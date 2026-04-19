//! Thread — the unit of long-lived agent work, modeled as tasks-as-data.
//!
//! A Thread is a serializable state machine. The scheduler drives it via two methods:
//!
//! - [`Thread::step`] advances the state synchronously. It returns a [`StepOutcome`]
//!   telling the scheduler whether to dispatch an I/O op, continue stepping, or pause
//!   until input arrives.
//! - [`Thread::apply_io_result`] integrates the completion of a previously-dispatched
//!   I/O op back into the task.
//!
//! Both methods push [`ThreadEvent`]s into an out-param so the scheduler can translate
//! them to wire-protocol events and broadcast to subscribers.
//!
//! The internal [`ThreadInternalState`] has finer distinctions than the public
//! [`ThreadStateLabel`] — the wire collapses them via [`Thread::public_state`]. This
//! indirection is the point of having a state machine: we can split a phase into
//! sub-phases (e.g. `NeedsModelCall` vs `AwaitingModelCall`) without touching
//! the wire.

use std::collections::{BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::{
    AllowMap, ApprovalChoice, BehaviorOrigin, ContentBlock, Conversation, Message, Role,
    ThreadBindings, ThreadConfig, ThreadSnapshot, ThreadStateLabel, ThreadSummary,
    ToolResultContent, TurnEntry, TurnLog, Usage,
};

use crate::functions::InFlightOps;
use crate::providers::model::ModelResponse;
use crate::tools::mcp::{CallToolResult, ToolAnnotations};

pub type OpId = u64;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Thread {
    pub id: String,
    /// Pod this thread belongs to. Matches a key in
    /// `Scheduler::pods`. Defaults to `id` for legacy persisted threads
    /// that pre-date the pod-aware load (Phase 2b shim treated each
    /// thread as its own pod, so id and pod_id were the same anyway).
    #[serde(default)]
    pub pod_id: String,
    pub created_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub title: Option<String>,
    pub config: ThreadConfig,
    /// Resource bindings — backend, sandbox, mcp host ids. Defaults to
    /// empty for threads persisted before Phase 3d.i (where the same info
    /// lived inline on `ThreadConfig`); the scheduler's load path
    /// re-pre-registers each binding so the registry is consistent again.
    #[serde(default)]
    pub bindings: ThreadBindings,
    pub conversation: Conversation,
    pub total_usage: Usage,
    /// Per-turn diagnostic log — one entry per assistant turn, in
    /// order. `#[serde(default)]` so threads persisted before this
    /// field existed load with an empty log.
    #[serde(default)]
    pub turn_log: TurnLog,
    #[serde(default)]
    pub archived: bool,
    /// Turns issued in the current user-message cycle; resets on user message.
    #[serde(default)]
    pub turns_in_cycle: u32,
    /// Snapshot of the pod's tool gate (`PodAllow.tools`) at thread-
    /// creation time. Consulted at approval-check time. Stored on the
    /// thread rather than looked up from the pod so mid-flight edits to
    /// the pod's allow table don't retroactively change a thread's
    /// active scope — rebind is the explicit path for that.
    #[serde(default = "AllowMap::allow_all")]
    pub tools_scope: AllowMap<String>,
    /// Per-thread "always allow" set keyed by tool name. Calls to a
    /// name in this set bypass a `AllowWithPrompt` disposition from
    /// `tools_scope` — the "remember this approval" action adds a name
    /// here. Persisted with the thread.
    #[serde(default)]
    pub tool_allowlist: BTreeSet<String>,
    /// Provenance stamp for threads spawned by a behavior trigger. `None`
    /// for interactive threads. Load-only plumbing: the scheduler stamps
    /// this on spawn and the on-completion hook reads it back.
    #[serde(default)]
    pub origin: Option<BehaviorOrigin>,
    /// When this thread was spawned as the continuation of a compacted
    /// thread, the id of that ancestor. Set once at spawn; never
    /// mutated afterward. `None` for threads that weren't created by
    /// compaction.
    #[serde(default)]
    pub continued_from: Option<String>,
    /// In-flight Function flags — e.g., `COMPACTING` is set between the
    /// moment a `/compact` command has appended the compaction prompt
    /// and the scheduler has finalized the continuation.
    ///
    /// Persisted so a compaction in flight during shutdown finalizes on
    /// the next startup when the model turn completes. Future restart
    /// semantics may clear non-resumable bits — see the "Restart note"
    /// in `docs/design_functions.md` — but today `COMPACTING` is
    /// genuinely resumable (the appended user message survives; the
    /// finalize logic is idempotent).
    #[serde(default)]
    pub in_flight: InFlightOps,
    /// Rendered `<dispatched-thread-notification>` envelopes queued
    /// for injection as fresh user messages once this thread reaches
    /// an idle turn boundary. Populated by
    /// `Scheduler::deliver_async_followup` when a dispatched child
    /// terminates while the parent is still Working; drained during
    /// `step_until_blocked` when the parent hits Idle/Completed.
    /// Transient — not persisted; a restart mid-delivery drops the
    /// queued follow-up (same lifecycle guarantee as other in-flight
    /// Function state).
    #[serde(default, skip)]
    pub pending_tool_result_followups: Vec<String>,
    /// Parent thread id when this thread was spawned by a parent's
    /// `dispatch_thread` tool call. `None` for top-level threads. Set
    /// once at spawn; never mutated afterward. Distinct from the
    /// transient "return my final message to this tool_use_id" mapping,
    /// which lives in scheduler state (not here) so it doesn't outlive
    /// the single tool call that produced it.
    #[serde(default)]
    pub dispatched_by: Option<String>,
    /// Dispatch nesting depth. Top-level threads are 0; each
    /// `dispatch_thread` call spawns a child at `parent.depth + 1`. The
    /// scheduler refuses to spawn past a fixed cap so a buggy agent
    /// can't recursively dispatch itself into the ground.
    #[serde(default)]
    pub dispatch_depth: u32,
    /// User's in-progress compose-box contents for this thread.
    /// Persisted so a partially-typed prompt survives reopening the
    /// thread or restarting the server. Mutated via `SetThreadDraft`
    /// wire messages and surfaced to subscribers via
    /// `ThreadDraftUpdated`. Empty string is the "no draft" state.
    #[serde(default)]
    pub draft: String,
    pub internal: ThreadInternalState,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ThreadInternalState {
    /// No conversation yet / never run.
    Idle,
    /// Previous loop ended on `end_turn`; accepts follow-up user messages.
    Completed,
    /// A user message has been appended but at least one bound resource
    /// (sandbox, primary MCP host) isn't Ready yet. The scheduler watches
    /// resource transitions and clears ids out of `needed`; when the set
    /// empties, the thread moves to `NeedsModelCall`.
    WaitingOnResources { needed: Vec<String> },
    /// Ready to dispatch a model call. The conversation already contains the latest
    /// user or tool_result message.
    NeedsModelCall,
    AwaitingModel {
        op_id: OpId,
        started_at: DateTime<Utc>,
    },
    /// Model responded with tool_uses. Each entry in `pending_dispatch` still needs to
    /// be fired at MCP; `pending_io` maps op_ids of in-flight tool calls to their
    /// tool_use_ids; `completed` accumulates ToolResult blocks as they return. A tool
    /// that was rejected during the approval phase is already synthesized into
    /// `completed` here with `is_error: true` so the model's next turn sees the denial.
    ///
    /// `approvals` carries the decision that led each tool to be dispatched so the audit
    /// log can record it when the tool completes (one line per call — see
    /// `design_permissions.md`). Keyed by tool_use_id.
    AwaitingTools {
        pending_dispatch: Vec<ToolUseReq>,
        pending_io: HashMap<OpId, String>,
        completed: Vec<ContentBlock>,
        #[serde(default)]
        approvals: HashMap<String, ApprovalRecord>,
    },
    /// Terminal: unrecoverable error.
    Failed { at_phase: String, message: String },
    /// Terminal: user-initiated cancellation. In-flight I/O may still complete; their
    /// results are discarded by the scheduler.
    Cancelled,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolUseReq {
    pub tool_use_id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Decision record threaded from an approval outcome into the dispatch phase so the
/// audit log captures who said yes (or that policy auto-approved).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ApprovalRecord {
    pub decision: String,    // "auto" | "approve" | "reject"
    pub who_decided: String, // "policy:read_only" | "policy:auto_approve_all" | "user:{conn}"
}

/// Request for an I/O operation to be dispatched by the scheduler.
///
/// Phase 3d.ii: only the per-thread, state-machine-driven ops live here.
/// Resource provisioning (sandbox + primary MCP host) is dispatched
/// separately by the scheduler at thread-creation time and routed through
/// [`crate::runtime::io_dispatch::SchedulerCompletion::Provision`].
#[derive(Debug, Clone)]
pub enum IoRequest {
    ModelCall {
        op_id: OpId,
    },
    ToolCall {
        op_id: OpId,
        tool_use_id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Successful result of an I/O op. Errors are carried in-band so
/// `apply_io_result` can transition the task to Failed with the right
/// phase tag.
pub enum IoResult {
    ModelCall(Result<ModelResponse, String>),
    ToolCall {
        tool_use_id: String,
        result: Result<CallToolResult, String>,
    },
}

#[derive(Debug)]
pub enum StepOutcome {
    /// Dispatch this I/O op, then wait for it.
    DispatchIo(IoRequest),
    /// State advanced synchronously; call step() again.
    Continue,
    /// No further progress possible until an I/O result or user input arrives.
    Paused,
}

/// Internal task event — translated by the scheduler into wire [`ServerToClient`] events.
#[derive(Debug, Clone)]
pub enum ThreadEvent {
    AssistantBegin {
        turn: u32,
    },
    ToolCallBegin {
        tool_use_id: String,
        name: String,
        args_preview: String,
        /// Full tool arguments (untruncated). Used by the webui for
        /// rich tool-specific renderers (e.g. unified diffs for
        /// edit_file). The router forwards this onto the wire.
        args: serde_json::Value,
    },
    ToolCallEnd {
        tool_use_id: String,
        result_preview: String,
        is_error: bool,
    },
    PendingApproval {
        approval_id: String,
        tool_use_id: String,
        name: String,
        args_preview: String,
        destructive: bool,
        read_only: bool,
    },
    ApprovalResolved {
        approval_id: String,
        decision: ApprovalChoice,
        decided_by_conn: Option<u64>,
    },
    AssistantEnd {
        stop_reason: Option<String>,
        usage: Usage,
    },
    LoopComplete,
    /// The task's `public_state()` flipped. Carries the new label so the
    /// scheduler's router can emit a wire `ThreadStateChanged` without looking
    /// the task back up.
    StateChanged {
        state: ThreadStateLabel,
    },
    Error {
        message: String,
    },
    /// The task's `tool_allowlist` set was modified — either a tool was added
    /// (via an approve-and-remember decision) or removed (via the revoke UI).
    /// Carries the full new set so the scheduler can broadcast it without
    /// recomputing.
    AllowlistChanged {
        allowlist: Vec<String>,
    },
    /// Tool call completed. Carried separately from the user-visible ToolCallEnd event
    /// so the scheduler can write an audit entry outside the task state machine.
    AuditToolCall {
        tool_name: String,
        args: serde_json::Value,
        is_error: bool,
        error_message: Option<String>,
        decision: String,
        who_decided: String,
    },
}

impl Thread {
    pub fn new(
        id: String,
        pod_id: String,
        config: ThreadConfig,
        bindings: ThreadBindings,
        tools_scope: AllowMap<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            pod_id,
            created_at: now,
            last_active: now,
            title: None,
            config,
            bindings,
            conversation: Conversation::new(),
            total_usage: Usage::default(),
            turn_log: TurnLog::default(),
            archived: false,
            turns_in_cycle: 0,
            tools_scope,
            tool_allowlist: BTreeSet::new(),
            origin: None,
            continued_from: None,
            in_flight: InFlightOps::empty(),
            pending_tool_result_followups: Vec::new(),
            dispatched_by: None,
            dispatch_depth: 0,
            draft: String::new(),
            internal: ThreadInternalState::Idle,
        }
    }

    /// Builder-style setter for behavior provenance. The scheduler uses
    /// this when spawning threads from a `RunBehavior` / trigger fire so
    /// the hook that updates `BehaviorState` on terminal transitions
    /// knows which behavior this thread belongs to.
    pub fn with_origin(mut self, origin: BehaviorOrigin) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Stamp the compaction-continuation ancestor. Called by the
    /// scheduler when spawning the continuation thread produced by
    /// `/compact` — never mutated after construction.
    pub fn with_continued_from(mut self, predecessor_id: String) -> Self {
        self.continued_from = Some(predecessor_id);
        self
    }

    /// Stamp the dispatch parent + depth. Called by the scheduler when
    /// spawning a thread from a `dispatch_thread` tool call — never
    /// mutated after construction.
    pub fn with_dispatched_by(mut self, parent_id: String, parent_depth: u32) -> Self {
        self.dispatched_by = Some(parent_id);
        self.dispatch_depth = parent_depth.saturating_add(1);
        self
    }

    pub fn touch(&mut self) {
        self.last_active = Utc::now();
    }

    /// Build a new thread by rewinding `self` to message index
    /// `from_message_index` (exclusive). Carries pod_id, config,
    /// bindings, and tool_allowlist over verbatim; resets per-run
    /// state (title, origin, lineage, cycle counter, compaction flag).
    ///
    /// Rejects mid-turn sources (working, awaiting approval,
    /// compacting) — the in-flight operation has no meaning in the
    /// derived conversation. Rejects non-user-role indices —
    /// truncating at a tool_use / tool_result boundary leaves an
    /// unanswered tool call, which the user-role restriction sidesteps
    /// in v1.
    pub fn fork_from(&self, new_id: String, from_message_index: usize) -> Result<Thread, String> {
        match &self.internal {
            ThreadInternalState::Idle
            | ThreadInternalState::Completed
            | ThreadInternalState::Cancelled
            | ThreadInternalState::Failed { .. } => {}
            _ => {
                return Err("cannot fork a thread that is mid-turn".into());
            }
        }
        if self.in_flight.contains(InFlightOps::COMPACTING) {
            return Err("cannot fork a thread while compaction is in flight".into());
        }
        let messages = self.conversation.messages();
        if from_message_index >= messages.len() {
            return Err(format!(
                "fork index {from_message_index} out of bounds (conversation has {} messages)",
                messages.len()
            ));
        }
        if messages[from_message_index].role != Role::User {
            return Err(format!(
                "fork index {from_message_index} is not a user-role message"
            ));
        }
        let mut conversation = self.conversation.clone();
        conversation.truncate(from_message_index);
        let assistant_turns = conversation
            .messages()
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .count();
        let mut turn_log = self.turn_log.clone();
        turn_log.entries.truncate(assistant_turns);
        let mut total_usage = Usage::default();
        for entry in &turn_log.entries {
            total_usage.add(&entry.usage);
        }
        let now = Utc::now();
        Ok(Thread {
            id: new_id,
            pod_id: self.pod_id.clone(),
            created_at: now,
            last_active: now,
            title: None,
            config: self.config.clone(),
            bindings: self.bindings.clone(),
            conversation,
            total_usage,
            turn_log,
            archived: false,
            turns_in_cycle: 0,
            tools_scope: self.tools_scope.clone(),
            tool_allowlist: self.tool_allowlist.clone(),
            origin: None,
            continued_from: None,
            in_flight: InFlightOps::empty(),
            pending_tool_result_followups: Vec::new(),
            dispatched_by: None,
            dispatch_depth: 0,
            // Client seeds the new thread's draft with the forked-from
            // user-message text via a follow-up `SetThreadDraft`; the
            // source's in-progress draft would be the wrong thing to
            // carry over.
            draft: String::new(),
            internal: ThreadInternalState::Idle,
        })
    }

    pub fn public_state(&self) -> ThreadStateLabel {
        match &self.internal {
            ThreadInternalState::Idle => ThreadStateLabel::Idle,
            ThreadInternalState::Completed => ThreadStateLabel::Completed,
            ThreadInternalState::WaitingOnResources { .. }
            | ThreadInternalState::NeedsModelCall
            | ThreadInternalState::AwaitingModel { .. }
            | ThreadInternalState::AwaitingTools { .. } => ThreadStateLabel::Working,
            ThreadInternalState::Failed { .. } => ThreadStateLabel::Failed,
            ThreadInternalState::Cancelled => ThreadStateLabel::Cancelled,
        }
    }

    pub fn summary(&self) -> ThreadSummary {
        ThreadSummary {
            thread_id: self.id.clone(),
            pod_id: self.pod_id.clone(),
            title: self.title.clone(),
            state: self.public_state(),
            created_at: self.created_at.to_rfc3339(),
            last_active: self.last_active.to_rfc3339(),
            origin: self.origin.clone(),
            continued_from: self.continued_from.clone(),
            dispatched_by: self.dispatched_by.clone(),
        }
    }

    pub fn snapshot(&self) -> ThreadSnapshot {
        ThreadSnapshot {
            thread_id: self.id.clone(),
            pod_id: self.pod_id.clone(),
            title: self.title.clone(),
            config: self.config.clone(),
            bindings: self.bindings.clone(),
            state: self.public_state(),
            conversation: self.conversation.clone(),
            total_usage: self.total_usage,
            turn_log: self.turn_log.clone(),
            draft: self.draft.clone(),
            created_at: self.created_at.to_rfc3339(),
            last_active: self.last_active.to_rfc3339(),
            failure: self.failure_detail(),
            tool_allowlist: self.tool_allowlist.iter().cloned().collect(),
            origin: self.origin.clone(),
            continued_from: self.continued_from.clone(),
            dispatched_by: self.dispatched_by.clone(),
        }
    }

    /// If the task is in the Failed state, return a human-readable description of
    /// why (combining phase + message). Returns None otherwise.
    pub fn failure_detail(&self) -> Option<String> {
        match &self.internal {
            ThreadInternalState::Failed { at_phase, message } => {
                Some(format!("{at_phase}: {message}"))
            }
            _ => None,
        }
    }

    /// Apply a user-submitted message. Appends to the conversation and starts
    /// the loop (or re-starts after a completed/failed cycle).
    ///
    /// `pending_resources` is the subset of `bindings.*` ids the scheduler
    /// hasn't observed transition to Ready yet — empty means "go straight
    /// to NeedsModelCall." When non-empty the thread parks in
    /// `WaitingOnResources` and the scheduler nudges it via
    /// [`Self::clear_waiting_resource`] as each id flips Ready.
    pub fn submit_user_message(&mut self, text: String, pending_resources: Vec<String>) {
        self.conversation.push(Message::user_text(text));
        self.turns_in_cycle = 0;
        self.internal = if pending_resources.is_empty() {
            ThreadInternalState::NeedsModelCall
        } else {
            ThreadInternalState::WaitingOnResources {
                needed: pending_resources,
            }
        };
        self.touch();
    }

    /// Append a tool-output text message and transition the thread
    /// toward its next model call. Same state transition as
    /// [`Self::submit_user_message`] — the only difference is the
    /// appended message's `Role` (`ToolResult` instead of `User`) so
    /// clients and adapters can classify the append without content-
    /// block inspection.
    pub fn submit_tool_result_text(&mut self, text: String, pending_resources: Vec<String>) {
        self.conversation.push(Message::tool_result_text(text));
        self.turns_in_cycle = 0;
        self.internal = if pending_resources.is_empty() {
            ThreadInternalState::NeedsModelCall
        } else {
            ThreadInternalState::WaitingOnResources {
                needed: pending_resources,
            }
        };
        self.touch();
    }

    /// Drop a now-Ready resource id from the `WaitingOnResources` set.
    /// Returns whether the thread is now ready to step (its `needed` set
    /// emptied as a result). No-op for any state other than
    /// `WaitingOnResources`.
    pub fn clear_waiting_resource(&mut self, resource_id: &str) -> bool {
        let ThreadInternalState::WaitingOnResources { needed } = &mut self.internal else {
            return false;
        };
        needed.retain(|id| id != resource_id);
        if needed.is_empty() {
            self.internal = ThreadInternalState::NeedsModelCall;
            self.touch();
            true
        } else {
            false
        }
    }

    pub fn cancel(&mut self) {
        self.internal = ThreadInternalState::Cancelled;
        self.touch();
    }

    pub fn fail(&mut self, phase: impl Into<String>, message: impl Into<String>) {
        self.internal = ThreadInternalState::Failed {
            at_phase: phase.into(),
            message: message.into(),
        };
        self.touch();
    }

    /// Drop a tool name from the allowlist. Returns true if the entry was
    /// present (so the caller knows whether to broadcast an update).
    pub fn remove_from_allowlist(&mut self, tool_name: &str) -> bool {
        let removed = self.tool_allowlist.remove(tool_name);
        if removed {
            self.touch();
        }
        removed
    }

    pub fn allowlist_snapshot(&self) -> Vec<String> {
        self.tool_allowlist.iter().cloned().collect()
    }

    /// Is the task accepting new user input right now?
    pub fn is_idle(&self) -> bool {
        matches!(
            self.internal,
            ThreadInternalState::Idle
                | ThreadInternalState::Completed
                | ThreadInternalState::Failed { .. }
                | ThreadInternalState::Cancelled
        )
    }

    /// Advance the state machine. Caller provides `next_op_id` for fresh I/O ops and
    /// `events` as the out-param for task events emitted during this step.
    pub fn step(&mut self, next_op_id: &mut OpId, events: &mut Vec<ThreadEvent>) -> StepOutcome {
        let prev_public = self.public_state();
        let outcome = self.step_inner(next_op_id, events);
        let new_public = self.public_state();
        if prev_public != new_public {
            events.push(ThreadEvent::StateChanged { state: new_public });
        }
        outcome
    }

    fn step_inner(&mut self, next_op_id: &mut OpId, events: &mut Vec<ThreadEvent>) -> StepOutcome {
        // Break the state out so we can replace it.
        let current = std::mem::replace(&mut self.internal, ThreadInternalState::Idle);
        match current {
            ThreadInternalState::Idle
            | ThreadInternalState::Completed
            | ThreadInternalState::Cancelled
            | ThreadInternalState::Failed { .. } => {
                // Restore and pause — caller shouldn't be stepping terminal states.
                self.internal = current;
                StepOutcome::Paused
            }
            ThreadInternalState::NeedsModelCall => {
                if self.turns_in_cycle >= self.config.max_turns {
                    tracing::warn!(max_turns = self.config.max_turns, thread_id = %self.id, "max_turns reached");
                    events.push(ThreadEvent::LoopComplete);
                    self.internal = ThreadInternalState::Completed;
                    self.touch();
                    return StepOutcome::Continue;
                }
                self.turns_in_cycle += 1;
                events.push(ThreadEvent::AssistantBegin {
                    turn: self.turns_in_cycle,
                });
                let op_id = next_id(next_op_id);
                self.internal = ThreadInternalState::AwaitingModel {
                    op_id,
                    started_at: Utc::now(),
                };
                self.touch();
                StepOutcome::DispatchIo(IoRequest::ModelCall { op_id })
            }
            ThreadInternalState::WaitingOnResources { .. }
            | ThreadInternalState::AwaitingModel { .. } => {
                // Not stepping these — waiting on resource provisioning
                // or a model response.
                self.internal = current;
                StepOutcome::Paused
            }
            ThreadInternalState::AwaitingTools {
                mut pending_dispatch,
                mut pending_io,
                completed,
                approvals,
            } => {
                if let Some(next) = pending_dispatch.pop() {
                    let op_id = next_id(next_op_id);
                    pending_io.insert(op_id, next.tool_use_id.clone());
                    events.push(ThreadEvent::ToolCallBegin {
                        tool_use_id: next.tool_use_id.clone(),
                        name: next.name.clone(),
                        args_preview: truncate(
                            serde_json::to_string(&next.input).unwrap_or_default(),
                            200,
                        ),
                        args: next.input.clone(),
                    });
                    let dispatch = IoRequest::ToolCall {
                        op_id,
                        tool_use_id: next.tool_use_id.clone(),
                        name: next.name.clone(),
                        input: next.input.clone(),
                    };
                    self.internal = ThreadInternalState::AwaitingTools {
                        pending_dispatch,
                        pending_io,
                        completed,
                        approvals,
                    };
                    self.touch();
                    StepOutcome::DispatchIo(dispatch)
                } else if pending_io.is_empty() {
                    // All tool calls done — append ToolResult blocks and loop back.
                    self.conversation
                        .push(Message::tool_result_blocks(completed));
                    self.internal = ThreadInternalState::NeedsModelCall;
                    self.touch();
                    StepOutcome::Continue
                } else {
                    self.internal = ThreadInternalState::AwaitingTools {
                        pending_dispatch,
                        pending_io,
                        completed,
                        approvals,
                    };
                    StepOutcome::Paused
                }
            }
        }
    }

    /// Apply an I/O completion. Pushes events describing the integration; the scheduler
    /// should call `step_until_blocked` afterward.
    ///
    /// `tool_annotations` is consulted when a model response arrives so the approval
    /// policy can be evaluated against each tool's hint flags.
    pub fn apply_io_result(
        &mut self,
        op_id: OpId,
        result: IoResult,
        tool_annotations: &HashMap<String, ToolAnnotations>,
        events: &mut Vec<ThreadEvent>,
    ) {
        let prev_public = self.public_state();
        self.apply_io_result_inner(op_id, result, tool_annotations, events);
        let new_public = self.public_state();
        if prev_public != new_public {
            events.push(ThreadEvent::StateChanged { state: new_public });
        }
    }

    fn apply_io_result_inner(
        &mut self,
        op_id: OpId,
        result: IoResult,
        tool_annotations: &HashMap<String, ToolAnnotations>,
        events: &mut Vec<ThreadEvent>,
    ) {
        self.touch();
        // Cancelled task: drop the result on the floor.
        if matches!(self.internal, ThreadInternalState::Cancelled) {
            return;
        }

        match (&self.internal, result) {
            (
                ThreadInternalState::AwaitingModel {
                    op_id: expected, ..
                },
                IoResult::ModelCall(res),
            ) if *expected == op_id => match res {
                Ok(response) => self.integrate_model_response(response, tool_annotations, events),
                Err(msg) => {
                    events.push(ThreadEvent::Error {
                        message: format!("model call failed: {msg}"),
                    });
                    self.fail("model_call", msg);
                }
            },
            (
                ThreadInternalState::AwaitingTools { .. },
                IoResult::ToolCall {
                    tool_use_id,
                    result,
                },
            ) => self.integrate_tool_result(op_id, tool_use_id, result, events),
            (state, result) => {
                tracing::warn!(
                    thread_id = %self.id,
                    op_id,
                    state = ?std::mem::discriminant(state),
                    result = ?std::mem::discriminant(&result),
                    "io result does not match current state — discarding"
                );
            }
        }
    }

    fn integrate_model_response(
        &mut self,
        response: ModelResponse,
        tool_annotations: &HashMap<String, ToolAnnotations>,
        events: &mut Vec<ThreadEvent>,
    ) {
        let ModelResponse {
            content: assistant_blocks,
            stop_reason,
            usage,
        } = response;
        self.total_usage.add(&usage);
        self.turn_log.entries.push(TurnEntry { usage });
        // Text and thinking blocks are NOT emitted as events here — the
        // scheduler's streaming consumer broadcasts them via
        // `ThreadAssistantTextDelta` / `ThreadAssistantReasoningDelta` during
        // the model call, and the assembled blocks are preserved on the
        // `Message` we push to `self.conversation` below for snapshot replay.
        let tool_uses: Vec<ToolUseReq> = assistant_blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => Some(ToolUseReq {
                    tool_use_id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }),
                _ => None,
            })
            .collect();
        events.push(ThreadEvent::AssistantEnd { stop_reason, usage });
        self.conversation
            .push(Message::assistant_blocks(assistant_blocks));

        if tool_uses.is_empty() {
            events.push(ThreadEvent::LoopComplete);
            self.internal = ThreadInternalState::Completed;
            return;
        }

        // Approval evaluation has moved to the scheduler's Function
        // registry (see `register_tool_function` in
        // `src/runtime/scheduler/functions.rs`). Every tool_use emerges
        // from here as AutoApproved; the scheduler either dispatches
        // it, defers it pending a user prompt, or synthesizes a denial
        // depending on the pod's tool-scope disposition.
        let _ = tool_annotations; // Consulted scheduler-side now.
        // Approvals are recorded for audit at the scheduler-side
        // Function layer; here we mark each call as "scheduler-gated"
        // in the thread's approvals map so the audit path stays uniform.
        let approvals: HashMap<String, ApprovalRecord> = tool_uses
            .iter()
            .map(|tu| {
                (
                    tu.tool_use_id.clone(),
                    ApprovalRecord {
                        decision: "auto".into(),
                        who_decided: "scheduler_gated".into(),
                    },
                )
            })
            .collect();
        self.internal = ThreadInternalState::AwaitingTools {
            pending_dispatch: make_dispatch_order(tool_uses),
            pending_io: HashMap::new(),
            completed: Vec::new(),
            approvals,
        };
    }

    fn integrate_tool_result(
        &mut self,
        op_id: OpId,
        tool_use_id: String,
        result: Result<CallToolResult, String>,
        events: &mut Vec<ThreadEvent>,
    ) {
        let ThreadInternalState::AwaitingTools {
            pending_dispatch,
            mut pending_io,
            mut completed,
            mut approvals,
        } = std::mem::replace(&mut self.internal, ThreadInternalState::Idle)
        else {
            unreachable!("matched guard ensures we're in AwaitingTools")
        };

        // The op_id should be tracked. Use it to verify; tool_use_id is the canonical
        // key for the conversation block either way.
        pending_io.remove(&op_id);

        let approval = approvals
            .remove(&tool_use_id)
            .unwrap_or_else(|| ApprovalRecord {
                decision: "dispatched".into(),
                who_decided: "policy".into(),
            });

        let (text, is_error, tool_name, args, err_msg) = match result {
            Ok(r) => {
                let text = join_mcp_text(&r.content);
                let tool_name = find_tool_name(&self.conversation, &tool_use_id);
                let args = find_tool_args(&self.conversation, &tool_use_id);
                (text, r.is_error, tool_name, args, None)
            }
            Err(msg) => {
                let tool_name = find_tool_name(&self.conversation, &tool_use_id);
                let args = find_tool_args(&self.conversation, &tool_use_id);
                (
                    format!("tool invocation failed: {msg}"),
                    true,
                    tool_name,
                    args,
                    Some(msg),
                )
            }
        };

        events.push(ThreadEvent::ToolCallEnd {
            tool_use_id: tool_use_id.clone(),
            result_preview: truncate(text.clone(), 200),
            is_error,
        });
        events.push(ThreadEvent::AuditToolCall {
            tool_name,
            args,
            is_error,
            error_message: err_msg,
            decision: approval.decision,
            who_decided: approval.who_decided,
        });

        completed.push(ContentBlock::ToolResult {
            tool_use_id,
            content: ToolResultContent::Text(text),
            is_error,
        });

        self.internal = ThreadInternalState::AwaitingTools {
            pending_dispatch,
            pending_io,
            completed,
            approvals,
        };
    }

}

/// `step()` pops from the back; feed it in reverse so the original order is preserved.
fn make_dispatch_order(tool_uses: Vec<ToolUseReq>) -> Vec<ToolUseReq> {
    reverse_for_pop(tool_uses)
}

fn reverse_for_pop<T>(mut v: Vec<T>) -> Vec<T> {
    v.reverse();
    v
}

/// Derive a title from the user's initial message: trim, collapse internal whitespace,
/// truncate to ~50 chars (rounded to a char boundary) with a trailing ellipsis.
pub fn derive_title(initial_message: &str) -> String {
    let collapsed: String = initial_message
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    const MAX: usize = 50;
    if collapsed.chars().count() <= MAX {
        collapsed
    } else {
        let mut out: String = collapsed.chars().take(MAX).collect();
        out.push('…');
        out
    }
}

pub fn new_task_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("task-{:016x}", nanos as u64)
}

fn next_id(counter: &mut OpId) -> OpId {
    *counter += 1;
    *counter
}

fn join_mcp_text(blocks: &[crate::tools::mcp::McpContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        match b {
            crate::tools::mcp::McpContentBlock::Text { text } => out.push_str(text),
        }
    }
    out
}

fn find_tool_name(conv: &Conversation, tool_use_id: &str) -> String {
    for msg in conv.messages().iter().rev() {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, name, .. } = block
                && id == tool_use_id
            {
                return name.clone();
            }
        }
    }
    String::new()
}

fn find_tool_args(conv: &Conversation, tool_use_id: &str) -> serde_json::Value {
    for msg in conv.messages().iter().rev() {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, input, .. } = block
                && id == tool_use_id
            {
                return input.clone();
            }
        }
    }
    serde_json::Value::Null
}

fn truncate(mut s: String, max: usize) -> String {
    if s.len() > max {
        // `max` is a byte count but may land inside a multi-byte UTF-8 sequence;
        // walk back to the nearest char boundary before cutting.
        let mut cut = max;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push('…');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn truncate_handles_multibyte_boundary() {
        // `…` is 3 bytes (E2 80 A6). Position the char so `max` lands inside it —
        // the naive `String::truncate(max)` panics here.
        let s = format!("{}…tail", "a".repeat(198));
        let out = truncate(s, 200);
        assert!(out.ends_with('…'));
        // The cut falls back to byte 198 (end of the "a" run, on a boundary).
        assert_eq!(out, format!("{}…", "a".repeat(198)));
    }

    #[test]
    fn truncate_no_op_when_short() {
        assert_eq!(truncate("héllo".into(), 200), "héllo");
    }

    #[test]
    fn truncate_cuts_on_ascii_boundary() {
        assert_eq!(truncate("abcdefghij".into(), 4), "abcd…");
    }

    // Approval-flow tests moved to the scheduler-side Function
    // registry in Phase 4c; Thread no longer owns that path. Unit tests
    // covering the scheduler's register_tool_function + resolve_tool_approval
    // arrive with later scheduler-integration-test work.

    fn basic_thread() -> Thread {
        let cfg = ThreadConfig {
            model: "test".into(),
            system_prompt: "test".into(),
            max_tokens: 100,
            max_turns: 10,
            compaction: Default::default(),
        };
        Thread::new(
            "t1".into(),
            "t1".into(),
            cfg,
            ThreadBindings::default(),
            AllowMap::allow_all(),
        )
    }

    #[test]
    fn remove_from_allowlist_returns_true_only_when_present() {
        let mut task = basic_thread();
        task.tool_allowlist.insert("edit_file".into());
        assert!(task.remove_from_allowlist("edit_file"));
        assert!(!task.remove_from_allowlist("edit_file"));
        assert!(!task.remove_from_allowlist("never_added"));
    }

    fn base_config_for_fork() -> ThreadConfig {
        ThreadConfig {
            model: "test".into(),
            system_prompt: "test".into(),
            max_tokens: 100,
            max_turns: 10,
            compaction: Default::default(),
        }
    }

    fn thread_with_two_turns() -> Thread {
        let mut task = Thread::new(
            "src".into(),
            "pod".into(),
            base_config_for_fork(),
            ThreadBindings::default(),
            AllowMap::allow_all(),
        );
        // [user, assistant, user, assistant] — two complete turns.
        task.conversation.push(Message::user_text("hi"));
        task.conversation
            .push(Message::assistant_blocks(vec![ContentBlock::Text {
                text: "hello".into(),
            }]));
        task.conversation.push(Message::user_text("again"));
        task.conversation
            .push(Message::assistant_blocks(vec![ContentBlock::Text {
                text: "hi again".into(),
            }]));
        task.turn_log.entries.push(TurnEntry {
            usage: Usage {
                input_tokens: 10,
                output_tokens: 2,
                ..Usage::default()
            },
        });
        task.turn_log.entries.push(TurnEntry {
            usage: Usage {
                input_tokens: 20,
                output_tokens: 4,
                ..Usage::default()
            },
        });
        task.total_usage = Usage {
            input_tokens: 30,
            output_tokens: 6,
            ..Usage::default()
        };
        task.tool_allowlist.insert("edit_file".into());
        task.internal = ThreadInternalState::Idle;
        task
    }

    #[test]
    fn fork_from_prefix_keeps_first_turn() {
        let src = thread_with_two_turns();
        let forked = src.fork_from("new".into(), 2).unwrap();
        // Prefix [user, assistant] survives; second user/assistant pair gone.
        assert_eq!(forked.conversation.len(), 2);
        assert_eq!(forked.turn_log.entries.len(), 1);
        assert_eq!(forked.total_usage.input_tokens, 10);
        assert_eq!(forked.total_usage.output_tokens, 2);
        assert!(forked.tool_allowlist.contains("edit_file"));
        assert_eq!(forked.id, "new");
        assert_eq!(forked.pod_id, "pod");
        assert!(!forked.archived);
        assert_eq!(forked.turns_in_cycle, 0);
        assert!(forked.title.is_none());
        // The draft is the client's unsaved typing buffer on the
        // source thread — not meaningful on the new thread. The
        // client seeds it explicitly with the forked-from message
        // text after receiving `ThreadCreated`, so the runtime
        // starts the fork with an empty draft regardless of what
        // the source had.
        assert_eq!(forked.draft, "");
    }

    #[test]
    fn fork_drops_source_draft() {
        let mut src = thread_with_two_turns();
        src.draft = "leftover typing from source".into();
        let forked = src.fork_from("new".into(), 2).unwrap();
        assert_eq!(forked.draft, "");
    }

    #[test]
    fn fork_from_index_zero_empties_conversation() {
        let src = thread_with_two_turns();
        let forked = src.fork_from("new".into(), 0).unwrap();
        assert!(forked.conversation.is_empty());
        assert!(forked.turn_log.entries.is_empty());
        assert_eq!(forked.total_usage.input_tokens, 0);
    }

    #[test]
    fn fork_rejects_non_user_index() {
        let src = thread_with_two_turns();
        // Index 1 is an assistant message.
        assert!(src.fork_from("new".into(), 1).is_err());
    }

    #[test]
    fn fork_rejects_out_of_bounds() {
        let src = thread_with_two_turns();
        // Conversation has 4 messages; index 4 is past the end. (v1 also rejects
        // == len since there'd be no user boundary to fork from.)
        assert!(src.fork_from("new".into(), 4).is_err());
    }

    #[test]
    fn fork_rejects_mid_turn() {
        let mut src = thread_with_two_turns();
        src.internal = ThreadInternalState::NeedsModelCall;
        assert!(src.fork_from("new".into(), 2).is_err());
    }

    #[test]
    fn fork_rejects_during_compaction() {
        let mut src = thread_with_two_turns();
        src.in_flight.insert(InFlightOps::COMPACTING);
        assert!(src.fork_from("new".into(), 2).is_err());
    }
}
