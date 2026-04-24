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

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::{
    BehaviorOrigin, ContentBlock, Conversation, Message, Role, ThreadBindings, ThreadConfig,
    ThreadSnapshot, ThreadStateLabel, ThreadSummary, ToolResultContent, ToolSurface, TurnEntry,
    TurnLog, Usage,
};

use crate::functions::InFlightOps;
use crate::permission::Scope;
use crate::providers::model::ModelResponse;
use crate::tools::mcp::{CallToolResult, ToolAnnotations};

pub type OpId = u64;

/// Reason text injected into synthesized `is_error` tool_result blocks
/// when a user cancel interrupts an in-flight or queued tool call. Read
/// by downstream consumers (compaction, cross-provider replay) that
/// need to know the ToolUse didn't complete naturally.
const CANCEL_REASON: &str = "cancelled";

/// Build an `is_error` ToolResult for a ToolUse that never got its
/// real result and push the matching [`ThreadEvent::ToolCallEnd`] into
/// `events`. Shared by `Thread::cancel` and
/// `Thread::synthesize_trailing_tool_results`.
fn synth_interrupted_tool_result(
    tool_use_id: &str,
    synth_text: &str,
    events: &mut Vec<ThreadEvent>,
) -> ContentBlock {
    events.push(ThreadEvent::ToolCallEnd {
        tool_use_id: tool_use_id.to_string(),
        result_preview: synth_text.to_string(),
        is_error: true,
    });
    ContentBlock::ToolResult {
        tool_use_id: tool_use_id.to_string(),
        content: ToolResultContent::Text(synth_text.to_string()),
        is_error: true,
    }
}

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
    /// The thread's permission scope, snapshotted from the pod's
    /// `[allow]` table at creation. Stored on the thread rather than
    /// looked up from the pod so mid-flight edits to the pod's allow
    /// table don't retroactively change a thread's active scope —
    /// pod-file edits apply to *future* threads. Escalation grants
    /// (when the request_escalation family lands in task #5) widen this
    /// in place.
    #[serde(default)]
    pub scope: Scope,
    /// How the thread presents its tool catalog. Composed at creation
    /// from the pod's `thread_defaults.tool_surface` and the behavior's
    /// optional override; frozen for the thread's lifetime. Affects
    /// `Role::Tools` content at seed time and the listing appended to
    /// the system prompt — but not what the thread can actually call
    /// (that's `scope.tools`).
    #[serde(default)]
    pub tool_surface: ToolSurface,
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
    /// denied by the thread's scope is synthesized into `completed` here with
    /// `is_error: true` so the model's next turn sees the denial.
    AwaitingTools {
        pending_dispatch: Vec<ToolUseReq>,
        pending_io: HashMap<OpId, String>,
        completed: Vec<ContentBlock>,
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
    /// Tool call completed. Carried separately from the user-visible ToolCallEnd event
    /// so the scheduler can write an audit entry outside the task state machine.
    AuditToolCall {
        tool_name: String,
        args: serde_json::Value,
        is_error: bool,
        error_message: Option<String>,
    },
}

impl Thread {
    pub fn new(
        id: String,
        pod_id: String,
        config: ThreadConfig,
        bindings: ThreadBindings,
        scope: Scope,
        tool_surface: ToolSurface,
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
            scope,
            tool_surface,
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
    /// bindings, and scope over verbatim; resets per-run state
    /// (title, origin, lineage, cycle counter, compaction flag).
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
            scope: self.scope.clone(),
            tool_surface: self.tool_surface.clone(),
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
            origin: self.origin.clone(),
            continued_from: self.continued_from.clone(),
            dispatched_by: self.dispatched_by.clone(),
            scope: self.scope.clone(),
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

    /// Bring the thread to a clean `Idle` state that can accept a new
    /// user message (or be stepped from scratch) without corrupting
    /// the conversation shape.
    ///
    /// - `AwaitingTools`: synthesize `is_error: true` `tool_result`
    ///   blocks for every unresolved `tool_use_id` and merge them with
    ///   any already-completed results into a single
    ///   `Role::ToolResult` message. Returns the interrupted
    ///   `tool_use_id`s so the caller can cancel their Function
    ///   entries (late-arriving I/O results will hit a state mismatch
    ///   in `apply_io_result` and be discarded).
    /// - `AwaitingModel` / `NeedsModelCall` / `WaitingOnResources`:
    ///   transition to `Idle`. These paths don't mutate the
    ///   conversation tail during their active phase (streaming
    ///   deltas go to subscribers, not `self.conversation`), so no
    ///   synthesized filler is needed.
    /// - `Idle` / `Completed` / `Failed` / `Cancelled`: no-op. The
    ///   terminal-state → `Idle` promotion belongs to an explicit
    ///   `recover()` action so "heal" doesn't silently clear a
    ///   terminal state out from under a caller that just wanted to
    ///   make sure the conversation was flushed.
    ///
    /// Exists to paper over the Anthropic-API constraint that every
    /// `tool_use` content block must be immediately followed by a
    /// matching `tool_result` block. Callers include: `SendUserMessage`
    /// while the thread is `AwaitingTools` (would otherwise append a
    /// bare `Role::User` after `assistant[tool_use]`) and the resume
    /// path on startup (would otherwise drop `AwaitingTools::completed`
    /// on the floor when marking the thread Failed).
    pub fn heal_to_idle(&mut self, reason: &str, events: &mut Vec<ThreadEvent>) -> Vec<String> {
        match &self.internal {
            ThreadInternalState::Idle
            | ThreadInternalState::Completed
            | ThreadInternalState::Failed { .. }
            | ThreadInternalState::Cancelled => return Vec::new(),
            ThreadInternalState::NeedsModelCall
            | ThreadInternalState::AwaitingModel { .. }
            | ThreadInternalState::WaitingOnResources { .. } => {
                self.internal = ThreadInternalState::Idle;
                self.touch();
                return Vec::new();
            }
            ThreadInternalState::AwaitingTools { .. } => {}
        }
        // `std::mem::replace` lets us move the fields out without
        // re-borrowing — we take ownership of `pending_dispatch`,
        // `pending_io`, and `completed`, then reinstate the state
        // below with `Idle`.
        let ThreadInternalState::AwaitingTools {
            pending_dispatch,
            pending_io,
            mut completed,
        } = std::mem::replace(&mut self.internal, ThreadInternalState::Idle)
        else {
            unreachable!("matched guard above");
        };

        let mut interrupted: Vec<String> = Vec::new();
        let synth_text = format!("tool call interrupted: {reason}");
        for req in pending_dispatch {
            events.push(ThreadEvent::ToolCallEnd {
                tool_use_id: req.tool_use_id.clone(),
                result_preview: synth_text.clone(),
                is_error: true,
            });
            completed.push(ContentBlock::ToolResult {
                tool_use_id: req.tool_use_id.clone(),
                content: ToolResultContent::Text(synth_text.clone()),
                is_error: true,
            });
            interrupted.push(req.tool_use_id);
        }
        for (_op_id, tool_use_id) in pending_io {
            events.push(ThreadEvent::ToolCallEnd {
                tool_use_id: tool_use_id.clone(),
                result_preview: synth_text.clone(),
                is_error: true,
            });
            completed.push(ContentBlock::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: ToolResultContent::Text(synth_text.clone()),
                is_error: true,
            });
            interrupted.push(tool_use_id);
        }
        if !completed.is_empty() {
            self.conversation
                .push(Message::tool_result_blocks(completed));
        }
        self.touch();
        interrupted
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

    /// Cancel this thread: close any trailing open ToolUse with an
    /// `is_error` ToolResult, then transition to
    /// [`ThreadInternalState::Cancelled`]. Synthesis events are pushed
    /// into `events` for the caller to broadcast — webui tool-call
    /// rows need the `ToolCallEnd` so they stop spinning.
    ///
    /// The `AwaitingTools` branch preserves already-completed tool
    /// results (real work) and merges synthesized fillers for the
    /// still-in-flight / still-queued ones into a single tool_result
    /// message. Other states delegate to
    /// [`Self::synthesize_trailing_tool_results`] as defensive heal
    /// for any orphaned ToolUse a wedged state might leave behind.
    pub fn cancel(&mut self, events: &mut Vec<ThreadEvent>) {
        if let ThreadInternalState::AwaitingTools { completed, .. } = &mut self.internal {
            let mut merged = std::mem::take(completed);
            let matched: std::collections::HashSet<String> = merged
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                    _ => None,
                })
                .collect();
            if let Some(last) = self.conversation.messages().last()
                && last.role == Role::Assistant
            {
                let synth_text = format!("tool call interrupted: {CANCEL_REASON}");
                for block in &last.content {
                    if let ContentBlock::ToolUse { id, .. } = block
                        && !matched.contains(id)
                    {
                        merged.push(synth_interrupted_tool_result(id, &synth_text, events));
                    }
                }
            }
            if !merged.is_empty() {
                self.conversation.push(Message::tool_result_blocks(merged));
            }
        } else {
            self.synthesize_trailing_tool_results(CANCEL_REASON, events);
        }
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

    /// Promote a `Failed` thread back to `Idle` so it can accept a new
    /// user message. Returns `false` (with no state change) if the
    /// thread isn't in `Failed`.
    ///
    /// Defensive tail-heal: if the conversation ends in a
    /// `Role::Assistant` message with `ToolUse` blocks that aren't
    /// followed by a `Role::ToolResult`, synth `is_error: true`
    /// tool_result blocks for them first. The fail-path in
    /// `pod::persist::load_one` already heals via `heal_to_idle`
    /// before marking the thread Failed, so this only matters for
    /// threads persisted before that heal wiring existed — cheap
    /// insurance, no cost for threads that were already clean.
    pub fn recover(&mut self, events: &mut Vec<ThreadEvent>) -> bool {
        if !matches!(self.internal, ThreadInternalState::Failed { .. }) {
            return false;
        }
        self.synthesize_trailing_tool_results("recovered after failure", events);
        self.internal = ThreadInternalState::Idle;
        self.touch();
        true
    }

    fn synthesize_trailing_tool_results(&mut self, reason: &str, events: &mut Vec<ThreadEvent>) {
        let Some(last) = self.conversation.messages().last() else {
            return;
        };
        if last.role != Role::Assistant {
            return;
        }
        let pending_ids: Vec<String> = last
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect();
        if pending_ids.is_empty() {
            return;
        }
        let synth_text = format!("tool call interrupted: {reason}");
        let blocks: Vec<ContentBlock> = pending_ids
            .iter()
            .map(|id| synth_interrupted_tool_result(id, &synth_text, events))
            .collect();
        self.conversation.push(Message::tool_result_blocks(blocks));
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

        // Scope admission lives at the scheduler's Function registry
        // (see `register_tool_function` in
        // `src/runtime/scheduler/functions.rs`): denied calls arrive
        // here as `is_error: true` tool_results, admitted calls run as
        // ordinary tool IO.
        let _ = tool_annotations;
        self.internal = ThreadInternalState::AwaitingTools {
            pending_dispatch: make_dispatch_order(tool_uses),
            pending_io: HashMap::new(),
            completed: Vec::new(),
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
        } = std::mem::replace(&mut self.internal, ThreadInternalState::Idle)
        else {
            unreachable!("matched guard ensures we're in AwaitingTools")
        };

        // The op_id should be tracked. Use it to verify; tool_use_id is the canonical
        // key for the conversation block either way.
        pending_io.remove(&op_id);

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

    fn base_config_for_fork() -> ThreadConfig {
        ThreadConfig {
            model: "test".into(),
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
            Scope::allow_all(),
            ToolSurface::default(),
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

    fn thread_awaiting_tools(
        pending_dispatch: Vec<ToolUseReq>,
        pending_io: HashMap<OpId, String>,
        completed: Vec<ContentBlock>,
    ) -> Thread {
        let mut task = Thread::new(
            "t".into(),
            "pod".into(),
            base_config_for_fork(),
            ThreadBindings::default(),
            Scope::allow_all(),
            ToolSurface::default(),
        );
        // Seed a prior assistant turn with the tool_uses so
        // `find_tool_name` / `find_tool_args` can resolve them if the
        // integration path ever walks them back. Not strictly needed
        // for `heal_to_idle`, but keeps the conversation
        // well-formed at the tool_use layer.
        let mut tool_use_blocks: Vec<ContentBlock> = Vec::new();
        for req in &pending_dispatch {
            tool_use_blocks.push(ContentBlock::ToolUse {
                id: req.tool_use_id.clone(),
                name: req.name.clone(),
                input: req.input.clone(),
                replay: None,
            });
        }
        for tool_use_id in pending_io.values() {
            tool_use_blocks.push(ContentBlock::ToolUse {
                id: tool_use_id.clone(),
                name: "test_tool".into(),
                input: serde_json::json!({}),
                replay: None,
            });
        }
        task.conversation
            .push(Message::assistant_blocks(tool_use_blocks));
        task.internal = ThreadInternalState::AwaitingTools {
            pending_dispatch,
            pending_io,
            completed,
        };
        task
    }

    #[test]
    fn heal_to_idle_noop_on_terminal_states() {
        let mut task = thread_with_two_turns(); // state = Idle
        let mut events = Vec::new();
        let interrupted = task.heal_to_idle("boom", &mut events);
        assert!(interrupted.is_empty());
        assert!(events.is_empty());
        // Conversation unchanged; Idle stays Idle.
        assert_eq!(task.conversation.messages().len(), 4);
        assert!(matches!(task.internal, ThreadInternalState::Idle));

        // Failed is a terminal state: heal_to_idle leaves it alone so a
        // later explicit `recover()` can own the Failed → Idle
        // promotion.
        task.fail("test", "boom");
        let interrupted = task.heal_to_idle("second", &mut events);
        assert!(interrupted.is_empty());
        assert!(events.is_empty());
        assert!(matches!(task.internal, ThreadInternalState::Failed { .. }));
    }

    #[test]
    fn heal_to_idle_flips_non_awaiting_work_states() {
        let mut task = thread_with_two_turns();
        let mut events = Vec::new();

        task.internal = ThreadInternalState::NeedsModelCall;
        let interrupted = task.heal_to_idle("reset", &mut events);
        assert!(interrupted.is_empty());
        assert!(events.is_empty());
        assert!(matches!(task.internal, ThreadInternalState::Idle));

        task.internal = ThreadInternalState::AwaitingModel {
            op_id: 42,
            started_at: Utc::now(),
        };
        let interrupted = task.heal_to_idle("reset", &mut events);
        assert!(interrupted.is_empty());
        assert!(events.is_empty());
        assert!(matches!(task.internal, ThreadInternalState::Idle));

        task.internal = ThreadInternalState::WaitingOnResources {
            needed: vec!["he-x".into()],
        };
        let interrupted = task.heal_to_idle("reset", &mut events);
        assert!(interrupted.is_empty());
        assert!(events.is_empty());
        assert!(matches!(task.internal, ThreadInternalState::Idle));

        // Conversation untouched through all three flips.
        assert_eq!(task.conversation.messages().len(), 4);
    }

    #[test]
    fn heal_to_idle_synthesizes_for_in_flight_and_queued() {
        // Two categories: one already-dispatched (in pending_io,
        // op_id 5), one still queued (in pending_dispatch, no op_id
        // yet). Plus one previously-completed result we must
        // preserve in the output message.
        let mut pending_io = HashMap::new();
        pending_io.insert(5, "toolu_inflight".to_string());
        let pending_dispatch = vec![ToolUseReq {
            tool_use_id: "toolu_queued".into(),
            name: "bash".into(),
            input: serde_json::json!({"command": "ls"}),
        }];
        let completed = vec![ContentBlock::ToolResult {
            tool_use_id: "toolu_done".into(),
            content: ToolResultContent::Text("ok".into()),
            is_error: false,
        }];
        let mut task = thread_awaiting_tools(pending_dispatch, pending_io, completed);

        let mut events = Vec::new();
        let mut interrupted = task.heal_to_idle("tests", &mut events);
        interrupted.sort();
        assert_eq!(
            interrupted,
            vec!["toolu_inflight".to_string(), "toolu_queued".into()]
        );
        // One ToolCallEnd event per synthesized result.
        assert_eq!(events.len(), 2);
        for ev in &events {
            let ThreadEvent::ToolCallEnd {
                is_error,
                result_preview,
                ..
            } = ev
            else {
                panic!("expected ToolCallEnd, got {ev:?}");
            };
            assert!(*is_error);
            assert!(result_preview.contains("tests"));
        }

        // State cleared; last conversation message is a Role::ToolResult
        // carrying: the already-completed block + one synthesized block
        // per interrupted tool_use_id.
        assert!(matches!(task.internal, ThreadInternalState::Idle));
        let last = task.conversation.messages().last().unwrap();
        assert_eq!(last.role, Role::ToolResult);
        let ids: Vec<&str> = last
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        assert!(ids.contains(&"toolu_done"));
        assert!(ids.contains(&"toolu_inflight"));
        assert!(ids.contains(&"toolu_queued"));
    }

    #[test]
    fn heal_then_fail_preserves_awaiting_tools_results() {
        // Mimics the persist.rs resume path: an AwaitingTools thread
        // loaded from disk with `completed` results. Without the heal
        // step, `fail()` replaces `internal` and those completed
        // results vanish from the thread (the conversation only has
        // the assistant[tool_use] turn, with no tool_result follow-up).
        // With the heal step, the completed results + synth errors
        // for pending calls land in the conversation before Failed.
        let mut pending_io = HashMap::new();
        pending_io.insert(9, "toolu_inflight".to_string());
        let pending_dispatch = vec![ToolUseReq {
            tool_use_id: "toolu_queued".into(),
            name: "bash".into(),
            input: serde_json::json!({}),
        }];
        let completed = vec![ContentBlock::ToolResult {
            tool_use_id: "toolu_done".into(),
            content: ToolResultContent::Text("ok".into()),
            is_error: false,
        }];
        let mut task = thread_awaiting_tools(pending_dispatch, pending_io, completed);

        let mut events = Vec::new();
        task.heal_to_idle("resume", &mut events);
        task.fail("resume", "was in-flight at shutdown");

        assert!(matches!(task.internal, ThreadInternalState::Failed { .. }));
        // Conversation ends in a tool_result message carrying all
        // three ids (one previously-completed, two synthesized).
        let last = task.conversation.messages().last().unwrap();
        assert_eq!(last.role, Role::ToolResult);
        let ids: Vec<&str> = last
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        assert!(ids.contains(&"toolu_done"));
        assert!(ids.contains(&"toolu_inflight"));
        assert!(ids.contains(&"toolu_queued"));
    }

    #[test]
    fn recover_rejects_non_failed_states() {
        let mut task = thread_with_two_turns(); // Idle
        let mut events = Vec::new();
        assert!(!task.recover(&mut events));
        assert!(matches!(task.internal, ThreadInternalState::Idle));

        task.internal = ThreadInternalState::Completed;
        assert!(!task.recover(&mut events));
        assert!(matches!(task.internal, ThreadInternalState::Completed));

        task.internal = ThreadInternalState::Cancelled;
        assert!(!task.recover(&mut events));
        assert!(matches!(task.internal, ThreadInternalState::Cancelled));

        task.internal = ThreadInternalState::NeedsModelCall;
        assert!(!task.recover(&mut events));
        assert!(matches!(task.internal, ThreadInternalState::NeedsModelCall));

        assert!(events.is_empty());
    }

    #[test]
    fn cancel_in_awaiting_tools_preserves_completed_and_synthesizes_missing() {
        // Mid-tool-dispatch cancel: `completed` has one real result,
        // one tool is in-flight, one is queued. After cancel the
        // conversation must end with a single tool_result message
        // carrying all three ids — preserved real result plus two
        // is_error synthesized results.
        let mut pending_io = HashMap::new();
        pending_io.insert(9, "toolu_inflight".to_string());
        let pending_dispatch = vec![ToolUseReq {
            tool_use_id: "toolu_queued".into(),
            name: "bash".into(),
            input: serde_json::json!({}),
        }];
        let completed = vec![ContentBlock::ToolResult {
            tool_use_id: "toolu_done".into(),
            content: ToolResultContent::Text("real-result".into()),
            is_error: false,
        }];
        let mut task = thread_awaiting_tools(pending_dispatch, pending_io, completed);

        let mut events = Vec::new();
        task.cancel(&mut events);

        assert!(matches!(task.internal, ThreadInternalState::Cancelled));
        // Two ToolCallEnd events — one per synthesized interrupted call.
        assert_eq!(events.len(), 2);
        let ended_ids: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ThreadEvent::ToolCallEnd { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        assert!(ended_ids.contains(&"toolu_inflight"));
        assert!(ended_ids.contains(&"toolu_queued"));

        let last = task.conversation.messages().last().unwrap();
        assert_eq!(last.role, Role::ToolResult);
        // Real result preserved verbatim; synthesized results are is_error.
        let mut saw_real = false;
        let mut saw_synth_inflight = false;
        let mut saw_synth_queued = false;
        for b in &last.content {
            let ContentBlock::ToolResult {
                tool_use_id,
                is_error,
                content,
            } = b
            else {
                continue;
            };
            match tool_use_id.as_str() {
                "toolu_done" => {
                    assert!(!is_error);
                    assert!(matches!(content, ToolResultContent::Text(t) if t == "real-result"));
                    saw_real = true;
                }
                "toolu_inflight" => {
                    assert!(*is_error);
                    saw_synth_inflight = true;
                }
                "toolu_queued" => {
                    assert!(*is_error);
                    saw_synth_queued = true;
                }
                other => panic!("unexpected tool_use_id: {other}"),
            }
        }
        assert!(saw_real && saw_synth_inflight && saw_synth_queued);
    }

    #[test]
    fn cancel_in_awaiting_model_leaves_conversation_untouched() {
        // A cancel during AwaitingModel happens before the assistant
        // turn has been persisted (partial deltas are broadcast but
        // not in the conversation). No synthesis needed — the
        // conversation stays well-formed as-is.
        let mut task = thread_with_two_turns();
        task.internal = ThreadInternalState::AwaitingModel {
            op_id: 1,
            started_at: Utc::now(),
        };
        let before_len = task.conversation.messages().len();
        let mut events = Vec::new();
        task.cancel(&mut events);

        assert!(matches!(task.internal, ThreadInternalState::Cancelled));
        assert_eq!(task.conversation.messages().len(), before_len);
        assert!(events.is_empty());
    }

    #[test]
    fn recover_flips_failed_to_idle() {
        let mut task = thread_with_two_turns();
        task.fail("model_call", "nope");
        let before = task.last_active;
        std::thread::sleep(std::time::Duration::from_millis(2));
        let mut events = Vec::new();
        assert!(task.recover(&mut events));
        assert!(matches!(task.internal, ThreadInternalState::Idle));
        assert!(task.last_active > before);
        // Conversation untouched — no dangling tool_use at the tail.
        assert_eq!(task.conversation.messages().len(), 4);
        assert!(events.is_empty());
    }

    #[test]
    fn recover_synthesizes_tool_results_for_dangling_tool_use() {
        // Simulates a Failed thread persisted before the heal-at-fail
        // wiring existed: the conversation ends in assistant[tool_use]
        // with no following tool_result. Recovery must synth filler so
        // the next model call doesn't 400 on shape.
        let mut task = thread_with_two_turns();
        task.conversation.push(Message::assistant_blocks(vec![
            ContentBlock::ToolUse {
                id: "toolu_orphan_a".into(),
                name: "bash".into(),
                input: serde_json::json!({}),
                replay: None,
            },
            ContentBlock::ToolUse {
                id: "toolu_orphan_b".into(),
                name: "bash".into(),
                input: serde_json::json!({}),
                replay: None,
            },
        ]));
        task.fail("resume", "in-flight at shutdown");
        let prior_len = task.conversation.messages().len();

        let mut events = Vec::new();
        assert!(task.recover(&mut events));
        assert!(matches!(task.internal, ThreadInternalState::Idle));

        // A new tool_result message landed at the tail covering both ids.
        assert_eq!(task.conversation.messages().len(), prior_len + 1);
        let last = task.conversation.messages().last().unwrap();
        assert_eq!(last.role, Role::ToolResult);
        let ids: Vec<&str> = last
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult {
                    tool_use_id,
                    is_error: true,
                    ..
                } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        assert!(ids.contains(&"toolu_orphan_a"));
        assert!(ids.contains(&"toolu_orphan_b"));
        // One ToolCallEnd event per synthesized result.
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn submit_user_message_after_interrupt_reaches_needs_model_call() {
        // End-to-end shape: AwaitingTools → interrupt → submit_user
        // leaves the thread with a valid conversation
        // (tool_result message sits between the assistant tool_use
        // and the new user message) and state = NeedsModelCall.
        let mut pending_io = HashMap::new();
        pending_io.insert(7, "toolu_x".to_string());
        let mut task = thread_awaiting_tools(Vec::new(), pending_io, Vec::new());
        let mut events = Vec::new();
        let interrupted = task.heal_to_idle("new_user_msg", &mut events);
        assert_eq!(interrupted, vec!["toolu_x".to_string()]);
        task.submit_user_message("follow-up".into(), Vec::new());
        assert!(matches!(task.internal, ThreadInternalState::NeedsModelCall));
        let msgs = task.conversation.messages();
        // [assistant[tool_use], tool_result, user]
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::Assistant);
        assert_eq!(msgs[1].role, Role::ToolResult);
        assert_eq!(msgs[2].role, Role::User);
    }
}
