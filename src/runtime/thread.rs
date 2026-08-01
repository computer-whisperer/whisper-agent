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

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::{
    BehaviorOrigin, ContentBlock, Conversation, GenerationContext, ImageSource, Message, Role,
    ThreadBindings, ThreadConfig, ThreadSnapshot, ThreadStateLabel, ThreadSummary,
    ToolResultContent, ToolSurface, TurnEntry, TurnLog, Usage,
};

use crate::permission::Scope;
use crate::providers::model::ModelResponse;
use crate::tools::mcp::CallToolResult;

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
    generation: &GenerationContext,
    events: &mut Vec<ThreadEvent>,
) -> ContentBlock {
    events.push(ThreadEvent::ToolCallEnd {
        generation: generation.clone(),
        tool_use_id: tool_use_id.to_string(),
        result_preview: synth_text.to_string(),
        is_error: true,
        attachments: Vec::new(),
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
    /// Legacy load shims for thread JSON written when driver state, the
    /// effect journal, and (earlier still) the raw cycle counter lived on
    /// the thread. Driver policy now lives on the ticking weave
    /// (`crate::runtime::weave`); load lifts these into the singleton
    /// weave and new snapshots omit all three fields.
    #[serde(default, skip_serializing)]
    pub driver_state: crate::runtime::driver::DriverState,
    #[serde(default, skip_serializing)]
    pub effect_journal: crate::runtime::driver::DriverEffectJournal,
    #[serde(default, skip_serializing)]
    pub turns_in_cycle: u32,
    /// The thread's permission scope, snapshotted from the pod's
    /// `[allow]` table at creation. Stored on the thread rather than
    /// looked up from the pod so mid-flight edits to the pod's allow
    /// table don't retroactively change a thread's active scope —
    /// pod-file edits apply to *future* threads. `sudo` grants widen
    /// this in place for one approved call.
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
    /// Server-generated knowledge nudges waiting to be inserted before
    /// the next model sub-turn. Transient: autoquery is opportunistic,
    /// so a restart can drop an in-flight nudge without corrupting the
    /// conversation.
    #[serde(default, skip)]
    pub pending_knowledge_nudges: Vec<String>,
    /// Knowledge hit keys already surfaced to this thread, either via
    /// explicit `knowledge_query` tool results or automatic knowledge
    /// nudges. Persisted with the thread so opportunistic retrieval
    /// does not repeat the same source record after a restart.
    #[serde(default)]
    pub seen_knowledge_hits: HashSet<String>,
    /// One-shot suppression flag for auto-injected nudges after the
    /// model explicitly asked for knowledge (`knowledge_query`) or
    /// manually drained queued nudges. Transient: after a restart,
    /// the persisted conversation already carries what was shown.
    #[serde(default, skip)]
    pub suppress_next_knowledge_nudge: bool,
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
    /// Shared per-turn sticky-routing slot, lifecycle-scoped to one
    /// user-message cycle (the tool loop kicked off by a single user
    /// input). Reset to an empty `OnceLock` inside
    /// [`Self::submit_user_message`]. The OpenAI Codex adapter reads
    /// the value (if any) and forwards it as the
    /// `x-codex-turn-state` request header; when the server's first
    /// response of the turn carries that header back it stores it
    /// here, and every later call within the turn replays it so the
    /// chatgpt.com backend keeps routing them to the same cache
    /// shard. Transient — recovering across restarts requires a
    /// fresh turn anyway.
    #[serde(default, skip)]
    pub turn_routing_token: std::sync::Arc<std::sync::OnceLock<String>>,
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
        #[serde(default)]
        generation: GenerationContext,
        /// Zero only for legacy snapshots created before effect journaling.
        #[serde(default)]
        effect_id: crate::runtime::driver::DriverEffectId,
    },
    /// Model responded with tool_uses. Each entry in `pending_dispatch` still needs to
    /// be fired at MCP; `pending_io` maps op_ids of in-flight tool calls to their
    /// tool_use_ids; `completed` accumulates ToolResult blocks as they return. A tool
    /// denied by the thread's scope is synthesized into `completed` here with
    /// `is_error: true` so the model's next turn sees the denial.
    AwaitingTools {
        #[serde(default)]
        generation: GenerationContext,
        /// The pending `DispatchTools` record for this generation. Zero is the
        /// legacy sentinel for snapshots with no effect journal.
        #[serde(default)]
        effect_id: crate::runtime::driver::DriverEffectId,
        pending_dispatch: Vec<ToolUseReq>,
        pending_io: HashMap<OpId, String>,
        completed: Vec<ContentBlock>,
    },
    /// Model response integrated; the tool calls it requested (possibly
    /// none) await driver interpretation via the ticking weave. Resolved
    /// within the same scheduler loop iteration in practice; persisted so
    /// the machine stays total across a crash at the boundary.
    AgentBoundary {
        #[serde(default)]
        generation: GenerationContext,
        /// The `RunAgent` record this response resolves. Zero is the
        /// legacy sentinel for snapshots with no effect journal.
        #[serde(default)]
        effect_id: crate::runtime::driver::DriverEffectId,
        #[serde(default)]
        pending_tool_uses: Vec<ToolUseReq>,
    },
    /// All requested tools resolved and their results appended; the cycle
    /// continuation awaits driver interpretation via the ticking weave.
    ToolsBoundary {
        #[serde(default)]
        generation: GenerationContext,
        /// The `DispatchTools` record these results resolve. Zero is the
        /// legacy sentinel for snapshots with no effect journal.
        #[serde(default)]
        effect_id: crate::runtime::driver::DriverEffectId,
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

/// Per-tool admission decision for [`Thread::resolve_tool_dispatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDecision {
    pub tool_use_id: String,
    pub allow: bool,
    /// Denial text shown to the model as the error tool_result.
    pub message: Option<String>,
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
        generation: GenerationContext,
    },
    ToolCall {
        op_id: OpId,
        generation: GenerationContext,
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
    /// A policy boundary. The thread does not know what happens next; the
    /// scheduler routes this through the ticking weave's driver and applies
    /// the returned effect via [`Thread::begin_model_call`] /
    /// [`Thread::begin_tool_dispatch`] / [`Thread::continue_cycle`] /
    /// [`Thread::finish_cycle`].
    Boundary(ThreadBoundary),
}

/// Policy boundary reached by the mechanical state machine.
#[derive(Debug, Clone)]
pub enum ThreadBoundary {
    /// `NeedsModelCall`: a runnable turn boundary — who (if anyone) runs
    /// next is the driver's call.
    TurnStart,
    /// A model response has been integrated into the conversation.
    AgentCompleted {
        generation: GenerationContext,
        /// The `RunAgent` record this response resolves (zero = legacy).
        effect_id: crate::runtime::driver::DriverEffectId,
        has_tool_calls: bool,
    },
    /// Every requested tool has resolved and the results are appended.
    ToolsCompleted {
        generation: GenerationContext,
        /// The `DispatchTools` record these results resolve (zero = legacy).
        effect_id: crate::runtime::driver::DriverEffectId,
    },
}

/// Internal task event — translated by the scheduler into wire [`ServerToClient`] events.
#[derive(Debug, Clone)]
pub enum ThreadEvent {
    AssistantBegin {
        generation: GenerationContext,
        turn: u32,
    },
    ToolCallBegin {
        generation: GenerationContext,
        tool_use_id: String,
        name: String,
        args_preview: String,
        /// Full tool arguments (untruncated). Used by the webui for
        /// rich tool-specific renderers (e.g. unified diffs for
        /// edit_file). The router forwards this onto the wire.
        args: serde_json::Value,
    },
    ToolCallEnd {
        generation: GenerationContext,
        tool_use_id: String,
        result_preview: String,
        is_error: bool,
        /// Image attachments lifted from the tool result's content
        /// blocks. Carried alongside `result_preview` so the live
        /// streaming path delivers the same images the snapshot
        /// path would — without this, MCP tools that return image
        /// content (mcp-imagegen, recall_image, …) wouldn't render
        /// in the webui until a refresh forced a snapshot resync.
        /// URL-source images stay too — the renderer picks how to
        /// display each.
        attachments: Vec<ImageSource>,
    },
    AssistantEnd {
        generation: GenerationContext,
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
    /// Participant currently driving the model/tool loop. Outside an active
    /// generation, fall back to the compatibility responder.
    pub fn active_participant_id(&self) -> &whisper_agent_protocol::ParticipantId {
        match &self.internal {
            ThreadInternalState::AwaitingModel { generation, .. }
            | ThreadInternalState::AwaitingTools { generation, .. }
            | ThreadInternalState::AgentBoundary { generation, .. }
            | ThreadInternalState::ToolsBoundary { generation, .. } => &generation.participant_id,
            _ => &self.config.participants.default_responder,
        }
    }

    pub fn scope_for(&self, participant_id: &whisper_agent_protocol::ParticipantId) -> &Scope {
        self.config
            .participant_profiles
            .get(participant_id)
            .map(|profile| &profile.scope)
            .unwrap_or(&self.scope)
    }

    pub fn bindings_for(
        &self,
        participant_id: &whisper_agent_protocol::ParticipantId,
    ) -> &ThreadBindings {
        self.config
            .participant_profiles
            .get(participant_id)
            .map(|profile| &profile.bindings)
            .unwrap_or(&self.bindings)
    }

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
            driver_state: Default::default(),
            effect_journal: Default::default(),
            turns_in_cycle: 0,
            scope,
            tool_surface,
            origin: None,
            pending_tool_result_followups: Vec::new(),
            pending_knowledge_nudges: Vec::new(),
            seen_knowledge_hits: HashSet::new(),
            suppress_next_knowledge_nudge: false,
            dispatched_by: None,
            dispatch_depth: 0,
            draft: String::new(),
            internal: ThreadInternalState::Idle,
            turn_routing_token: std::sync::Arc::new(std::sync::OnceLock::new()),
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
    /// Rejects mid-turn sources (working, awaiting approval) — the
    /// in-flight operation has no meaning in the derived conversation.
    /// (Mid-compaction forks are refused at the scheduler level, where
    /// the weave's compacting marker lives.) Rejects non-user-role
    /// indices —
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
            driver_state: Default::default(),
            effect_journal: Default::default(),
            turns_in_cycle: 0,
            scope: self.scope.clone(),
            tool_surface: self.tool_surface.clone(),
            origin: None,
            pending_tool_result_followups: Vec::new(),
            pending_knowledge_nudges: Vec::new(),
            seen_knowledge_hits: HashSet::new(),
            suppress_next_knowledge_nudge: false,
            dispatched_by: None,
            dispatch_depth: 0,
            // Client seeds the new thread's draft with the forked-from
            // user-message text via a follow-up `SetThreadDraft`; the
            // source's in-progress draft would be the wrong thing to
            // carry over.
            draft: String::new(),
            internal: ThreadInternalState::Idle,
            turn_routing_token: std::sync::Arc::new(std::sync::OnceLock::new()),
        })
    }

    pub fn public_state(&self) -> ThreadStateLabel {
        match &self.internal {
            ThreadInternalState::Idle => ThreadStateLabel::Idle,
            ThreadInternalState::Completed => ThreadStateLabel::Completed,
            ThreadInternalState::WaitingOnResources { .. }
            | ThreadInternalState::NeedsModelCall
            | ThreadInternalState::AwaitingModel { .. }
            | ThreadInternalState::AwaitingTools { .. }
            | ThreadInternalState::AgentBoundary { .. }
            | ThreadInternalState::ToolsBoundary { .. } => ThreadStateLabel::Working,
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
            dispatched_by: self.dispatched_by.clone(),
            // A thread has no knowledge of its weave relationships; the
            // scheduler decorates these from the ticker index before
            // anything client-facing ships (`Scheduler::decorate_summary`).
            weave_id: None,
            weave_role: None,
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
    /// `attachments` are media items the client bundled with the message
    /// (images today, audio/documents later). They're lowered into
    /// [`ContentBlock`]s and appended after the text block, in arrival
    /// order — providers receive `[text, image1, image2, ...]` which
    /// matches the "images come before prompt" guidance when the text
    /// block itself is the question.
    ///
    /// `pending_resources` is the subset of `bindings.*` ids the scheduler
    /// hasn't observed transition to Ready yet — empty means "go straight
    /// to NeedsModelCall." When non-empty the thread parks in
    /// `WaitingOnResources` and the scheduler nudges it via
    /// [`Self::clear_waiting_resource`] as each id flips Ready.
    pub fn submit_user_message(
        &mut self,
        text: String,
        attachments: Vec<whisper_agent_protocol::Attachment>,
        pending_resources: Vec<String>,
    ) {
        let msg = if attachments.is_empty() {
            Message::user_text(text)
        } else {
            let mut blocks = Vec::with_capacity(1 + attachments.len());
            if !text.is_empty() {
                blocks.push(whisper_agent_protocol::ContentBlock::Text { text });
            }
            for a in attachments {
                blocks.push(a.into_content_block());
            }
            Message::user_blocks(blocks)
        }
        .with_author(self.config.participants.default_input.clone());
        self.conversation.push(msg);
        // Fresh per-turn sticky-routing slot. A new user message marks
        // the boundary between turns in codex's vocabulary — replaying
        // a prior turn's `x-codex-turn-state` into the next turn would
        // violate the client/server contract per the codex CLI's own
        // comment (see codex-rs/core/src/client.rs:227).
        self.turn_routing_token = std::sync::Arc::new(std::sync::OnceLock::new());
        self.internal = if pending_resources.is_empty() {
            ThreadInternalState::NeedsModelCall
        } else {
            ThreadInternalState::WaitingOnResources {
                needed: pending_resources,
            }
        };
        self.touch();
    }

    /// Append a server-generated system reminder that should be
    /// consumed by the model as part of the current user cycle. Unlike
    /// a real user message, this does not reset the driver's turn counter;
    /// it is an agent sub-turn nudge, not a new user request.
    pub fn submit_server_nudge(&mut self, text: String, pending_resources: Vec<String>) {
        self.conversation.push(
            Message::system_text(text)
                .with_author(self.config.participants.default_responder.clone()),
        );
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
            | ThreadInternalState::ToolsBoundary { .. }
            | ThreadInternalState::WaitingOnResources { .. } => {
                self.internal = ThreadInternalState::Idle;
                self.touch();
                return Vec::new();
            }
            // The conversation tail is an assistant message whose ToolUse
            // blocks were never dispatched (no Function entries exist for
            // them, so nothing to interrupt at the caller).
            ThreadInternalState::AgentBoundary { .. } => {
                self.internal = ThreadInternalState::Idle;
                self.synthesize_trailing_tool_results(reason, events);
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
            generation,
            effect_id: _,
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
                generation: generation.clone(),
                tool_use_id: req.tool_use_id.clone(),
                result_preview: synth_text.clone(),
                is_error: true,
                attachments: Vec::new(),
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
                generation: generation.clone(),
                tool_use_id: tool_use_id.clone(),
                result_preview: synth_text.clone(),
                is_error: true,
                attachments: Vec::new(),
            });
            completed.push(ContentBlock::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: ToolResultContent::Text(synth_text.clone()),
                is_error: true,
            });
            interrupted.push(tool_use_id);
        }
        if !completed.is_empty() {
            self.conversation.push(
                Message::tool_result_blocks(completed)
                    .with_author(generation.participant_id.clone())
                    .with_run_id(generation.run_id),
            );
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
        self.conversation.push(
            Message::tool_result_text(text)
                .with_author(self.config.participants.default_responder.clone()),
        );
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
        if let ThreadInternalState::AwaitingTools {
            generation,
            completed,
            ..
        } = &mut self.internal
        {
            let generation = generation.clone();
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
                        merged.push(synth_interrupted_tool_result(
                            id,
                            &synth_text,
                            &generation,
                            events,
                        ));
                    }
                }
            }
            if !merged.is_empty() {
                self.conversation.push(
                    Message::tool_result_blocks(merged)
                        .with_author(generation.participant_id)
                        .with_run_id(generation.run_id),
                );
            }
        } else {
            self.synthesize_trailing_tool_results(CANCEL_REASON, events);
        }
        self.internal = ThreadInternalState::Cancelled;
        self.touch();
    }

    /// Terminal failure. The caller (scheduler) resolves any pending
    /// records in the ticking weave's effect journal — the thread no
    /// longer holds driver state.
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
        let generation = GenerationContext {
            run_id: last.run_id.clone().unwrap_or_default(),
            participant_id: last.effective_author(),
        };
        let blocks: Vec<ContentBlock> = pending_ids
            .iter()
            .map(|id| synth_interrupted_tool_result(id, &synth_text, &generation, events))
            .collect();
        self.conversation.push(
            Message::tool_result_blocks(blocks)
                .with_author(generation.participant_id)
                .with_run_id(generation.run_id),
        );
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

    /// Is the task anywhere inside an active cycle (a turn queued, waiting
    /// on resources, or mid-generation)? Ticker release is refused while
    /// this holds — releasing would strand the cycle with no weave to
    /// route its next boundary. Also the persister's in-flight predicate.
    pub fn is_in_flight(&self) -> bool {
        !self.is_idle()
    }

    /// Is a provider request in flight or the tool loop mid-way? Narrower
    /// than [`Self::is_in_flight`]: a queued-but-unbuilt model call
    /// (`NeedsModelCall` / `WaitingOnResources`) does not count, because
    /// the upcoming request is built from the log and will see any entry
    /// appended now. Cross-thread appends are refused while this holds so
    /// the materialized log never contains an entry sequenced before
    /// output that was generated without seeing it.
    pub fn is_mid_generation(&self) -> bool {
        matches!(
            self.internal,
            ThreadInternalState::AwaitingModel { .. }
                | ThreadInternalState::AwaitingTools { .. }
                | ThreadInternalState::AgentBoundary { .. }
                | ThreadInternalState::ToolsBoundary { .. }
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
                // Runnable turn boundary — the ticking weave's driver
                // decides who (if anyone) runs next.
                self.internal = current;
                StepOutcome::Boundary(ThreadBoundary::TurnStart)
            }
            ThreadInternalState::AgentBoundary {
                ref generation,
                effect_id,
                ref pending_tool_uses,
            } => {
                let boundary = ThreadBoundary::AgentCompleted {
                    generation: generation.clone(),
                    effect_id,
                    has_tool_calls: !pending_tool_uses.is_empty(),
                };
                self.internal = current;
                StepOutcome::Boundary(boundary)
            }
            ThreadInternalState::ToolsBoundary {
                ref generation,
                effect_id,
            } => {
                let boundary = ThreadBoundary::ToolsCompleted {
                    generation: generation.clone(),
                    effect_id,
                };
                self.internal = current;
                StepOutcome::Boundary(boundary)
            }
            ThreadInternalState::WaitingOnResources { .. }
            | ThreadInternalState::AwaitingModel { .. } => {
                // Not stepping these — waiting on resource provisioning
                // or a model response.
                self.internal = current;
                StepOutcome::Paused
            }
            ThreadInternalState::AwaitingTools {
                generation,
                effect_id,
                mut pending_dispatch,
                mut pending_io,
                completed,
            } => {
                if let Some(next) = pending_dispatch.pop() {
                    let op_id = next_id(next_op_id);
                    pending_io.insert(op_id, next.tool_use_id.clone());
                    events.push(ThreadEvent::ToolCallBegin {
                        generation: generation.clone(),
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
                        generation: generation.clone(),
                        tool_use_id: next.tool_use_id.clone(),
                        name: next.name.clone(),
                        input: next.input.clone(),
                    };
                    self.internal = ThreadInternalState::AwaitingTools {
                        generation,
                        effect_id,
                        pending_dispatch,
                        pending_io,
                        completed,
                    };
                    self.touch();
                    StepOutcome::DispatchIo(dispatch)
                } else if pending_io.is_empty() {
                    // All tool calls done — append the ToolResult blocks
                    // and surface the boundary on the next step().
                    self.conversation.push(
                        Message::tool_result_blocks(completed)
                            .with_author(generation.participant_id.clone())
                            .with_run_id(generation.run_id.clone()),
                    );
                    self.internal = ThreadInternalState::ToolsBoundary {
                        generation,
                        effect_id,
                    };
                    self.touch();
                    StepOutcome::Continue
                } else {
                    self.internal = ThreadInternalState::AwaitingTools {
                        generation,
                        effect_id,
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
    /// should call `step_until_blocked` afterward (a model result parks the thread at
    /// [`ThreadInternalState::AgentBoundary`], which the step loop routes through the
    /// ticking weave).
    pub fn apply_io_result(
        &mut self,
        op_id: OpId,
        result: IoResult,
        events: &mut Vec<ThreadEvent>,
    ) {
        let prev_public = self.public_state();
        self.apply_io_result_inner(op_id, result, events);
        let new_public = self.public_state();
        if prev_public != new_public {
            events.push(ThreadEvent::StateChanged { state: new_public });
        }
    }

    fn apply_io_result_inner(
        &mut self,
        op_id: OpId,
        result: IoResult,
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
                    op_id: expected,
                    generation,
                    effect_id,
                    ..
                },
                IoResult::ModelCall(res),
            ) if *expected == op_id => match res {
                Ok(response) => {
                    self.integrate_model_response(response, generation.clone(), *effect_id, events)
                }
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

    // ---------- weave effect application ----------
    //
    // Mechanical transitions the scheduler applies after the ticking
    // weave's driver has interpreted a [`ThreadBoundary`]. Each returns
    // false (leaving state untouched) when the thread is not at the
    // boundary the effect targets — a stale or misrouted decision is
    // discarded rather than corrupting the machine.

    /// Apply a `RunAgent` effect at the `TurnStart` boundary. `effect_id`
    /// is the pending `RunAgent` record the scheduler journaled on the
    /// weave before calling this.
    ///
    /// Also applies from `Idle` / `Completed`: a scripted driver may run
    /// a turn on a thread that has no queued input — a freshly derived
    /// thread whose context is its seed, or a finished checker being
    /// reused after a new question was appended. The request is built
    /// from the thread's current log either way.
    pub fn begin_model_call(
        &mut self,
        op_id: OpId,
        generation: GenerationContext,
        effect_id: crate::runtime::driver::DriverEffectId,
        turn: u32,
        events: &mut Vec<ThreadEvent>,
    ) -> Option<IoRequest> {
        if !matches!(
            self.internal,
            ThreadInternalState::NeedsModelCall
                | ThreadInternalState::Idle
                | ThreadInternalState::Completed
        ) {
            return None;
        }
        let prev_public = self.public_state();
        events.push(ThreadEvent::AssistantBegin {
            generation: generation.clone(),
            turn,
        });
        self.internal = ThreadInternalState::AwaitingModel {
            op_id,
            started_at: Utc::now(),
            generation: generation.clone(),
            effect_id,
        };
        self.touch();
        // Driver-run turns start from Idle/Completed, a public-state
        // flip the step loop never sees — emit it here so clients don't
        // show a generating thread as finished.
        let new_public = self.public_state();
        if prev_public != new_public {
            events.push(ThreadEvent::StateChanged { state: new_public });
        }
        Some(IoRequest::ModelCall { op_id, generation })
    }

    /// Apply a `DispatchTools` effect at the `AgentCompleted` boundary.
    /// `effect_id` is the pending `DispatchTools` record the scheduler
    /// journaled on the weave before calling this.
    ///
    /// Scope admission lives at the scheduler's Function registry (see
    /// `register_tool_function` in `src/runtime/scheduler/functions.rs`):
    /// denied calls arrive back as `is_error: true` tool_results, admitted
    /// calls run as ordinary tool IO.
    pub fn begin_tool_dispatch(
        &mut self,
        effect_id: crate::runtime::driver::DriverEffectId,
    ) -> bool {
        if !matches!(self.internal, ThreadInternalState::AgentBoundary { .. }) {
            return false;
        }
        let ThreadInternalState::AgentBoundary {
            generation,
            pending_tool_uses,
            ..
        } = std::mem::replace(&mut self.internal, ThreadInternalState::Idle)
        else {
            unreachable!("matched guard above");
        };
        self.internal = ThreadInternalState::AwaitingTools {
            generation,
            effect_id,
            pending_dispatch: make_dispatch_order(pending_tool_uses),
            pending_io: HashMap::new(),
            completed: Vec::new(),
        };
        self.touch();
        true
    }

    /// Apply a `ResolveTools` effect at the `AgentCompleted` boundary:
    /// per-tool admission decided by the ticking driver. Denied requests
    /// (and any request without a decision) are closed immediately with
    /// synthesized error tool_results; admitted requests dispatch as
    /// usual. With zero admissions the tool_result message flushes on
    /// the next step and the thread surfaces `ToolsCompleted` — the
    /// model sees every denial in one Anthropic-valid batch.
    pub fn resolve_tool_dispatch(
        &mut self,
        effect_id: crate::runtime::driver::DriverEffectId,
        decisions: &[ToolDecision],
        events: &mut Vec<ThreadEvent>,
    ) -> bool {
        if !matches!(self.internal, ThreadInternalState::AgentBoundary { .. }) {
            return false;
        }
        let ThreadInternalState::AgentBoundary {
            generation,
            pending_tool_uses,
            ..
        } = std::mem::replace(&mut self.internal, ThreadInternalState::Idle)
        else {
            unreachable!("matched guard above");
        };
        let mut approved = Vec::new();
        let mut completed = Vec::new();
        for req in pending_tool_uses {
            let decision = decisions.iter().find(|d| d.tool_use_id == req.tool_use_id);
            match decision {
                Some(d) if d.allow => approved.push(req),
                other => {
                    let text = other
                        .and_then(|d| d.message.clone())
                        .unwrap_or_else(|| "tool call denied by driver".to_string());
                    completed.push(synth_interrupted_tool_result(
                        &req.tool_use_id,
                        &text,
                        &generation,
                        events,
                    ));
                }
            }
        }
        self.internal = ThreadInternalState::AwaitingTools {
            generation,
            effect_id,
            pending_dispatch: make_dispatch_order(approved),
            pending_io: HashMap::new(),
            completed,
        };
        self.touch();
        true
    }

    /// Apply a `Continue` effect at the `ToolsCompleted` boundary.
    pub fn continue_cycle(&mut self) -> bool {
        if !matches!(self.internal, ThreadInternalState::ToolsBoundary { .. }) {
            return false;
        }
        self.internal = ThreadInternalState::NeedsModelCall;
        self.touch();
        true
    }

    /// Apply a `Finish` effect at any boundary. Defensive: a driver that
    /// finishes an `AgentCompleted` boundary despite requested tool calls
    /// would orphan the trailing ToolUse blocks, so they are synthesized
    /// closed first.
    pub fn finish_cycle(&mut self, events: &mut Vec<ThreadEvent>) -> bool {
        let orphaned_tool_uses = match &self.internal {
            ThreadInternalState::NeedsModelCall | ThreadInternalState::ToolsBoundary { .. } => {
                false
            }
            ThreadInternalState::AgentBoundary {
                pending_tool_uses, ..
            } => !pending_tool_uses.is_empty(),
            _ => return false,
        };
        let prev_public = self.public_state();
        if orphaned_tool_uses {
            self.synthesize_trailing_tool_results(
                "driver finished with undispatched tool calls",
                events,
            );
        }
        events.push(ThreadEvent::LoopComplete);
        self.internal = ThreadInternalState::Completed;
        self.touch();
        let new_public = self.public_state();
        if prev_public != new_public {
            events.push(ThreadEvent::StateChanged { state: new_public });
        }
        true
    }

    fn integrate_model_response(
        &mut self,
        response: ModelResponse,
        generation: GenerationContext,
        run_effect_id: crate::runtime::driver::DriverEffectId,
        events: &mut Vec<ThreadEvent>,
    ) {
        let ModelResponse {
            content: assistant_blocks,
            stop_reason,
            usage,
        } = response;
        self.total_usage.add(&usage);
        self.turn_log.entries.push(TurnEntry {
            run_id: generation.run_id.clone(),
            participant_id: generation.participant_id.clone(),
            usage,
        });
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
        events.push(ThreadEvent::AssistantEnd {
            generation: generation.clone(),
            stop_reason,
            usage,
        });
        self.conversation.push(
            Message::assistant_blocks(assistant_blocks)
                .with_author(generation.participant_id.clone())
                .with_run_id(generation.run_id.clone()),
        );
        // Park at the boundary; the next step() surfaces it and the
        // scheduler routes it through the ticking weave's driver.
        self.internal = ThreadInternalState::AgentBoundary {
            generation,
            effect_id: run_effect_id,
            pending_tool_uses: tool_uses,
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
            generation,
            effect_id,
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

        let (content, preview, is_error, tool_name, args, err_msg) = match result {
            Ok(r) => {
                let content = mcp_content_to_tool_result(&r.content);
                let preview = tool_result_preview(&content);
                let tool_name = find_tool_name(&self.conversation, &tool_use_id);
                let args = find_tool_args(&self.conversation, &tool_use_id);
                (content, preview, r.is_error, tool_name, args, None)
            }
            Err(msg) => {
                let text = format!("tool invocation failed: {msg}");
                let tool_name = find_tool_name(&self.conversation, &tool_use_id);
                let args = find_tool_args(&self.conversation, &tool_use_id);
                (
                    ToolResultContent::Text(text.clone()),
                    text,
                    true,
                    tool_name,
                    args,
                    Some(msg),
                )
            }
        };

        // Lift any image attachments out of the tool result so the
        // live streaming path delivers them to webui subscribers.
        // Without this, MCP tools that return image content (e.g.
        // mcp-imagegen, recall_image) wouldn't render until a
        // refresh forced a snapshot resync — the image bytes are
        // already in `content` but the wire-level ToolCallEnd only
        // ships `result_preview`.
        //
        // `result_preview` ships the full text now. Worker-side caps
        // (bash: BASH_MAX_OUTPUT_BYTES = 30 KiB, MCP tools similar)
        // already bound the size, so a wire-level second truncate just
        // hides content the UI is otherwise ready to render — the live
        // event was clipping non-streaming tools (read_file, grep, …)
        // to 200 chars even though the snapshot path showed the full
        // text on reload. The field keeps its `_preview` name for
        // protocol stability.
        let attachments: Vec<ImageSource> = content.image_sources().into_iter().cloned().collect();
        events.push(ThreadEvent::ToolCallEnd {
            generation: generation.clone(),
            tool_use_id: tool_use_id.clone(),
            result_preview: preview,
            is_error,
            attachments,
        });
        events.push(ThreadEvent::AuditToolCall {
            tool_name,
            args,
            is_error,
            error_message: err_msg,
        });

        completed.push(ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        });

        self.internal = ThreadInternalState::AwaitingTools {
            generation,
            effect_id,
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

/// Fold an MCP-decoded content vector into our canonical
/// [`ToolResultContent`]. Pure text flows out as `Text`; mixed text +
/// images flow out as `Blocks` so provider adapters can translate each
/// side to the appropriate wire shape. Image blocks whose declared MIME
/// isn't in our accepted set (or whose payload fails to decode) degrade
/// to a text note rather than erroring the call — tools can't always
/// control what their transitive dependencies emit, and a lost
/// thumbnail shouldn't kill a useful result.
fn mcp_content_to_tool_result(blocks: &[crate::tools::mcp::McpContentBlock]) -> ToolResultContent {
    use crate::tools::mcp::McpContentBlock;
    if blocks
        .iter()
        .all(|b| matches!(b, McpContentBlock::Text { .. }))
    {
        let mut out = String::new();
        for b in blocks {
            if let McpContentBlock::Text { text } = b {
                out.push_str(text);
            }
        }
        return ToolResultContent::Text(out);
    }
    let out: Vec<ContentBlock> = blocks.iter().map(|b| b.to_content_block()).collect();
    ToolResultContent::Blocks(out)
}

/// Short human-readable preview of a tool result, used for the
/// `ToolCallEnd` event the webui renders as the collapsed-row label.
/// Text results show their first line; block-form results summarize
/// attachments alongside whatever leading text we have.
fn tool_result_preview(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(s) => s.clone(),
        ToolResultContent::Blocks(blocks) => {
            let mut text = String::new();
            let mut image_count = 0usize;
            for b in blocks {
                match b {
                    ContentBlock::Text { text: t } => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(t);
                    }
                    ContentBlock::Image { .. } => image_count += 1,
                    _ => {}
                }
            }
            match (text.is_empty(), image_count) {
                (true, 0) => String::new(),
                (true, 1) => "[image]".into(),
                (true, n) => format!("[{n} images]"),
                (false, 0) => text,
                (false, 1) => format!("{text} [+image]"),
                (false, n) => format!("{text} [+{n} images]"),
            }
        }
    }
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
    use crate::runtime::driver::{DriverFinishReason, PersistedDriverEffect};

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
            participants: Default::default(),
            driver: Default::default(),
            model: "test".into(),
            max_tokens: 100,
            max_turns: 10,
            compaction: Default::default(),
            autoquery: Default::default(),
            tunables: Default::default(),
            participant_profiles: Default::default(),
        }
    }

    /// Minimal stand-in for the scheduler's boundary router
    /// (`Scheduler::apply_thread_boundary`): route `Boundary` outcomes
    /// through `weave`, apply the decision, and collect dispatched I/O
    /// until the thread pauses. Keeps thread tests exercising the same
    /// choreography the runtime uses.
    fn drive_until_blocked(
        task: &mut Thread,
        weave: &mut crate::runtime::weave::Weave,
        next_op_id: &mut OpId,
        events: &mut Vec<ThreadEvent>,
    ) -> Vec<IoRequest> {
        use crate::runtime::driver::{DriverEffect, DriverFinishReason, PersistedDriverEffect};
        let mut dispatched = Vec::new();
        loop {
            match task.step(next_op_id, events) {
                StepOutcome::DispatchIo(req) => dispatched.push(req),
                StepOutcome::Continue => {}
                StepOutcome::Paused => break,
                StepOutcome::Boundary(ThreadBoundary::TurnStart) => {
                    let effect = weave
                        .next_effect(&task.config.participants, task.config.max_turns)
                        .unwrap();
                    match effect {
                        DriverEffect::RunAgent {
                            participant_id,
                            turn,
                        } => {
                            let generation = GenerationContext::new(
                                uuid::Uuid::new_v4().to_string(),
                                participant_id,
                            );
                            let effect_id =
                                weave.record_pending_effect(PersistedDriverEffect::RunAgent {
                                    generation: generation.clone(),
                                    turn,
                                });
                            let op_id = next_id(next_op_id);
                            let req = task
                                .begin_model_call(op_id, generation, effect_id, turn, events)
                                .expect("RunAgent applies at turn start");
                            dispatched.push(req);
                        }
                        DriverEffect::Finish => {
                            weave.record_completed_effect(PersistedDriverEffect::Finish {
                                generation: None,
                                reason: DriverFinishReason::TurnLimit,
                            });
                            task.finish_cycle(events);
                        }
                        other => panic!("unexpected effect at turn start: {other:?}"),
                    }
                }
                StepOutcome::Boundary(ThreadBoundary::AgentCompleted {
                    generation,
                    effect_id,
                    has_tool_calls,
                }) => {
                    weave.complete_effect(effect_id);
                    let effect = weave
                        .agent_completed(
                            &generation.participant_id,
                            &task.config.participants,
                            has_tool_calls,
                        )
                        .unwrap();
                    match effect {
                        DriverEffect::DispatchTools => {
                            let tool_use_ids = match &task.internal {
                                ThreadInternalState::AgentBoundary {
                                    pending_tool_uses, ..
                                } => pending_tool_uses
                                    .iter()
                                    .map(|t| t.tool_use_id.clone())
                                    .collect(),
                                _ => Vec::new(),
                            };
                            let dispatch_id =
                                weave.record_pending_effect(PersistedDriverEffect::DispatchTools {
                                    generation,
                                    tool_use_ids,
                                });
                            assert!(task.begin_tool_dispatch(dispatch_id));
                        }
                        DriverEffect::Finish => {
                            weave.record_completed_effect(PersistedDriverEffect::Finish {
                                generation: Some(generation),
                                reason: DriverFinishReason::AgentCompleted,
                            });
                            task.finish_cycle(events);
                        }
                        other => panic!("unexpected effect after agent: {other:?}"),
                    }
                }
                StepOutcome::Boundary(ThreadBoundary::ToolsCompleted {
                    generation,
                    effect_id,
                }) => {
                    weave.complete_effect(effect_id);
                    let effect = weave
                        .tools_completed(&generation.participant_id, &task.config.participants)
                        .unwrap();
                    match effect {
                        DriverEffect::Continue => {
                            weave.record_completed_effect(PersistedDriverEffect::Continue {
                                generation,
                            });
                            assert!(task.continue_cycle());
                        }
                        DriverEffect::Finish => {
                            weave.record_completed_effect(PersistedDriverEffect::Finish {
                                generation: Some(generation),
                                reason: DriverFinishReason::ToolsCompleted,
                            });
                            task.finish_cycle(events);
                        }
                        other => panic!("unexpected effect after tools: {other:?}"),
                    }
                }
            }
        }
        dispatched
    }

    fn singleton_weave_for(task: &Thread) -> crate::runtime::weave::Weave {
        let mut weave = crate::runtime::weave::Weave::singleton_for_thread(
            task.id.clone(),
            task.pod_id.clone(),
            task.config.driver.clone(),
        );
        weave.input_accepted().unwrap();
        weave
    }

    #[test]
    fn legacy_driver_fields_lift_into_a_singleton_weave() {
        let task = Thread::new(
            "legacy-driver".into(),
            "pod".into(),
            base_config_for_fork(),
            ThreadBindings::default(),
            Scope::allow_all(),
            ToolSurface::default(),
        );
        // Oldest generation: a bare pre-driver cycle counter.
        let mut json = serde_json::to_value(task).unwrap();
        let object = json.as_object_mut().unwrap();
        object.insert("turns_in_cycle".into(), serde_json::json!(3));

        let mut decoded: Thread = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.turns_in_cycle, 3);

        let mut weave = crate::runtime::weave::Weave::singleton_for_thread(
            decoded.id.clone(),
            decoded.pod_id.clone(),
            decoded.config.driver.clone(),
        );
        weave.import_thread_driver_state(
            std::mem::take(&mut decoded.driver_state),
            std::mem::take(&mut decoded.effect_journal),
            decoded.turns_in_cycle,
        );
        decoded.turns_in_cycle = 0;
        assert_eq!(
            crate::runtime::driver::turns_in_cycle(&weave.driver_state),
            3
        );

        // New thread snapshots carry no driver fields at all.
        let migrated = serde_json::to_value(decoded).unwrap();
        assert!(migrated.get("turns_in_cycle").is_none());
        assert!(migrated.get("driver_state").is_none());
        assert!(migrated.get("effect_journal").is_none());
    }

    #[test]
    fn participant_profile_overrides_execution_scope_and_bindings() {
        let mut task = Thread::new(
            "profiled".into(),
            "pod".into(),
            base_config_for_fork(),
            ThreadBindings {
                backend: "default-backend".into(),
                ..Default::default()
            },
            Scope::allow_all(),
            ToolSurface::default(),
        );
        task.config.participant_profiles.insert(
            "reviewer".into(),
            whisper_agent_protocol::ParticipantExecutionProfile {
                model: "review-model".into(),
                max_tokens: 50,
                system_prompt: "review".into(),
                bindings: ThreadBindings {
                    backend: "review-backend".into(),
                    ..Default::default()
                },
                scope: Scope::deny_all(),
                tool_surface: ToolSurface::default(),
                tunables: Default::default(),
                tools: Vec::new(),
                context: Vec::new(),
            },
        );

        assert_eq!(
            task.bindings_for(&"reviewer".into()).backend,
            "review-backend"
        );
        assert_eq!(
            task.bindings_for(&"missing".into()).backend,
            "default-backend"
        );
        assert_eq!(task.scope_for(&"reviewer".into()), &Scope::deny_all());
        assert_eq!(task.scope_for(&"missing".into()), &Scope::allow_all());
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
            run_id: Default::default(),
            participant_id: whisper_agent_protocol::DEFAULT_MODEL_PARTICIPANT_ID.into(),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 2,
                ..Usage::default()
            },
        });
        task.turn_log.entries.push(TurnEntry {
            run_id: Default::default(),
            participant_id: whisper_agent_protocol::DEFAULT_MODEL_PARTICIPANT_ID.into(),
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
    fn configured_participants_author_new_transcript_messages() {
        let mut config = base_config_for_fork();
        config.participants = whisper_agent_protocol::ThreadParticipants {
            members: vec![
                whisper_agent_protocol::ThreadParticipant {
                    id: "human".into(),
                    kind: whisper_agent_protocol::ThreadParticipantKind::Client,
                    display_name: None,
                },
                whisper_agent_protocol::ThreadParticipant {
                    id: "builder".into(),
                    kind: whisper_agent_protocol::ThreadParticipantKind::Model,
                    display_name: None,
                },
            ],
            default_input: "human".into(),
            default_responder: "builder".into(),
        };
        let mut task = Thread::new(
            "authored".into(),
            "pod".into(),
            config,
            ThreadBindings::default(),
            Scope::allow_all(),
            ToolSurface::default(),
        );
        task.submit_user_message("hello".into(), Vec::new(), Vec::new());
        assert_eq!(
            task.conversation.messages()[0].effective_author().as_str(),
            "human"
        );

        let mut events = Vec::new();
        task.integrate_model_response(
            ModelResponse {
                content: vec![ContentBlock::Text {
                    text: "response".into(),
                }],
                stop_reason: Some("end_turn".into()),
                usage: Usage::default(),
            },
            GenerationContext::new("test-run", "builder"),
            0,
            &mut events,
        );
        // Integration is mechanical now: the thread parks at the agent
        // boundary awaiting the ticking weave's decision.
        assert!(matches!(
            task.internal,
            ThreadInternalState::AgentBoundary { .. }
        ));
        assert_eq!(
            task.conversation.messages()[1].effective_author().as_str(),
            "builder"
        );
        assert_eq!(
            task.conversation.messages()[1]
                .run_id
                .as_ref()
                .map(|id| id.as_str()),
            Some("test-run")
        );
        assert_eq!(task.turn_log.entries[0].participant_id.as_str(), "builder");
        assert_eq!(task.turn_log.entries[0].run_id.as_str(), "test-run");
    }

    #[test]
    fn model_dispatch_has_one_stable_participant_scoped_generation() {
        let mut config = base_config_for_fork();
        config.participants.members[1].id = "builder".into();
        config.participants.default_responder = "builder".into();
        let mut task = Thread::new(
            "generated".into(),
            "pod".into(),
            config,
            ThreadBindings::default(),
            Scope::allow_all(),
            ToolSurface::default(),
        );
        task.submit_user_message("hello".into(), Vec::new(), Vec::new());
        let mut weave = singleton_weave_for(&task);

        let mut next_op_id = 1;
        let mut events = Vec::new();
        let dispatched = drive_until_blocked(&mut task, &mut weave, &mut next_op_id, &mut events);
        let begin = events.iter().find_map(|event| match event {
            ThreadEvent::AssistantBegin { generation, .. } => Some(generation),
            _ => None,
        });
        let Some(begin) = begin else {
            panic!("missing AssistantBegin")
        };
        let [IoRequest::ModelCall { op_id, generation }] = dispatched.as_slice() else {
            panic!("expected exactly one model dispatch, got {dispatched:?}")
        };

        assert!(!generation.run_id.is_empty());
        assert_eq!(generation.participant_id.as_str(), "builder");
        assert_eq!(begin, generation);
        let ThreadInternalState::AwaitingModel {
            generation: persisted,
            effect_id,
            ..
        } = &task.internal
        else {
            panic!("expected AwaitingModel")
        };
        assert_eq!(persisted, generation);
        let record = &weave.effect_journal.records()[0];
        assert_eq!(record.id, *effect_id);
        assert!(matches!(
            &record.effect,
            PersistedDriverEffect::RunAgent {
                generation: recorded,
                turn: 1,
            } if recorded == generation
        ));
        assert_eq!(
            record.outcome,
            crate::runtime::driver::DriverEffectOutcome::Pending
        );

        // The pending record (on the weave) and its internal-state link (on
        // the thread) both serialize before the lazy model future is polled
        // by the scheduler; the thread JSON no longer carries a journal.
        let weave_json = serde_json::to_value(&weave).unwrap();
        assert_eq!(weave_json["effect_journal"]["records"][0]["id"], *effect_id);
        let thread_json = serde_json::to_value(&task).unwrap();
        assert_eq!(thread_json["internal"]["effect_id"], *effect_id);
        assert!(thread_json.get("effect_journal").is_none());

        let op_id = *op_id;
        task.apply_io_result(
            op_id,
            IoResult::ModelCall(Ok(ModelResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: Some("end_turn".into()),
                usage: Usage::default(),
            })),
            &mut events,
        );
        let followup = drive_until_blocked(&mut task, &mut weave, &mut next_op_id, &mut events);
        assert!(followup.is_empty());
        assert!(matches!(task.internal, ThreadInternalState::Completed));
        assert_eq!(weave.effect_journal.records().len(), 2);
        assert!(weave.effect_journal.records().iter().all(|record| {
            record.outcome == crate::runtime::driver::DriverEffectOutcome::Completed
        }));
        assert!(matches!(
            weave.effect_journal.records()[1].effect,
            PersistedDriverEffect::Finish {
                reason: DriverFinishReason::AgentCompleted,
                ..
            }
        ));
    }

    #[test]
    fn tool_cycle_resolves_dispatch_effect_before_driver_continues() {
        use crate::tools::mcp::McpContentBlock;

        let mut task = Thread::new(
            "journal-tools".into(),
            "pod".into(),
            base_config_for_fork(),
            ThreadBindings::default(),
            Scope::allow_all(),
            ToolSurface::default(),
        );
        task.submit_user_message("use a tool".into(), Vec::new(), Vec::new());
        let mut weave = singleton_weave_for(&task);
        let mut next_op_id = 1;
        let mut events = Vec::new();
        let dispatched = drive_until_blocked(&mut task, &mut weave, &mut next_op_id, &mut events);
        let [IoRequest::ModelCall { op_id, .. }] = dispatched.as_slice() else {
            panic!("expected model dispatch, got {dispatched:?}")
        };

        let op_id = *op_id;
        task.apply_io_result(
            op_id,
            IoResult::ModelCall(Ok(ModelResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "toolu-1".into(),
                    name: "lookup".into(),
                    input: serde_json::json!({ "q": "x" }),
                    replay: None,
                }],
                stop_reason: Some("tool_use".into()),
                usage: Usage::default(),
            })),
            &mut events,
        );
        // Driving routes the agent boundary through the weave: the
        // DispatchTools record goes pending, then the queued tool call
        // dispatches.
        let dispatched = drive_until_blocked(&mut task, &mut weave, &mut next_op_id, &mut events);
        let [
            IoRequest::ToolCall {
                op_id: tool_op_id,
                tool_use_id,
                ..
            },
        ] = dispatched.as_slice()
        else {
            panic!("expected tool call, got {dispatched:?}")
        };
        let ThreadInternalState::AwaitingTools { effect_id, .. } = &task.internal else {
            panic!("expected tool dispatch")
        };
        assert_eq!(
            weave.effect_journal.records()[0].outcome,
            crate::runtime::driver::DriverEffectOutcome::Completed
        );
        assert_eq!(weave.effect_journal.records()[1].id, *effect_id);
        assert_eq!(
            weave.effect_journal.records()[1].outcome,
            crate::runtime::driver::DriverEffectOutcome::Pending
        );

        let tool_op_id = *tool_op_id;
        let tool_use_id = tool_use_id.clone();
        task.apply_io_result(
            tool_op_id,
            IoResult::ToolCall {
                tool_use_id,
                result: Ok(CallToolResult {
                    content: vec![McpContentBlock::Text { text: "ok".into() }],
                    is_error: false,
                }),
            },
            &mut events,
        );

        // Driving resolves the tools boundary (Continue) and then the next
        // turn boundary spawns the second model call.
        let dispatched = drive_until_blocked(&mut task, &mut weave, &mut next_op_id, &mut events);
        assert!(matches!(
            dispatched.as_slice(),
            [IoRequest::ModelCall { .. }]
        ));
        assert!(matches!(
            task.internal,
            ThreadInternalState::AwaitingModel { .. }
        ));
        // The journal fully encodes the choreography: first turn's RunAgent
        // and DispatchTools resolved, the synchronous Continue recorded
        // completed, and the second turn's RunAgent pending.
        let records = weave.effect_journal.records();
        assert_eq!(records.len(), 4);
        assert!(matches!(
            records[0].effect,
            PersistedDriverEffect::RunAgent { turn: 1, .. }
        ));
        assert!(matches!(
            records[1].effect,
            PersistedDriverEffect::DispatchTools { .. }
        ));
        assert!(matches!(
            records[2].effect,
            PersistedDriverEffect::Continue { .. }
        ));
        assert!(matches!(
            records[3].effect,
            PersistedDriverEffect::RunAgent { turn: 2, .. }
        ));
        assert!(records[..3].iter().all(|record| {
            record.outcome == crate::runtime::driver::DriverEffectOutcome::Completed
        }));
        assert_eq!(
            records[3].outcome,
            crate::runtime::driver::DriverEffectOutcome::Pending
        );
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
            generation: GenerationContext::default(),
            effect_id: 0,
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
            generation: GenerationContext::default(),
            effect_id: 0,
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
    fn restart_healing_interrupts_a_durable_pending_effect() {
        let mut task = Thread::new(
            "restart-journal".into(),
            "pod".into(),
            base_config_for_fork(),
            ThreadBindings::default(),
            Scope::allow_all(),
            ToolSurface::default(),
        );
        task.submit_user_message("hello".into(), Vec::new(), Vec::new());
        let mut weave = singleton_weave_for(&task);
        let mut next_op_id = 1;
        let mut events = Vec::new();
        let dispatched = drive_until_blocked(&mut task, &mut weave, &mut next_op_id, &mut events);
        assert!(matches!(
            dispatched.as_slice(),
            [IoRequest::ModelCall { .. }]
        ));
        assert!(weave.effect_journal.has_pending());

        // Restart healing: the thread heals to Idle, and the runtime
        // (persist load / scheduler) interrupts the ticking weave's
        // pending records with the same reason.
        task.heal_to_idle("task was in-flight at last shutdown", &mut events);
        weave.interrupt_pending("task was in-flight at last shutdown");
        assert!(matches!(task.internal, ThreadInternalState::Idle));
        assert!(!weave.effect_journal.has_pending());
        assert!(matches!(
            &weave.effect_journal.records()[0].outcome,
            crate::runtime::driver::DriverEffectOutcome::Interrupted { reason }
                if reason == "task was in-flight at last shutdown"
        ));
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
            generation: GenerationContext::default(),
            effect_id: 0,
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
    fn mcp_text_only_result_lowers_to_text() {
        use crate::tools::mcp::McpContentBlock;
        let blocks = [McpContentBlock::Text { text: "ok".into() }];
        match mcp_content_to_tool_result(&blocks) {
            ToolResultContent::Text(s) => assert_eq!(s, "ok"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn mcp_mixed_text_and_image_lowers_to_blocks() {
        use crate::tools::mcp::McpContentBlock;
        use base64::Engine;
        use base64::engine::general_purpose::STANDARD;
        let png_bytes = vec![0x89, b'P', b'N', b'G'];
        let blocks = [
            McpContentBlock::Text {
                text: "screenshot:".into(),
            },
            McpContentBlock::Image {
                data: STANDARD.encode(&png_bytes),
                mime_type: "image/png".into(),
            },
        ];
        match mcp_content_to_tool_result(&blocks) {
            ToolResultContent::Blocks(bs) => {
                assert_eq!(bs.len(), 2);
                assert!(matches!(bs[0], ContentBlock::Text { ref text } if text == "screenshot:"));
                let ContentBlock::Image {
                    source:
                        whisper_agent_protocol::ImageSource::Bytes {
                            ref data,
                            media_type,
                        },
                    ..
                } = bs[1]
                else {
                    panic!("expected Image block, got {:?}", bs[1]);
                };
                assert_eq!(data, &png_bytes);
                assert_eq!(media_type, whisper_agent_protocol::ImageMime::Png);
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn mcp_unsupported_mime_falls_back_to_text_note() {
        use crate::tools::mcp::McpContentBlock;
        let blocks = [McpContentBlock::Image {
            data: "AAAA".into(),
            mime_type: "image/tiff".into(),
        }];
        match mcp_content_to_tool_result(&blocks) {
            ToolResultContent::Blocks(bs) => {
                let ContentBlock::Text { ref text } = bs[0] else {
                    panic!("expected text fallback, got {:?}", bs[0]);
                };
                assert!(text.contains("image/tiff"), "got: {text}");
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    fn empty_thread() -> Thread {
        Thread::new(
            "t".into(),
            "pod".into(),
            base_config_for_fork(),
            ThreadBindings::default(),
            Scope::allow_all(),
            ToolSurface::default(),
        )
    }

    #[test]
    fn submit_user_message_resets_turn_routing_token() {
        // The codex contract is: each "turn" (one user input + the
        // assistant's tool-loop reply) gets a fresh sticky-routing
        // slot. Replaying a prior turn's token into the next turn
        // violates client/server expectations and can cause routing
        // bugs (see codex-rs/core/src/client.rs:227). The boundary
        // signal is `submit_user_message` — that's where we mint a
        // fresh `OnceLock` regardless of whether the previous slot
        // was populated.
        let mut task = empty_thread();
        // Simulate the prior turn having captured a token.
        task.turn_routing_token
            .set("prior-turn-token".to_string())
            .expect("slot starts empty");
        assert_eq!(
            task.turn_routing_token.get().map(String::as_str),
            Some("prior-turn-token")
        );
        let prior_arc = std::sync::Arc::clone(&task.turn_routing_token);

        task.submit_user_message("hello".into(), Vec::new(), Vec::new());

        // New slot, not the same Arc, and empty.
        assert!(
            !std::sync::Arc::ptr_eq(&prior_arc, &task.turn_routing_token),
            "submit_user_message should swap in a fresh Arc, not mutate the existing one"
        );
        assert!(
            task.turn_routing_token.get().is_none(),
            "new turn must start with an empty routing slot"
        );
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
        task.submit_user_message("follow-up".into(), Vec::new(), Vec::new());
        assert!(matches!(task.internal, ThreadInternalState::NeedsModelCall));
        let msgs = task.conversation.messages();
        // [assistant[tool_use], tool_result, user]
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::Assistant);
        assert_eq!(msgs[1].role, Role::ToolResult);
        assert_eq!(msgs[2].role, Role::User);
    }
}
