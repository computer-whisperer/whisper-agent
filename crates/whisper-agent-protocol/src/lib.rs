//! Wire protocol between the whisper-agent server and its clients (webui, CLI).
//!
//! Both directions are CBOR-encoded enums — see the helper functions at the bottom for
//! the (de)serialization entry points.
//!
//! This crate also owns the canonical conversation types (`Role`, `Message`,
//! `ContentBlock`, `Conversation`). They're modeled after Anthropic's content-block shape
//! so serde serializes directly into Anthropic's request body, and they're shared between
//! the server (which builds them) and the client (which renders them from task snapshots).

pub mod conversation;
pub mod sandbox;

pub use conversation::{ContentBlock, Conversation, Message, Role, ToolResultContent};
pub use sandbox::SandboxSpec;

use serde::{Deserialize, Serialize};

// ---------- Task-level types ----------

/// Public-facing state label for a task. The scheduler's internal state has finer
/// distinctions (e.g. `AwaitingModel` vs `AwaitingTools`) but clients see the collapsed
/// form so the protocol is stable against scheduler internals.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStateLabel {
    /// Ready for new user input.
    Idle,
    /// Model call or tool call in flight.
    Working,
    /// Paused waiting for a human approval decision.
    AwaitingApproval,
    /// Turn finished with `end_turn`; open for follow-ups or archive.
    Completed,
    /// Terminally failed. `detail` in the last event gives the reason.
    Failed,
    /// User cancelled.
    Cancelled,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_input_tokens: u32,
    pub cache_creation_input_tokens: u32,
}

impl Usage {
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
    }
}

/// The per-task configuration the server holds. Clients can override pieces of it at
/// task-creation time via [`TaskConfigOverride`].
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskConfig {
    /// Name of the backend in the server's catalog. Empty string means "use the
    /// server's default backend." Kept as `default`-able so tasks persisted before
    /// multi-backend support still deserialize.
    #[serde(default)]
    pub backend: String,
    pub model: String,
    pub system_prompt: String,
    pub mcp_host_url: String,
    pub max_tokens: u32,
    pub max_turns: u32,
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
    #[serde(default)]
    pub sandbox: SandboxSpec,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TaskConfigOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_host_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<ApprovalPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SandboxSpec>,
}

/// Pattern-1 approval policy. See `docs/design_permissions.md`.
///
/// Default is `AutoApproveAll` because the sandbox layer (landlock /
/// containers) is the primary safety boundary. Opt into `PromptDestructive`
/// when running without a sandbox or when human-in-loop is desired.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalPolicy {
    /// Auto-approve every tool call. Default.
    #[default]
    AutoApproveAll,
    /// Auto-approve tools the MCP server marked `readOnlyHint: true`; prompt the user
    /// for everything else (destructive, open-world, or unannotated).
    PromptDestructive,
}

/// User's decision on a pending approval.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalChoice {
    Approve,
    Reject,
}

/// Lightweight per-task summary. Broadcast to every connected client and used by
/// `ListTasks` responses.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskSummary {
    pub task_id: String,
    pub title: Option<String>,
    pub state: TaskStateLabel,
    /// ISO-8601 timestamp. Kept as a plain string on the wire so the protocol crate
    /// doesn't pull in chrono.
    pub created_at: String,
    pub last_active: String,
}

/// Entry in a `BackendsList` response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendSummary {
    /// User-chosen alias — matches `TaskConfig.backend`.
    pub name: String,
    /// Which protocol this backend speaks (`"anthropic"`, `"openai_chat"`, …).
    /// Clients can use this for labels or icons; the server doesn't rely on it.
    pub kind: String,
    /// Model id the server would fall back to if a task doesn't specify one.
    /// None when the backend has no configured default — the UI should fetch the
    /// model list and pick one (typical for single-model local endpoints).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
}

/// Entry in a `ModelsList` response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelSummary {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
}

/// Full per-task snapshot. Sent in response to a `SubscribeToTask` so the client can
/// render the entire conversation from a cold start.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskSnapshot {
    pub task_id: String,
    pub title: Option<String>,
    pub config: TaskConfig,
    pub state: TaskStateLabel,
    pub conversation: Conversation,
    pub total_usage: Usage,
    pub created_at: String,
    pub last_active: String,
    /// Reason the task entered the `Failed` state, if any. Populated from the task's
    /// internal `Failed { at_phase, message }` so clients subscribing after the
    /// failure can still render why. `None` for non-Failed tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
    /// Tool names this task has marked "always allow" — they bypass the approval
    /// prompt for the rest of the task's lifetime. Sorted for stable display.
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
}

// ---------- Wire enums ----------

/// Messages the client sends to the server.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientToServer {
    // --- Task lifecycle ---
    /// Create a new task. `initial_message` becomes the first user message; the server
    /// starts driving the loop immediately.
    CreateTask {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        initial_message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        config_override: Option<TaskConfigOverride>,
    },
    /// Append a follow-up user message to an existing task.
    SendUserMessage { task_id: String, text: String },
    /// Respond to a pending tool-call approval. If `remember` is true and the
    /// decision is `Approve`, the tool's name is added to the task's allowlist —
    /// future calls to that tool skip the prompt for the rest of the task's
    /// lifetime. `remember` with `Reject` is currently ignored (no
    /// "always reject" semantics).
    ApprovalDecision {
        task_id: String,
        approval_id: String,
        decision: ApprovalChoice,
        #[serde(default)]
        remember: bool,
    },
    /// Remove a tool name from the task's allowlist. Future calls to that tool
    /// will prompt again under the task's policy.
    RemoveToolAllowlistEntry { task_id: String, tool_name: String },
    /// Cancel a task. Stubbed for now — flips state to `Cancelled` without rolling back
    /// any in-flight tool work. Proper rollback semantics are a v0.3 problem.
    CancelTask { task_id: String },
    /// Archive (hide) a task. It remains on disk but drops off the broadcast list.
    ArchiveTask { task_id: String },

    // --- Observation ---
    /// Start receiving per-turn events for this task. Server responds with a
    /// `TaskSnapshot` and then streams subsequent events.
    SubscribeToTask { task_id: String },
    UnsubscribeFromTask { task_id: String },
    /// Request the current list of non-archived tasks.
    ListTasks {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },

    // --- Model catalog ---
    /// Request the list of configured backends. Cheap / synchronous on the server.
    ListBackends {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// Request the list of models advertised by a specific backend. Server hits the
    /// backend's `list_models` endpoint — may take a round-trip.
    ListModels {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        backend: String,
    },
}

/// Messages the server sends to the client.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerToClient {
    // --- Task-list tier (broadcast to every connected client) ---
    TaskCreated {
        task_id: String,
        summary: TaskSummary,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    TaskStateChanged {
        task_id: String,
        state: TaskStateLabel,
    },
    TaskTitleUpdated {
        task_id: String,
        title: String,
    },
    TaskArchived {
        task_id: String,
    },

    // --- Request / response ---
    TaskList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        tasks: Vec<TaskSummary>,
    },
    TaskSnapshot {
        task_id: String,
        snapshot: TaskSnapshot,
    },

    // --- Per-task turn tier (only to subscribers of `task_id`) ---
    TaskAssistantBegin {
        task_id: String,
        turn: u32,
    },
    /// A complete text block emitted by the assistant. Always emitted at turn-end so
    /// clients reconnecting mid-stream see consistent state once the turn settles.
    TaskAssistantText {
        task_id: String,
        text: String,
    },
    /// Streaming text partial (reserved — not emitted until SSE streaming lands).
    TaskAssistantTextDelta {
        task_id: String,
        delta: String,
    },
    /// A tool call needs user approval before it can be dispatched.
    TaskPendingApproval {
        task_id: String,
        approval_id: String,
        tool_use_id: String,
        name: String,
        args_preview: String,
        /// Server-declared `destructiveHint` (default false if unannotated).
        destructive: bool,
        /// Server-declared `readOnlyHint` (default false if unannotated).
        read_only: bool,
    },
    /// A previously-pending approval has been resolved (by any connected client).
    TaskApprovalResolved {
        task_id: String,
        approval_id: String,
        decision: ApprovalChoice,
        /// Connection id that submitted the decision, if known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decided_by_conn: Option<u64>,
    },
    /// Task's tool allowlist changed. Sent whenever a tool is added (via an
    /// `ApprovalDecision` with `remember: true`) or removed (via
    /// `RemoveToolAllowlistEntry`). Carries the full new list so clients don't
    /// have to track add/remove deltas.
    TaskAllowlistUpdated {
        task_id: String,
        tool_allowlist: Vec<String>,
    },
    TaskToolCallBegin {
        task_id: String,
        tool_use_id: String,
        name: String,
        args_preview: String,
    },
    TaskToolCallEnd {
        task_id: String,
        tool_use_id: String,
        result_preview: String,
        is_error: bool,
    },
    TaskAssistantEnd {
        task_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
        usage: Usage,
    },
    TaskLoopComplete {
        task_id: String,
    },

    // --- Model catalog responses ---
    BackendsList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        default_backend: String,
        backends: Vec<BackendSummary>,
    },
    ModelsList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        backend: String,
        models: Vec<ModelSummary>,
    },

    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<String>,
        message: String,
    },
}

// ---------- (de)serialization ----------

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("encode: {0}")]
    Encode(String),
    #[error("decode: {0}")]
    Decode(String),
}

pub fn encode_to_server(msg: &ClientToServer) -> Result<Vec<u8>, CodecError> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(msg, &mut buf).map_err(|e| CodecError::Encode(e.to_string()))?;
    Ok(buf)
}

pub fn decode_from_client(bytes: &[u8]) -> Result<ClientToServer, CodecError> {
    ciborium::de::from_reader(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}

pub fn encode_to_client(msg: &ServerToClient) -> Result<Vec<u8>, CodecError> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(msg, &mut buf).map_err(|e| CodecError::Encode(e.to_string()))?;
    Ok(buf)
}

pub fn decode_from_server(bytes: &[u8]) -> Result<ServerToClient, CodecError> {
    ciborium::de::from_reader(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}
