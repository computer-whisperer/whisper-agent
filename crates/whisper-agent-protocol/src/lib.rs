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
pub mod pod;
pub mod sandbox;

pub use conversation::{ContentBlock, Conversation, Message, Role, ToolResultContent};
pub use pod::{NamedSandboxSpec, PodAllow, PodConfig, PodLimits, PodSnapshot, PodSummary, ThreadDefaults};
pub use sandbox::SandboxSpec;

use serde::{Deserialize, Serialize};

// ---------- Task-level types ----------

/// Public-facing state label for a task. The scheduler's internal state has finer
/// distinctions (e.g. `AwaitingModel` vs `AwaitingTools`) but clients see the collapsed
/// form so the protocol is stable against scheduler internals.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThreadStateLabel {
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

/// The per-thread configuration the server holds. Carries only "what the loop
/// looks like" — model, prompts, limits, policy. Resource-side concerns
/// (which backend, which sandbox, which MCP hosts) live on
/// [`ThreadBindings`]; the split was introduced in Phase 3d.i so resource
/// lifecycles can move independently of thread config.
///
/// Clients can override pieces of this at thread-creation time via
/// [`ThreadConfigOverride`]; bindings are overridden separately via
/// [`ThreadBindingsRequest`].
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThreadConfig {
    pub model: String,
    pub system_prompt: String,
    pub max_tokens: u32,
    pub max_turns: u32,
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ThreadConfigOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<ApprovalPolicy>,
}

/// Concrete resource bindings for a thread. Each field names an entry in
/// the scheduler's resource registry — `backend` by catalog name, `sandbox`
/// by `SandboxId` (content-hash of the spec), `mcp_hosts` by `McpHostId`
/// (the ordered list the thread routes tool calls through, primary first).
///
/// `tool_filter`, when present, narrows the visible tool catalog to a
/// whitelist of names — used by subagent / fork flows that want to expose
/// less than the full surface their bound MCP hosts advertise. (Reserved;
/// not yet honored.)
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ThreadBindings {
    /// Backend catalog name (e.g. `"anthropic"`). Empty string means "use the
    /// server's default backend." Kept as a name rather than a `BackendId`
    /// so display surfaces (UI labels, logs) don't have to strip the
    /// `backend-` prefix.
    #[serde(default)]
    pub backend: String,
    /// Bound `SandboxId` (full `sb-…` form) when the thread runs inside a
    /// sandbox; `None` for bare-metal threads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<String>,
    /// Bound `McpHostId`s, in routing precedence order. The primary
    /// (per-thread filesystem) host is at index 0; shared hosts follow in
    /// the order they were declared on the pod.
    #[serde(default)]
    pub mcp_hosts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_filter: Option<Vec<String>>,
}

/// Client-side overrides for the bindings the pod's `thread_defaults` would
/// otherwise produce. Each `Some` field replaces the corresponding default;
/// each `None` inherits. Values are validated against the pod's `[allow]`
/// table on the server — backends and shared MCP host names must already
/// appear there. Inline `sandbox` specs are accepted as-is until pod
/// editing UI ships; see `resolve_bindings_choice` for the rationale.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ThreadBindingsRequest {
    /// Backend catalog name. Empty → server default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Inline sandbox spec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SandboxSpec>,
    /// Catalog names of shared MCP hosts the thread should bind to. `None`
    /// inherits the pod default; `Some(vec)` replaces exactly (empty vec
    /// means "no shared hosts beyond the primary").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_hosts: Option<Vec<String>>,
}

/// Mid-thread rebind patch. Applied to the live `ThreadBindings` of a
/// running thread; each `Some` field replaces the matching binding, `None`
/// leaves it alone. Validated against the pod's `[allow]` table — same
/// rules as initial binding (modulo the inline-sandbox carve-out).
///
/// In-flight I/O against the *previous* bindings completes against its
/// captured resources; only future ops see the new state. When the
/// sandbox or MCP host set changes, the scheduler appends a synthetic
/// conversation note so the model knows the execution environment shifted.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ThreadBindingsPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// `Some(SandboxRebind::Inline(spec))` re-binds; `Some(Clear)` drops
    /// the binding entirely; `None` leaves it alone. Two layers of Option
    /// because "set to nothing" and "leave alone" are distinct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SandboxRebind>,
    /// Replace the shared MCP host list. `Some(vec)` replaces exactly
    /// (empty vec means "no shared hosts"); `None` leaves the current set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_hosts: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SandboxRebind {
    /// Replace the bound sandbox with this inline spec.
    Inline { spec: SandboxSpec },
    /// Drop the sandbox binding entirely (thread runs bare-metal).
    Clear,
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
/// `ListThreads` responses.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThreadSummary {
    pub thread_id: String,
    /// Pod the thread belongs to. Empty for legacy snapshots from before the
    /// pod-aware scheduler landed (Phase 2d.iii).
    #[serde(default)]
    pub pod_id: String,
    pub title: Option<String>,
    pub state: ThreadStateLabel,
    /// ISO-8601 timestamp. Kept as a plain string on the wire so the protocol crate
    /// doesn't pull in chrono.
    pub created_at: String,
    pub last_active: String,
}

/// Entry in a `BackendsList` response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendSummary {
    /// User-chosen alias — matches `ThreadConfig.backend`.
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

// ---------- Resource registry ----------

/// Lifecycle state for a resource, collapsed for the wire. Internal scheduler
/// distinctions (e.g. provisioning op_id) are dropped — clients only see the
/// state they can act on.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceStateLabel {
    Provisioning,
    Ready,
    Errored,
    TornDown,
}

/// Discriminator for resource events. The resource id strings are namespaced
/// by prefix (`sb-`, `mcp-primary-`, `mcp-shared-`, `backend-`) but carrying
/// the kind explicitly lets clients route without prefix-parsing.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    Sandbox,
    McpHost,
    Backend,
}

/// One resource entry, in a shape that's serializable and stable for clients.
/// Mirrors `crate::resources::*Entry` types in the server but keeps wire-only
/// concerns (no `Arc<McpSession>`, timestamps as RFC-3339 strings, tool list
/// reduced to names).
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResourceSnapshot {
    Sandbox {
        id: String,
        spec: SandboxSpec,
        state: ResourceStateLabel,
        /// User ids (task ids today, thread ids after Phase 2). Sorted.
        users: Vec<String>,
        pinned: bool,
        created_at: String,
        last_used: String,
    },
    McpHost {
        id: String,
        url: String,
        label: String,
        per_task: bool,
        state: ResourceStateLabel,
        /// Names of tools the host advertises. Empty until the first
        /// `tools/list` lands. Full descriptors are not on the wire — clients
        /// that need them can subscribe to a task and read its snapshot.
        tools: Vec<String>,
        users: Vec<String>,
        pinned: bool,
        created_at: String,
        last_used: String,
    },
    Backend {
        id: String,
        name: String,
        backend_kind: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default_model: Option<String>,
        state: ResourceStateLabel,
        users: Vec<String>,
        pinned: bool,
        created_at: String,
        last_used: String,
    },
}

impl ResourceSnapshot {
    pub fn id(&self) -> &str {
        match self {
            Self::Sandbox { id, .. } | Self::McpHost { id, .. } | Self::Backend { id, .. } => id,
        }
    }
    pub fn kind(&self) -> ResourceKind {
        match self {
            Self::Sandbox { .. } => ResourceKind::Sandbox,
            Self::McpHost { .. } => ResourceKind::McpHost,
            Self::Backend { .. } => ResourceKind::Backend,
        }
    }
}

/// Full per-task snapshot. Sent in response to a `SubscribeToThread` so the client can
/// render the entire conversation from a cold start.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThreadSnapshot {
    pub thread_id: String,
    /// Pod this thread belongs to. Empty for legacy snapshots.
    #[serde(default)]
    pub pod_id: String,
    pub title: Option<String>,
    pub config: ThreadConfig,
    /// Resource bindings — backend, sandbox, mcp hosts. Defaults to empty
    /// for snapshots from before Phase 3d.i (where binding info still lived
    /// inline on `ThreadConfig`).
    #[serde(default)]
    pub bindings: ThreadBindings,
    pub state: ThreadStateLabel,
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
// `CreateThread` carries a `ThreadConfigOverride` (~300 bytes of Options); other
// variants are tens of bytes. Boxing the override would change every
// construction site for a bytes-saved-per-message that's a rounding error
// against typical `initial_message` payloads. Revisit if profiling shows it.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientToServer {
    // --- Task lifecycle ---
    /// Create a new task. `initial_message` becomes the first user message; the server
    /// starts driving the loop immediately.
    CreateThread {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        /// Which pod the new thread should land in. `None` routes the thread
        /// to the server's default pod — currently the only path the webui
        /// uses, since multi-pod creation arrives in Phase 2e/4.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pod_id: Option<String>,
        initial_message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        config_override: Option<ThreadConfigOverride>,
        /// Override resource bindings (backend / sandbox / shared MCP hosts).
        /// `None` inherits the pod's `thread_defaults` for every binding;
        /// `Some` replaces just the fields it carries.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bindings_request: Option<ThreadBindingsRequest>,
    },
    /// Append a follow-up user message to an existing task.
    SendUserMessage {
        thread_id: String,
        text: String,
    },
    /// Respond to a pending tool-call approval. If `remember` is true and the
    /// decision is `Approve`, the tool's name is added to the task's allowlist —
    /// future calls to that tool skip the prompt for the rest of the task's
    /// lifetime. `remember` with `Reject` is currently ignored (no
    /// "always reject" semantics).
    ApprovalDecision {
        thread_id: String,
        approval_id: String,
        decision: ApprovalChoice,
        #[serde(default)]
        remember: bool,
    },
    /// Remove a tool name from the task's allowlist. Future calls to that tool
    /// will prompt again under the task's policy.
    RemoveToolAllowlistEntry {
        thread_id: String,
        tool_name: String,
    },
    /// Replace pieces of a running thread's resource bindings (backend,
    /// sandbox, shared MCP hosts). The patch is validated against the
    /// pod's `[allow]` table; on success the scheduler swaps the bindings
    /// in place, adjusts resource refcounts, re-provisions the primary
    /// MCP if the sandbox changed, and appends a synthetic conversation
    /// note so the model is aware the execution environment may have
    /// shifted. In-flight ops complete against their captured resources.
    RebindThread {
        thread_id: String,
        patch: ThreadBindingsPatch,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// Cancel a task. Stubbed for now — flips state to `Cancelled` without rolling back
    /// any in-flight tool work. Proper rollback semantics are a v0.3 problem.
    CancelThread {
        thread_id: String,
    },
    /// Archive (hide) a task. It remains on disk but drops off the broadcast list.
    ArchiveThread {
        thread_id: String,
    },

    // --- Observation ---
    /// Start receiving per-turn events for this task. Server responds with a
    /// `ThreadSnapshot` and then streams subsequent events.
    SubscribeToThread {
        thread_id: String,
    },
    UnsubscribeFromThread {
        thread_id: String,
    },
    /// Request the current list of non-archived tasks.
    ListThreads {
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

    // --- Resource registry (read-only in Phase 1b) ---
    /// Request a snapshot of every entry in the scheduler's resource registry
    /// (sandboxes, MCP hosts, backends). Server responds with `ResourceList`.
    /// Subsequent registry mutations are pushed unsolicited via
    /// `ResourceCreated` / `ResourceUpdated` / `ResourceDestroyed`.
    ListResources {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },

    // --- Pod registry (Phase 2d.i — wire surface only; the scheduler
    //     becomes pod-aware in 2d.iii). ---
    /// Walk `<pods_root>/` and return every non-archived pod's summary.
    ListPods {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// Read one pod's config + thread summaries.
    GetPod {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
    },
    /// Create a new pod directory + write its `pod.toml`. Fails if the pod
    /// id already exists (use `UpdatePodConfig` to mutate).
    CreatePod {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        config: PodConfig,
    },
    /// Replace the pod's `pod.toml` with the given text. Server parses and
    /// validates before writing; on validation failure no file changes.
    UpdatePodConfig {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        toml_text: String,
    },
    /// Move the pod's directory under `<pods_root>/.archived/`. Threads
    /// inside become unreachable (they're not loaded from .archived/).
    ArchivePod {
        pod_id: String,
    },
}

/// Messages the server sends to the client.
// `ThreadSnapshot` is much larger (~470 bytes — full conversation, config, usage)
// than the streaming variants (~50 bytes each). Snapshots are sent rarely (one
// per subscribe), so the per-clone cost during scheduler broadcasts is the
// streaming-variant size, not the snapshot's. Boxing would change every
// construction site for marginal gain. Revisit if snapshot broadcasts ever land
// on the hot path.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerToClient {
    // --- Task-list tier (broadcast to every connected client) ---
    ThreadCreated {
        thread_id: String,
        summary: ThreadSummary,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    ThreadStateChanged {
        thread_id: String,
        state: ThreadStateLabel,
    },
    ThreadTitleUpdated {
        thread_id: String,
        title: String,
    },
    ThreadArchived {
        thread_id: String,
    },
    /// A `RebindThread` was applied (or denied — see correlation_id +
    /// matching Error). Carries the new bindings so subscribed clients
    /// can refresh any rendered binding info.
    ThreadBindingsChanged {
        thread_id: String,
        bindings: ThreadBindings,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },

    // --- Request / response ---
    ThreadList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        tasks: Vec<ThreadSummary>,
    },
    ThreadSnapshot {
        thread_id: String,
        snapshot: ThreadSnapshot,
    },

    // --- Per-task turn tier (only to subscribers of `thread_id`) ---
    ThreadAssistantBegin {
        thread_id: String,
        turn: u32,
    },
    /// A complete text block emitted by the assistant. Always emitted at turn-end so
    /// clients reconnecting mid-stream see consistent state once the turn settles.
    ThreadAssistantText {
        thread_id: String,
        text: String,
    },
    /// Chain-of-thought block emitted by the assistant — Anthropic
    /// extended-thinking, OpenAI-compat `reasoning_content`, or inline
    /// `<think>...</think>`. Order-preserved with `ThreadAssistantText` so the
    /// client can render reasoning, replies, and tool-calls in the order the
    /// model emitted them.
    ThreadAssistantReasoning {
        thread_id: String,
        text: String,
    },
    /// Streaming text partial (reserved — not emitted until SSE streaming lands).
    ThreadAssistantTextDelta {
        thread_id: String,
        delta: String,
    },
    /// A tool call needs user approval before it can be dispatched.
    ThreadPendingApproval {
        thread_id: String,
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
    ThreadApprovalResolved {
        thread_id: String,
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
    ThreadAllowlistUpdated {
        thread_id: String,
        tool_allowlist: Vec<String>,
    },
    ThreadToolCallBegin {
        thread_id: String,
        tool_use_id: String,
        name: String,
        args_preview: String,
    },
    ThreadToolCallEnd {
        thread_id: String,
        tool_use_id: String,
        result_preview: String,
        is_error: bool,
    },
    ThreadAssistantEnd {
        thread_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
        usage: Usage,
    },
    ThreadLoopComplete {
        thread_id: String,
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

    // --- Resource registry tier (broadcast to every connected client) ---
    /// Snapshot response to `ListResources`.
    ResourceList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        resources: Vec<ResourceSnapshot>,
    },
    /// A new entry appeared in the registry.
    ResourceCreated {
        resource: ResourceSnapshot,
    },
    /// An existing entry's fields changed (state, users, tool list, etc.).
    /// Carries a full snapshot rather than a delta — entries are small, and
    /// full snapshots let clients ignore field-by-field churn.
    ResourceUpdated {
        resource: ResourceSnapshot,
    },
    /// An entry was removed from the registry. Reserved — Phase 1b doesn't
    /// emit it (TornDown is a state on the Updated event); a future GC pass
    /// that reaps TornDown entries will start emitting it.
    ResourceDestroyed {
        id: String,
        kind: ResourceKind,
    },

    // --- Pod registry tier (broadcast to every connected client) ---
    PodList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pods: Vec<PodSummary>,
        /// Pod the server routes `CreateThread { pod_id: None }` to. Lets
        /// clients clone its config when bootstrapping fresh pods, so a
        /// "+ New pod" flow inherits a known-working sandbox + mcp host
        /// setup instead of starting from a minimal stub.
        #[serde(default)]
        default_pod_id: String,
    },
    PodSnapshot {
        snapshot: PodSnapshot,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    PodCreated {
        pod: PodSummary,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    PodConfigUpdated {
        pod_id: String,
        toml_text: String,
        parsed: PodConfig,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    PodArchived {
        pod_id: String,
    },

    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thread_id: Option<String>,
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
