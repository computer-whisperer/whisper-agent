//! Wire protocol between the whisper-agent server and its clients (webui, CLI).
//!
//! Both directions are CBOR-encoded enums — see the helper functions at the bottom for
//! the (de)serialization entry points.
//!
//! This crate also owns the canonical conversation types (`Role`, `Message`,
//! `ContentBlock`, `Conversation`). They're modeled after Anthropic's content-block shape
//! so serde serializes directly into Anthropic's request body, and they're shared between
//! the server (which builds them) and the client (which renders them from task snapshots).

pub mod behavior;
pub mod conversation;
pub mod permission;
pub mod pod;
pub mod sandbox;

pub use permission::{AllowMap, Disposition};

pub use behavior::{
    BehaviorBindingsOverride, BehaviorConfig, BehaviorOrigin, BehaviorOutcome, BehaviorSnapshot,
    BehaviorState, BehaviorSummary, BehaviorThreadOverride, CatchUp, Overlap, RetentionPolicy,
    TriggerSpec,
};
pub use conversation::{
    ContentBlock, Conversation, Message, ProviderReplay, Role, ToolResultContent,
};
pub use pod::{
    CompactionConfig, FsEntry, NamedHostEnv, PodAllow, PodConfig, PodLimits, PodSnapshot, PodState,
    PodSummary, ThreadDefaults,
};
// `SystemPromptChoice` is defined below; re-export here so other
// crates can refer to it as `whisper_agent_protocol::SystemPromptChoice`.
pub use sandbox::HostEnvSpec;

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

/// Diagnostic log of per-LLM-call metadata. Grows by one [`TurnEntry`]
/// on every assistant turn (i.e. every `integrate_model_response` in
/// the runtime). Distinct from [`Usage`] aggregated in
/// `ThreadSnapshot::total_usage`: `total_usage` is the running sum,
/// `turn_log` preserves the per-call shape so cache hit/miss rates can
/// be audited after the fact.
///
/// Entry count matches the number of `Role::Assistant` messages in
/// the conversation in temporal order — the runtime pushes exactly
/// one entry per model response. Threads that pre-date this log load
/// with `entries: []` via `#[serde(default)]`.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TurnLog {
    pub entries: Vec<TurnEntry>,
}

/// One LLM call's diagnostic record. Only `usage` today; additional
/// fields (model actually used, latency, cache-breakpoint layout at
/// call time, retries) can grow here without touching callers that
/// just read `usage`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TurnEntry {
    pub usage: Usage,
}

/// The per-thread configuration the server holds. Carries only "what the loop
/// looks like" — model, limits, policy. Resource-side concerns
/// (which backend, which sandbox, which MCP hosts) live on
/// [`ThreadBindings`]; the split was introduced in Phase 3d.i so resource
/// lifecycles can move independently of thread config.
///
/// The system prompt and tool manifest used to live here too, but they
/// now ride at the head of [`Conversation`] as `Role::System` and
/// `Role::Tools` messages. Both are cache-fingerprint-critical — any
/// change busts the prompt cache — so anchoring them to the immutable
/// conversation log matches the operational reality and keeps the chat
/// record identical to what the model actually saw.
///
/// Clients can override pieces of this at thread-creation time via
/// [`ThreadConfigOverride`]; bindings are overridden separately via
/// [`ThreadBindingsRequest`].
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThreadConfig {
    pub model: String,
    pub max_tokens: u32,
    pub max_turns: u32,
    /// Policy for compacting an overlong thread into a continuation.
    /// Inherits from the pod's `thread_defaults.compaction`; per-thread
    /// overrides come in via [`ThreadConfigOverride.compaction`].
    #[serde(default)]
    pub compaction: CompactionConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ThreadConfigOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    /// Creation-time choice of system prompt for the new thread.
    /// `None` inherits the pod's cached `system_prompt` (the text of
    /// the file named by `thread_defaults.system_prompt_file`).
    /// `Some(File { name })` reads a sibling file from the pod dir;
    /// `Some(Text { text })` uses the literal string. Resolved once
    /// during thread creation and frozen into `conversation[0]` as
    /// the `Role::System` message — not preserved on `ThreadConfig`,
    /// since the conversation log is the source of truth for what the
    /// model actually saw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<SystemPromptChoice>,
    /// Per-field compaction override. Fields left `None` inherit from
    /// the pod's defaults; any set field replaces it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactionConfigOverride>,
}

/// How a thread-creation caller specifies the system prompt to seed
/// `conversation[0]` with. Pod-relative file lookup or inline text;
/// more variants (e.g. named-template reference) can join later
/// without changing call sites that already match on `File`/`Text`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SystemPromptChoice {
    /// Read the prompt from `<pod_dir>/<name>`. Missing file logs a
    /// warning and falls back to the pod's cached default prompt —
    /// same graceful-degrade story as an unreadable
    /// `thread_defaults.system_prompt_file`. Pod-relative (not
    /// absolute) so the pod's filesystem boundary is respected.
    File { name: String },
    /// Inline prompt text used verbatim.
    Text { text: String },
}

/// Per-thread compaction override. Shape mirrors [`CompactionConfig`] but
/// every field is `Option` — `None` means "inherit from the pod default."
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct CompactionConfigOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_regex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_threshold: Option<Option<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation_template: Option<String>,
}

/// Concrete resource bindings for a thread. Each field names an entry in
/// the scheduler's resource registry — `backend` by catalog name, `sandbox`
/// by `HostEnvId` (content-hash of provider+spec), `mcp_hosts` by
/// `McpHostId` (the ordered list the thread routes tool calls through,
/// primary first).
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
    /// Host envs the thread is bound to, in declared order. Empty vec
    /// means "no host env" — the thread runs under the always-built-in
    /// `bare` provider (in-process, no isolation). Each non-empty entry
    /// resolves to its own runtime `HostEnvId` at registry-touch time;
    /// the scheduler provisions and tears down envs independently.
    ///
    /// Persisted as an array. Legacy thread.json files (pre-multi-env)
    /// stored a single `HostEnvBinding` under `host_env` — the custom
    /// deserializer below accepts that object form and wraps it into a
    /// one-entry vec, so old files load without a separate migration.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_host_env_bindings"
    )]
    pub host_env: Vec<HostEnvBinding>,
    /// Bound `McpHostId`s, in routing precedence order. Per-thread
    /// host-env MCPs come first (one per entry in `host_env`, in the
    /// same order); shared hosts follow in the order they were
    /// declared on the pod.
    #[serde(default)]
    pub mcp_hosts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_filter: Option<Vec<String>>,
}

/// Accept both the legacy singular shape (`null` or a `HostEnvBinding`
/// object) and the new plural shape (a `Vec<HostEnvBinding>`). The
/// legacy form wraps into a one- or zero-entry vec so persisted thread
/// JSONs from before the multi-host-env refactor load cleanly.
fn deserialize_host_env_bindings<'de, D>(d: D) -> Result<Vec<HostEnvBinding>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Shape {
        Many(Vec<HostEnvBinding>),
        One(HostEnvBinding),
    }
    Ok(match Option::<Shape>::deserialize(d)? {
        None => Vec::new(),
        Some(Shape::Many(v)) => v,
        Some(Shape::One(b)) => vec![b],
    })
}

/// What a thread's `host_env` slot can hold.
///
/// The only shape today is [`Named`](Self::Named) — a reference to a
/// `[[allow.host_env]]` entry in the owning pod's config. The pod
/// owns the (provider, spec); editing the entry in `pod.toml`
/// propagates to every thread bound to it on next provision. Pod
/// `[allow]` is a real capability cap: every user-originated binding
/// resolves to a named entry, nothing more.
///
/// [`Inline`](Self::Inline) is reserved for a future subagent /
/// out-of-pod spawn path that needs to pin a spec the pod doesn't
/// declare. It is **not** reachable from user-facing request types —
/// [`ThreadBindingsRequest`] and [`ThreadBindingsPatch`] both name
/// entries by string only — so the wire can't sneak a spec past the
/// pod cap. Kept as a variant (rather than collapsing the enum to
/// plain strings) so the stored-state type is stable when the
/// subagent flow lands.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostEnvBinding {
    Named {
        name: String,
    },
    /// Reserved: ad-hoc/subagent path — not constructable from any
    /// current wire request type.
    Inline {
        provider: String,
        spec: HostEnvSpec,
    },
}

/// Client-side overrides for the bindings the pod's `thread_defaults` would
/// otherwise produce. Each `Some` field replaces the corresponding default;
/// each `None` inherits. Every field is validated against the pod's
/// `[allow]` table on the server; the pod cap is authoritative.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ThreadBindingsRequest {
    /// Backend catalog name. Empty → server default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Names of `[[allow.host_env]]` entries in the target pod. `None`
    /// inherits the pod's `thread_defaults.host_env`; `Some(vec)`
    /// replaces exactly (empty vec means "no host envs — bare thread").
    /// Each name must resolve in the pod's allow list — the server
    /// rejects unknown names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_env: Option<Vec<String>>,
    /// Catalog names of shared MCP hosts the thread should bind to. `None`
    /// inherits the pod default; `Some(vec)` replaces exactly (empty vec
    /// means "no shared hosts beyond the primary").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_hosts: Option<Vec<String>>,
}

/// Mid-thread rebind patch. Applied to the live `ThreadBindings` of a
/// running thread; each `Some` field replaces the matching binding, `None`
/// leaves it alone. Validated against the pod's `[allow]` table — same
/// rules as initial binding.
///
/// In-flight I/O against the *previous* bindings completes against its
/// captured resources; only future ops see the new state. When the
/// host env or MCP host set changes, the scheduler appends a synthetic
/// conversation note so the model knows the execution environment shifted.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct ThreadBindingsPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Replace the bound host-env list. `Some(vec)` replaces exactly —
    /// empty vec drops all bindings (thread falls back to the bare
    /// provider), non-empty rebinds to the listed allow entries.
    /// `None` leaves the current list alone. Each name must resolve in
    /// the pod's allow list at rebind time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_env: Option<Vec<String>>,
    /// Replace the shared MCP host list. `Some(vec)` replaces exactly
    /// (empty vec means "no shared hosts"); `None` leaves the current set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_hosts: Option<Vec<String>>,
}

/// Server-advertised entry from the host-env-provider catalog. Sent in
/// response to `ListHostEnvProviders` so the webui's pod editor can
/// populate its provider dropdown. Every entry represents a
/// daemon-backed provider — there are no built-ins.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct HostEnvProviderInfo {
    /// User-facing catalog name (matches `NamedHostEnv.provider`).
    pub name: String,
    /// Daemon URL the scheduler talks to for provision / teardown.
    /// Included so a WebUI management panel can display it; clients
    /// that only need the name (pod-editor dropdown) can ignore it.
    pub url: String,
    /// How this entry entered the catalog. `Seeded` entries were
    /// imported from `whisper-agent.toml`'s `[[host_env_providers]]`
    /// table on startup; `Manual` entries were added at runtime
    /// (WebUI / RPC). `RuntimeOverlay` entries come from CLI flags
    /// and are not persisted — they live only for this server run.
    pub origin: HostEnvProviderOrigin,
    /// Whether the entry carries a control-plane bearer token.
    /// The token itself is never sent over the wire; clients get a
    /// boolean so the UI can show "authenticated" / "anonymous"
    /// status without exposing the secret.
    pub has_token: bool,
}

/// Provenance enum for `HostEnvProviderInfo.origin`. Mirrors the
/// server's internal `CatalogOrigin` plus a `RuntimeOverlay` variant
/// for CLI-provided entries that never land in the durable catalog.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HostEnvProviderOrigin {
    Seeded,
    Manual,
    RuntimeOverlay,
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
    /// Behavior-origin stamp when this thread was spawned by a behavior
    /// trigger. `None` for interactive threads. Present on the list tier
    /// because clients want to badge / group behavior-spawned threads
    /// without fetching the full snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<BehaviorOrigin>,
    /// When this thread was spawned as the continuation of a compacted
    /// thread, the id of that ancestor. `None` for threads that weren't
    /// created by compaction. Exposed on the list tier so the UI can
    /// badge "continued from …" without fetching the full snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continued_from: Option<String>,
    /// When this thread was spawned by a parent thread's `dispatch_thread`
    /// tool call, the id of the parent. `None` for top-level threads.
    /// Exposed on the list tier so the UI can nest dispatched children
    /// under their parent in the sidebar without fetching the full snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatched_by: Option<String>,
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
/// by prefix (`he-`, `mcp-primary-`, `mcp-shared-`, `backend-`) but carrying
/// the kind explicitly lets clients route without prefix-parsing.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    HostEnv,
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
    HostEnv {
        id: String,
        /// Provider name from the catalog this host env is bound to.
        provider: String,
        spec: HostEnvSpec,
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
            Self::HostEnv { id, .. } | Self::McpHost { id, .. } | Self::Backend { id, .. } => id,
        }
    }
    pub fn kind(&self) -> ResourceKind {
        match self {
            Self::HostEnv { .. } => ResourceKind::HostEnv,
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
    /// Per-turn usage log, parallel to the `Role::Assistant` messages
    /// in `conversation`. `#[serde(default)]` so threads persisted
    /// before this field was added load with an empty log.
    #[serde(default)]
    pub turn_log: TurnLog,
    /// User's in-progress compose-box contents for this thread.
    /// Opaque to the server; mutated via `SetThreadDraft` and
    /// echoed via `ThreadDraftUpdated`. Persists on disk so
    /// partial prompts survive reopen. Empty = no draft.
    #[serde(default)]
    pub draft: String,
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
    /// Behavior-origin stamp for threads spawned by a behavior trigger.
    /// Carries the full payload (`ThreadSummary` carries the same field
    /// but payload-size concerns may trim it there later).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<BehaviorOrigin>,
    /// Ancestor thread id when this thread is a compaction continuation.
    /// `None` for threads that weren't created by compaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continued_from: Option<String>,
    /// Parent thread id when this thread was spawned by a
    /// `dispatch_thread` tool call. `None` for top-level threads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatched_by: Option<String>,
}

// ---------- Wire enums ----------

/// Messages the client sends to the server.
// `CreateThread` carries a `ThreadConfigOverride` (~300 bytes of Options); other
// variants are tens of bytes. Boxing the override would change every
// construction site for a bytes-saved-per-message that's a rounding error
// against typical `initial_message` payloads. Revisit if profiling shows it.
/// Stable display-facing tag describing what kind of in-flight
/// operation a Function represents. The server owns a richer
/// `Function` enum with per-variant execution specs; this is a
/// flattening of just the kind, for UI surfaces that track active
/// Functions without needing to parse variant payloads.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FunctionKind {
    CreateThread,
    CompactThread,
    RebindThread,
    CancelThread,
    RunBehavior,
    BuiltinToolCall,
    McpToolUse,
}

/// Coarse outcome tag sent with `FunctionEnded`. The server's full
/// `FunctionOutcome` carries per-variant terminal payloads; the webui
/// only needs the kind of ending for chip semantics ("done" vs
/// "errored" vs "cancelled") — variant-specific terminal data rides
/// its own dedicated events (`ThreadStateChanged`, `ThreadCompacted`,
/// tool-result messages).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FunctionOutcomeTag {
    Success,
    Error,
    Cancelled,
}

/// Display snapshot of one in-flight Function. Sent as the payload of
/// `FunctionStarted` and as list entries in `FunctionList`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionSummary {
    /// Server-assigned monotonic id. Not stable across restarts; only
    /// used to correlate `FunctionStarted` / `FunctionEnded` pairs
    /// within a session.
    pub function_id: u64,
    pub kind: FunctionKind,
    /// Thread the Function targets, when applicable (Cancel, Compact,
    /// Rebind, or `dispatch_thread`-shaped Creates).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// Pod the Function is scoped to, when knowable from the spec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pod_id: Option<String>,
    /// Tool name for `BuiltinToolCall` / `McpToolUse`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Behavior id for `RunBehavior`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_id: Option<String>,
    /// Short audit tag from the CallerLink — "ws/42",
    /// "tool/thread-123:tu-4", "lua/5", "internal". Lets the UI show
    /// "who asked for this" without exposing internal struct shape.
    pub caller_tag: String,
    /// RFC3339 timestamp for when the Function was registered. UI
    /// computes elapsed time relative to this.
    pub started_at: String,
}

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
    /// Manually compact `thread_id`. Appends the thread's configured
    /// compaction prompt as a final user message; on turn completion,
    /// the scheduler spawns a new thread seeded with the extracted
    /// summary and sends `ThreadCompacted` linking the two. Rejected
    /// when the thread is mid-turn or its config has
    /// `compaction.enabled = false`.
    CompactThread {
        thread_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// Fork `thread_id` at `from_message_index` into a new thread in
    /// the same pod. The new thread's conversation is the prefix
    /// `[0..from_message_index)`; bindings, config, and tool
    /// allowlist carry over; total_usage is recomputed from the
    /// truncated turn_log; dispatch lineage / behavior origin do
    /// not propagate.
    ///
    /// `from_message_index` must point at a `Role::User` message
    /// (v1 keeps tool_use / tool_result atomicity trivial).
    /// Rejected mid-turn (working / awaiting approval / compacting).
    ///
    /// `archive_original` additionally flips the source's
    /// `archived` flag. Announcement rides the standard
    /// `ThreadCreated` broadcast; the requester gets a correlated
    /// copy so it can subscribe to the new thread id.
    ForkThread {
        thread_id: String,
        from_message_index: usize,
        #[serde(default)]
        archive_original: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// Replace the server-side draft for `thread_id`. Broadcasts
    /// `ThreadDraftUpdated` to every subscriber except the sender
    /// — the sender already has the text locally, and echoing it
    /// back would yank the cursor mid-type. Empty `text` = clear.
    SetThreadDraft {
        thread_id: String,
        text: String,
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

    /// Snapshot of the server's host-env-provider catalog. Used by the
    /// pod editor to populate the per-entry "provider" dropdown.
    /// Server responds with `HostEnvProvidersList`.
    ListHostEnvProviders {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },

    /// Register a new host-env provider with the durable catalog. The
    /// server validates the name (non-empty, not already in use) and
    /// persists the entry to `host_env_providers.toml` before
    /// responding. Responds with `HostEnvProviderAdded` on success or
    /// `Error` on validation failure.
    AddHostEnvProvider {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        name: String,
        url: String,
        /// Control-plane bearer the daemon expects. Omit for dev
        /// daemons running without `--control-token-file`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token: Option<String>,
    },

    /// Replace url / token on an existing catalog entry. Origin and
    /// created_at are preserved. Responds with `HostEnvProviderUpdated`
    /// or `Error` (unknown name, or the entry is a `RuntimeOverlay`
    /// that can't be persisted).
    UpdateHostEnvProvider {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        name: String,
        url: String,
        /// When `None`, the existing token is cleared. To keep the
        /// existing token, clients must read it from — wait, they
        /// can't; tokens are never sent to the client. So: clients
        /// that don't intend to change the token must omit the field
        /// entirely (it'd still be `None` on the wire), which means
        /// "clear it". The UI should make this distinction obvious;
        /// a three-way toggle (keep / set / clear) is the natural
        /// shape, but until we need it we keep the wire simple.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token: Option<String>,
    },

    /// Remove a provider from the catalog. Refused if any loaded pod's
    /// `[[allow.host_env]]` table references the name — the response
    /// is an `Error` listing the referencing pods so the operator can
    /// edit them first. Responds with `HostEnvProviderRemoved` on
    /// success.
    RemoveHostEnvProvider {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        name: String,
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
    /// Shallow directory listing under `<pods_root>/<pod_id>/<path>`. An
    /// empty / absent `path` lists the pod's root. The webui file-tree
    /// panel issues one of these per directory the user expands —
    /// directories aren't enumerated recursively, so a pod with many
    /// threads doesn't pay the full walk cost up front.
    ///
    /// Server filters hidden (dotfile) entries and rejects paths that
    /// escape the pod root; responses flow back as `PodDirListing` on
    /// success or `Error` on failure.
    ListPodDir {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        /// Path relative to the pod directory. Empty / absent = pod root.
        /// Must be a plain relative path — no `..` components, no
        /// absolute prefix.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
    /// Read one text file under a pod. Used by the webui's generic
    /// file viewer for paths that don't route to a specialized editor
    /// (pod.toml and behaviors/* go to their own modals). The server
    /// rejects files larger than a fixed cap and files that look
    /// binary (null bytes in the first 8 KB) — the viewer is a plain
    /// text surface and can't render either. Responses: `PodFileContent`
    /// on success or `Error` on failure.
    ReadPodFile {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        /// Path relative to the pod directory. Same normalization rules
        /// as `ListPodDir.path` (no `..`, no absolute prefix).
        path: String,
    },
    /// Overwrite one text file under a pod. Enforced server-side: the
    /// target must not be on the read-only list (thread JSONs,
    /// `pod_state.json`, behaviors/*/state.json). The webui is trusted
    /// beyond the agent's `pod_write_file` allowlist — if the user can
    /// see a file in the tree and it isn't read-only, they can edit
    /// it. Ack: `PodFileWritten` on success, `Error` on failure.
    WritePodFile {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        path: String,
        content: String,
    },

    // --- Behavior registry (read-only in phase 1 — see
    //     docs/design_behaviors.md). Create / update / delete / run arrive
    //     in phase 2 alongside manual-trigger support. ---
    /// List the behaviors declared under one pod.
    ListBehaviors {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
    },
    /// Read one behavior's full config + prompt + persisted state.
    GetBehavior {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
    },
    /// Manually trigger a behavior: spawn a thread with the behavior's
    /// templated prompt + resolved config + `BehaviorOrigin`, run to
    /// terminal. Request carries an optional JSON payload that is
    /// substituted into the prompt template (`{{payload}}`) and stamped
    /// onto the thread's origin.
    RunBehavior {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
        /// Payload exposed to the prompt template and recorded on
        /// `BehaviorOrigin.trigger_payload`. `None` → treated as
        /// `serde_json::Value::Null`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        payload: Option<serde_json::Value>,
    },
    /// Create a new behavior under the named pod. Writes
    /// `<pod>/behaviors/<behavior_id>/{behavior.toml,prompt.md,state.json}`.
    /// Fails on id collision; to mutate an existing behavior use
    /// `UpdateBehavior`.
    CreateBehavior {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
        config: BehaviorConfig,
        /// Contents of the sibling `prompt.md`. May be empty.
        #[serde(default)]
        prompt: String,
    },
    /// Replace an existing behavior's config + prompt. Does NOT reset
    /// `state.json` — run_count / last_fired_at / etc. are
    /// scheduler-maintained and should survive config edits.
    UpdateBehavior {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
        config: BehaviorConfig,
        #[serde(default)]
        prompt: String,
    },
    /// Remove a behavior. Idempotent-ish — unknown behavior returns an
    /// error, but a repeated call against the same id after deletion is
    /// a different user-level state (id was just removed).
    DeleteBehavior {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
    },
    /// Pause / resume a single behavior. `enabled = false` gates the
    /// cron tick, the webhook endpoint, and startup catch-up for this
    /// behavior; any `QueueOne`-parked payload is dropped on pause.
    /// Manual `RunBehavior` continues to work regardless.
    SetBehaviorEnabled {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
        enabled: bool,
    },
    /// Pod-level master switch for automatic behaviors. Overrides every
    /// behavior in the pod while set — individual `enabled` flags are
    /// still tracked and resume when the pod is re-enabled.
    SetPodBehaviorsEnabled {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        enabled: bool,
    },

    // --- Function registry (display surface) ---
    /// Snapshot of every Function the scheduler currently has in its
    /// `active_functions` registry. Issued on connect so a
    /// newly-connected client isn't blind to Functions that started
    /// before it joined — subsequent lifecycle is delivered via
    /// `FunctionStarted` / `FunctionEnded` broadcasts.
    ListFunctions {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
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
    /// `Thread.draft` for `thread_id` changed. Fanout excludes the
    /// connection that issued the originating `SetThreadDraft` so
    /// the sender's live cursor isn't disrupted by its own echo.
    ThreadDraftUpdated {
        thread_id: String,
        text: String,
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
    /// Compaction finished on `thread_id`: the scheduler spawned
    /// `new_thread_id` seeded with the extracted summary, and
    /// `new_thread_id.continued_from == thread_id`. The original thread
    /// is left in-place (Completed) so clients can still render its
    /// history — a deliberate choice to preserve the compaction
    /// boundary as a historical artifact rather than rewrite the past.
    /// `summary_text` is the body that was extracted and used to seed
    /// the continuation; clients may render it inline without having
    /// to re-parse the old thread.
    ThreadCompacted {
        thread_id: String,
        new_thread_id: String,
        summary_text: String,
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
    /// A user-role message was appended to the thread's conversation.
    /// Fires both for user-typed follow-ups (via `SendUserMessage`) and
    /// for server-injected messages — behavior-trigger prompts,
    /// compaction continuation seeds, `dispatch_thread` async
    /// notification callbacks. Subscribers that are already rendering
    /// the thread append this to their local view; a webui that just
    /// sent a `SendUserMessage` receives its own echo and should not
    /// dedupe optimistically (the wire echo is the source of truth).
    ThreadUserMessage {
        thread_id: String,
        text: String,
    },
    /// A `Role::ToolResult` text message was appended to the
    /// conversation. Fires for async `dispatch_thread` callbacks
    /// (where the child's final result can't bind to the original
    /// tool_use_id because the sync ack already consumed it) and any
    /// future server-injected tool-output text. Distinct from
    /// `ThreadUserMessage` so the webui can render it as a tool
    /// output (parse the dispatched-thread-notification envelope,
    /// route the inner `<result>` into the originating tool call's
    /// result slot, collapsed by default) rather than as a plain
    /// user turn. Structured `ToolResult` content blocks (the normal
    /// synchronous tool-call result path) continue to flow through
    /// `ThreadToolCallEnd` unchanged.
    ThreadToolResultMessage {
        thread_id: String,
        text: String,
    },
    ThreadAssistantBegin {
        thread_id: String,
        turn: u32,
    },
    /// Streaming text fragment. Emitted repeatedly during a turn as the model
    /// produces text. Non-streaming backends synthesize a single delta
    /// carrying the full block so clients only ever handle one event type.
    /// Reconstruct full blocks by concatenating consecutive deltas until the
    /// next non-text event (ReasoningDelta, ToolCallBegin, AssistantEnd).
    ThreadAssistantTextDelta {
        thread_id: String,
        delta: String,
    },
    /// Streaming chain-of-thought fragment — Anthropic extended-thinking,
    /// OpenAI Responses `reasoning`, Gemini `thought:true` text. Same
    /// accumulation rule as [`ThreadAssistantTextDelta`] but opens a
    /// separate, visually-distinct block.
    ThreadAssistantReasoningDelta {
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
        /// Full structured arguments. Carried so the webui can render
        /// rich tool-specific views (e.g. unified diff for edit_file)
        /// without having to wait for a snapshot rebuild. Optional
        /// because the snapshot path doesn't re-emit Begin events
        /// individually — and to give us room to add server-side size
        /// caps later.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<serde_json::Value>,
    },
    /// Streaming content fragment emitted while a tool call is in flight.
    /// Lands between `ThreadToolCallBegin` and `ThreadToolCallEnd`. Each
    /// event carries one `ContentBlock` — for the MVP bash-style MCP tool
    /// this is a `ContentBlock::Text` chunk of stdout/stderr, but the
    /// shape admits images / structured content without further wire
    /// changes. Streaming-decorative: the final integrated tool result
    /// still arrives in the conversation via the snapshot path.
    ThreadToolCallContent {
        thread_id: String,
        tool_use_id: String,
        block: ContentBlock,
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
    /// Snapshot response to `ListHostEnvProviders`. Entries appear in
    /// name-sorted order; no built-ins.
    HostEnvProvidersList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        providers: Vec<HostEnvProviderInfo>,
    },
    /// Successful response to `AddHostEnvProvider`. Carries the
    /// canonicalized entry so the client can update its local view
    /// without re-fetching the whole list.
    HostEnvProviderAdded {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        provider: HostEnvProviderInfo,
    },
    /// Successful response to `UpdateHostEnvProvider`.
    HostEnvProviderUpdated {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        provider: HostEnvProviderInfo,
    },
    /// Successful response to `RemoveHostEnvProvider`. The name is
    /// echoed so clients can identify which local entry to drop.
    HostEnvProviderRemoved {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        name: String,
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
    /// The pod's system-prompt file was rewritten (via the builtin
    /// `pod_write_file` / `pod_edit_file` tool, or a future wire
    /// command). Carries the full new text so subscribed clients can
    /// refresh any rendered view of the prompt.
    PodSystemPromptUpdated {
        pod_id: String,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    PodArchived {
        pod_id: String,
    },
    /// Reply to `ListPodDir`. Echoes the requested `pod_id` + `path`
    /// (empty string for the pod root) alongside the entries so clients
    /// can route the response into the right slot of their tree cache
    /// when multiple expansions are in flight.
    PodDirListing {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        path: String,
        entries: Vec<FsEntry>,
    },
    /// Reply to `ReadPodFile`. `readonly` mirrors `FsEntry.readonly`
    /// so the viewer can hide the Save button without having to
    /// re-classify the path itself.
    PodFileContent {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        path: String,
        content: String,
        readonly: bool,
    },
    /// Ack for a successful `WritePodFile`. Clients use the
    /// `correlation_id` to clear the "saving…" state of a specific
    /// save; the new content on disk is whatever the caller just sent,
    /// so no payload echo is needed.
    PodFileWritten {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        path: String,
    },

    // --- Behavior registry ---
    BehaviorList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        behaviors: Vec<BehaviorSummary>,
    },
    BehaviorSnapshot {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        snapshot: BehaviorSnapshot,
    },
    /// A behavior's persisted state changed (a trigger fired, a run
    /// finished, a queued payload was consumed). Carries a full snapshot
    /// of the state so clients can render badges / last-run info
    /// without tracking deltas.
    BehaviorStateChanged {
        pod_id: String,
        behavior_id: String,
        state: BehaviorState,
    },
    /// A new behavior was created (via `CreateBehavior`). Broadcast so
    /// every client's pod-detail view can fold in the new entry without
    /// a refetch.
    BehaviorCreated {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        summary: BehaviorSummary,
    },
    /// An existing behavior's config / prompt was replaced (via
    /// `UpdateBehavior`). Carries the full new snapshot so clients can
    /// drop any cached config for this behavior and re-render.
    BehaviorUpdated {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        snapshot: BehaviorSnapshot,
    },
    /// A behavior was deleted (via `DeleteBehavior`). Spawned threads
    /// keep their `origin.behavior_id` — the UI resolves those as
    /// "orphaned runs" of a now-deleted behavior.
    BehaviorDeleted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
    },
    /// Pod-level `behaviors_enabled` flag changed (via
    /// `SetPodBehaviorsEnabled`). Individual `BehaviorStateChanged`
    /// events still fire for per-behavior edits; this one covers the
    /// pod-wide toggle so clients can badge the pod header without
    /// inspecting every behavior.
    PodBehaviorsEnabledChanged {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        pod_id: String,
        enabled: bool,
    },

    // --- Function registry (broadcast to every connected client) ---
    /// Reply to `ListFunctions`. Empty list is legitimate — the
    /// scheduler may genuinely have no Functions in flight.
    FunctionList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        functions: Vec<FunctionSummary>,
    },
    /// A new Function was admitted to the registry. Fanout to every
    /// connected client; UI surfaces (status-bar chip, popover list)
    /// use this as the "here's something new" signal.
    FunctionStarted {
        summary: FunctionSummary,
    },
    /// A Function completed (success / error / cancelled). UI drops
    /// the corresponding `function_id` from its local active set.
    FunctionEnded {
        function_id: u64,
        outcome: FunctionOutcomeTag,
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
