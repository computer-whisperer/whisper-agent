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
pub mod tool_surface;

pub use permission::{AllowMap, Disposition};

pub use behavior::{
    BehaviorBindingsOverride, BehaviorConfig, BehaviorOrigin, BehaviorOutcome, BehaviorScope,
    BehaviorScopeCaps, BehaviorSnapshot, BehaviorState, BehaviorSummary, BehaviorThreadOverride,
    CatchUp, Overlap, RetentionPolicy, TriggerSpec,
};
pub use conversation::{
    Attachment, ContentBlock, ContentCapabilities, Conversation, DocumentMime, DocumentSource,
    ImageMime, ImageSource, MediaSupport, Message, ProviderReplay, Role, ToolResultContent,
};
pub use permission::{BehaviorOpsCap, DispatchCap, PodModifyCap};
pub use pod::{
    CompactionConfig, FsEntry, NamedHostEnv, PodAllow, PodAllowCaps, PodConfig, PodLimits,
    PodSnapshot, PodState, PodSummary, ThreadDefaultCaps, ThreadDefaults,
};
pub use tool_surface::{ActivationSurface, CoreTools, InitialListing, ToolSurface};
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
    /// Per-request output cap: the maximum number of tokens the model
    /// is allowed to generate in a single response. **Not** the
    /// model's total context window — that's an upstream property of
    /// the model itself, reported in `ModelSummary::context_window`
    /// when the provider exposes it.
    pub max_tokens: u32,
    /// Maximum assistant turns in a single user-message cycle before
    /// the scheduler stops generating. Prevents runaway tool-loop
    /// chains; resets each time the user sends a new message.
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
    /// Override for [`ThreadConfig::max_tokens`] — the per-request
    /// output cap on this thread. `None` inherits the pod default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Override for [`ThreadConfig::max_turns`] — per-cycle assistant
    /// turn limit. `None` inherits the pod default.
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
    /// Backend catalog name (e.g. `"anthropic"`). Empty string means
    /// "no backend bound" — model calls fail until the thread is
    /// rebound. Kept as a name rather than a `BackendId` so display
    /// surfaces (UI labels, logs) don't have to strip the `backend-`
    /// prefix.
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
/// [`ThreadBindingsRequest`] names entries by string only — so the
/// wire can't sneak a spec past the pod cap. Kept as a variant
/// (rather than collapsing the enum to plain strings) so the
/// stored-state type is stable when the subagent flow lands.
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
    /// Last-known daemon liveness for this provider. Updated from
    /// both periodic probes (on a background timer) and opportunistic
    /// signals (provision success / connect-fail). `Unknown` means
    /// the probe hasn't run yet — fresh registrations start here.
    #[serde(default)]
    pub reachability: HostEnvReachability,
}

/// Daemon-liveness status carried on `HostEnvProviderInfo`. Times are
/// RFC-3339 strings so the protocol crate can stay chrono-free.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum HostEnvReachability {
    /// No probe has completed yet. Also the state on boot before the
    /// first tick fires.
    #[default]
    Unknown,
    /// Last probe or provision succeeded. `at` is the observation time.
    Reachable { at: String },
    /// Last probe failed. `since` is the first failed observation
    /// after the last `Reachable`, or first failure at all if we've
    /// never seen it. `last_error` carries the most recent failure
    /// message for operator display.
    Unreachable { since: String, last_error: String },
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

/// Server-advertised entry from the shared-MCP-host catalog. Unlike
/// host-env providers (which are our own daemons we spawn / own),
/// shared MCP hosts are endpoints the operator points us at — often
/// third-party servers that require their own authentication. The
/// catalog lets operators register these at runtime instead of only
/// via TOML at boot, and carries enough auth metadata to connect
/// without leaking secrets back to clients.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SharedMcpHostInfo {
    /// User-facing catalog name; matches `ThreadConfig.shared_mcp_hosts`
    /// entries and the key in pod `bindings.mcp_hosts`.
    pub name: String,
    /// Endpoint URL (streamable-HTTP or SSE MCP transport).
    pub url: String,
    /// How this entry entered the catalog. Mirrors the host-env shape
    /// (`Seeded` from `[shared_mcp_hosts]` TOML, `Manual` from WebUI /
    /// RPC, `RuntimeOverlay` from a CLI `--shared-mcp-host` flag).
    pub origin: HostEnvProviderOrigin,
    /// Non-secret auth classification. The secret itself (bearer token,
    /// OAuth tokens) is never sent to the client; the UI gets just
    /// enough to render "anonymous" / "bearer-auth" / etc.
    pub auth: SharedMcpAuthPublic,
    /// Whether the server currently holds a connected MCP session for
    /// this host. `true` means `tools/list` has succeeded at least once;
    /// `false` means the connect attempt failed and the entry is
    /// registered but non-functional until the next successful update.
    pub connected: bool,
    /// Latest connect error if `connected` is false. Empty when
    /// connected or when the host has never been attempted. Operators
    /// use this to diagnose bad URLs / invalid tokens without digging
    /// in server logs.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub last_error: String,
}

/// Public view of a shared-MCP-host's authentication configuration —
/// the bits a client is allowed to see. Tagged union keyed by `kind`;
/// new auth methods land additively (Oauth2 added for third-party
/// servers that require a browser-driven authorization flow).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SharedMcpAuthPublic {
    /// Anonymous access — the server attaches no `Authorization`
    /// header. Used by dev loopback servers and MCPs that don't
    /// require auth.
    #[default]
    None,
    /// Static bearer token provided out-of-band (e.g. Slack app bot
    /// tokens, long-lived PATs). The token is stored server-side; the
    /// client only learns that one is configured.
    Bearer,
    /// OAuth 2.1 handshake completed. The catalog holds an access
    /// token + refresh token; `issuer` and `scope` are the non-secret
    /// pieces the UI shows so operators can tell OAuth backings apart.
    Oauth2 {
        issuer: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
    },
}

/// Client-supplied auth payload for Add/Update. Carries secrets in
/// the `Bearer` variant and the request-shape for `Oauth2Start`; never
/// echoed back. `Oauth2Start` begins a flow rather than completing
/// one: the server replies with a `SharedMcpOauthFlowStarted` event
/// carrying the URL the webui should open, and finishes the flow
/// itself when the authorization server redirects to
/// `/oauth/callback`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SharedMcpAuthInput {
    #[default]
    None,
    Bearer {
        token: String,
    },
    /// Start an OAuth 2.1 authorization_code + PKCE flow. `scope` is
    /// optional; omit to let the server ask the AS metadata what it
    /// needs. Only valid on `AddSharedMcpHost` (OAuth hosts can't be
    /// created as anonymous-then-upgraded via Update — the discovery
    /// + DCR only makes sense up front).
    Oauth2Start {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
        /// Origin the webui is served from — e.g.
        /// `http://127.0.0.1:8080`. The server uses this to build
        /// the redirect URI (`<redirect_base>/oauth/callback`) that's
        /// sent to the authorization server and embedded in the URL
        /// the webui opens. Letting the webui dictate this keeps the
        /// server correct across deployments (loopback-only, LAN,
        /// reverse-proxied) without any infer-from-Host guesswork.
        redirect_base: String,
    },
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
    /// Auth mode the backend was built with (`"api_key"`,
    /// `"chatgpt_subscription"`, `"google_oauth"`). None when the
    /// backend skips auth entirely (local OpenAI-compatible endpoints).
    /// Surfaced so the settings panel can show where each credential
    /// lives without leaking the credential itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_mode: Option<String>,
}

/// Entry in a `ModelsList` response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelSummary {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Maximum input context the model accepts, in tokens. `None`
    /// means the provider's `/models` endpoint doesn't publish this
    /// — today Anthropic and OpenAI both omit it, Gemini's API-key
    /// route reports it, llama.cpp reports it via `/props`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Provider-declared ceiling on output tokens per request. `None`
    /// when not exposed by the upstream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Which media kinds/MIMEs the model accepts and emits. Adapters
    /// populate the input side from a per-model table; the output
    /// side is empty today (v2 when Gemini / OpenAI-Responses image
    /// output lands). Empty for models we haven't explicitly
    /// classified — treat that as "text only" at the UI layer.
    #[serde(default, skip_serializing_if = "ContentCapabilities_is_default")]
    pub capabilities: ContentCapabilities,
}

#[allow(non_snake_case)]
fn ContentCapabilities_is_default(c: &ContentCapabilities) -> bool {
    c.input.is_empty() && c.output.is_empty()
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
    /// Live entry suddenly became unusable after having been Ready —
    /// the daemon backing it lost our session (typical cause: the
    /// daemon's host powered off or the daemon restarted). Distinct
    /// from `Errored` (which means provisioning failed from the
    /// start) so the UI can distinguish "this never worked" from
    /// "this was working and went away".
    Lost,
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
    /// The thread's effective permission scope, as snapshotted at
    /// creation and widened by any escalation grants so far. Lets
    /// clients render what the thread is allowed to do without
    /// re-deriving from pod config.
    #[serde(default)]
    pub scope: crate::permission::Scope,
}

// ---------- Knowledge buckets ----------

/// Wire snapshot of a knowledge bucket as the server's bucket registry
/// knows it. The `id` is the bucket's directory name and is permanent;
/// `name` is the user-editable display label from `bucket.toml`.
///
/// `scope` is `"server"` today — pod-scoped buckets land in a follow-up
/// slice, when this becomes a tagged enum. Until then a plain string is
/// enough for the WebUI to label entries without locking the protocol
/// shape down prematurely.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct BucketSummary {
    pub id: String,
    pub scope: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// `"stored"`, `"linked"`, or `"managed"` — matches the
    /// `bucket.toml [source]` tag.
    pub source_kind: String,
    /// Human-readable source target (path / archive). `None` for
    /// `managed` buckets, which have no external source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_detail: Option<String>,
    /// Provider name from `bucket.toml [defaults] embedder` — matches an
    /// entry in `embedding_providers.*` from the server config.
    pub embedder_provider: String,
    pub dense_enabled: bool,
    pub sparse_enabled: bool,
    /// RFC-3339 — `bucket.toml [created_at]`.
    pub created_at: String,
    /// `None` when the bucket has no slot yet (just created, never
    /// built) or its only slot failed before reaching `Ready`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_slot: Option<ActiveSlotSummary>,
}

/// Wire snapshot of the bucket's active slot manifest, surfaced
/// alongside the bucket. Heavy artifacts (chunks, vectors, dense /
/// sparse indexes) are not exposed — only what the UI needs to show
/// "is this bucket ready, how big is it, what did it index."
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ActiveSlotSummary {
    pub slot_id: String,
    pub state: SlotStateLabel,
    /// Embedder model id the slot was actually built against — captured
    /// from the provider's `/info` at build time, not the config name.
    /// Lets the UI flag drift if the provider has been silently
    /// rebound to a different model.
    pub embedder_model: String,
    pub dimension: u32,
    pub chunk_count: u64,
    pub vector_count: u64,
    pub disk_size_bytes: u64,
    /// RFC-3339 — slot creation time (set when the build started).
    pub created_at: String,
    /// RFC-3339 — when the slot first reached `Ready`. `None` while
    /// `state` is `Planning` or `Building`, or for slots that `Failed`
    /// before completing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub built_at: Option<String>,
}

/// Lifecycle state of a slot. Mirrors the server-side
/// `knowledge::manifest::SlotState`; serialized snake_case to match
/// `manifest.toml`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SlotStateLabel {
    Planning,
    Building,
    Ready,
    Failed,
    Archived,
}

/// One reranked candidate returned by a `QueryBuckets` request.
/// Mirrors the server-side `knowledge::query::RerankedCandidate` —
/// `chunk_id` is hex, `source_path` is the retrieval path
/// (`"dense"` or `"sparse"`) the candidate originated on, and
/// `source_score`'s magnitude is path-and-provider-specific (compare
/// only within a single response and only per-path; use
/// `rerank_score` for ordering decisions).
///
/// `source_id` and `source_locator` carry the chunk's adapter-level
/// provenance — for `markdown_dir` `source_id` is the file path; for
/// `mediawiki_xml` it's the article title. `source_locator` is the
/// chunker's per-chunk pointer into the record (`"chars 200-700"`).
/// Both are surfaced so the WebUI can cite the source and the
/// `knowledge_query` builtin can include it in tool output.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct QueryHit {
    pub bucket_id: String,
    pub chunk_id: String,
    pub chunk_text: String,
    pub source_path: String,
    pub source_score: f32,
    pub rerank_score: f32,
    #[serde(default)]
    pub source_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_locator: Option<String>,
}

/// Source-of-truth descriptor for a bucket creation request. Mirrors the
/// server-side `SourceConfig` enum but kept narrow to what the WebUI form
/// captures — no compaction / rescan tunables yet (those default at
/// bucket-toml synthesis time and can be edited later through raw TOML).
///
/// `adapter` is a free-form string; the server validates it against the
/// adapters it actually knows how to build (`mediawiki_xml` for `stored`,
/// `markdown_dir` for `linked`). Sending an unknown combo lands an
/// `Error` at create time rather than a parse failure on next registry
/// load.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BucketSourceInput {
    Stored {
        adapter: String,
        archive_path: String,
    },
    Linked {
        adapter: String,
        path: String,
    },
    Managed {},
}

/// Form payload for `CreateBucket`. The server synthesizes a fresh
/// `bucket.toml` from this and writes it under
/// `<buckets_root>/<id>/bucket.toml`. Defaults that aren't user-tunable
/// at v1 (compaction thresholds, serving mode, quantization) are filled
/// in server-side rather than asking the WebUI to surface every knob.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct BucketCreateInput {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub source: BucketSourceInput,
    /// Provider name from `[embedding_providers.*]` server config.
    pub embedder: String,
    pub chunk_tokens: u32,
    pub overlap_tokens: u32,
    pub dense_enabled: bool,
    pub sparse_enabled: bool,
}

/// Coarse phase tag emitted by `BucketBuildProgress`. The build pipeline
/// has more sub-stages internally (per-batch embed calls, vector store
/// flushes); this rolls them into the five the UI actually wants to
/// distinguish — optional download for tracked buckets, record
/// enumeration, the long streaming phase, the dominant HNSW build,
/// then the quick finalize-and-promote tail.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BucketBuildPhase {
    /// `kind = "tracked"` only — the feed driver is fetching the base
    /// snapshot into the bucket's `source-cache/` directory before any
    /// indexing work begins. Per-byte progress is not yet surfaced;
    /// expect minutes to hours depending on dataset (~24 GB for
    /// enwiki). Stored / linked / managed buckets skip this phase
    /// entirely.
    Downloading,
    /// Walking the source, hashing each record, recording `Planned`
    /// entries in `build.state`. Cheap relative to `Indexing`;
    /// produces the record-count total that makes the indexing
    /// progress bar meaningful.
    Planning,
    /// Source → chunks → embed → write. `chunks` ticks up here.
    Indexing,
    /// HNSW graph build over the just-written vectors. The single
    /// dominant cost on wikipedia-scale builds; counters don't move
    /// during this phase, only the phase tag does.
    BuildingDense,
    /// Open readers, write final manifest, atomic active-pointer flip.
    /// Sub-second on every build measured so far.
    Finalizing,
}

/// Terminal state for a build, carried by `BucketBuildEnded`. `Error`
/// folds in the failure reason so the UI doesn't need a sibling event
/// for it.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "tag", rename_all = "snake_case")]
pub enum BucketBuildOutcome {
    Success,
    Error { message: String },
    Cancelled,
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
    CancelThread,
    RunBehavior,
    BuiltinToolCall,
    McpToolUse,
    /// A model-initiated `sudo` call — the model wants to invoke a
    /// named tool with explicit user approval. Stays in the registry
    /// until the owning interactive channel resolves it via
    /// `ClientToServer::ResolveSudo`.
    Sudo,
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
        /// Media attached to the initial user message (images today,
        /// audio/documents later). Rendered by the scheduler into
        /// additional content blocks on the first user turn,
        /// interleaved after the text. Empty on every pre-existing
        /// client.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        initial_attachments: Vec<Attachment>,
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
        /// Media attached to this user message. Rendered by the
        /// scheduler as additional content blocks after the text.
        /// Empty when the client sends a plain text-only turn.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<Attachment>,
    },
    /// Cancel a task. Stubbed for now — flips state to `Cancelled` without rolling back
    /// any in-flight tool work. Proper rollback semantics are a v0.3 problem.
    CancelThread {
        thread_id: String,
    },
    /// Promote a `Failed` thread back to `Idle` so it can accept a
    /// new user message. Rejected (`Error` reply) when the thread
    /// isn't in `Failed`. Defensively synthesizes trailing
    /// `tool_result` blocks if the conversation ends with an
    /// unmatched `assistant[tool_use]`, so the recovered thread's
    /// next model call lands on a well-formed conversation.
    RecoverThread {
        thread_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
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
    /// `[0..from_message_index)`; `total_usage` is recomputed from
    /// the truncated turn_log; dispatch lineage / behavior origin
    /// do not propagate.
    ///
    /// `reset_capabilities` picks between two capability-inheritance
    /// modes:
    /// - `false` (default): the fork inherits the source's live
    ///   bindings, scope, config, and tool_surface verbatim. Good
    ///   for "continue this conversation from here with all my
    ///   sudo-remembers and current settings intact."
    /// - `true`: the fork re-derives bindings, scope, config, and
    ///   tool_surface from the pod's current `thread_defaults` —
    ///   equivalent to a fresh thread carrying the source's
    ///   conversation prefix. Use this to pick up newly-added MCP
    ///   hosts, sandbox bindings, or cap changes that post-date
    ///   the source thread.
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
        #[serde(default)]
        reset_capabilities: bool,
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

    /// Enumerate shared-MCP-host catalog entries. Server responds with
    /// `SharedMcpHostsList`.
    ListSharedMcpHosts {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },

    /// Register a shared MCP host with the durable catalog. The server
    /// attempts to open an MCP session using the supplied auth before
    /// persisting — on handshake failure, the entry is not added and
    /// the response is an `Error`. Admin-only. Responds with
    /// `SharedMcpHostAdded` on success.
    AddSharedMcpHost {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        name: String,
        url: String,
        #[serde(default)]
        auth: SharedMcpAuthInput,
    },

    /// Replace url / auth on an existing shared-MCP-host entry. Opens a
    /// fresh session against the new url+auth; if that succeeds the old
    /// session is swapped out (in-flight tool calls keep running on the
    /// old `Arc<McpSession>` until they complete). Admin-only. Omit
    /// `auth` to leave it unchanged; pass `{ "kind": "none" }` to
    /// explicitly clear. Responds with `SharedMcpHostUpdated`.
    UpdateSharedMcpHost {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        name: String,
        url: String,
        /// `None` means "leave auth unchanged" so a URL-only edit
        /// doesn't force the client to re-enter a bearer token it
        /// can't see. To clear auth, pass `Some(SharedMcpAuthInput::None)`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        auth: Option<SharedMcpAuthInput>,
    },

    /// Remove a shared MCP host. Refused when any live thread currently
    /// opts into it (the registry's `users` set is non-empty) — the
    /// operator should end / retarget those threads first. Admin-only.
    /// Responds with `SharedMcpHostRemoved`.
    RemoveSharedMcpHost {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        name: String,
    },

    // --- Server-settings mutations (admin-only) ---
    /// Replace the Codex `auth.json` contents for a ChatGPT-subscription
    /// backend. Server validates the JSON against the Codex schema,
    /// writes it to the backend's configured path, and swaps the
    /// in-memory auth state so in-flight and subsequent requests pick
    /// it up without a restart. Admin-only — plain client tokens are
    /// rejected. Server responds with `CodexAuthUpdated` on success or
    /// `Error` on validation / IO / permission failure.
    UpdateCodexAuth {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        /// Backend name as it appears in `BackendsList` / the TOML
        /// `[backends.<name>]` table. Must reference a backend whose
        /// auth_mode is `chatgpt_subscription`.
        backend: String,
        /// Full `auth.json` contents — typically the result of
        /// `cat ~/.codex/auth.json` on a machine that just ran
        /// `codex login`.
        contents: String,
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
    /// Read the server-level `whisper-agent.toml` raw text. Admin-only —
    /// the file contains API keys and bearer tokens. The server replies
    /// with `ServerConfigFetched` on success, `Error` if not authorized
    /// or if the server was started without `--config` (no path to
    /// read).
    FetchServerConfig {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// Replace `whisper-agent.toml` with the given text. Admin-only.
    /// Server parses and validates the new config, then atomically
    /// hot-swaps the in-memory backend catalog: any thread bound to a
    /// removed or modified backend is cancelled before the swap so it
    /// can't silently transition to a new provider mid-conversation.
    /// Other config sections (shared_mcp_hosts, host_env_providers,
    /// secrets, auth) are persisted to disk but require a server
    /// restart to take effect — the server reports them in the
    /// response's `restart_required_sections` list. Response:
    /// `ServerConfigUpdateResult` on success, `Error` on validation or
    /// disk-write failure.
    UpdateServerConfig {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
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

    // --- Knowledge buckets ---
    /// Snapshot every bucket the registry knows about. Server responds
    /// with `BucketsList`. Cheap; reads cached registry state, no I/O.
    ListBuckets {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// Run a hybrid (dense + sparse + reranker) query against one or
    /// more buckets. Server resolves each bucket's embedder via its
    /// `bucket.toml`'s `[defaults] embedder = "..."`, picks any
    /// configured reranker (slice 9 punts on per-bucket reranker
    /// selection), and responds with `QueryResults` or `Error`.
    ///
    /// First query against an unloaded bucket pays the slot-load cost
    /// (HNSW rebuild from `vectors.bin`); subsequent queries hit the
    /// scheduler-side bucket cache and return in milliseconds.
    QueryBuckets {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        /// Bucket ids to query across. Slice 9 supports exactly one;
        /// multi-bucket fan-out (which already works in `QueryEngine`)
        /// arrives once the dimension-matching UX is figured out.
        bucket_ids: Vec<String>,
        query: String,
        /// Final result count returned to the caller after reranking.
        /// `0` is rejected with `Error` rather than silently returning
        /// nothing.
        top_k: u32,
    },
    /// Create a new bucket. Server validates `id` (filesystem-safe ASCII)
    /// and the embedder reference, synthesizes `bucket.toml` from
    /// `config`, writes it under `<buckets_root>/<id>/`, inserts the
    /// entry into the in-memory registry, and broadcasts
    /// `BucketCreated`. The freshly-created bucket has no active slot
    /// yet — `StartBucketBuild` is a separate call.
    CreateBucket {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        id: String,
        config: BucketCreateInput,
    },
    /// Remove a bucket from the registry and rmdir its directory. Cancels
    /// any in-flight build for the same id first. Idempotent on the wire
    /// (deleting an unknown id returns `Error`); broadcasts `BucketDeleted`
    /// on success so peer clients drop their local entry.
    DeleteBucket {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        id: String,
    },
    /// Kick off a slot build for `id`. The build runs as a detached
    /// background task off the scheduler thread (HNSW build is
    /// `spawn_blocking`'d so it doesn't pin a runtime worker).
    /// Refusal is immediate via `Error` (already-building, unknown
    /// bucket, missing embedder, missing source archive); acceptance is
    /// signalled by a `BucketBuildStarted` broadcast carrying the new
    /// slot id, followed by periodic `BucketBuildProgress` updates and a
    /// terminal `BucketBuildEnded`.
    StartBucketBuild {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        id: String,
    },
    /// Cancel an in-flight build for `id`. Fires the build task's
    /// cancellation token; the task observes it at the next chunk-batch
    /// boundary or HNSW build entry, finalizes the failed slot, and
    /// emits `BucketBuildEnded { outcome: Cancelled }`. Idempotent —
    /// cancelling when no build is running returns `Error` rather than
    /// silently succeeding so the UI can keep its model honest.
    CancelBucketBuild {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        id: String,
    },
    /// Manually trigger a tracked-bucket feed worker to poll its
    /// driver immediately rather than waiting for the next cadence
    /// tick. Idempotent on the wire — a poll already in flight
    /// coalesces silently (the trigger channel is bounded at 1).
    /// Refusal (`Error`) for unknown bucket id or for non-tracked
    /// buckets that don't have a worker. Acceptance is signalled
    /// synchronously via an empty-body success ack; the resulting
    /// `TickOutcome` lands through the worker's normal observer
    /// path (today's wiring is `NoopObserver` in production, so
    /// callers infer success from the bucket's updated state on
    /// the next `ListBuckets` rather than a per-tick wire event —
    /// per-tick broadcasting is a follow-up).
    PollFeedNow {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        id: String,
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

    // --- Sudo ---
    /// User's answer to a pending `Sudo` Function — the approval reply
    /// to a model's `sudo` tool call. On approve (once or remember) the
    /// scheduler dispatches the wrapped tool with pod-ceiling caps and
    /// sends the result back as the sudo call's tool_result. On
    /// `approve_remember`, the wrapped tool name is additionally flipped
    /// to `Allow` in the thread's `scope.tools` so future direct calls
    /// skip the approval prompt. On reject the `reason` (if any) is
    /// delivered to the model as the tool_result error text.
    ResolveSudo {
        function_id: u64,
        decision: crate::permission::SudoDecision,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
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
        /// Media that was attached alongside the text, in the order
        /// the scheduler placed them in the underlying content-block
        /// sequence. Empty for server-injected messages (behavior
        /// prompts, compaction seeds, etc.) and for text-only
        /// `SendUserMessage` echoes.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<Attachment>,
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
    /// Mid-prefill progress heartbeat. Only emitted by backends that can
    /// actually observe prefill — today, the `llamacpp` driver via its
    /// `/slots` poller. Other backends are silent between
    /// [`ServerToClient::ThreadAssistantBegin`] and the first
    /// [`ServerToClient::ThreadAssistantTextDelta`] /
    /// [`ServerToClient::ThreadAssistantReasoningDelta`]. Ephemeral:
    /// clients render a transient progress bar while the stream of
    /// these events is active and drop it on first delta. Nothing is
    /// persisted.
    ThreadPrefillProgress {
        thread_id: String,
        /// Prompt tokens ingested so far.
        tokens_processed: u32,
        /// Total prompt length the backend is working through.
        tokens_total: u32,
    },
    /// In-flight tool-call placeholder. Emitted while the model is
    /// still streaming its arguments JSON — before the scheduler has
    /// dispatched the call and fired
    /// [`ServerToClient::ThreadToolCallBegin`]. Lets the UI frame the
    /// row with the tool's name and a running char count instead of
    /// dead silence during a long args-JSON write. Clients drop the
    /// placeholder on the matching `ThreadToolCallBegin` (same
    /// `tool_use_id`). Ephemeral; not persisted.
    ThreadToolCallStreaming {
        thread_id: String,
        tool_use_id: String,
        name: String,
        /// Cumulative length of the args-JSON buffered so far. Not a
        /// token count — providers don't surface those mid-stream.
        args_chars: u32,
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
    /// A model-emitted image attachment on the current assistant turn —
    /// fired once per `inlineData` part for providers that can return
    /// native image output (Gemini's `gemini-2.5-flash-image*` family
    /// today; OpenAI's `image_generation` built-in tool will flow
    /// through a different path). Bytes arrive whole, not as a stream
    /// of deltas, so this event is one-shot per image. The persisted
    /// `ContentBlock::Image` lands in the conversation snapshot via
    /// `ThreadSnapshot`; this event is the live-update channel so
    /// connected webuis can render the thumbnail immediately rather
    /// than waiting on a snapshot rebuild.
    ThreadAssistantImage {
        thread_id: String,
        source: ImageSource,
    },
    /// Streaming chain-of-thought fragment — Anthropic extended-thinking,
    /// OpenAI Responses `reasoning`, Gemini `thought:true` text. Same
    /// accumulation rule as [`ThreadAssistantTextDelta`] but opens a
    /// separate, visually-distinct block.
    ThreadAssistantReasoningDelta {
        thread_id: String,
        delta: String,
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
    /// Snapshot response to `ListSharedMcpHosts`. Entries in
    /// name-sorted order.
    SharedMcpHostsList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        hosts: Vec<SharedMcpHostInfo>,
    },
    /// Successful response to `AddSharedMcpHost`. Broadcast so every
    /// connected client's view stays in sync without re-fetching the
    /// full list.
    SharedMcpHostAdded {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        host: SharedMcpHostInfo,
    },
    /// Successful response to `UpdateSharedMcpHost`. Broadcast.
    SharedMcpHostUpdated {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        host: SharedMcpHostInfo,
    },
    /// Successful response to `RemoveSharedMcpHost`. Broadcast.
    SharedMcpHostRemoved {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        name: String,
    },
    /// Intermediate response to an `AddSharedMcpHost` with
    /// `auth = Oauth2Start`. The server has discovered the AS
    /// metadata, optionally DCR'd, and built the authorization URL.
    /// The webui should open `authorization_url` in a new browser
    /// window; the final outcome arrives later as
    /// `SharedMcpHostAdded` (success) or `Error` (failure).
    SharedMcpOauthFlowStarted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        /// Target name the host will be filed under on success. Echoed
        /// so the webui can keep its pending-form state tied to the
        /// right future completion.
        name: String,
        /// URL the webui opens in a new tab/window for the user to
        /// authorize. Carries `state` so the callback can be matched
        /// back to this in-flight flow.
        authorization_url: String,
    },
    /// ACK for `UpdateCodexAuth`. The backend named `backend` has its
    /// in-memory auth state refreshed; subsequent requests will use the
    /// new tokens. Broadcast to every connected client so a concurrent
    /// admin session sees the change.
    CodexAuthUpdated {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        backend: String,
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
    /// Reply to `FetchServerConfig`. Sent only to the originating
    /// connection — the file contains secrets and is not broadcast.
    ServerConfigFetched {
        toml_text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
    /// Reply to a successful `UpdateServerConfig`. Sent only to the
    /// originator (other clients are notified about the catalog change
    /// via a follow-up `BackendsList` broadcast). `cancelled_threads`
    /// lists thread ids that were cancelled because their backend was
    /// removed or modified. `restart_required_sections` lists config
    /// sections that changed but only take effect on next server
    /// start. `pods_with_missing_backends` lists pod ids whose
    /// `[allow].backends` references one of the removed backends —
    /// the user is expected to edit those pods or risk thread-creation
    /// failures.
    ServerConfigUpdateResult {
        cancelled_threads: Vec<String>,
        restart_required_sections: Vec<String>,
        pods_with_missing_backends: Vec<String>,
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

    // --- Knowledge buckets ---
    /// Snapshot response to `ListBuckets`. Entries appear in id order
    /// (bucket directory name). Empty when the server has no buckets
    /// configured — the WebUI then offers the "create your first
    /// bucket" affordance.
    BucketsList {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        buckets: Vec<BucketSummary>,
    },
    /// Response to a successful `QueryBuckets`. `query` echoes the
    /// request so concurrent queries don't muddle their result panes
    /// when ordering across the wire is non-deterministic. Hits are
    /// already sorted by descending `rerank_score`.
    QueryResults {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        query: String,
        hits: Vec<QueryHit>,
    },
    /// A bucket was created (via `CreateBucket`). Broadcast so peer
    /// clients can fold the new entry into their list view without a
    /// `ListBuckets` round-trip.
    BucketCreated {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        summary: BucketSummary,
    },
    /// A bucket was removed (via `DeleteBucket`). Broadcast so peer
    /// clients drop the row + any open progress UI keyed off this id.
    BucketDeleted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        id: String,
    },
    /// A `StartBucketBuild` was accepted and the build task started.
    /// Broadcast so every client paints the row as building, not just
    /// the requester. `slot_id` is the freshly-minted slot id; it stays
    /// stable across the build's lifetime.
    BucketBuildStarted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        bucket_id: String,
        slot_id: String,
    },
    /// Periodic progress tick during a build. Throttled to ~1Hz on the
    /// server side so this stream stays cheap regardless of how fast
    /// chunks land. Counters monotonically increase within a single
    /// build and reset implicitly when a new build starts.
    BucketBuildProgress {
        bucket_id: String,
        slot_id: String,
        phase: BucketBuildPhase,
        source_records: u64,
        chunks: u64,
    },
    /// Terminal event for a build. `summary` carries the bucket's new
    /// snapshot when the build succeeded (so clients can update the row
    /// without `ListBuckets`); `None` when the slot never reached
    /// Ready (cancelled / failed mid-build).
    BucketBuildEnded {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        bucket_id: String,
        slot_id: String,
        outcome: BucketBuildOutcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<BucketSummary>,
    },
    /// `PollFeedNow` was accepted — the worker has been signalled.
    /// Sent only to the requesting client (it's an ack of *their*
    /// click), not broadcast. The actual `TickOutcome` of the
    /// triggered poll lands through the worker's normal observer
    /// path — UI consumers refresh the bucket via `ListBuckets` to
    /// see the post-poll state.
    FeedPollAccepted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        bucket_id: String,
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

    // --- Sudo ---
    /// A thread's model issued `sudo(tool_name, args, reason)`. Delivered
    /// only to the thread's interactive channel; autonomous threads
    /// cannot emit this (the `sudo` tool hard-errors on threads without
    /// an interactive approver). The client renders an approval UI
    /// showing the wrapped tool name, the inner args, and the model's
    /// reason, then replies with `ClientToServer::ResolveSudo`.
    SudoRequested {
        function_id: u64,
        thread_id: String,
        tool_name: String,
        args: serde_json::Value,
        /// Model-supplied justification. Free-form string.
        reason: String,
    },
    /// A sudo Function completed. Broadcast to subscribers so any
    /// observer (not just the approver) can reflect the resolution.
    SudoResolved {
        function_id: u64,
        thread_id: String,
        decision: crate::permission::SudoDecision,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fork_thread_reset_capabilities_defaults_false() {
        // Old clients (and anyone sending a minimal message) must
        // still parse into the current-behavior branch. `#[serde(default)]`
        // is the contract — guard against someone renaming the field
        // or dropping the attribute.
        let json = r#"{
            "type": "fork_thread",
            "thread_id": "task-1",
            "from_message_index": 2
        }"#;
        let msg: ClientToServer = serde_json::from_str(json).unwrap();
        match msg {
            ClientToServer::ForkThread {
                reset_capabilities,
                archive_original,
                ..
            } => {
                assert!(!reset_capabilities);
                assert!(!archive_original);
            }
            other => panic!("expected ForkThread, got {other:?}"),
        }
    }

    #[test]
    fn fork_thread_reset_capabilities_roundtrip() {
        let json = r#"{
            "type": "fork_thread",
            "thread_id": "task-1",
            "from_message_index": 2,
            "reset_capabilities": true
        }"#;
        let msg: ClientToServer = serde_json::from_str(json).unwrap();
        match msg {
            ClientToServer::ForkThread {
                reset_capabilities, ..
            } => assert!(reset_capabilities),
            other => panic!("expected ForkThread, got {other:?}"),
        }
    }
}
