//! Central scheduler: single tokio task driving all tasks through their state machines.
//!
//! Owns the authoritative state — tasks, clients, subscriptions, per-task MCP sessions
//! and tool descriptor caches. The server's WebSocket handler only encodes/decodes
//! frames and forwards decoded [`ClientToServer`] messages to this scheduler via an
//! mpsc inbox.
//!
//! The run loop ([`run`]) is a `select!` over:
//!   - **Inbox** messages: client frames, client registration/unregistration.
//!   - **Pending I/O** completions: model responses, MCP connects, tool results.
//!
//! Each event triggers:
//!   1. `apply_*` — mutate task state, collect [`ThreadEvent`]s from the state machine.
//!   2. `router.dispatch_events` — translate task events to wire events and broadcast.
//!   3. `step_until_blocked` — keep calling `task.step()` until the task requests I/O
//!      or pauses; pushes fresh I/O futures to the `FuturesUnordered`.
//!
//! Tasks-as-data discipline: state mutation happens exclusively in this module (via
//! `task.step()` / `task.apply_io_result()`). No other code path writes to a Thread.

mod behaviors;
mod bindings;
mod buckets;
mod client_messages;
mod compaction;
mod config_updates;
mod dispatch;
mod feed_workers;
mod functions;
mod retention;
mod server_config;
mod thread_config;
mod triggers;

pub use self::thread_config::build_default_pod_config;

use self::bindings::resolve_bindings_choice;
use self::retention::{RetentionAction, archive_thread_json, delete_thread_json};
use self::thread_config::{apply_config_override, base_thread_config_from_pod};

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use whisper_agent_protocol::{
    ClientToServer, HostEnvBinding, ResourceKind, ServerToClient, ThreadBindings,
    ThreadBindingsRequest, ThreadConfigOverride, ThreadStateLabel,
};

use crate::knowledge::BucketRegistry;
use crate::pod::persist::{LoadedState, Persister};
use crate::pod::resources::{
    BackendId, CompleteHostEnvOutcome, HostEnvId, McpHostId, ResourceRegistry,
};
use crate::pod::{Pod, PodId};
use crate::providers::embedding::EmbeddingProvider;
use crate::providers::model::ModelProvider;
use crate::providers::rerank::RerankProvider;
use crate::runtime::audit::AuditLog;
use crate::runtime::io_dispatch::{
    self, IoCompletion, ProvisionCompletion, ProvisionPhase, ProvisionResult, SchedulerCompletion,
    SchedulerFuture,
};
use crate::runtime::thread::{IoResult, OpId, StepOutcome, Thread, derive_title, new_task_id};
use crate::server::thread_router::ThreadEventRouter;
use crate::tools::mcp::{McpSession, ToolAnnotations, ToolDescriptor as McpTool};
use crate::tools::sandbox::HostEnvProvider;
use whisper_agent_protocol::HostEnvSpec;

pub type ConnId = u64;

/// How `route_tool` wants a given tool invocation handled. Produced
/// during dispatch and consumed by `io_dispatch::tool_call`.
pub enum ToolRoute {
    /// Handle in-process via [`crate::tools::builtin_tools::dispatch`], scoped
    /// to the pod's own directory.
    Builtin { pod_id: PodId },
    /// Forward to the named MCP session via JSON-RPC. `host` is the
    /// scheduler-side host name (used as the `host:` field on the
    /// `Function::McpToolUse` spec). `real_name` is the name the MCP
    /// host advertises server-side — equal to the public tool name for
    /// shared MCPs, and the public name minus the `{env_name}_` prefix
    /// for host-env MCPs (so the wire call is untainted by the model-
    /// facing prefix that disambiguates across multiple host envs).
    Mcp {
        session: Arc<McpSession>,
        host: String,
        real_name: String,
    },
}

/// An MCP host entry the scheduler is routing a specific thread's tool
/// call through, paired with the prefix the model sees on every tool
/// this MCP advertises. `prefix: Some(name)` means the model sees
/// `{name}_{tool}` and the MCP receives the bare `tool`; `prefix: None`
/// is pass-through (shared MCPs and — by convention — the reserved
/// Inline host-env path).
struct BoundMcp<'a> {
    entry: &'a crate::pod::resources::McpHostEntry,
    /// Prefix applied to tool names for host-env MCPs. `None` for
    /// shared MCPs (tools unprefixed on the wire).
    prefix: Option<&'a str>,
    /// Human-meaningful source name — the host-env binding name for
    /// host-env MCPs, the shared catalog host name for shared MCPs.
    /// Used for listing categorization; distinct from `entry.id` which
    /// is a registry-internal opaque handle.
    source_name: String,
}

/// Snapshot of a pod's on-disk directory + parsed config + behavior ids.
/// Cloned out of the scheduler when dispatching a builtin tool future so
/// the future doesn't need a back-borrow of scheduler state.
pub struct PodSnapshot {
    pub pod_dir: std::path::PathBuf,
    pub config: whisper_agent_protocol::PodConfig,
    /// Ids of the behaviors currently registered in-memory. Used by the
    /// builtin file tools to build the dynamic allowlist (each id
    /// contributes `behaviors/<id>/behavior.toml` and `prompt.md`) and
    /// to decide whether a write targets a create vs an update.
    pub behavior_ids: Vec<String>,
}

/// Messages accepted by the scheduler inbox.
// `ClientMessage` carries a `ClientToServer` (the protocol's largest variant
// is `CreateThread`, ~300 bytes). The other variants are tens of bytes. Boxing
// would touch every send-site for a per-message saving that's a rounding
// error against typical payloads. See `whisper-agent-protocol::ClientToServer`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum SchedulerMsg {
    ClientMessage {
        conn_id: ConnId,
        msg: ClientToServer,
    },
    RegisterClient {
        conn_id: ConnId,
        outbound: mpsc::UnboundedSender<ServerToClient>,
        /// True when the connection authed with an admin token (or is
        /// loopback). Promoted connections pass privileged handlers
        /// like `UpdateCodexAuth`; non-admin connections receive
        /// `Error` responses for those messages.
        is_admin: bool,
    },
    UnregisterClient {
        conn_id: ConnId,
    },
    /// An external trigger fired (webhook today; future event sources
    /// route here too). The scheduler validates the target is an
    /// actually-webhook-triggered behavior, then dispatches via the
    /// shared `fire_trigger` path. `reply` receives the validation
    /// outcome so the HTTP handler can map it to the right status.
    TriggerFired {
        pod_id: String,
        behavior_id: String,
        payload: serde_json::Value,
        reply: tokio::sync::oneshot::Sender<Result<(), TriggerFireError>>,
    },
    /// OAuth redirect landed on the server's `/oauth/callback` route.
    /// The axum handler extracts `code` + `state` from the query and
    /// forwards here; the scheduler looks up the matching pending
    /// flow, dispatches the token-exchange future, and eventually
    /// replies to the originating webui connection with
    /// `SharedMcpHostAdded` or `Error`. `error` is populated when
    /// the AS redirected with `?error=...` instead of `?code=...`
    /// (e.g. the user declined consent).
    OauthCallback {
        state: String,
        code: Option<String>,
        error: Option<String>,
    },
}

/// Per-fire validation result surfaced back to whatever transport
/// delivered the trigger (today: the `/triggers/...` HTTP handler).
/// Successful fires — whether they spawn a thread or queue a payload
/// — return `Ok(())`; this enum enumerates only the reject cases.
#[derive(Debug, Clone)]
pub enum TriggerFireError {
    UnknownPod,
    UnknownBehavior,
    /// Behavior exists but isn't webhook-triggered. Holds the actual
    /// trigger kind ("manual" / "cron") for a descriptive error.
    NotWebhookTrigger(&'static str),
    /// Behavior is present but its `behavior.toml` failed to parse.
    BehaviorLoadError(String),
    /// Behavior or its pod is paused. Webhook handler maps this to a
    /// 503 so the caller knows to retry after resume.
    Paused,
}

impl std::fmt::Display for TriggerFireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPod => write!(f, "unknown pod"),
            Self::UnknownBehavior => write!(f, "unknown behavior"),
            Self::NotWebhookTrigger(kind) => {
                write!(f, "behavior is {kind}-triggered, not webhook")
            }
            Self::BehaviorLoadError(msg) => write!(f, "behavior load error: {msg}"),
            Self::Paused => write!(f, "behavior or pod is paused"),
        }
    }
}

/// Entry in the scheduler's backend catalog.
pub struct BackendEntry {
    pub provider: Arc<dyn ModelProvider>,
    pub kind: String,
    /// Preferred model id when the task doesn't pick one. None for single-model
    /// endpoints (most local llama.cpp setups) — the scheduler passes an empty
    /// model through in that case, which those endpoints ignore.
    pub default_model: Option<String>,
    /// Auth mode tag this backend was built with (`"api_key"`,
    /// `"chatgpt_subscription"`, `"google_oauth"`), or None when the
    /// backend skips auth entirely. Retained after construction so the
    /// settings panel can describe the credential slot without holding
    /// the credential itself.
    pub auth_mode: Option<String>,
    /// Source `BackendConfig` this entry was built from — kept so
    /// runtime config-edit handlers can diff old vs new without
    /// re-parsing the on-disk TOML and so they can decide whether to
    /// rebuild the provider.
    pub source: crate::pod::config::BackendConfig,
}

/// Entry in the scheduler's embedding-provider catalog. Mirrors
/// [`BackendEntry`] minus the `default_model` slot — embedding providers
/// like TEI serve a single model per process, so there's no per-call
/// model choice to default. When OpenAI-shaped providers land they'll
/// expose model selection via the request itself.
pub struct EmbeddingProviderEntry {
    pub provider: Arc<dyn EmbeddingProvider>,
    pub kind: String,
    pub auth_mode: Option<String>,
    pub source: crate::pod::config::EmbeddingProviderConfig,
}

/// Entry in the scheduler's reranker catalog. Same shape as
/// [`EmbeddingProviderEntry`].
pub struct RerankProviderEntry {
    pub provider: Arc<dyn RerankProvider>,
    pub kind: String,
    pub auth_mode: Option<String>,
    pub source: crate::pod::config::RerankProviderConfig,
}

/// CLI-provided shared MCP host (`--shared-mcp-host name=url`). These
/// are runtime-overlays: they don't land in the durable catalog, and
/// they can't be edited through the WebUI (the catalog entry would
/// shadow them after the next add). Always anonymous — the catalog is
/// the place to attach a bearer token.
#[derive(Debug, Clone)]
pub struct SharedHostOverlay {
    pub name: String,
    pub url: String,
}

/// In-flight OAuth 2.1 flow for a pending `AddSharedMcpHost`. Keyed
/// in `Scheduler.pending_oauth_flows` by the random `state` parameter
/// we embed in the authorization URL. `state` is what the AS echoes
/// back to our `/oauth/callback`, letting us match a redirect to the
/// webui connection that started the flow.
///
/// The struct holds everything the post-authorization phase (code →
/// token exchange) needs, because by the time the callback fires,
/// the webui connection may have gone away and we can't re-derive
/// endpoints from it.
#[derive(Debug, Clone)]
pub(crate) struct PendingOauthFlow {
    pub(crate) name: String,
    pub(crate) url: String,
    pub(crate) issuer: String,
    pub(crate) token_endpoint: String,
    pub(crate) registration_endpoint: Option<String>,
    pub(crate) client_id: String,
    pub(crate) client_secret: Option<String>,
    pub(crate) resource: String,
    pub(crate) scope: Option<String>,
    pub(crate) pkce_verifier: String,
    pub(crate) redirect_uri: String,
    /// Connection that initiated the flow. Used to route the final
    /// SharedMcpHostAdded / Error reply. If the connection has gone
    /// away by the time the callback fires, the reply is dropped
    /// silently — the catalog entry still lands.
    pub(crate) conn_id: ConnId,
    pub(crate) correlation_id: Option<String>,
    /// Wall-clock deadline. Flows older than this are swept by
    /// `sweep_expired_oauth_flows` on the GC tick so a user who
    /// opened the authorization window and walked away doesn't leave
    /// an indefinite memory residue.
    pub(crate) expires_at: chrono::DateTime<chrono::Utc>,
}

/// How long a Ready resource can sit with zero users before the GC sweep
/// tears it down. 5 minutes is comfortably longer than a typical
/// "thread finishes a turn, user thinks, sends another message" gap, so
/// active conversations don't reprovision needlessly, but short enough
/// that abandoned threads don't keep their sandbox running forever.
const IDLE_RESOURCE_SECS: i64 = 300;
/// How long a TornDown / Errored resource entry lingers in the registry
/// before GC evicts it from the map. Kept for inspection ("yes, that
/// sandbox existed and was used by task X") for an hour after teardown.
const TERMINAL_RETENTION_SECS: i64 = 3600;
/// How often the GC sweep runs. Coarse — the thresholds above are minutes
/// or hours, so a 60s tick is plenty fine-grained.
const GC_TICK_SECS: u64 = 60;
/// How often cron-triggered behaviors are evaluated for fire windows.
/// 30 s means a minute-granularity cron ("0 9 * * *") fires within 30 s
/// of its scheduled time — acceptable slop for the human-scale
/// schedules behaviors encode. Tighter cadences (seconds, sub-minute)
/// aren't supported by the 5-field crontab we parse.
const CRON_TICK_SECS: u64 = 30;
/// How often retention sweeps run. Day-granularity policies don't
/// need sub-hour precision; an hourly tick means a thread past its
/// retention window is swept within 60 min. Missed tick behavior
/// (Delay) prevents bursts when the process resumes after suspension.
const RETENTION_TICK_SECS: u64 = 3600;
/// How often host-env provider reachability is probed. 30 s means a
/// daemon outage is visible in the WebUI within 30 s of the daemon
/// going down. Also bounds how stale the reachability indicator can
/// be when nothing else is exercising the provider.
const HOST_ENV_PROBE_TICK_SECS: u64 = 30;
/// How often the OAuth refresh sweep runs. 60s is short enough that
/// the 5-minute safety margin (`OAUTH_REFRESH_MARGIN_SECS`) covers
/// ~5 sweep attempts per token lifetime — plenty of headroom for a
/// transient token-endpoint blip to recover before expiry actually
/// hits. Longer would mean fewer retries; shorter would spend
/// scheduler cycles scanning a cold catalog.
const OAUTH_REFRESH_TICK_SECS: u64 = 60;

/// Interim event a dispatched I/O task pushes back to the scheduler for
/// immediate fan-out to subscribers. Distinct from the per-task I/O
/// *completion* (which flows via `FuturesUnordered`): an `StreamUpdate`
/// carries a pre-shaped wire event the scheduler just forwards to
/// subscribers of `thread_id`. Model-call streaming deltas ride this
/// channel today; future MCP tool-output streaming plugs into the same
/// path (the tool-call future clones the sender on dispatch).
pub struct StreamUpdate {
    pub thread_id: String,
    pub event: ServerToClient,
}

pub struct Scheduler {
    /// MCP host URL passed through to threads when their pod doesn't carry a
    /// Pod the scheduler picks when `CreateThread` doesn't name one. Always
    /// a key in `pods`; the server bootstrap synthesizes it on first start.
    default_pod_id: PodId,
    /// All pods the scheduler knows about. Loaded from `<pods_root>/`
    /// at startup; mutated when threads are created/archived/etc.
    pods: HashMap<PodId, Pod>,
    /// Named backends: `ThreadConfig.backend` resolves against this map.
    backends: HashMap<String, BackendEntry>,
    /// Named embedding providers — populated from `[embedding_providers.*]`.
    /// Hot-swappable through the same config-edit path as `backends`. Empty
    /// is valid: a server with no embedding providers can still drive chat
    /// threads, it just can't host knowledge buckets.
    embedding_providers: HashMap<String, EmbeddingProviderEntry>,
    /// Named rerank providers — populated from `[rerank_providers.*]`.
    rerank_providers: HashMap<String, RerankProviderEntry>,
    /// Server-scoped knowledge buckets discovered under `<buckets_root>/`
    /// at startup. Empty / `root: None` when the server boots without a
    /// buckets root configured — the rest of the scheduler degrades
    /// gracefully (no bucket-targeted operations succeed, but chat
    /// threads and pod CRUD are unaffected).
    bucket_registry: BucketRegistry,
    /// Path to the `whisper-agent.toml` the server was loaded from.
    /// `None` when started with the env-key fallback (no `--config`).
    /// Runtime config-edit handlers refuse to write back without it.
    server_config_path: Option<PathBuf>,
    persister: Option<Persister>,
    /// Live catalog of host-env providers, keyed by name. Empty when
    /// the server was started without any catalog entries or TOML-seed
    /// `[[host_env_providers]]` — threads in such a server can still
    /// run, just without any host-env MCP (so their tool catalog is
    /// shared-only). Mutated at runtime via `AddHostEnvProvider` /
    /// `UpdateHostEnvProvider` / `RemoveHostEnvProvider` commands.
    host_env_registry: crate::tools::sandbox::HostEnvRegistry,
    /// Durable backing store mirroring the registry minus any CLI-only
    /// runtime-overlay providers. Add/update/remove commands write
    /// through here so the catalog survives restart.
    host_env_catalog: crate::tools::host_env_catalog::CatalogStore,
    /// Durable catalog of shared MCP hosts. The scheduler is the
    /// authority; `ResourceRegistry.mcp_hosts` holds the live
    /// `McpSession` per name, which we keep in sync on every
    /// Add/Update/Remove. CLI overlays don't write here.
    shared_mcp_catalog: crate::tools::shared_mcp_catalog::CatalogStore,
    /// CLI `--shared-mcp-host` runtime-overlay names. Used for origin
    /// classification on list responses — an entry in this set that
    /// isn't in the catalog reports as `RuntimeOverlay`. Add/Update
    /// commands refuse overlay names (restart without the flag to
    /// unshadow the catalog entry).
    shared_mcp_overlay_names: std::collections::BTreeSet<String>,
    /// In-flight OAuth 2.1 flows for `AddSharedMcpHost { auth: Oauth2Start }`,
    /// keyed by the `state` parameter we embed in the authorization
    /// URL. Populated when the pre-authorization phase (discovery +
    /// DCR + URL build) completes; drained when the `/oauth/callback`
    /// redirect arrives and dispatches the post-authorization phase
    /// (code exchange + MCP connect). Swept for expiry on the GC
    /// tick so walked-away users don't leak.
    pending_oauth_flows: HashMap<String, PendingOauthFlow>,
    /// Catalog entry names whose OAuth refresh is currently
    /// dispatched but hasn't completed yet. Prevents the refresh
    /// ticker from enqueueing a second refresh for the same entry
    /// while the first is in flight.
    oauth_refresh_in_flight: HashSet<String>,
    /// Catalog entry names we've already logged a "no refresh_token"
    /// warning for. Once-per-entry log to avoid spamming the
    /// journal on every refresh tick for non-refreshable entries.
    oauth_refresh_no_refresh_token_warned: HashSet<String>,
    /// Consecutive refresh failure count per catalog entry name.
    /// Reset to 0 on any successful refresh. When it reaches
    /// `OAUTH_REFRESH_FAIL_THRESHOLD`, the MCP host entry is marked
    /// Errored so the operator sees the real state in the webui
    /// instead of `connected: true` backed by an about-to-expire
    /// token. Not persisted — a server restart reloads the catalog,
    /// reconnects via `connect_shared_mcp_on_boot`, and either the
    /// next refresh works or the boot reconnect fails with its own
    /// clear error.
    oauth_refresh_failure_counts: HashMap<String, u32>,

    tasks: HashMap<String, Thread>,
    /// Per-thread cancel signal. Cloned into every dispatched I/O
    /// future; firing it aborts the in-flight HTTP request (model
    /// call or MCP invoke) instead of merely discarding the result
    /// on return. Kept on the scheduler rather than on `Thread` so
    /// it doesn't interfere with `Thread`'s `Clone + Serialize`
    /// shape. Lifecycle matches `tasks`: insert on task-insert,
    /// remove on task-remove. A Cancelled thread's token stays
    /// fired for the thread's remaining lifetime.
    cancel_tokens: HashMap<String, tokio_util::sync::CancellationToken>,
    /// Owns the connection registry, subscription map, audit log, and the
    /// `ThreadEvent → ServerToClient` translation layer.
    router: ThreadEventRouter,

    /// The registry is the single source of truth for MCP sessions,
    /// tool descriptors, and host-env handles.
    resources: ResourceRegistry,

    next_op_id: OpId,
    /// Tasks modified during the current scheduler-loop iteration. Flushed to the
    /// persister before the next iteration starts.
    dirty: HashSet<String>,
    /// Behavior ids whose `state.json` needs writeback. Keyed by
    /// `(pod_id, behavior_id)`. Serialization through this set (rather
    /// than tokio::spawn per update) means two rapid state changes
    /// produce exactly one disk write per iteration.
    dirty_behaviors: HashSet<(String, String)>,
    /// `HostEnvId`s whose provisioning future is currently in flight.
    /// Keyed by the deduped host-env id (not thread) so threads that
    /// share a binding (same provider + spec) ride on a single
    /// provision attempt. A thread with multiple host-env bindings
    /// contributes one entry per binding; each entry clears
    /// independently when its provision completes.
    provisioning_in_flight: HashSet<HostEnvId>,
    /// Sender half of the stream-update channel; cloned and handed to each
    /// dispatched I/O future that wants to push interim events (streaming
    /// model deltas today, MCP tool-output chunks tomorrow). The receiver
    /// lives on the scheduler's run loop.
    stream_tx: mpsc::UnboundedSender<StreamUpdate>,

    /// In-flight bucket builds, keyed by bucket id. The token is the
    /// build task's cancel handle; firing it from the
    /// `CancelBucketBuild` handler stops the build at the next
    /// chunk-batch boundary or HNSW build entry. The entry is removed
    /// when the build task terminates (success / error / cancel) and
    /// emits its `BucketBuildEnded`. Capped to one in-flight build
    /// per bucket; concurrent `StartBucketBuild` for the same id is
    /// rejected.
    active_bucket_builds: HashMap<String, tokio_util::sync::CancellationToken>,
    /// Live progress state for every in-flight bucket build, keyed by
    /// bucket id. Mirrors `active_bucket_builds`'s lifetime: inserted
    /// in `handle_start_bucket_build`, removed in
    /// `apply_bucket_task_update`'s `BuildEnded` arm. The build task
    /// shares this `Arc` and ticks the atomics; the scheduler reads it
    /// when a fresh client connects (so they see ongoing progress
    /// instead of a stale `Building` row with zero counters).
    active_bucket_progress:
        HashMap<String, std::sync::Arc<crate::runtime::scheduler::buckets::ProgressShared>>,
    /// Cancellation tokens for the per-tracked-bucket
    /// [`FeedWorker`](crate::knowledge::FeedWorker) tasks spawned at
    /// scheduler-construction time. Keyed by bucket id; entries
    /// present only for buckets whose source is
    /// [`SourceConfig::Tracked`]. `DeleteBucket` pops the matching
    /// entry and fires `cancel()` to wind the worker down.
    active_feed_workers: HashMap<String, tokio_util::sync::CancellationToken>,
    /// Sender for `BucketTaskUpdate`s pushed from detached build
    /// tasks back to the scheduler loop. The matching receiver is
    /// returned from `Scheduler::new` and drained by the run loop.
    bucket_task_tx: mpsc::UnboundedSender<buckets::BucketTaskUpdate>,

    /// In-flight Function registry. Populated synchronously by
    /// `register_function`; each entry is removed when its Function
    /// terminates (success / error / cancel). Non-persistent.
    ///
    /// Phase 2 wires only `Function::CancelThread` through this registry;
    /// later commits migrate the rest.
    active_functions: HashMap<crate::functions::FunctionId, functions::ActiveFunctionEntry>,
    /// Monotonic counter for assigning `FunctionId`s at registration time.
    next_function_id: crate::functions::FunctionId,
    /// Connection ids that authed with an admin token (or came in over
    /// loopback). Consulted by settings-mutation handlers to decide
    /// whether to honour or reject the request. Populated on
    /// `RegisterClient`, cleared on `UnregisterClient`.
    admin_connections: HashSet<ConnId>,
}

impl Scheduler {
    /// Async because we eagerly connect to every configured shared MCP host at
    /// startup — surfacing misconfiguration at server boot rather than at first
    /// task creation. Returns Err if any shared host fails to handshake.
    ///
    /// `default_pod` becomes the in-memory entry the scheduler routes
    /// no-pod-specified `CreateThread` requests to. The caller is responsible
    /// for materializing it on disk (the server does this via
    /// `Persister::ensure_pod_toml`); the scheduler trusts what it's handed.
    // Grouping these into a `SchedulerInit` struct doesn't buy clarity —
    // every field is independent and the call sites read fine positionally.
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        default_pod: Pod,
        host_id: String,
        backends: HashMap<String, BackendEntry>,
        embedding_providers: HashMap<String, EmbeddingProviderEntry>,
        rerank_providers: HashMap<String, RerankProviderEntry>,
        bucket_registry: BucketRegistry,
        audit: AuditLog,
        host_env_registry: crate::tools::sandbox::HostEnvRegistry,
        host_env_catalog: crate::tools::host_env_catalog::CatalogStore,
        shared_mcp_catalog: crate::tools::shared_mcp_catalog::CatalogStore,
        shared_mcp_overlays: Vec<SharedHostOverlay>,
        server_config_path: Option<PathBuf>,
    ) -> anyhow::Result<(
        Self,
        mpsc::UnboundedReceiver<StreamUpdate>,
        mpsc::UnboundedReceiver<buckets::BucketTaskUpdate>,
    )> {
        let mut resources = ResourceRegistry::new();
        for (name, entry) in &backends {
            resources.insert_backend(
                name.clone(),
                entry.kind.clone(),
                entry.default_model.clone(),
            );
        }

        // Connect shared MCP hosts: catalog entries first (authoritative
        // list, may carry bearer tokens), then CLI overlays (anonymous
        // only; catalog name wins on collision). Connect failures are
        // logged and the entry is dropped from the registry — a
        // misconfigured shared host doesn't brick server startup, and
        // the operator can fix it via the WebUI (re-add triggers
        // another connect attempt).
        for entry in shared_mcp_catalog.entries() {
            connect_shared_mcp_on_boot(
                &mut resources,
                &entry.name,
                &entry.url,
                entry.auth.bearer().map(str::to_string),
                "catalog",
            )
            .await;
        }
        // `overlay_names` tracks CLI entries only; used later for
        // origin classification on list responses.
        let mut overlay_names: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        for overlay in shared_mcp_overlays {
            if shared_mcp_catalog.contains(&overlay.name) {
                info!(
                    name = %overlay.name,
                    "CLI --shared-mcp-host flag shadowed by catalog entry; using catalog"
                );
                continue;
            }
            overlay_names.insert(overlay.name.clone());
            connect_shared_mcp_on_boot(
                &mut resources,
                &overlay.name,
                &overlay.url,
                None,
                "cli-overlay",
            )
            .await;
        }

        let default_pod_id = default_pod.id.clone();
        let mut pods = HashMap::new();
        pods.insert(default_pod_id.clone(), default_pod);

        let (stream_tx, stream_rx) = mpsc::unbounded_channel();
        let (bucket_task_tx, bucket_task_rx) = mpsc::unbounded_channel();
        // Spawn one feed worker per tracked-source bucket. Workers
        // tick on cadence; their cancellation tokens are stored on
        // the scheduler so `DeleteBucket` can stop them. Skipped for
        // stored / linked / managed buckets.
        let active_feed_workers =
            feed_workers::spawn_for_registry(&bucket_registry, &embedding_providers);
        Ok((
            Self {
                default_pod_id,
                pods,
                backends,
                embedding_providers,
                rerank_providers,
                bucket_registry,
                server_config_path,
                persister: None,
                host_env_registry,
                host_env_catalog,
                shared_mcp_catalog,
                shared_mcp_overlay_names: overlay_names,
                pending_oauth_flows: HashMap::new(),
                oauth_refresh_in_flight: HashSet::new(),
                oauth_refresh_no_refresh_token_warned: HashSet::new(),
                oauth_refresh_failure_counts: HashMap::new(),
                tasks: HashMap::new(),
                cancel_tokens: HashMap::new(),
                router: ThreadEventRouter::new(audit, host_id),
                resources,
                next_op_id: 1,
                dirty: HashSet::new(),
                dirty_behaviors: HashSet::new(),
                provisioning_in_flight: HashSet::new(),
                stream_tx,
                active_bucket_builds: HashMap::new(),
                active_bucket_progress: HashMap::new(),
                active_feed_workers,
                bucket_task_tx,
                active_functions: HashMap::new(),
                next_function_id: 1,
                admin_connections: HashSet::new(),
            },
            stream_rx,
            bucket_task_rx,
        ))
    }

    /// Clone of the stream-update sender. Dispatched I/O futures grab one
    /// of these at dispatch time and push interim events through it; the
    /// scheduler's `run` loop forwards anything the receiver gets to the
    /// router.
    pub(crate) fn stream_sender(&self) -> mpsc::UnboundedSender<StreamUpdate> {
        self.stream_tx.clone()
    }

    /// Clone of the bucket-task sender. Detached build tasks grab one
    /// at spawn time and push terminal/Broadcast updates back to the
    /// scheduler loop, which applies registry mutations under
    /// `&mut self`.
    pub(crate) fn bucket_task_sender(&self) -> mpsc::UnboundedSender<buckets::BucketTaskUpdate> {
        self.bucket_task_tx.clone()
    }

    /// Read-only access to the Phase 1a shadow registry. Currently only used by
    /// tests and (in Phase 1b) the wire layer.
    pub fn resources(&self) -> &ResourceRegistry {
        &self.resources
    }

    /// Convenience for the pod-CRUD handlers (Phase 2d.i): grabs a clone of
    /// the persister and the requester's outbound channel together. Returns
    /// `None` after sending an error to the requester when either piece is
    /// missing (no persister configured, or the connection has dropped).
    fn persister_and_outbound(
        &self,
        conn_id: ConnId,
    ) -> Option<(Persister, mpsc::UnboundedSender<ServerToClient>)> {
        let outbound = self.router.outbound(conn_id)?;
        let Some(persister) = self.persister.clone() else {
            let _ = outbound.send(ServerToClient::Error {
                correlation_id: None,
                thread_id: None,
                message: "server has no persister configured".into(),
            });
            return None;
        };
        Some((persister, outbound))
    }

    // ---------- Resource event emission (Phase 1b) ----------
    //
    // Each mutation of the registry is followed by one of these so connected
    // clients see the change without polling. Created fires once per entry's
    // appearance; Updated fires on every subsequent field change. Entries
    // populated at startup (backends, shared MCP hosts) don't get a Created
    // event — there are no clients connected yet, and a fresh client gets the
    // full set via `ListResources`.

    fn emit_host_env_updated(&self, id: &HostEnvId) {
        if let Some(snap) = self.resources.snapshot_host_env(id) {
            self.router
                .broadcast_resource(ServerToClient::ResourceUpdated { resource: snap });
        }
    }
    fn emit_mcp_host_updated(&self, id: &McpHostId) {
        if let Some(snap) = self.resources.snapshot_mcp_host(id) {
            self.router
                .broadcast_resource(ServerToClient::ResourceUpdated { resource: snap });
        }
    }
    fn emit_backend_updated(&self, id: &BackendId) {
        if let Some(snap) = self.resources.snapshot_backend(id) {
            self.router
                .broadcast_resource(ServerToClient::ResourceUpdated { resource: snap });
        }
    }

    // ---------- Read-only accessors used by `io_dispatch` ----------

    pub(crate) fn task(&self, thread_id: &str) -> Option<&Thread> {
        self.tasks.get(thread_id)
    }

    /// Owned clone of the per-thread cancel token, or a fresh
    /// never-fired token if the thread was removed mid-dispatch.
    /// Callers in io_dispatch take one of these on entry and hand it
    /// to the provider / tool future — the user's cancel signal then
    /// reaches the inner HTTP request, not just the scheduler's
    /// result sink.
    pub(crate) fn cancel_token_or_default(
        &self,
        thread_id: &str,
    ) -> tokio_util::sync::CancellationToken {
        self.cancel_tokens
            .get(thread_id)
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn backend(&self, name: &str) -> Option<&BackendEntry> {
        self.backends.get(name)
    }

    /// Look up a host-env provider in the catalog by name.
    pub(crate) fn host_env_provider(&self, name: &str) -> Option<&Arc<dyn HostEnvProvider>> {
        self.host_env_registry.get(name)
    }

    /// Classify a registered provider's origin for protocol output.
    /// Anything in the registry but absent from the durable catalog is
    /// a `RuntimeOverlay` (CLI-flag provided); anything in the catalog
    /// reports the catalog's stored origin.
    fn host_env_provider_origin(
        &self,
        name: &str,
    ) -> whisper_agent_protocol::HostEnvProviderOrigin {
        use crate::tools::host_env_catalog::CatalogOrigin;
        use whisper_agent_protocol::HostEnvProviderOrigin;
        match self.host_env_catalog.get(name).map(|e| e.origin) {
            Some(CatalogOrigin::Seeded) => HostEnvProviderOrigin::Seeded,
            Some(CatalogOrigin::Manual) => HostEnvProviderOrigin::Manual,
            None => HostEnvProviderOrigin::RuntimeOverlay,
        }
    }

    /// Snapshot the registry for `ListHostEnvProviders`. Entries appear
    /// in name-sorted order (via `HostEnvRegistry::snapshot`).
    pub(crate) fn host_env_provider_snapshot(
        &self,
    ) -> Vec<whisper_agent_protocol::HostEnvProviderInfo> {
        self.host_env_registry
            .snapshot(|name| self.host_env_provider_origin(name))
    }

    /// Build a single `HostEnvProviderInfo` from the live registry
    /// state, with its origin classified. Returns `None` when the
    /// name isn't in the registry. Used by the Add/Update response
    /// paths so the wire record always reflects what the registry
    /// actually holds (including reachability, which is live state).
    pub(crate) fn host_env_provider_info(
        &self,
        name: &str,
    ) -> Option<whisper_agent_protocol::HostEnvProviderInfo> {
        let entry = self.host_env_registry.entry(name)?;
        Some(whisper_agent_protocol::HostEnvProviderInfo {
            name: name.to_string(),
            url: entry.url.clone(),
            origin: self.host_env_provider_origin(name),
            has_token: entry.token.is_some(),
            reachability: entry.reachability.clone(),
        })
    }

    /// Enumerate one reachability-probe future per registered
    /// provider. Each future resolves to a `SchedulerCompletion::Probe`
    /// the run loop feeds back into `apply_probe_completion`. Clones
    /// the `Arc<dyn HostEnvProvider>` so the probe future is
    /// self-contained — the registry can mutate concurrently without
    /// invalidating in-flight probes.
    pub(crate) fn spawn_reachability_probes(
        &self,
    ) -> Vec<crate::runtime::io_dispatch::SchedulerFuture> {
        use crate::runtime::io_dispatch::{ProbeCompletion, SchedulerCompletion};
        let mut futs: Vec<crate::runtime::io_dispatch::SchedulerFuture> = Vec::new();
        for name in self.host_env_registry.names() {
            let Some(entry) = self.host_env_registry.entry(&name) else {
                continue;
            };
            let provider = entry.provider.clone();
            let name_owned = name.clone();
            futs.push(Box::pin(async move {
                let result = provider.probe().await.map_err(|e| e.to_string());
                SchedulerCompletion::Probe(ProbeCompletion {
                    name: name_owned,
                    observed_at: chrono::Utc::now(),
                    result,
                })
            }));
        }
        futs
    }

    /// Update the registry entry's reachability from a probe result
    /// and emit a `HostEnvProviderUpdated` push if the state actually
    /// changed. The "changed?" gate matters — a reachable daemon
    /// would otherwise generate a push every probe tick.
    pub(crate) fn apply_probe_completion(
        &mut self,
        probe: crate::runtime::io_dispatch::ProbeCompletion,
    ) {
        let crate::runtime::io_dispatch::ProbeCompletion {
            name,
            observed_at,
            result,
        } = probe;
        // Log transitions at WARN / INFO, repeats at DEBUG. With the
        // 30s probe tick a permanently-down daemon would otherwise
        // spam WARN every tick forever; operators want the first edge
        // + the recovery, not a ticker in the journal.
        let changed = match result {
            Ok(()) => {
                let changed = self.host_env_registry.mark_reachable(&name, observed_at);
                if changed {
                    info!(provider = %name, "host-env provider probe recovered");
                } else {
                    debug!(provider = %name, "host-env provider probe ok");
                }
                changed
            }
            Err(msg) => {
                let changed =
                    self.host_env_registry
                        .mark_unreachable(&name, observed_at, msg.clone());
                if changed {
                    warn!(provider = %name, error = %msg, "host-env provider probe failed");
                } else {
                    debug!(provider = %name, error = %msg, "host-env provider probe still failing");
                }
                changed
            }
        };
        if changed && let Some(info) = self.host_env_provider_info(&name) {
            self.router.broadcast_resource(
                whisper_agent_protocol::ServerToClient::HostEnvProviderUpdated {
                    correlation_id: None,
                    provider: info,
                },
            );
        }
    }

    /// Pod IDs whose `[[allow.host_env]]` references the given
    /// provider name. Used by the remove path to refuse removal while
    /// any loaded pod binding would be orphaned. Archived pods aren't
    /// loaded into `self.pods` and therefore aren't checked — that's
    /// intentional: archived pods are already unreachable.
    fn pods_referencing_host_env_provider(&self, name: &str) -> Vec<String> {
        let mut refs: Vec<String> = self
            .pods
            .iter()
            .filter(|(_, pod)| {
                pod.config
                    .allow
                    .host_env
                    .iter()
                    .any(|entry| entry.provider == name)
            })
            .map(|(id, _)| id.to_string())
            .collect();
        refs.sort();
        refs
    }

    /// Add a provider to the durable catalog and the live registry.
    /// Returns the canonical snapshot on success so the caller can
    /// echo it back to clients.
    pub(crate) fn add_host_env_provider(
        &mut self,
        name: String,
        url: String,
        token: Option<String>,
    ) -> anyhow::Result<whisper_agent_protocol::HostEnvProviderInfo> {
        if name.is_empty() {
            anyhow::bail!("provider name must be non-empty");
        }
        if url.is_empty() {
            anyhow::bail!("provider url must be non-empty");
        }
        if self.host_env_registry.contains(&name) {
            anyhow::bail!(
                "provider `{name}` already registered (as {})",
                match self.host_env_provider_origin(&name) {
                    whisper_agent_protocol::HostEnvProviderOrigin::Seeded => "seeded",
                    whisper_agent_protocol::HostEnvProviderOrigin::Manual => "manual",
                    whisper_agent_protocol::HostEnvProviderOrigin::RuntimeOverlay =>
                        "runtime-overlay (CLI --host-env-provider flag)",
                }
            );
        }
        let now = chrono::Utc::now();
        self.host_env_catalog
            .insert(crate::tools::host_env_catalog::new_manual_entry(
                name.clone(),
                url.clone(),
                token.clone(),
                now,
            ))?;
        self.host_env_registry
            .insert_or_replace(name.clone(), url, token);
        Ok(self.host_env_provider_info(&name).expect("just inserted"))
    }

    /// Replace `url` and `token` on an existing catalog entry. Refuses
    /// runtime-overlay entries (they're not in the catalog to edit).
    pub(crate) fn update_host_env_provider(
        &mut self,
        name: String,
        url: String,
        token: Option<String>,
    ) -> anyhow::Result<whisper_agent_protocol::HostEnvProviderInfo> {
        if url.is_empty() {
            anyhow::bail!("provider url must be non-empty");
        }
        if !self.host_env_catalog.contains(&name) {
            if self.host_env_registry.contains(&name) {
                anyhow::bail!(
                    "provider `{name}` is a runtime-overlay from a CLI flag and can't be updated at runtime; remove the --host-env-provider flag and restart to manage it here"
                );
            }
            anyhow::bail!("provider `{name}` not in catalog");
        }
        let now = chrono::Utc::now();
        self.host_env_catalog
            .update(&name, url.clone(), token.clone(), now)?;
        self.host_env_registry
            .insert_or_replace(name.clone(), url, token);
        Ok(self.host_env_provider_info(&name).expect("just updated"))
    }

    /// Remove a provider from the catalog and live registry. Refuses
    /// with a listing of referencing pod IDs if any loaded pod's
    /// `[[allow.host_env]]` still names this provider. Also refuses
    /// runtime-overlay entries (not in the catalog). Existing live
    /// host-env handles already provisioned against the provider keep
    /// working until GC'd — the handle cached the URL and http client.
    pub(crate) fn remove_host_env_provider(&mut self, name: &str) -> anyhow::Result<()> {
        if !self.host_env_catalog.contains(name) {
            if self.host_env_registry.contains(name) {
                anyhow::bail!(
                    "provider `{name}` is a runtime-overlay from a CLI flag; remove the flag and restart to unregister"
                );
            }
            anyhow::bail!("provider `{name}` not in catalog");
        }
        let refs = self.pods_referencing_host_env_provider(name);
        if !refs.is_empty() {
            anyhow::bail!(
                "provider `{name}` is referenced by pod `[[allow.host_env]]` in: [{}]. Edit those pods first",
                refs.join(", ")
            );
        }
        self.host_env_catalog.remove(name)?;
        self.host_env_registry.remove(name);
        Ok(())
    }

    // ---------- Shared-MCP-host catalog surface ----------

    /// Classify a shared-MCP-host's origin for protocol output.
    /// Same shape as `host_env_provider_origin`: catalog wins over
    /// CLI overlay, unknown names report as `RuntimeOverlay` (they
    /// exist in the registry but nowhere durable).
    fn shared_mcp_host_origin(&self, name: &str) -> whisper_agent_protocol::HostEnvProviderOrigin {
        use crate::tools::host_env_catalog::CatalogOrigin;
        use whisper_agent_protocol::HostEnvProviderOrigin;
        match self.shared_mcp_catalog.get(name).map(|e| e.origin) {
            Some(CatalogOrigin::Seeded) => HostEnvProviderOrigin::Seeded,
            Some(CatalogOrigin::Manual) => HostEnvProviderOrigin::Manual,
            None => HostEnvProviderOrigin::RuntimeOverlay,
        }
    }

    /// Build a `SharedMcpHostInfo` for one name, combining catalog
    /// state (origin + auth classification) with live-registry state
    /// (connected + last_error). Returns `None` when the name is in
    /// neither source — caller shouldn't be asking about unknown
    /// names.
    fn shared_mcp_host_info(
        &self,
        name: &str,
    ) -> Option<whisper_agent_protocol::SharedMcpHostInfo> {
        use crate::pod::resources::{McpHostId, ResourceState};
        let catalog_entry = self.shared_mcp_catalog.get(name);
        let is_overlay = self.shared_mcp_overlay_names.contains(name);
        if catalog_entry.is_none() && !is_overlay {
            return None;
        }
        let auth = catalog_entry
            .map(|e| e.auth.public())
            .unwrap_or(whisper_agent_protocol::SharedMcpAuthPublic::None);
        let url = catalog_entry
            .map(|e| e.url.clone())
            .or_else(|| {
                // Runtime overlays stash their URL in the registry's
                // McpHostEntry.spec.url; this is the one source of
                // truth for an overlay's URL.
                self.resources
                    .mcp_hosts
                    .get(&McpHostId::shared(name))
                    .map(|e| e.spec.url.clone())
            })
            .unwrap_or_default();
        let (connected, last_error) = match self.resources.mcp_hosts.get(&McpHostId::shared(name)) {
            Some(entry) => match &entry.state {
                ResourceState::Ready => (true, String::new()),
                ResourceState::Errored { message } | ResourceState::Lost { message } => {
                    (false, message.clone())
                }
                _ => (false, String::new()),
            },
            None => (false, String::new()),
        };
        Some(whisper_agent_protocol::SharedMcpHostInfo {
            name: name.to_string(),
            url,
            origin: self.shared_mcp_host_origin(name),
            auth,
            connected,
            last_error,
        })
    }

    /// Snapshot the catalog + overlays for `ListSharedMcpHosts`.
    /// Name-sorted. Merges catalog entries and CLI-overlay names so
    /// a CLI-overlay-only host still appears.
    pub(crate) fn shared_mcp_hosts_snapshot(
        &self,
    ) -> Vec<whisper_agent_protocol::SharedMcpHostInfo> {
        let mut names: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for e in self.shared_mcp_catalog.entries() {
            names.insert(e.name.as_str());
        }
        for n in &self.shared_mcp_overlay_names {
            names.insert(n.as_str());
        }
        names
            .into_iter()
            .filter_map(|n| self.shared_mcp_host_info(n))
            .collect()
    }

    /// Pre-flight validation for Add. Runs on the scheduler loop
    /// before any network I/O; failures surface immediately as an
    /// `Error` response. Returns the validated catalog-auth shape
    /// the connect future will use.
    pub(crate) fn validate_shared_mcp_add(&self, name: &str, url: &str) -> anyhow::Result<()> {
        if name.is_empty() {
            anyhow::bail!("host name must be non-empty");
        }
        if url.is_empty() {
            anyhow::bail!("host url must be non-empty");
        }
        if self.shared_mcp_catalog.contains(name) {
            anyhow::bail!("host `{name}` already exists in catalog");
        }
        if self.shared_mcp_overlay_names.contains(name) {
            anyhow::bail!(
                "host `{name}` is a CLI --shared-mcp-host overlay; remove the flag and restart to manage it through the catalog"
            );
        }
        Ok(())
    }

    /// Pre-flight validation for Update. Resolves the auth the connect
    /// future should use — explicit `new_auth` when set, otherwise the
    /// existing catalog entry's.
    pub(crate) fn validate_shared_mcp_update(
        &self,
        name: &str,
        url: &str,
        new_auth: Option<&crate::tools::shared_mcp_catalog::SharedMcpAuth>,
    ) -> anyhow::Result<crate::tools::shared_mcp_catalog::SharedMcpAuth> {
        if url.is_empty() {
            anyhow::bail!("host url must be non-empty");
        }
        if !self.shared_mcp_catalog.contains(name) {
            if self.shared_mcp_overlay_names.contains(name) {
                anyhow::bail!(
                    "host `{name}` is a CLI --shared-mcp-host overlay; remove the flag and restart to manage it through the catalog"
                );
            }
            anyhow::bail!("host `{name}` not in catalog");
        }
        let effective = match new_auth {
            Some(a) => a.clone(),
            None => self
                .shared_mcp_catalog
                .get(name)
                .expect("checked above")
                .auth
                .clone(),
        };
        Ok(effective)
    }

    /// Apply a completed shared-MCP connect attempt (Add or Update).
    /// Runs on the scheduler loop so catalog + registry writes stay
    /// single-threaded. On success, mutates state and sends the
    /// matching `SharedMcpHost{Added,Updated}` to the originating
    /// connection. On failure, sends an `Error`.
    pub(crate) fn apply_shared_mcp_completion(
        &mut self,
        completion: crate::runtime::io_dispatch::SharedMcpCompletion,
    ) {
        use crate::runtime::io_dispatch::SharedMcpOp;
        let crate::runtime::io_dispatch::SharedMcpCompletion {
            conn_id,
            correlation_id,
            name,
            url,
            auth,
            op,
            result,
        } = completion;
        match result {
            Ok((session, tools)) => {
                // Write the catalog first, then the registry — same
                // order as the direct mutation APIs so failure mid-
                // way can't leave a live registry entry the catalog
                // doesn't reflect.
                let now = chrono::Utc::now();
                let catalog_write = match op {
                    SharedMcpOp::Add => self.shared_mcp_catalog.insert(
                        crate::tools::shared_mcp_catalog::new_manual_entry(
                            name.clone(),
                            url.clone(),
                            auth.clone(),
                            now,
                        ),
                    ),
                    SharedMcpOp::Update => {
                        self.shared_mcp_catalog
                            .update(&name, url.clone(), Some(auth.clone()), now)
                    }
                };
                if let Err(e) = catalog_write {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("shared_mcp_host catalog write: {e}"),
                        },
                    );
                    return;
                }
                let id = self
                    .resources
                    .insert_shared_mcp_host(name.clone(), url, session);
                let annotations: HashMap<String, ToolAnnotations> = tools
                    .iter()
                    .map(|t| (t.name.clone(), t.annotations.clone()))
                    .collect();
                self.resources.populate_mcp_tools(&id, tools, annotations);
                let host = self.shared_mcp_host_info(&name).expect("just applied");
                let reply = match op {
                    SharedMcpOp::Add => ServerToClient::SharedMcpHostAdded {
                        correlation_id,
                        host,
                    },
                    SharedMcpOp::Update => ServerToClient::SharedMcpHostUpdated {
                        correlation_id,
                        host,
                    },
                };
                self.router.send_to_client(conn_id, reply);
            }
            Err(message) => {
                let verb = match op {
                    SharedMcpOp::Add => "add_shared_mcp_host",
                    SharedMcpOp::Update => "update_shared_mcp_host",
                };
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::Error {
                        correlation_id,
                        thread_id: None,
                        message: format!("{verb}: {message}"),
                    },
                );
            }
        }
    }

    /// Remove a shared MCP host from the catalog and registry.
    /// Refuses when any live thread holds the host in its `users` set
    /// — the caller should retarget / finish those threads first.
    /// Refuses overlay-only entries (CLI-only) too.
    pub(crate) fn remove_shared_mcp_host(&mut self, name: &str) -> anyhow::Result<()> {
        use crate::pod::resources::McpHostId;
        if !self.shared_mcp_catalog.contains(name) {
            if self.shared_mcp_overlay_names.contains(name) {
                anyhow::bail!(
                    "host `{name}` is a CLI --shared-mcp-host overlay; remove the flag and restart to unregister"
                );
            }
            anyhow::bail!("host `{name}` not in catalog");
        }
        let id = McpHostId::shared(name);
        if let Some(entry) = self.resources.mcp_hosts.get(&id)
            && !entry.users.is_empty()
        {
            let mut users: Vec<&str> = entry.users.iter().map(String::as_str).collect();
            users.sort();
            anyhow::bail!(
                "host `{name}` is currently in use by threads: [{}]. End or retarget those threads first",
                users.join(", ")
            );
        }
        self.shared_mcp_catalog.remove(name)?;
        self.resources.mcp_hosts.remove(&id);
        Ok(())
    }

    // ---------- OAuth 2.1 flow for third-party shared MCP hosts ----------

    /// How long a pending OAuth flow can sit unaclaimed before the
    /// GC sweep evicts it. 10 min is long enough for a user to open
    /// the authorization URL, read the consent screen, and click
    /// Allow; short enough that a walked-away user doesn't leak.
    const OAUTH_FLOW_TTL_SECS: i64 = 600;

    /// Kick off the pre-authorization phase of an
    /// `AddSharedMcpHost { auth: Oauth2Start }`. The preconditions
    /// (admin gate, name/url validity, no-collision) have already
    /// been checked by the client-message handler; this method just
    /// dispatches the discovery+DCR+authz-URL future.
    // Wire-shape args (conn + correlation + name + url + scope +
    // redirect + pending_io) all land here from the client-message
    // handler; grouping into a struct is churn without clarity.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_shared_mcp_oauth(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        name: String,
        url: String,
        scope: Option<String>,
        redirect_base: String,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        pending_io.push(crate::runtime::io_dispatch::build_oauth_start_future(
            conn_id,
            correlation_id,
            name,
            url,
            scope,
            redirect_base,
        ));
    }

    /// Handle the pre-auth phase completion: either stash the
    /// pending flow + tell the webui to open the authorization URL,
    /// or surface the discovery/DCR failure.
    pub(crate) fn apply_oauth_start_completion(
        &mut self,
        completion: crate::runtime::io_dispatch::OauthStartCompletion,
    ) {
        let crate::runtime::io_dispatch::OauthStartCompletion {
            conn_id,
            correlation_id,
            name,
            url,
            redirect_uri,
            scope,
            result,
        } = completion;
        match result {
            Ok(data) => {
                let crate::runtime::io_dispatch::OauthStartData {
                    authorization_url,
                    state,
                    pkce_verifier,
                    issuer,
                    token_endpoint,
                    registration_endpoint,
                    client_id,
                    client_secret,
                    resource,
                } = data;
                let expires_at =
                    chrono::Utc::now() + chrono::Duration::seconds(Self::OAUTH_FLOW_TTL_SECS);
                self.pending_oauth_flows.insert(
                    state.clone(),
                    PendingOauthFlow {
                        name: name.clone(),
                        url,
                        issuer,
                        token_endpoint,
                        registration_endpoint,
                        client_id,
                        client_secret,
                        resource,
                        scope,
                        pkce_verifier,
                        redirect_uri,
                        conn_id,
                        correlation_id: correlation_id.clone(),
                        expires_at,
                    },
                );
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::SharedMcpOauthFlowStarted {
                        correlation_id,
                        name,
                        authorization_url,
                    },
                );
            }
            Err(message) => {
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::Error {
                        correlation_id,
                        thread_id: None,
                        message: format!("add_shared_mcp_host (oauth2): {message}"),
                    },
                );
            }
        }
    }

    /// Called from the inbox when the `/oauth/callback` route hands
    /// us a redirect. Looks up the pending flow by `state`; on miss,
    /// logs + drops (the callback page was already served). On hit,
    /// dispatches the token-exchange future. `error` is populated
    /// when the AS reported e.g. `access_denied`.
    pub(crate) fn apply_oauth_callback(
        &mut self,
        state: String,
        code: Option<String>,
        error: Option<String>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(flow) = self.pending_oauth_flows.remove(&state) else {
            warn!(%state, "oauth callback for unknown or expired state; ignoring");
            return;
        };
        if let Some(err) = error {
            self.router.send_to_client(
                flow.conn_id,
                ServerToClient::Error {
                    correlation_id: flow.correlation_id,
                    thread_id: None,
                    message: format!(
                        "add_shared_mcp_host (oauth2): authorization server returned error: {err}"
                    ),
                },
            );
            return;
        }
        let Some(code) = code else {
            self.router.send_to_client(
                flow.conn_id,
                ServerToClient::Error {
                    correlation_id: flow.correlation_id,
                    thread_id: None,
                    message: "add_shared_mcp_host (oauth2): authorization server redirected without a code".into(),
                },
            );
            return;
        };
        pending_io.push(crate::runtime::io_dispatch::build_oauth_complete_future(
            flow.conn_id,
            flow.correlation_id,
            flow.name,
            flow.url,
            flow.issuer,
            flow.token_endpoint,
            flow.registration_endpoint,
            flow.client_id,
            flow.client_secret,
            flow.resource,
            flow.scope,
            code,
            flow.pkce_verifier,
            flow.redirect_uri,
        ));
    }

    /// Apply the post-authorization phase result: write the catalog
    /// entry, insert the MCP session into the registry, send the
    /// originating connection a `SharedMcpHostAdded` (or Error on
    /// failure). Because the connection may have dropped between
    /// flow start and callback, `send_to_client` is best-effort —
    /// the catalog entry still lands.
    pub(crate) fn apply_oauth_complete_completion(
        &mut self,
        completion: crate::runtime::io_dispatch::OauthCompleteCompletion,
    ) {
        let crate::runtime::io_dispatch::OauthCompleteCompletion {
            conn_id,
            correlation_id,
            name,
            url,
            issuer,
            token_endpoint,
            registration_endpoint,
            client_id,
            client_secret,
            resource,
            scope,
            result,
        } = completion;
        match result {
            Ok(data) => {
                let crate::runtime::io_dispatch::OauthCompleteData {
                    session,
                    tools,
                    access_token,
                    refresh_token,
                    expires_at,
                    scope_granted,
                } = data;
                // Granted scope wins over requested scope — the AS
                // may have down-granted and we want to refresh with
                // exactly what we have.
                let effective_scope = scope_granted.or(scope);
                let auth = crate::tools::shared_mcp_catalog::SharedMcpAuth::Oauth2(Box::new(
                    crate::tools::shared_mcp_catalog::SharedMcpOauth2 {
                        issuer,
                        token_endpoint,
                        registration_endpoint,
                        client_id,
                        client_secret,
                        access_token,
                        refresh_token,
                        expires_at,
                        scope: effective_scope,
                        resource,
                    },
                ));
                let now = chrono::Utc::now();
                if let Err(e) = self.shared_mcp_catalog.insert(
                    crate::tools::shared_mcp_catalog::new_manual_entry(
                        name.clone(),
                        url.clone(),
                        auth,
                        now,
                    ),
                ) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!(
                                "add_shared_mcp_host (oauth2): catalog write failed: {e}"
                            ),
                        },
                    );
                    return;
                }
                let id = self
                    .resources
                    .insert_shared_mcp_host(name.clone(), url, session);
                let annotations: HashMap<String, ToolAnnotations> = tools
                    .iter()
                    .map(|t| (t.name.clone(), t.annotations.clone()))
                    .collect();
                self.resources.populate_mcp_tools(&id, tools, annotations);
                let host = self.shared_mcp_host_info(&name).expect("just inserted");
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::SharedMcpHostAdded {
                        correlation_id,
                        host,
                    },
                );
            }
            Err(message) => {
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::Error {
                        correlation_id,
                        thread_id: None,
                        message: format!("add_shared_mcp_host (oauth2): {message}"),
                    },
                );
            }
        }
    }

    /// How far before expiry the refresh sweep starts trying to
    /// rotate a token. 5 minutes is generous relative to typical
    /// tool-call latencies (seconds to tens of seconds) so in-flight
    /// calls almost never straddle a rotation boundary. Tokens with
    /// no `expires_at` are refreshed on demand (their `refresh_token`
    /// still rotates eventually if the AS revokes the access one).
    const OAUTH_REFRESH_MARGIN_SECS: i64 = 300;

    /// How many consecutive refresh failures before we flip the MCP
    /// registry entry to Errored. 3 is a middle ground — a transient
    /// token-endpoint blip (one failed tick) shouldn't redecorate
    /// the operator's UI, but a structural failure (revoked
    /// refresh_token, AS outage, DCR client wiped) should surface
    /// within ~3 minutes rather than silently rotting until the
    /// access_token expires for real.
    const OAUTH_REFRESH_FAIL_THRESHOLD: u32 = 3;

    /// Walk every catalog entry and dispatch a refresh future for
    /// each OAuth entry whose access_token is within the safety
    /// margin of expiring (or already expired). Returns a vec of
    /// `SchedulerFuture`s the caller extends into `pending_io`.
    /// `in_flight` is populated with the names we just dispatched
    /// so the next tick doesn't re-enqueue entries still waiting
    /// for their refresh to complete — the `apply_oauth_refresh_completion`
    /// handler removes the name on landing.
    pub(crate) fn spawn_oauth_refresh_futures(
        &mut self,
    ) -> Vec<crate::runtime::io_dispatch::SchedulerFuture> {
        let now = chrono::Utc::now().timestamp();
        let margin = Self::OAUTH_REFRESH_MARGIN_SECS;
        let mut out: Vec<crate::runtime::io_dispatch::SchedulerFuture> = Vec::new();
        for entry in self.shared_mcp_catalog.entries() {
            let crate::tools::shared_mcp_catalog::SharedMcpAuth::Oauth2(o) = &entry.auth else {
                continue;
            };
            if self.oauth_refresh_in_flight.contains(&entry.name) {
                continue;
            }
            let Some(refresh_token) = o.refresh_token.clone() else {
                // No refresh token → can't refresh this entry. The
                // access token will expire on its own; tool calls
                // will start 401ing; the operator has to remove +
                // re-add through the webui. Logged once per entry
                // the first time we skip it.
                if self
                    .oauth_refresh_no_refresh_token_warned
                    .insert(entry.name.clone())
                {
                    warn!(
                        host = %entry.name,
                        "oauth host has no refresh_token; will not auto-renew when access_token expires"
                    );
                }
                continue;
            };
            // Skip when expiry is far out. `expires_at = None`
            // falls through (the `<` against now + margin is false
            // when the option is None), so no-expiry entries don't
            // get preemptively refreshed. They'll still be
            // refreshed reactively when the access token starts
            // failing (followup in step 2c).
            let needs_refresh = match o.expires_at {
                Some(exp) => exp < now + margin,
                None => false,
            };
            if !needs_refresh {
                continue;
            }
            self.oauth_refresh_in_flight.insert(entry.name.clone());
            out.push(crate::runtime::io_dispatch::build_oauth_refresh_future(
                crate::runtime::io_dispatch::OauthRefreshArgs {
                    name: entry.name.clone(),
                    url: entry.url.clone(),
                    token_endpoint: o.token_endpoint.clone(),
                    client_id: o.client_id.clone(),
                    client_secret: o.client_secret.clone(),
                    refresh_token,
                    resource: o.resource.clone(),
                    scope: o.scope.clone(),
                },
            ));
        }
        out
    }

    /// Apply a completed refresh: on success, rotate tokens in the
    /// catalog and swap the session in the registry; on failure,
    /// log + leave the old session in place (it may still work
    /// until the token actually expires, and the next tick will
    /// retry). Either way, `oauth_refresh_in_flight` is updated so
    /// the entry becomes eligible for the next sweep.
    pub(crate) fn apply_oauth_refresh_completion(
        &mut self,
        completion: crate::runtime::io_dispatch::OauthRefreshCompletion,
    ) {
        let crate::runtime::io_dispatch::OauthRefreshCompletion { name, url, result } = completion;
        self.oauth_refresh_in_flight.remove(&name);
        match result {
            Ok(data) => {
                let crate::runtime::io_dispatch::OauthRefreshData {
                    session,
                    access_token,
                    refresh_token,
                    expires_at,
                    scope,
                } = data;
                let now = chrono::Utc::now();
                if let Err(e) = self.shared_mcp_catalog.rotate_oauth_tokens(
                    &name,
                    access_token,
                    refresh_token,
                    expires_at,
                    scope,
                    now,
                ) {
                    warn!(
                        host = %name,
                        error = %e,
                        "oauth refresh: catalog rotate failed; session not swapped"
                    );
                    return;
                }
                let swapped = self
                    .resources
                    .replace_shared_mcp_session(&name, url, session);
                if !swapped {
                    warn!(
                        host = %name,
                        "oauth refresh: registry entry missing; catalog rotated but session not swapped (will be picked up on next connect attempt)"
                    );
                    return;
                }
                // Successful rotation clears any accumulated failure
                // count. If we'd marked the entry Errored earlier,
                // `replace_shared_mcp_session` already flipped it
                // back to Ready.
                self.oauth_refresh_failure_counts.remove(&name);
                info!(host = %name, "oauth refresh: rotated tokens and swapped session");
            }
            Err(message) => {
                // Bounded retry policy: one or two failures don't
                // flip the entry's state (a transient token-endpoint
                // blip recovers on the next tick). After
                // `OAUTH_REFRESH_FAIL_THRESHOLD` consecutive
                // failures we mark the MCP entry Errored so the
                // operator sees the real state in the webui instead
                // of a `connected: true` lie backed by a token
                // that's about to expire.
                // Increment then drop the map borrow before touching
                // `self.resources` — `mark_mcp_errored` + `emit_...`
                // need their own &mut self and can't coexist with a
                // live `HashMap::entry` reference.
                let count = {
                    let slot = self
                        .oauth_refresh_failure_counts
                        .entry(name.clone())
                        .or_insert(0);
                    *slot += 1;
                    *slot
                };
                if count >= Self::OAUTH_REFRESH_FAIL_THRESHOLD {
                    use crate::pod::resources::McpHostId;
                    let id = McpHostId::shared(&name);
                    self.resources.mark_mcp_errored(
                        &id,
                        format!("oauth refresh failed {count} times: {message}"),
                    );
                    self.emit_mcp_host_updated(&id);
                    warn!(
                        host = %name,
                        count,
                        error = %message,
                        "oauth refresh failed repeatedly; marked MCP entry Errored"
                    );
                } else {
                    warn!(host = %name, count, error = %message, "oauth refresh failed");
                }
            }
        }
    }

    /// Evict pending OAuth flows whose deadline has passed. Called
    /// from the GC tick. Counts the eviction so operators can
    /// correlate a missing callback to a TTL expiry.
    pub(crate) fn sweep_expired_oauth_flows(&mut self) {
        let now = chrono::Utc::now();
        let before = self.pending_oauth_flows.len();
        self.pending_oauth_flows.retain(|_, f| f.expires_at > now);
        let evicted = before - self.pending_oauth_flows.len();
        if evicted > 0 {
            info!(evicted, "swept expired shared-mcp oauth flows");
        }
    }

    /// Resolve a [`HostEnvBinding`] into the (provider, spec) pair the
    /// registry needs. Inline bindings carry both directly; Named
    /// bindings walk the pod's `[[allow.host_env]]` table for a match.
    /// Returns `None` when the named entry has been removed (pod was
    /// edited) or the pod itself isn't loaded — caller decides
    /// recovery (typically: re-bind to the pod's current default).
    pub(crate) fn resolve_binding(
        &self,
        pod_id: &str,
        binding: &HostEnvBinding,
    ) -> Option<(String, HostEnvSpec)> {
        match binding {
            HostEnvBinding::Named { name } => {
                let pod = self.pods.get(pod_id)?;
                pod.config
                    .allow
                    .host_env
                    .iter()
                    .find(|nh| &nh.name == name)
                    .map(|nh| (nh.provider.clone(), nh.spec.clone()))
            }
            HostEnvBinding::Inline { provider, spec } => Some((provider.clone(), spec.clone())),
        }
    }

    /// Compute the runtime `HostEnvId`s for every binding on the
    /// thread, in the binding list's declared order. Empty when the
    /// thread has no host envs; entries whose `resolve_binding` lookup
    /// fails (named entry removed from pod allow list mid-flight) are
    /// silently skipped rather than failing the whole list.
    pub(crate) fn host_env_ids_for_thread(&self, thread_id: &str) -> Vec<HostEnvId> {
        let Some(task) = self.tasks.get(thread_id) else {
            return Vec::new();
        };
        task.bindings
            .host_env
            .iter()
            .filter_map(|b| self.resolve_binding(&task.pod_id, b))
            .map(|(provider, spec)| HostEnvId::for_provider_spec(&provider, &spec))
            .collect()
    }

    /// Enumerate every tool the thread could reach — either admitted
    /// by scope now, or askable via `request_escalation` (in scope's
    /// pod ceiling but denied by current scope).
    ///
    /// Single primitive used by four surfaces: the wire `tools:`
    /// array (`wire_tool_descriptors`), the system-prompt listing
    /// (`render_initial_listing`), `find_tool`'s regex search, and
    /// `describe_tool`'s exact lookup. Keeps classification logic
    /// in one place instead of scattered `scope_denies` closures.
    ///
    /// Tool names are already prefix-adjusted for host-env MCPs
    /// (`{env_name}_{tool}`) so callers can filter / match against
    /// the wire-level identifier directly.
    pub(crate) fn admissible_and_askable_tools(
        &self,
        thread_id: &str,
    ) -> Vec<crate::runtime::tool_listing::AdmissibleTool> {
        use crate::runtime::tool_listing::{
            AdmissibleTool, ToolAdmission, ToolCategory, classify_admission,
        };
        let Some(task) = self.tasks.get(thread_id) else {
            return Vec::new();
        };
        let pod_ceiling_tools = self
            .pods
            .get(&task.pod_id)
            .map(|p| p.config.allow.tools.clone())
            .unwrap_or_else(whisper_agent_protocol::AllowMap::deny_all);
        let escalation_available = task.scope.escalation.is_interactive();
        // The runtime side of "MCP has no in-band signal for whether
        // the client model accepts image / document input": ask the
        // bound provider synchronously what the active model can
        // consume, then drop media-producing tools (`view_image`,
        // `view_pdf`) from the advertised set when the answer is
        // "nothing". Belt-and-suspenders only — even if a tool sneaks
        // through, the MCP media content the runtime surfaces back is
        // structurally inert for a text-only model; the filter is
        // purely to keep the model from trying.
        let active_caps = self
            .backends
            .get(&task.bindings.backend)
            .map(|entry| entry.provider.capabilities_for(&task.config.model))
            .unwrap_or_default();
        let model_takes_images = !active_caps.input.image.is_empty();
        let model_takes_documents = !active_caps.input.document.is_empty();
        let mut out: Vec<AdmissibleTool> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        let classify = |name: &str| -> ToolAdmission {
            classify_admission(&task.scope.tools, &pod_ceiling_tools, name)
        };

        for tool in crate::tools::builtin_tools::descriptors() {
            // `sudo` is only meaningful with an interactive approver
            // attached. Autonomous threads hide the tool entirely — no
            // use listing a tool whose calls would go nowhere.
            if tool.name == crate::tools::builtin_tools::SUDO && !escalation_available {
                continue;
            }
            let admission = classify(&tool.name);
            let requires_escalation = match admission {
                ToolAdmission::Admitted => false,
                ToolAdmission::Askable => true,
                ToolAdmission::OutOfReach => continue,
            };
            if seen.insert(tool.name.clone()) {
                out.push(AdmissibleTool {
                    name: tool.name,
                    description: tool.description,
                    input_schema: tool.input_schema,
                    annotations: tool.annotations,
                    category: ToolCategory::Builtin,
                    requires_escalation,
                });
            }
        }

        for bound in self.bound_mcp_hosts(thread_id) {
            for tool in &bound.entry.tools {
                // Hide media-producing tools from models that can't
                // ingest the matching modality. Tools still live in
                // the host catalog and dispatch if invoked — this
                // filter is just to keep them out of sight.
                if !model_takes_images && tool.name == "view_image" {
                    continue;
                }
                if !model_takes_documents && tool.name == "view_pdf" {
                    continue;
                }
                let (public_name, category) = match bound.prefix {
                    Some(prefix) => (
                        format!("{prefix}_{}", tool.name),
                        ToolCategory::HostEnv {
                            env_name: prefix.to_string(),
                        },
                    ),
                    None => (
                        tool.name.clone(),
                        ToolCategory::SharedMcp {
                            host_name: bound.source_name.clone(),
                        },
                    ),
                };
                let admission = classify(&public_name);
                let requires_escalation = match admission {
                    ToolAdmission::Admitted => false,
                    ToolAdmission::Askable => true,
                    ToolAdmission::OutOfReach => continue,
                };
                if seen.insert(public_name.clone()) {
                    out.push(AdmissibleTool {
                        name: public_name,
                        description: tool.description.clone(),
                        input_schema: tool.input_schema.clone(),
                        annotations: tool.annotations.clone(),
                        category,
                        requires_escalation,
                    });
                }
            }
        }

        out
    }

    /// Tool descriptors for the thread's wire `tools:` array
    /// (`Role::Tools` content at position 1). Every admitted
    /// (non-askable) tool lands on the wire so the model can actually
    /// invoke it — llama.cpp-backed endpoints build a grammar from
    /// this array and mask every other tool name at sample time, so
    /// tools advertised only in prose can't be called on those
    /// providers. The `core_tools` policy therefore only controls
    /// *representation verbosity*, not which tools appear:
    ///
    /// - Tools named in `core_tools` (or all tools when `CoreTools::All`)
    ///   carry their full description and full input-schema — this is
    ///   where we spend tokens to help the model pick the right tool.
    /// - Every other admitted tool is stripped: empty `description`,
    ///   and `input_schema` has every `description` / `title`
    ///   metadata field removed (preserving `type`, `properties`,
    ///   `required`, `enum`, combinators — everything the grammar
    ///   mask and argument validator need). The model can still
    ///   invoke these tools; when it needs the prose, `describe_tool`
    ///   re-fetches the full schema, which is kept server-side.
    pub(crate) fn wire_tool_descriptors(&self, thread_id: &str) -> Vec<McpTool> {
        use whisper_agent_protocol::tool_surface::CoreTools;
        let Some(task) = self.tasks.get(thread_id) else {
            return Vec::new();
        };
        let full_desc_all = matches!(task.tool_surface.core_tools, CoreTools::All);
        let full_desc_set: HashSet<String> = match &task.tool_surface.core_tools {
            CoreTools::All => HashSet::new(),
            CoreTools::Named(names) => names.iter().cloned().collect(),
        };
        self.admissible_and_askable_tools(thread_id)
            .into_iter()
            .filter(|t| !t.requires_escalation)
            .map(|t| {
                let is_core = full_desc_all || full_desc_set.contains(&t.name);
                let (description, input_schema) = if is_core {
                    (t.description, t.input_schema)
                } else {
                    (
                        String::new(),
                        crate::runtime::tool_listing::strip_schema_descriptions(&t.input_schema),
                    )
                };
                McpTool {
                    name: t.name,
                    description,
                    input_schema,
                    annotations: t.annotations,
                }
            })
            .collect()
    }

    /// Render the tool-catalog listing to append to the system prompt
    /// at thread seed. Empty string when `initial_listing = None`.
    pub(crate) fn render_initial_listing(&self, thread_id: &str) -> String {
        use whisper_agent_protocol::tool_surface::CoreTools;
        let Some(task) = self.tasks.get(thread_id) else {
            return String::new();
        };
        let escalation_available = task.scope.escalation.is_interactive();
        let tools = self.admissible_and_askable_tools(thread_id);
        let core_names: Vec<String> = match &task.tool_surface.core_tools {
            CoreTools::All => tools
                .iter()
                .filter(|t| !t.requires_escalation)
                .map(|t| t.name.clone())
                .collect(),
            CoreTools::Named(n) => n.clone(),
        };
        crate::runtime::tool_listing::render_listing(
            &tools,
            task.tool_surface.initial_listing,
            escalation_available,
            &core_names,
        )
    }

    /// Resolve which handler should receive a tool invocation.
    /// Routing order:
    ///
    /// 1. **Builtin**: exact name match against the builtin list
    ///    (`pod_*`, `dispatch_thread`). Always wins — builtins live
    ///    in the top-level namespace.
    /// 2. **Host-env MCP**: for every bound host-env entry with
    ///    prefix `P`, if `tool_name` starts with `P_`, strip the
    ///    prefix and match the remainder against that MCP's tool
    ///    catalog. Order follows `bindings.host_env` declaration
    ///    order so ambiguous prefix collisions between envs are
    ///    deterministic (first-declared wins).
    /// 3. **Shared MCP**: walk each shared host (unprefixed) and
    ///    match `tool_name` exactly.
    ///
    /// Returns the real tool name on `Mcp` routes (the prefix is
    /// stripped) so the MCP wire call carries the server-side name
    /// the host expects. `None` when nothing claims the name.
    pub(crate) fn route_tool(&self, thread_id: &str, tool_name: &str) -> Option<ToolRoute> {
        if crate::tools::builtin_tools::is_builtin(tool_name) {
            let pod_id = self.tasks.get(thread_id)?.pod_id.clone();
            return Some(ToolRoute::Builtin { pod_id });
        }
        for bound in self.bound_mcp_hosts(thread_id) {
            let Some(prefix) = bound.prefix else {
                // Shared MCP — match by exact name.
                if bound.entry.tools.iter().any(|t| t.name == tool_name)
                    && let Some(session) = bound.entry.session.clone()
                {
                    return Some(ToolRoute::Mcp {
                        session,
                        host: bound.entry.id.0.clone(),
                        real_name: tool_name.to_string(),
                    });
                }
                continue;
            };
            let expected = format!("{prefix}_");
            let Some(stripped) = tool_name.strip_prefix(&expected) else {
                continue;
            };
            if bound.entry.tools.iter().any(|t| t.name == stripped)
                && let Some(session) = bound.entry.session.clone()
            {
                return Some(ToolRoute::Mcp {
                    session,
                    host: bound.entry.id.0.clone(),
                    real_name: stripped.to_string(),
                });
            }
        }
        None
    }

    /// Snapshot of a pod's on-disk dir + current config — what a builtin
    /// tool future needs to operate without holding a scheduler borrow.
    pub(crate) fn pod_snapshot(&self, pod_id: &str) -> Option<PodSnapshot> {
        let pod = self.pods.get(pod_id)?;
        Some(PodSnapshot {
            pod_dir: pod.dir.clone(),
            config: pod.config.clone(),
            behavior_ids: pod.behaviors.keys().cloned().collect(),
        })
    }

    /// Iterate the thread's bound MCP host entries in precedence
    /// order: every bound host-env MCP first (in the thread's
    /// `bindings.host_env` declaration order, each carrying the env's
    /// name as its tool-prefix), then each shared MCP host in pod-
    /// declared order (no prefix). Skips ids whose entry isn't yet in
    /// the registry (host-env MCP is created when provisioning
    /// dispatches; shared entries exist from startup). `Inline` host-
    /// env bindings don't carry a stable short name and are
    /// therefore prefix-less too — in practice they're reserved for a
    /// future subagent path and don't reach this code today.
    fn bound_mcp_hosts(&self, thread_id: &str) -> Vec<BoundMcp<'_>> {
        let mut out = Vec::new();
        let Some(task) = self.tasks.get(thread_id) else {
            return out;
        };
        for binding in &task.bindings.host_env {
            let Some((provider, spec)) = self.resolve_binding(&task.pod_id, binding) else {
                continue;
            };
            let he_id = HostEnvId::for_provider_spec(&provider, &spec);
            let mcp_id = McpHostId::for_host_env(&he_id);
            let Some(entry) = self.resources.mcp_hosts.get(&mcp_id) else {
                continue;
            };
            let (prefix, source_name) = match binding {
                HostEnvBinding::Named { name } => (Some(name.as_str()), name.clone()),
                HostEnvBinding::Inline { .. } => (None, "<inline>".to_string()),
            };
            out.push(BoundMcp {
                entry,
                prefix,
                source_name,
            });
        }
        for name in &task.bindings.mcp_hosts {
            let id = McpHostId::shared(name);
            if let Some(entry) = self.resources.mcp_hosts.get(&id) {
                out.push(BoundMcp {
                    entry,
                    prefix: None,
                    source_name: name.clone(),
                });
            }
        }
        out
    }

    /// Walk the thread's bindings and return every bound resource id whose
    /// registry entry isn't yet Ready (or is missing entirely — happens
    /// post-restart for the host-env MCP). Used by
    /// `send_user_message` to park the thread in `WaitingOnResources`
    /// when provisioning is still in flight. Covers every host-env
    /// binding the thread has (not just the first) — the thread stays
    /// blocked until all of them finish provisioning.
    fn pending_resources_for(&self, thread_id: &str) -> Vec<String> {
        let Some(task) = self.tasks.get(thread_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for he_id in self.host_env_ids_for_thread(thread_id) {
            match self.resources.host_envs.get(&he_id) {
                Some(e) if e.state.is_ready() => {}
                _ => out.push(he_id.0.clone()),
            }
            let mcp_id = McpHostId::for_host_env(&he_id);
            match self.resources.mcp_hosts.get(&mcp_id) {
                Some(e) if e.state.is_ready() => {}
                _ => out.push(mcp_id.0),
            }
        }
        for name in &task.bindings.mcp_hosts {
            let id = McpHostId::shared(name);
            match self.resources.mcp_hosts.get(&id) {
                Some(e) if e.state.is_ready() => {}
                _ => out.push(id.0),
            }
        }
        out
    }

    /// Ensure a host-env provisioning future is in flight for every
    /// binding this thread has that isn't yet Ready. No-op when the
    /// thread has no host-env bindings — in that case the thread's
    /// tool set is just the shared MCPs it bound to, no provisioning
    /// needed. Called from `create_task` and `send_user_message`
    /// (post-restart recovery).
    ///
    /// The `provisioning_in_flight` guard is keyed per `HostEnvId`:
    /// threads that share a binding (deduped to the same id) reuse a
    /// single in-flight provision instead of each spawning their own.
    /// A thread with multiple host-envs dispatches one future per id
    /// (minus any id that already has an in-flight provision from
    /// another thread or from a prior call).
    fn ensure_host_env_provisioning(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let host_env_ids = self.host_env_ids_for_thread(thread_id);
        if host_env_ids.is_empty() {
            return;
        }
        for host_env_id in host_env_ids {
            if self.provisioning_in_flight.contains(&host_env_id) {
                continue;
            }
            let mcp_id = McpHostId::for_host_env(&host_env_id);
            let needs_dispatch = match self.resources.mcp_hosts.get(&mcp_id) {
                Some(entry) => !entry.state.is_ready(),
                None => true,
            };
            if !needs_dispatch {
                continue;
            }
            // Register the MCP entry up front so waiters can observe
            // Provisioning. Idempotent: re-registering for the same
            // host_env just touches last_used and adds the thread user.
            self.resources
                .pre_register_host_env_mcp(&host_env_id, thread_id);
            self.emit_mcp_host_updated(&mcp_id);
            self.provisioning_in_flight.insert(host_env_id.clone());
            let fut = io_dispatch::provision_host_env_mcp(self, thread_id.to_string(), host_env_id);
            pending_io.push(fut);
        }
    }

    /// The thread's current `pod_modify` cap, or `None` if the thread is
    /// unknown. Read-only access for tool-dispatch sites that need the
    /// cap to gate path-scoped pod-file operations.
    pub(crate) fn thread_pod_modify_cap(&self, thread_id: &str) -> crate::permission::PodModifyCap {
        self.tasks
            .get(thread_id)
            .map(|t| t.scope.pod_modify)
            .unwrap_or(crate::permission::PodModifyCap::None)
    }

    /// The thread's current `behaviors` cap, or `None` if the thread is
    /// unknown. Used to gate `pod_run_behavior` / `pod_set_behavior_enabled`
    /// at tool-dispatch entry.
    pub(crate) fn thread_behaviors_cap(
        &self,
        thread_id: &str,
    ) -> crate::permission::BehaviorOpsCap {
        self.tasks
            .get(thread_id)
            .map(|t| t.scope.behaviors)
            .unwrap_or(crate::permission::BehaviorOpsCap::None)
    }

    /// Apply the behavior-authoring gate. When `tool_name` is a pod-file
    /// write targeting `behaviors/<id>/behavior.toml`, compare the
    /// declared scope against the author thread's scope per
    /// [`crate::permission::BehaviorOpsCap`] rules and return
    /// `Some(error_message)` on a denial. Returns `None` (admits) for
    /// every other tool / path, and for writes that fail to parse
    /// (let `prepare_update` surface the parse error downstream).
    ///
    /// Thin wrapper around [`check_behavior_authoring_pure`] that looks
    /// up the caller's scope and pod ceiling from scheduler state. The
    /// decision logic itself is pure and covered by that helper's tests.
    pub(crate) fn check_behavior_authoring(
        &self,
        thread_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Option<String> {
        use crate::tools::builtin_tools::{POD_EDIT_FILE, POD_WRITE_FILE};

        if !matches!(tool_name, POD_WRITE_FILE | POD_EDIT_FILE) {
            return None;
        }
        let filename = args.get("filename").and_then(|v| v.as_str())?;
        let (_id, suffix) = crate::pod::fs::parse_behavior_path(filename)?;
        if suffix != crate::pod::behaviors::BEHAVIOR_TOML {
            return None;
        }

        let task = self.tasks.get(thread_id)?;
        let caller_scope = task.scope.clone();
        let pod_ceiling = self.pod_scope_ceiling(&task.pod_id);

        let content = args.get("content").and_then(|v| v.as_str())?;
        let Ok(parsed) = crate::pod::behaviors::parse_toml(content) else {
            return None;
        };
        check_behavior_authoring_pure(&caller_scope, &pod_ceiling, filename, &parsed.scope)
    }

    /// Sudo-path authoring gate. Same rules as
    /// [`Self::check_behavior_authoring`] but runs with the caller's
    /// scope effectively raised to `pod_ceiling` — the single-call
    /// cap bypass sudo grants. So on an `author_narrower` ceiling the
    /// strictly-narrower subset check still applies (fire_scope must
    /// be a strict subset of the pod ceiling), and on `author_any`
    /// every admissible behavior write passes. Runs at sudo-register
    /// time so the user never sees an approval prompt for a write
    /// that would fail this gate anyway.
    pub(crate) fn check_behavior_authoring_for_sudo(
        &self,
        thread_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Option<String> {
        use crate::tools::builtin_tools::{POD_EDIT_FILE, POD_WRITE_FILE};

        if !matches!(tool_name, POD_WRITE_FILE | POD_EDIT_FILE) {
            return None;
        }
        let filename = args.get("filename").and_then(|v| v.as_str())?;
        let (_id, suffix) = crate::pod::fs::parse_behavior_path(filename)?;
        if suffix != crate::pod::behaviors::BEHAVIOR_TOML {
            return None;
        }

        let task = self.tasks.get(thread_id)?;
        let pod_ceiling = self.pod_scope_ceiling(&task.pod_id);

        let content = args.get("content").and_then(|v| v.as_str())?;
        let Ok(parsed) = crate::pod::behaviors::parse_toml(content) else {
            return None;
        };
        // pod_ceiling twice: as caller_scope (sudo-elevated) and as
        // the ceiling narrowing fire_scope — identical by construction
        // under sudo.
        check_behavior_authoring_pure(&pod_ceiling, &pod_ceiling, filename, &parsed.scope)
    }

    /// Derive the effective [`Scope`] a newly-created thread in `pod_id`
    /// starts with. The bindings + tools come from the pod's `[allow]`
    /// table; typed caps come from `[thread_defaults.caps]` in the pod
    /// TOML (conservative defaults when the section is omitted — see
    /// `ThreadDefaultCaps::default`). Escalation channels are attached
    /// at `create_task` per-caller, so this snapshot always starts
    /// `Escalation::None`.
    pub(super) fn pod_thread_scope(&self, pod_id: &str) -> crate::permission::Scope {
        use crate::permission::{Escalation, Scope, SetOrAll};
        let Some(pod) = self.pods.get(pod_id) else {
            return Scope::deny_all();
        };
        let allow = &pod.config.allow;
        let td_caps = pod.config.thread_defaults.caps;
        Scope {
            backends: SetOrAll::only(allow.backends.iter().cloned()),
            host_envs: SetOrAll::only(allow.host_env.iter().map(|h| h.name.clone())),
            mcp_hosts: SetOrAll::only(allow.mcp_hosts.iter().cloned()),
            tools: allow.tools.clone(),
            pod_modify: td_caps.pod_modify,
            dispatch: td_caps.dispatch,
            behaviors: td_caps.behaviors,
            escalation: Escalation::None,
        }
    }

    /// Build a `Scope` representing the pod's `[allow]` ceiling — the
    /// hardest bound threads in this pod can hold. Used as the ceiling
    /// argument to [`crate::permission::Scope::check_escalation`] so a
    /// grant can't widen a thread past what the pod allows. Differs
    /// from [`Self::pod_thread_scope`] only on the typed caps — which
    /// read from `allow.caps` rather than `thread_defaults.caps`.
    pub(super) fn pod_scope_ceiling(&self, pod_id: &str) -> crate::permission::Scope {
        use crate::permission::{Escalation, Scope, SetOrAll};
        let Some(pod) = self.pods.get(pod_id) else {
            return Scope::deny_all();
        };
        let allow = &pod.config.allow;
        Scope {
            backends: SetOrAll::only(allow.backends.iter().cloned()),
            host_envs: SetOrAll::only(allow.host_env.iter().map(|h| h.name.clone())),
            mcp_hosts: SetOrAll::only(allow.mcp_hosts.iter().cloned()),
            tools: allow.tools.clone(),
            pod_modify: allow.caps.pod_modify,
            dispatch: allow.caps.dispatch,
            behaviors: allow.caps.behaviors,
            escalation: Escalation::None,
        }
    }

    pub fn with_persister(mut self, persister: Persister) -> Self {
        self.persister = Some(persister);
        self
    }

    /// Seed the scheduler with pods + threads loaded from disk. The persister
    /// should have already transitioned any in-flight internal states to
    /// Failed before handoff. Pods loaded here win over the default pod
    /// inserted at construction time (in case a real on-disk default exists).
    pub fn load_state(&mut self, state: LoadedState) {
        for pod in state.pods {
            // Replace the construction-time default if disk has a pod with
            // the same id — disk wins.
            self.pods.insert(pod.id.clone(), pod);
        }
        for mut task in state.threads {
            // Re-pre-register each of the thread's host-env bindings so
            // the registry knows their (provider, spec) pairs. For
            // `Named` bindings we look up the pod entry by name; for
            // `Inline` the spec lives in the binding itself. Bindings
            // that can't be resolved (pod allow entry removed, Inline
            // spec no longer matches any allow entry) are dropped; if
            // the resulting list is empty and the pod has a default,
            // we re-seed from the default.
            let mut bindings_dirty = false;
            // Escalation is a live-conn pointer; conn ids don't
            // survive a process restart. Any persisted
            // `Interactive{via_conn}` refers to a conn that no longer
            // exists — keeping it would leave the thread reporting
            // `is_interactive() == true` while `sudo` prompts silently
            // route to a dead conn. Clear to `None`; the first
            // `SendUserMessage` from a live conn rebinds via
            // `rebind_escalation_if_orphaned`.
            if let crate::permission::Escalation::Interactive { .. } = task.scope.escalation {
                task.scope.escalation = crate::permission::Escalation::None;
            }
            // Migrate legacy Inline bindings: walk the pod's allow
            // list for a structurally equivalent entry and rebind to
            // the Named reference; otherwise drop the binding.
            let original = std::mem::take(&mut task.bindings.host_env);
            let mut migrated: Vec<HostEnvBinding> = Vec::with_capacity(original.len());
            for binding in original {
                match binding {
                    HostEnvBinding::Inline { provider, spec } => {
                        let matching = self.pods.get(&task.pod_id).and_then(|pod| {
                            pod.config
                                .allow
                                .host_env
                                .iter()
                                .find(|nh| nh.provider == provider && nh.spec == spec)
                                .map(|nh| nh.name.clone())
                        });
                        match matching {
                            Some(name) => {
                                warn!(
                                    thread_id = %task.id,
                                    rebound_to = %name,
                                    "migrating Inline host_env binding to Named reference",
                                );
                                migrated.push(HostEnvBinding::Named { name });
                                bindings_dirty = true;
                            }
                            None => {
                                warn!(
                                    thread_id = %task.id,
                                    "Inline host_env binding has no structural match in pod.allow.host_env; dropping",
                                );
                                bindings_dirty = true;
                            }
                        }
                    }
                    HostEnvBinding::Named { name } => {
                        migrated.push(HostEnvBinding::Named { name });
                    }
                }
            }
            task.bindings.host_env = migrated;

            // Resolve each surviving binding; drop those whose allow
            // entry has gone missing. If the list ends up empty and the
            // pod has defaults, re-seed from `thread_defaults.host_env`
            // so threads don't silently land on "no host env."
            let resolvable: Vec<HostEnvBinding> = task
                .bindings
                .host_env
                .iter()
                .filter(|b| self.resolve_binding(&task.pod_id, b).is_some())
                .cloned()
                .collect();
            if resolvable.len() != task.bindings.host_env.len() {
                warn!(
                    thread_id = %task.id,
                    "loaded thread has host_env bindings that could not be resolved; dropping them",
                );
                task.bindings.host_env = resolvable;
                bindings_dirty = true;
            }
            if task.bindings.host_env.is_empty()
                && let Some(pod) = self.pods.get(&task.pod_id)
                && !pod.config.thread_defaults.host_env.is_empty()
            {
                let defaulted: Vec<HostEnvBinding> = pod
                    .config
                    .thread_defaults
                    .host_env
                    .iter()
                    .map(|name| HostEnvBinding::Named { name: name.clone() })
                    .collect();
                warn!(
                    thread_id = %task.id,
                    replaced_with = ?pod.config.thread_defaults.host_env,
                    "loaded thread had no resolvable host_env bindings; rebinding to pod defaults",
                );
                task.bindings.host_env = defaulted;
                bindings_dirty = true;
            }
            for binding in task.bindings.host_env.clone() {
                if let Some((provider, spec)) = self.resolve_binding(&task.pod_id, &binding) {
                    self.resources
                        .pre_register_host_env(&task.id, provider, spec);
                }
            }

            let backend_id = BackendId::for_name(&task.bindings.backend);
            self.resources.add_backend_user(&backend_id, &task.id);
            // Refcount the shared MCP hosts this thread is bound to.
            // Shared registry entries exist from server startup, so
            // this is just bookkeeping for GC-correctness (idle
            // shared MCPs don't get torn down while users reference
            // them).
            for name in &task.bindings.mcp_hosts {
                self.resources
                    .add_mcp_user(&McpHostId::shared(name), &task.id);
            }

            // Make sure the owning pod knows about this thread. If the pod
            // wasn't in `state.pods` (shouldn't happen — load_pod walks both
            // together) we drop the thread on the floor with a warning.
            if let Some(pod) = self.pods.get_mut(&task.pod_id) {
                pod.threads.insert(task.id.clone());
                let task_id = task.id.clone();
                self.cancel_tokens
                    .insert(task_id.clone(), tokio_util::sync::CancellationToken::new());
                self.tasks.insert(task_id.clone(), task);
                if bindings_dirty {
                    self.mark_dirty(&task_id);
                }
            } else {
                warn!(
                    thread_id = %task.id,
                    pod_id = %task.pod_id,
                    "loaded thread references unknown pod; dropping",
                );
            }
        }
    }

    fn mark_dirty(&mut self, thread_id: &str) {
        if self.persister.is_some() {
            self.dirty.insert(thread_id.to_string());
        }
    }

    /// Mark a behavior's `state.json` for writeback at the next flush.
    /// Mirrors `mark_dirty` for threads — the batched flush at the end
    /// of each scheduler-loop iteration serializes writes per-behavior,
    /// avoiding the tokio::spawn race where two quick updates could
    /// land their disk writes out of order. No-op when no persister is
    /// configured (the in-memory state is already up to date).
    fn mark_behavior_dirty(&mut self, pod_id: &str, behavior_id: &str) {
        if self.persister.is_some() {
            self.dirty_behaviors
                .insert((pod_id.to_string(), behavior_id.to_string()));
        }
    }

    async fn flush_dirty(&mut self) {
        let Some(persister) = &self.persister else {
            self.dirty.clear();
            self.dirty_behaviors.clear();
            return;
        };
        let dirty = std::mem::take(&mut self.dirty);
        for thread_id in dirty {
            let Some(task) = self.tasks.get(&thread_id) else {
                continue;
            };
            if let Err(e) = persister.flush(task).await {
                error!(thread_id = %thread_id, error = %e, "persist flush failed");
            }
        }
        let dirty_behaviors = std::mem::take(&mut self.dirty_behaviors);
        for (pod_id, behavior_id) in dirty_behaviors {
            let Some(pod) = self.pods.get(&pod_id) else {
                continue;
            };
            let Some(behavior) = pod.behaviors.get(&behavior_id) else {
                continue;
            };
            if let Err(e) =
                crate::pod::behaviors::write_state(&pod.dir, &behavior_id, &behavior.state).await
            {
                error!(
                    pod_id = %pod_id,
                    behavior_id = %behavior_id,
                    error = %e,
                    "persist behavior state failed"
                );
            }
        }
    }

    /// Apply an inbox message. Returns the task_ids (usually 0 or 1) whose state may
    /// have advanced and need `step_until_blocked` run.
    fn apply_input(
        &mut self,
        input: SchedulerMsg,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        match input {
            SchedulerMsg::RegisterClient {
                conn_id,
                outbound,
                is_admin,
            } => {
                self.router.register_client(conn_id, outbound);
                if is_admin {
                    self.admin_connections.insert(conn_id);
                }
                // Replay in-flight bucket builds so a freshly-connected
                // client sees their `Started` + current `Progress`
                // snapshot rather than waiting until the next terminal
                // `BuildEnded`.
                self.replay_active_builds_to_client(conn_id);
            }
            SchedulerMsg::UnregisterClient { conn_id } => {
                self.router.unregister_client(conn_id);
                self.admin_connections.remove(&conn_id);
                self.revoke_escalation_channel(conn_id, pending_io);
            }
            SchedulerMsg::ClientMessage { conn_id, msg } => {
                self.apply_client_message(conn_id, msg, pending_io);
            }
            SchedulerMsg::TriggerFired {
                pod_id,
                behavior_id,
                payload,
                reply,
            } => {
                let result =
                    self.handle_webhook_trigger(&pod_id, &behavior_id, payload, pending_io);
                // Ignore send errors — the HTTP handler may have given up
                // (client disconnect / timeout). The fire already happened
                // if it was going to; nothing to clean up.
                let _ = reply.send(result);
            }
            SchedulerMsg::OauthCallback { state, code, error } => {
                self.apply_oauth_callback(state, code, error, pending_io);
            }
        }
    }

    /// Spawn a fresh thread under `pod_id`. `requester` identifies the
    /// connection that asked for it, if any; trigger-driven spawns
    /// (behaviors, cron, etc.) pass `None`. `origin` stamps behavior
    /// provenance on the thread; `None` for interactive work.
    ///
    /// `base_scope_override` lets a caller (notably `run_behavior`)
    /// supply a pre-composed base scope that differs from the pod's
    /// `thread_defaults` baseline. Behavior fires use the pod.allow
    /// ceiling narrowed by the behavior's declared `[scope]` — a
    /// wider starting point than interactive threads, because the
    /// behavior author already committed to a scope at author time.
    /// `None` uses the pod's thread_defaults (interactive baseline).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_task(
        &mut self,
        requester: Option<ConnId>,
        correlation_id: Option<String>,
        pod_id: Option<String>,
        config_override: Option<ThreadConfigOverride>,
        bindings_request: Option<ThreadBindingsRequest>,
        origin: Option<whisper_agent_protocol::BehaviorOrigin>,
        // Optional dispatch lineage: (parent_thread_id, parent_depth).
        // When provided, the new thread is stamped before the
        // ThreadCreated broadcast fires so subscribers see lineage on
        // arrival and the sidebar's dispatch-nesting view renders
        // correctly without a second patch event.
        dispatched_by_parent: Option<(String, u32)>,
        base_scope_override: Option<crate::permission::Scope>,
        // Behavior fires pass their `scope.tool_surface` override
        // composed onto the pod baseline; other callers (WS create,
        // dispatch, compaction) pass `None` and inherit the pod's
        // `thread_defaults.tool_surface`.
        tool_surface_override: Option<whisper_agent_protocol::ToolSurface>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<String, String> {
        // Resolve which pod this thread lands in. None routes to the
        // server's default pod.
        let pod_id = pod_id.unwrap_or_else(|| self.default_pod_id.clone());
        let pod = self
            .pods
            .get(&pod_id)
            .ok_or_else(|| format!("unknown pod `{pod_id}`"))?;

        // Pull the system-prompt choice off the override before
        // collapsing the rest into `ThreadConfig`. `ThreadConfig`
        // intentionally doesn't carry the prompt anymore — it lands
        // in `conversation[0]` via `seed_thread_setup` below — so the
        // override field is a creation-time-only input.
        let system_prompt_choice = config_override
            .as_ref()
            .and_then(|o| o.system_prompt.clone());
        // Resolve the thread's plain config (model, limits, policy).
        let base_config = base_thread_config_from_pod(pod);
        let config = apply_config_override(base_config, config_override);

        // Resolve the binding choices (backend / sandbox / shared MCP hosts):
        // start from pod defaults, layer the request on top, validate every
        // choice against the pod's `[allow]` cap — and, for dispatched
        // children, against the dispatching parent's scope. This is the
        // "child bindings must live within parent scope" half of the
        // narrowing invariant; the scope-narrowing below is the other half.
        let parent_scope = dispatched_by_parent
            .as_ref()
            .and_then(|(parent_id, _)| self.tasks.get(parent_id))
            .map(|p| p.scope.clone());
        let resolved = resolve_bindings_choice(pod, bindings_request, parent_scope.as_ref())?;
        // Validate shared MCP host names against the server catalog too —
        // a name allowed by the pod but not actually wired up at the server
        // is a misconfiguration we want to surface at create-time.
        for name in &resolved.shared_host_names {
            if !self
                .resources
                .mcp_hosts
                .contains_key(&McpHostId::shared(name))
            {
                return Err(format!(
                    "shared MCP host `{name}` not configured on this server (pod `{pod_id}` allows it but no upstream is wired up)",
                ));
            }
        }

        let thread_id = new_task_id();
        let bindings = ThreadBindings {
            backend: resolved.backend_name.clone(),
            host_env: resolved.host_env.clone(),
            mcp_hosts: resolved.shared_host_names.clone(),
            tool_filter: None,
        };

        // Snapshot the pod's effective Scope into the thread; mid-flight
        // edits to the pod's allow table don't retroactively change the
        // thread's active scope. Caps default to the `thread_defaults`
        // baseline described in `docs/design_permissions_rework.md`
        // until pod.toml schema grows `[allow.caps]` /
        // `[thread_defaults.caps]` (task #6).
        //
        // Pod-allow-derived base, then narrow by parent's scope when the
        // thread is dispatched — the child can never hold more than the
        // parent on bindings, tools, or caps. Non-dispatched threads
        // (WS-client create, cron-fired behavior) get the raw pod-
        // derived scope.
        //
        // Escalation is set separately — it's a runtime channel pointer,
        // not an authority bound, so it isn't narrowed against the pod
        // ceiling (which doesn't configure conn ids):
        //   - Dispatched child: inherit the parent's channel verbatim.
        //     A child with `Interactive{parent_conn}` can request
        //     widening; if that conn drops, the escalation sweep
        //     revokes it.
        //   - Top-level WS create: attach `Interactive{via_conn: conn_id}`.
        //   - Behavior fire / auto-compact / cron: stays `None`;
        //     `request_escalation` is filtered out of the catalog.
        let base_scope = base_scope_override.unwrap_or_else(|| self.pod_thread_scope(&pod_id));
        let mut scope = match &dispatched_by_parent {
            Some((parent_id, _)) => match self.tasks.get(parent_id) {
                Some(parent) => base_scope.narrow(&parent.scope),
                None => base_scope,
            },
            None => base_scope,
        };
        scope.escalation = match &dispatched_by_parent {
            Some((parent_id, _)) => self
                .tasks
                .get(parent_id)
                .map(|p| p.scope.escalation)
                .unwrap_or(crate::permission::Escalation::None),
            None => match requester {
                Some(conn_id) => crate::permission::Escalation::Interactive { via_conn: conn_id },
                None => crate::permission::Escalation::None,
            },
        };
        // Compose the effective tool surface: pod baseline, optionally
        // replaced by the behavior-declared override. `compose` is a
        // wholesale replace (not per-field narrowing) — tool-surface
        // knobs are presentation preferences, not permissions.
        let tool_surface = pod
            .config
            .thread_defaults
            .tool_surface
            .compose(tool_surface_override.as_ref());
        let mut task = Thread::new(
            thread_id.clone(),
            pod_id.clone(),
            config,
            bindings,
            scope,
            tool_surface,
        );
        if let Some(origin) = origin {
            task = task.with_origin(origin);
        }
        if let Some((parent_id, parent_depth)) = dispatched_by_parent {
            task = task.with_dispatched_by(parent_id, parent_depth);
        }

        let thread_id = self.register_new_task(task, requester, correlation_id, pending_io);
        self.seed_thread_setup(&thread_id, system_prompt_choice.as_ref());
        info!(thread_id = %thread_id, pod_id = %pod_id, "task created");
        Ok(thread_id)
    }

    /// Push the setup prefix onto a freshly-registered thread's
    /// conversation. System prompt is pod config — always available
    /// synchronously, pushed immediately. Tool manifest depends on
    /// resource state — if the thread has a host-env binding whose
    /// MCP hasn't finished its initial `tools/list` yet, the
    /// `Role::Tools` push is **deferred** to the
    /// `HostEnvMcpCompleted` handler so the snapshot captures the
    /// real catalog rather than an empty placeholder.
    ///
    /// Why defer instead of refresh-later: the setup prefix is
    /// cache-fingerprint-critical to the model provider — once the
    /// thread starts trading messages with the LLM, mutating
    /// `messages[1]` busts the entire cached prefix and forces a
    /// rewrite on every subsequent turn. Getting it right the first
    /// time beats fixing it up after.
    ///
    /// `override_choice` — when `Some`, takes precedence over the
    /// pod's cached `system_prompt`. `File { name }` reads
    /// `<pod_dir>/<name>` synchronously (pod-relative, no parent-
    /// escape); `Text { text }` uses the literal verbatim. Missing
    /// or unreadable file falls back to the pod default with a
    /// warn-level log — same graceful-degrade policy as the pod
    /// loader when `system_prompt_file` points at nothing.
    fn seed_thread_setup(
        &mut self,
        thread_id: &str,
        override_choice: Option<&whisper_agent_protocol::SystemPromptChoice>,
    ) {
        let system_prompt = {
            let pod = self
                .tasks
                .get(thread_id)
                .and_then(|t| self.pods.get(&t.pod_id));
            match (override_choice, pod) {
                (Some(whisper_agent_protocol::SystemPromptChoice::Text { text }), _) => {
                    text.clone()
                }
                (Some(whisper_agent_protocol::SystemPromptChoice::File { name }), Some(pod)) => {
                    resolve_system_prompt_file(pod, name)
                }
                (None, Some(pod)) => pod.system_prompt.clone(),
                // Thread/pod missing — defensive no-prompt. Shouldn't
                // happen in practice (task was just registered).
                (_, None) => String::new(),
            }
        };
        if let Some(task) = self.tasks.get_mut(thread_id) {
            task.conversation
                .push(whisper_agent_protocol::Message::system_text(system_prompt));
        }
        // Tools manifest + initial listing + memory snapshot all land
        // together as a single idempotent operation. Self-gates on
        // host-env MCP readiness: if some binding is still
        // provisioning, the hook in `apply_provision_completion`
        // calls back once it lands.
        self.finalize_setup_prefix(thread_id);
        self.mark_dirty(thread_id);
    }

    /// Are every one of the thread's host-env MCPs ready to advertise
    /// tools? True when the thread has no host-env bindings, or when
    /// every bound host-env MCP entry is in `ResourceState::Ready`.
    /// False means the `Role::Tools` seed must wait for the
    /// remaining `HostEnvMcpCompleted` hooks — we merge every env's
    /// tools into a single manifest message, so deferring until the
    /// slowest env lands keeps the snapshot complete instead of
    /// freezing a partial tool list.
    fn host_env_mcp_is_ready(&self, thread_id: &str) -> bool {
        let host_env_ids = self.host_env_ids_for_thread(thread_id);
        if host_env_ids.is_empty() {
            return true; // no host-env bindings — nothing to wait for
        }
        host_env_ids.iter().all(|he_id| {
            let mcp_id = McpHostId::for_host_env(he_id);
            self.resources
                .mcp_hosts
                .get(&mcp_id)
                .map(|e| e.state.is_ready())
                .unwrap_or(false)
        })
    }

    /// Finalize the setup prefix for a freshly-seeded thread: insert
    /// the `Role::Tools` manifest, the initial tool-catalog listing
    /// (a `Role::System` block), and the memory-index snapshot (also
    /// `Role::System`), in that order, at `setup_prefix_end()`.
    ///
    /// Single idempotent handler — replaces the prior trio of
    /// `push_*_snapshot` methods that each carried their own marker
    /// check. Those markers overlapped (listing and memory are both
    /// `Role::System` at the same position), which meant ordering
    /// between the three calls had to be just right or one would
    /// false-match another. Consolidating them here makes the
    /// idempotency gate one thing: does `Role::Tools` exist in the
    /// conversation? If yes, the prefix already finalized and nothing
    /// more is needed. If no, assemble and insert all three together.
    ///
    /// Self-gates on `host_env_mcp_is_ready` — returns early if any
    /// host-env MCP is still provisioning. The completion hook
    /// (`apply_provision_completion`) calls us again once the last
    /// MCP lands, so there's one well-defined trigger: "all
    /// ingredients ready, assemble now."
    ///
    /// Steady-state layout is `[System, Tools, Listing?, Memory,
    /// ...body]` — listing sits adjacent to Tools (it's a rendering
    /// of the same catalog) and memory closes the prefix before the
    /// first body turn. Listing is skipped when
    /// `tool_surface.initial_listing = None` (empty render).
    ///
    /// Non-append mutation: inserting at `setup_prefix_end()` shifts
    /// later messages, so streaming subscribers would drift from
    /// server state. One `ThreadSnapshot` broadcast at the end
    /// rebuilds their view from authoritative state in a single pass.
    fn finalize_setup_prefix(&mut self, thread_id: &str) {
        if !self.host_env_mcp_is_ready(thread_id) {
            return;
        }
        let Some(task) = self.tasks.get(thread_id) else {
            return;
        };
        let already_finalized = task
            .conversation
            .messages()
            .iter()
            .any(|m| m.role == whisper_agent_protocol::Role::Tools);
        if already_finalized {
            return;
        }

        let tool_blocks: Vec<whisper_agent_protocol::ContentBlock> = self
            .wire_tool_descriptors(thread_id)
            .into_iter()
            .map(|t| whisper_agent_protocol::ContentBlock::ToolSchema {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
            })
            .collect();
        let tools_msg = whisper_agent_protocol::Message::tools_manifest(tool_blocks);

        let listing_text = self.render_initial_listing(thread_id);

        let pod_dir = self
            .tasks
            .get(thread_id)
            .and_then(|t| self.pods.get(&t.pod_id))
            .map(|p| p.dir.clone());
        let memory_msg = pod_dir
            .map(|dir| crate::runtime::memory_snapshot::build_block(&dir, chrono::Utc::now()));

        let snapshot = if let Some(task) = self.tasks.get_mut(thread_id) {
            let mut at = task.conversation.setup_prefix_end();
            task.conversation.insert(at, tools_msg);
            at += 1;
            if !listing_text.is_empty() {
                task.conversation.insert(
                    at,
                    whisper_agent_protocol::Message::system_text(listing_text),
                );
                at += 1;
            }
            if let Some(memory_msg) = memory_msg {
                task.conversation.insert(at, memory_msg);
            }
            Some(task.snapshot())
        } else {
            None
        };
        self.mark_dirty(thread_id);
        if let Some(snapshot) = snapshot {
            self.router.broadcast_to_subscribers(
                thread_id,
                ServerToClient::ThreadSnapshot {
                    thread_id: thread_id.to_string(),
                    snapshot,
                },
            );
        }
    }

    /// Binding-side prefix a given thread would apply to tool names
    /// from `target_mcp_id`, if any. `Some(prefix)` for host-env
    /// bindings (host-env tools are prefixed `{env_name}_{tool}` on
    /// the wire); `None` for shared MCPs or if the thread isn't
    /// bound to the target MCP. Used by the mid-conversation
    /// activation path to render the right public names.
    fn prefix_for_bound_mcp(&self, thread_id: &str, target_mcp_id: &McpHostId) -> Option<String> {
        self.bound_mcp_hosts(thread_id)
            .into_iter()
            .find(|b| &b.entry.id == target_mcp_id)
            .and_then(|b| b.prefix.map(|s| s.to_string()))
    }

    /// Append a tail-end `Role::System` activation message for a set
    /// of newly-available tool names. Honors the thread's
    /// `activation_surface`: `Announce` renders names + one-liners,
    /// `InjectSchema` renders full schemas inline.
    ///
    /// Used by the escalation-approve path (for `AddTool` grants) and
    /// by mid-conversation MCP-attach paths. Append-only by
    /// construction — never mutates position-1 `Role::Tools`, so the
    /// static prefix cache stays hot.
    ///
    /// Tools passed as `names` that aren't admissible to the thread
    /// right now (denied, or out of pod ceiling) are silently dropped
    /// — the caller doesn't need to pre-filter.
    pub(crate) fn push_tool_activation_message(&mut self, thread_id: &str, names: &[String]) {
        use whisper_agent_protocol::tool_surface::ActivationSurface;
        if names.is_empty() {
            return;
        }
        let Some(task) = self.tasks.get(thread_id) else {
            return;
        };
        let surface = task.tool_surface.activation_surface;
        let all = self.admissible_and_askable_tools(thread_id);
        let wanted: std::collections::HashSet<&String> = names.iter().collect();
        let matches: Vec<_> = all
            .into_iter()
            .filter(|t| wanted.contains(&t.name) && !t.requires_escalation)
            .collect();
        if matches.is_empty() {
            return;
        }
        let text = match surface {
            ActivationSurface::Announce => {
                crate::runtime::tool_listing::render_activation_announce(&matches)
            }
            ActivationSurface::InjectSchema => {
                crate::runtime::tool_listing::render_activation_inject(&matches)
            }
        };
        if text.is_empty() {
            return;
        }
        let snapshot = if let Some(task) = self.tasks.get_mut(thread_id) {
            task.conversation
                .push(whisper_agent_protocol::Message::system_text(text));
            Some(task.snapshot())
        } else {
            None
        };
        self.mark_dirty(thread_id);
        if let Some(snapshot) = snapshot {
            self.router.broadcast_to_subscribers(
                thread_id,
                ServerToClient::ThreadSnapshot {
                    thread_id: thread_id.to_string(),
                    snapshot,
                },
            );
        }
    }

    /// Shared tail for freshly-constructed threads (minted by
    /// `create_task`, `fork_task`, etc). Pre-registers the host env
    /// if `task.bindings.host_env` is set, inserts `task` into the
    /// scheduler map and the owning pod's `threads` set, takes up
    /// backend / shared-MCP resource slots, kicks off host-env
    /// provisioning, and broadcasts `ThreadCreated` (routing the
    /// correlation_id only to the requester). The caller must
    /// finalize `task.bindings` / origin / dispatch lineage before
    /// calling — this function never mutates those.
    ///
    /// Returns the thread id. Safe to call with a `requester` of
    /// `None` (trigger-driven spawns from behaviors / cron): falls
    /// back to a single uncorrelated broadcast reaching every
    /// client.
    fn register_new_task(
        &mut self,
        task: Thread,
        requester: Option<ConnId>,
        correlation_id: Option<String>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> String {
        let thread_id = task.id.clone();
        let pod_id = task.pod_id.clone();

        // Deterministic `HostEnvId` means a second thread with the
        // same (provider, spec) joins the existing entry as a user.
        // Empty binding list → no pre-register; tools come from shared
        // MCPs alone.
        let host_env_ids: Vec<HostEnvId> = task
            .bindings
            .host_env
            .iter()
            .map(|binding| {
                let (provider, spec) = self
                    .resolve_binding(&pod_id, binding)
                    .expect("caller already validated this binding");
                self.resources
                    .pre_register_host_env(&thread_id, provider, spec)
            })
            .collect();

        let backend_name = task.bindings.backend.clone();
        let shared_hosts = task.bindings.mcp_hosts.clone();
        let summary = task.summary();
        self.cancel_tokens.insert(
            thread_id.clone(),
            tokio_util::sync::CancellationToken::new(),
        );
        self.tasks.insert(thread_id.clone(), task);
        if let Some(pod) = self.pods.get_mut(&pod_id) {
            pod.threads.insert(thread_id.clone());
        }

        let backend_id = BackendId::for_name(&backend_name);
        self.resources.add_backend_user(&backend_id, &thread_id);
        self.emit_backend_updated(&backend_id);
        for id in &host_env_ids {
            self.emit_host_env_updated(id);
        }
        for name in &shared_hosts {
            let host_id = McpHostId::shared(name);
            self.resources.add_mcp_user(&host_id, &thread_id);
            self.emit_mcp_host_updated(&host_id);
        }
        self.ensure_host_env_provisioning(&thread_id, pending_io);

        match (requester, correlation_id) {
            (Some(conn), Some(cid)) => {
                self.router.broadcast_task_list_except(
                    ServerToClient::ThreadCreated {
                        thread_id: thread_id.clone(),
                        summary: summary.clone(),
                        correlation_id: None,
                    },
                    conn,
                );
                self.router.send_to_client(
                    conn,
                    ServerToClient::ThreadCreated {
                        thread_id: thread_id.clone(),
                        summary,
                        correlation_id: Some(cid),
                    },
                );
            }
            _ => {
                self.router
                    .broadcast_task_list(ServerToClient::ThreadCreated {
                        thread_id: thread_id.clone(),
                        summary,
                        correlation_id: None,
                    });
            }
        }
        thread_id
    }

    /// Fork a thread at a prefix boundary. Mirrors the resource-wiring
    /// tail of [`Self::create_task`]. The new thread's conversation
    /// prefix comes from [`Thread::fork_from`]; capabilities come
    /// from one of two paths:
    ///
    /// - `reset_capabilities = false` (default): the fork inherits
    ///   the source's live bindings / scope / config / tool_surface
    ///   verbatim. "Continue this conversation with all my
    ///   sudo-remembers and current settings intact."
    /// - `reset_capabilities = true`: bindings, scope, config, and
    ///   tool_surface are re-derived from the pod's current
    ///   `thread_defaults` — equivalent to a fresh thread carrying
    ///   the source's conversation prefix. Use this to pick up
    ///   pod-config changes that post-date the source (newly-added
    ///   MCP hosts, sandbox bindings, cap widenings).
    ///
    /// Returns the new thread id on success. Failures before
    /// insertion leave no partial registration: the source thread is
    /// untouched and no resource refcounts move.
    #[allow(clippy::too_many_arguments)]
    fn fork_task(
        &mut self,
        requester: Option<ConnId>,
        correlation_id: Option<String>,
        source_thread_id: &str,
        from_message_index: usize,
        reset_capabilities: bool,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<String, String> {
        let new_id = crate::runtime::thread::new_task_id();
        let mut task = {
            let source = self
                .tasks
                .get(source_thread_id)
                .ok_or_else(|| format!("unknown thread `{source_thread_id}`"))?;
            source.fork_from(new_id.clone(), from_message_index)?
        };
        let pod_id = task.pod_id.clone();
        if !self.pods.contains_key(&pod_id) {
            return Err(format!("unknown pod `{pod_id}`"));
        }
        if reset_capabilities {
            use crate::pod::resources::McpHostId;
            // Resolve bindings first — it's the failure-prone step
            // (shared MCP host not actually wired up, etc.) and we
            // want to bail before mutating `task`.
            let resolved = {
                let pod = self
                    .pods
                    .get(&pod_id)
                    .ok_or_else(|| format!("unknown pod `{pod_id}`"))?;
                resolve_bindings_choice(pod, None, None)?
            };
            for name in &resolved.shared_host_names {
                if !self
                    .resources
                    .mcp_hosts
                    .contains_key(&McpHostId::shared(name))
                {
                    return Err(format!(
                        "shared MCP host `{name}` not configured on this server (pod `{pod_id}` allows it but no upstream is wired up)",
                    ));
                }
            }
            task.bindings = ThreadBindings {
                backend: resolved.backend_name.clone(),
                host_env: resolved.host_env.clone(),
                mcp_hosts: resolved.shared_host_names.clone(),
                tool_filter: None,
            };
            let pod = self.pods.get(&pod_id).expect("pod presence checked above");
            task.config = apply_config_override(base_thread_config_from_pod(pod), None);
            task.tool_surface = pod.config.thread_defaults.tool_surface.compose(None);
            task.scope = self.pod_thread_scope(&pod_id);
        }
        // Escalation is a live-channel pointer, not part of the
        // source's conversation prefix. The forking client owns the
        // new thread, so bind the channel to the forking conn (or
        // clear it if no conn is known). Applies to both the
        // inherit and reset paths — inheriting leaks the source's
        // stale `via_conn`; resetting starts from `None`.
        task.scope.escalation = match requester {
            Some(conn_id) => crate::permission::Escalation::Interactive { via_conn: conn_id },
            None => crate::permission::Escalation::None,
        };
        // Pre-validate the host_env binding before handing the task
        // off to `register_new_task` (which `.expect()`s that the
        // binding resolves). Relevant to the inherit path: a Named
        // binding can dangle if the entry was removed from the
        // pod's allow list after the source thread was created. The
        // reset path re-derived from `pod.allow` above, so this
        // loop is a no-op there.
        for binding in task.bindings.host_env.iter() {
            if self.resolve_binding(&pod_id, binding).is_none() {
                return Err(
                    "source thread's host_env binding no longer resolves in the pod's allow list \
                     (fork with `reset_capabilities: true` to re-derive from pod defaults)"
                        .into(),
                );
            }
        }

        let new_id = self.register_new_task(task, requester, correlation_id, pending_io);
        self.mark_dirty(&new_id);
        info!(
            new_thread_id = %new_id,
            source_thread_id = %source_thread_id,
            pod_id = %pod_id,
            from_message_index,
            "thread forked"
        );
        Ok(new_id)
    }

    /// Archive `thread_id`: remove the thread from every in-memory
    /// surface and move its JSON from `<pod>/threads/<id>.json` to
    /// `<pod>/.archived/threads/<id>.json`. Shares the disk-move +
    /// teardown path with `retention_sweep` so introspection tools
    /// that walk `<pod>/threads/` never see archived threads. When
    /// the thread is non-terminal, cancels any in-flight model/tool
    /// work first so Function-registry entries and resource users
    /// unwind cleanly before the thread is yanked from memory.
    /// Silent no-op when the thread is unknown.
    pub(crate) fn archive_thread(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let (terminal, pod_id) = match self.tasks.get(thread_id) {
            Some(task) => {
                let terminal = matches!(
                    task.public_state(),
                    ThreadStateLabel::Completed
                        | ThreadStateLabel::Failed
                        | ThreadStateLabel::Cancelled
                );
                (terminal, task.pod_id.clone())
            }
            None => return,
        };
        if !terminal {
            self.execute_cancel_thread(thread_id, pending_io);
        }
        self.sweep_thread(thread_id, &pod_id, RetentionAction::Archive);
    }

    /// Promote a `Failed` thread back to `Idle` so the user can send
    /// a follow-up message. Returns `Err` with a human-readable
    /// reason when the thread doesn't exist or isn't in `Failed`;
    /// the caller turns that into an `Error` wire reply.
    pub(crate) fn recover_thread(&mut self, thread_id: &str) -> Result<(), String> {
        let Some(task) = self.tasks.get_mut(thread_id) else {
            return Err("unknown thread".into());
        };
        let mut events = Vec::new();
        if !task.recover(&mut events) {
            return Err(format!(
                "thread is not in Failed state (current: {:?})",
                task.public_state()
            ));
        }
        let new_state = task.public_state();
        self.mark_dirty(thread_id);
        self.router.dispatch_events(thread_id, events);
        self.router
            .broadcast_task_list(ServerToClient::ThreadStateChanged {
                thread_id: thread_id.to_string(),
                state: new_state,
            });
        Ok(())
    }

    fn send_user_message(
        &mut self,
        thread_id: &str,
        text: String,
        attachments: Vec<whisper_agent_protocol::Attachment>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        self.mark_dirty(thread_id);
        // Post-restart: a loaded thread's host-env MCP entry may not
        // exist yet (the in-memory registry doesn't survive process
        // restart). ensure_host_env_provisioning is a no-op when
        // create_task already dispatched in this cycle or when the
        // thread has no host_env binding.
        self.ensure_host_env_provisioning(thread_id, pending_io);
        // If the thread was mid tool-call-cycle (AwaitingTools) when
        // this user message arrived, synthesize error-tagged
        // tool_results for every unresolved tool_use so the
        // conversation stays Anthropic-valid (`assistant[tool_use]`
        // must be followed by `tool_result`, not bare `user[text]`).
        // Then cancel the corresponding Function entries so late-
        // arriving I/O results are discarded at the scheduler layer
        // rather than re-integrated against a finalized conversation.
        let interrupted = {
            let mut events = Vec::new();
            let task = self.tasks.get_mut(thread_id).expect("task exists");
            let list = task.heal_to_idle("superseded by new user message", &mut events);
            self.router.dispatch_events(thread_id, events);
            list
        };
        for tool_use_id in &interrupted {
            if let Some(fn_id) = self.find_tool_function_for(thread_id, tool_use_id) {
                self.complete_function(
                    fn_id,
                    crate::functions::FunctionOutcome::Cancelled(
                        crate::functions::CancelReason::ExplicitCancel,
                    ),
                    pending_io,
                );
            }
        }
        // Derive title on the first user message.
        let title_new = {
            let task = self.tasks.get_mut(thread_id).expect("task exists");
            if task.title.is_none() {
                let t = derive_title(&text);
                task.title = Some(t.clone());
                Some(t)
            } else {
                None
            }
        };
        if let Some(title) = title_new {
            self.router
                .broadcast_task_list(ServerToClient::ThreadTitleUpdated {
                    thread_id: thread_id.to_string(),
                    title,
                });
        }

        // Phase 3d.ii: walk the thread's bindings against the registry to
        // figure out which resources aren't Ready yet. Anything not Ready
        // gets carried in `WaitingOnResources { needed }`; the scheduler
        // nudges the thread out via `clear_waiting_resource` as each
        // resource transitions Ready (sandbox or primary MCP).
        let pending_resources = self.pending_resources_for(thread_id);
        let new_state = {
            let task = self.tasks.get_mut(thread_id).expect("task exists");
            task.submit_user_message(text.clone(), attachments.clone(), pending_resources);
            task.public_state()
        };
        // Emit the user message to subscribers BEFORE the state change
        // so the conversation view lands in order: user message first,
        // then the state flips to Working for the model turn it kicks.
        self.router.broadcast_to_subscribers(
            thread_id,
            ServerToClient::ThreadUserMessage {
                thread_id: thread_id.to_string(),
                text,
                attachments,
            },
        );
        self.router
            .broadcast_task_list(ServerToClient::ThreadStateChanged {
                thread_id: thread_id.to_string(),
                state: new_state,
            });
    }

    /// Append a `Role::ToolResult` + text message to `thread_id`'s
    /// conversation and kick the model turn, same shape as
    /// [`Self::send_user_message`] but broadcasts a
    /// `ThreadToolResultMessage` event so clients can render it as
    /// tool output rather than user input. Used for async
    /// `dispatch_thread` callbacks (and any future server-injected
    /// tool output) — the originating `tool_use_id` has already been
    /// consumed by the synchronous ack, so the callback can't bind
    /// to it structurally; the role carries the intent and the webui
    /// parses the embedded XML envelope to reattach the payload to
    /// the right tool-call item.
    pub(super) fn send_tool_result_text(
        &mut self,
        thread_id: &str,
        text: String,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        self.mark_dirty(thread_id);
        self.ensure_host_env_provisioning(thread_id, pending_io);
        // Deliberately don't derive a title from this text — a
        // machine-rendered notification isn't a useful thread title.
        let pending_resources = self.pending_resources_for(thread_id);
        let new_state = {
            let task = self.tasks.get_mut(thread_id).expect("task exists");
            task.submit_tool_result_text(text.clone(), pending_resources);
            task.public_state()
        };
        self.router.broadcast_to_subscribers(
            thread_id,
            ServerToClient::ThreadToolResultMessage {
                thread_id: thread_id.to_string(),
                text,
            },
        );
        self.router
            .broadcast_task_list(ServerToClient::ThreadStateChanged {
                thread_id: thread_id.to_string(),
                state: new_state,
            });
    }

    /// Apply one per-thread I/O completion to its task and dispatch any
    /// resulting events. Resource-provisioning completions take a
    /// different path; see [`Self::apply_provision_completion`].
    fn apply_io_completion(
        &mut self,
        completion: IoCompletion,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let IoCompletion {
            thread_id,
            op_id,
            result,
            pod_update,
            scheduler_command,
            host_env_lost,
        } = completion;
        let mut events = Vec::new();

        // Builtin pod-modify tools produce a side effect when their disk
        // write succeeds. Apply it BEFORE the task sees the tool result
        // so subscribed clients observe PodConfigUpdated /
        // PodSystemPromptUpdated before the ToolCallEnd that caused it —
        // matches the "state-updates-first" ordering used by every other
        // broadcast pair.
        if let Some(update) = pod_update {
            self.apply_pod_update(&thread_id, update);
        }

        // Handle transport-lost signal from the MCP tool-call path.
        // Mark both the host env and its MCP entry Lost, broadcast
        // resource updates, and — since the session won't recover —
        // don't emit a teardown RPC (mark_host_env_lost drops the
        // cached handle itself). The thread sees the normal tool
        // error below and keeps running; the next thread to arrive
        // with the same (provider, spec) re-provisions via the
        // existing Errored-reset path in pre_register_host_env.
        if let Some(signal) = host_env_lost {
            let message = signal.message;
            if let Some(host_env_id) = signal.mcp_host_id.host_env_id() {
                if self
                    .resources
                    .mark_host_env_lost(&host_env_id, message.clone())
                {
                    self.emit_host_env_updated(&host_env_id);
                }
                if self.resources.mark_mcp_lost(&signal.mcp_host_id, message) {
                    self.emit_mcp_host_updated(&signal.mcp_host_id);
                }
                warn!(
                    %thread_id,
                    host_env_id = %host_env_id.0,
                    "host env marked Lost after transport failure"
                );
            }
        }
        if let Some(command) = scheduler_command {
            self.apply_scheduler_command(&thread_id, command, pending_io);
        }

        // Surface any IO-level error to the operator even if no client is subscribed.
        match &result {
            IoResult::ModelCall(Err(e)) => {
                warn!(%thread_id, op_id, error = %e, "model call failed");
            }
            IoResult::ToolCall {
                tool_use_id,
                result: Err(e),
            } => {
                warn!(%thread_id, %tool_use_id, error = %e, "tool call failed");
            }
            _ => {}
        }

        // Build per-tool-name annotation map so the task's approval policy can consult it.
        let annotations = self.annotations_for(&thread_id);
        // Snapshot the tool-call outcome before handing the result to
        // the thread (which consumes it by move). The snapshot lets us
        // close out the matching `ActiveFunctionEntry` after the
        // thread has integrated the result.
        let tool_result_snapshot: Option<(
            String,
            Result<crate::tools::mcp::CallToolResult, String>,
        )> = match &result {
            IoResult::ToolCall {
                tool_use_id,
                result: tc_result,
            } => Some((tool_use_id.clone(), tc_result.clone())),
            _ => None,
        };
        if let Some(task) = self.tasks.get_mut(&thread_id) {
            task.apply_io_result(op_id, result, &annotations, &mut events);
        } else {
            warn!(%thread_id, op_id, "io completion for unknown task");
            return;
        }
        self.mark_dirty(&thread_id);
        self.router.dispatch_events(&thread_id, events);
        self.teardown_host_env_if_terminal(&thread_id);
        // Close out the tool-call Function entry, if any. No-op when
        // the completion is a model call (ModelCall isn't a Function)
        // or a dispatch_thread intercept (no Function registered at
        // this layer — CreateThread Function lands in Phase 5).
        if let Some((tool_use_id, tc_result)) = tool_result_snapshot {
            self.complete_tool_function(&thread_id, &tool_use_id, &tc_result, pending_io);
        }
    }

    /// Apply a resource-provisioning completion. Updates the registry
    /// (sandbox handle + primary MCP session + tool catalog), broadcasts
    /// resource events, and nudges every thread parked in
    /// `WaitingOnResources` whose `needed` set contained one of the now-
    /// Ready ids. Newly-Ready threads get stepped.
    fn apply_provision_completion(
        &mut self,
        completion: ProvisionCompletion,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let ProvisionCompletion { thread_id, result } = completion;
        // Per-id guard: clear using the completion's host_env_id. The
        // thread_id rides along only so the scheduler can tie errors
        // back to who originally dispatched this provision; the guard
        // itself is keyed per deduped id because dedup-joining threads
        // share one in-flight provision.
        let completed_id = match &result {
            ProvisionResult::HostEnvMcpReady { host_env_id, .. } => host_env_id.clone(),
            ProvisionResult::HostEnvMcpFailed { host_env_id, .. } => host_env_id.clone(),
        };
        self.provisioning_in_flight.remove(&completed_id);
        // Opportunistic reachability: if the provision reached (or
        // went past) the HostEnv phase, the daemon responded, so the
        // provider is live; if it failed at the HostEnv phase, the
        // daemon is unreachable or rejecting control-plane requests.
        // Later phase failures (McpConnect, ListTools) imply the
        // daemon *is* reachable — the sandbox came up — so we still
        // report Reachable. We look up the provider name through the
        // host_env_id's entry in `resources`.
        let provider_name = self
            .resources
            .host_envs
            .get(&completed_id)
            .map(|e| e.provider.clone());
        if let Some(provider_name) = provider_name {
            let now = chrono::Utc::now();
            let changed = match &result {
                ProvisionResult::HostEnvMcpReady { .. } => {
                    self.host_env_registry.mark_reachable(&provider_name, now)
                }
                ProvisionResult::HostEnvMcpFailed {
                    phase: ProvisionPhase::HostEnv,
                    message,
                    ..
                } => self
                    .host_env_registry
                    .mark_unreachable(&provider_name, now, message.clone()),
                ProvisionResult::HostEnvMcpFailed { .. } => {
                    self.host_env_registry.mark_reachable(&provider_name, now)
                }
            };
            if changed && let Some(info) = self.host_env_provider_info(&provider_name) {
                self.router.broadcast_resource(
                    whisper_agent_protocol::ServerToClient::HostEnvProviderUpdated {
                        correlation_id: None,
                        provider: info,
                    },
                );
            }
        }
        match result {
            ProvisionResult::HostEnvMcpReady {
                host_env_id,
                host_env_handle,
                mcp_url,
                mcp_token,
                mcp_session,
                tools,
            } => {
                let mcp_id = McpHostId::for_host_env(&host_env_id);
                // 1. Host env — attach the handle (or recognize the
                //    dedup race and tear ours down).
                let outcome = self.resources.complete_host_env_provisioning(
                    &host_env_id,
                    Some(mcp_url.clone()),
                    mcp_token,
                    Some(host_env_handle),
                );
                match outcome {
                    CompleteHostEnvOutcome::Completed => {
                        self.emit_host_env_updated(&host_env_id);
                    }
                    CompleteHostEnvOutcome::AlreadyCompleted { unused_handle }
                    | CompleteHostEnvOutcome::NotPreRegistered { unused_handle } => {
                        self.emit_host_env_updated(&host_env_id);
                        if let Some(mut h) = unused_handle {
                            tokio::spawn(async move {
                                if let Err(e) = h.teardown().await {
                                    warn!(error = %e, "teardown of dedup-loser host env failed");
                                }
                            });
                        }
                    }
                }

                // 2. Host-env MCP host — attach session + tools.
                let attached = self
                    .resources
                    .complete_host_env_mcp(&mcp_id, mcp_url, mcp_session);
                if !attached {
                    warn!(%thread_id, "host-env mcp already attached or missing — completion dropped");
                }
                let annotations: HashMap<String, ToolAnnotations> = tools
                    .iter()
                    .map(|t| (t.name.clone(), t.annotations.clone()))
                    .collect();
                self.resources
                    .populate_mcp_tools(&mcp_id, tools, annotations);
                self.emit_mcp_host_updated(&mcp_id);

                // 3. Finalize the setup prefix for every thread bound
                //    to this MCP *whose remaining host-env MCPs are
                //    also Ready*. Threads that bind to multiple envs
                //    merge every env's tools into one manifest;
                //    pushing partway would freeze a tool list missing
                //    the slower envs' contributions.
                //    `finalize_setup_prefix` self-gates on
                //    `host_env_mcp_is_ready` and is idempotent (keyed
                //    on `Role::Tools` existence), so it's safe to
                //    call against every bound thread. Runs BEFORE
                //    `step_until_blocked` below — so no model call
                //    has fired yet, and the setup-prefix write is
                //    still free (nothing has been cached).
                let bound_to_mcp: Vec<String> = self
                    .resources
                    .mcp_hosts
                    .get(&mcp_id)
                    .map(|e| e.users.iter().cloned().collect())
                    .unwrap_or_default();
                // Snapshot the raw tool names now — the registry was
                // populated above and we'll need them for the
                // activation-message branch below (the setup-prefix
                // path handles already-seeded threads itself; this is
                // only for threads whose prefix already finalized).
                let fresh_tool_names: Vec<String> = self
                    .resources
                    .mcp_hosts
                    .get(&mcp_id)
                    .map(|e| e.tools.iter().map(|t| t.name.clone()).collect())
                    .unwrap_or_default();
                for tid in &bound_to_mcp {
                    let already_seeded = self
                        .tasks
                        .get(tid)
                        .map(|t| {
                            t.conversation
                                .messages()
                                .iter()
                                .any(|m| m.role == whisper_agent_protocol::Role::Tools)
                        })
                        .unwrap_or(false);
                    if already_seeded {
                        // Rare case: the thread already crossed its
                        // setup boundary but this MCP is arriving now
                        // (e.g. re-provision after a Lost state). The
                        // static Role::Tools is immutable per cache-
                        // coherence — emit an append-only activation
                        // message instead so the model sees the new
                        // names.
                        let prefix = self.prefix_for_bound_mcp(tid, &mcp_id);
                        let prefixed: Vec<String> = fresh_tool_names
                            .iter()
                            .map(|n| match &prefix {
                                Some(p) => format!("{p}_{n}"),
                                None => n.clone(),
                            })
                            .collect();
                        self.push_tool_activation_message(tid, &prefixed);
                    } else {
                        self.finalize_setup_prefix(tid);
                    }
                }

                // 4. Nudge any threads that were waiting on these
                //    resources. Dedup means multiple threads can share
                //    one host env + MCP pair.
                let mut newly_ready: Vec<String> = Vec::new();
                let resource_ids = [host_env_id.0.clone(), mcp_id.0.clone()];
                for id in &resource_ids {
                    for task in self.tasks.values_mut() {
                        if task.clear_waiting_resource(id) {
                            newly_ready.push(task.id.clone());
                        }
                    }
                }
                for id in newly_ready {
                    self.mark_dirty(&id);
                    self.step_until_blocked(&id, pending_io);
                }
            }
            ProvisionResult::HostEnvMcpFailed {
                phase,
                message,
                host_env_id,
            } => {
                warn!(%thread_id, phase = phase.as_str(), error = %message, "host env provisioning failed");
                let mcp_id = McpHostId::for_host_env(&host_env_id);
                // Mark both entries Errored so future threads observe
                // the failure and don't endlessly retry.
                self.resources.mark_mcp_errored(&mcp_id, message.clone());
                self.emit_mcp_host_updated(&mcp_id);
                if matches!(phase, ProvisionPhase::HostEnv) {
                    self.resources
                        .mark_host_env_errored(&host_env_id, message.clone());
                    self.emit_host_env_updated(&host_env_id);
                }
                // Fail the requesting thread.
                if let Some(task) = self.tasks.get_mut(&thread_id) {
                    task.fail(phase.as_str(), message);
                    let mut events = Vec::new();
                    events.push(crate::runtime::thread::ThreadEvent::Error {
                        message: format!(
                            "{}: {}",
                            phase.as_str(),
                            task.failure_detail().unwrap_or_default()
                        ),
                    });
                    self.mark_dirty(&thread_id);
                    self.router.dispatch_events(&thread_id, events);
                    self.teardown_host_env_if_terminal(&thread_id);
                    self.on_behavior_thread_terminal(&thread_id, pending_io);
                }
            }
        }
    }

    fn annotations_for(&self, thread_id: &str) -> HashMap<String, ToolAnnotations> {
        let mut out: HashMap<String, ToolAnnotations> = crate::tools::builtin_tools::annotations();
        for bound in self.bound_mcp_hosts(thread_id) {
            for (name, ann) in &bound.entry.annotations {
                // Annotations are keyed by the *public* tool name so
                // approval lookups match what the model called. For
                // host-env MCPs that's `{env_prefix}_{name}`; shared
                // MCPs and builtins use the bare name.
                let public_name = match bound.prefix {
                    Some(prefix) => format!("{prefix}_{name}"),
                    None => name.clone(),
                };
                // First-wins matches `route_tool`'s builtin-then-host
                // precedence, so the approval decision agrees with
                // where the call will actually land.
                out.entry(public_name).or_insert_with(|| ann.clone());
            }
        }
        out
    }

    /// Advance `thread_id`'s state machine until it pauses, pushing new I/O to
    /// `pending_io` as requested.
    fn step_until_blocked(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        self.mark_dirty(thread_id);
        loop {
            let mut events = Vec::new();
            let outcome = {
                let Some(task) = self.tasks.get_mut(thread_id) else {
                    return;
                };
                task.step(&mut self.next_op_id, &mut events)
            };
            self.router.dispatch_events(thread_id, events);

            match outcome {
                StepOutcome::DispatchIo(req) => {
                    match req {
                        req @ crate::runtime::thread::IoRequest::ToolCall { .. } => {
                            // Single-entry dispatch:
                            // `register_tool_function` handles the tool-
                            // pool alias check (dispatch_thread →
                            // Function::CreateThread), normal tool
                            // routing, admission, and future-pushing.
                            self.register_tool_function(thread_id, req, pending_io);
                        }
                        other => {
                            let fut =
                                io_dispatch::build_io_future(self, thread_id.to_string(), other);
                            pending_io.push(fut);
                        }
                    };
                }
                StepOutcome::Continue => continue,
                StepOutcome::Paused => {
                    // Idle turn boundary. If this thread has queued
                    // `<dispatched-thread-notification>` envelopes
                    // waiting for an idle parent (async dispatch
                    // follow-ups), inject the first one as a user
                    // message and re-enter the step loop — it will
                    // kick a fresh turn. Remaining follow-ups wait
                    // for the next boundary.
                    let has_followup = self
                        .tasks
                        .get(thread_id)
                        .map(|t| !t.pending_tool_result_followups.is_empty())
                        .unwrap_or(false);
                    if has_followup && let Some(task) = self.tasks.get_mut(thread_id) {
                        let notification = task.pending_tool_result_followups.remove(0);
                        self.mark_dirty(thread_id);
                        self.send_tool_result_text(thread_id, notification, pending_io);
                        continue;
                    }
                    break;
                }
            }
        }
        // Catch Completed/Failed/Cancelled transitions for behavior-spawned
        // threads. Idempotent (no-op when origin is None or state isn't
        // terminal). Covers every stepping path; CancelThread and the
        // failed-provisioning site don't go through here and call the
        // hook directly. Passes `pending_io` through so the hook can
        // re-fire on a queued QueueOne payload.
        self.on_behavior_thread_terminal(thread_id, pending_io);
        // If the thread was mid-compaction and has just reached
        // Completed, parse the summary and spawn a continuation.
        // No-op otherwise. Runs after the behavior hook so a
        // compacted-then-continued behavior thread records the
        // Completed outcome on its original thread first.
        self.finalize_pending_compaction(thread_id, pending_io);
        // Then check whether this (or a freshly spawned continuation)
        // has crossed its auto-compaction token threshold. The
        // already-compacted gate inside the hook keeps the just-
        // -finalized parent from retriggering.
        self.maybe_auto_compact(thread_id, pending_io);
        // Dispatched-child terminal-state fan-out: if this thread is
        // a child awaited by a `Function::CreateThread{ThreadTerminal}`,
        // fire that Function's terminal — the delivery tag on the
        // entry routes the final text back to the parent (sync
        // oneshot or async follow-up message). If this thread is a
        // parent and has just died, cancel any Functions still
        // targeting it as their `ThreadToolCall` caller.
        self.complete_functions_awaiting_thread(thread_id, pending_io);
        self.cascade_cancel_caller_gone(thread_id, pending_io);
    }

    /// If the task is in a terminal state, tear down its sandbox (if any).
    /// Drop `thread_id` from every resource user set its bindings refer to,
    /// and tear down the per-thread primary MCP host (its lifecycle is 1:1
    /// with the thread). Call this whenever a thread is being removed from
    /// `self.tasks` so resource refcounts stay honest — without it the GC
    /// pass would never see those resources as idle.
    ///
    /// Sandbox teardown is intentionally NOT done here. Sandboxes are
    /// shared across threads with matching specs; the GC pass is the
    /// thing that decides "no users left, idle long enough, tear it down."
    /// Eager teardown the moment a thread leaves would defeat dedup.
    fn release_thread_resources(
        &mut self,
        thread_id: &str,
        pod_id: &str,
        bindings: &whisper_agent_protocol::ThreadBindings,
    ) {
        for binding in &bindings.host_env {
            if let Some((provider, spec)) = self.resolve_binding(pod_id, binding) {
                let sid = HostEnvId::for_provider_spec(&provider, &spec);
                let _ = self.resources.release_host_env_user(&sid, thread_id);
                self.emit_host_env_updated(&sid);
                // The host-env MCP is scoped to the same (provider, spec)
                // dedup, so drop this thread from its user set too.
                let mcp_id = McpHostId::for_host_env(&sid);
                self.resources.remove_mcp_user(&mcp_id, thread_id);
                self.emit_mcp_host_updated(&mcp_id);
            }
        }
        let backend_id = BackendId::for_name(&bindings.backend);
        self.resources.remove_backend_user(&backend_id, thread_id);
        for shared in &bindings.mcp_hosts {
            let mid = McpHostId::shared(shared);
            self.resources.remove_mcp_user(&mid, thread_id);
            self.emit_mcp_host_updated(&mid);
        }
    }

    /// One retention sweep over behavior-spawned threads. Walks every
    /// in-memory thread with a `BehaviorOrigin`, resolves the owning
    /// behavior's `on_completion` policy, and if the thread is in a
    /// terminal state past the retention window, either archives or
    /// deletes it. Interactive threads (origin is None) are never
    /// swept — retention is a behavior-thread concern only.
    ///
    /// Archiving moves the JSON to `<pod>/.archived/threads/<id>.json`
    /// (preserves the audit trail; thread is gone from every wire
    /// surface but forensically accessible on disk). Deleting removes
    /// the JSON and the in-memory state entirely.
    ///
    /// Orphan threads (behavior was deleted) are left alone — the
    /// design explicitly preserves their audit trail.
    ///
    /// The sweep itself is synchronous: state mutation + broadcasts
    /// happen inline, and the filesystem ops are spawned as detached
    /// futures (fire-and-forget). A missed disk op on one iteration
    /// retries on the next sweep because the in-memory thread has
    /// already been removed and flush_dirty won't re-create the file.
    fn retention_sweep(&mut self) {
        use whisper_agent_protocol::RetentionPolicy;
        let now = chrono::Utc::now();

        let mut sweep: Vec<(String, String, RetentionAction)> = Vec::new();
        for (thread_id, task) in &self.tasks {
            let Some(origin) = task.origin.as_ref() else {
                continue; // interactive thread — never swept
            };
            // Only terminal threads are candidates.
            let terminal = matches!(
                task.public_state(),
                ThreadStateLabel::Completed
                    | ThreadStateLabel::Failed
                    | ThreadStateLabel::Cancelled
            );
            if !terminal {
                continue;
            }
            // Resolve the behavior — orphaned threads (behavior
            // deleted) are left alone.
            let Some(pod) = self.pods.get(&task.pod_id) else {
                continue;
            };
            let Some(behavior) = pod.behaviors.get(&origin.behavior_id) else {
                continue;
            };
            let Some(policy) = behavior.config.as_ref().map(|c| &c.on_completion) else {
                continue;
            };
            let days = match policy {
                RetentionPolicy::Keep => continue,
                RetentionPolicy::ArchiveAfterDays { days }
                | RetentionPolicy::DeleteAfterDays { days } => *days,
            };
            let age = now.signed_duration_since(task.last_active);
            if age < chrono::Duration::days(days as i64) {
                continue;
            }
            let action = match policy {
                RetentionPolicy::ArchiveAfterDays { .. } => RetentionAction::Archive,
                RetentionPolicy::DeleteAfterDays { .. } => RetentionAction::Delete,
                RetentionPolicy::Keep => unreachable!(),
            };
            sweep.push((thread_id.clone(), task.pod_id.clone(), action));
        }

        for (thread_id, pod_id, action) in sweep {
            self.sweep_thread(&thread_id, &pod_id, action);
        }
    }

    /// Inner helper: removes the thread from every in-memory surface
    /// (tasks map, pod membership, router subscriptions, resource user
    /// sets), broadcasts `ThreadArchived`, and spawns the disk op.
    /// Shared between `retention_sweep` (terminal behavior threads)
    /// and `archive_thread` (user-initiated archive of any thread,
    /// after it has been cancelled if non-terminal).
    fn sweep_thread(&mut self, thread_id: &str, pod_id: &str, action: RetentionAction) {
        let pod_dir = match self.pods.get(pod_id) {
            Some(pod) => pod.dir.clone(),
            None => return,
        };
        // Capture bindings BEFORE removing from self.tasks — we need
        // them to release resource users.
        let bindings = match self.tasks.get(thread_id) {
            Some(t) => t.bindings.clone(),
            None => return,
        };
        self.tasks.remove(thread_id);
        self.cancel_tokens.remove(thread_id);
        self.dirty.remove(thread_id);
        // Provisioning guard is now keyed per HostEnvId, not per thread
        // — shared across all threads that bind to the same deduped
        // host-env. Don't clear on thread removal; other threads may
        // still be riding the same provision.
        if let Some(pod) = self.pods.get_mut(pod_id) {
            pod.threads.remove(thread_id);
        }
        self.release_thread_resources(thread_id, pod_id, &bindings);
        self.router.drop_thread(thread_id);
        self.router
            .broadcast_task_list(ServerToClient::ThreadArchived {
                thread_id: thread_id.to_string(),
            });

        // Disk op is fire-and-forget. The in-memory removal is the
        // load-bearing effect; a failed disk op logs warn and gets
        // retried by the next sweep (the thread will no longer be in
        // self.tasks but the JSON still exists, so nothing references
        // it — it'll just sit on disk harmlessly until an operator
        // cleans it up).
        let tid = thread_id.to_string();
        tokio::spawn(async move {
            let result = match action {
                RetentionAction::Archive => archive_thread_json(&pod_dir, &tid).await,
                RetentionAction::Delete => delete_thread_json(&pod_dir, &tid).await,
            };
            if let Err(e) = result {
                warn!(thread_id = %tid, error = %e, "retention sweep disk op failed");
            }
        });
    }

    /// One GC sweep over the resource registry. Tears down idle Ready
    /// resources (sandboxes, per-thread MCP hosts) and removes terminal
    /// entries that have been retained long enough. Spawns async sandbox
    /// teardown futures and broadcasts `ResourceUpdated` /
    /// `ResourceDestroyed` events for every change.
    fn gc_tick(&mut self) {
        self.sweep_expired_oauth_flows();
        // Ground-truth "in use" sets. A thread's `bindings.host_env` +
        // `bindings.mcp_hosts` pin the corresponding registry entries
        // against reap for as long as the thread is live — regardless
        // of whether the registry's `users` refcount has drifted. A
        // thread parked on a slow model call or a dispatched child
        // touches nothing in the registry, so refcount-only idle
        // detection would race live work. Failed / Cancelled threads
        // don't count — their bindings are already released via
        // `teardown_host_env_if_terminal` and they're on their way
        // out anyway.
        let mut in_use_host_envs: HashSet<HostEnvId> = HashSet::new();
        let mut in_use_mcp_hosts: HashSet<McpHostId> = HashSet::new();
        for task in self.tasks.values() {
            if matches!(
                task.public_state(),
                ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
            ) {
                continue;
            }
            for binding in &task.bindings.host_env {
                if let Some((provider, spec)) = self.resolve_binding(&task.pod_id, binding) {
                    let hid = HostEnvId::for_provider_spec(&provider, &spec);
                    let mid = McpHostId::for_host_env(&hid);
                    in_use_host_envs.insert(hid);
                    in_use_mcp_hosts.insert(mid);
                }
            }
            for name in &task.bindings.mcp_hosts {
                in_use_mcp_hosts.insert(McpHostId::shared(name));
            }
        }
        let plan = self.resources.reap_idle(
            chrono::Utc::now(),
            chrono::Duration::seconds(IDLE_RESOURCE_SECS),
            chrono::Duration::seconds(TERMINAL_RETENTION_SECS),
            &in_use_host_envs,
            &in_use_mcp_hosts,
        );
        if plan.torn_down_host_envs.is_empty()
            && plan.torn_down_mcp_hosts.is_empty()
            && plan.destroyed_host_envs.is_empty()
            && plan.destroyed_mcp_hosts.is_empty()
        {
            return;
        }
        for (id, handle) in plan.torn_down_host_envs {
            info!(sandbox_id = %id.0, "GC: tearing down idle sandbox");
            self.emit_host_env_updated(&id);
            if let Some(mut h) = handle {
                let id_str = id.0.clone();
                tokio::spawn(async move {
                    if let Err(e) = h.teardown().await {
                        warn!(sandbox_id = %id_str, error = %e, "GC sandbox teardown failed");
                    }
                });
            }
        }
        for id in plan.torn_down_mcp_hosts {
            info!(mcp_id = %id.0, "GC: tearing down idle MCP host");
            self.emit_mcp_host_updated(&id);
        }
        for id in plan.destroyed_host_envs {
            info!(sandbox_id = %id.0, "GC: removing stale terminal sandbox entry");
            self.router
                .broadcast_resource(ServerToClient::ResourceDestroyed {
                    id: id.0,
                    kind: ResourceKind::HostEnv,
                });
        }
        for id in plan.destroyed_mcp_hosts {
            info!(mcp_id = %id.0, "GC: removing stale terminal MCP host entry");
            self.router
                .broadcast_resource(ServerToClient::ResourceDestroyed {
                    id: id.0,
                    kind: ResourceKind::McpHost,
                });
        }
    }

    fn teardown_host_env_if_terminal(&mut self, thread_id: &str) {
        // Completed is NOT terminal — the task can receive follow-up messages.
        // Only tear down on truly irreversible states.
        let is_terminal = self.tasks.get(thread_id).is_some_and(|t| {
            matches!(
                t.public_state(),
                ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
            )
        });
        if !is_terminal {
            return;
        }
        // Each bound host env (and its MCP) is shared across threads
        // with matching (provider, spec); drop this thread from every
        // bound entry's user set and tear down only those whose count
        // hits zero and aren't pinned. No-op for threads with zero
        // host_env bindings — nothing was ever provisioned.
        let host_env_ids = self.host_env_ids_for_thread(thread_id);
        for sandbox_id in host_env_ids {
            // Drop the thread's reservation on the host-env MCP too.
            let mcp_id = McpHostId::for_host_env(&sandbox_id);
            self.resources.remove_mcp_user(&mcp_id, thread_id);
            self.emit_mcp_host_updated(&mcp_id);
            let remaining = self.resources.release_host_env_user(&sandbox_id, thread_id);
            let pinned = self
                .resources
                .host_envs
                .get(&sandbox_id)
                .map(|e| e.pinned)
                .unwrap_or(false);
            if remaining == 0 && !pinned {
                let handle = self.resources.take_host_env_handle(&sandbox_id);
                self.resources.mark_host_env_torn_down(&sandbox_id);
                self.emit_host_env_updated(&sandbox_id);
                if let Some(mut handle) = handle {
                    let tid = thread_id.to_string();
                    tokio::spawn(async move {
                        if let Err(e) = handle.teardown().await {
                            warn!(thread_id = %tid, error = %e, "sandbox teardown failed");
                        }
                    });
                }
            } else {
                // Sandbox lives on for the other threads using it; the user
                // list changed so subscribers want a refresh.
                self.emit_host_env_updated(&sandbox_id);
            }
        }
    }
}

/// Pure decision logic for the behavior-authoring gate. Split out of
/// [`Scheduler::check_behavior_authoring`] so the three
/// `BehaviorOpsCap` branches + the "strictly narrower" subset-check
/// can be unit-tested without building a full Scheduler.
///
/// Rules (from `docs/design_permissions_rework.md`):
///   - `None` / `Read`: authoring denied outright.
///   - `AuthorNarrower`: the behavior's fire-time scope
///     (`pod.allow.narrow(behavior.scope)`) must be **strictly**
///     narrower than the caller's. Equal-scope writes are denied —
///     AuthorNarrower lets an author mint a less-privileged behavior,
///     not one at their own power level.
///   - `AuthorAny`: admits anything; the pod.allow narrowing at fire
///     time is the only remaining bound.
pub(crate) fn check_behavior_authoring_pure(
    caller_scope: &crate::permission::Scope,
    pod_ceiling: &crate::permission::Scope,
    filename: &str,
    declared_scope: &whisper_agent_protocol::BehaviorScope,
) -> Option<String> {
    use crate::permission::BehaviorOpsCap;

    let fire_scope = pod_ceiling.narrow(&declared_scope.as_scope_narrower());
    match caller_scope.behaviors {
        BehaviorOpsCap::None | BehaviorOpsCap::Read => Some(format!(
            "`{filename}`: authoring a behavior requires behaviors cap ≥ \
             author_narrower (have: {:?}). Ask for scope widening if \
             this action is needed.",
            caller_scope.behaviors
        )),
        BehaviorOpsCap::AuthorNarrower => {
            // `inner.narrow(&outer) == inner` iff `inner ⊆ outer`.
            let is_subset = fire_scope.narrow(caller_scope) == fire_scope;
            if !is_subset {
                return Some(format!(
                    "`{filename}`: author_narrower requires the behavior's \
                     effective scope to fit within the author's; one or more \
                     fields exceed the caller's scope."
                ));
            }
            if fire_scope == *caller_scope {
                return Some(format!(
                    "`{filename}`: author_narrower requires the behavior's \
                     effective scope to be strictly narrower than the \
                     author's; this write would produce a scope equal to \
                     the caller's."
                ));
            }
            None
        }
        BehaviorOpsCap::AuthorAny => None,
    }
}

/// Read a pod-relative prompt file for a creation-time system-prompt
/// override. Rejects any name containing path separators or a parent
/// component — the override is a filename inside the pod dir, not an
/// escape hatch. A missing or unreadable file falls back to the pod's
/// cached default with a warn-level log, matching the pod loader's
/// graceful-degrade policy for `thread_defaults.system_prompt_file`.
fn resolve_system_prompt_file(pod: &crate::pod::Pod, name: &str) -> String {
    // Pod-relative path: accept nested directories (e.g.
    // `behaviors/<id>/system_prompt.md`) but reject traversal, absolute
    // paths, and Windows-style separators. The `PodModifyCap` tier
    // admits writes across the same surface — allowing the loader to
    // read a file the cap would let the agent author in the first
    // place.
    if name.is_empty()
        || name.starts_with('/')
        || name.starts_with('\\')
        || name.contains('\\')
        || name
            .split('/')
            .any(|seg| seg.is_empty() || seg == "." || seg == "..")
    {
        warn!(
            pod_id = %pod.id,
            file = %name,
            "system_prompt File override name rejected (must be a pod-relative path with no \
             traversal) — falling back to pod default"
        );
        return pod.system_prompt.clone();
    }
    let path = pod.dir.join(name);
    match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => {
            warn!(
                pod_id = %pod.id,
                file = %name,
                error = %e,
                "system_prompt File override unreadable — falling back to pod default"
            );
            pod.system_prompt.clone()
        }
    }
}

// ---------- Run loop ----------

pub async fn run(
    mut scheduler: Scheduler,
    mut inbox: mpsc::UnboundedReceiver<SchedulerMsg>,
    mut stream_rx: mpsc::UnboundedReceiver<StreamUpdate>,
    mut bucket_task_rx: mpsc::UnboundedReceiver<buckets::BucketTaskUpdate>,
) {
    let mut pending_io: FuturesUnordered<SchedulerFuture> = FuturesUnordered::new();
    let mut gc_ticker = tokio::time::interval(Duration::from_secs(GC_TICK_SECS));
    let mut cron_ticker = tokio::time::interval(Duration::from_secs(CRON_TICK_SECS));
    let mut retention_ticker = tokio::time::interval(Duration::from_secs(RETENTION_TICK_SECS));
    let mut probe_ticker = tokio::time::interval(Duration::from_secs(HOST_ENV_PROBE_TICK_SECS));
    let mut oauth_refresh_ticker =
        tokio::time::interval(Duration::from_secs(OAUTH_REFRESH_TICK_SECS));
    // Skip the first immediate tick on all — `interval` fires at t=0 by default.
    gc_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    gc_ticker.tick().await;
    cron_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    cron_ticker.tick().await;
    retention_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    retention_ticker.tick().await;
    // Probe ticker differs: we *want* the first tick to fire immediately
    // so reachability transitions from Unknown to Reachable/Unreachable
    // as soon as the server is up, instead of staring at Unknown for
    // 30 s. MissedTickBehavior::Skip prevents thundering-herd bursts
    // after the process suspends and resumes (laptop lids etc.).
    probe_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Oauth refresh: skip the first tick (no point refreshing at
    // boot — freshly loaded tokens are, by definition, fresh) and
    // drop missed ticks instead of bursting. Suspended laptops can
    // otherwise fire a dozen refreshes at once on resume.
    oauth_refresh_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    oauth_refresh_ticker.tick().await;

    // Apply each cron behavior's CatchUp policy to its persisted
    // cursor. Runs once at boot, before any tick evaluates for real.
    // Does not fire anything — the first `cron_ticker.tick()` handles
    // that for the `One`/`All` cases via their natural flow.
    scheduler.evaluate_cron_catch_up();

    info!("scheduler starting");
    loop {
        tokio::select! {
            biased;
            maybe_input = inbox.recv() => {
                match maybe_input {
                    Some(input) => scheduler.apply_input(input, &mut pending_io),
                    None => {
                        info!("scheduler inbox closed — shutting down");
                        break;
                    }
                }
            }
            maybe_completion = next_completion(&mut pending_io), if !pending_io.is_empty() => {
                if let Some(completion) = maybe_completion {
                    match completion {
                        SchedulerCompletion::Io(io) => {
                            let thread_id = io.thread_id.clone();
                            scheduler.apply_io_completion(io, &mut pending_io);
                            scheduler.step_until_blocked(&thread_id, &mut pending_io);
                        }
                        SchedulerCompletion::Provision(prov) => {
                            scheduler.apply_provision_completion(prov, &mut pending_io);
                        }
                        SchedulerCompletion::Probe(probe) => {
                            scheduler.apply_probe_completion(probe);
                        }
                        SchedulerCompletion::SharedMcp(done) => {
                            scheduler.apply_shared_mcp_completion(done);
                        }
                        SchedulerCompletion::OauthStart(done) => {
                            scheduler.apply_oauth_start_completion(done);
                        }
                        SchedulerCompletion::OauthComplete(done) => {
                            scheduler.apply_oauth_complete_completion(done);
                        }
                        SchedulerCompletion::OauthRefresh(done) => {
                            scheduler.apply_oauth_refresh_completion(done);
                        }
                        SchedulerCompletion::SudoInner(done) => {
                            scheduler.apply_sudo_inner_completion(done, &mut pending_io);
                        }
                    }
                }
            }
            Some(update) = stream_rx.recv() => {
                // Fire-and-forget fan-out: the router handles "no
                // subscribers" gracefully (drops the event), so an
                // unsubscribed thread's deltas cost us nothing beyond
                // the channel push.
                scheduler.router.broadcast_to_subscribers(&update.thread_id, update.event);
            }
            Some(update) = bucket_task_rx.recv() => {
                scheduler.apply_bucket_task_update(update).await;
            }
            _ = gc_ticker.tick() => {
                scheduler.gc_tick();
            }
            _ = cron_ticker.tick() => {
                scheduler.fire_due_cron_behaviors(&mut pending_io);
            }
            _ = retention_ticker.tick() => {
                scheduler.retention_sweep();
            }
            _ = probe_ticker.tick() => {
                pending_io.extend(scheduler.spawn_reachability_probes());
            }
            _ = oauth_refresh_ticker.tick() => {
                pending_io.extend(scheduler.spawn_oauth_refresh_futures());
            }
        }
        scheduler.flush_dirty().await;
    }

    // Drain pending I/O on shutdown — give it a grace period so in-flight HTTP can
    // complete cleanly, but don't block shutdown forever.
    info!(pending = pending_io.len(), "scheduler draining pending I/O");
    let _ = tokio::time::timeout(Duration::from_secs(5), async {
        while pending_io.next().await.is_some() {}
    })
    .await;
}

async fn next_completion(
    pending: &mut FuturesUnordered<SchedulerFuture>,
) -> Option<SchedulerCompletion> {
    pending.next().await
}

/// Best-effort shared-MCP-host connect at startup. Success populates
/// the registry as `Ready` with tools listed; failure registers the
/// entry as `Errored` with the connect message so the operator can
/// see the error in the WebUI list without digging in server logs.
/// `source` is included in logs to distinguish catalog vs CLI-overlay
/// entries.
async fn connect_shared_mcp_on_boot(
    resources: &mut ResourceRegistry,
    name: &str,
    url: &str,
    bearer: Option<String>,
    source: &str,
) {
    info!(name = %name, url = %url, %source, "connecting to shared MCP host");
    let session = match McpSession::connect(url, bearer).await {
        Ok(s) => Arc::new(s),
        Err(e) => {
            warn!(
                name = %name,
                url = %url,
                error = %e,
                "shared MCP connect failed at startup; entry registered as Errored"
            );
            resources.insert_shared_mcp_host_errored(
                name.to_string(),
                url.to_string(),
                format!("connect: {e}"),
            );
            return;
        }
    };
    let tools = match session.list_tools().await {
        Ok(t) => t,
        Err(e) => {
            warn!(
                name = %name,
                url = %url,
                error = %e,
                "shared MCP tools/list failed at startup; entry registered as Errored"
            );
            resources.insert_shared_mcp_host_errored(
                name.to_string(),
                url.to_string(),
                format!("tools/list: {e}"),
            );
            return;
        }
    };
    let id = resources.insert_shared_mcp_host(name.to_string(), url.to_string(), session);
    let annotations: HashMap<String, ToolAnnotations> = tools
        .iter()
        .map(|t| (t.name.clone(), t.annotations.clone()))
        .collect();
    resources.populate_mcp_tools(&id, tools, annotations);
}

#[cfg(test)]
mod authoring_gate_tests {
    use super::*;
    use crate::permission::{BehaviorOpsCap, PodModifyCap, Scope, SetOrAll};
    use whisper_agent_protocol::{BehaviorScope, BehaviorScopeCaps};

    fn pod_ceiling_allow_all() -> Scope {
        Scope::allow_all()
    }

    /// A caller scope with full resources but a specific `behaviors`
    /// cap and `pod_modify` narrowed to `Content` (typical interactive
    /// authoring baseline).
    fn caller_with_behaviors(cap: BehaviorOpsCap) -> Scope {
        let mut s = Scope::allow_all();
        s.behaviors = cap;
        s.pod_modify = PodModifyCap::Content;
        s
    }

    #[test]
    fn none_cap_denied() {
        let r = check_behavior_authoring_pure(
            &caller_with_behaviors(BehaviorOpsCap::None),
            &pod_ceiling_allow_all(),
            "behaviors/x/behavior.toml",
            &BehaviorScope::default(),
        );
        assert!(r.is_some());
        assert!(r.unwrap().contains("author_narrower"));
    }

    #[test]
    fn read_cap_denied() {
        // Read admits list/read but never authoring.
        let r = check_behavior_authoring_pure(
            &caller_with_behaviors(BehaviorOpsCap::Read),
            &pod_ceiling_allow_all(),
            "behaviors/x/behavior.toml",
            &BehaviorScope::default(),
        );
        assert!(r.is_some());
    }

    #[test]
    fn author_any_admits_any_scope() {
        // AuthorAny == "within pod.allow", which narrowing enforces
        // automatically — the gate itself imposes no extra check.
        let r = check_behavior_authoring_pure(
            &caller_with_behaviors(BehaviorOpsCap::AuthorAny),
            &pod_ceiling_allow_all(),
            "behaviors/x/behavior.toml",
            &BehaviorScope::default(),
        );
        assert!(r.is_none());
    }

    #[test]
    fn author_narrower_denied_for_equal_scope() {
        // Caller has behaviors=AuthorNarrower (narrower than pod ceiling);
        // declared narrows behaviors cap to match exactly, every other
        // field defaults. Fire-time scope matches the caller on every
        // dimension → strictly-narrower rule rejects the equal case.
        let mut caller = Scope::allow_all();
        caller.behaviors = BehaviorOpsCap::AuthorNarrower;
        let declared = BehaviorScope {
            caps: BehaviorScopeCaps {
                behaviors: Some(BehaviorOpsCap::AuthorNarrower),
                ..Default::default()
            },
            ..Default::default()
        };
        let r = check_behavior_authoring_pure(
            &caller,
            &pod_ceiling_allow_all(),
            "behaviors/x/behavior.toml",
            &declared,
        );
        assert!(r.is_some());
        assert!(r.unwrap().contains("strictly narrower"));
    }

    #[test]
    fn author_narrower_denied_when_fire_scope_exceeds_caller() {
        // Caller narrower than pod.allow on host_envs; behavior declares
        // no `[scope]` block, so fire_scope inherits the pod ceiling —
        // wider than the caller on host_envs AND on behaviors cap
        // (pod's AuthorAny vs caller's AuthorNarrower). Subset fails.
        let mut caller = Scope::allow_all();
        caller.behaviors = BehaviorOpsCap::AuthorNarrower;
        caller.host_envs = SetOrAll::only(["narrow".to_string()]);
        let r = check_behavior_authoring_pure(
            &caller,
            &pod_ceiling_allow_all(),
            "behaviors/x/behavior.toml",
            &BehaviorScope::default(),
        );
        assert!(r.is_some());
        assert!(r.unwrap().contains("fit within the author"));
    }

    #[test]
    fn author_narrower_admits_strict_subset() {
        // Caller has host_envs ∈ {a, b}; behavior restricts to {a} AND
        // narrows behaviors cap to match caller. Fire scope ⊆ caller
        // and is strictly smaller on host_envs.
        let mut caller = Scope::allow_all();
        caller.behaviors = BehaviorOpsCap::AuthorNarrower;
        caller.host_envs = SetOrAll::only(["a".to_string(), "b".to_string()]);
        let declared = BehaviorScope {
            host_envs: Some(vec!["a".to_string()]),
            caps: BehaviorScopeCaps {
                behaviors: Some(BehaviorOpsCap::AuthorNarrower),
                ..Default::default()
            },
            ..Default::default()
        };
        let r = check_behavior_authoring_pure(
            &caller,
            &pod_ceiling_allow_all(),
            "behaviors/x/behavior.toml",
            &declared,
        );
        assert!(r.is_none(), "expected admission, got: {r:?}");
    }

    #[test]
    fn author_narrower_admits_when_a_cap_is_strictly_less() {
        // Caller = pod.allow on every field (including behaviors=AuthorAny
        // so fire_scope doesn't inherit wider). Declared narrows
        // pod_modify to None — strictly less on that one dimension.
        let caller = Scope::allow_all();
        // Leave caller.behaviors at AuthorAny so no cap mismatch.
        let declared = BehaviorScope {
            caps: BehaviorScopeCaps {
                pod_modify: Some(PodModifyCap::None),
                ..Default::default()
            },
            ..Default::default()
        };
        // Flip behaviors cap to AuthorNarrower on a clone of caller to
        // activate the AuthorNarrower branch.
        let mut author = caller.clone();
        author.behaviors = BehaviorOpsCap::AuthorNarrower;
        // But we need the author's scope to *contain* fire_scope — which
        // it does because fire_scope only narrows pod_modify further,
        // and behaviors=AuthorNarrower ≤ AuthorAny (the inherited default).
        // Wait: fire_scope.behaviors = AuthorAny (inherited) > author's
        // AuthorNarrower. Same inheritance trap as above — declared must
        // match the author's behaviors cap for subset to hold.
        let declared = BehaviorScope {
            caps: BehaviorScopeCaps {
                pod_modify: Some(PodModifyCap::None),
                behaviors: Some(BehaviorOpsCap::AuthorNarrower),
                ..Default::default()
            },
            ..declared
        };
        let r = check_behavior_authoring_pure(
            &author,
            &pod_ceiling_allow_all(),
            "behaviors/x/behavior.toml",
            &declared,
        );
        assert!(r.is_none(), "expected admission, got: {r:?}");
    }

    #[test]
    fn author_narrower_inheriting_pod_ceiling_on_behaviors_is_denied() {
        // Documents the inheritance trap: an AuthorNarrower caller who
        // forgets to narrow `[scope.caps.behaviors]` in their behavior
        // TOML gets denied because the fire-time `behaviors` cap
        // inherits the pod ceiling (usually AuthorAny), which exceeds
        // the author's cap. This is intentional — authors must be
        // explicit to avoid minting behaviors more privileged than
        // themselves. The denial message surfaces the "fit within"
        // subset failure so the author can diagnose.
        let mut caller = Scope::allow_all();
        caller.behaviors = BehaviorOpsCap::AuthorNarrower;
        let declared = BehaviorScope::default();
        let r = check_behavior_authoring_pure(
            &caller,
            &pod_ceiling_allow_all(),
            "behaviors/x/behavior.toml",
            &declared,
        );
        assert!(r.is_some());
        assert!(r.unwrap().contains("fit within the author"));
    }
}

#[cfg(test)]
mod resolve_system_prompt_file_tests {
    use super::resolve_system_prompt_file;
    use crate::pod::Pod;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use whisper_agent_protocol::PodConfig;

    fn sample_pod(dir: PathBuf, fallback: &str) -> Pod {
        // Minimal Pod fixture: only the fields the function touches
        // (id for log scope, dir for path resolution, system_prompt
        // for the fallback body) matter; the rest use `Pod::new`'s
        // defaults.
        let toml = r#"name = "test"
created_at = "2026-04-22T00:00:00Z"
[allow]
backends = ["anthropic"]
mcp_hosts = []
host_env = []
[thread_defaults]
backend = "anthropic"
model = "x"
system_prompt_file = "system_prompt.md"
max_tokens = 1000
max_turns = 10
"#;
        let cfg: PodConfig = crate::pod::parse_toml(toml).unwrap();
        Pod::new("test".into(), dir, cfg, toml.to_string(), fallback.into())
    }

    fn temp_pod_dir() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("wa-resolve-syspm-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn accepts_nested_pod_relative_path() {
        // Regression: prior to the fix, any name containing `/` was
        // rejected — so a behavior referencing
        // `behaviors/<id>/system_prompt.md` could never load its own
        // prompt. Verify we now read it off disk.
        let dir = temp_pod_dir();
        std::fs::create_dir_all(dir.join("behaviors/researcher")).unwrap();
        std::fs::write(
            dir.join("behaviors/researcher/system_prompt.md"),
            "researcher sys",
        )
        .unwrap();
        let pod = sample_pod(dir, "fallback");
        let out = resolve_system_prompt_file(&pod, "behaviors/researcher/system_prompt.md");
        assert_eq!(out, "researcher sys");
    }

    #[test]
    fn accepts_flat_filename() {
        // The existing flat-filename case must keep working.
        let dir = temp_pod_dir();
        std::fs::write(dir.join("custom.md"), "flat").unwrap();
        let pod = sample_pod(dir, "fallback");
        assert_eq!(resolve_system_prompt_file(&pod, "custom.md"), "flat");
    }

    #[test]
    fn rejects_traversal_absolute_and_empty() {
        let dir = temp_pod_dir();
        let pod = sample_pod(dir, "fallback-body");
        for bad in [
            "",
            "/etc/passwd",
            "../../etc/passwd",
            "behaviors/../escape",
            "a/./b.md",
            "behaviors//x.md",
            r"behaviors\researcher.md",
        ] {
            let out = resolve_system_prompt_file(&pod, bad);
            assert_eq!(
                out, "fallback-body",
                "name `{bad}` should have been rejected"
            );
        }
    }

    #[test]
    fn missing_file_falls_back_to_pod_default() {
        let dir = temp_pod_dir();
        let pod = sample_pod(dir, "fallback-default");
        let out = resolve_system_prompt_file(&pod, "behaviors/ghost/absent.md");
        assert_eq!(out, "fallback-default");
    }
}
