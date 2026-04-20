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
mod client_messages;
mod compaction;
mod config_updates;
mod dispatch;
mod functions;
mod retention;
mod thread_config;
mod triggers;

pub use self::thread_config::build_default_pod_config;

use self::bindings::resolve_bindings_choice;
use self::retention::{RetentionAction, archive_thread_json, delete_thread_json};
use self::thread_config::{apply_config_override, base_thread_config_from_pod};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use whisper_agent_protocol::{
    ClientToServer, HostEnvBinding, Message, ResourceKind, ServerToClient, ThreadBindings,
    ThreadBindingsPatch, ThreadBindingsRequest, ThreadConfigOverride, ThreadStateLabel,
};

use crate::pod::persist::{LoadedState, Persister};
use crate::pod::resources::{
    BackendId, CompleteHostEnvOutcome, HostEnvId, McpHostId, ResourceRegistry,
};
use crate::pod::{Pod, PodId};
use crate::providers::model::ModelProvider;
use crate::runtime::audit::AuditLog;
use crate::runtime::io_dispatch::{
    self, IoCompletion, ProvisionCompletion, ProvisionPhase, ProvisionResult, SchedulerCompletion,
    SchedulerFuture,
};
use crate::runtime::thread::{
    IoResult, OpId, StepOutcome, Thread, ThreadInternalState, derive_title, new_task_id,
};
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
    prefix: Option<&'a str>,
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
}

/// Entry in the scheduler's shared-MCP-host catalog. Configured at server
/// start; one connection per host shared across all tasks that opt in via
/// `ThreadConfig.shared_mcp_hosts`.
#[derive(Debug, Clone)]
pub struct SharedHostConfig {
    /// Stable name; what `ThreadConfig.shared_mcp_hosts` references.
    pub name: String,
    /// MCP endpoint URL (typically `http://127.0.0.1:9830/mcp`).
    pub url: String,
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
    /// Fallback when a task doesn't specify a backend. Must be a key in `backends`.
    default_backend: String,
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

    tasks: HashMap<String, Thread>,
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

    /// In-flight Function registry. Populated synchronously by
    /// `register_function`; each entry is removed when its Function
    /// terminates (success / error / cancel). Non-persistent.
    ///
    /// Phase 2 wires only `Function::CancelThread` through this registry;
    /// later commits migrate the rest.
    active_functions: HashMap<crate::functions::FunctionId, functions::ActiveFunctionEntry>,
    /// Monotonic counter for assigning `FunctionId`s at registration time.
    next_function_id: crate::functions::FunctionId,
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
        default_backend: String,
        audit: AuditLog,
        host_env_registry: crate::tools::sandbox::HostEnvRegistry,
        host_env_catalog: crate::tools::host_env_catalog::CatalogStore,
        shared_host_configs: Vec<SharedHostConfig>,
    ) -> anyhow::Result<(Self, mpsc::UnboundedReceiver<StreamUpdate>)> {
        assert!(
            backends.contains_key(&default_backend),
            "default_backend must be in backends"
        );

        let mut resources = ResourceRegistry::new();
        for (name, entry) in &backends {
            resources.insert_backend(
                name.clone(),
                entry.kind.clone(),
                entry.default_model.clone(),
            );
        }

        for cfg in shared_host_configs {
            info!(name = %cfg.name, url = %cfg.url, "connecting to shared MCP host");
            // Shared hosts use anonymous MCP access today; adding auth
            // here is a separate change with its own config shape
            // (they're long-lived endpoints, not per-sandbox).
            let session = Arc::new(McpSession::connect(&cfg.url, None).await.map_err(|e| {
                anyhow::anyhow!("shared MCP host `{}` ({}): {e}", cfg.name, cfg.url)
            })?);
            // Phase 3c: list tools at startup so per-thread routing can
            // walk the registry without a per-thread fan-out. Failure
            // here is a startup failure — a misbehaving shared host that
            // can't list tools is the same kind of fatal misconfiguration
            // as one that can't handshake.
            let tools = session
                .list_tools()
                .await
                .map_err(|e| anyhow::anyhow!("list_tools on shared MCP `{}`: {e}", cfg.name))?;
            let id = resources.insert_shared_mcp_host(cfg.name.clone(), cfg.url.clone(), session);
            let annotations: HashMap<String, ToolAnnotations> = tools
                .iter()
                .map(|t| (t.name.clone(), t.annotations.clone()))
                .collect();
            resources.populate_mcp_tools(&id, tools, annotations);
        }

        let default_pod_id = default_pod.id.clone();
        let mut pods = HashMap::new();
        pods.insert(default_pod_id.clone(), default_pod);

        let (stream_tx, stream_rx) = mpsc::unbounded_channel();
        Ok((
            Self {
                default_pod_id,
                pods,
                backends,
                default_backend,
                persister: None,
                host_env_registry,
                host_env_catalog,
                tasks: HashMap::new(),
                router: ThreadEventRouter::new(audit, host_id),
                resources,
                next_op_id: 1,
                dirty: HashSet::new(),
                dirty_behaviors: HashSet::new(),
                provisioning_in_flight: HashSet::new(),
                stream_tx,
                active_functions: HashMap::new(),
                next_function_id: 1,
            },
            stream_rx,
        ))
    }

    /// Clone of the stream-update sender. Dispatched I/O futures grab one
    /// of these at dispatch time and push interim events through it; the
    /// scheduler's `run` loop forwards anything the receiver gets to the
    /// router.
    pub(crate) fn stream_sender(&self) -> mpsc::UnboundedSender<StreamUpdate> {
        self.stream_tx.clone()
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

    /// Returns the backend name the thread is bound to, falling back to the
    /// server's default if the binding's `backend` field is empty. Does NOT
    /// validate existence.
    pub(crate) fn resolve_backend_name<'a>(&'a self, task: &'a Thread) -> &'a str {
        if task.bindings.backend.is_empty() {
            &self.default_backend
        } else {
            &task.bindings.backend
        }
    }

    // ---------- Read-only accessors used by `io_dispatch` ----------

    pub(crate) fn task(&self, thread_id: &str) -> Option<&Thread> {
        self.tasks.get(thread_id)
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
        let changed = match result {
            Ok(()) => self.host_env_registry.mark_reachable(&name, observed_at),
            Err(msg) => {
                warn!(provider = %name, error = %msg, "host-env provider probe failed");
                self.host_env_registry
                    .mark_unreachable(&name, observed_at, msg)
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

    /// Build the thread's effective tool catalog — the model-facing
    /// pool. Prepends the builtin pod-editing tools, then every host-env
    /// MCP's tools (prefixed `{env_name}_`), then each shared MCP host
    /// in pod-declared order (unprefixed). The prefix scheme keeps
    /// host-env tools addressable independently even when two envs
    /// advertise the same bare name — `c-dtop_write_file` vs
    /// `c-srv3_write_file` coexist on the wire. Builtins and shared
    /// MCPs live in a single flat namespace because they're already
    /// unique by construction (builtins are `pod_*`, shared MCPs are
    /// singletons per server).
    ///
    /// Tools whose scope disposition is `Deny` for this thread are
    /// omitted entirely — no point advertising a tool to the model that
    /// will be synthesize-denied on call. `Allow` and `AllowWithPrompt`
    /// both surface; the approval layer handles the latter at dispatch
    /// time. Scope evaluation uses the *prefixed* name so policies can
    /// target a specific env's tool without touching the others.
    pub(crate) fn tool_descriptors(&self, thread_id: &str) -> Vec<McpTool> {
        let scope_denies = |name: &str| -> bool {
            self.tasks
                .get(thread_id)
                .map(|t| !t.tools_scope.disposition(&name.to_string()).admits())
                .unwrap_or(false)
        };
        let mut out: Vec<McpTool> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for tool in crate::tools::builtin_tools::descriptors() {
            if scope_denies(&tool.name) {
                continue;
            }
            if seen.insert(tool.name.clone()) {
                out.push(tool);
            }
        }
        for bound in self.bound_mcp_hosts(thread_id) {
            for tool in &bound.entry.tools {
                let public_name = match bound.prefix {
                    Some(prefix) => format!("{prefix}_{}", tool.name),
                    None => tool.name.clone(),
                };
                if scope_denies(&public_name) {
                    continue;
                }
                if seen.insert(public_name.clone()) {
                    let mut renamed = tool.clone();
                    renamed.name = public_name;
                    out.push(renamed);
                }
            }
        }
        out
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
            let prefix = match binding {
                HostEnvBinding::Named { name } => Some(name.as_str()),
                HostEnvBinding::Inline { .. } => None,
            };
            out.push(BoundMcp { entry, prefix });
        }
        for name in &task.bindings.mcp_hosts {
            let id = McpHostId::shared(name);
            if let Some(entry) = self.resources.mcp_hosts.get(&id) {
                out.push(BoundMcp {
                    entry,
                    prefix: None,
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

    /// Apply a [`ThreadBindingsPatch`] to a running thread.
    ///
    /// Validates the patch against the pod's `[allow]` table (backend +
    /// shared MCP host names; sandbox specs are accepted as-is — see
    /// `resolve_bindings_choice` for why), swaps the bindings in place,
    /// adjusts every affected resource's user count, re-provisions the
    /// primary MCP if the sandbox changed, and appends a synthetic
    /// conversation note when the execution environment shifted. Threads
    /// parked in `WaitingOnResources` get a recomputed `needed` list and
    /// transition to `NeedsModelCall` if the new set is empty.
    pub(super) fn apply_rebind(
        &mut self,
        thread_id: &str,
        patch: ThreadBindingsPatch,
        correlation_id: Option<String>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), String> {
        // --- Phase 1: snapshot read-only state ---
        let (pod_id, old_bindings) = {
            let task = self
                .tasks
                .get(thread_id)
                .ok_or_else(|| format!("unknown thread `{thread_id}`"))?;
            (task.pod_id.clone(), task.bindings.clone())
        };
        let pod_allow = self
            .pods
            .get(&pod_id)
            .ok_or_else(|| format!("thread `{thread_id}` references unknown pod `{pod_id}`"))?
            .config
            .allow
            .clone();

        // --- Phase 2: compute new pieces (no mutation yet — we want
        //              validation failures to leave the thread untouched) ---
        let new_backend = patch
            .backend
            .clone()
            .unwrap_or_else(|| old_bindings.backend.clone());

        let new_shared_hosts: Vec<String> = match &patch.mcp_hosts {
            Some(list) => list.clone(),
            None => old_bindings
                .mcp_hosts
                .iter()
                .filter_map(|raw| raw.strip_prefix("mcp-shared-").map(String::from))
                .collect(),
        };

        let new_host_env_bindings: Vec<HostEnvBinding> = match &patch.host_env {
            None => old_bindings.host_env.clone(),
            Some(names) => names
                .iter()
                .map(|name| {
                    // Validate against the pod's allow list — cap is real.
                    if !pod_allow.host_env.iter().any(|nh| &nh.name == name) {
                        return Err(format!(
                            "host env `{name}` not in pod `{}`'s allow.host_env",
                            pod_id,
                        ));
                    }
                    Ok(HostEnvBinding::Named { name: name.clone() })
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        // Runtime ids derived per binding for refcounting + emit.
        // Bindings whose Named entry has gone missing (pod edited
        // mid-flight) resolve to `None` and get filtered — we skip the
        // user-count drop on those, matching the pre-refactor behavior.
        let old_host_env_ids: Vec<HostEnvId> = old_bindings
            .host_env
            .iter()
            .filter_map(|b| self.resolve_binding(&pod_id, b))
            .map(|(p, s)| HostEnvId::for_provider_spec(&p, &s))
            .collect();

        // --- Phase 3: validate against pod allowlist ---
        if !new_backend.is_empty() && !pod_allow.backends.iter().any(|b| b == &new_backend) {
            return Err(format!(
                "backend `{}` not in pod `{}`'s allow.backends ({})",
                new_backend,
                pod_id,
                pod_allow.backends.join(", ")
            ));
        }
        for name in &new_shared_hosts {
            if !pod_allow.mcp_hosts.iter().any(|h| h == name) {
                return Err(format!(
                    "shared MCP host `{name}` not in pod `{}`'s allow.mcp_hosts ({})",
                    pod_id,
                    pod_allow.mcp_hosts.join(", "),
                ));
            }
            if !self
                .resources
                .mcp_hosts
                .contains_key(&McpHostId::shared(name))
            {
                return Err(format!(
                    "shared MCP host `{name}` not configured on this server",
                ));
            }
        }

        // --- Phase 4: figure out what actually changed ---
        let backend_changed = new_backend != old_bindings.backend;
        let host_env_changed = new_host_env_bindings != old_bindings.host_env;
        let old_shared: HashSet<String> = old_bindings
            .mcp_hosts
            .iter()
            .filter_map(|raw| raw.strip_prefix("mcp-shared-").map(String::from))
            .collect();
        let new_shared: HashSet<String> = new_shared_hosts.iter().cloned().collect();
        let mcp_hosts_changed = old_shared != new_shared;

        // --- Phase 5: refcount adjustments ---
        if backend_changed {
            let old_resolved: &str = if old_bindings.backend.is_empty() {
                &self.default_backend
            } else {
                &old_bindings.backend
            };
            let new_resolved: &str = if new_backend.is_empty() {
                &self.default_backend
            } else {
                &new_backend
            };
            let old_id = BackendId::for_name(old_resolved);
            let new_id = BackendId::for_name(new_resolved);
            self.resources.remove_backend_user(&old_id, thread_id);
            self.resources.add_backend_user(&new_id, thread_id);
            self.emit_backend_updated(&old_id);
            self.emit_backend_updated(&new_id);
        }

        if host_env_changed {
            // Diff old vs new host-env ids: drop refcounts on removed
            // ids, pre-register each added id. Resolve NEW ids fresh
            // here rather than trusting the vec — bindings that fail to
            // resolve (spec missing from pod allow) would error out at
            // validation already, so resolution is expected to succeed.
            let new_host_env_ids: Vec<HostEnvId> = new_host_env_bindings
                .iter()
                .map(|b| {
                    self.resolve_binding(&pod_id, b)
                        .map(|(p, s)| HostEnvId::for_provider_spec(&p, &s))
                        .ok_or_else(|| "rebind: binding could not be resolved".to_string())
                })
                .collect::<Result<_, _>>()?;
            let old_set: HashSet<HostEnvId> = old_host_env_ids.iter().cloned().collect();
            let new_set: HashSet<HostEnvId> = new_host_env_ids.iter().cloned().collect();

            for old_id in old_host_env_ids.iter().filter(|id| !new_set.contains(id)) {
                // Release the old host env; tear it down if we were the
                // last user and the entry isn't pinned.
                let remaining = self.resources.release_host_env_user(old_id, thread_id);
                let pinned = self
                    .resources
                    .host_envs
                    .get(old_id)
                    .map(|e| e.pinned)
                    .unwrap_or(false);
                if remaining == 0 && !pinned {
                    let handle = self.resources.take_host_env_handle(old_id);
                    self.resources.mark_host_env_torn_down(old_id);
                    self.emit_host_env_updated(old_id);
                    if let Some(mut h) = handle {
                        let tid = thread_id.to_string();
                        tokio::spawn(async move {
                            if let Err(e) = h.teardown().await {
                                warn!(thread_id = %tid, error = %e, "host env teardown after rebind failed");
                            }
                        });
                    } else {
                        self.emit_host_env_updated(old_id);
                    }
                } else {
                    self.emit_host_env_updated(old_id);
                }
            }
            let added: Vec<HostEnvBinding> = new_host_env_bindings
                .iter()
                .filter(|b| {
                    let id = self
                        .resolve_binding(&pod_id, b)
                        .map(|(p, s)| HostEnvId::for_provider_spec(&p, &s));
                    id.map(|i| !old_set.contains(&i)).unwrap_or(false)
                })
                .cloned()
                .collect();
            for binding in added {
                let (provider, spec) = self
                    .resolve_binding(&pod_id, &binding)
                    .ok_or_else(|| "rebind: binding could not be resolved".to_string())?;
                let new_id = self
                    .resources
                    .pre_register_host_env(thread_id, provider, spec);
                self.emit_host_env_updated(&new_id);
            }
        }

        if mcp_hosts_changed {
            for name in old_shared.difference(&new_shared) {
                let id = McpHostId::shared(name);
                self.resources.remove_mcp_user(&id, thread_id);
                self.emit_mcp_host_updated(&id);
            }
            for name in new_shared.difference(&old_shared) {
                let id = McpHostId::shared(name);
                self.resources.add_mcp_user(&id, thread_id);
                self.emit_mcp_host_updated(&id);
            }
        }

        // --- Phase 6: host env change → re-provision its MCP ---
        if host_env_changed {
            // Mark every old host-env MCP entry that's no longer bound
            // as torn down. New bindings will get fresh MCP entries
            // when the next provision dispatches.
            let new_host_env_ids: HashSet<HostEnvId> = new_host_env_bindings
                .iter()
                .filter_map(|b| self.resolve_binding(&pod_id, b))
                .map(|(p, s)| HostEnvId::for_provider_spec(&p, &s))
                .collect();
            for old_id in old_host_env_ids
                .iter()
                .filter(|id| !new_host_env_ids.contains(id))
            {
                let mcp_id = McpHostId::for_host_env(old_id);
                self.resources.mark_mcp_torn_down(&mcp_id);
                self.emit_mcp_host_updated(&mcp_id);
            }
            // Drop the in-flight guard so ensure_host_env_provisioning
            // dispatches a fresh future. Swap bindings first so the
            // dispatcher sees the new binding.
        }

        // --- Phase 7: swap bindings + synthetic note + recompute waiting ---
        let new_bindings = ThreadBindings {
            backend: new_backend,
            host_env: new_host_env_bindings.clone(),
            mcp_hosts: new_shared_hosts.clone(),
            tool_filter: old_bindings.tool_filter.clone(),
        };

        let env_changed = host_env_changed || mcp_hosts_changed;
        let was_waiting = {
            let task = self.tasks.get_mut(thread_id).expect("checked in phase 1");
            task.bindings = new_bindings.clone();
            task.touch();
            if env_changed {
                task.conversation.push(Message::user_text(
                    "*Note: the execution environment changed at this point. \
                     Files, processes, and tool availability may differ from \
                     earlier in this conversation.*",
                ));
            }
            matches!(
                task.internal,
                ThreadInternalState::WaitingOnResources { .. }
            )
        };

        // Refresh the thread-prefix `Role::Tools` manifest in place.
        // The tool manifest is a cache-fingerprint-critical snapshot
        // of what the model sees as its capability surface; a
        // host-env or MCP-host rebind changes that surface, so the
        // snapshot has to move with it. Mutates `conversation[1]`
        // (or `[0]` if the thread somehow lacks a system-prefix) —
        // adapters will pick the new manifest up on the next model
        // call. Host-env provisioning is async; tools that arrive
        // after the MCP host finishes connecting will still reach
        // the model via the same wire path, just not this snapshot.
        if env_changed {
            let new_tool_blocks: Vec<whisper_agent_protocol::ContentBlock> = self
                .tool_descriptors(thread_id)
                .into_iter()
                .map(|t| whisper_agent_protocol::ContentBlock::ToolSchema {
                    name: t.name,
                    description: t.description,
                    input_schema: t.input_schema,
                })
                .collect();
            if let Some(task) = self.tasks.get_mut(thread_id)
                && let Some(tools_msg) = task.conversation.tools_message_mut()
            {
                tools_msg.content = new_tool_blocks;
            }
        }

        if host_env_changed {
            // Per-id guard doesn't need thread-scoped clearing on
            // rebind — new ids aren't in the set, so ensure will
            // dispatch for them; old ids either keep running (shared
            // with other threads) or their future completes and
            // clears the guard itself.
            self.ensure_host_env_provisioning(thread_id, pending_io);
        }

        let mut now_unblocked = false;
        if was_waiting {
            let new_needed = self.pending_resources_for(thread_id);
            if let Some(task) = self.tasks.get_mut(thread_id)
                && let ThreadInternalState::WaitingOnResources { needed } = &mut task.internal
            {
                *needed = new_needed;
                if needed.is_empty() {
                    task.internal = ThreadInternalState::NeedsModelCall;
                    now_unblocked = true;
                }
            }
        }

        // --- Phase 8: persist + broadcast ---
        self.mark_dirty(thread_id);
        self.router.broadcast_to_subscribers(
            thread_id,
            ServerToClient::ThreadBindingsChanged {
                thread_id: thread_id.to_string(),
                bindings: new_bindings,
                correlation_id,
            },
        );
        if now_unblocked {
            self.step_until_blocked(thread_id, pending_io);
        }
        Ok(())
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

            let backend_id = BackendId::for_name(self.resolve_backend_name(&task));
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
            SchedulerMsg::RegisterClient { conn_id, outbound } => {
                self.router.register_client(conn_id, outbound);
            }
            SchedulerMsg::UnregisterClient { conn_id } => {
                self.router.unregister_client(conn_id);
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
        }
    }

    /// Spawn a fresh thread under `pod_id`. `requester` identifies the
    /// connection that asked for it, if any; trigger-driven spawns
    /// (behaviors, cron, etc.) pass `None`. `origin` stamps behavior
    /// provenance on the thread; `None` for interactive work.
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
        // choice against the pod's `[allow]` cap.
        let resolved = resolve_bindings_choice(pod, bindings_request)?;
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

        // Snapshot the pod's tool-scope into the thread; mid-flight
        // edits to the pod's allow.tools won't retroactively change
        // the thread's active scope (rebind is the explicit path).
        let tools_scope = self
            .pods
            .get(&pod_id)
            .map(|p| p.config.allow.tools.clone())
            .unwrap_or_else(whisper_agent_protocol::AllowMap::allow_all);
        let mut task = Thread::new(
            thread_id.clone(),
            pod_id.clone(),
            config,
            bindings,
            tools_scope,
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
        // Tools: only if every resource advertising them is already
        // Ready. Otherwise wait for the provisioning-complete hook.
        if self.host_env_mcp_is_ready(thread_id) {
            self.push_tools_snapshot(thread_id);
        }
        // Memory index: read at thread-creation time and persist into
        // the conversation as a Role::System message right after the
        // tool manifest (or after System if Tools hasn't landed yet —
        // the tight `setup_prefix_end` logic routes Tools into its
        // slot on arrival, shuffling Memory to index 2). Idempotent:
        // skipped silently if a prior call already planted the block.
        self.push_memory_snapshot(thread_id);
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

    /// Insert a fresh `Role::Tools` manifest into the thread's
    /// conversation at the setup-prefix boundary (right after the
    /// `Role::System` message, before any body turns), built from
    /// the current [`Self::tool_descriptors`] view. Idempotent: if
    /// the thread already has a `Role::Tools` message this is a
    /// no-op — keeps the `HostEnvMcpCompleted` handler's per-user
    /// fan-out safe to call against every bound thread without
    /// per-thread bookkeeping.
    ///
    /// Insertion (rather than append) matters when a user message
    /// has already landed before provisioning completed — a
    /// `CreateThread`-with-initial_message while the host-env MCP
    /// is still spinning up leaves the conversation as
    /// `[System, User]`. Pushing Tools at the tail would yield
    /// `[System, User, Tools]`, which makes `setup_prefix_end()`
    /// return 1 and hides the tool manifest from the adapter's
    /// `tools:` wire field. Inserting at `setup_prefix_end()` keeps
    /// the invariant that the setup prefix stays contiguous at the
    /// head of the conversation.
    ///
    /// Because the insertion shifts later-message indices, it's a
    /// non-append mutation — subscribed clients' incremental views
    /// would drift from the server's conversation if they only saw
    /// streaming events. To keep them honest we follow the insert
    /// with a full `ThreadSnapshot` broadcast: subscribers rebuild
    /// their view from authoritative state in one step, so `Tools`
    /// becomes visible inline without waiting for a reload /
    /// re-subscribe. One-time cost — the setup prefix only
    /// finalizes once per thread.
    fn push_tools_snapshot(&mut self, thread_id: &str) {
        let already_has = self
            .tasks
            .get(thread_id)
            .map(|t| {
                t.conversation
                    .messages()
                    .iter()
                    .any(|m| m.role == whisper_agent_protocol::Role::Tools)
            })
            .unwrap_or(true);
        if already_has {
            return;
        }
        let tool_blocks: Vec<whisper_agent_protocol::ContentBlock> = self
            .tool_descriptors(thread_id)
            .into_iter()
            .map(|t| whisper_agent_protocol::ContentBlock::ToolSchema {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
            })
            .collect();
        let snapshot = if let Some(task) = self.tasks.get_mut(thread_id) {
            let at = task.conversation.setup_prefix_end();
            task.conversation.insert(
                at,
                whisper_agent_protocol::Message::tools_manifest(tool_blocks),
            );
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

    /// Insert the memory-index snapshot (a `Role::System` message
    /// rendered from `<pod_dir>/memory/MEMORY.md` + a fresh timestamp)
    /// into the thread's conversation at the setup-prefix boundary.
    /// Idempotent: no-op if a `Role::System` message is already
    /// present at that position (meaning the snapshot has already
    /// been planted by a prior call — `seed_thread_setup`,
    /// `HostEnvMcpCompleted` fan-out, compaction finalize, etc).
    ///
    /// Lives at the setup-prefix boundary (`setup_prefix_end()`), so
    /// before `Role::Tools` lands the memory slot is index 1 (just
    /// after System); after Tools arrives, the tools insert at index
    /// 1 shifts Memory to index 2. Ordering `[System, Tools, Memory,
    /// User, ...]` is the final steady state regardless of when each
    /// piece arrived.
    ///
    /// The block is persisted into the log like any other setup-frame
    /// message — same logic as why System prompt + Tools manifest are
    /// persisted. The log is the faithful record of what the model
    /// saw; re-reading MEMORY.md on every send would break prompt-
    /// cache stability for no gain.
    fn push_memory_snapshot(&mut self, thread_id: &str) {
        let Some(task) = self.tasks.get(thread_id) else {
            return;
        };
        let at = task.conversation.setup_prefix_end();
        let already_has = task
            .conversation
            .messages()
            .get(at)
            .is_some_and(|m| m.role == whisper_agent_protocol::Role::System);
        if already_has {
            return;
        }
        let pod_dir = match self.pods.get(&task.pod_id) {
            Some(p) => p.dir.clone(),
            None => return,
        };
        let block = crate::runtime::memory_snapshot::build_block(&pod_dir, chrono::Utc::now());
        let snapshot = if let Some(task) = self.tasks.get_mut(thread_id) {
            let at = task.conversation.setup_prefix_end();
            task.conversation.insert(at, block);
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
    /// tail of [`Self::create_task`] but against bindings copied from
    /// the source thread rather than re-resolved from pod defaults +
    /// a patch. The new thread's [`ThreadBindings`] are a verbatim
    /// clone of the source's, so every backend / host_env / shared
    /// MCP registration follows from those fields directly.
    ///
    /// Returns the new thread id on success. Failures before
    /// insertion leave no partial registration: the source thread is
    /// untouched and no resource refcounts move.
    fn fork_task(
        &mut self,
        requester: Option<ConnId>,
        correlation_id: Option<String>,
        source_thread_id: &str,
        from_message_index: usize,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<String, String> {
        let new_id = crate::runtime::thread::new_task_id();
        let task = {
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
        // Pre-validate the host_env binding before handing the task
        // off to `register_new_task` (which `.expect()`s that the
        // binding resolves). A Named binding can dangle if the entry
        // was removed from the pod's allow list after the source
        // thread was created; fail cleanly without touching
        // resources or the task map.
        for binding in task.bindings.host_env.iter() {
            if self.resolve_binding(&pod_id, binding).is_none() {
                return Err(
                    "source thread's host_env binding no longer resolves in the pod's allow list"
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

    /// Archive `thread_id`: flip the persisted `archived` flag, touch
    /// last_active, mark dirty for the next flush, and broadcast
    /// `ThreadArchived` so clients can drop the thread off their
    /// sidebars. Silent no-op when the thread is unknown — matches
    /// the existing `ArchiveThread` dispatch semantics.
    pub(crate) fn archive_thread(&mut self, thread_id: &str) {
        let Some(task) = self.tasks.get_mut(thread_id) else {
            return;
        };
        task.archived = true;
        task.touch();
        self.mark_dirty(thread_id);
        self.router
            .broadcast_task_list(ServerToClient::ThreadArchived {
                thread_id: thread_id.to_string(),
            });
    }

    fn send_user_message(
        &mut self,
        thread_id: &str,
        text: String,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        self.mark_dirty(thread_id);
        // Post-restart: a loaded thread's host-env MCP entry may not
        // exist yet (the in-memory registry doesn't survive process
        // restart). ensure_host_env_provisioning is a no-op when
        // create_task already dispatched in this cycle or when the
        // thread has no host_env binding.
        self.ensure_host_env_provisioning(thread_id, pending_io);
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
            task.submit_user_message(text.clone(), pending_resources);
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

                // 3. Finalize the `Role::Tools` setup message for every
                //    thread bound to this MCP *whose remaining host-
                //    env MCPs are also Ready*. Threads that bind to
                //    multiple envs merge every env's tools into one
                //    manifest; pushing partway would freeze a tool
                //    list missing the slower envs' contributions.
                //    `host_env_mcp_is_ready` returns true both for
                //    "no host-env bindings" and "every binding is
                //    Ready." Runs BEFORE `step_until_blocked` below
                //    — so no model call has fired yet, and the
                //    setup-prefix write is still free (no cache-bust
                //    because nothing has been cached yet). Idempotent:
                //    `push_tools_snapshot` no-ops for threads that
                //    already have a `Role::Tools` message.
                let bound_to_mcp: Vec<String> = self
                    .resources
                    .mcp_hosts
                    .get(&mcp_id)
                    .map(|e| e.users.iter().cloned().collect())
                    .unwrap_or_default();
                for tid in &bound_to_mcp {
                    if self.host_env_mcp_is_ready(tid) {
                        self.push_tools_snapshot(tid);
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

    /// Inner helper for `retention_sweep`: removes the thread from
    /// every in-memory surface (tasks map, pod membership, router
    /// subscriptions, resource user sets), broadcasts `ThreadArchived`,
    /// and spawns the disk op. Used only for behavior-spawned threads.
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
        let plan = self.resources.reap_idle(
            chrono::Utc::now(),
            chrono::Duration::seconds(IDLE_RESOURCE_SECS),
            chrono::Duration::seconds(TERMINAL_RETENTION_SECS),
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

/// Build the `ThreadPendingApproval` events that a newly-subscribed
/// client needs to render the approval UI. Scans the Function registry
/// for tool-call entries on this thread that are waiting on approval.
pub(super) fn pending_approvals_of(scheduler: &Scheduler, task: &Thread) -> Vec<ServerToClient> {
    let mut out = Vec::new();
    for entry in scheduler.active_functions.values() {
        // Only tool-call Functions on this thread that are paused
        // awaiting an approval decision contribute to the snapshot.
        let crate::functions::CallerLink::ThreadToolCall {
            thread_id,
            tool_use_id,
        } = &entry.caller
        else {
            continue;
        };
        if thread_id != &task.id {
            continue;
        }
        let Some(io_req) = &entry.pending_approval_io else {
            continue;
        };
        let crate::runtime::thread::IoRequest::ToolCall { name, input, .. } = io_req else {
            continue;
        };
        let approval_id = format!("ap-{tool_use_id}");
        let annotations = scheduler.annotations_for(&task.id);
        let empty = ToolAnnotations::default();
        let ann = annotations.get(name).unwrap_or(&empty);
        out.push(ServerToClient::ThreadPendingApproval {
            thread_id: task.id.clone(),
            approval_id,
            tool_use_id: tool_use_id.clone(),
            name: name.clone(),
            args_preview: truncate(serde_json::to_string(input).unwrap_or_default(), 200),
            destructive: ann.is_destructive(),
            read_only: ann.is_read_only(),
        });
    }
    out
}

fn truncate(mut s: String, max: usize) -> String {
    if s.len() > max {
        let mut cut = max;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push('…');
    }
    s
}

/// Read a pod-relative prompt file for a creation-time system-prompt
/// override. Rejects any name containing path separators or a parent
/// component — the override is a filename inside the pod dir, not an
/// escape hatch. A missing or unreadable file falls back to the pod's
/// cached default with a warn-level log, matching the pod loader's
/// graceful-degrade policy for `thread_defaults.system_prompt_file`.
fn resolve_system_prompt_file(pod: &crate::pod::Pod, name: &str) -> String {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        warn!(
            pod_id = %pod.id,
            file = %name,
            "system_prompt File override name rejected (must be a plain filename inside the pod dir) \
             — falling back to pod default"
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
) {
    let mut pending_io: FuturesUnordered<SchedulerFuture> = FuturesUnordered::new();
    let mut gc_ticker = tokio::time::interval(Duration::from_secs(GC_TICK_SECS));
    let mut cron_ticker = tokio::time::interval(Duration::from_secs(CRON_TICK_SECS));
    let mut retention_ticker = tokio::time::interval(Duration::from_secs(RETENTION_TICK_SECS));
    let mut probe_ticker = tokio::time::interval(Duration::from_secs(HOST_ENV_PROBE_TICK_SECS));
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
