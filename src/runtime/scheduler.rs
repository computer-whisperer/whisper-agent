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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use whisper_agent_protocol::{
    BackendSummary, ClientToServer, HostEnvBinding, HostEnvRebind, Message, ModelSummary,
    ResourceKind, ServerToClient, ThreadBindings, ThreadBindingsPatch, ThreadBindingsRequest,
    ThreadConfig, ThreadConfigOverride, ThreadStateLabel,
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
    ApprovalDisposition, IoResult, OpId, StepOutcome, Thread, ThreadInternalState, derive_title,
    new_task_id,
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
    /// Forward to the named MCP session via JSON-RPC.
    Mcp(Arc<McpSession>),
}

/// Snapshot of a pod's on-disk directory + parsed config. Cloned out of
/// the scheduler when dispatching a builtin tool future so the future
/// doesn't need a back-borrow of scheduler state.
pub struct PodSnapshot {
    pub pod_dir: std::path::PathBuf,
    pub config: whisper_agent_protocol::PodConfig,
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
    /// Catalog of host-env providers, keyed by name. Empty when the
    /// server was started without any `[[host_env_providers]]` entries
    /// — threads in such a server can still run, just without any
    /// host-env MCP (so their tool catalog is shared-only).
    host_env_registry: crate::tools::sandbox::HostEnvRegistry,

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
    /// Thread ids whose host-env provisioning future is currently in
    /// flight. Used to deduplicate
    /// `ensure_host_env_provisioning` calls across the create_task /
    /// send_user_message paths so we don't fan out redundant provision
    /// futures for the same thread.
    provisioning_in_flight: HashSet<String>,
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
        shared_host_configs: Vec<SharedHostConfig>,
    ) -> anyhow::Result<Self> {
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
            let session = Arc::new(McpSession::connect(&cfg.url).await.map_err(|e| {
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

        Ok(Self {
            default_pod_id,
            pods,
            backends,
            default_backend,
            persister: None,
            host_env_registry,
            tasks: HashMap::new(),
            router: ThreadEventRouter::new(audit, host_id),
            resources,
            next_op_id: 1,
            dirty: HashSet::new(),
            provisioning_in_flight: HashSet::new(),
        })
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

    /// Compute the runtime `HostEnvId` for a thread's binding. Returns
    /// `None` when the thread has no host env binding or when
    /// `resolve_binding` couldn't find the named entry.
    pub(crate) fn host_env_id_for_thread(&self, thread_id: &str) -> Option<HostEnvId> {
        let task = self.tasks.get(thread_id)?;
        let binding = task.bindings.host_env.as_ref()?;
        let (provider, spec) = self.resolve_binding(&task.pod_id, binding)?;
        Some(HostEnvId::for_provider_spec(&provider, &spec))
    }

    /// Look up the registry entry for a thread's bound host env.
    /// Returns `None` when the thread has no binding, the binding
    /// can't be resolved (named entry removed), or when
    /// `pre_register_host_env` somehow wasn't called.
    pub(crate) fn host_env_for_thread(
        &self,
        thread_id: &str,
    ) -> Option<&crate::pod::resources::HostEnvEntry> {
        let id = self.host_env_id_for_thread(thread_id)?;
        self.resources.host_envs.get(&id)
    }

    /// Phase 3d.i: build the thread's effective tool catalog. Prepends
    /// the builtin pod-editing tools (always available to every thread),
    /// then walks `task.bindings.mcp_hosts` in precedence order — host-env
    /// MCP first, then each shared host the thread is bound to. Tool-name
    /// collisions resolve in favor of the earlier entry, so builtins win
    /// if an MCP host tries to advertise a colliding name.
    pub(crate) fn tool_descriptors(&self, thread_id: &str) -> Vec<McpTool> {
        let mut out: Vec<McpTool> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for tool in crate::tools::builtin_tools::descriptors() {
            if seen.insert(tool.name.clone()) {
                out.push(tool);
            }
        }
        for host in self.bound_mcp_hosts(thread_id) {
            for tool in &host.tools {
                if seen.insert(tool.name.clone()) {
                    out.push(tool.clone());
                }
            }
        }
        out
    }

    /// Resolve which handler should receive a tool invocation. Builtin
    /// tools are checked first (they always win); otherwise walks the
    /// thread's bound MCP hosts in precedence order and returns the
    /// first host whose tool catalog includes `tool_name`. `None` when
    /// nothing claims the name — the model called a tool we didn't
    /// advertise.
    pub(crate) fn route_tool(&self, thread_id: &str, tool_name: &str) -> Option<ToolRoute> {
        if crate::tools::builtin_tools::is_builtin(tool_name) {
            let pod_id = self.tasks.get(thread_id)?.pod_id.clone();
            return Some(ToolRoute::Builtin { pod_id });
        }
        for host in self.bound_mcp_hosts(thread_id) {
            if host.tools.iter().any(|t| t.name == tool_name)
                && let Some(s) = host.session.clone()
            {
                return Some(ToolRoute::Mcp(s));
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
        })
    }

    /// Iterate the thread's bound MCP host entries in precedence
    /// order: the host-env MCP first (if any), then each shared MCP
    /// host in pod-declared order. Skips ids whose entry isn't yet in
    /// the registry (host-env MCP is created when provisioning
    /// dispatches; shared entries exist from startup).
    fn bound_mcp_hosts(&self, thread_id: &str) -> Vec<&crate::pod::resources::McpHostEntry> {
        let mut out = Vec::new();
        if let Some(he_id) = self.host_env_id_for_thread(thread_id) {
            let mcp_id = McpHostId::for_host_env(&he_id);
            if let Some(entry) = self.resources.mcp_hosts.get(&mcp_id) {
                out.push(entry);
            }
        }
        let Some(task) = self.tasks.get(thread_id) else {
            return out;
        };
        for name in &task.bindings.mcp_hosts {
            let id = McpHostId::shared(name);
            if let Some(entry) = self.resources.mcp_hosts.get(&id) {
                out.push(entry);
            }
        }
        out
    }

    /// Walk the thread's bindings and return every bound resource id whose
    /// registry entry isn't yet Ready (or is missing entirely — happens
    /// post-restart for the host-env MCP). Used by
    /// `send_user_message` to park the thread in `WaitingOnResources`
    /// when provisioning is still in flight.
    fn pending_resources_for(&self, thread_id: &str) -> Vec<String> {
        let Some(task) = self.tasks.get(thread_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        if let Some(he_id) = self.host_env_id_for_thread(thread_id) {
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

    /// Ensure a host-env provisioning future is in flight for this
    /// thread, if it has a host_env binding. No-op when the thread
    /// has no binding — in that case the thread's tool set is just
    /// the shared MCPs it bound to, no provisioning needed. Called
    /// from `create_task` and `send_user_message` (post-restart
    /// recovery). The `provisioning_in_flight` guard prevents
    /// double-dispatch.
    fn ensure_host_env_provisioning(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if self.provisioning_in_flight.contains(thread_id) {
            return;
        }
        let Some(host_env_id) = self.host_env_id_for_thread(thread_id) else {
            return;
        };
        let mcp_id = McpHostId::for_host_env(&host_env_id);
        let needs_dispatch = match self.resources.mcp_hosts.get(&mcp_id) {
            Some(entry) => !entry.state.is_ready(),
            None => true,
        };
        if !needs_dispatch {
            return;
        }
        // Register the MCP entry up front so waiters can observe
        // Provisioning. Idempotent: re-registering for the same
        // host_env just touches last_used and adds the thread user.
        self.resources
            .pre_register_host_env_mcp(&host_env_id, thread_id);
        self.emit_mcp_host_updated(&mcp_id);
        self.provisioning_in_flight.insert(thread_id.to_string());
        let fut = io_dispatch::provision_host_env_mcp(self, thread_id.to_string());
        pending_io.push(fut);
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
    fn apply_rebind(
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

        let new_host_env_binding: Option<HostEnvBinding> = match &patch.host_env {
            None => old_bindings.host_env.clone(),
            Some(HostEnvRebind::Clear) => None,
            Some(HostEnvRebind::Named { name }) => {
                // Validate against the pod's allow list — cap is real.
                if !pod_allow.host_env.iter().any(|nh| &nh.name == name) {
                    return Err(format!(
                        "host env `{name}` not in pod `{}`'s allow.host_env",
                        pod_id,
                    ));
                }
                Some(HostEnvBinding::Named { name: name.clone() })
            }
        };
        // The runtime id is derived per binding for refcounting + emit.
        // `None` here means the thread had no host env binding, or a
        // Named binding whose entry has gone missing (already torn
        // down or pod was edited mid-flight — we just skip the
        // user-count drop).
        let old_host_env_id = old_bindings
            .host_env
            .as_ref()
            .and_then(|b| self.resolve_binding(&pod_id, b))
            .map(|(p, s)| HostEnvId::for_provider_spec(&p, &s));

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
        let host_env_changed = new_host_env_binding != old_bindings.host_env;
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
            // Release the old host env; tear it down if we were the last
            // user and the entry isn't pinned. Tolerates a missing id
            // (named entry was removed from pod.toml mid-life — the old
            // entry is effectively orphaned and we just drop the user
            // count if it's still around).
            if let Some(old_id) = old_host_env_id {
                let remaining = self.resources.release_host_env_user(&old_id, thread_id);
                let pinned = self
                    .resources
                    .host_envs
                    .get(&old_id)
                    .map(|e| e.pinned)
                    .unwrap_or(false);
                if remaining == 0 && !pinned {
                    let handle = self.resources.take_host_env_handle(&old_id);
                    self.resources.mark_host_env_torn_down(&old_id);
                    self.emit_host_env_updated(&old_id);
                    if let Some(mut h) = handle {
                        let tid = thread_id.to_string();
                        tokio::spawn(async move {
                            if let Err(e) = h.teardown().await {
                                warn!(thread_id = %tid, error = %e, "host env teardown after rebind failed");
                            }
                        });
                    } else {
                        self.emit_host_env_updated(&old_id);
                    }
                } else {
                    self.emit_host_env_updated(&old_id);
                }
            }
            // Pre-register the new host env (if any).
            if let Some(binding) = &new_host_env_binding {
                let (provider, spec) = self
                    .resolve_binding(&pod_id, binding)
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
            // Mark the OLD host-env MCP entry torn down (if there was one).
            // The new binding will get its own fresh MCP entry when the
            // next provision dispatches.
            if let Some(old_id) = old_bindings
                .host_env
                .as_ref()
                .and_then(|b| self.resolve_binding(&pod_id, b))
                .map(|(p, s)| HostEnvId::for_provider_spec(&p, &s))
            {
                let mcp_id = McpHostId::for_host_env(&old_id);
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
            host_env: new_host_env_binding.clone(),
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

        if host_env_changed {
            self.provisioning_in_flight.remove(thread_id);
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
            // Re-pre-register the thread's host env so the registry knows
            // its (provider, spec). For a Named binding we look up the
            // pod entry by name; for Inline the spec lives in the
            // binding itself. If a Named binding can't be resolved
            // (entry was removed from the pod) and the pod still has
            // a default, we silently re-bind — surfacing the change as
            // a `mark_dirty` so the new binding flushes on next tick.
            // This is also the migration path for pre-refactor threads
            // whose persisted host_env was a bare id string: persist
            // normalizes that to None on load, and we re-bind here.
            let mut bindings_dirty = false;
            // Migrate legacy Inline bindings: the pod's `allow.host_env`
            // is now the sole source of truth for user-composed
            // threads. Walk the pod's allow list for a structurally
            // equivalent entry; if one exists, rebind to the Named
            // reference. Otherwise fall through to the default-rebind
            // path below (Inline becomes None, then Named default).
            if let Some(HostEnvBinding::Inline { provider, spec }) = task.bindings.host_env.clone()
                && let Some(pod) = self.pods.get(&task.pod_id)
            {
                let matching = pod
                    .config
                    .allow
                    .host_env
                    .iter()
                    .find(|nh| nh.provider == provider && nh.spec == spec);
                match matching {
                    Some(nh) => {
                        warn!(
                            thread_id = %task.id,
                            rebound_to = %nh.name,
                            "migrating Inline host_env binding to Named reference",
                        );
                        task.bindings.host_env = Some(HostEnvBinding::Named {
                            name: nh.name.clone(),
                        });
                        bindings_dirty = true;
                    }
                    None => {
                        warn!(
                            thread_id = %task.id,
                            "Inline host_env binding has no structural match in pod.allow.host_env; rebinding to pod default",
                        );
                        task.bindings.host_env = None;
                        bindings_dirty = true;
                    }
                }
            }
            if task.bindings.host_env.is_none()
                && let Some(pod) = self.pods.get(&task.pod_id)
                && !pod.config.thread_defaults.host_env.is_empty()
            {
                task.bindings.host_env = Some(HostEnvBinding::Named {
                    name: pod.config.thread_defaults.host_env.clone(),
                });
                bindings_dirty = true;
            }
            if let Some(binding) = task.bindings.host_env.clone() {
                match self.resolve_binding(&task.pod_id, &binding) {
                    Some((provider, spec)) => {
                        self.resources
                            .pre_register_host_env(&task.id, provider, spec);
                    }
                    None => {
                        // Named entry missing — fall back to pod default
                        // when one exists, else drop the binding.
                        let fallback = self
                            .pods
                            .get(&task.pod_id)
                            .map(|p| p.config.thread_defaults.host_env.clone())
                            .filter(|s| !s.is_empty());
                        match fallback {
                            Some(default_name) => {
                                let new_binding = HostEnvBinding::Named {
                                    name: default_name.clone(),
                                };
                                if let Some((provider, spec)) =
                                    self.resolve_binding(&task.pod_id, &new_binding)
                                {
                                    warn!(
                                        thread_id = %task.id,
                                        replaced_with = %default_name,
                                        "loaded thread's host_env binding could not be resolved; rebinding to pod default",
                                    );
                                    self.resources
                                        .pre_register_host_env(&task.id, provider, spec);
                                    task.bindings.host_env = Some(new_binding);
                                    bindings_dirty = true;
                                }
                            }
                            None => {
                                warn!(
                                    thread_id = %task.id,
                                    "loaded thread's host_env binding could not be resolved and pod has no default; dropping binding",
                                );
                                task.bindings.host_env = None;
                                bindings_dirty = true;
                            }
                        }
                    }
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

    async fn flush_dirty(&mut self) {
        let Some(persister) = &self.persister else {
            self.dirty.clear();
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
        }
    }

    fn apply_client_message(
        &mut self,
        conn_id: ConnId,
        msg: ClientToServer,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        match msg {
            ClientToServer::CreateThread {
                correlation_id,
                pod_id,
                initial_message,
                config_override,
                bindings_request,
            } => match self.create_task(
                conn_id,
                correlation_id.clone(),
                pod_id,
                config_override,
                bindings_request,
                pending_io,
            ) {
                Ok(thread_id) => {
                    self.mark_dirty(&thread_id);
                    self.send_user_message(&thread_id, initial_message, pending_io);
                    self.step_until_blocked(&thread_id, pending_io);
                }
                Err(e) => {
                    warn!(error = %e, conn_id, "create_task rejected");
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("create_task: {e}"),
                        },
                    );
                }
            },
            ClientToServer::SendUserMessage { thread_id, text } => {
                if self.tasks.contains_key(&thread_id) {
                    self.send_user_message(&thread_id, text, pending_io);
                    self.step_until_blocked(&thread_id, pending_io);
                } else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id),
                            message: "unknown task".into(),
                        },
                    );
                }
            }
            ClientToServer::ApprovalDecision {
                thread_id,
                approval_id,
                decision,
                remember,
            } => {
                let mut events = Vec::new();
                let known = if let Some(task) = self.tasks.get_mut(&thread_id) {
                    task.apply_approval_decision(
                        &approval_id,
                        decision,
                        remember,
                        Some(conn_id),
                        &mut events,
                    );
                    true
                } else {
                    false
                };
                if !known {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id),
                            message: "unknown task".into(),
                        },
                    );
                } else {
                    self.mark_dirty(&thread_id);
                    self.router.dispatch_events(&thread_id, events);
                    self.teardown_host_env_if_terminal(&thread_id);
                    self.step_until_blocked(&thread_id, pending_io);
                }
            }
            ClientToServer::RemoveToolAllowlistEntry {
                thread_id,
                tool_name,
            } => {
                let removed = self
                    .tasks
                    .get_mut(&thread_id)
                    .map(|t| (t.remove_from_allowlist(&tool_name), t.allowlist_snapshot()));
                match removed {
                    Some((true, snapshot)) => {
                        self.mark_dirty(&thread_id);
                        self.router.broadcast_to_subscribers(
                            &thread_id,
                            ServerToClient::ThreadAllowlistUpdated {
                                thread_id: thread_id.clone(),
                                tool_allowlist: snapshot,
                            },
                        );
                    }
                    Some((false, _)) => { /* nothing to remove — silent no-op */ }
                    None => self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id),
                            message: "unknown task".into(),
                        },
                    ),
                }
            }
            ClientToServer::RebindThread {
                thread_id,
                patch,
                correlation_id,
            } => {
                if let Err(e) =
                    self.apply_rebind(&thread_id, patch, correlation_id.clone(), pending_io)
                {
                    warn!(error = %e, conn_id, %thread_id, "rebind rejected");
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: Some(thread_id),
                            message: format!("rebind: {e}"),
                        },
                    );
                }
            }
            ClientToServer::CancelThread { thread_id } => {
                if let Some(task) = self.tasks.get_mut(&thread_id) {
                    task.cancel();
                    self.mark_dirty(&thread_id);
                    self.router
                        .broadcast_task_list(ServerToClient::ThreadStateChanged {
                            thread_id: thread_id.clone(),
                            state: ThreadStateLabel::Cancelled,
                        });
                    self.teardown_host_env_if_terminal(&thread_id);
                }
            }
            ClientToServer::ArchiveThread { thread_id } => {
                if let Some(task) = self.tasks.get_mut(&thread_id) {
                    task.archived = true;
                    task.touch();
                    self.mark_dirty(&thread_id);
                    self.router
                        .broadcast_task_list(ServerToClient::ThreadArchived { thread_id });
                }
            }
            ClientToServer::SubscribeToThread { thread_id } => {
                if let Some(task) = self.tasks.get(&thread_id) {
                    self.router.subscribe(conn_id, &thread_id);
                    let snapshot = task.snapshot();
                    // Rehydrate any still-pending approvals so the newly-subscribed
                    // client can render the approval UI. The snapshot itself doesn't
                    // carry approval state.
                    let pending = pending_approvals_of(task);
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::ThreadSnapshot {
                            thread_id: thread_id.clone(),
                            snapshot,
                        },
                    );
                    for event in pending {
                        self.router.send_to_client(conn_id, event);
                    }
                } else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id),
                            message: "unknown task".into(),
                        },
                    );
                }
            }
            ClientToServer::UnsubscribeFromThread { thread_id } => {
                self.router.unsubscribe(conn_id, &thread_id);
            }
            ClientToServer::ListThreads { correlation_id } => {
                let tasks = self
                    .tasks
                    .values()
                    .filter(|t| !t.archived)
                    .map(|t| t.summary())
                    .collect();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::ThreadList {
                        correlation_id,
                        tasks,
                    },
                );
            }
            ClientToServer::ListBackends { correlation_id } => {
                let backends: Vec<BackendSummary> = self
                    .backends
                    .iter()
                    .map(|(name, entry)| BackendSummary {
                        name: name.clone(),
                        kind: entry.kind.clone(),
                        default_model: entry.default_model.clone(),
                    })
                    .collect();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::BackendsList {
                        correlation_id,
                        default_backend: self.default_backend.clone(),
                        backends,
                    },
                );
            }
            ClientToServer::ListResources { correlation_id } => {
                let resources = self.resources.snapshot_all();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::ResourceList {
                        correlation_id,
                        resources,
                    },
                );
            }
            ClientToServer::ListHostEnvProviders { correlation_id } => {
                let providers = self.host_env_registry.snapshot();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::HostEnvProvidersList {
                        correlation_id,
                        providers,
                    },
                );
            }
            ClientToServer::ListPods { correlation_id } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                let default_pod_id = self.default_pod_id.clone();
                tokio::spawn(async move {
                    match persister.list_pods().await {
                        Ok(pods) => {
                            let _ = outbound.send(ServerToClient::PodList {
                                correlation_id,
                                pods,
                                default_pod_id,
                            });
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("list_pods: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::GetPod {
                correlation_id,
                pod_id,
            } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                tokio::spawn(async move {
                    match persister.get_pod(&pod_id).await {
                        Ok(Some(snapshot)) => {
                            let _ = outbound.send(ServerToClient::PodSnapshot {
                                snapshot,
                                correlation_id,
                            });
                        }
                        Ok(None) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("pod `{pod_id}` not found"),
                            });
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("get_pod: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::CreatePod {
                correlation_id,
                pod_id,
                mut config,
            } => {
                // Sync work first: validate + register in-memory so a
                // CreateThread that races behind this can find the pod.
                // The disk write happens in the background (logged on
                // failure); the in-memory state is what subsequent
                // CreateThread / GetPod calls consult.
                if let Err(e) = crate::pod::persist::validate_pod_id(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("create_pod: {e}"),
                        },
                    );
                    return;
                }
                if config.created_at.is_empty() {
                    config.created_at = chrono::Utc::now().to_rfc3339();
                }
                if let Err(e) = crate::pod::validate(&config) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("create_pod: {e}"),
                        },
                    );
                    return;
                }
                if self.pods.contains_key(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("create_pod: pod `{pod_id}` already exists"),
                        },
                    );
                    return;
                }
                let raw_toml = match crate::pod::to_toml(&config) {
                    Ok(t) => t,
                    Err(e) => {
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("create_pod: encode toml: {e}"),
                            },
                        );
                        return;
                    }
                };
                let pod_dir = self
                    .persister
                    .as_ref()
                    .map(|p| p.dir().join(&pod_id))
                    .unwrap_or_else(|| std::path::PathBuf::from(&pod_id));
                let pod = Pod::new(
                    pod_id.clone(),
                    pod_dir,
                    config.clone(),
                    raw_toml,
                    String::new(),
                );
                let summary = whisper_agent_protocol::PodSummary {
                    pod_id: pod_id.clone(),
                    name: config.name.clone(),
                    description: config.description.clone(),
                    created_at: config.created_at.clone(),
                    thread_count: 0,
                    archived: false,
                };
                self.pods.insert(pod_id.clone(), pod);

                // Broadcast PodCreated to every connected client.
                let ev = ServerToClient::PodCreated {
                    pod: summary,
                    correlation_id,
                };
                for tx in self.router.outbound_snapshot() {
                    let _ = tx.send(ev.clone());
                }

                // Disk write runs in the background. Failure logs warn —
                // the in-memory pod stays usable but won't survive a
                // restart. (No good story for rollback today; the user
                // can retry by deleting and recreating.)
                if let Some(persister) = self.persister.clone() {
                    let pid = pod_id.clone();
                    tokio::spawn(async move {
                        if let Err(e) = persister.create_pod(&pid, config).await {
                            warn!(pod_id = %pid, error = %e, "create_pod disk write failed");
                        }
                    });
                }
            }
            ClientToServer::UpdatePodConfig {
                correlation_id,
                pod_id,
                toml_text,
            } => {
                if let Err(e) = crate::pod::persist::validate_pod_id(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("update_pod_config: {e}"),
                        },
                    );
                    return;
                }
                if !self.pods.contains_key(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("update_pod_config: unknown pod `{pod_id}`"),
                        },
                    );
                    return;
                }
                let parsed = match crate::pod::parse_toml(&toml_text) {
                    Ok(c) => c,
                    Err(e) => {
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("update_pod_config: {e}"),
                            },
                        );
                        return;
                    }
                };
                self.apply_pod_config_update(&pod_id, toml_text.clone(), parsed, correlation_id);
                self.persist_pod_config(pod_id, toml_text, Some(conn_id));
            }
            ClientToServer::ArchivePod { pod_id } => {
                if let Err(e) = crate::pod::persist::validate_pod_id(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: None,
                            message: format!("archive_pod: {e}"),
                        },
                    );
                    return;
                }
                if pod_id == self.default_pod_id {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: None,
                            message: "archive_pod: refusing to archive the default pod".into(),
                        },
                    );
                    return;
                }
                let Some(pod) = self.pods.remove(&pod_id) else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: None,
                            message: format!("archive_pod: unknown pod `{pod_id}`"),
                        },
                    );
                    return;
                };
                // Drop in-memory thread state for every thread under this pod.
                // The pod directory is moving to .archived/, so these threads
                // become unreachable — we'd otherwise leak entries in
                // `tasks` / `dirty` / `provisioning_in_flight` / `router`
                // subs, and the resource-registry user sets would carry
                // stale thread_ids that prevent GC from ever marking those
                // resources idle.
                for thread_id in &pod.threads {
                    let bindings = self.tasks.get(thread_id).map(|t| t.bindings.clone());
                    self.tasks.remove(thread_id);
                    self.dirty.remove(thread_id);
                    self.provisioning_in_flight.remove(thread_id);
                    self.router.drop_thread(thread_id);
                    if let Some(bindings) = bindings {
                        self.release_thread_resources(thread_id, &pod_id, &bindings);
                    }
                }
                // Broadcast first so every client clears its view before the
                // disk move completes; the disk write is best-effort.
                let ev = ServerToClient::PodArchived {
                    pod_id: pod_id.clone(),
                };
                for tx in self.router.outbound_snapshot() {
                    let _ = tx.send(ev.clone());
                }
                if let Some(persister) = self.persister.clone() {
                    let outbound = self.router.outbound(conn_id);
                    tokio::spawn(async move {
                        if let Err(e) = persister.archive_pod(&pod_id).await {
                            warn!(pod_id = %pod_id, error = %e, "archive_pod disk move failed");
                            if let Some(tx) = outbound {
                                let _ = tx.send(ServerToClient::Error {
                                    correlation_id: None,
                                    thread_id: None,
                                    message: format!("archive_pod: disk move failed: {e}"),
                                });
                            }
                        }
                    });
                }
            }
            ClientToServer::ListBehaviors {
                correlation_id,
                pod_id,
            } => {
                let Some(pod) = self.pods.get(&pod_id) else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("list_behaviors: unknown pod `{pod_id}`"),
                        },
                    );
                    return;
                };
                let behaviors: Vec<_> = pod.behaviors.values().map(|b| b.summary()).collect();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::BehaviorList {
                        correlation_id,
                        pod_id,
                        behaviors,
                    },
                );
            }
            ClientToServer::GetBehavior {
                correlation_id,
                pod_id,
                behavior_id,
            } => {
                let Some(pod) = self.pods.get(&pod_id) else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("get_behavior: unknown pod `{pod_id}`"),
                        },
                    );
                    return;
                };
                let Some(behavior) = pod.behaviors.get(&behavior_id) else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!(
                                "get_behavior: unknown behavior `{behavior_id}` under pod `{pod_id}`"
                            ),
                        },
                    );
                    return;
                };
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::BehaviorSnapshot {
                        correlation_id,
                        snapshot: behavior.snapshot(),
                    },
                );
            }
            ClientToServer::ListModels {
                correlation_id,
                backend,
            } => {
                let entry = self.backends.get(&backend);
                let Some(entry) = entry else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("unknown backend `{backend}`"),
                        },
                    );
                    return;
                };
                // Spawn a detached task so we don't block the scheduler loop on the
                // backend's network round-trip. The task writes directly to the
                // client's outbound channel — the scheduler holds no intermediate state.
                let provider = entry.provider.clone();
                let Some(outbound) = self.router.outbound(conn_id) else {
                    return;
                };
                tokio::spawn(async move {
                    match provider.list_models().await {
                        Ok(models) => {
                            let models: Vec<ModelSummary> = models
                                .into_iter()
                                .map(|m| ModelSummary {
                                    id: m.id,
                                    display_name: m.display_name,
                                })
                                .collect();
                            let _ = outbound.send(ServerToClient::ModelsList {
                                correlation_id,
                                backend,
                                models,
                            });
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("list_models failed: {e}"),
                            });
                        }
                    }
                });
            }
        }
    }

    fn create_task(
        &mut self,
        requester: ConnId,
        correlation_id: Option<String>,
        pod_id: Option<String>,
        config_override: Option<ThreadConfigOverride>,
        bindings_request: Option<ThreadBindingsRequest>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<String, String> {
        // Resolve which pod this thread lands in. None routes to the
        // server's default pod.
        let pod_id = pod_id.unwrap_or_else(|| self.default_pod_id.clone());
        let pod = self
            .pods
            .get(&pod_id)
            .ok_or_else(|| format!("unknown pod `{pod_id}`"))?;

        // Resolve the thread's plain config (model, prompts, limits, policy).
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

        // Pre-register the host env so the registry holds the
        // Pre-register the host env (if any). The deterministic
        // HostEnvId means a second thread with the same (provider,
        // spec) just joins the existing entry as a user. No binding
        // → no pre-register, no provisioning, no host-env MCP. The
        // thread's tool set in that case is just the shared MCPs it
        // bound to.
        let host_env_id = match resolved.host_env.as_ref() {
            None => None,
            Some(binding) => {
                let (provider, spec) = self
                    .resolve_binding(&pod_id, binding)
                    .expect("resolve_bindings_choice already validated this");
                Some(
                    self.resources
                        .pre_register_host_env(&thread_id, provider, spec),
                )
            }
        };
        let bindings = ThreadBindings {
            backend: resolved.backend_name.clone(),
            host_env: resolved.host_env.clone(),
            mcp_hosts: resolved.shared_host_names.clone(),
            tool_filter: None,
        };

        let task = Thread::new(thread_id.clone(), pod_id.clone(), config, bindings);
        let summary = task.summary();
        self.tasks.insert(thread_id.clone(), task);
        // Mirror the new thread into the owning pod's `threads` set.
        if let Some(pod) = self.pods.get_mut(&pod_id) {
            pod.threads.insert(thread_id.clone());
        }

        // Register this task as a user of its backend and of every
        // shared MCP host it references. pre_register_host_env
        // already counted us as a host-env user.
        let backend_id = BackendId::for_name(&resolved.backend_name);
        self.resources.add_backend_user(&backend_id, &thread_id);
        self.emit_backend_updated(&backend_id);
        if let Some(id) = host_env_id.as_ref() {
            self.emit_host_env_updated(id);
        }
        for name in &resolved.shared_host_names {
            let host_id = McpHostId::shared(name);
            self.resources.add_mcp_user(&host_id, &thread_id);
            self.emit_mcp_host_updated(&host_id);
        }
        // Kick off host-env provisioning if the thread has a host_env
        // binding. No-op for threads bound only to shared MCPs.
        self.ensure_host_env_provisioning(&thread_id, pending_io);

        info!(thread_id = %thread_id, pod_id = %pod_id, "task created");
        // Every client gets exactly one ThreadCreated; the requester's copy carries
        // correlation_id if they provided one.
        if correlation_id.is_some() {
            self.router.broadcast_task_list_except(
                ServerToClient::ThreadCreated {
                    thread_id: thread_id.clone(),
                    summary: summary.clone(),
                    correlation_id: None,
                },
                requester,
            );
            self.router.send_to_client(
                requester,
                ServerToClient::ThreadCreated {
                    thread_id: thread_id.clone(),
                    summary,
                    correlation_id,
                },
            );
        } else {
            self.router
                .broadcast_task_list(ServerToClient::ThreadCreated {
                    thread_id: thread_id.clone(),
                    summary,
                    correlation_id: None,
                });
        }
        Ok(thread_id)
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
            task.submit_user_message(text, pending_resources);
            task.public_state()
        };
        self.router
            .broadcast_task_list(ServerToClient::ThreadStateChanged {
                thread_id: thread_id.to_string(),
                state: new_state,
            });
    }

    /// Apply one per-thread I/O completion to its task and dispatch any
    /// resulting events. Resource-provisioning completions take a
    /// different path; see [`Self::apply_provision_completion`].
    fn apply_io_completion(&mut self, completion: IoCompletion) {
        let IoCompletion {
            thread_id,
            op_id,
            result,
            pod_update,
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
        if let Some(task) = self.tasks.get_mut(&thread_id) {
            task.apply_io_result(op_id, result, &annotations, &mut events);
        } else {
            warn!(%thread_id, op_id, "io completion for unknown task");
            return;
        }
        self.mark_dirty(&thread_id);
        self.router.dispatch_events(&thread_id, events);
        self.teardown_host_env_if_terminal(&thread_id);
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
        self.provisioning_in_flight.remove(&thread_id);
        match result {
            ProvisionResult::HostEnvMcpReady {
                host_env_id,
                host_env_handle,
                mcp_url,
                mcp_session,
                tools,
            } => {
                let mcp_id = McpHostId::for_host_env(&host_env_id);
                // 1. Host env — attach the handle (or recognize the
                //    dedup race and tear ours down).
                let outcome = self.resources.complete_host_env_provisioning(
                    &host_env_id,
                    Some(mcp_url.clone()),
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

                // 3. Nudge any threads that were waiting on these
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
                }
            }
        }
    }

    fn annotations_for(&self, thread_id: &str) -> HashMap<String, ToolAnnotations> {
        let mut out: HashMap<String, ToolAnnotations> = crate::tools::builtin_tools::annotations();
        for host in self.bound_mcp_hosts(thread_id) {
            for (name, ann) in &host.annotations {
                // First-wins matches `route_tool`'s builtin-then-host
                // precedence, so the approval decision agrees with
                // where the call will actually land.
                out.entry(name.clone()).or_insert_with(|| ann.clone());
            }
        }
        out
    }

    /// Apply a [`PodUpdate`] side effect produced by a builtin tool.
    /// Dispatches to the specific apply helper based on the update
    /// kind; both helpers handle the in-memory refresh + broadcast.
    /// The builtin tool already wrote the change to disk — no further
    /// persist work is needed here.
    fn apply_pod_update(
        &mut self,
        thread_id: &str,
        update: crate::tools::builtin_tools::PodUpdate,
    ) {
        let Some(task) = self.tasks.get(thread_id) else {
            warn!(%thread_id, "apply_pod_update: task not found");
            return;
        };
        let pod_id = task.pod_id.clone();
        match update {
            crate::tools::builtin_tools::PodUpdate::Config { toml_text, parsed } => {
                self.apply_pod_config_update(&pod_id, toml_text, *parsed, None);
            }
            crate::tools::builtin_tools::PodUpdate::SystemPrompt { text } => {
                self.apply_system_prompt_update(&pod_id, text, None);
            }
        }
    }

    /// Refresh the in-memory pod with a validated new config + raw TOML
    /// and broadcast `PodConfigUpdated` to every connected client. Does
    /// NOT touch disk — callers that originate on-wire updates (the
    /// UpdatePodConfig handler) separately invoke [`persist_pod_config`];
    /// the builtin-tool path has already written the file before
    /// reaching here.
    fn apply_pod_config_update(
        &mut self,
        pod_id: &str,
        toml_text: String,
        parsed: whisper_agent_protocol::PodConfig,
        correlation_id: Option<String>,
    ) {
        if let Some(pod) = self.pods.get_mut(pod_id) {
            pod.config = parsed.clone();
            pod.raw_toml = toml_text.clone();
        }
        let ev = ServerToClient::PodConfigUpdated {
            pod_id: pod_id.to_string(),
            toml_text,
            parsed,
            correlation_id,
        };
        for tx in self.router.outbound_snapshot() {
            let _ = tx.send(ev.clone());
        }
    }

    /// Refresh the in-memory pod's cached system prompt and broadcast
    /// `PodSystemPromptUpdated`. Same split as
    /// [`apply_pod_config_update`] — builtin-tool callers have already
    /// written the file; wire-originated callers spawn
    /// [`persist_system_prompt`] alongside.
    fn apply_system_prompt_update(
        &mut self,
        pod_id: &str,
        text: String,
        correlation_id: Option<String>,
    ) {
        if let Some(pod) = self.pods.get_mut(pod_id) {
            pod.system_prompt = text.clone();
        }
        let ev = ServerToClient::PodSystemPromptUpdated {
            pod_id: pod_id.to_string(),
            text,
            correlation_id,
        };
        for tx in self.router.outbound_snapshot() {
            let _ = tx.send(ev.clone());
        }
    }

    /// Spawn a background task that flushes a new pod.toml to disk. On
    /// failure the pod is still live in memory (the broadcast already
    /// went out); we log and optionally surface an error to the
    /// originating connection.
    fn persist_pod_config(&self, pod_id: String, toml_text: String, err_conn: Option<ConnId>) {
        let Some(persister) = self.persister.clone() else {
            return;
        };
        let outbound = err_conn.and_then(|c| self.router.outbound(c));
        tokio::spawn(async move {
            if let Err(e) = persister.update_pod_config(&pod_id, &toml_text).await {
                warn!(pod_id = %pod_id, error = %e, "update_pod_config disk write failed");
                if let Some(tx) = outbound {
                    let _ = tx.send(ServerToClient::Error {
                        correlation_id: None,
                        thread_id: None,
                        message: format!("update_pod_config: disk write failed: {e}"),
                    });
                }
            }
        });
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
                    // Push the future but keep stepping — `AwaitingTools` dispatches all
                    // N tool calls in parallel before pausing.
                    let fut = io_dispatch::build_io_future(self, thread_id.to_string(), req);
                    pending_io.push(fut);
                }
                StepOutcome::Continue => continue,
                StepOutcome::Paused => break,
            }
        }
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
        if let Some(binding) = &bindings.host_env
            && let Some((provider, spec)) = self.resolve_binding(pod_id, binding)
        {
            let sid = HostEnvId::for_provider_spec(&provider, &spec);
            let _ = self.resources.release_host_env_user(&sid, thread_id);
            self.emit_host_env_updated(&sid);
            // The host-env MCP is scoped to the same (provider, spec)
            // dedup, so drop this thread from its user set too.
            let mcp_id = McpHostId::for_host_env(&sid);
            self.resources.remove_mcp_user(&mcp_id, thread_id);
            self.emit_mcp_host_updated(&mcp_id);
        }
        let backend_id = BackendId::for_name(&bindings.backend);
        self.resources.remove_backend_user(&backend_id, thread_id);
        for shared in &bindings.mcp_hosts {
            let mid = McpHostId::shared(shared);
            self.resources.remove_mcp_user(&mid, thread_id);
            self.emit_mcp_host_updated(&mid);
        }
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
        // The host env (and its MCP) is shared across threads with
        // matching (provider, spec); drop this thread from the user
        // set and only tear down when the count hits zero and the
        // entry isn't pinned. No-op for threads with no host_env
        // binding — nothing was ever provisioned.
        let Some(sandbox_id) = self.host_env_id_for_thread(thread_id) else {
            return;
        };
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

/// Build the `ThreadPendingApproval` events that a newly-subscribed client needs to
/// render the approval UI. Returns empty if the task isn't in AwaitingApproval.
fn pending_approvals_of(task: &Thread) -> Vec<ServerToClient> {
    let ThreadInternalState::AwaitingApproval {
        tool_uses,
        dispositions,
    } = &task.internal
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (tool_use, disposition) in tool_uses.iter().zip(dispositions.iter()) {
        if let ApprovalDisposition::Pending {
            approval_id,
            destructive,
            read_only,
        } = disposition
        {
            out.push(ServerToClient::ThreadPendingApproval {
                thread_id: task.id.clone(),
                approval_id: approval_id.clone(),
                tool_use_id: tool_use.tool_use_id.clone(),
                name: tool_use.name.clone(),
                args_preview: truncate(
                    serde_json::to_string(&tool_use.input).unwrap_or_default(),
                    200,
                ),
                destructive: *destructive,
                read_only: *read_only,
            });
        }
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

/// Synthesize a `PodConfig` from the server's runtime defaults — used to
/// bootstrap the in-memory default pod when no real pod.toml exists.
///
/// The resulting `[allow]` table reflects exactly what the server has wired
/// up at startup: every configured backend, every shared MCP host, and
/// optionally a single named host-env slot. When `default_host_env` is
/// `None`, the synthesized pod has an empty `allow.host_env` and its
/// `thread_defaults.host_env` is the empty string — threads inside the
/// default pod run with no host-env MCP connection.
pub fn build_default_pod_config(
    pod_id: &str,
    config: &ThreadConfig,
    default_backend: &str,
    default_host_env: Option<(String, HostEnvSpec)>,
    backend_names: &[String],
    shared_host_names: &[String],
) -> whisper_agent_protocol::PodConfig {
    use whisper_agent_protocol::{NamedHostEnv, PodAllow, PodConfig, PodLimits, ThreadDefaults};
    const HOST_ENV_NAME: &str = "default";
    let (host_env_entries, default_host_env_name) = match default_host_env {
        Some((provider, spec)) => (
            vec![NamedHostEnv {
                name: HOST_ENV_NAME.to_string(),
                provider,
                spec,
            }],
            HOST_ENV_NAME.to_string(),
        ),
        None => (Vec::new(), String::new()),
    };
    let now = chrono::Utc::now().to_rfc3339();
    PodConfig {
        name: pod_id.to_string(),
        description: Some(
            "Auto-synthesized default pod. Mutate via UpdatePodConfig (or hand-edit \
             pod.toml) to change thread defaults or tighten the allow cap."
                .into(),
        ),
        created_at: now,
        allow: PodAllow {
            backends: backend_names.to_vec(),
            mcp_hosts: shared_host_names.to_vec(),
            host_env: host_env_entries,
        },
        thread_defaults: ThreadDefaults {
            backend: default_backend.to_string(),
            model: config.model.clone(),
            system_prompt_file: "system_prompt.md".into(),
            max_tokens: config.max_tokens,
            max_turns: config.max_turns,
            approval_policy: config.approval_policy,
            host_env: default_host_env_name,
            mcp_hosts: shared_host_names.to_vec(),
        },
        limits: PodLimits::default(),
    }
}

/// Render the pod's `thread_defaults` (model, prompt, limits, policy) into
/// a fresh `ThreadConfig`. Binding-side defaults (backend, sandbox, shared
/// hosts) are produced separately by [`resolve_bindings_choice`].
fn base_thread_config_from_pod(pod: &Pod) -> ThreadConfig {
    let defaults = &pod.config.thread_defaults;
    ThreadConfig {
        model: defaults.model.clone(),
        system_prompt: pod.system_prompt.clone(),
        max_tokens: defaults.max_tokens,
        max_turns: defaults.max_turns,
        approval_policy: defaults.approval_policy,
    }
}

fn apply_config_override(base: ThreadConfig, ov: Option<ThreadConfigOverride>) -> ThreadConfig {
    let Some(ov) = ov else { return base };
    ThreadConfig {
        model: ov.model.unwrap_or(base.model),
        system_prompt: ov.system_prompt.unwrap_or(base.system_prompt),
        max_tokens: ov.max_tokens.unwrap_or(base.max_tokens),
        max_turns: ov.max_turns.unwrap_or(base.max_turns),
        approval_policy: ov.approval_policy.unwrap_or(base.approval_policy),
    }
}

/// Resolved binding-side choices for a fresh thread. Validated against
/// the pod's `[allow]` cap before construction.
struct ResolvedBindings {
    /// Backend catalog name. Empty string = "use server default".
    backend_name: String,
    /// Persisted on the thread. `None` when the pod declares no host
    /// envs (threads there get shared MCPs only). `Named` references
    /// a pod `[[allow.host_env]]` entry; `Inline` carries its own
    /// (provider, spec) for ad-hoc threads.
    host_env: Option<HostEnvBinding>,
    /// Catalog names of shared MCP hosts the thread is bound to.
    shared_host_names: Vec<String>,
}

/// Layer the request on top of pod `thread_defaults`, then verify each
/// chosen value against the pod's `[allow]` table. Inline host-env
/// requests carry both provider + spec, so no name lookup is needed.
fn resolve_bindings_choice(
    pod: &Pod,
    request: Option<ThreadBindingsRequest>,
) -> Result<ResolvedBindings, String> {
    let defaults = &pod.config.thread_defaults;
    let allow = &pod.config.allow;
    let request = request.unwrap_or_default();

    let backend_name = request.backend.unwrap_or_else(|| defaults.backend.clone());
    if !backend_name.is_empty() && !allow.backends.iter().any(|b| b == &backend_name) {
        return Err(format!(
            "backend `{}` not in pod `{}`'s allow.backends ({})",
            backend_name,
            pod.id,
            allow.backends.join(", ")
        ));
    }

    // Every user-originated binding resolves to a named entry in the
    // pod's allow.host_env table. Unknown names are rejected; the
    // pod's allow list is authoritative.
    let chosen_name: Option<String> = match request.host_env {
        Some(name) => Some(name),
        None if defaults.host_env.is_empty() => None,
        None => Some(defaults.host_env.clone()),
    };
    let host_env: Option<HostEnvBinding> = match chosen_name {
        None => None,
        Some(name) => {
            if !allow.host_env.iter().any(|nh| nh.name == name) {
                return Err(format!(
                    "host env `{name}` not in pod `{}`'s allow.host_env ({})",
                    pod.id,
                    if allow.host_env.is_empty() {
                        "empty".into()
                    } else {
                        allow
                            .host_env
                            .iter()
                            .map(|nh| nh.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    },
                ));
            }
            Some(HostEnvBinding::Named { name })
        }
    };

    let shared_host_names = request
        .mcp_hosts
        .unwrap_or_else(|| defaults.mcp_hosts.clone());
    for name in &shared_host_names {
        if !allow.mcp_hosts.iter().any(|h| h == name) {
            return Err(format!(
                "shared MCP host `{name}` not in pod `{}`'s allow.mcp_hosts ({})",
                pod.id,
                allow.mcp_hosts.join(", "),
            ));
        }
    }

    Ok(ResolvedBindings {
        backend_name,
        host_env,
        shared_host_names,
    })
}

// ---------- Run loop ----------

pub async fn run(mut scheduler: Scheduler, mut inbox: mpsc::UnboundedReceiver<SchedulerMsg>) {
    let mut pending_io: FuturesUnordered<SchedulerFuture> = FuturesUnordered::new();
    let mut gc_ticker = tokio::time::interval(Duration::from_secs(GC_TICK_SECS));
    // Skip the first immediate tick — `interval` fires at t=0 by default.
    gc_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    gc_ticker.tick().await;

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
                            scheduler.apply_io_completion(io);
                            scheduler.step_until_blocked(&thread_id, &mut pending_io);
                        }
                        SchedulerCompletion::Provision(prov) => {
                            scheduler.apply_provision_completion(prov, &mut pending_io);
                        }
                    }
                }
            }
            _ = gc_ticker.tick() => {
                scheduler.gc_tick();
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
