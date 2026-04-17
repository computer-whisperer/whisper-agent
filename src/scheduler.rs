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
    BackendSummary, ClientToServer, Message, ModelSummary, ResourceKind, HostEnvRebind,
    ServerToClient, ThreadBindings, ThreadBindingsPatch, ThreadBindingsRequest, ThreadConfig,
    ThreadConfigOverride, ThreadStateLabel,
};

use crate::audit::AuditLog;
use crate::io_dispatch::{
    self, IoCompletion, ProvisionCompletion, ProvisionPhase, ProvisionResult, SchedulerCompletion,
    SchedulerFuture,
};
use crate::mcp::{McpSession, ToolAnnotations, ToolDescriptor as McpTool};
use crate::model::ModelProvider;
use crate::persist::{LoadedState, Persister};
use crate::pod::{Pod, PodId};
use crate::resources::{
    BackendId, CompleteHostEnvOutcome, McpHostId, ResourceRegistry, HostEnvId,
};
use crate::sandbox::HostEnvProvider;
use crate::thread::{
    ApprovalDisposition, IoResult, OpId, StepOutcome, Thread, ThreadInternalState, derive_title,
    new_task_id,
};
use crate::thread_router::ThreadEventRouter;
use whisper_agent_protocol::HostEnvSpec;

pub type ConnId = u64;

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
    /// sandbox-provided URL. Pods don't model this — it's a server-level
    /// concern (where the filesystem MCP daemon lives) that pre-dates the
    /// pod-aware refactor. Stays here as a fallback until Phase 3 / 4
    /// rework makes per-thread bindings authoritative.
    default_mcp_host_url: String,
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
    /// Catalog of host-env providers, keyed by name. Always contains
    /// `bare`; configured entries arrive via `whisper-agent.toml`.
    host_env_registry: crate::sandbox::HostEnvRegistry,

    tasks: HashMap<String, Thread>,
    /// Owns the connection registry, subscription map, audit log, and the
    /// `ThreadEvent → ServerToClient` translation layer.
    router: ThreadEventRouter,

    /// Phase 3c: the registry is the single source of truth for MCP
    /// sessions, tool descriptors, and sandbox handles. Per-task
    /// `mcp_pools` / `tool_descriptors` / `sandbox_handles` are gone;
    /// the io_dispatch accessors walk this on demand.
    resources: ResourceRegistry,

    next_op_id: OpId,
    /// Tasks modified during the current scheduler-loop iteration. Flushed to the
    /// persister before the next iteration starts.
    dirty: HashSet<String>,
    /// Thread ids whose primary-MCP provisioning future is currently in
    /// flight (dispatched but not yet completed). Used to deduplicate
    /// `ensure_primary_mcp_provisioning` calls across the create_task /
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
        default_mcp_host_url: String,
        default_pod: Pod,
        host_id: String,
        backends: HashMap<String, BackendEntry>,
        default_backend: String,
        audit: AuditLog,
        host_env_registry: crate::sandbox::HostEnvRegistry,
        shared_host_configs: Vec<SharedHostConfig>,
    ) -> anyhow::Result<Self> {
        assert!(
            backends.contains_key(&default_backend),
            "default_backend must be in backends"
        );

        let mut resources = ResourceRegistry::new();
        for (name, entry) in &backends {
            resources.insert_backend(name.clone(), entry.kind.clone(), entry.default_model.clone());
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
            let id = resources.insert_shared_mcp_host(
                cfg.name.clone(),
                cfg.url.clone(),
                session,
            );
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
            default_mcp_host_url,
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
    pub(crate) fn host_env_provider(
        &self,
        name: &str,
    ) -> Option<&Arc<dyn HostEnvProvider>> {
        self.host_env_registry.get(name)
    }

    /// Server-level fallback MCP URL — used by `io_dispatch` when a
    /// thread's bound host env doesn't (yet) carry its own URL.
    pub(crate) fn default_mcp_url(&self) -> &str {
        &self.default_mcp_host_url
    }

    /// Look up the registry entry for a thread's bound host env.
    /// Returns `None` when the thread doesn't bind one (the `bare`
    /// path) or when `pre_register_host_env` somehow wasn't called.
    pub(crate) fn host_env_for_thread(
        &self,
        thread_id: &str,
    ) -> Option<&crate::resources::HostEnvEntry> {
        let task = self.tasks.get(thread_id)?;
        let id_str = task.bindings.host_env.as_ref()?;
        self.resources.host_envs.get(&HostEnvId(id_str.clone()))
    }

    /// Phase 3d.i: build the thread's effective tool catalog by walking
    /// `task.bindings.mcp_hosts` in precedence order — primary first
    /// (index 0), then each shared host the thread is bound to. Tool-name
    /// collisions resolve in favor of the earlier host (matching the prior
    /// `route_tool` behavior). Empty when the primary hasn't completed
    /// `list_tools` yet AND no shared hosts are bound.
    pub(crate) fn tool_descriptors(&self, thread_id: &str) -> Vec<McpTool> {
        let mut out: Vec<McpTool> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for host in self.bound_mcp_hosts(thread_id) {
            for tool in &host.tools {
                if seen.insert(tool.name.clone()) {
                    out.push(tool.clone());
                }
            }
        }
        out
    }

    /// Resolve which MCP session should receive a tool invocation. Walks
    /// the thread's bound hosts in precedence order and returns the first
    /// host whose tool catalog includes `tool_name`. Falls back to the
    /// primary's session when nothing matches — defensive, since the
    /// model shouldn't be calling tools we didn't advertise.
    pub(crate) fn route_tool(&self, thread_id: &str, tool_name: &str) -> Option<Arc<McpSession>> {
        for host in self.bound_mcp_hosts(thread_id) {
            if host.tools.iter().any(|t| t.name == tool_name)
                && let Some(s) = host.session.clone()
            {
                return Some(s);
            }
        }
        let primary_id = McpHostId::primary_for_task(thread_id);
        self.resources
            .mcp_hosts
            .get(&primary_id)
            .and_then(|e| e.session.clone())
    }

    /// Iterate the thread's bound MCP host entries in `bindings.mcp_hosts`
    /// order (primary first by convention; shared hosts in pod-declared
    /// order). Skips ids that don't have a registry entry yet — the
    /// per-thread primary is created lazily when McpConnect succeeds.
    fn bound_mcp_hosts(&self, thread_id: &str) -> Vec<&crate::resources::McpHostEntry> {
        let Some(task) = self.tasks.get(thread_id) else {
            return Vec::new();
        };
        task.bindings
            .mcp_hosts
            .iter()
            .filter_map(|raw| self.resources.mcp_hosts.get(&McpHostId(raw.clone())))
            .collect()
    }

    /// Walk the thread's bindings and return every bound resource id whose
    /// registry entry isn't yet Ready (or is missing entirely — happens
    /// post-restart for the per-thread primary MCP). Used by
    /// `send_user_message` to park the thread in `WaitingOnResources`
    /// when provisioning is still in flight; backends and shared MCP
    /// hosts (Ready from startup) drop out naturally.
    fn pending_resources_for(&self, thread_id: &str) -> Vec<String> {
        let Some(task) = self.tasks.get(thread_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        if let Some(sb) = &task.bindings.host_env {
            let id = HostEnvId(sb.clone());
            match self.resources.host_envs.get(&id) {
                Some(e) if e.state.is_ready() => {}
                _ => out.push(sb.clone()),
            }
        }
        for raw in &task.bindings.mcp_hosts {
            let id = McpHostId(raw.clone());
            match self.resources.mcp_hosts.get(&id) {
                Some(e) if e.state.is_ready() => {}
                _ => out.push(raw.clone()),
            }
        }
        out
    }

    /// Ensure a `provision_primary_mcp` future is in flight for this
    /// thread. Called from `create_task` (always dispatches) and from
    /// `send_user_message` (covers the post-restart case where the
    /// loaded thread's registry entry was lost). The
    /// `provisioning_in_flight` guard prevents double-dispatch when both
    /// paths run for the same thread within one create+message cycle.
    fn ensure_primary_mcp_provisioning(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if self.provisioning_in_flight.contains(thread_id) {
            return;
        }
        let primary_id = McpHostId::primary_for_task(thread_id);
        let needs_dispatch = match self.resources.mcp_hosts.get(&primary_id) {
            Some(entry) => !entry.state.is_ready(),
            None => true,
        };
        if !needs_dispatch {
            return;
        }
        // Make sure the registry has a Provisioning entry to attach to.
        // Calling pre_register on an existing Provisioning entry is a
        // no-op apart from touching last_used.
        self.resources
            .pre_register_primary_mcp_host(thread_id, self.default_mcp_host_url.clone());
        self.emit_mcp_host_updated(&primary_id);
        self.provisioning_in_flight.insert(thread_id.to_string());
        let fut = io_dispatch::provision_primary_mcp(self, thread_id.to_string());
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

        let new_host_env_id_str: Option<String> = match &patch.host_env {
            None => old_bindings.host_env.clone(),
            Some(HostEnvRebind::Clear) => None,
            Some(HostEnvRebind::Inline { binding }) => Some(
                HostEnvId::for_provider_spec(&binding.provider, &binding.spec).0,
            ),
        };

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
        let host_env_changed = new_host_env_id_str != old_bindings.host_env;
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
            // Release the old sandbox; tear it down if we were the last
            // user and the entry isn't pinned.
            if let Some(old_id_str) = &old_bindings.host_env {
                let old_id = HostEnvId(old_id_str.clone());
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
                                warn!(thread_id = %tid, error = %e, "sandbox teardown after rebind failed");
                            }
                        });
                    }
                } else {
                    self.emit_host_env_updated(&old_id);
                }
            }
            // Pre-register the new host env (if any).
            if let Some(HostEnvRebind::Inline { binding }) = &patch.host_env {
                let new_id = self.resources.pre_register_host_env(
                    thread_id,
                    binding.provider.clone(),
                    binding.spec.clone(),
                );
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

        // --- Phase 6: sandbox change → reconnect primary MCP ---
        if host_env_changed {
            let primary_id = McpHostId::primary_for_task(thread_id);
            self.resources.mark_mcp_torn_down(&primary_id);
            self.emit_mcp_host_updated(&primary_id);
            // Drop the in-flight guard so ensure_primary_mcp_provisioning
            // dispatches a fresh future. The old one is still in flight;
            // its completion will hit AlreadyCompleted on the new
            // primary entry's state and harmlessly drop the session.
            self.provisioning_in_flight.remove(thread_id);
            self.ensure_primary_mcp_provisioning(thread_id, pending_io);
        }

        // --- Phase 7: swap bindings + synthetic note + recompute waiting ---
        let primary_mcp_id = McpHostId::primary_for_task(thread_id).0;
        let mut new_mcp_hosts: Vec<String> = vec![primary_mcp_id];
        for name in &new_shared_hosts {
            new_mcp_hosts.push(McpHostId::shared(name).0);
        }
        let new_bindings = ThreadBindings {
            backend: new_backend,
            host_env: new_host_env_id_str.clone(),
            mcp_hosts: new_mcp_hosts,
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
            matches!(task.internal, ThreadInternalState::WaitingOnResources { .. })
        };

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
        for task in state.threads {
            // Re-pre-register the thread's host env so the registry knows
            // its (provider, spec) — the binding only carries the
            // deterministic id, so we look up the matching entry by
            // hashing each pod entry's (provider, spec) pair.
            if let Some(host_env_id_str) = task.bindings.host_env.clone() {
                let entry = self.pods.get(&task.pod_id).and_then(|p| {
                    p.config.allow.host_env.iter().find_map(|nh| {
                        if HostEnvId::for_provider_spec(&nh.provider, &nh.spec).0
                            == host_env_id_str
                        {
                            Some((nh.provider.clone(), nh.spec.clone()))
                        } else {
                            None
                        }
                    })
                });
                match entry {
                    Some((provider, spec)) => {
                        self.resources
                            .pre_register_host_env(&task.id, provider, spec);
                    }
                    None => warn!(
                        thread_id = %task.id,
                        host_env_id = %host_env_id_str,
                        "loaded thread's host env id has no matching entry in pod allow.host_env; skipping pre-register",
                    ),
                }
            }

            let backend_id = BackendId::for_name(self.resolve_backend_name(&task));
            self.resources.add_backend_user(&backend_id, &task.id);
            // bindings.mcp_hosts holds both the per-task primary id (which
            // doesn't have a registry entry until McpConnect lands) and the
            // shared host ids. Only the shared ones need user counts here.
            for raw in &task.bindings.mcp_hosts {
                if let Some(name) = raw.strip_prefix("mcp-shared-") {
                    self.resources
                        .add_mcp_user(&McpHostId::shared(name), &task.id);
                }
            }

            // Make sure the owning pod knows about this thread. If the pod
            // wasn't in `state.pods` (shouldn't happen — load_pod walks both
            // together) we drop the thread on the floor with a warning.
            if let Some(pod) = self.pods.get_mut(&task.pod_id) {
                pod.threads.insert(task.id.clone());
                self.tasks.insert(task.id.clone(), task);
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
    fn apply_input(&mut self, input: SchedulerMsg, pending_io: &mut FuturesUnordered<SchedulerFuture>) {
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
            ClientToServer::RemoveToolAllowlistEntry { thread_id, tool_name } => {
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
                if let Err(e) = crate::persist::validate_pod_id(&pod_id) {
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
                if let Err(e) = crate::persist::validate_pod_id(&pod_id) {
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
                // Refresh in-memory pod so subsequent create_task /
                // RebindThread paths see the new config without waiting
                // for the disk round-trip.
                if let Some(pod) = self.pods.get_mut(&pod_id) {
                    pod.config = parsed.clone();
                    pod.raw_toml = toml_text.clone();
                }
                let ev = ServerToClient::PodConfigUpdated {
                    pod_id: pod_id.clone(),
                    toml_text: toml_text.clone(),
                    parsed,
                    correlation_id,
                };
                for tx in self.router.outbound_snapshot() {
                    let _ = tx.send(ev.clone());
                }
                if let Some(persister) = self.persister.clone() {
                    let outbound = self.router.outbound(conn_id);
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
            }
            ClientToServer::ArchivePod { pod_id } => {
                if let Err(e) = crate::persist::validate_pod_id(&pod_id) {
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
                    let bindings = self
                        .tasks
                        .get(thread_id)
                        .map(|t| t.bindings.clone());
                    self.tasks.remove(thread_id);
                    self.dirty.remove(thread_id);
                    self.provisioning_in_flight.remove(thread_id);
                    self.router.drop_thread(thread_id);
                    if let Some(bindings) = bindings {
                        self.release_thread_resources(thread_id, &bindings);
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
        // (provider, spec) from creation onward — io_dispatch reads it
        // back at McpConnect time. The deterministic HostEnvId means a
        // second thread with the same (provider, spec) just adds itself
        // as a user to the existing entry. `bare` provider entries are
        // not pre-registered — the thread's `bindings.host_env` stays
        // None and io_dispatch falls back to the default MCP URL.
        let host_env_id = if resolved.host_env_provider == crate::sandbox::BARE_PROVIDER_NAME
            && matches!(resolved.host_env_spec, HostEnvSpec::None)
        {
            None
        } else {
            Some(self.resources.pre_register_host_env(
                &thread_id,
                resolved.host_env_provider.clone(),
                resolved.host_env_spec.clone(),
            ))
        };
        let primary_mcp_id = McpHostId::primary_for_task(&thread_id);
        let mut mcp_host_ids: Vec<String> = Vec::with_capacity(1 + resolved.shared_host_names.len());
        mcp_host_ids.push(primary_mcp_id.0.clone());
        for name in &resolved.shared_host_names {
            mcp_host_ids.push(McpHostId::shared(name).0);
        }
        let bindings = ThreadBindings {
            backend: resolved.backend_name.clone(),
            host_env: host_env_id.as_ref().map(|id| id.0.clone()),
            mcp_hosts: mcp_host_ids,
            tool_filter: None,
        };

        let task = Thread::new(thread_id.clone(), pod_id.clone(), config, bindings);
        let summary = task.summary();
        self.tasks.insert(thread_id.clone(), task);
        // Mirror the new thread into the owning pod's `threads` set.
        if let Some(pod) = self.pods.get_mut(&pod_id) {
            pod.threads.insert(thread_id.clone());
        }

        // Register this task as a user of its backend and of every shared
        // MCP host it references. The sandbox already counted us via
        // `pre_register_host_env`. The per-task primary MCP host gets a
        // Provisioning entry below — its session lands when the
        // provision_primary_mcp future completes.
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
        // Phase 3d.ii: kick off the resource provisioning future eagerly
        // so the sandbox + primary MCP land in parallel with the user
        // composing their first message. The thread itself stays in Idle
        // until submit_user_message; if provisioning is still in flight at
        // that point, the thread parks in WaitingOnResources.
        self.ensure_primary_mcp_provisioning(&thread_id, pending_io);

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
        // Post-restart: a loaded thread's primary MCP entry may not exist
        // yet (the in-memory registry doesn't survive process restart).
        // ensure_primary_mcp_provisioning is a no-op when create_task
        // already dispatched in this cycle.
        self.ensure_primary_mcp_provisioning(thread_id, pending_io);
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
        } = completion;
        let mut events = Vec::new();

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
        let primary_id = McpHostId::primary_for_task(&thread_id);
        match result {
            ProvisionResult::PrimaryMcpReady {
                host_env_id,
                host_env_handle,
                mcp_url,
                mcp_session,
                tools,
            } => {
                // 1. Host env first — attach the handle (or recognize the
                //    dedup race and tear ours down).
                if let Some(id) = host_env_id.as_ref() {
                    let outcome = self.resources.complete_host_env_provisioning(
                        id,
                        Some(mcp_url.clone()),
                        host_env_handle,
                    );
                    match outcome {
                        CompleteHostEnvOutcome::Completed => {
                            self.emit_host_env_updated(id);
                        }
                        CompleteHostEnvOutcome::AlreadyCompleted { unused_handle }
                        | CompleteHostEnvOutcome::NotPreRegistered { unused_handle } => {
                            self.emit_host_env_updated(id);
                            if let Some(mut h) = unused_handle {
                                tokio::spawn(async move {
                                    if let Err(e) = h.teardown().await {
                                        warn!(error = %e, "teardown of dedup-loser sandbox failed");
                                    }
                                });
                            }
                        }
                    }
                }

                // 2. Primary MCP host — attach the session + tools and flip
                //    Ready. The pre-registered entry already has the user
                //    set populated, so we don't re-add it here.
                let attached =
                    self.resources
                        .complete_primary_mcp_host(&primary_id, mcp_url, mcp_session);
                if !attached {
                    warn!(%thread_id, "primary mcp host already attached or missing — completion dropped");
                }
                let annotations: HashMap<String, ToolAnnotations> = tools
                    .iter()
                    .map(|t| (t.name.clone(), t.annotations.clone()))
                    .collect();
                self.resources
                    .populate_mcp_tools(&primary_id, tools, annotations);
                self.emit_mcp_host_updated(&primary_id);

                // 3. Nudge any threads that were waiting on these resources.
                //    Walk both ids — sandbox can have multiple waiters via
                //    dedup; primary MCP only has its single owner thread.
                let mut newly_ready: Vec<String> = Vec::new();
                let resource_ids: Vec<String> = host_env_id
                    .iter()
                    .map(|id| id.0.clone())
                    .chain(std::iter::once(primary_id.0.clone()))
                    .collect();
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
            ProvisionResult::PrimaryMcpFailed {
                phase,
                message,
                host_env_id,
            } => {
                warn!(%thread_id, phase = phase.as_str(), error = %message, "primary mcp provisioning failed");
                // Mark registry entries Errored so future threads observe
                // the failure and don't endlessly retry under the same id.
                self.resources
                    .mark_mcp_errored(&primary_id, message.clone());
                self.emit_mcp_host_updated(&primary_id);
                if matches!(phase, ProvisionPhase::HostEnv)
                    && let Some(id) = host_env_id.as_ref()
                {
                    self.resources.mark_host_env_errored(id, message.clone());
                    self.emit_host_env_updated(id);
                }
                // Fail the requesting thread.
                if let Some(task) = self.tasks.get_mut(&thread_id) {
                    task.fail(phase.as_str(), message);
                    let mut events = Vec::new();
                    events.push(crate::thread::ThreadEvent::Error {
                        message: format!("{}: {}", phase.as_str(), task.failure_detail().unwrap_or_default()),
                    });
                    self.mark_dirty(&thread_id);
                    self.router.dispatch_events(&thread_id, events);
                    self.teardown_host_env_if_terminal(&thread_id);
                }
            }
        }
    }

    fn annotations_for(&self, thread_id: &str) -> HashMap<String, ToolAnnotations> {
        let mut out = HashMap::new();
        for host in self.bound_mcp_hosts(thread_id) {
            for (name, ann) in &host.annotations {
                // First host wins on collision — matches `route_tool`'s
                // primary-first precedence so the policy decision agrees
                // with where the call will actually land.
                out.entry(name.clone()).or_insert_with(|| ann.clone());
            }
        }
        out
    }

    /// Advance `thread_id`'s state machine until it pauses, pushing new I/O to
    /// `pending_io` as requested.
    fn step_until_blocked(&mut self, thread_id: &str, pending_io: &mut FuturesUnordered<SchedulerFuture>) {
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
        bindings: &whisper_agent_protocol::ThreadBindings,
    ) {
        if let Some(sandbox) = &bindings.host_env {
            let sid = HostEnvId(sandbox.clone());
            let _ = self.resources.release_host_env_user(&sid, thread_id);
            self.emit_host_env_updated(&sid);
        }
        let backend_id = BackendId::for_name(&bindings.backend);
        self.resources.remove_backend_user(&backend_id, thread_id);
        for shared in &bindings.mcp_hosts {
            let mid = McpHostId::shared(shared);
            self.resources.remove_mcp_user(&mid, thread_id);
            self.emit_mcp_host_updated(&mid);
        }
        let primary = McpHostId::primary_for_task(thread_id);
        self.resources.mark_mcp_torn_down(&primary);
        self.emit_mcp_host_updated(&primary);
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
        // Phase 3c: the per-thread primary MCP host owns its session in
        // the registry; marking it torn down drops the session Arc and
        // the routing accessors stop returning it. Backend and
        // shared-host user counts are intentionally left alone — the
        // task itself still exists in `self.tasks` and the existing
        // code never removes terminal tasks.
        let primary_id = McpHostId::primary_for_task(thread_id);
        self.resources.mark_mcp_torn_down(&primary_id);
        self.emit_mcp_host_updated(&primary_id);

        // Phase 3d.i: the sandbox id lives on `task.bindings.host_env` —
        // O(1) lookup instead of the prior linear scan. The sandbox is
        // shared across threads with matching specs; drop this thread from
        // the user set and only tear it down when the count hits zero and
        // the entry isn't pinned.
        let Some(sandbox_id) = self
            .tasks
            .get(thread_id)
            .and_then(|t| t.bindings.host_env.clone())
            .map(HostEnvId)
        else {
            return;
        };
        let remaining = self
            .resources
            .release_host_env_user(&sandbox_id, thread_id);
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
/// up at startup: every configured backend, every shared MCP host, and a
/// single named sandbox slot carrying `sandbox_spec`. The pod isn't
/// restrictive — it allows everything the server can do today, so
/// `CreateThread` against the default pod has the same effective surface as
/// the pre-pod-aware code path.
#[allow(clippy::too_many_arguments)]
pub fn build_default_pod_config(
    pod_id: &str,
    config: &ThreadConfig,
    default_backend: &str,
    default_host_env_provider: &str,
    default_host_env_spec: HostEnvSpec,
    backend_names: &[String],
    shared_host_names: &[String],
) -> whisper_agent_protocol::PodConfig {
    use whisper_agent_protocol::{
        NamedHostEnv, PodAllow, PodConfig, PodLimits, ThreadDefaults,
    };
    let host_env_name = "default".to_string();
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
            host_env: vec![NamedHostEnv {
                name: host_env_name.clone(),
                provider: default_host_env_provider.to_string(),
                spec: default_host_env_spec,
            }],
        },
        thread_defaults: ThreadDefaults {
            backend: default_backend.to_string(),
            model: config.model.clone(),
            system_prompt_file: "system_prompt.md".into(),
            max_tokens: config.max_tokens,
            max_turns: config.max_turns,
            approval_policy: config.approval_policy,
            host_env: host_env_name,
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
    /// Catalog name of the host-env provider to dispatch to. Always
    /// populated — falls back to `bare` when the pod has no host envs
    /// configured.
    host_env_provider: String,
    /// Resolved host-env spec (used for `pre_register_host_env`).
    host_env_spec: HostEnvSpec,
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

    let (host_env_provider, host_env_spec) = match request.host_env {
        // Inline (provider, spec) override — accepted as-is for the
        // ad-hoc-bindings flow; the server-level catalog lookup at
        // dispatch time will still error if the provider name is
        // unknown.
        Some(inline) => (inline.provider, inline.spec),
        None => {
            if defaults.host_env.is_empty() {
                (
                    crate::sandbox::BARE_PROVIDER_NAME.to_string(),
                    HostEnvSpec::None,
                )
            } else {
                allow
                    .host_env
                    .iter()
                    .find(|nh| nh.name == defaults.host_env)
                    .map(|nh| (nh.provider.clone(), nh.spec.clone()))
                    .ok_or_else(|| {
                        format!(
                            "pod `{}` thread_defaults.host_env = `{}` but no matching [[allow.host_env]] entry",
                            pod.id, defaults.host_env,
                        )
                    })?
            }
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
        host_env_provider,
        host_env_spec,
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
