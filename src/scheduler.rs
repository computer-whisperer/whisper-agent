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
    BackendSummary, ClientToServer, ModelSummary, ServerToClient, ThreadConfig, ThreadConfigOverride,
    ThreadStateLabel,
};

use crate::audit::AuditLog;
use crate::io_dispatch::{self, IoCompletion, IoFuture};
use crate::mcp::{McpSession, ToolAnnotations, ToolDescriptor as McpTool};
use crate::model::ModelProvider;
use crate::persist::{LoadedState, Persister};
use crate::pod::{Pod, PodId};
use crate::resources::{BackendId, McpHostId, ResourceRegistry, SandboxId};
use crate::sandbox::SandboxProvider;
use crate::thread::{
    ApprovalDisposition, IoResult, OpId, StepOutcome, Thread, ThreadInternalState, derive_title,
    new_task_id,
};
use crate::thread_router::ThreadEventRouter;
use whisper_agent_protocol::SandboxSpec;

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

/// Per-task bundle of MCP sessions. Routing is built from the merged
/// `tools/list` response across every session in the pool — primary first so
/// its tool names win on collision. Empty `routing` means `tools/list` hasn't
/// completed yet.
struct McpPool {
    /// Session 0 is the per-task primary (sandbox MCP). Sessions 1.. are
    /// clones of shared singletons selected by the task's allowlist.
    sessions: Vec<Arc<McpSession>>,
    /// `tool_name -> sessions index`. Filled when `ListToolsSuccess` lands.
    routing: HashMap<String, usize>,
}

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
    sandbox_provider: Arc<dyn SandboxProvider>,

    tasks: HashMap<String, Thread>,
    /// Owns the connection registry, subscription map, audit log, and the
    /// `ThreadEvent → ServerToClient` translation layer.
    router: ThreadEventRouter,

    /// Per-task MCP host pool. Index 0 of each pool's `sessions` vec is the
    /// task's primary (sandbox-provisioned filesystem); index 1.. are clones
    /// of `shared_hosts` Arcs filtered by the task's `shared_mcp_hosts`
    /// allowlist.
    mcp_pools: HashMap<String, McpPool>,
    /// Connections to shared (singleton) MCP hosts, keyed by name. Connected
    /// once at scheduler start. Tasks reference these by name via
    /// `ThreadConfig.shared_mcp_hosts`.
    shared_hosts: HashMap<String, Arc<McpSession>>,
    /// Per-task tool catalog from the last `tools/list`. Stored in MCP shape (with
    /// annotations) so the approval layer can consult `read_only_hint` /
    /// `destructive_hint` at dispatch time; we convert to Anthropic shape on-demand
    /// when building a model request.
    tool_descriptors: HashMap<String, Vec<McpTool>>,

    /// Phase 1a shadow registry — populated alongside the per-task plumbing
    /// above, never read by the I/O dispatch path. Phase 1b exposes it on the
    /// wire; later phases promote it to the source of truth and the per-task
    /// fields above go away. See `docs/design_pod_thread_scheduler.md`.
    resources: ResourceRegistry,

    next_op_id: OpId,
    /// Tasks modified during the current scheduler-loop iteration. Flushed to the
    /// persister before the next iteration starts.
    dirty: HashSet<String>,
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
        sandbox_provider: Arc<dyn SandboxProvider>,
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

        let mut shared_hosts: HashMap<String, Arc<McpSession>> = HashMap::new();
        for cfg in shared_host_configs {
            info!(name = %cfg.name, url = %cfg.url, "connecting to shared MCP host");
            let session = Arc::new(McpSession::connect(&cfg.url).await.map_err(|e| {
                anyhow::anyhow!("shared MCP host `{}` ({}): {e}", cfg.name, cfg.url)
            })?);
            resources.insert_shared_mcp_host(cfg.name.clone(), cfg.url.clone(), session.clone());
            shared_hosts.insert(cfg.name, session);
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
            sandbox_provider,
            tasks: HashMap::new(),
            router: ThreadEventRouter::new(audit, host_id),
            mcp_pools: HashMap::new(),
            shared_hosts,
            tool_descriptors: HashMap::new(),
            resources,
            next_op_id: 1,
            dirty: HashSet::new(),
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

    fn emit_sandbox_created(&self, id: &SandboxId) {
        if let Some(snap) = self.resources.snapshot_sandbox(id) {
            self.router
                .broadcast_resource(ServerToClient::ResourceCreated { resource: snap });
        }
    }
    fn emit_sandbox_updated(&self, id: &SandboxId) {
        if let Some(snap) = self.resources.snapshot_sandbox(id) {
            self.router
                .broadcast_resource(ServerToClient::ResourceUpdated { resource: snap });
        }
    }
    fn emit_mcp_host_created(&self, id: &McpHostId) {
        if let Some(snap) = self.resources.snapshot_mcp_host(id) {
            self.router
                .broadcast_resource(ServerToClient::ResourceCreated { resource: snap });
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

    /// Returns the backend name the task is configured to use, falling back to the
    /// server's default if the task left it empty. Does NOT validate existence.
    pub(crate) fn resolve_backend_name<'a>(&'a self, task: &'a Thread) -> &'a str {
        if task.config.backend.is_empty() {
            &self.default_backend
        } else {
            &task.config.backend
        }
    }

    // ---------- Read-only accessors used by `io_dispatch` ----------

    pub(crate) fn task(&self, thread_id: &str) -> Option<&Thread> {
        self.tasks.get(thread_id)
    }

    pub(crate) fn backend(&self, name: &str) -> Option<&BackendEntry> {
        self.backends.get(name)
    }

    pub(crate) fn sandbox_provider(&self) -> &Arc<dyn SandboxProvider> {
        &self.sandbox_provider
    }

    pub(crate) fn tool_descriptors(&self, thread_id: &str) -> Option<&[McpTool]> {
        self.tool_descriptors.get(thread_id).map(|v| v.as_slice())
    }

    /// Snapshot of every MCP session in the task's pool, primary first. Used by
    /// `ListTools` to fan out the `tools/list` call across the pool.
    pub(crate) fn pool_sessions(&self, thread_id: &str) -> Option<Vec<Arc<McpSession>>> {
        self.mcp_pools.get(thread_id).map(|p| p.sessions.clone())
    }

    /// Resolve which pool session a tool call should be routed to. Falls back
    /// to the primary if the routing map hasn't been filled in (defensive —
    /// `ToolCall` follows `ListToolsSuccess`, so routing should always exist).
    pub(crate) fn route_tool(&self, thread_id: &str, tool_name: &str) -> Option<Arc<McpSession>> {
        let pool = self.mcp_pools.get(thread_id)?;
        let idx = *pool.routing.get(tool_name).unwrap_or(&0);
        pool.sessions.get(idx).cloned()
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
            // Phase 1a: mirror the registry registrations that create_task
            // would have done. Per-task primary MCP + sandbox entries don't
            // exist for persisted tasks yet — they're created lazily on the
            // next McpConnect, same as fresh tasks.
            let backend_id = BackendId::for_name(self.resolve_backend_name(&task));
            self.resources.add_backend_user(&backend_id, &task.id);
            for name in &task.config.shared_mcp_hosts {
                self.resources
                    .add_mcp_user(&McpHostId::shared(name), &task.id);
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
    fn apply_input(&mut self, input: SchedulerMsg, pending_io: &mut FuturesUnordered<IoFuture>) {
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
        pending_io: &mut FuturesUnordered<IoFuture>,
    ) {
        match msg {
            ClientToServer::CreateThread {
                correlation_id,
                pod_id,
                initial_message,
                config_override,
            } => match self.create_task(conn_id, correlation_id.clone(), pod_id, config_override) {
                Ok(thread_id) => {
                    self.mark_dirty(&thread_id);
                    self.send_user_message(&thread_id, initial_message);
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
                    self.send_user_message(&thread_id, text);
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
                    self.teardown_sandbox_if_terminal(&thread_id);
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
            ClientToServer::CancelThread { thread_id } => {
                if let Some(task) = self.tasks.get_mut(&thread_id) {
                    task.cancel();
                    self.mark_dirty(&thread_id);
                    self.router
                        .broadcast_task_list(ServerToClient::ThreadStateChanged {
                            thread_id: thread_id.clone(),
                            state: ThreadStateLabel::Cancelled,
                        });
                    self.teardown_sandbox_if_terminal(&thread_id);
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
            ClientToServer::ListPods { correlation_id } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                tokio::spawn(async move {
                    match persister.list_pods().await {
                        Ok(pods) => {
                            let _ = outbound.send(ServerToClient::PodList {
                                correlation_id,
                                pods,
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
                config,
            } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                let broadcast = self.router.outbound_snapshot();
                tokio::spawn(async move {
                    match persister.create_pod(&pod_id, config).await {
                        Ok(summary) => {
                            let ev = ServerToClient::PodCreated {
                                pod: summary,
                                correlation_id,
                            };
                            for tx in broadcast {
                                let _ = tx.send(ev.clone());
                            }
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("create_pod: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::UpdatePodConfig {
                correlation_id,
                pod_id,
                toml_text,
            } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                let broadcast = self.router.outbound_snapshot();
                tokio::spawn(async move {
                    match persister.update_pod_config(&pod_id, &toml_text).await {
                        Ok(parsed) => {
                            let ev = ServerToClient::PodConfigUpdated {
                                pod_id,
                                toml_text,
                                parsed,
                                correlation_id,
                            };
                            for tx in broadcast {
                                let _ = tx.send(ev.clone());
                            }
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("update_pod_config: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::ArchivePod { pod_id } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                let broadcast = self.router.outbound_snapshot();
                tokio::spawn(async move {
                    match persister.archive_pod(&pod_id).await {
                        Ok(()) => {
                            let ev = ServerToClient::PodArchived { pod_id };
                            for tx in broadcast {
                                let _ = tx.send(ev.clone());
                            }
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id: None,
                                thread_id: None,
                                message: format!("archive_pod: {e}"),
                            });
                        }
                    }
                });
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
    ) -> Result<String, String> {
        // Resolve which pod this thread lands in. None routes to the
        // server's default pod.
        let pod_id = pod_id.unwrap_or_else(|| self.default_pod_id.clone());
        let pod = self
            .pods
            .get(&pod_id)
            .ok_or_else(|| format!("unknown pod `{pod_id}`"))?;

        // Build the base ThreadConfig from the pod's thread_defaults, then
        // layer the override on top. Server-level fallback URL fills in the
        // mcp_host_url field — pods don't model it yet.
        let base = base_thread_config_from_pod(pod, &self.default_mcp_host_url);
        let config = apply_override(base, config_override);

        // Validate against the pod's allowlist: backend and shared MCP host
        // names must each appear in `[allow]`. Sandbox spec validation
        // arrives in Phase 3 with the resource resolver.
        validate_thread_against_pod(&config, pod)?;

        // Validate shared MCP host names against the server catalog too —
        // a name allowed by the pod but not actually wired up at the server
        // is a misconfiguration we want to surface at create-time.
        for name in &config.shared_mcp_hosts {
            if !self.shared_hosts.contains_key(name) {
                return Err(format!(
                    "shared MCP host `{name}` not configured on this server (pod `{pod_id}` allows it but no upstream is wired up)",
                ));
            }
        }

        let thread_id = new_task_id();
        let task = Thread::new(thread_id.clone(), pod_id.clone(), config);
        let summary = task.summary();
        self.tasks.insert(thread_id.clone(), task);
        // Mirror the new thread into the owning pod's `threads` set.
        if let Some(pod) = self.pods.get_mut(&pod_id) {
            pod.threads.insert(thread_id.clone());
        }

        // Phase 1a: register this task as a user of its backend and of every
        // shared MCP host it references. Counts are added once at creation;
        // the teardown path clears per-task entries (primary MCP + sandbox)
        // but leaves backend/shared-host associations alone — Completed tasks
        // keep their bindings so follow-up messages stay routable.
        let task_ref = self
            .tasks
            .get(&thread_id)
            .expect("just-inserted task is present");
        let backend_id = BackendId::for_name(self.resolve_backend_name(task_ref));
        let shared_names = task_ref.config.shared_mcp_hosts.clone();
        self.resources.add_backend_user(&backend_id, &thread_id);
        self.emit_backend_updated(&backend_id);
        for name in &shared_names {
            let host_id = McpHostId::shared(name);
            self.resources.add_mcp_user(&host_id, &thread_id);
            self.emit_mcp_host_updated(&host_id);
        }

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

    fn send_user_message(&mut self, thread_id: &str, text: String) {
        self.mark_dirty(thread_id);
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

        let mcp_connected = self.mcp_pools.contains_key(thread_id);
        let tools_listed = self.tool_descriptors.contains_key(thread_id);
        let new_state = {
            let task = self.tasks.get_mut(thread_id).expect("task exists");
            task.submit_user_message(text, mcp_connected, tools_listed);
            task.public_state()
        };
        self.router
            .broadcast_task_list(ServerToClient::ThreadStateChanged {
                thread_id: thread_id.to_string(),
                state: new_state,
            });
    }

    /// Apply one I/O completion to its task and dispatch any resulting events.
    fn apply_io_completion(&mut self, completion: IoCompletion) {
        let IoCompletion {
            thread_id,
            op_id,
            result,
        } = completion;
        let mut events = Vec::new();

        // Surface any IO-level error to the operator even if no client is subscribed.
        // The task's own Failed state captures it for the wire, but operator-visible
        // logs are how we catch misconfigured backends / unreachable endpoints without
        // round-tripping through the UI.
        match &result {
            IoResult::McpConnect(Err(e)) => {
                warn!(%thread_id, op_id, error = %e, "mcp connect failed");
            }
            IoResult::ListTools(Err(e)) => {
                warn!(%thread_id, op_id, error = %e, "list_tools failed");
            }
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

        // Handle IoResult::McpConnect / ListTools specially because the
        // session Arcs and routing map are scheduler-owned resources that
        // get peeled off before the task sees the slim variant.
        let result = match result {
            IoResult::McpConnectSuccess {
                session,
                sandbox_handle,
            } => {
                // Build the task's pool: primary first, then shared singletons
                // selected by the task's allowlist. Unknown names are warn-skipped
                // — they shouldn't survive create_task validation, but a persisted
                // task whose host was removed since save lands here.
                let (allow, sandbox_spec) = self
                    .tasks
                    .get(&thread_id)
                    .map(|t| (t.config.shared_mcp_hosts.clone(), t.config.sandbox.clone()))
                    .unwrap_or_default();
                let mut sessions = vec![session.clone()];
                for name in &allow {
                    match self.shared_hosts.get(name) {
                        Some(s) => sessions.push(s.clone()),
                        None => warn!(
                            %thread_id, host = %name,
                            "task references unknown shared MCP host (skipping)",
                        ),
                    }
                }
                // Phase 1a: mirror the just-acquired primary MCP + sandbox
                // into the resource registry. Phase 1b: emit ResourceCreated
                // so subscribed clients see them appear. Phase 3a: the
                // sandbox handle moves into the registry too, replacing the
                // dropped `scheduler.sandbox_handles` map.
                let primary_url = sandbox_handle
                    .as_ref()
                    .and_then(|h| h.mcp_url().map(str::to_string))
                    .unwrap_or_else(|| {
                        self.tasks
                            .get(&thread_id)
                            .map(|t| t.config.mcp_host_url.clone())
                            .unwrap_or_default()
                    });
                let primary_id =
                    self.resources
                        .insert_primary_mcp_host(&thread_id, primary_url, session);
                self.emit_mcp_host_created(&primary_id);
                if sandbox_handle.is_some() {
                    let sandbox_id = self.resources.insert_sandbox_for_task(
                        &thread_id,
                        sandbox_spec,
                        sandbox_handle,
                    );
                    self.emit_sandbox_created(&sandbox_id);
                }

                self.mcp_pools.insert(
                    thread_id.clone(),
                    McpPool {
                        sessions,
                        routing: HashMap::new(),
                    },
                );
                IoResult::McpConnect(Ok(()))
            }
            IoResult::ListToolsSuccess { tools, routing } => {
                if let Some(pool) = self.mcp_pools.get_mut(&thread_id) {
                    pool.routing = routing;
                }
                // Phase 1a: mirror the per-task primary's tools into the
                // registry. The merged list contains tools from every session
                // in the pool; we only attribute to the primary entry here
                // because in Phase 1a there's no shared-host tools-list flow
                // (shared hosts publish tools too, but that wiring lands in
                // Phase 1b alongside the wire surface).
                let primary_id = McpHostId::primary_for_task(&thread_id);
                let annotations: HashMap<String, ToolAnnotations> = tools
                    .iter()
                    .map(|t| (t.name.clone(), t.annotations.clone()))
                    .collect();
                self.resources
                    .populate_mcp_tools(&primary_id, tools.clone(), annotations);
                self.emit_mcp_host_updated(&primary_id);
                self.tool_descriptors.insert(thread_id.clone(), tools);
                IoResult::ListTools(Ok(()))
            }
            other => other,
        };

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
        self.teardown_sandbox_if_terminal(&thread_id);
    }

    fn annotations_for(&self, thread_id: &str) -> HashMap<String, ToolAnnotations> {
        self.tool_descriptors
            .get(thread_id)
            .map(|tools| {
                tools
                    .iter()
                    .map(|t| (t.name.clone(), t.annotations.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Advance `thread_id`'s state machine until it pauses, pushing new I/O to
    /// `pending_io` as requested.
    fn step_until_blocked(&mut self, thread_id: &str, pending_io: &mut FuturesUnordered<IoFuture>) {
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
    fn teardown_sandbox_if_terminal(&mut self, thread_id: &str) {
        // Completed is NOT terminal — the task can receive follow-up messages.
        // Only tear down on truly irreversible states.
        let is_terminal = self.tasks.get(thread_id).is_some_and(|t| {
            matches!(
                t.public_state(),
                ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
            )
        });
        if is_terminal {
            // Pool's primary session points at the sandbox MCP host that's
            // about to die. Drop the pool too — shared session Arcs survive
            // because `self.shared_hosts` still holds them.
            self.mcp_pools.remove(thread_id);
            // Phase 1a: mark the per-task primary MCP + sandbox as torn down
            // in the registry so inspectability tools see the right state.
            // Backend and shared-host user counts are intentionally left
            // alone — the task itself still exists in `self.tasks` and the
            // existing code never removes terminal tasks.
            let primary_id = McpHostId::primary_for_task(thread_id);
            let sandbox_id = SandboxId::for_task(thread_id);
            self.resources.mark_mcp_torn_down(&primary_id);
            self.emit_mcp_host_updated(&primary_id);
            // Pull the handle out of the registry before flipping state so
            // we don't race a future inspector that consults the entry.
            let handle = self.resources.take_sandbox_handle(&sandbox_id);
            self.resources.mark_sandbox_torn_down(&sandbox_id);
            self.emit_sandbox_updated(&sandbox_id);
            if let Some(mut handle) = handle {
                let tid = thread_id.to_string();
                tokio::spawn(async move {
                    if let Err(e) = handle.teardown().await {
                        warn!(thread_id = %tid, error = %e, "sandbox teardown failed");
                    }
                });
            }
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
/// single named sandbox slot pulled from `default_task_config.sandbox`. The
/// pod isn't restrictive — it allows everything the server can do today, so
/// `CreateThread` against the default pod has the same effective surface as
/// the pre-pod-aware code path.
pub fn build_default_pod_config(
    pod_id: &str,
    defaults: &ThreadConfig,
    backend_names: &[String],
    shared_host_names: &[String],
) -> whisper_agent_protocol::PodConfig {
    use whisper_agent_protocol::{
        NamedSandboxSpec, PodAllow, PodConfig, PodLimits, ThreadDefaults,
    };
    let sandbox_name = "default".to_string();
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
            sandbox: vec![NamedSandboxSpec {
                name: sandbox_name.clone(),
                spec: defaults.sandbox.clone(),
            }],
        },
        thread_defaults: ThreadDefaults {
            backend: defaults.backend.clone(),
            model: defaults.model.clone(),
            system_prompt_file: "system_prompt.md".into(),
            max_tokens: defaults.max_tokens,
            max_turns: defaults.max_turns,
            approval_policy: defaults.approval_policy,
            sandbox: sandbox_name,
            mcp_hosts: defaults.shared_mcp_hosts.clone(),
        },
        limits: PodLimits::default(),
    }
}

/// Render the pod's `thread_defaults` into a fresh `ThreadConfig`.
///
/// `default_mcp_url` is the server's filesystem-MCP fallback — pods don't
/// model it (it's where the MCP daemon lives, a server-level concern). The
/// resolved `sandbox` field is filled by looking the named entry up in
/// `pod.config.allow.sandbox`; if not present, defaults to `SandboxSpec::None`
/// (matches today's behavior when no sandbox is configured).
fn base_thread_config_from_pod(pod: &Pod, default_mcp_url: &str) -> ThreadConfig {
    let defaults = &pod.config.thread_defaults;
    let sandbox = pod
        .config
        .allow
        .sandbox
        .iter()
        .find(|s| s.name == defaults.sandbox)
        .map(|s| s.spec.clone())
        .unwrap_or(SandboxSpec::None);
    ThreadConfig {
        backend: defaults.backend.clone(),
        model: defaults.model.clone(),
        system_prompt: pod.system_prompt.clone(),
        mcp_host_url: default_mcp_url.to_string(),
        max_tokens: defaults.max_tokens,
        max_turns: defaults.max_turns,
        approval_policy: defaults.approval_policy,
        sandbox,
        shared_mcp_hosts: defaults.mcp_hosts.clone(),
    }
}

/// Reject thread configs that escape the pod's `[allow]` cap.
///
/// Today this checks `backend` and `shared_mcp_hosts` against the pod's
/// allowed names. Sandbox-spec validation is deferred to Phase 3 alongside
/// the resource resolver (the resolver is the natural place to verify a
/// requested SandboxSpec deep-equals one of `[[allow.sandbox]]`).
fn validate_thread_against_pod(config: &ThreadConfig, pod: &Pod) -> Result<(), String> {
    if !config.backend.is_empty()
        && !pod.config.allow.backends.iter().any(|b| b == &config.backend)
    {
        return Err(format!(
            "backend `{}` not in pod `{}`'s allow.backends ({})",
            config.backend,
            pod.id,
            pod.config.allow.backends.join(", ")
        ));
    }
    for name in &config.shared_mcp_hosts {
        if !pod.config.allow.mcp_hosts.iter().any(|h| h == name) {
            return Err(format!(
                "shared MCP host `{name}` not in pod `{}`'s allow.mcp_hosts ({})",
                pod.id,
                pod.config.allow.mcp_hosts.join(", ")
            ));
        }
    }
    Ok(())
}

fn apply_override(base: ThreadConfig, ov: Option<ThreadConfigOverride>) -> ThreadConfig {
    let Some(ov) = ov else { return base };
    // If the override picks a different backend without specifying a model, don't
    // inherit the model from the default backend — leave it empty so build_model_request
    // resolves via the picked backend's default_model at call time (or passes empty to
    // a single-model endpoint that ignores the field).
    let backend_changed = ov.backend.as_deref().is_some_and(|b| b != base.backend);
    let model = match ov.model {
        Some(m) => m,
        None if backend_changed => String::new(),
        None => base.model.clone(),
    };
    ThreadConfig {
        backend: ov.backend.unwrap_or(base.backend),
        model,
        system_prompt: ov.system_prompt.unwrap_or(base.system_prompt),
        mcp_host_url: ov.mcp_host_url.unwrap_or(base.mcp_host_url),
        max_tokens: ov.max_tokens.unwrap_or(base.max_tokens),
        max_turns: ov.max_turns.unwrap_or(base.max_turns),
        approval_policy: ov.approval_policy.unwrap_or(base.approval_policy),
        sandbox: ov.sandbox.unwrap_or(base.sandbox),
        shared_mcp_hosts: ov.shared_mcp_hosts.unwrap_or(base.shared_mcp_hosts),
    }
}

// ---------- Run loop ----------

pub async fn run(mut scheduler: Scheduler, mut inbox: mpsc::UnboundedReceiver<SchedulerMsg>) {
    let mut pending_io: FuturesUnordered<IoFuture> = FuturesUnordered::new();

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
                    let thread_id = completion.thread_id.clone();
                    scheduler.apply_io_completion(completion);
                    scheduler.step_until_blocked(&thread_id, &mut pending_io);
                }
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

async fn next_completion(pending: &mut FuturesUnordered<IoFuture>) -> Option<IoCompletion> {
    pending.next().await
}
