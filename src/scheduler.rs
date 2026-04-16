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
//!   1. `apply_*` — mutate task state, collect [`TaskEvent`]s from the state machine.
//!   2. `dispatch_task_events` — translate task events to wire events and broadcast.
//!   3. `step_until_blocked` — keep calling `task.step()` until the task requests I/O
//!      or pauses; pushes fresh I/O futures to the `FuturesUnordered`.
//!
//! Tasks-as-data discipline: state mutation happens exclusively in this module (via
//! `task.step()` / `task.apply_io_result()`). No other code path writes to a Task.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use whisper_agent_protocol::{
    BackendSummary, ClientToServer, ModelSummary, ServerToClient, TaskConfig, TaskConfigOverride,
    TaskStateLabel,
};

use crate::audit::{AuditLog, ToolCallEntry, ToolCallOutcome};
use crate::io_dispatch::{self, IoCompletion, IoFuture};
use crate::mcp::{McpSession, ToolAnnotations, ToolDescriptor as McpTool};
use crate::model::ModelProvider;
use crate::sandbox::{SandboxHandle, SandboxProvider};
use crate::persist::Persister;
use crate::task::{
    ApprovalDisposition, IoResult, OpId, StepOutcome, Task, TaskEvent, TaskInternalState,
    derive_title, new_task_id,
};

pub type ConnId = u64;

/// Messages accepted by the scheduler inbox.
#[derive(Debug)]
pub enum SchedulerMsg {
    ClientMessage { conn_id: ConnId, msg: ClientToServer },
    RegisterClient {
        conn_id: ConnId,
        outbound: mpsc::UnboundedSender<ServerToClient>,
    },
    UnregisterClient { conn_id: ConnId },
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
/// `TaskConfig.shared_mcp_hosts`.
#[derive(Debug, Clone)]
pub struct SharedHostConfig {
    /// Stable name; what `TaskConfig.shared_mcp_hosts` references.
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
    default_config: TaskConfig,
    host_id: String,
    /// Named backends: `TaskConfig.backend` resolves against this map.
    backends: HashMap<String, BackendEntry>,
    /// Fallback when a task doesn't specify a backend. Must be a key in `backends`.
    default_backend: String,
    audit: AuditLog,
    persister: Option<Persister>,
    sandbox_provider: Arc<dyn SandboxProvider>,

    tasks: HashMap<String, Task>,
    clients: HashMap<ConnId, mpsc::UnboundedSender<ServerToClient>>,
    subscriptions: HashMap<String, HashSet<ConnId>>,

    /// Per-task MCP host pool. Index 0 of each pool's `sessions` vec is the
    /// task's primary (sandbox-provisioned filesystem); index 1.. are clones
    /// of `shared_hosts` Arcs filtered by the task's `shared_mcp_hosts`
    /// allowlist.
    mcp_pools: HashMap<String, McpPool>,
    sandbox_handles: HashMap<String, Box<dyn SandboxHandle>>,
    /// Connections to shared (singleton) MCP hosts, keyed by name. Connected
    /// once at scheduler start. Tasks reference these by name via
    /// `TaskConfig.shared_mcp_hosts`.
    shared_hosts: HashMap<String, Arc<McpSession>>,
    /// Per-task tool catalog from the last `tools/list`. Stored in MCP shape (with
    /// annotations) so the approval layer can consult `read_only_hint` /
    /// `destructive_hint` at dispatch time; we convert to Anthropic shape on-demand
    /// when building a model request.
    tool_descriptors: HashMap<String, Vec<McpTool>>,

    next_op_id: OpId,
    /// Tasks modified during the current scheduler-loop iteration. Flushed to the
    /// persister before the next iteration starts.
    dirty: HashSet<String>,
}

impl Scheduler {
    /// Async because we eagerly connect to every configured shared MCP host at
    /// startup — surfacing misconfiguration at server boot rather than at first
    /// task creation. Returns Err if any shared host fails to handshake.
    pub async fn new(
        default_config: TaskConfig,
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

        let mut shared_hosts: HashMap<String, Arc<McpSession>> = HashMap::new();
        for cfg in shared_host_configs {
            info!(name = %cfg.name, url = %cfg.url, "connecting to shared MCP host");
            let session = McpSession::connect(&cfg.url).await.map_err(|e| {
                anyhow::anyhow!("shared MCP host `{}` ({}): {e}", cfg.name, cfg.url)
            })?;
            shared_hosts.insert(cfg.name, Arc::new(session));
        }

        Ok(Self {
            default_config,
            host_id,
            backends,
            default_backend,
            audit,
            persister: None,
            sandbox_provider,
            tasks: HashMap::new(),
            clients: HashMap::new(),
            subscriptions: HashMap::new(),
            mcp_pools: HashMap::new(),
            sandbox_handles: HashMap::new(),
            shared_hosts,
            tool_descriptors: HashMap::new(),
            next_op_id: 1,
            dirty: HashSet::new(),
        })
    }

    /// Returns the backend name the task is configured to use, falling back to the
    /// server's default if the task left it empty. Does NOT validate existence.
    pub(crate) fn resolve_backend_name<'a>(&'a self, task: &'a Task) -> &'a str {
        if task.config.backend.is_empty() {
            &self.default_backend
        } else {
            &task.config.backend
        }
    }

    // ---------- Read-only accessors used by `io_dispatch` ----------

    pub(crate) fn task(&self, task_id: &str) -> Option<&Task> {
        self.tasks.get(task_id)
    }

    pub(crate) fn backend(&self, name: &str) -> Option<&BackendEntry> {
        self.backends.get(name)
    }

    pub(crate) fn sandbox_provider(&self) -> &Arc<dyn SandboxProvider> {
        &self.sandbox_provider
    }

    pub(crate) fn tool_descriptors(&self, task_id: &str) -> Option<&[McpTool]> {
        self.tool_descriptors.get(task_id).map(|v| v.as_slice())
    }

    /// Snapshot of every MCP session in the task's pool, primary first. Used by
    /// `ListTools` to fan out the `tools/list` call across the pool.
    pub(crate) fn pool_sessions(&self, task_id: &str) -> Option<Vec<Arc<McpSession>>> {
        self.mcp_pools.get(task_id).map(|p| p.sessions.clone())
    }

    /// Resolve which pool session a tool call should be routed to. Falls back
    /// to the primary if the routing map hasn't been filled in (defensive —
    /// `ToolCall` follows `ListToolsSuccess`, so routing should always exist).
    pub(crate) fn route_tool(&self, task_id: &str, tool_name: &str) -> Option<Arc<McpSession>> {
        let pool = self.mcp_pools.get(task_id)?;
        let idx = *pool.routing.get(tool_name).unwrap_or(&0);
        pool.sessions.get(idx).cloned()
    }

    pub fn with_persister(mut self, persister: Persister) -> Self {
        self.persister = Some(persister);
        self
    }

    /// Seed the scheduler with previously-persisted tasks. The persister should have
    /// already transitioned any in-flight internal states to Failed before handoff.
    pub fn load_tasks(&mut self, tasks: Vec<Task>) {
        for task in tasks {
            self.tasks.insert(task.id.clone(), task);
        }
    }

    fn mark_dirty(&mut self, task_id: &str) {
        if self.persister.is_some() {
            self.dirty.insert(task_id.to_string());
        }
    }

    async fn flush_dirty(&mut self) {
        let Some(persister) = &self.persister else {
            self.dirty.clear();
            return;
        };
        let dirty = std::mem::take(&mut self.dirty);
        for task_id in dirty {
            let Some(task) = self.tasks.get(&task_id) else {
                continue;
            };
            if let Err(e) = persister.flush(task).await {
                error!(task_id = %task_id, error = %e, "persist flush failed");
            }
        }
    }

    /// Apply an inbox message. Returns the task_ids (usually 0 or 1) whose state may
    /// have advanced and need `step_until_blocked` run.
    fn apply_input(&mut self, input: SchedulerMsg, pending_io: &mut FuturesUnordered<IoFuture>) {
        match input {
            SchedulerMsg::RegisterClient { conn_id, outbound } => {
                self.clients.insert(conn_id, outbound);
            }
            SchedulerMsg::UnregisterClient { conn_id } => {
                self.clients.remove(&conn_id);
                for subs in self.subscriptions.values_mut() {
                    subs.remove(&conn_id);
                }
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
            ClientToServer::CreateTask {
                correlation_id,
                initial_message,
                config_override,
            } => {
                match self.create_task(conn_id, correlation_id.clone(), config_override) {
                    Ok(task_id) => {
                        self.mark_dirty(&task_id);
                        self.send_user_message(&task_id, initial_message);
                        self.step_until_blocked(&task_id, pending_io);
                    }
                    Err(e) => {
                        warn!(error = %e, conn_id, "create_task rejected");
                        self.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                task_id: None,
                                message: format!("create_task: {e}"),
                            },
                        );
                    }
                }
            }
            ClientToServer::SendUserMessage { task_id, text } => {
                if self.tasks.contains_key(&task_id) {
                    self.send_user_message(&task_id, text);
                    self.step_until_blocked(&task_id, pending_io);
                } else {
                    self.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            task_id: Some(task_id),
                            message: "unknown task".into(),
                        },
                    );
                }
            }
            ClientToServer::ApprovalDecision {
                task_id,
                approval_id,
                decision,
                remember,
            } => {
                let mut events = Vec::new();
                let known = if let Some(task) = self.tasks.get_mut(&task_id) {
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
                    self.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            task_id: Some(task_id),
                            message: "unknown task".into(),
                        },
                    );
                } else {
                    self.mark_dirty(&task_id);
                    self.dispatch_task_events(&task_id, events);
                    self.teardown_sandbox_if_terminal(&task_id);
                    self.step_until_blocked(&task_id, pending_io);
                }
            }
            ClientToServer::RemoveToolAllowlistEntry { task_id, tool_name } => {
                let removed = self
                    .tasks
                    .get_mut(&task_id)
                    .map(|t| (t.remove_from_allowlist(&tool_name), t.allowlist_snapshot()));
                match removed {
                    Some((true, snapshot)) => {
                        self.mark_dirty(&task_id);
                        self.broadcast_to_subscribers(
                            &task_id,
                            ServerToClient::TaskAllowlistUpdated {
                                task_id: task_id.clone(),
                                tool_allowlist: snapshot,
                            },
                        );
                    }
                    Some((false, _)) => { /* nothing to remove — silent no-op */ }
                    None => self.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            task_id: Some(task_id),
                            message: "unknown task".into(),
                        },
                    ),
                }
            }
            ClientToServer::CancelTask { task_id } => {
                if let Some(task) = self.tasks.get_mut(&task_id) {
                    task.cancel();
                    self.mark_dirty(&task_id);
                    self.broadcast_task_list(ServerToClient::TaskStateChanged {
                        task_id: task_id.clone(),
                        state: TaskStateLabel::Cancelled,
                    });
                    self.teardown_sandbox_if_terminal(&task_id);
                }
            }
            ClientToServer::ArchiveTask { task_id } => {
                if let Some(task) = self.tasks.get_mut(&task_id) {
                    task.archived = true;
                    task.touch();
                    self.mark_dirty(&task_id);
                    self.broadcast_task_list(ServerToClient::TaskArchived { task_id });
                }
            }
            ClientToServer::SubscribeToTask { task_id } => {
                if let Some(task) = self.tasks.get(&task_id) {
                    self.subscriptions
                        .entry(task_id.clone())
                        .or_default()
                        .insert(conn_id);
                    let snapshot = task.snapshot();
                    // Rehydrate any still-pending approvals so the newly-subscribed
                    // client can render the approval UI. The snapshot itself doesn't
                    // carry approval state.
                    let pending = pending_approvals_of(task);
                    self.send_to_client(
                        conn_id,
                        ServerToClient::TaskSnapshot {
                            task_id: task_id.clone(),
                            snapshot,
                        },
                    );
                    for event in pending {
                        self.send_to_client(conn_id, event);
                    }
                } else {
                    self.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            task_id: Some(task_id),
                            message: "unknown task".into(),
                        },
                    );
                }
            }
            ClientToServer::UnsubscribeFromTask { task_id } => {
                if let Some(subs) = self.subscriptions.get_mut(&task_id) {
                    subs.remove(&conn_id);
                }
            }
            ClientToServer::ListTasks { correlation_id } => {
                let tasks = self
                    .tasks
                    .values()
                    .filter(|t| !t.archived)
                    .map(|t| t.summary())
                    .collect();
                self.send_to_client(
                    conn_id,
                    ServerToClient::TaskList { correlation_id, tasks },
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
                self.send_to_client(
                    conn_id,
                    ServerToClient::BackendsList {
                        correlation_id,
                        default_backend: self.default_backend.clone(),
                        backends,
                    },
                );
            }
            ClientToServer::ListModels { correlation_id, backend } => {
                let entry = self.backends.get(&backend);
                let Some(entry) = entry else {
                    self.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            task_id: None,
                            message: format!("unknown backend `{backend}`"),
                        },
                    );
                    return;
                };
                // Spawn a detached task so we don't block the scheduler loop on the
                // backend's network round-trip. The task writes directly to the
                // client's outbound channel — the scheduler holds no intermediate state.
                let provider = entry.provider.clone();
                let outbound = self.clients.get(&conn_id).cloned();
                let Some(outbound) = outbound else {
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
                                task_id: None,
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
        config_override: Option<TaskConfigOverride>,
    ) -> Result<String, String> {
        let config = apply_override(self.default_config.clone(), config_override);
        // Validate shared MCP host names — fresh tasks must reference only
        // hosts that are actually configured. (Persisted tasks loaded from
        // disk may legitimately reference hosts that have since been removed
        // — those get warn-skipped at pool-build time, see McpConnectSuccess.)
        for name in &config.shared_mcp_hosts {
            if !self.shared_hosts.contains_key(name) {
                return Err(format!(
                    "unknown shared MCP host `{name}`; configured: [{}]",
                    self.shared_hosts.keys().cloned().collect::<Vec<_>>().join(", ")
                ));
            }
        }
        let task_id = new_task_id();
        let task = Task::new(task_id.clone(), config);
        let summary = task.summary();
        self.tasks.insert(task_id.clone(), task);

        info!(task_id = %task_id, "task created");
        // Every client gets exactly one TaskCreated; the requester's copy carries
        // correlation_id if they provided one.
        if correlation_id.is_some() {
            self.broadcast_task_list_except(
                ServerToClient::TaskCreated {
                    task_id: task_id.clone(),
                    summary: summary.clone(),
                    correlation_id: None,
                },
                requester,
            );
            self.send_to_client(
                requester,
                ServerToClient::TaskCreated {
                    task_id: task_id.clone(),
                    summary,
                    correlation_id,
                },
            );
        } else {
            self.broadcast_task_list(ServerToClient::TaskCreated {
                task_id: task_id.clone(),
                summary,
                correlation_id: None,
            });
        }
        Ok(task_id)
    }

    fn send_user_message(&mut self, task_id: &str, text: String) {
        self.mark_dirty(task_id);
        // Derive title on the first user message.
        let title_new = {
            let task = self.tasks.get_mut(task_id).expect("task exists");
            if task.title.is_none() {
                let t = derive_title(&text);
                task.title = Some(t.clone());
                Some(t)
            } else {
                None
            }
        };
        if let Some(title) = title_new {
            self.broadcast_task_list(ServerToClient::TaskTitleUpdated {
                task_id: task_id.to_string(),
                title,
            });
        }

        let mcp_connected = self.mcp_pools.contains_key(task_id);
        let tools_listed = self.tool_descriptors.contains_key(task_id);
        let new_state = {
            let task = self.tasks.get_mut(task_id).expect("task exists");
            task.submit_user_message(text, mcp_connected, tools_listed);
            task.public_state()
        };
        self.broadcast_task_list(ServerToClient::TaskStateChanged {
            task_id: task_id.to_string(),
            state: new_state,
        });
    }

    /// Apply one I/O completion to its task and dispatch any resulting events.
    fn apply_io_completion(&mut self, completion: IoCompletion) {
        let IoCompletion { task_id, op_id, result } = completion;
        let mut events = Vec::new();

        // Surface any IO-level error to the operator even if no client is subscribed.
        // The task's own Failed state captures it for the wire, but operator-visible
        // logs are how we catch misconfigured backends / unreachable endpoints without
        // round-tripping through the UI.
        match &result {
            IoResult::McpConnect(Err(e)) => {
                warn!(%task_id, op_id, error = %e, "mcp connect failed");
            }
            IoResult::ListTools(Err(e)) => {
                warn!(%task_id, op_id, error = %e, "list_tools failed");
            }
            IoResult::ModelCall(Err(e)) => {
                warn!(%task_id, op_id, error = %e, "model call failed");
            }
            IoResult::ToolCall { tool_use_id, result: Err(e) } => {
                warn!(%task_id, %tool_use_id, error = %e, "tool call failed");
            }
            _ => {}
        }

        // Handle IoResult::McpConnect / ListTools specially because the
        // session Arcs and routing map are scheduler-owned resources that
        // get peeled off before the task sees the slim variant.
        let result = match result {
            IoResult::McpConnectSuccess { session, sandbox_handle } => {
                // Build the task's pool: primary first, then shared singletons
                // selected by the task's allowlist. Unknown names are warn-skipped
                // — they shouldn't survive create_task validation, but a persisted
                // task whose host was removed since save lands here.
                let allow = self
                    .tasks
                    .get(&task_id)
                    .map(|t| t.config.shared_mcp_hosts.clone())
                    .unwrap_or_default();
                let mut sessions = vec![session];
                for name in &allow {
                    match self.shared_hosts.get(name) {
                        Some(s) => sessions.push(s.clone()),
                        None => warn!(
                            %task_id, host = %name,
                            "task references unknown shared MCP host (skipping)",
                        ),
                    }
                }
                self.mcp_pools.insert(
                    task_id.clone(),
                    McpPool { sessions, routing: HashMap::new() },
                );
                if let Some(handle) = sandbox_handle {
                    self.sandbox_handles.insert(task_id.clone(), handle);
                }
                IoResult::McpConnect(Ok(()))
            }
            IoResult::ListToolsSuccess { tools, routing } => {
                if let Some(pool) = self.mcp_pools.get_mut(&task_id) {
                    pool.routing = routing;
                }
                self.tool_descriptors.insert(task_id.clone(), tools);
                IoResult::ListTools(Ok(()))
            }
            other => other,
        };

        // Build per-tool-name annotation map so the task's approval policy can consult it.
        let annotations = self.annotations_for(&task_id);
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.apply_io_result(op_id, result, &annotations, &mut events);
        } else {
            warn!(%task_id, op_id, "io completion for unknown task");
            return;
        }
        self.mark_dirty(&task_id);
        self.dispatch_task_events(&task_id, events);
        self.teardown_sandbox_if_terminal(&task_id);
    }

    fn annotations_for(&self, task_id: &str) -> HashMap<String, ToolAnnotations> {
        self.tool_descriptors
            .get(task_id)
            .map(|tools| {
                tools
                    .iter()
                    .map(|t| (t.name.clone(), t.annotations.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Advance `task_id`'s state machine until it pauses, pushing new I/O to
    /// `pending_io` as requested.
    fn step_until_blocked(&mut self, task_id: &str, pending_io: &mut FuturesUnordered<IoFuture>) {
        self.mark_dirty(task_id);
        loop {
            let mut events = Vec::new();
            let outcome = {
                let Some(task) = self.tasks.get_mut(task_id) else {
                    return;
                };
                task.step(&mut self.next_op_id, &mut events)
            };
            self.dispatch_task_events(task_id, events);

            match outcome {
                StepOutcome::DispatchIo(req) => {
                    // Push the future but keep stepping — `AwaitingTools` dispatches all
                    // N tool calls in parallel before pausing.
                    let fut = io_dispatch::build_io_future(self, task_id.to_string(), req);
                    pending_io.push(fut);
                }
                StepOutcome::Continue => continue,
                StepOutcome::Paused => break,
            }
        }
    }

    /// If the task is in a terminal state, tear down its sandbox (if any).
    fn teardown_sandbox_if_terminal(&mut self, task_id: &str) {
        // Completed is NOT terminal — the task can receive follow-up messages.
        // Only tear down on truly irreversible states.
        let is_terminal = self
            .tasks
            .get(task_id)
            .is_some_and(|t| matches!(
                t.public_state(),
                TaskStateLabel::Failed | TaskStateLabel::Cancelled
            ));
        if is_terminal {
            // Pool's primary session points at the sandbox MCP host that's
            // about to die. Drop the pool too — shared session Arcs survive
            // because `self.shared_hosts` still holds them.
            self.mcp_pools.remove(task_id);
            if let Some(mut handle) = self.sandbox_handles.remove(task_id) {
                let tid = task_id.to_string();
                tokio::spawn(async move {
                    if let Err(e) = handle.teardown().await {
                        warn!(task_id = %tid, error = %e, "sandbox teardown failed");
                    }
                });
            }
        }
    }

    // ---------- Event dispatch ----------

    fn dispatch_task_events(&self, task_id: &str, events: Vec<TaskEvent>) {
        for event in events {
            match event {
                TaskEvent::AssistantBegin { turn } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskAssistantBegin {
                            task_id: task_id.to_string(),
                            turn,
                        },
                    );
                }
                TaskEvent::AssistantText { text } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskAssistantText {
                            task_id: task_id.to_string(),
                            text,
                        },
                    );
                }
                TaskEvent::AssistantReasoning { text } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskAssistantReasoning {
                            task_id: task_id.to_string(),
                            text,
                        },
                    );
                }
                TaskEvent::ToolCallBegin { tool_use_id, name, args_preview } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskToolCallBegin {
                            task_id: task_id.to_string(),
                            tool_use_id,
                            name,
                            args_preview,
                        },
                    );
                }
                TaskEvent::ToolCallEnd { tool_use_id, result_preview, is_error } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskToolCallEnd {
                            task_id: task_id.to_string(),
                            tool_use_id,
                            result_preview,
                            is_error,
                        },
                    );
                }
                TaskEvent::PendingApproval {
                    approval_id,
                    tool_use_id,
                    name,
                    args_preview,
                    destructive,
                    read_only,
                } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskPendingApproval {
                            task_id: task_id.to_string(),
                            approval_id,
                            tool_use_id,
                            name,
                            args_preview,
                            destructive,
                            read_only,
                        },
                    );
                }
                TaskEvent::ApprovalResolved { approval_id, decision, decided_by_conn } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskApprovalResolved {
                            task_id: task_id.to_string(),
                            approval_id,
                            decision,
                            decided_by_conn,
                        },
                    );
                }
                TaskEvent::AssistantEnd { stop_reason, usage } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskAssistantEnd {
                            task_id: task_id.to_string(),
                            stop_reason,
                            usage,
                        },
                    );
                }
                TaskEvent::LoopComplete => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskLoopComplete {
                            task_id: task_id.to_string(),
                        },
                    );
                }
                TaskEvent::StateChanged => {
                    let Some(task) = self.tasks.get(task_id) else { continue };
                    self.broadcast_task_list(ServerToClient::TaskStateChanged {
                        task_id: task_id.to_string(),
                        state: task.public_state(),
                    });
                }
                TaskEvent::Error { message } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            task_id: Some(task_id.to_string()),
                            message,
                        },
                    );
                }
                TaskEvent::AllowlistChanged { allowlist } => {
                    self.broadcast_to_subscribers(
                        task_id,
                        ServerToClient::TaskAllowlistUpdated {
                            task_id: task_id.to_string(),
                            tool_allowlist: allowlist,
                        },
                    );
                }
                TaskEvent::AuditToolCall {
                    tool_name,
                    args,
                    is_error,
                    error_message,
                    decision,
                    who_decided,
                } => {
                    self.write_audit(
                        task_id,
                        tool_name,
                        args,
                        is_error,
                        error_message,
                        decision,
                        who_decided,
                    );
                }
            }
        }
    }

    fn write_audit(
        &self,
        task_id: &str,
        tool_name: String,
        args: serde_json::Value,
        is_error: bool,
        error_message: Option<String>,
        decision: String,
        who_decided: String,
    ) {
        // Fire-and-forget — audit writes don't block task progression.
        let audit = self.audit.clone();
        let task_id = task_id.to_string();
        let host_id = self.host_id.clone();
        tokio::spawn(async move {
            let outcome = match &error_message {
                Some(msg) => ToolCallOutcome::Failed { message: msg },
                None => ToolCallOutcome::Ok { is_error },
            };
            let entry = ToolCallEntry {
                timestamp: chrono::Utc::now(),
                task_id: &task_id,
                host_id: &host_id,
                tool_name: &tool_name,
                args,
                decision: &decision,
                who_decided: &who_decided,
                outcome,
            };
            if let Err(e) = audit.write(&entry).await {
                error!(error = %e, "audit write failed");
            }
        });
    }

    fn send_to_client(&self, conn_id: ConnId, event: ServerToClient) {
        if let Some(tx) = self.clients.get(&conn_id) {
            if tx.send(event).is_err() {
                warn!(conn_id, "send_to_client: outbound channel closed");
            }
        }
    }

    fn broadcast_task_list(&self, event: ServerToClient) {
        for tx in self.clients.values() {
            let _ = tx.send(event.clone());
        }
    }

    fn broadcast_task_list_except(&self, event: ServerToClient, skip: ConnId) {
        for (conn_id, tx) in &self.clients {
            if *conn_id != skip {
                let _ = tx.send(event.clone());
            }
        }
    }

    fn broadcast_to_subscribers(&self, task_id: &str, event: ServerToClient) {
        let Some(subs) = self.subscriptions.get(task_id) else {
            return;
        };
        for conn_id in subs {
            if let Some(tx) = self.clients.get(conn_id) {
                let _ = tx.send(event.clone());
            }
        }
    }
}

/// Build the `TaskPendingApproval` events that a newly-subscribed client needs to
/// render the approval UI. Returns empty if the task isn't in AwaitingApproval.
fn pending_approvals_of(task: &Task) -> Vec<ServerToClient> {
    let TaskInternalState::AwaitingApproval { tool_uses, dispositions } = &task.internal else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (tool_use, disposition) in tool_uses.iter().zip(dispositions.iter()) {
        if let ApprovalDisposition::Pending { approval_id, destructive, read_only } = disposition {
            out.push(ServerToClient::TaskPendingApproval {
                task_id: task.id.clone(),
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

fn apply_override(base: TaskConfig, ov: Option<TaskConfigOverride>) -> TaskConfig {
    let Some(ov) = ov else { return base };
    // If the override picks a different backend without specifying a model, don't
    // inherit the model from the default backend — leave it empty so build_model_request
    // resolves via the picked backend's default_model at call time (or passes empty to
    // a single-model endpoint that ignores the field).
    let backend_changed = ov
        .backend
        .as_deref()
        .is_some_and(|b| b != base.backend);
    let model = match ov.model {
        Some(m) => m,
        None if backend_changed => String::new(),
        None => base.model.clone(),
    };
    TaskConfig {
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

pub async fn run(
    mut scheduler: Scheduler,
    mut inbox: mpsc::UnboundedReceiver<SchedulerMsg>,
) {
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
                    let task_id = completion.task_id.clone();
                    scheduler.apply_io_completion(completion);
                    scheduler.step_until_blocked(&task_id, &mut pending_io);
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
