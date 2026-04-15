//! TaskManager — owns all tasks, client connections, and subscriptions.
//!
//! Responsibilities:
//!
//! - **Task storage** as `HashMap<TaskId, TaskHandle>`. Each handle carries an
//!   `Arc<tokio::sync::Mutex<Task>>` plus an mpsc inbox that the per-task runner reads.
//! - **Client registry**: every connected WebSocket registers an outbound mpsc sender
//!   and receives an opaque `ConnId`. The manager routes wire events to the appropriate
//!   client senders.
//! - **Two-tier event delivery**:
//!   - Task-list events (created / state / title / archived) broadcast to all clients.
//!   - Per-task turn events only to subscribers of that task id.
//! - **Per-task runner** (placeholder): each task spawns one tokio task that reads from
//!   its inbox and drives `turn::run` against a per-task MCP session. This will be
//!   replaced by a single central scheduler in the next commit.
//!
//! Snapshot consistency during a turn: the runner clones the conversation when a turn
//! starts, drives the clone, and writes it back at turn end. Clients that subscribe
//! mid-turn see the pre-turn state in their snapshot plus the event stream from the
//! subscribe point onward. That's consistent with the design doc's "clients that
//! reconnect mid-stream get a consistent view once the turn settles".

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tracing::{error, info, warn};
use whisper_agent_protocol::{
    ClientToServer, ServerToClient, TaskConfig, TaskConfigOverride, TaskStateLabel, Usage,
};

use crate::anthropic::{AnthropicClient, Usage as AnthropicUsage};
use crate::audit::AuditLog;
use crate::mcp::McpSession;
use crate::task::{Task, derive_title, new_task_id};
use crate::turn::{self, TurnConfig, TurnEvent};

pub type ConnId = u64;

#[derive(Debug)]
enum TaskInput {
    UserMessage(String),
    Cancel,
}

struct TaskHandle {
    task: Arc<AsyncMutex<Task>>,
    inbox: mpsc::UnboundedSender<TaskInput>,
}

pub struct TaskManager {
    default_config: TaskConfig,
    host_id: String,
    anthropic: Arc<AnthropicClient>,
    audit: AuditLog,
    inner: Mutex<Inner>,
}

struct Inner {
    tasks: HashMap<String, TaskHandle>,
    clients: HashMap<ConnId, mpsc::UnboundedSender<ServerToClient>>,
    subscriptions: HashMap<String, HashSet<ConnId>>,
    next_conn_id: ConnId,
}

impl TaskManager {
    pub fn new(
        default_config: TaskConfig,
        host_id: String,
        anthropic: Arc<AnthropicClient>,
        audit: AuditLog,
    ) -> Self {
        Self {
            default_config,
            host_id,
            anthropic,
            audit,
            inner: Mutex::new(Inner {
                tasks: HashMap::new(),
                clients: HashMap::new(),
                subscriptions: HashMap::new(),
                next_conn_id: 1,
            }),
        }
    }

    // ---------- Client registry ----------

    pub fn register_client(&self, outbound: mpsc::UnboundedSender<ServerToClient>) -> ConnId {
        let mut inner = self.inner.lock().unwrap();
        let id = inner.next_conn_id;
        inner.next_conn_id += 1;
        inner.clients.insert(id, outbound);
        id
    }

    pub fn unregister_client(&self, conn_id: ConnId) {
        let mut inner = self.inner.lock().unwrap();
        inner.clients.remove(&conn_id);
        for subs in inner.subscriptions.values_mut() {
            subs.remove(&conn_id);
        }
    }

    // ---------- Inbound dispatch ----------

    /// Handle one decoded [`ClientToServer`] from a connected client.
    pub async fn handle_client_message(self: &Arc<Self>, conn_id: ConnId, msg: ClientToServer) {
        match msg {
            ClientToServer::CreateTask {
                correlation_id,
                initial_message,
                config_override,
            } => {
                self.create_task(conn_id, correlation_id, initial_message, config_override);
            }
            ClientToServer::SendUserMessage { task_id, text } => {
                self.send_user_message(&task_id, text);
            }
            ClientToServer::CancelTask { task_id } => {
                self.cancel_task(&task_id);
            }
            ClientToServer::ArchiveTask { task_id } => {
                self.archive_task(&task_id).await;
            }
            ClientToServer::SubscribeToTask { task_id } => {
                self.subscribe(conn_id, task_id).await;
            }
            ClientToServer::UnsubscribeFromTask { task_id } => {
                self.unsubscribe(conn_id, &task_id);
            }
            ClientToServer::ListTasks { correlation_id } => {
                self.list_tasks_for_client(conn_id, correlation_id).await;
            }
        }
    }

    // ---------- Task lifecycle ----------

    fn create_task(
        self: &Arc<Self>,
        requester_conn_id: ConnId,
        correlation_id: Option<String>,
        initial_message: String,
        config_override: Option<TaskConfigOverride>,
    ) {
        let config = apply_override(self.default_config.clone(), config_override);
        let task_id = new_task_id();
        let title = Some(derive_title(&initial_message));

        let mut task = Task::new(task_id.clone(), config);
        task.title = title.clone();

        let summary = task.summary();
        let (inbox_tx, inbox_rx) = mpsc::unbounded_channel::<TaskInput>();

        let handle = TaskHandle {
            task: Arc::new(AsyncMutex::new(task)),
            inbox: inbox_tx.clone(),
        };
        let task_arc = handle.task.clone();

        {
            let mut inner = self.inner.lock().unwrap();
            inner.tasks.insert(task_id.clone(), handle);
        }

        info!(task_id = %task_id, "task created");

        // Every client gets exactly one TaskCreated. If the requester supplied a
        // correlation_id, their copy carries it; others get a None-correlation copy.
        if correlation_id.is_some() {
            self.broadcast_task_list_except(
                ServerToClient::TaskCreated {
                    task_id: task_id.clone(),
                    summary: summary.clone(),
                    correlation_id: None,
                },
                requester_conn_id,
            );
            self.send_to_client(
                requester_conn_id,
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
        if let Some(t) = &title {
            self.broadcast_task_list(ServerToClient::TaskTitleUpdated {
                task_id: task_id.clone(),
                title: t.clone(),
            });
        }

        // Enqueue the initial user message and spawn the runner.
        let _ = inbox_tx.send(TaskInput::UserMessage(initial_message));
        let mgr = self.clone();
        tokio::spawn(async move {
            mgr.run_task(task_id, task_arc, inbox_rx).await;
        });
    }

    fn send_user_message(&self, task_id: &str, text: String) {
        let handle = {
            let inner = self.inner.lock().unwrap();
            inner.tasks.get(task_id).map(|h| h.inbox.clone())
        };
        match handle {
            Some(tx) => {
                if tx.send(TaskInput::UserMessage(text)).is_err() {
                    warn!(task_id, "send_user_message: task runner has exited");
                }
            }
            None => warn!(task_id, "send_user_message: unknown task"),
        }
    }

    fn cancel_task(&self, task_id: &str) {
        let handle = {
            let inner = self.inner.lock().unwrap();
            inner.tasks.get(task_id).map(|h| h.inbox.clone())
        };
        match handle {
            Some(tx) => {
                let _ = tx.send(TaskInput::Cancel);
            }
            None => warn!(task_id, "cancel_task: unknown task"),
        }
    }

    async fn archive_task(&self, task_id: &str) {
        let task_arc = {
            let inner = self.inner.lock().unwrap();
            inner.tasks.get(task_id).map(|h| h.task.clone())
        };
        let Some(task_arc) = task_arc else {
            warn!(task_id, "archive_task: unknown task");
            return;
        };
        {
            let mut t = task_arc.lock().await;
            t.archived = true;
            t.touch();
        }
        self.broadcast_task_list(ServerToClient::TaskArchived { task_id: task_id.to_string() });
    }

    // ---------- Subscriptions ----------

    async fn subscribe(&self, conn_id: ConnId, task_id: String) {
        let task_arc = {
            let mut inner = self.inner.lock().unwrap();
            let Some(handle) = inner.tasks.get(&task_id) else {
                drop(inner);
                self.send_to_client(
                    conn_id,
                    ServerToClient::Error {
                        correlation_id: None,
                        task_id: Some(task_id),
                        message: "unknown task".into(),
                    },
                );
                return;
            };
            let arc = handle.task.clone();
            inner.subscriptions.entry(task_id.clone()).or_default().insert(conn_id);
            arc
        };
        let snapshot = task_arc.lock().await.snapshot();
        self.send_to_client(
            conn_id,
            ServerToClient::TaskSnapshot { task_id, snapshot },
        );
    }

    fn unsubscribe(&self, conn_id: ConnId, task_id: &str) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(subs) = inner.subscriptions.get_mut(task_id) {
            subs.remove(&conn_id);
        }
    }

    async fn list_tasks_for_client(&self, conn_id: ConnId, correlation_id: Option<String>) {
        // Collect task Arcs under the sync lock; then lock each task for a snapshot.
        let task_arcs: Vec<Arc<AsyncMutex<Task>>> = {
            let inner = self.inner.lock().unwrap();
            inner.tasks.values().map(|h| h.task.clone()).collect()
        };
        let mut tasks = Vec::with_capacity(task_arcs.len());
        for arc in task_arcs {
            let t = arc.lock().await;
            if !t.archived {
                tasks.push(t.summary());
            }
        }
        self.send_to_client(
            conn_id,
            ServerToClient::TaskList { correlation_id, tasks },
        );
    }

    // ---------- Event emission ----------

    fn send_to_client(&self, conn_id: ConnId, event: ServerToClient) {
        let tx = {
            let inner = self.inner.lock().unwrap();
            inner.clients.get(&conn_id).cloned()
        };
        if let Some(tx) = tx {
            if tx.send(event).is_err() {
                warn!(conn_id, "send_to_client: outbound channel closed");
            }
        }
    }

    fn broadcast_task_list(&self, event: ServerToClient) {
        let senders: Vec<_> = {
            let inner = self.inner.lock().unwrap();
            inner.clients.values().cloned().collect()
        };
        for tx in senders {
            let _ = tx.send(event.clone());
        }
    }

    fn broadcast_task_list_except(&self, event: ServerToClient, skip_conn_id: ConnId) {
        let senders: Vec<_> = {
            let inner = self.inner.lock().unwrap();
            inner
                .clients
                .iter()
                .filter_map(|(id, tx)| (*id != skip_conn_id).then(|| tx.clone()))
                .collect()
        };
        for tx in senders {
            let _ = tx.send(event.clone());
        }
    }

    fn broadcast_to_subscribers(&self, task_id: &str, event: ServerToClient) {
        let senders: Vec<_> = {
            let inner = self.inner.lock().unwrap();
            let Some(subs) = inner.subscriptions.get(task_id) else {
                return;
            };
            subs.iter()
                .filter_map(|conn_id| inner.clients.get(conn_id).cloned())
                .collect()
        };
        for tx in senders {
            let _ = tx.send(event.clone());
        }
    }

    fn set_task_state_and_broadcast(&self, task_id: &str, state: TaskStateLabel) {
        self.broadcast_task_list(ServerToClient::TaskStateChanged {
            task_id: task_id.to_string(),
            state,
        });
    }

    fn emit_turn_event(&self, task_id: &str, event: TurnEvent) {
        let wire = match event {
            TurnEvent::AssistantBegin { turn } => ServerToClient::TaskAssistantBegin {
                task_id: task_id.to_string(),
                turn,
            },
            TurnEvent::AssistantText { text } => ServerToClient::TaskAssistantText {
                task_id: task_id.to_string(),
                text,
            },
            TurnEvent::ToolCallBegin { tool_use_id, name, args_preview } => {
                ServerToClient::TaskToolCallBegin {
                    task_id: task_id.to_string(),
                    tool_use_id,
                    name,
                    args_preview,
                }
            }
            TurnEvent::ToolCallEnd { tool_use_id, result_preview, is_error } => {
                ServerToClient::TaskToolCallEnd {
                    task_id: task_id.to_string(),
                    tool_use_id,
                    result_preview,
                    is_error,
                }
            }
            TurnEvent::AssistantEnd { stop_reason, usage } => ServerToClient::TaskAssistantEnd {
                task_id: task_id.to_string(),
                stop_reason,
                usage,
            },
            TurnEvent::LoopComplete => ServerToClient::TaskLoopComplete {
                task_id: task_id.to_string(),
            },
        };
        self.broadcast_to_subscribers(task_id, wire);
    }

    // ---------- Per-task runner ----------

    async fn run_task(
        self: Arc<Self>,
        task_id: String,
        task_arc: Arc<AsyncMutex<Task>>,
        mut inbox: mpsc::UnboundedReceiver<TaskInput>,
    ) {
        let mcp_url = task_arc.lock().await.config.mcp_host_url.clone();
        let mcp = match McpSession::connect(&mcp_url).await {
            Ok(s) => s,
            Err(e) => {
                error!(task_id, error = %e, "mcp connect failed");
                let msg = format!("mcp connect failed: {e}");
                {
                    let mut t = task_arc.lock().await;
                    t.state = TaskStateLabel::Failed;
                }
                self.set_task_state_and_broadcast(&task_id, TaskStateLabel::Failed);
                self.broadcast_to_subscribers(
                    &task_id,
                    ServerToClient::Error {
                        correlation_id: None,
                        task_id: Some(task_id.clone()),
                        message: msg,
                    },
                );
                return;
            }
        };

        while let Some(input) = inbox.recv().await {
            match input {
                TaskInput::UserMessage(text) => {
                    self.handle_user_message_turn(&task_id, &task_arc, &mcp, text).await;
                }
                TaskInput::Cancel => {
                    {
                        let mut t = task_arc.lock().await;
                        t.state = TaskStateLabel::Cancelled;
                        t.touch();
                    }
                    self.set_task_state_and_broadcast(&task_id, TaskStateLabel::Cancelled);
                    return;
                }
            }
        }
    }

    async fn handle_user_message_turn(
        self: &Arc<Self>,
        task_id: &str,
        task_arc: &Arc<AsyncMutex<Task>>,
        mcp: &McpSession,
        text: String,
    ) {
        // Phase 1: flip to Working, take a clone of the conversation.
        let (config, mut conversation) = {
            let mut t = task_arc.lock().await;
            t.state = TaskStateLabel::Working;
            t.touch();
            (t.config.clone(), t.conversation.clone())
        };
        self.set_task_state_and_broadcast(task_id, TaskStateLabel::Working);

        // Phase 2: drive the turn.
        let (tx, mut rx) = mpsc::unbounded_channel::<TurnEvent>();
        let forwarder = {
            let mgr = self.clone();
            let tid = task_id.to_string();
            tokio::spawn(async move {
                while let Some(e) = rx.recv().await {
                    mgr.emit_turn_event(&tid, e);
                }
            })
        };

        let tcfg = TurnConfig {
            model: &config.model,
            max_tokens: config.max_tokens,
            system_prompt: &config.system_prompt,
            task_id,
            host_id: &self.host_id,
            max_turns: config.max_turns,
        };
        let result = turn::run(
            &tcfg,
            &self.anthropic,
            mcp,
            &self.audit,
            &mut conversation,
            text,
            Some(tx),
        )
        .await;
        let _ = forwarder.await;

        // Phase 3: commit conversation + state back.
        let (new_state, error_msg) = {
            let mut t = task_arc.lock().await;
            t.conversation = conversation;
            t.touch();
            match &result {
                Ok(outcome) => {
                    t.total_usage.add(&convert_usage(&outcome.usage_total));
                    t.state = TaskStateLabel::Completed;
                    (t.state, None)
                }
                Err(e) => {
                    t.state = TaskStateLabel::Failed;
                    (t.state, Some(format!("loop failed: {e}")))
                }
            }
        };
        self.set_task_state_and_broadcast(task_id, new_state);
        if let Some(msg) = error_msg {
            error!(task_id, error = %msg, "loop failed");
            self.broadcast_to_subscribers(
                task_id,
                ServerToClient::Error {
                    correlation_id: None,
                    task_id: Some(task_id.to_string()),
                    message: msg,
                },
            );
        }
    }
}

fn apply_override(base: TaskConfig, ov: Option<TaskConfigOverride>) -> TaskConfig {
    let Some(ov) = ov else { return base };
    TaskConfig {
        model: ov.model.unwrap_or(base.model),
        system_prompt: ov.system_prompt.unwrap_or(base.system_prompt),
        mcp_host_url: ov.mcp_host_url.unwrap_or(base.mcp_host_url),
        max_tokens: ov.max_tokens.unwrap_or(base.max_tokens),
        max_turns: ov.max_turns.unwrap_or(base.max_turns),
    }
}

fn convert_usage(u: &AnthropicUsage) -> Usage {
    Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        cache_read_input_tokens: u.cache_read_input_tokens,
        cache_creation_input_tokens: u.cache_creation_input_tokens,
    }
}
