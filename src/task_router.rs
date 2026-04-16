//! Task event routing: connection registry, subscriptions, and the
//! `TaskEvent → ServerToClient` translation that the scheduler hands off after
//! every `task.step()` / `task.apply_io_result()` cycle.
//!
//! Owns:
//!   - **Clients**: `ConnId → outbound mpsc::Sender<ServerToClient>`. The server's
//!     WebSocket task registers a sender on connect and unregisters on close.
//!   - **Subscriptions**: `task_id → set<ConnId>`. Per-task event tier (assistant
//!     text, tool calls, approvals) only goes to subscribers.
//!   - **Audit log**: tool-call audit entries are written here as fire-and-forget
//!     tokio tasks driven by [`TaskEvent::AuditToolCall`].
//!
//! Adding a new wire event type — or changing how events are batched / filtered —
//! is a local edit on this module. The scheduler's run loop and I/O dispatch stay
//! untouched.

use std::collections::{HashMap, HashSet};

use tokio::sync::mpsc;
use tracing::{error, warn};
use whisper_agent_protocol::ServerToClient;

use crate::audit::{AuditLog, ToolCallEntry, ToolCallOutcome};
use crate::scheduler::ConnId;
use crate::task::TaskEvent;

pub(crate) struct TaskEventRouter {
    clients: HashMap<ConnId, mpsc::UnboundedSender<ServerToClient>>,
    subscriptions: HashMap<String, HashSet<ConnId>>,
    audit: AuditLog,
    host_id: String,
}

impl TaskEventRouter {
    pub(crate) fn new(audit: AuditLog, host_id: String) -> Self {
        Self {
            clients: HashMap::new(),
            subscriptions: HashMap::new(),
            audit,
            host_id,
        }
    }

    // ---------- Connection lifecycle ----------

    pub(crate) fn register_client(
        &mut self,
        conn_id: ConnId,
        outbound: mpsc::UnboundedSender<ServerToClient>,
    ) {
        self.clients.insert(conn_id, outbound);
    }

    pub(crate) fn unregister_client(&mut self, conn_id: ConnId) {
        self.clients.remove(&conn_id);
        for subs in self.subscriptions.values_mut() {
            subs.remove(&conn_id);
        }
    }

    // ---------- Subscription lifecycle ----------

    pub(crate) fn subscribe(&mut self, conn_id: ConnId, task_id: &str) {
        self.subscriptions
            .entry(task_id.to_string())
            .or_default()
            .insert(conn_id);
    }

    pub(crate) fn unsubscribe(&mut self, conn_id: ConnId, task_id: &str) {
        if let Some(subs) = self.subscriptions.get_mut(task_id) {
            subs.remove(&conn_id);
        }
    }

    // ---------- Send / broadcast ----------

    pub(crate) fn send_to_client(&self, conn_id: ConnId, event: ServerToClient) {
        if let Some(tx) = self.clients.get(&conn_id)
            && tx.send(event).is_err()
        {
            warn!(conn_id, "send_to_client: outbound channel closed");
        }
    }

    pub(crate) fn broadcast_task_list(&self, event: ServerToClient) {
        for tx in self.clients.values() {
            let _ = tx.send(event.clone());
        }
    }

    /// Broadcast a resource-tier event to every connected client. Same fan-out
    /// as `broadcast_task_list`; named separately so the call sites are honest
    /// about which event tier they're emitting.
    pub(crate) fn broadcast_resource(&self, event: ServerToClient) {
        for tx in self.clients.values() {
            let _ = tx.send(event.clone());
        }
    }

    pub(crate) fn broadcast_task_list_except(&self, event: ServerToClient, skip: ConnId) {
        for (conn_id, tx) in &self.clients {
            if *conn_id != skip {
                let _ = tx.send(event.clone());
            }
        }
    }

    pub(crate) fn broadcast_to_subscribers(&self, task_id: &str, event: ServerToClient) {
        let Some(subs) = self.subscriptions.get(task_id) else {
            return;
        };
        for conn_id in subs {
            if let Some(tx) = self.clients.get(conn_id) {
                let _ = tx.send(event.clone());
            }
        }
    }

    /// Clone of a connection's outbound channel for code that needs to send
    /// from a detached `tokio::spawn` (currently only the `ListModels` handler,
    /// which round-trips through the backend before responding).
    pub(crate) fn outbound(
        &self,
        conn_id: ConnId,
    ) -> Option<mpsc::UnboundedSender<ServerToClient>> {
        self.clients.get(&conn_id).cloned()
    }

    // ---------- Event translation ----------

    /// Translate a batch of [`TaskEvent`]s for `task_id` into wire events and
    /// dispatch them to the right audience: per-task subscribers for the turn
    /// tier, every client for state-list updates, the audit log for tool calls.
    pub(crate) fn dispatch_events(&self, task_id: &str, events: Vec<TaskEvent>) {
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
                TaskEvent::ToolCallBegin {
                    tool_use_id,
                    name,
                    args_preview,
                } => {
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
                TaskEvent::ToolCallEnd {
                    tool_use_id,
                    result_preview,
                    is_error,
                } => {
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
                TaskEvent::ApprovalResolved {
                    approval_id,
                    decision,
                    decided_by_conn,
                } => {
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
                TaskEvent::StateChanged { state } => {
                    self.broadcast_task_list(ServerToClient::TaskStateChanged {
                        task_id: task_id.to_string(),
                        state,
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

    /// Fire-and-forget audit write — never blocks task progression.
    // The args mirror `TaskEvent::AuditToolCall` 1:1; grouping them into a
    // struct just shuffles the same fields without buying clarity.
    #[allow(clippy::too_many_arguments)]
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
}
