//! Thread event routing: connection registry, subscriptions, and the
//! `ThreadEvent → ServerToClient` translation that the scheduler hands off after
//! every `task.step()` / `task.apply_io_result()` cycle.
//!
//! Owns:
//!   - **Clients**: `ConnId → outbound mpsc::Sender<ServerToClient>`. The server's
//!     WebSocket task registers a sender on connect and unregisters on close.
//!   - **Subscriptions**: `thread_id → set<ConnId>`. Per-task event tier (assistant
//!     text, tool calls, approvals) only goes to subscribers.
//!   - **Audit log**: tool-call audit entries are written here as fire-and-forget
//!     tokio tasks driven by [`ThreadEvent::AuditToolCall`].
//!
//! Adding a new wire event type — or changing how events are batched / filtered —
//! is a local edit on this module. The scheduler's run loop and I/O dispatch stay
//! untouched.

use std::collections::{HashMap, HashSet};

use tokio::sync::mpsc;
use tracing::{error, warn};
use whisper_agent_protocol::ServerToClient;

use crate::runtime::audit::{AuditLog, ToolCallEntry, ToolCallOutcome};
use crate::runtime::scheduler::ConnId;
use crate::runtime::thread::ThreadEvent;

pub(crate) struct ThreadEventRouter {
    clients: HashMap<ConnId, mpsc::UnboundedSender<ServerToClient>>,
    subscriptions: HashMap<String, HashSet<ConnId>>,
    audit: AuditLog,
    host_id: String,
}

impl ThreadEventRouter {
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

    pub(crate) fn subscribe(&mut self, conn_id: ConnId, thread_id: &str) {
        self.subscriptions
            .entry(thread_id.to_string())
            .or_default()
            .insert(conn_id);
    }

    pub(crate) fn unsubscribe(&mut self, conn_id: ConnId, thread_id: &str) {
        if let Some(subs) = self.subscriptions.get_mut(thread_id) {
            subs.remove(&conn_id);
        }
    }

    pub(crate) fn drop_thread(&mut self, thread_id: &str) {
        self.subscriptions.remove(thread_id);
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

    /// Snapshot of every connected client's outbound channel. Used by code
    /// that broadcasts from a detached `tokio::spawn` (currently pod CRUD,
    /// which awaits async file I/O before fanning out the result).
    pub(crate) fn outbound_snapshot(&self) -> Vec<mpsc::UnboundedSender<ServerToClient>> {
        self.clients.values().cloned().collect()
    }

    pub(crate) fn broadcast_task_list_except(&self, event: ServerToClient, skip: ConnId) {
        for (conn_id, tx) in &self.clients {
            if *conn_id != skip {
                let _ = tx.send(event.clone());
            }
        }
    }

    pub(crate) fn broadcast_to_subscribers(&self, thread_id: &str, event: ServerToClient) {
        let Some(subs) = self.subscriptions.get(thread_id) else {
            return;
        };
        for conn_id in subs {
            if let Some(tx) = self.clients.get(conn_id) {
                let _ = tx.send(event.clone());
            }
        }
    }

    /// [`broadcast_to_subscribers`] skipping `skip`. For events
    /// that echo a sender's own mutation where the echo would
    /// disrupt local state (e.g. `ThreadDraftUpdated`).
    pub(crate) fn broadcast_to_subscribers_except(
        &self,
        thread_id: &str,
        event: ServerToClient,
        skip: ConnId,
    ) {
        let Some(subs) = self.subscriptions.get(thread_id) else {
            return;
        };
        for conn_id in subs {
            if *conn_id == skip {
                continue;
            }
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

    /// Translate a batch of [`ThreadEvent`]s for `thread_id` into wire events and
    /// dispatch them to the right audience: per-task subscribers for the turn
    /// tier, every client for state-list updates, the audit log for tool calls.
    pub(crate) fn dispatch_events(&self, thread_id: &str, events: Vec<ThreadEvent>) {
        for event in events {
            match event {
                ThreadEvent::AssistantBegin { turn } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadAssistantBegin {
                            thread_id: thread_id.to_string(),
                            turn,
                        },
                    );
                }
                ThreadEvent::AssistantText { text } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadAssistantText {
                            thread_id: thread_id.to_string(),
                            text,
                        },
                    );
                }
                ThreadEvent::AssistantReasoning { text } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadAssistantReasoning {
                            thread_id: thread_id.to_string(),
                            text,
                        },
                    );
                }
                ThreadEvent::ToolCallBegin {
                    tool_use_id,
                    name,
                    args_preview,
                    args,
                } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadToolCallBegin {
                            thread_id: thread_id.to_string(),
                            tool_use_id,
                            name,
                            args_preview,
                            args: Some(args),
                        },
                    );
                }
                ThreadEvent::ToolCallEnd {
                    tool_use_id,
                    result_preview,
                    is_error,
                } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadToolCallEnd {
                            thread_id: thread_id.to_string(),
                            tool_use_id,
                            result_preview,
                            is_error,
                        },
                    );
                }
                ThreadEvent::PendingApproval {
                    approval_id,
                    tool_use_id,
                    name,
                    args_preview,
                    destructive,
                    read_only,
                } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadPendingApproval {
                            thread_id: thread_id.to_string(),
                            approval_id,
                            tool_use_id,
                            name,
                            args_preview,
                            destructive,
                            read_only,
                        },
                    );
                }
                ThreadEvent::ApprovalResolved {
                    approval_id,
                    decision,
                    decided_by_conn,
                } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadApprovalResolved {
                            thread_id: thread_id.to_string(),
                            approval_id,
                            decision,
                            decided_by_conn,
                        },
                    );
                }
                ThreadEvent::AssistantEnd { stop_reason, usage } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadAssistantEnd {
                            thread_id: thread_id.to_string(),
                            stop_reason,
                            usage,
                        },
                    );
                }
                ThreadEvent::LoopComplete => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadLoopComplete {
                            thread_id: thread_id.to_string(),
                        },
                    );
                }
                ThreadEvent::StateChanged { state } => {
                    self.broadcast_task_list(ServerToClient::ThreadStateChanged {
                        thread_id: thread_id.to_string(),
                        state,
                    });
                }
                ThreadEvent::Error { message } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id.to_string()),
                            message,
                        },
                    );
                }
                ThreadEvent::AllowlistChanged { allowlist } => {
                    self.broadcast_to_subscribers(
                        thread_id,
                        ServerToClient::ThreadAllowlistUpdated {
                            thread_id: thread_id.to_string(),
                            tool_allowlist: allowlist,
                        },
                    );
                }
                ThreadEvent::AuditToolCall {
                    tool_name,
                    args,
                    is_error,
                    error_message,
                    decision,
                    who_decided,
                } => {
                    self.write_audit(
                        thread_id,
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
    // The args mirror `ThreadEvent::AuditToolCall` 1:1; grouping them into a
    // struct just shuffles the same fields without buying clarity.
    #[allow(clippy::too_many_arguments)]
    fn write_audit(
        &self,
        thread_id: &str,
        tool_name: String,
        args: serde_json::Value,
        is_error: bool,
        error_message: Option<String>,
        decision: String,
        who_decided: String,
    ) {
        let audit = self.audit.clone();
        let thread_id = thread_id.to_string();
        let host_id = self.host_id.clone();
        tokio::spawn(async move {
            let outcome = match &error_message {
                Some(msg) => ToolCallOutcome::Failed { message: msg },
                None => ToolCallOutcome::Ok { is_error },
            };
            let entry = ToolCallEntry {
                timestamp: chrono::Utc::now(),
                thread_id: &thread_id,
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
