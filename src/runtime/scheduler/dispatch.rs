//! `dispatch_thread` scheduler intercept.
//!
//! The builtin tool `dispatch_thread` spawns a child thread in the same
//! pod as the caller. Under the Function model, the intercept is a thin
//! shim over `Function::CreateThread`:
//!
//! - The parent's `ToolCall` IO is replaced by a future that either
//!   parks on the Function's terminal (sync) or returns an immediate
//!   "Dispatched task-X" ack (async).
//! - `Function::CreateThread { wait_mode: ThreadTerminal, parent:
//!   Some, .. }` is registered with a delivery tag describing how its
//!   eventual terminal reaches the parent:
//!     - `FunctionDelivery::ToolResultChannel(tx)` — sync: the
//!       terminal's final text fires `tx`, the parent's SchedulerFuture
//!       reads it as the tool_result.
//!     - `FunctionDelivery::ToolResultFollowup { parent_thread_id,
//!       parent_tool_use_id }` — async: `complete_function` renders a
//!       `<dispatched-thread-notification>` envelope from the terminal
//!       and injects it as a fresh user message on the parent (kicks
//!       a new turn).
//! - Cancellation cascade: when a parent thread terminates while it
//!   still has Functions registered with a `ThreadToolCall` caller-link
//!   pointing back at it, `cascade_cancel_caller_gone` cancels those
//!   Functions — which in turn cancels each dispatched child (since the
//!   Function owns the "I'm waiting on this child" link via
//!   `awaiting_child_thread_id`).
//!
//! No side-map lives outside the Function registry — `pending_dispatches`
//! was removed during the Phase 5c rewrite. A restart mid-dispatch still
//! orphans the parent the same way it orphans an in-flight MCP tool
//! call: Functions aren't persistent, so a crash loses the linkage and
//! the user recovers with `CancelThread`.

use futures::stream::FuturesUnordered;
use tokio::sync::oneshot;
use tracing::debug;
use super::Scheduler;
use super::functions::{FunctionDelivery, MAX_DISPATCH_DEPTH};
use crate::runtime::io_dispatch::{IoCompletion, SchedulerCompletion, SchedulerFuture};
use crate::runtime::thread::{IoResult, OpId};
use crate::tools::builtin_tools::DISPATCH_THREAD;
use crate::tools::builtin_tools::dispatch_thread::parse_args;
use crate::tools::mcp::{CallToolResult, McpContentBlock};

impl Scheduler {
    /// If `tool_name == "dispatch_thread"`, consume the request: register
    /// `Function::CreateThread` for the child, wire up the parent's
    /// tool-call future, and return it. Returns `None` for any other
    /// tool name — the caller falls through to the normal
    /// `io_dispatch::build_io_future` path.
    pub(super) fn try_intercept_dispatch_thread(
        &mut self,
        parent_thread_id: &str,
        op_id: OpId,
        tool_use_id: &str,
        tool_name: &str,
        input: serde_json::Value,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Option<SchedulerFuture> {
        if tool_name != DISPATCH_THREAD {
            return None;
        }
        let parent_id = parent_thread_id.to_string();
        let tool_use_id = tool_use_id.to_string();

        let args = match parse_args(input) {
            Ok(a) => a,
            Err(e) => return Some(immediate_err(parent_id, op_id, tool_use_id, e)),
        };

        // The parent must exist and not have recursed past the cap.
        // These checks duplicate the Function precondition check so we
        // can surface as a clean tool error rather than a Function
        // RejectReason; the Function-level check remains as defense in
        // depth.
        let parent_pod = match self.tasks.get(parent_thread_id) {
            Some(t) => {
                if t.dispatch_depth >= MAX_DISPATCH_DEPTH {
                    return Some(immediate_err(
                        parent_id,
                        op_id,
                        tool_use_id,
                        format!(
                            "dispatch_thread: nesting cap reached (parent depth {}, max {MAX_DISPATCH_DEPTH})",
                            t.dispatch_depth
                        ),
                    ));
                }
                t.pod_id.clone()
            }
            None => {
                return Some(immediate_err(
                    parent_id,
                    op_id,
                    tool_use_id,
                    "dispatch_thread: parent thread vanished".into(),
                ));
            }
        };

        let spec = crate::functions::Function::CreateThread {
            pod_id: Some(parent_pod),
            initial_message: Some(args.prompt.clone()),
            parent: Some(crate::functions::ParentLink {
                thread_id: parent_id.clone(),
                tool_use_id: tool_use_id.clone(),
            }),
            wait_mode: crate::functions::WaitMode::ThreadTerminal,
            config_override: args.config_override,
            bindings_request: args.bindings_override,
        };
        let scope = self.internal_scope();
        let caller = crate::functions::CallerLink::ThreadToolCall {
            thread_id: parent_id.clone(),
            tool_use_id: tool_use_id.clone(),
        };

        if args.sync {
            let (tx, rx) = oneshot::channel::<Result<String, String>>();
            let delivery = FunctionDelivery::ToolResultChannel(tx);
            if let Err(e) = self.register_function_with_delivery(spec, scope, caller, delivery) {
                return Some(immediate_err(
                    parent_id,
                    op_id,
                    tool_use_id,
                    format!("dispatch_thread: {}", super::client_messages::reject_reason_detail(&e)),
                ));
            }
            // Launch is keyed off the most-recent Function id. Use a
            // straightforward scan rather than threading the id back
            // from register_function_with_delivery.
            let fn_id = self
                .active_functions
                .keys()
                .max()
                .copied()
                .expect("just inserted");
            self.launch_function(fn_id, pending_io);
            debug!(
                parent = %parent_id, tool_use_id = %tool_use_id,
                "dispatch_thread: sync — parent parked awaiting Function terminal"
            );
            Some(Box::pin(async move {
                let result = match rx.await {
                    Ok(Ok(text)) => Ok(text_tool_result(text)),
                    Ok(Err(msg)) => Err(msg),
                    Err(_) => Err(
                        "dispatch_thread: Function terminated without delivering a result".to_string(),
                    ),
                };
                SchedulerCompletion::Io(IoCompletion {
                    thread_id: parent_id,
                    op_id,
                    result: IoResult::ToolCall {
                        tool_use_id,
                        result,
                    },
                    pod_update: None,
                    scheduler_command: None,
                })
            }))
        } else {
            let delivery = FunctionDelivery::ToolResultFollowup {
                parent_thread_id: parent_id.clone(),
                parent_tool_use_id: tool_use_id.clone(),
            };
            if let Err(e) = self.register_function_with_delivery(spec, scope, caller, delivery) {
                return Some(immediate_err(
                    parent_id,
                    op_id,
                    tool_use_id,
                    format!("dispatch_thread: {}", super::client_messages::reject_reason_detail(&e)),
                ));
            }
            let fn_id = self
                .active_functions
                .keys()
                .max()
                .copied()
                .expect("just inserted");
            self.launch_function(fn_id, pending_io);
            // The child id is now readable off the Function entry's
            // awaiting_child_thread_id; use it in the ack so the
            // parent's next turn can reference it.
            let child_id = self
                .active_functions
                .get(&fn_id)
                .and_then(|e| e.awaiting_child_thread_id.clone())
                .unwrap_or_default();
            let ack = format!(
                "Dispatched child thread `{child_id}` in the background. Its final result will be delivered as a fresh user message once the child terminates; you can continue working in the meantime."
            );
            debug!(
                parent = %parent_id, child = %child_id,
                "dispatch_thread: async — parent continues, child result will land as a later user message"
            );
            Some(Box::pin(async move {
                SchedulerCompletion::Io(IoCompletion {
                    thread_id: parent_id,
                    op_id,
                    result: IoResult::ToolCall {
                        tool_use_id,
                        result: Ok(text_tool_result(ack)),
                    },
                    pod_update: None,
                    scheduler_command: None,
                })
            }))
        }
    }
}

fn text_tool_result(text: String) -> CallToolResult {
    CallToolResult {
        content: vec![McpContentBlock::Text { text }],
        is_error: false,
    }
}

fn immediate_err(
    parent_thread_id: String,
    op_id: OpId,
    tool_use_id: String,
    message: String,
) -> SchedulerFuture {
    Box::pin(async move {
        SchedulerCompletion::Io(IoCompletion {
            thread_id: parent_thread_id,
            op_id,
            result: IoResult::ToolCall {
                tool_use_id,
                result: Err(message),
            },
            pod_update: None,
            scheduler_command: None,
        })
    })
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DispatchStatus {
    Completed,
    Failed,
    Cancelled,
}

impl DispatchStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Render the `<dispatched-thread-notification>` envelope for the
/// async delivery path. Shape mirrors Claude Code's `<task-notification>`:
/// stamps (thread-id + original parent tool_use_id) for correlation,
/// a short summary, the child's final `<result>` inline (XML-escaped),
/// and a `<usage>` block. The resulting string is injected as the text
/// content of a fresh user message on the parent's conversation.
pub(super) fn render_dispatch_notification(
    task: &crate::runtime::thread::Thread,
    parent_tool_use_id: &str,
    status: DispatchStatus,
    body: &str,
) -> String {
    let title = task.title.as_deref().unwrap_or(task.id.as_str());
    let summary = format!("dispatched thread \"{}\" {}", title, status.as_str());
    let total_tokens = task
        .total_usage
        .input_tokens
        .saturating_add(task.total_usage.output_tokens);
    let tool_uses = count_tool_uses(&task.conversation);
    let duration_ms = (task.last_active - task.created_at)
        .num_milliseconds()
        .max(0);
    let mut out = String::new();
    out.push_str("<dispatched-thread-notification>\n");
    out.push_str(&format!(
        "  <thread-id>{}</thread-id>\n",
        escape_xml(&task.id)
    ));
    out.push_str(&format!(
        "  <tool-use-id>{}</tool-use-id>\n",
        escape_xml(parent_tool_use_id)
    ));
    out.push_str(&format!("  <status>{}</status>\n", status.as_str()));
    out.push_str(&format!("  <summary>{}</summary>\n", escape_xml(&summary)));
    out.push_str(&format!("  <result>{}</result>\n", escape_xml(body)));
    out.push_str(&format!(
        "  <usage><total_tokens>{total_tokens}</total_tokens><tool_uses>{tool_uses}</tool_uses><duration_ms>{duration_ms}</duration_ms></usage>\n"
    ));
    out.push_str("</dispatched-thread-notification>");
    out
}

fn count_tool_uses(conv: &whisper_agent_protocol::Conversation) -> u32 {
    use whisper_agent_protocol::{ContentBlock, Role};
    let mut n: u32 = 0;
    for msg in conv.messages() {
        if msg.role != Role::Assistant {
            continue;
        }
        for block in &msg.content {
            if matches!(block, ContentBlock::ToolUse { .. }) {
                n = n.saturating_add(1);
            }
        }
    }
    n
}

fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_xml_handles_angle_and_amp() {
        assert_eq!(escape_xml("<T>&"), "&lt;T&gt;&amp;");
        assert_eq!(escape_xml("plain"), "plain");
    }

    #[test]
    fn dispatch_status_str() {
        assert_eq!(DispatchStatus::Completed.as_str(), "completed");
        assert_eq!(DispatchStatus::Failed.as_str(), "failed");
        assert_eq!(DispatchStatus::Cancelled.as_str(), "cancelled");
    }
}
