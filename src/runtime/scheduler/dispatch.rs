//! `dispatch_thread` scheduler intercept + terminal-state hooks.
//!
//! The builtin tool `dispatch_thread` spawns a child thread in the same
//! pod as the caller and either parks the caller's tool call until the
//! child terminates (sync) or returns immediately with a "dispatched"
//! acknowledgment (async).
//!
//! Shape:
//!
//! - [`Scheduler::try_intercept_dispatch_thread`] runs from
//!   `step_until_blocked` whenever the parent emits a `ToolCall` I/O
//!   request. If the tool name is `dispatch_thread`, it consumes the
//!   request by spawning the child and producing the `SchedulerFuture`
//!   that eventually yields the parent's tool result.
//!
//! - Sync case: the future awaits a `oneshot::Receiver<Result<String,
//!   String>>`. The sender is stashed in [`Scheduler::pending_dispatches`]
//!   keyed by the child's thread id. When the child reaches a terminal
//!   state, [`Scheduler::resolve_pending_dispatch`] (a terminal-state
//!   hook bolted onto `step_until_blocked`) extracts the child's final
//!   assistant text and fires the sender.
//!
//! - Async case: the future resolves immediately with a text result
//!   carrying the child's id. The child runs detached; nothing is
//!   registered in `pending_dispatches`.
//!
//! - Cancellation cascade: when any thread reaches terminal while it
//!   still has dispatched-child senders keyed to it as a parent,
//!   [`Scheduler::cascade_cancel_dispatched_children`] cancels those
//!   children and drops the senders. The parent's model turn will
//!   never consume their results, so leaving them running would be
//!   wasted work.
//!
//! The side-map is transient (not persisted) — a restart mid-sync-
//! dispatch orphans the parent in `AwaitingTools` the same way it
//! orphans an in-flight MCP tool call. User recovery is `CancelThread`.

use futures::stream::FuturesUnordered;
use tokio::sync::oneshot;
use tracing::{debug, warn};
use whisper_agent_protocol::ThreadStateLabel;

use super::Scheduler;
use super::compaction::extract_last_assistant_text;
use crate::runtime::io_dispatch::{IoCompletion, SchedulerCompletion, SchedulerFuture};
use crate::runtime::thread::{IoResult, OpId};
use crate::tools::builtin_tools::DISPATCH_THREAD;
use crate::tools::builtin_tools::dispatch_thread::parse_args;
use crate::tools::mcp::{CallToolResult, McpContentBlock};

/// Compile-in cap on `dispatch_thread` nesting depth. A dispatched
/// child's own `dispatch_thread` call would be depth 2; we allow a few
/// levels for legitimate delegation chains and refuse past this so a
/// buggy agent can't recursively dispatch itself into OOM.
const MAX_DISPATCH_DEPTH: u32 = 4;

/// Per-child sender + parent linkage. Removed by `resolve_pending_dispatch`
/// once the child reaches terminal, or dropped by `cascade_cancel_dispatched_children`
/// when the parent terminates first. The parent's `tool_use_id` lives
/// inside the receiver-awaiting future's closure, not here — this map
/// exists only for the reverse `parent → pending children` lookup the
/// cascade-cancel hook needs.
pub(super) struct PendingDispatch {
    pub parent_thread_id: String,
    pub sender: oneshot::Sender<Result<String, String>>,
}

impl Scheduler {
    /// If `tool_name == "dispatch_thread"`, consume the request: spawn
    /// the child, set up return plumbing, and return the parent's
    /// tool-call future. Returns `None` for any other tool name — the
    /// caller falls through to the normal `io_dispatch::build_io_future`
    /// path.
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

        let (parent_depth, pod_id) = match self.tasks.get(parent_thread_id) {
            Some(t) => (t.dispatch_depth, Some(t.pod_id.clone())),
            None => {
                return Some(immediate_err(
                    parent_id,
                    op_id,
                    tool_use_id,
                    "dispatch_thread: parent thread vanished".into(),
                ));
            }
        };
        if parent_depth >= MAX_DISPATCH_DEPTH {
            return Some(immediate_err(
                parent_id,
                op_id,
                tool_use_id,
                format!(
                    "dispatch_thread: nesting cap reached (parent depth {parent_depth}, max {MAX_DISPATCH_DEPTH})"
                ),
            ));
        }

        let child_id = match self.create_task(
            None,
            None,
            pod_id,
            args.config_override,
            args.bindings_override,
            None,
            Some((parent_id.clone(), parent_depth)),
            pending_io,
        ) {
            Ok(id) => id,
            Err(e) => {
                return Some(immediate_err(
                    parent_id,
                    op_id,
                    tool_use_id,
                    format!("dispatch_thread: {e}"),
                ));
            }
        };

        // For sync dispatch, register the oneshot sender BEFORE we step
        // the child. The first `step_until_blocked` call below can
        // reach a terminal state in one pass (e.g. the child's model
        // call errors immediately, or the model returns `end_turn`
        // without any tool calls). The terminal-state hook is the one
        // that fires the sender — if we register afterwards, a
        // terminal-before-insert race leaves the entry unread and the
        // parent's future hangs forever. Inserting first is safe: the
        // oneshot buffers a send even with no receiver yet attached.
        let sync_rx = if args.sync {
            let (tx, rx) = oneshot::channel();
            self.pending_dispatches.insert(
                child_id.clone(),
                PendingDispatch {
                    parent_thread_id: parent_id.clone(),
                    sender: tx,
                },
            );
            Some(rx)
        } else {
            None
        };

        // Seed + kick.
        self.send_user_message(&child_id, args.prompt.clone(), pending_io);
        self.step_until_blocked(&child_id, pending_io);

        if let Some(rx) = sync_rx {
            debug!(
                parent = %parent_id, child = %child_id, depth = parent_depth + 1,
                "dispatch_thread: sync — parent parked awaiting child terminal"
            );
            Some(Box::pin(async move {
                let result = match rx.await {
                    Ok(Ok(text)) => Ok(text_tool_result(text)),
                    Ok(Err(msg)) => Err(msg),
                    Err(_) => Err(
                        "dispatch_thread: child terminated without delivering a result".to_string(),
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
            let ack = format!(
                "Dispatched child thread `{child_id}` in the background. Its conversation is visible via the UI; this tool call does not wait for its result."
            );
            debug!(
                parent = %parent_id, child = %child_id, depth = parent_depth + 1,
                "dispatch_thread: async — parent continues, child detached"
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

    /// Terminal-state hook: if `thread_id` is a dispatched child with a
    /// waiting parent, extract its final assistant text and fire the
    /// registered oneshot. No-op otherwise.
    pub(super) fn resolve_pending_dispatch(
        &mut self,
        thread_id: &str,
        _pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if !self.pending_dispatches.contains_key(thread_id) {
            return;
        }
        let Some(task) = self.tasks.get(thread_id) else {
            // Thread vanished — drop the pending entry, receiver gets
            // a RecvError which the future translates to a terminal
            // error result.
            self.pending_dispatches.remove(thread_id);
            return;
        };
        let state = task.public_state();
        let msg = match state {
            ThreadStateLabel::Completed => Ok(extract_last_assistant_text(task)),
            ThreadStateLabel::Failed => Err(format!(
                "dispatched child `{thread_id}` failed: {}",
                task.failure_detail()
                    .unwrap_or_else(|| "unknown error".into())
            )),
            ThreadStateLabel::Cancelled => {
                Err(format!("dispatched child `{thread_id}` was cancelled"))
            }
            // Not terminal — leave the entry in place, wait for the
            // next step that pushes this thread to terminal.
            ThreadStateLabel::Idle
            | ThreadStateLabel::Working
            | ThreadStateLabel::AwaitingApproval => return,
        };
        if let Some(pending) = self.pending_dispatches.remove(thread_id) {
            // send() only fails when the receiver has been dropped —
            // i.e. the parent already terminated. Not a scheduler-level
            // error; the parent's future is gone, nothing to deliver to.
            let _ = pending.sender.send(msg);
        }
    }

    /// Terminal-state hook: if `thread_id` was a parent of any
    /// outstanding dispatched children, cancel those children and drop
    /// their senders. The parent's tool call is no longer going to
    /// consume the result, so keeping the child running is wasted
    /// compute.
    pub(super) fn cascade_cancel_dispatched_children(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        // Only cascade when this thread has actually reached terminal;
        // an interim park on a parent shouldn't evict children.
        let Some(task) = self.tasks.get(thread_id) else {
            return;
        };
        let terminal = matches!(
            task.public_state(),
            ThreadStateLabel::Completed | ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
        );
        if !terminal {
            return;
        }
        // A terminal parent with pending dispatches means sync
        // dispatches whose parent died before the child did — odd but
        // possible (parent Cancelled mid-wait, parent Failed on some
        // unrelated op, parent Completed without consuming the pending
        // tool result). Walk the map and evict.
        let orphaned_children: Vec<String> = self
            .pending_dispatches
            .iter()
            .filter(|(_, pd)| pd.parent_thread_id == thread_id)
            .map(|(child_id, _)| child_id.clone())
            .collect();
        for child_id in orphaned_children {
            // Drop the entry first — the child's own terminal hook
            // will then see no pending entry and skip the oneshot send.
            self.pending_dispatches.remove(&child_id);
            if let Some(child) = self.tasks.get_mut(&child_id) {
                let already_terminal = matches!(
                    child.public_state(),
                    ThreadStateLabel::Completed
                        | ThreadStateLabel::Failed
                        | ThreadStateLabel::Cancelled
                );
                if already_terminal {
                    continue;
                }
                child.cancel();
                self.mark_dirty(&child_id);
                self.router.broadcast_task_list(
                    whisper_agent_protocol::ServerToClient::ThreadStateChanged {
                        thread_id: child_id.clone(),
                        state: ThreadStateLabel::Cancelled,
                    },
                );
                warn!(
                    parent = %thread_id, child = %child_id,
                    "dispatch_thread: parent terminated; cascading cancel to dispatched child"
                );
                // Teardown + behavior-origin hooks, matching the
                // explicit CancelThread path so resource refcounts
                // stay honest.
                self.teardown_host_env_if_terminal(&child_id);
                self.on_behavior_thread_terminal(&child_id, pending_io);
            }
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
