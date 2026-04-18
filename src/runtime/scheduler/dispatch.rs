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

/// Per-child linkage + delivery intent. One entry per dispatched
/// child thread (sync or async). The parent's `tool_use_id` only
/// matters for the sync variant and lives inside its oneshot-awaiting
/// future's closure — this map is keyed by child id because the child
/// is what the terminal-state hooks look up, and `parent_thread_id`
/// serves the reverse lookup used by the cascade-cancel hook.
pub(super) struct PendingDispatch {
    pub parent_thread_id: String,
    pub delivery: DispatchDelivery,
}

/// How to deliver the child's final assistant text back to the parent.
///
/// - [`ToolResult`] — sync: fill the parent's parked `tool_use_id` via
///   the oneshot. Parent resumes its current turn with the child's
///   final text as the dispatch_thread tool result.
///
/// - [`Async`] — async: the parent's dispatch_thread tool call already
///   returned an immediate "Dispatched task-X" ack, so there's no
///   parked tool result to fill. Instead, when the child terminates
///   we inject the final text as a fresh **user** message on the
///   parent's conversation (if the parent can accept one), which
///   kicks a new turn. Matches the "MCP returns a message, LLM gets
///   another turn" pattern — async is the same routing as sync, just
///   delivered via a new turn rather than the tool-result slot.
///   `pending_text` is `None` until the child terminates; once set,
///   the delivery is flushed to the parent the next time the parent
///   reaches an idle-enough state (`Thread::is_idle`). A parent that
///   ends up Failed or Cancelled before flush drops the delivery.
pub(super) enum DispatchDelivery {
    ToolResult {
        sender: oneshot::Sender<Result<String, String>>,
    },
    Async {
        pending_text: Option<String>,
    },
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

        // Register the delivery intent BEFORE stepping the child.
        // The first `step_until_blocked` call below can reach terminal
        // in a single pass (immediate error, no-tool-call turn, etc.),
        // and the terminal-state hook is what fires the delivery — if
        // we register afterwards a terminal-before-insert race leaves
        // the entry uninstalled and the parent never hears back.
        // Both sync and async register: the unified side-map carries
        // the routing regardless of whether the parent parks.
        let sync_rx = if args.sync {
            let (tx, rx) = oneshot::channel();
            self.pending_dispatches.insert(
                child_id.clone(),
                PendingDispatch {
                    parent_thread_id: parent_id.clone(),
                    delivery: DispatchDelivery::ToolResult { sender: tx },
                },
            );
            Some(rx)
        } else {
            self.pending_dispatches.insert(
                child_id.clone(),
                PendingDispatch {
                    parent_thread_id: parent_id.clone(),
                    delivery: DispatchDelivery::Async { pending_text: None },
                },
            );
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
                "Dispatched child thread `{child_id}` in the background. Its final result will be delivered as a fresh user message once the child terminates; you can continue working in the meantime."
            );
            debug!(
                parent = %parent_id, child = %child_id, depth = parent_depth + 1,
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

    /// Terminal-state hook: if `thread_id` is a dispatched child with a
    /// registered delivery intent, run the delivery. For sync children
    /// that fires the oneshot; for async children it stashes the final
    /// text on the entry so [`Scheduler::flush_ready_async_deliveries`]
    /// can inject it as a user message on the parent once the parent
    /// is idle enough to accept one. No-op otherwise.
    pub(super) fn resolve_pending_dispatch(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if !self.pending_dispatches.contains_key(thread_id) {
            return;
        }
        let Some(task) = self.tasks.get(thread_id) else {
            // Thread vanished — drop the pending entry. For sync, the
            // receiver gets a RecvError which the future maps to a
            // terminal error result. For async, nothing to deliver.
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
        let Some(entry) = self.pending_dispatches.get_mut(thread_id) else {
            return;
        };
        match &mut entry.delivery {
            DispatchDelivery::ToolResult { .. } => {
                // Take ownership to fire send() — send() consumes the
                // sender.
                let removed = self
                    .pending_dispatches
                    .remove(thread_id)
                    .expect("just-checked");
                if let DispatchDelivery::ToolResult { sender } = removed.delivery {
                    // send() only fails when the receiver has been
                    // dropped — i.e. the parent already terminated. Not
                    // a scheduler-level error; nothing to deliver to.
                    let _ = sender.send(msg);
                }
            }
            DispatchDelivery::Async { pending_text } => {
                // Stash the final text (or the error text) on the
                // entry; the flush hook will inject it as a user
                // message once the parent is idle enough. An async
                // dispatch surfaces child failures as a human-readable
                // note rather than an error tool result — the parent's
                // dispatch_thread call already closed out with the
                // "dispatched" ack, so there's no error channel per se.
                let text = match msg {
                    Ok(t) => format!("[dispatched thread `{thread_id}` completed]\n{t}"),
                    Err(e) => format!("[dispatched thread `{thread_id}` ended] {e}"),
                };
                *pending_text = Some(text);
                // Try flushing now — if the parent happens to already
                // be idle, we don't want to wait for its next step to
                // notice. Common case: parent already Completed before
                // its long-running async child finished.
                let parent_id = entry.parent_thread_id.clone();
                self.flush_ready_async_deliveries(&parent_id, pending_io);
            }
        }
    }

    /// Per-parent hook: walk `pending_dispatches` for entries whose
    /// parent is this thread AND whose delivery is an async-ready
    /// message. For each such entry, if the parent can accept a user
    /// message (`Thread::is_idle` — includes Idle/Completed; Failed
    /// and Cancelled drop the delivery), append the message and
    /// remove the entry. Kicks a new parent turn via the standard
    /// `send_user_message` path so the LLM gets a follow-up turn
    /// analogous to an MCP tool result arriving mid-conversation.
    pub(super) fn flush_ready_async_deliveries(
        &mut self,
        parent_thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        // Collect candidate child ids with ready text. Two-phase loop
        // because `send_user_message` needs `&mut self` and we're
        // iterating `&self.pending_dispatches`.
        let ready: Vec<(String, String)> = self
            .pending_dispatches
            .iter()
            .filter_map(|(child_id, pd)| {
                if pd.parent_thread_id != parent_thread_id {
                    return None;
                }
                match &pd.delivery {
                    DispatchDelivery::Async {
                        pending_text: Some(text),
                    } => Some((child_id.clone(), text.clone())),
                    _ => None,
                }
            })
            .collect();
        if ready.is_empty() {
            return;
        }
        // Parent state check: Idle and Completed accept new user
        // messages; Failed and Cancelled drop the delivery silently.
        // For in-between states (Working, AwaitingApproval) leave the
        // text stashed — we'll retry on the next step_until_blocked
        // for this parent.
        let Some(parent) = self.tasks.get(parent_thread_id) else {
            // Parent vanished — drop all queued deliveries.
            for (child_id, _) in &ready {
                self.pending_dispatches.remove(child_id);
            }
            return;
        };
        let parent_state = parent.public_state();
        let can_accept = matches!(
            parent_state,
            ThreadStateLabel::Idle | ThreadStateLabel::Completed
        );
        let parent_terminal = matches!(
            parent_state,
            ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
        );
        if parent_terminal {
            for (child_id, _) in &ready {
                debug!(
                    parent = %parent_thread_id, child = %child_id,
                    state = ?parent_state,
                    "dispatch_thread: dropping async delivery to terminal parent"
                );
                self.pending_dispatches.remove(child_id);
            }
            return;
        }
        if !can_accept {
            return;
        }
        for (child_id, text) in ready {
            self.pending_dispatches.remove(&child_id);
            debug!(
                parent = %parent_thread_id, child = %child_id,
                "dispatch_thread: flushing async delivery as user message"
            );
            self.send_user_message(parent_thread_id, text, pending_io);
            // Step the parent so the new user message actually kicks
            // a model call instead of sitting idle. Subsequent
            // deliveries in this loop will find the parent in a
            // non-idle state on the next flush attempt — that's fine,
            // they'll land on the following turn boundary.
            self.step_until_blocked(parent_thread_id, pending_io);
            // If stepping the parent made it non-idle, bail so the
            // remaining deliveries wait for the next boundary rather
            // than getting interleaved into the same turn.
            let still_idle = self
                .tasks
                .get(parent_thread_id)
                .map(|t| {
                    matches!(
                        t.public_state(),
                        ThreadStateLabel::Idle | ThreadStateLabel::Completed
                    )
                })
                .unwrap_or(false);
            if !still_idle {
                break;
            }
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
