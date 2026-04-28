//! Function registry on the scheduler.
//!
//! The registry is the scheduler's source of truth for in-flight
//! caller-initiated operations. Each entry is an `ActiveFunctionEntry` —
//! spec + scope + caller-link. `register_function` is the synchronous
//! accept/deny path; per-variant `launch_*` methods run the actual work
//! (which may complete inline for sync variants, or span many scheduler
//! ticks for async variants), and `complete_function` removes the entry
//! and emits the terminal.
//!
//! Currently wired:
//!
//! - `Function::CancelThread` — synchronous; launch completes inline.
//! - `Function::CompactThread` — asynchronous; launches the compaction
//!   turn and stays in the registry until `finalize_pending_compaction`
//!   fires `complete_function`.

use chrono::{DateTime, Utc};
use futures::stream::FuturesUnordered;
use tokio::sync::oneshot;
use tracing::{debug, warn};
use whisper_agent_protocol::{
    FunctionOutcomeTag, FunctionSummary, ServerToClient, ThreadStateLabel,
};

use super::{ConnId, Scheduler};
use crate::functions::{
    CallerLink, Function, FunctionId, FunctionOutcome, FunctionTerminal, InFlightOps, RejectReason,
};
use crate::runtime::io_dispatch::SchedulerFuture;

/// Compile-in cap on `dispatch_thread` nesting depth. A dispatched
/// child's own `dispatch_thread` call would be depth 2; we allow a
/// few levels for legitimate delegation chains and refuse past this
/// so a buggy agent can't recursively dispatch itself into OOM.
pub(super) const MAX_DISPATCH_DEPTH: u32 = 4;

/// Server-owned runtime state of an in-flight Function.
///
/// Phase-2/3 shape: minimal — enough to support synchronous variants
/// (CancelThread) and multi-tick async variants that carry only a
/// thread_id (CompactThread). Full progress/terminal channels land when
/// the tool-call variants migrate in Commit 4.
#[derive(Debug)]
// Fields are written at registration and read by variant launch/complete
// paths + the cancel-by-thread sweep; silence early dead-code until
// those consumers land in Commit 4+.
#[allow(dead_code)]
pub struct ActiveFunctionEntry {
    pub id: FunctionId,
    pub spec: Function,
    pub caller: CallerLink,
    /// For `Function::CreateThread { wait_mode: ThreadTerminal, .. }`:
    /// the child thread whose terminal completes this Function. The
    /// scheduler's thread-terminal hook reads this to decide which
    /// Functions to fire.
    pub awaiting_child_thread_id: Option<String>,
    /// How the terminal is delivered to the caller. Set at
    /// registration; consumed by `complete_function`.
    pub delivery: FunctionDelivery,
    /// Wall-clock at registration time. Emitted on the wire via
    /// `FunctionSummary.started_at` so UI surfaces can compute
    /// elapsed time — which is the primary signal that distinguishes
    /// "quick tool call flickering by" from "long-running dispatch
    /// worth attention." Captured once and never mutated.
    pub started_at: DateTime<Utc>,
}

impl ActiveFunctionEntry {
    /// Project the entry down to the display-facing wire summary used
    /// in `FunctionStarted` and `FunctionList`. Cheap — strings are
    /// cloned but all of them are small (ids + a RFC3339 timestamp).
    pub(crate) fn wire_summary(&self) -> FunctionSummary {
        let (thread_id, pod_id, tool_name, behavior_id) = self.spec.wire_scope_fields();
        FunctionSummary {
            function_id: self.id,
            kind: self.spec.wire_kind(),
            thread_id,
            pod_id,
            tool_name,
            behavior_id,
            caller_tag: self.caller.audit_tag(),
            started_at: self.started_at.to_rfc3339(),
        }
    }
}

/// How the terminal of an in-flight Function gets delivered back to
/// its caller beyond the default wire-Error routing in
/// `complete_function`. Most variants use `None`; dispatch-driven
/// tool calls populate one of the tool-result variants so the
/// parent's tool_use_id eventually sees the child's result.
pub enum FunctionDelivery {
    /// Default — no variant-specific delivery. Errors for `WsClient`
    /// callers still produce a wire `Error`; successful terminals
    /// rely on variant-level broadcasts (`ThreadCompacted`, etc.).
    None,
    /// Sync tool-call parking: fire this oneshot with the terminal's
    /// final text (`Ok` on Success, `Err` on Error/Cancelled). The
    /// parent's tool-call SchedulerFuture awaits the receiver and
    /// converts the result into an `IoResult::ToolCall` — normal tool
    /// result delivery.
    ToolResultChannel(oneshot::Sender<Result<String, String>>),
    /// Async tool-call follow-up: render a
    /// `<dispatched-thread-notification>` envelope from the terminal
    /// and inject it as a fresh user message on the parent thread.
    /// The parent's tool call already returned a synthetic ack, so
    /// there's no parked oneshot to fire.
    ToolResultFollowup {
        parent_thread_id: String,
        parent_tool_use_id: String,
    },
}

impl std::fmt::Debug for FunctionDelivery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::ToolResultChannel(_) => write!(f, "ToolResultChannel(<sender>)"),
            Self::ToolResultFollowup {
                parent_thread_id,
                parent_tool_use_id,
            } => f
                .debug_struct("ToolResultFollowup")
                .field("parent_thread_id", parent_thread_id)
                .field("parent_tool_use_id", parent_tool_use_id)
                .finish(),
        }
    }
}

impl Scheduler {
    /// Fresh `FunctionId`, incrementing the scheduler's counter.
    fn next_function_id(&mut self) -> FunctionId {
        let id = self.next_function_id;
        self.next_function_id = self.next_function_id.wrapping_add(1);
        id
    }

    /// Synchronous accept-or-deny registration. On accept the
    /// `ActiveFunctionEntry` enters the registry and its `FunctionId` is
    /// returned. Anything that requires awaiting I/O happens later, in
    /// the variant's launch path.
    pub(super) fn register_function(
        &mut self,
        spec: Function,
        caller: CallerLink,
    ) -> Result<FunctionId, RejectReason> {
        self.register_function_with_delivery(spec, caller, FunctionDelivery::None)
    }

    /// Full registration shape used by callers that need to attach a
    /// non-default terminal delivery (dispatch_thread's sync oneshot or
    /// async follow-up). Most callers use `register_function` and get
    /// `FunctionDelivery::None`.
    pub(super) fn register_function_with_delivery(
        &mut self,
        spec: Function,
        caller: CallerLink,
        delivery: FunctionDelivery,
    ) -> Result<FunctionId, RejectReason> {
        self.variant_precondition_check(&spec)?;
        let id = self.next_function_id();
        let entry = ActiveFunctionEntry {
            id,
            spec,
            caller,
            awaiting_child_thread_id: None,
            delivery,
            started_at: Utc::now(),
        };
        debug!(
            function_id = id,
            caller = %entry.caller.audit_tag(),
            "Function registered"
        );
        // Broadcast before insertion so the summary snapshot reflects
        // the just-registered state without an extra lookup round-trip.
        let summary = entry.wire_summary();
        self.active_functions.insert(id, entry);
        self.router
            .broadcast_task_list(ServerToClient::FunctionStarted { summary });
        Ok(id)
    }

    /// Variant-specific precondition check. Called synchronously
    /// from `register_function` before admission. The permissions
    /// layer is mid-rework — capability-admission checks return here
    /// when the new `Scope` type lands (task #3 / #4).
    fn variant_precondition_check(&self, spec: &Function) -> Result<(), RejectReason> {
        match spec {
            Function::CancelThread { thread_id } => {
                self.tasks
                    .get(thread_id)
                    .ok_or_else(|| RejectReason::PreconditionFailed {
                        detail: format!("unknown thread {thread_id}"),
                    })?;
                Ok(())
            }
            Function::CompactThread { thread_id } => {
                let task =
                    self.tasks
                        .get(thread_id)
                        .ok_or_else(|| RejectReason::PreconditionFailed {
                            detail: format!("unknown thread {thread_id}"),
                        })?;
                if !task.config.compaction.enabled {
                    return Err(RejectReason::PreconditionFailed {
                        detail: "compaction is disabled on this thread".into(),
                    });
                }
                if task.in_flight.contains(InFlightOps::COMPACTING) {
                    return Err(RejectReason::ResourceBusy {
                        detail: "compaction already in progress".into(),
                    });
                }
                if !task.is_idle() {
                    return Err(RejectReason::PreconditionFailed {
                        detail: "thread is not at a clean turn boundary".into(),
                    });
                }
                Ok(())
            }
            Function::CreateThread { pod_id, parent, .. } => {
                let effective_pod = pod_id
                    .clone()
                    .unwrap_or_else(|| self.default_pod_id.clone());
                if !self.pods.contains_key(&effective_pod) {
                    return Err(RejectReason::PreconditionFailed {
                        detail: format!("unknown pod `{effective_pod}`"),
                    });
                }
                if let Some(link) = parent {
                    let parent_task = self.tasks.get(&link.thread_id).ok_or_else(|| {
                        RejectReason::PreconditionFailed {
                            detail: format!("unknown parent thread `{}`", link.thread_id),
                        }
                    })?;
                    if parent_task.dispatch_depth >= MAX_DISPATCH_DEPTH {
                        return Err(RejectReason::PreconditionFailed {
                            detail: format!(
                                "dispatch_thread nesting cap reached (parent depth {}, max {})",
                                parent_task.dispatch_depth, MAX_DISPATCH_DEPTH
                            ),
                        });
                    }
                }
                Ok(())
            }
            Function::RunBehavior {
                pod_id,
                behavior_id,
                ..
            } => {
                let pod =
                    self.pods
                        .get(pod_id)
                        .ok_or_else(|| RejectReason::PreconditionFailed {
                            detail: format!("unknown pod `{pod_id}`"),
                        })?;
                let behavior = pod.behaviors.get(behavior_id).ok_or_else(|| {
                    RejectReason::PreconditionFailed {
                        detail: format!("unknown behavior `{behavior_id}` under pod `{pod_id}`"),
                    }
                })?;
                if let Some(err) = &behavior.load_error {
                    return Err(RejectReason::PreconditionFailed {
                        detail: format!("behavior `{behavior_id}` failed to load: {err}"),
                    });
                }
                if behavior.config.is_none() {
                    return Err(RejectReason::PreconditionFailed {
                        detail: format!("behavior `{behavior_id}` has no parsed config"),
                    });
                }
                Ok(())
            }
            Function::BuiltinToolCall { .. } | Function::McpToolUse { .. } => Ok(()),
            Function::Sudo { thread_id, .. } => {
                let task =
                    self.tasks
                        .get(thread_id)
                        .ok_or_else(|| RejectReason::PreconditionFailed {
                            detail: format!("unknown thread {thread_id}"),
                        })?;
                if !task.scope.escalation.is_interactive() {
                    return Err(RejectReason::ScopeDenied {
                        detail: "thread has no interactive approver for `sudo`".into(),
                    });
                }
                Ok(())
            }
        }
    }

    /// Launch an admitted Function. For synchronous variants this also
    /// emits the terminal (via `complete_function`) before returning;
    /// async variants return with the entry still in `active_functions`
    /// and complete later (e.g., when a subsequent scheduler tick fires
    /// the completion hook).
    pub(super) fn launch_function(
        &mut self,
        id: FunctionId,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let spec = match self.active_functions.get(&id) {
            Some(e) => e.spec.clone(),
            None => {
                warn!(function_id = id, "launch_function: no such entry");
                return;
            }
        };
        match spec {
            Function::CancelThread { thread_id } => {
                self.execute_cancel_thread(&thread_id, pending_io);
                self.complete_function(
                    id,
                    FunctionOutcome::Success(FunctionTerminal::CancelThread),
                    pending_io,
                );
            }
            Function::CompactThread { thread_id } => {
                self.launch_compact_thread(&thread_id, pending_io);
                // Function stays in the registry until
                // `finalize_pending_compaction` calls `complete_function`.
            }
            Function::CreateThread {
                pod_id,
                initial_message,
                initial_attachments,
                parent,
                wait_mode,
                config_override,
                bindings_request,
            } => {
                self.launch_create_thread(
                    id,
                    pod_id,
                    initial_message,
                    initial_attachments,
                    parent,
                    wait_mode,
                    config_override,
                    bindings_request,
                    pending_io,
                );
            }
            Function::RunBehavior {
                pod_id,
                behavior_id,
                payload,
            } => {
                // Pull requester + correlation_id from the caller-link
                // so `create_task` routes `ThreadCreated` broadcasts the
                // same way interactive / trigger-driven paths used to.
                let (requester, correlation_id) = match self.active_functions.get(&id) {
                    Some(entry) => match &entry.caller {
                        CallerLink::WsClient {
                            conn_id,
                            correlation_id,
                        } => (Some(*conn_id), correlation_id.clone()),
                        _ => (None, None),
                    },
                    None => (None, None),
                };
                match self.run_behavior(
                    requester,
                    correlation_id,
                    &pod_id,
                    &behavior_id,
                    Some(payload),
                    pending_io,
                ) {
                    Ok(thread_id) => self.complete_function(
                        id,
                        FunctionOutcome::Success(FunctionTerminal::RunBehavior(
                            crate::functions::RunBehaviorTerminal { thread_id },
                        )),
                        pending_io,
                    ),
                    Err(e) => self.complete_function(
                        id,
                        FunctionOutcome::Error(crate::functions::FunctionError {
                            kind: crate::functions::FunctionErrorKind::Execution,
                            detail: e,
                        }),
                        pending_io,
                    ),
                }
            }
            Function::BuiltinToolCall { .. } | Function::McpToolUse { .. } => {
                // The tool call's actual work is pumped by the existing
                // IO-future mechanism in `build_io_future`; the Function
                // just records the call in the registry. Phase 4b keeps
                // this wrap-only shape — Phase 4c moves approval
                // evaluation into launch, Phase 4d moves the dispatch
                // itself into the Function's future.
                //
                // The Function stays admitted until
                // `apply_io_completion` locates it by caller-link
                // (ThreadToolCall { thread_id, tool_use_id }) and calls
                // `complete_function`.
            }
            Function::Sudo {
                thread_id,
                tool_name,
                args,
                reason,
            } => {
                // Parallel to RequestEscalation's launch path: emit the
                // wire approval request to the thread's interactive
                // channel; the Function stays admitted until
                // `resolve_sudo` fires.
                let via_conn = match self.tasks.get(&thread_id).map(|t| t.scope.escalation) {
                    Some(crate::permission::Escalation::Interactive { via_conn }) => via_conn,
                    _ => {
                        self.complete_function(
                            id,
                            FunctionOutcome::Error(crate::functions::FunctionError {
                                kind: crate::functions::FunctionErrorKind::Execution,
                                detail: "interactive approver unavailable".into(),
                            }),
                            pending_io,
                        );
                        return;
                    }
                };
                self.router.send_to_client(
                    via_conn,
                    ServerToClient::SudoRequested {
                        function_id: id,
                        thread_id: thread_id.clone(),
                        tool_name: tool_name.clone(),
                        args: args.clone(),
                        reason: reason.clone(),
                    },
                );
            }
        }
    }

    /// Remove the entry from `active_functions` and log/audit the
    /// terminal, then run any caller-specific delivery. Error
    /// terminals for `WsClient` callers with a correlation_id surface
    /// as `ServerToClient::Error`. Terminals with
    /// `FunctionDelivery::ToolResultChannel` fire the parked oneshot
    /// so the parent's parked tool-call SchedulerFuture resumes;
    /// `ToolResultFollowup` renders the terminal text as a
    /// `<dispatched-thread-notification>` envelope and injects it as
    /// a user message on the parent.
    pub(super) fn complete_function(
        &mut self,
        id: FunctionId,
        outcome: FunctionOutcome,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(entry) = self.active_functions.remove(&id) else {
            warn!(function_id = id, "complete_function: no such entry");
            return;
        };
        debug!(
            function_id = id,
            caller = %entry.caller.audit_tag(),
            outcome = ?outcome,
            "Function terminal"
        );
        // Paired with the `FunctionStarted` broadcast at
        // registration time. UI surfaces use the outcome tag to pick
        // success / error / cancelled chip colors when they animate
        // out; the variant-specific terminal payload rides other
        // dedicated events (`ThreadStateChanged`, tool-result messages,
        // etc.).
        let outcome_tag = match &outcome {
            FunctionOutcome::Success(_) => FunctionOutcomeTag::Success,
            FunctionOutcome::Error(_) => FunctionOutcomeTag::Error,
            FunctionOutcome::Cancelled(_) => FunctionOutcomeTag::Cancelled,
        };
        self.router
            .broadcast_task_list(ServerToClient::FunctionEnded {
                function_id: id,
                outcome: outcome_tag,
            });
        let wire_error_for_ws_client = if let (
            FunctionOutcome::Error(err),
            CallerLink::WsClient {
                conn_id,
                correlation_id,
            },
        ) = (&outcome, &entry.caller)
        {
            let thread_id = entry.spec.primary_thread_id().map(str::to_string);
            Some((
                *conn_id,
                correlation_id.clone(),
                thread_id,
                err.detail.clone(),
            ))
        } else {
            None
        };
        if let Some((conn_id, correlation_id, thread_id, message)) = wire_error_for_ws_client {
            self.router.send_to_client(
                conn_id,
                ServerToClient::Error {
                    correlation_id,
                    thread_id,
                    message,
                },
            );
        }

        match entry.delivery {
            FunctionDelivery::None => {}
            FunctionDelivery::ToolResultChannel(sender) => {
                let msg = outcome_to_sync_tool_result(&outcome);
                // send() fails only if the receiver has been dropped
                // — e.g. the parent already terminated. Nothing to do
                // in that case; the cascade-cancel handled the parent
                // side.
                let _ = sender.send(msg);
            }
            FunctionDelivery::ToolResultFollowup {
                parent_thread_id,
                parent_tool_use_id,
            } => {
                self.deliver_async_followup(
                    &parent_thread_id,
                    &parent_tool_use_id,
                    &entry.spec,
                    &outcome,
                    pending_io,
                );
            }
        }
    }

    /// Render a `<dispatched-thread-notification>` envelope from the
    /// completed Function's terminal and inject it as a fresh user
    /// message on the parent thread. No-op if the parent can't
    /// currently accept one (parent gone, in a terminal state of its
    /// own). The Function spec's `CreateThread`-variant carries the
    /// child thread id; the terminal carries its final text.
    fn deliver_async_followup(
        &mut self,
        parent_thread_id: &str,
        parent_tool_use_id: &str,
        spec: &Function,
        outcome: &FunctionOutcome,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let child_thread_id = match spec {
            Function::CreateThread { .. } => match outcome {
                FunctionOutcome::Success(FunctionTerminal::CreateThread(term)) => {
                    term.thread_id.clone()
                }
                _ => return,
            },
            _ => return,
        };
        let Some(child_task) = self.tasks.get(&child_thread_id) else {
            return;
        };
        let (status, body) = match outcome {
            FunctionOutcome::Success(FunctionTerminal::CreateThread(term)) => (
                crate::runtime::scheduler::dispatch::DispatchStatus::Completed,
                term.final_result
                    .as_ref()
                    .and_then(|s| s.final_text.clone())
                    .unwrap_or_default(),
            ),
            FunctionOutcome::Error(err) => (
                crate::runtime::scheduler::dispatch::DispatchStatus::Failed,
                format!("child failed: {}", err.detail),
            ),
            FunctionOutcome::Cancelled(_) => (
                crate::runtime::scheduler::dispatch::DispatchStatus::Cancelled,
                String::new(),
            ),
            FunctionOutcome::Success(_) => return,
        };
        let notification = crate::runtime::scheduler::dispatch::render_dispatch_notification(
            child_task,
            parent_tool_use_id,
            status,
            &body,
        );
        let Some(parent) = self.tasks.get_mut(parent_thread_id) else {
            // Parent gone — drop silently.
            return;
        };
        let parent_state = parent.public_state();
        let parent_terminal = matches!(
            parent_state,
            ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
        );
        if parent_terminal {
            debug!(
                parent = %parent_thread_id,
                child = %child_thread_id,
                state = ?parent_state,
                "dispatch async follow-up: parent in terminal state; dropping delivery"
            );
            return;
        }
        let parent_can_accept = matches!(
            parent_state,
            ThreadStateLabel::Idle | ThreadStateLabel::Completed
        );
        if !parent_can_accept {
            // Parent is mid-turn; queue the notification on the
            // parent's state. `drain_pending_tool_result_followups`
            // (called from `step_until_blocked` when the parent
            // reaches an idle boundary) injects each queued entry as
            // a fresh user message.
            debug!(
                parent = %parent_thread_id,
                child = %child_thread_id,
                state = ?parent_state,
                "dispatch async follow-up: parent busy; queueing for next idle boundary"
            );
            parent.pending_tool_result_followups.push(notification);
            self.mark_dirty(parent_thread_id);
            return;
        }
        debug!(
            parent = %parent_thread_id,
            child = %child_thread_id,
            "dispatch async follow-up: injecting notification as user message"
        );
        self.send_tool_result_text(parent_thread_id, notification, pending_io);
        self.step_until_blocked(parent_thread_id, pending_io);
    }

    /// Find the FunctionId of the in-flight `CompactThread` targeting
    /// `thread_id`, if any. Linear scan; `active_functions` stays small
    /// enough that a scan is fine at the expected scale.
    pub(super) fn find_compact_function_for(&self, thread_id: &str) -> Option<FunctionId> {
        self.active_functions
            .iter()
            .find_map(|(id, entry)| match &entry.spec {
                Function::CompactThread { thread_id: t } if t == thread_id => Some(*id),
                _ => None,
            })
    }

    /// Find the FunctionId of the in-flight tool-call Function (i.e.,
    /// `BuiltinToolCall` / `McpToolUse`) for this `(thread_id,
    /// tool_use_id)` pair. Restricted to tool-call specs on purpose —
    /// `dispatch_thread` shares the same caller-link shape via
    /// `Function::CreateThread`, but its terminal is driven by the
    /// child thread's lifecycle, not by a tool-IO completion. Matching
    /// it here would race the child and complete the Function twice.
    pub(super) fn find_tool_function_for(
        &self,
        thread_id: &str,
        tool_use_id: &str,
    ) -> Option<FunctionId> {
        self.active_functions.iter().find_map(|(id, entry)| {
            let is_tool_variant = matches!(
                entry.spec,
                Function::BuiltinToolCall { .. } | Function::McpToolUse { .. }
            );
            if !is_tool_variant {
                return None;
            }
            if let CallerLink::ThreadToolCall {
                thread_id: t,
                tool_use_id: u,
            } = &entry.caller
                && t == thread_id
                && u == tool_use_id
            {
                return Some(*id);
            }
            None
        })
    }

    /// Single entry point for tool-use dispatch. Recognizes the
    /// `dispatch_thread` pool-alias and routes it to
    /// `Function::CreateThread`; every other tool name maps to
    /// `Function::BuiltinToolCall` / `Function::McpToolUse` via
    /// `route_tool`. Handles admission (tool disposition +
    /// registration preconditions), registers the Function, and
    /// pushes the appropriate IO future into `pending_io`.
    pub(super) fn register_tool_function(
        &mut self,
        thread_id: &str,
        io_request: crate::runtime::thread::IoRequest,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        use super::ToolRoute;
        let crate::runtime::thread::IoRequest::ToolCall {
            op_id,
            tool_use_id,
            name,
            input,
        } = io_request.clone()
        else {
            // register_tool_function is only ever called with
            // ToolCall. ModelCall takes a different path.
            return;
        };

        let Some(disposition) = self.tool_disposition(thread_id, &name) else {
            // Unknown thread — defensive; upstream route_tool checks
            // will surface the error.
            let fut = crate::runtime::io_dispatch::build_io_future(
                self,
                thread_id.to_string(),
                io_request,
            );
            pending_io.push(fut);
            return;
        };

        // `dispatch_thread` is a model-facing alias for
        // `Function::CreateThread`. Deny stops at the tool layer;
        // Allow proceeds with the CreateThread spec.
        if name == crate::tools::builtin_tools::DISPATCH_THREAD {
            self.register_dispatch_thread_tool(
                thread_id,
                op_id,
                tool_use_id,
                input,
                disposition,
                pending_io,
            );
            return;
        }

        // `sudo` is intercepted at the scheduler layer — register a
        // Function::Sudo that emits the approval request on launch and
        // parks the tool call on its terminal. Approval dispatches the
        // wrapped tool; rejection returns the user's reason.
        if name == crate::tools::builtin_tools::SUDO {
            self.register_sudo_tool(
                thread_id,
                op_id,
                tool_use_id,
                input,
                disposition,
                pending_io,
            );
            return;
        }

        // `describe_tool` and `find_tool` are synchronous — no
        // Function, no async waiting. We compute the result from the
        // scheduler's tool registry and push a pre-resolved future
        // that fires with the result on the next scheduler iteration.
        // The disposition gate still applies: the user can deny
        // these via `scope.tools` if they want to disable introspection.
        if name == crate::tools::builtin_tools::DESCRIBE_TOOL {
            self.complete_describe_tool_call(
                thread_id,
                op_id,
                tool_use_id,
                input,
                disposition,
                pending_io,
            );
            return;
        }
        if name == crate::tools::builtin_tools::FIND_TOOL {
            self.complete_find_tool_call(
                thread_id,
                op_id,
                tool_use_id,
                input,
                disposition,
                pending_io,
            );
            return;
        }
        if name == crate::tools::builtin_tools::LIST_LLM_PROVIDERS {
            self.complete_list_llm_providers_call(
                thread_id,
                op_id,
                tool_use_id,
                input,
                disposition,
                pending_io,
            );
            return;
        }
        if name == crate::tools::builtin_tools::LIST_MCP_HOSTS {
            self.complete_list_mcp_hosts_call(
                thread_id,
                op_id,
                tool_use_id,
                input,
                disposition,
                pending_io,
            );
            return;
        }
        if name == crate::tools::builtin_tools::LIST_HOST_ENV_PROVIDERS {
            self.complete_list_host_env_providers_call(
                thread_id,
                op_id,
                tool_use_id,
                input,
                disposition,
                pending_io,
            );
            return;
        }
        if name == crate::tools::builtin_tools::KNOWLEDGE_QUERY {
            self.complete_knowledge_query_call(
                thread_id,
                op_id,
                tool_use_id,
                input,
                disposition,
                pending_io,
            );
            return;
        }
        if name == crate::tools::builtin_tools::KNOWLEDGE_MODIFY {
            self.complete_knowledge_modify_call(
                thread_id,
                op_id,
                tool_use_id,
                input,
                disposition,
                pending_io,
            );
            return;
        }

        let spec = match self.route_tool(thread_id, &name) {
            Some(ToolRoute::Builtin { .. }) => Function::BuiltinToolCall {
                name: name.clone(),
                args: input.clone(),
            },
            Some(ToolRoute::Mcp { host, .. }) => Function::McpToolUse {
                host: host.clone(),
                name: name.clone(),
                args: input.clone(),
            },
            None => {
                // No route — the thread's turn will surface this as a
                // tool-not-found error when the IO future runs. Let it
                // through.
                let fut = crate::runtime::io_dispatch::build_io_future(
                    self,
                    thread_id.to_string(),
                    io_request,
                );
                pending_io.push(fut);
                return;
            }
        };

        let caller = CallerLink::ThreadToolCall {
            thread_id: thread_id.to_string(),
            tool_use_id: tool_use_id.clone(),
        };

        match disposition {
            crate::permission::Disposition::Allow => {
                let _ = self.register_function(spec, caller);
                let fut = crate::runtime::io_dispatch::build_io_future(
                    self,
                    thread_id.to_string(),
                    io_request,
                );
                pending_io.push(fut);
            }
            crate::permission::Disposition::Deny => {
                // Reject — don't register the Function at all.
                // Synthesize a ToolCall completion future that fires
                // with an IoResult::Err so the thread integrates the
                // denial as a tool_result and continues its turn.
                pending_io.push(make_denial_future(
                    thread_id.to_string(),
                    tool_use_id,
                    op_id,
                    name,
                ));
            }
        }
    }

    /// `dispatch_thread`-specific registration path. Parses the tool
    /// args into a `Function::CreateThread` spec with
    /// `wait_mode: ThreadTerminal`, picks a `FunctionDelivery` based
    /// on `args.sync`, registers + launches the Function, and pushes
    /// the parent-facing tool-call future (oneshot-awaiting for sync
    /// or immediate ack for async).
    #[allow(clippy::too_many_arguments)]
    fn register_dispatch_thread_tool(
        &mut self,
        parent_thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        disposition: crate::permission::Disposition,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if matches!(disposition, crate::permission::Disposition::Deny) {
            pending_io.push(make_denial_future(
                parent_thread_id.to_string(),
                tool_use_id,
                op_id,
                crate::tools::builtin_tools::DISPATCH_THREAD.to_string(),
            ));
            return;
        }
        // DispatchCap gate: a thread whose scope admits no dispatch
        // can't spawn children regardless of what `[allow.tools]`
        // says about the tool name.
        let dispatch_admitted = self
            .tasks
            .get(parent_thread_id)
            .map(|t| t.scope.dispatch)
            .unwrap_or(crate::permission::DispatchCap::None)
            != crate::permission::DispatchCap::None;
        if !dispatch_admitted {
            pending_io.push(immediate_tool_error(
                parent_thread_id.to_string(),
                op_id,
                tool_use_id,
                "dispatch_thread denied: thread's scope does not admit dispatching \
                 (DispatchCap::None). Ask for scope widening if this is needed."
                    .into(),
            ));
            return;
        }

        let args = match crate::tools::builtin_tools::dispatch_thread::parse_args(input) {
            Ok(a) => a,
            Err(e) => {
                pending_io.push(immediate_tool_error(
                    parent_thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    e,
                ));
                return;
            }
        };

        // Parent's pod is the target; ParentLink carries the lineage.
        let pod_id = self.tasks.get(parent_thread_id).map(|t| t.pod_id.clone());
        let spec = Function::CreateThread {
            pod_id,
            initial_message: Some(args.prompt.clone()),
            initial_attachments: Vec::new(),
            parent: Some(crate::functions::ParentLink {
                thread_id: parent_thread_id.to_string(),
                tool_use_id: tool_use_id.clone(),
            }),
            wait_mode: crate::functions::WaitMode::ThreadTerminal,
            config_override: args.config_override,
            bindings_request: args.bindings_override,
        };
        let caller = CallerLink::ThreadToolCall {
            thread_id: parent_thread_id.to_string(),
            tool_use_id: tool_use_id.clone(),
        };

        if args.sync {
            let (tx, rx) = tokio::sync::oneshot::channel::<Result<String, String>>();
            let delivery = FunctionDelivery::ToolResultChannel(tx);
            let fn_id = match self.register_function_with_delivery(spec, caller, delivery) {
                Ok(id) => id,
                Err(e) => {
                    pending_io.push(immediate_tool_error(
                        parent_thread_id.to_string(),
                        op_id,
                        tool_use_id,
                        format!(
                            "dispatch_thread: {}",
                            super::client_messages::reject_reason_detail(&e)
                        ),
                    ));
                    return;
                }
            };
            self.launch_function(fn_id, pending_io);
            let parent_id = parent_thread_id.to_string();
            pending_io.push(Box::pin(async move {
                let result = match rx.await {
                    Ok(Ok(text)) => Ok(crate::tools::mcp::CallToolResult {
                        content: vec![crate::tools::mcp::McpContentBlock::Text { text }],
                        is_error: false,
                    }),
                    Ok(Err(msg)) => Err(msg),
                    Err(_) => Err(
                        "dispatch_thread: Function terminated without delivering a result"
                            .to_string(),
                    ),
                };
                crate::runtime::io_dispatch::SchedulerCompletion::Io(
                    crate::runtime::io_dispatch::IoCompletion {
                        thread_id: parent_id,
                        op_id,
                        result: crate::runtime::thread::IoResult::ToolCall {
                            tool_use_id,
                            result,
                        },
                        pod_update: None,
                        scheduler_command: None,
                        host_env_lost: None,
                    },
                )
            }));
        } else {
            let delivery = FunctionDelivery::ToolResultFollowup {
                parent_thread_id: parent_thread_id.to_string(),
                parent_tool_use_id: tool_use_id.clone(),
            };
            let fn_id = match self.register_function_with_delivery(spec, caller, delivery) {
                Ok(id) => id,
                Err(e) => {
                    pending_io.push(immediate_tool_error(
                        parent_thread_id.to_string(),
                        op_id,
                        tool_use_id,
                        format!(
                            "dispatch_thread: {}",
                            super::client_messages::reject_reason_detail(&e)
                        ),
                    ));
                    return;
                }
            };
            self.launch_function(fn_id, pending_io);
            let child_id = self
                .active_functions
                .get(&fn_id)
                .and_then(|e| e.awaiting_child_thread_id.clone())
                .unwrap_or_default();
            let ack = format!(
                "Dispatched child thread `{child_id}` in the background. Its final result will \
                 be delivered as a fresh user message once the child terminates; you can \
                 continue working in the meantime."
            );
            let parent_id = parent_thread_id.to_string();
            pending_io.push(Box::pin(async move {
                crate::runtime::io_dispatch::SchedulerCompletion::Io(
                    crate::runtime::io_dispatch::IoCompletion {
                        thread_id: parent_id,
                        op_id,
                        result: crate::runtime::thread::IoResult::ToolCall {
                            tool_use_id,
                            result: Ok(crate::tools::mcp::CallToolResult {
                                content: vec![crate::tools::mcp::McpContentBlock::Text {
                                    text: ack,
                                }],
                                is_error: false,
                            }),
                        },
                        pod_update: None,
                        scheduler_command: None,
                        host_env_lost: None,
                    },
                )
            }));
        }
    }

    /// `describe_tool`-specific synchronous path. No Function is
    /// registered — the result is computed from the scheduler's tool
    /// registry and delivered on the next tick via a pre-resolved
    /// future. Respects `scope.tools` disposition (Deny stops here)
    /// and admits tools in the pod ceiling so the model can read
    /// schemas of askable tools before deciding to escalate.
    fn complete_describe_tool_call(
        &mut self,
        thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        disposition: crate::permission::Disposition,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if matches!(disposition, crate::permission::Disposition::Deny) {
            pending_io.push(make_denial_future(
                thread_id.to_string(),
                tool_use_id,
                op_id,
                crate::tools::builtin_tools::DESCRIBE_TOOL.to_string(),
            ));
            return;
        }
        let args = match crate::tools::builtin_tools::describe_tool::parse_args(input) {
            Ok(a) => a,
            Err(e) => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    e,
                ));
                return;
            }
        };
        let found = self
            .admissible_and_askable_tools(thread_id)
            .into_iter()
            .find(|t| t.name == args.name);
        let text = match found {
            Some(t) => {
                let status = if t.requires_escalation {
                    "askable via sudo"
                } else {
                    "admitted"
                };
                let schema = serde_json::to_string_pretty(&t.input_schema)
                    .unwrap_or_else(|e| format!("<schema serialization failed: {e}>"));
                format!(
                    "Tool `{name}` ({status})\n\
                     Category: {cat}\n\
                     Description: {desc}\n\
                     \n\
                     Input schema:\n{schema}",
                    name = t.name,
                    cat = t.category.label(),
                    desc = t.description,
                )
            }
            None => format!(
                "tool `{}` not found in this thread's catalog (check the name — \
                 host-env tools are prefixed, e.g. `rustdev_bash`; use `find_tool` \
                 to search)",
                args.name
            ),
        };
        pending_io.push(immediate_tool_success(
            thread_id.to_string(),
            op_id,
            tool_use_id,
            text,
        ));
    }

    /// `find_tool`-specific synchronous path. Matches regex over
    /// name+description, filters by coarse category, honors
    /// `include_escalation`, paginates via `limit`+`offset`. Returns
    /// a structured text result (name/desc/cat/requires_escalation
    /// per entry) suitable for the model to parse or scan.
    fn complete_find_tool_call(
        &mut self,
        thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        disposition: crate::permission::Disposition,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if matches!(disposition, crate::permission::Disposition::Deny) {
            pending_io.push(make_denial_future(
                thread_id.to_string(),
                tool_use_id,
                op_id,
                crate::tools::builtin_tools::FIND_TOOL.to_string(),
            ));
            return;
        }
        let args = match crate::tools::builtin_tools::find_tool::parse_args(input) {
            Ok(a) => a,
            Err(e) => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    e,
                ));
                return;
            }
        };
        let regex = match regex::Regex::new(&args.pattern) {
            Ok(r) => r,
            Err(e) => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    format!("invalid regex pattern `{}`: {e}", args.pattern),
                ));
                return;
            }
        };
        let all = self.admissible_and_askable_tools(thread_id);
        let limit = args.effective_limit() as usize;
        let offset = args.effective_offset() as usize;
        let (page, total) = crate::runtime::tool_listing::filter_tools_for_find(
            &all,
            &regex,
            args.category.as_deref(),
            args.include_escalation,
            limit,
            offset,
        );
        let shown = page.len();
        let mut out = String::new();
        out.push_str(&format!(
            "Found {total} tool(s) matching `{pattern}`; showing {shown} (offset {offset}, limit {limit}).\n\n",
            pattern = args.pattern
        ));
        for t in page {
            let tag = if t.requires_escalation { " [sudo]" } else { "" };
            out.push_str(&format!(
                "- `{name}` [{cat}]{tag} — {desc}\n",
                name = t.name,
                cat = t.category.label(),
                desc = one_line(&t.description),
            ));
        }
        if total > offset + shown {
            out.push_str(&format!(
                "\n… {} more. Page with `offset={}`.\n",
                total - offset - shown,
                offset + shown
            ));
        }
        pending_io.push(immediate_tool_success(
            thread_id.to_string(),
            op_id,
            tool_use_id,
            out,
        ));
    }

    /// `list_llm_providers`-specific synchronous path. Without args,
    /// reads the scheduler's backend catalog and emits an immediate
    /// tool result. With `backend`, resolves the provider, clones its
    /// `Arc<dyn ModelProvider>`, and pushes an async future that
    /// awaits `list_models()` — the scheduler loop doesn't block on
    /// the HTTP round-trip.
    fn complete_list_llm_providers_call(
        &mut self,
        thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        disposition: crate::permission::Disposition,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if matches!(disposition, crate::permission::Disposition::Deny) {
            pending_io.push(make_denial_future(
                thread_id.to_string(),
                tool_use_id,
                op_id,
                crate::tools::builtin_tools::LIST_LLM_PROVIDERS.to_string(),
            ));
            return;
        }
        let args = match crate::tools::builtin_tools::list_llm_providers::parse_args(input) {
            Ok(a) => a,
            Err(e) => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    e,
                ));
                return;
            }
        };

        // Resolve the caller's pod so we can annotate each backend
        // with whether the pod's `[allow.backends]` admits it. Out-of-
        // scope backends are still listed — the agent may want to
        // know what exists at the server level even if adding one to
        // `pod.toml` requires `pod_modify`.
        let in_scope: std::collections::HashSet<String> = self
            .tasks
            .get(thread_id)
            .and_then(|t| self.pods.get(&t.pod_id))
            .map(|pod| pod.config.allow.backends.iter().cloned().collect())
            .unwrap_or_default();

        // Always prepare the backend summary header — it's the cheap
        // part and callers expanding one backend still want the
        // roster for context.
        let mut backend_lines: Vec<(String, String)> = self
            .backends
            .iter()
            .map(|(name, entry)| {
                let default_model = entry.default_model.as_deref().unwrap_or("—");
                let auth = entry.auth_mode.as_deref().unwrap_or("none");
                let scope = if in_scope.contains(name) {
                    "in scope"
                } else {
                    "not in scope"
                };
                (
                    name.clone(),
                    format!(
                        "- `{name}` [{scope}] — kind={kind}, default_model={dm}, auth={auth}",
                        kind = entry.kind,
                        dm = default_model,
                    ),
                )
            })
            .collect();
        backend_lines.sort_by(|a, b| a.0.cmp(&b.0));
        let mut header = String::new();
        header.push_str(
            "Configured LLM backends (in scope = admitted by this pod's \
             `[allow.backends]`; out-of-scope entries require pod_modify to add):\n",
        );
        for (_, line) in &backend_lines {
            header.push_str(line);
            header.push('\n');
        }

        let Some(backend_name) = args.backend else {
            // No expansion requested — cheap path, synchronous.
            pending_io.push(immediate_tool_success(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                header,
            ));
            return;
        };

        // Expand models for the named backend. Clone the provider Arc
        // out so the future can own it; surface an immediate error
        // when the name doesn't resolve — and include the roster in
        // the error so the agent doesn't have to make a second call
        // to see the valid names.
        let Some(entry) = self.backends.get(&backend_name) else {
            pending_io.push(immediate_tool_error(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                format!("unknown backend `{backend_name}`.\n\n{header}"),
            ));
            return;
        };
        let provider = entry.provider.clone();
        let thread_id_s = thread_id.to_string();
        pending_io.push(Box::pin(async move {
            let body = match provider.list_models().await {
                Ok(models) => {
                    let mut s = header;
                    s.push('\n');
                    s.push_str(&format!(
                        "Models on `{backend_name}` ({count}):\n",
                        count = models.len()
                    ));
                    if models.is_empty() {
                        s.push_str(
                            "  (provider returned no models — this is typical for \
                             single-model local endpoints; use an empty model id)\n",
                        );
                    } else {
                        for m in models {
                            let head = match m.display_name {
                                Some(ref dn) if !dn.is_empty() && dn != &m.id => {
                                    format!("- `{}` — {}", m.id, dn)
                                }
                                _ => format!("- `{}`", m.id),
                            };
                            let caps = match (m.context_window, m.max_output_tokens) {
                                (Some(ctx), Some(out)) => {
                                    format!(" (ctx {ctx}, max_out {out})")
                                }
                                (Some(ctx), None) => format!(" (ctx {ctx})"),
                                (None, Some(out)) => format!(" (max_out {out})"),
                                (None, None) => String::new(),
                            };
                            s.push_str(&head);
                            s.push_str(&caps);
                            s.push('\n');
                        }
                    }
                    Ok(s)
                }
                Err(e) => Err(format!("list_models(`{backend_name}`) failed: {e}")),
            };
            let result = match body {
                Ok(text) => crate::runtime::thread::IoResult::ToolCall {
                    tool_use_id,
                    result: Ok(crate::tools::mcp::CallToolResult {
                        content: vec![crate::tools::mcp::McpContentBlock::Text { text }],
                        is_error: false,
                    }),
                },
                Err(message) => crate::runtime::thread::IoResult::ToolCall {
                    tool_use_id,
                    result: Err(message),
                },
            };
            crate::runtime::io_dispatch::SchedulerCompletion::Io(
                crate::runtime::io_dispatch::IoCompletion {
                    thread_id: thread_id_s,
                    op_id,
                    result,
                    pod_update: None,
                    scheduler_command: None,
                    host_env_lost: None,
                },
            )
        }));
    }

    /// `list_mcp_hosts`-specific synchronous path. Reads the
    /// scheduler's shared-MCP catalog snapshot (no tokens, public
    /// auth classification only) and annotates each entry with
    /// whether the current pod's `[allow.mcp_hosts]` admits it.
    fn complete_list_mcp_hosts_call(
        &mut self,
        thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        disposition: crate::permission::Disposition,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if matches!(disposition, crate::permission::Disposition::Deny) {
            pending_io.push(make_denial_future(
                thread_id.to_string(),
                tool_use_id,
                op_id,
                crate::tools::builtin_tools::LIST_MCP_HOSTS.to_string(),
            ));
            return;
        }
        if let Err(e) = crate::tools::builtin_tools::list_mcp_hosts::parse_args(input) {
            pending_io.push(immediate_tool_error(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                e,
            ));
            return;
        }
        let in_scope: std::collections::HashSet<String> = self
            .tasks
            .get(thread_id)
            .and_then(|t| self.pods.get(&t.pod_id))
            .map(|pod| pod.config.allow.mcp_hosts.iter().cloned().collect())
            .unwrap_or_default();

        let hosts = self.shared_mcp_hosts_snapshot();
        let mut out = String::new();
        out.push_str(
            "Shared MCP hosts (in scope = listed in this pod's `[allow.mcp_hosts]`; \
             out-of-scope entries require pod_modify to add):\n",
        );
        if hosts.is_empty() {
            out.push_str("  (no shared MCP hosts configured on this server)\n");
        } else {
            for h in hosts {
                let scope = if in_scope.contains(&h.name) {
                    "in scope"
                } else {
                    "not in scope"
                };
                let auth = match &h.auth {
                    whisper_agent_protocol::SharedMcpAuthPublic::None => "none".to_string(),
                    whisper_agent_protocol::SharedMcpAuthPublic::Bearer => "bearer".to_string(),
                    whisper_agent_protocol::SharedMcpAuthPublic::Oauth2 { issuer, .. } => {
                        format!("oauth2 (issuer: {issuer})")
                    }
                };
                let connected = if h.connected {
                    "connected"
                } else {
                    "disconnected"
                };
                let origin = match h.origin {
                    whisper_agent_protocol::HostEnvProviderOrigin::Seeded => "seeded",
                    whisper_agent_protocol::HostEnvProviderOrigin::Manual => "manual",
                    whisper_agent_protocol::HostEnvProviderOrigin::RuntimeOverlay => {
                        "runtime_overlay"
                    }
                };
                out.push_str(&format!(
                    "- `{name}` [{scope}] — url={url}, auth={auth}, {connected}, origin={origin}\n",
                    name = h.name,
                    url = h.url,
                ));
            }
        }
        pending_io.push(immediate_tool_success(
            thread_id.to_string(),
            op_id,
            tool_use_id,
            out,
        ));
    }

    /// `list_host_env_providers`-specific synchronous path. Reads the
    /// scheduler's host-env catalog snapshot (no control-plane tokens,
    /// only a `has_token` presence flag) and annotates each entry with
    /// whether any of the current pod's `[[allow.host_env]]` entries
    /// reference the provider.
    fn complete_list_host_env_providers_call(
        &mut self,
        thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        disposition: crate::permission::Disposition,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if matches!(disposition, crate::permission::Disposition::Deny) {
            pending_io.push(make_denial_future(
                thread_id.to_string(),
                tool_use_id,
                op_id,
                crate::tools::builtin_tools::LIST_HOST_ENV_PROVIDERS.to_string(),
            ));
            return;
        }
        if let Err(e) = crate::tools::builtin_tools::list_host_env_providers::parse_args(input) {
            pending_io.push(immediate_tool_error(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                e,
            ));
            return;
        }
        let in_scope: std::collections::HashSet<String> = self
            .tasks
            .get(thread_id)
            .and_then(|t| self.pods.get(&t.pod_id))
            .map(|pod| {
                pod.config
                    .allow
                    .host_env
                    .iter()
                    .map(|e| e.provider.clone())
                    .collect()
            })
            .unwrap_or_default();

        let providers = self.host_env_provider_snapshot();
        let mut out = String::new();
        out.push_str(
            "Host-env (sandbox) providers (in scope = referenced by at least one \
             `[[allow.host_env]]` entry in this pod; out-of-scope entries require \
             pod_modify to add):\n",
        );
        if providers.is_empty() {
            out.push_str("  (no host-env providers registered on this server)\n");
        } else {
            for p in providers {
                let scope = if in_scope.contains(&p.name) {
                    "in scope"
                } else {
                    "not in scope"
                };
                let token = if p.has_token {
                    "authenticated"
                } else {
                    "anonymous"
                };
                let reach = match &p.reachability {
                    whisper_agent_protocol::HostEnvReachability::Unknown => "unknown".to_string(),
                    whisper_agent_protocol::HostEnvReachability::Reachable { .. } => {
                        "reachable".to_string()
                    }
                    whisper_agent_protocol::HostEnvReachability::Unreachable {
                        last_error, ..
                    } => format!("unreachable ({last_error})"),
                };
                let origin = match p.origin {
                    whisper_agent_protocol::HostEnvProviderOrigin::Seeded => "seeded",
                    whisper_agent_protocol::HostEnvProviderOrigin::Manual => "manual",
                    whisper_agent_protocol::HostEnvProviderOrigin::RuntimeOverlay => {
                        "runtime_overlay"
                    }
                };
                out.push_str(&format!(
                    "- `{name}` [{scope}] — url={url}, {token}, {reach}, origin={origin}\n",
                    name = p.name,
                    url = p.url,
                ));
            }
        }
        pending_io.push(immediate_tool_success(
            thread_id.to_string(),
            op_id,
            tool_use_id,
            out,
        ));
    }

    /// `knowledge_query`-specific path. Resolves the pod's
    /// `[allow.knowledge_buckets]`, intersects with the caller's
    /// optional `buckets` arg, then runs `QueryEngine` over the
    /// admitted set on a detached future so the multi-second HNSW +
    /// embedder + reranker round-trip doesn't block the scheduler.
    fn complete_knowledge_query_call(
        &mut self,
        thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        disposition: crate::permission::Disposition,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if matches!(disposition, crate::permission::Disposition::Deny) {
            pending_io.push(make_denial_future(
                thread_id.to_string(),
                tool_use_id,
                op_id,
                crate::tools::builtin_tools::KNOWLEDGE_QUERY.to_string(),
            ));
            return;
        }
        let args = match crate::tools::builtin_tools::knowledge_query::parse_args(input) {
            Ok(a) => a,
            Err(e) => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    e,
                ));
                return;
            }
        };
        let top_k = args
            .top_k
            .unwrap_or(crate::tools::builtin_tools::knowledge_query::DEFAULT_TOP_K);

        // Resolve in-scope buckets from the pod's allow list.
        let in_scope: Vec<String> = self
            .tasks
            .get(thread_id)
            .and_then(|t| self.pods.get(&t.pod_id))
            .map(|pod| pod.config.allow.knowledge_buckets.clone())
            .unwrap_or_default();

        if in_scope.is_empty() {
            pending_io.push(immediate_tool_error(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                "knowledge_query: this pod's `[allow.knowledge_buckets]` is empty. \
                 Ask the operator to add bucket ids before retry."
                    .into(),
            ));
            return;
        }

        // Intersect the explicit `buckets` arg (if any) with the in-
        // scope set. Out-of-scope names error with what IS in scope so
        // the model can correct in one round-trip.
        let target_ids: Vec<String> = if args.buckets.is_empty() {
            in_scope.clone()
        } else {
            let scope_set: std::collections::HashSet<&String> = in_scope.iter().collect();
            let bad: Vec<&String> = args
                .buckets
                .iter()
                .filter(|b| !scope_set.contains(b))
                .collect();
            if !bad.is_empty() {
                let bad_list = bad
                    .iter()
                    .map(|s| format!("`{s}`"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let scope_list = in_scope
                    .iter()
                    .map(|s| format!("`{s}`"))
                    .collect::<Vec<_>>()
                    .join(", ");
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    format!(
                        "knowledge_query: bucket(s) {bad_list} not in this pod's \
                         `[allow.knowledge_buckets]`. In scope: {scope_list}."
                    ),
                ));
                return;
            }
            args.buckets.clone()
        };

        // Resolve embedder + reranker. Different buckets may want
        // different embedders (each slot has the dimension baked into
        // its manifest); the reranker is global. We pick per-bucket
        // embedders inside the future after the buckets are loaded;
        // for the launch-time check we just need *some* embedder
        // to exist. Reranker is required.
        let reranker = match self.rerank_providers.values().next() {
            Some(r) => r.provider.clone(),
            None => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    "knowledge_query: no rerank providers configured \
                     (`[rerank_providers.*]`). Knowledge queries need a reranker."
                        .into(),
                ));
                return;
            }
        };

        // Validate each target bucket exists in the registry up-front
        // so the failure message is "unknown bucket" with the bad id,
        // not a generic load error from inside the future.
        let mut bucket_configs: Vec<(String, String)> = Vec::with_capacity(target_ids.len());
        for id in &target_ids {
            let entry = match self.bucket_registry.buckets.get(id) {
                Some(e) => e,
                None => {
                    pending_io.push(immediate_tool_error(
                        thread_id.to_string(),
                        op_id,
                        tool_use_id,
                        format!(
                            "knowledge_query: bucket `{id}` is in the pod's allow list \
                             but no longer exists in the registry — likely deleted \
                             out from under the pod config. Update `[allow.knowledge_buckets]`."
                        ),
                    ));
                    return;
                }
            };
            bucket_configs.push((id.clone(), entry.config.defaults.embedder.clone()));
        }

        // Resolve each bucket's embedder by its bucket.toml `defaults.embedder`.
        // All embedded query buckets must agree on the embedder for the
        // dense path to fuse cleanly, but mismatched dimensions only
        // skip the dense path silently (see QueryEngine). For the v1
        // tool we still pick a single embedder per call — the first
        // bucket's. Future: per-bucket dispatch with per-embedder
        // dispatch fanning out. Today this is consistent with the
        // WebUI handler's behavior.
        let primary_embedder_name = bucket_configs[0].1.clone();
        let embedder = match self.embedding_providers.get(&primary_embedder_name) {
            Some(e) => e.provider.clone(),
            None => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    format!(
                        "knowledge_query: bucket `{}` references embedder `{}` which is \
                         not configured under `[embedding_providers.*]`.",
                        bucket_configs[0].0, primary_embedder_name,
                    ),
                ));
                return;
            }
        };

        let registry = self.bucket_registry.clone();
        let thread_id_s = thread_id.to_string();
        let query_text = args.query.clone();
        pending_io.push(Box::pin(async move {
            let cancel = tokio_util::sync::CancellationToken::new();

            // Load each bucket. First-load pays the slot-load cost;
            // hot loads come from the cache.
            let mut buckets: Vec<std::sync::Arc<dyn crate::knowledge::Bucket>> =
                Vec::with_capacity(target_ids.len());
            for id in &target_ids {
                match registry.loaded_bucket(id).await {
                    Ok(b) => buckets.push(b),
                    Err(e) => {
                        return tool_error_completion(
                            thread_id_s,
                            op_id,
                            tool_use_id,
                            format!("knowledge_query: load bucket `{id}` failed: {e}"),
                        );
                    }
                }
            }

            let engine = crate::knowledge::QueryEngine::new(embedder, reranker);
            let params = crate::knowledge::QueryParams {
                top_k: top_k as usize,
                ..Default::default()
            };
            let hits = match engine.query(&buckets, &query_text, &params, &cancel).await {
                Ok(h) => h,
                Err(e) => {
                    return tool_error_completion(
                        thread_id_s,
                        op_id,
                        tool_use_id,
                        format!("knowledge_query failed: {e}"),
                    );
                }
            };

            let body = format_hits(&query_text, &target_ids, &hits);
            tool_success_completion(thread_id_s, op_id, tool_use_id, body)
        }));
    }

    /// `knowledge_modify`-specific path. Same pod-scope gate as
    /// `complete_knowledge_query_call`; the bucket's source-kind
    /// check happens inside [`DiskBucket::insert_record`] /
    /// [`DiskBucket::tombstone_by_source`] (only `managed` buckets
    /// accept LLM-driven mutations). Dispatches to a detached future
    /// because the embed + index work can take seconds for large
    /// `content` payloads, and the scheduler thread should not pin
    /// on it.
    fn complete_knowledge_modify_call(
        &mut self,
        thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        disposition: crate::permission::Disposition,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if matches!(disposition, crate::permission::Disposition::Deny) {
            pending_io.push(make_denial_future(
                thread_id.to_string(),
                tool_use_id,
                op_id,
                crate::tools::builtin_tools::KNOWLEDGE_MODIFY.to_string(),
            ));
            return;
        }
        let args = match crate::tools::builtin_tools::knowledge_modify::parse_args(input) {
            Ok(a) => a,
            Err(e) => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    e,
                ));
                return;
            }
        };

        // Same pod-scope check as knowledge_query — the bucket id
        // must be in `[allow.knowledge_buckets]`. The source-kind
        // check (managed-only) happens inside the bucket method,
        // not here, so that error message names the actual kind.
        let in_scope: Vec<String> = self
            .tasks
            .get(thread_id)
            .and_then(|t| self.pods.get(&t.pod_id))
            .map(|pod| pod.config.allow.knowledge_buckets.clone())
            .unwrap_or_default();
        if !in_scope.iter().any(|b| b == &args.bucket_id) {
            let scope_list = if in_scope.is_empty() {
                "(empty — ask the operator to populate `[allow.knowledge_buckets]`)".to_string()
            } else {
                in_scope
                    .iter()
                    .map(|s| format!("`{s}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            pending_io.push(immediate_tool_error(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                format!(
                    "knowledge_modify: bucket `{}` not in this pod's \
                     `[allow.knowledge_buckets]`. In scope: {scope_list}.",
                    args.bucket_id,
                ),
            ));
            return;
        }

        // Validate bucket exists in the registry up-front for a
        // clearer error than the future's load failure.
        let entry = match self.bucket_registry.buckets.get(&args.bucket_id) {
            Some(e) => e.clone(),
            None => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    format!(
                        "knowledge_modify: bucket `{}` is in the pod's allow list but no \
                         longer exists in the registry — likely deleted. Update \
                         `[allow.knowledge_buckets]`.",
                        args.bucket_id,
                    ),
                ));
                return;
            }
        };

        // Insert needs a live embedder; tombstone doesn't. Resolve
        // the embedder up-front for both paths to keep the failure
        // mode simple ("missing embedder" surfaces at args time, not
        // mid-future), even though the tombstone path won't call it.
        let embedder = match self
            .embedding_providers
            .get(&entry.config.defaults.embedder)
        {
            Some(p) => p.provider.clone(),
            None => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    format!(
                        "knowledge_modify: bucket `{}` references embedder `{}` which is \
                         not configured under `[embedding_providers.*]`.",
                        args.bucket_id, entry.config.defaults.embedder,
                    ),
                ));
                return;
            }
        };

        let registry = self.bucket_registry.clone();
        let thread_id_s = thread_id.to_string();
        let bucket_id = args.bucket_id.clone();
        let source_id = args.source_id.clone();
        let action = args.action.clone();
        let content = args.content.unwrap_or_default();
        let embedder_name = entry.config.defaults.embedder.clone();
        pending_io.push(Box::pin(async move {
            let cancel = tokio_util::sync::CancellationToken::new();
            let bucket = match registry.loaded_bucket(&bucket_id).await {
                Ok(b) => b,
                Err(e) => {
                    return tool_error_completion(
                        thread_id_s,
                        op_id,
                        tool_use_id,
                        format!("knowledge_modify: load bucket `{bucket_id}` failed: {e}"),
                    );
                }
            };
            // The bucket may have a stale embedder from a prior
            // `set_embedder` call (or none if just opened). Re-bind
            // each time — `set_embedder` short-circuits when the
            // model id matches, so this is cheap on the hot path.
            if let Err(e) = bucket.set_embedder(embedder, &embedder_name) {
                return tool_error_completion(
                    thread_id_s,
                    op_id,
                    tool_use_id,
                    format!("knowledge_modify: set_embedder on `{bucket_id}` failed: {e}"),
                );
            }
            match action {
                crate::tools::builtin_tools::knowledge_modify::KnowledgeModifyAction::Insert => {
                    match bucket.insert_record(&source_id, &content, &cancel).await {
                        Ok(n) => tool_success_completion(
                            thread_id_s,
                            op_id,
                            tool_use_id,
                            format!(
                                "Inserted `{source_id}` into `{bucket_id}` ({n} chunk{}).",
                                if n == 1 { "" } else { "s" },
                            ),
                        ),
                        Err(e) => tool_error_completion(
                            thread_id_s,
                            op_id,
                            tool_use_id,
                            format!("knowledge_modify insert failed: {e}"),
                        ),
                    }
                }
                crate::tools::builtin_tools::knowledge_modify::KnowledgeModifyAction::Tombstone => {
                    match bucket.tombstone_by_source(&source_id).await {
                        Ok(0) => tool_success_completion(
                            thread_id_s,
                            op_id,
                            tool_use_id,
                            format!(
                                "Tombstone `{source_id}` in `{bucket_id}`: no-op \
                                 (source not found, already absent).",
                            ),
                        ),
                        Ok(n) => tool_success_completion(
                            thread_id_s,
                            op_id,
                            tool_use_id,
                            format!(
                                "Tombstoned `{source_id}` in `{bucket_id}` ({n} chunk{}).",
                                if n == 1 { "" } else { "s" },
                            ),
                        ),
                        Err(e) => tool_error_completion(
                            thread_id_s,
                            op_id,
                            tool_use_id,
                            format!("knowledge_modify tombstone failed: {e}"),
                        ),
                    }
                }
            }
        }));
    }

    /// `sudo`-specific registration path. Parses the tool args, refuses
    /// at the tool layer if the thread's scope lacks an interactive
    /// approver, otherwise registers a `Function::Sudo` (which fires
    /// the `SudoRequested` event on launch) and parks the tool call on
    /// its terminal via `FunctionDelivery::ToolResultChannel`. The
    /// parent's turn resumes when the user resolves the sudo.
    fn register_sudo_tool(
        &mut self,
        thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        disposition: crate::permission::Disposition,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if matches!(disposition, crate::permission::Disposition::Deny) {
            pending_io.push(make_denial_future(
                thread_id.to_string(),
                tool_use_id,
                op_id,
                crate::tools::builtin_tools::SUDO.to_string(),
            ));
            return;
        }
        let interactive = self
            .tasks
            .get(thread_id)
            .map(|t| t.scope.escalation.is_interactive())
            .unwrap_or(false);
        if !interactive {
            pending_io.push(immediate_tool_error(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                "sudo denied: thread has no interactive approver. Autonomous \
                 threads cannot run sudo — use the tool directly if admitted \
                 by scope, or have a parent widen this thread's scope."
                    .into(),
            ));
            return;
        }
        let args = match crate::tools::builtin_tools::sudo::parse_args(input) {
            Ok(a) => a,
            Err(e) => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    e,
                ));
                return;
            }
        };
        // Reject recursive / meta targets at register time so the user
        // never sees an approval prompt for them.
        let disallowed_targets = [
            crate::tools::builtin_tools::SUDO,
            crate::tools::builtin_tools::DISPATCH_THREAD,
        ];
        if disallowed_targets.contains(&args.tool_name.as_str()) {
            pending_io.push(immediate_tool_error(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                format!("sudo cannot wrap `{}`. Call it directly.", args.tool_name),
            ));
            return;
        }
        // The wrapped tool must at least be routable in this thread's
        // pod (builtin or bound MCP) — don't prompt the user for a
        // name we can't resolve.
        if self.route_tool(thread_id, &args.tool_name).is_none() {
            pending_io.push(immediate_tool_error(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                format!(
                    "sudo: no bound host advertises `{}`. Check `find_tool` \
                     for the exact name.",
                    args.tool_name
                ),
            ));
            return;
        }
        // Behavior-authoring gate runs BEFORE we prompt the user. A
        // sudo'd write of `behaviors/<id>/behavior.toml` whose fire
        // scope exceeds the pod ceiling (under `author_narrower`)
        // would fail anyway after approval — refuse up front so the
        // approver isn't bothered with an impossible request.
        if let Some(denial) =
            self.check_behavior_authoring_for_sudo(thread_id, &args.tool_name, &args.args)
        {
            pending_io.push(immediate_tool_error(
                thread_id.to_string(),
                op_id,
                tool_use_id,
                format!("sudo: {denial}"),
            ));
            return;
        }
        let spec = Function::Sudo {
            thread_id: thread_id.to_string(),
            tool_name: args.tool_name,
            args: args.args,
            reason: args.reason,
        };
        let caller = CallerLink::ThreadToolCall {
            thread_id: thread_id.to_string(),
            tool_use_id: tool_use_id.clone(),
        };
        let (tx, rx) = tokio::sync::oneshot::channel::<Result<String, String>>();
        let delivery = FunctionDelivery::ToolResultChannel(tx);
        let fn_id = match self.register_function_with_delivery(spec, caller, delivery) {
            Ok(id) => id,
            Err(e) => {
                pending_io.push(immediate_tool_error(
                    thread_id.to_string(),
                    op_id,
                    tool_use_id,
                    format!("sudo: {}", super::client_messages::reject_reason_detail(&e)),
                ));
                return;
            }
        };
        self.launch_function(fn_id, pending_io);
        let parent_id = thread_id.to_string();
        pending_io.push(Box::pin(async move {
            let result = match rx.await {
                Ok(Ok(text)) => Ok(crate::tools::mcp::CallToolResult {
                    content: vec![crate::tools::mcp::McpContentBlock::Text { text }],
                    is_error: false,
                }),
                Ok(Err(msg)) => Ok(crate::tools::mcp::CallToolResult {
                    content: vec![crate::tools::mcp::McpContentBlock::Text { text: msg }],
                    is_error: true,
                }),
                Err(_) => Err("sudo: Function terminated without delivering a result".into()),
            };
            crate::runtime::io_dispatch::SchedulerCompletion::Io(
                crate::runtime::io_dispatch::IoCompletion {
                    thread_id: parent_id,
                    op_id,
                    result: crate::runtime::thread::IoResult::ToolCall {
                        tool_use_id,
                        result,
                    },
                    pod_update: None,
                    scheduler_command: None,
                    host_env_lost: None,
                },
            )
        }));
    }

    /// Resolve an in-flight `Sudo` Function. On approve (once or
    /// remember), runs the wrapped tool with pod-ceiling caps and
    /// sends the inner result back through the parked tool-call
    /// oneshot. `ApproveRemember` additionally widens `scope.tools`
    /// so future direct calls of the wrapped tool skip the prompt.
    /// On reject, surfaces the user's reason through the tool result.
    pub(crate) fn resolve_sudo(
        &mut self,
        resolver_conn: ConnId,
        function_id: crate::functions::FunctionId,
        decision: crate::permission::SudoDecision,
        reason: Option<String>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let entry = match self.active_functions.get(&function_id) {
            Some(e) => e,
            None => {
                warn!(function_id, resolver_conn, "resolve_sudo: no such Function");
                self.router.send_to_client(
                    resolver_conn,
                    ServerToClient::Error {
                        correlation_id: None,
                        thread_id: None,
                        message: format!("resolve_sudo: unknown function_id {function_id}"),
                    },
                );
                return;
            }
        };
        let Function::Sudo {
            thread_id,
            tool_name,
            args,
            reason: sudo_reason,
        } = entry.spec.clone()
        else {
            self.router.send_to_client(
                resolver_conn,
                ServerToClient::Error {
                    correlation_id: None,
                    thread_id: entry.spec.primary_thread_id().map(str::to_string),
                    message: format!("resolve_sudo: function {function_id} is not a Sudo"),
                },
            );
            return;
        };
        let channel_conn = self
            .tasks
            .get(&thread_id)
            .and_then(|t| match t.scope.escalation {
                crate::permission::Escalation::Interactive { via_conn } => Some(via_conn),
                crate::permission::Escalation::None => None,
            });
        if channel_conn != Some(resolver_conn) {
            self.router.send_to_client(
                resolver_conn,
                ServerToClient::Error {
                    correlation_id: None,
                    thread_id: Some(thread_id.clone()),
                    message: "resolve_sudo: caller is not the thread's interactive channel"
                        .to_string(),
                },
            );
            return;
        }
        match decision {
            crate::permission::SudoDecision::Reject => {
                let denial_text = match &reason {
                    Some(r) if !r.trim().is_empty() => format!("sudo rejected: {r}"),
                    _ => "sudo rejected by user.".to_string(),
                };
                self.router.write_sudo_audit(
                    &thread_id,
                    &tool_name,
                    args.clone(),
                    sudo_reason,
                    crate::runtime::audit::SudoDecisionTag::Reject,
                    reason.clone(),
                    None,
                );
                self.router
                    .broadcast_task_list(ServerToClient::SudoResolved {
                        function_id,
                        thread_id: thread_id.clone(),
                        decision,
                    });
                self.complete_function(
                    function_id,
                    FunctionOutcome::Error(crate::functions::FunctionError {
                        kind: crate::functions::FunctionErrorKind::Execution,
                        detail: denial_text,
                    }),
                    pending_io,
                );
            }
            crate::permission::SudoDecision::ApproveOnce
            | crate::permission::SudoDecision::ApproveRemember => {
                if matches!(decision, crate::permission::SudoDecision::ApproveRemember)
                    && let Some(task) = self.tasks.get_mut(&thread_id)
                {
                    task.scope.tools.set_allow(tool_name.clone());
                    let snapshot = task.snapshot();
                    self.mark_dirty(&thread_id);
                    self.router.broadcast_to_subscribers(
                        &thread_id,
                        ServerToClient::ThreadSnapshot {
                            thread_id: thread_id.clone(),
                            snapshot,
                        },
                    );
                }
                self.router
                    .broadcast_task_list(ServerToClient::SudoResolved {
                        function_id,
                        thread_id: thread_id.clone(),
                        decision,
                    });
                // Dispatch the wrapped tool with pod-ceiling caps. The
                // future emits SchedulerCompletion::SudoInner, routed to
                // `apply_sudo_inner_completion`, which completes the
                // Function with the inner tool's result.
                let inner = self.build_sudo_inner_future(
                    function_id,
                    thread_id.clone(),
                    decision,
                    tool_name,
                    args,
                );
                pending_io.push(inner);
            }
        }
    }

    /// Build the future that runs a sudo'd inner tool call with
    /// pod-ceiling caps, emitting a `SchedulerCompletion::SudoInner`
    /// that the scheduler loop converts into a Function completion.
    fn build_sudo_inner_future(
        &self,
        function_id: crate::functions::FunctionId,
        thread_id: String,
        decision: crate::permission::SudoDecision,
        tool_name: String,
        args: serde_json::Value,
    ) -> SchedulerFuture {
        use crate::runtime::io_dispatch::{SchedulerCompletion, SudoInnerCompletion};
        let route = self.route_tool(&thread_id, &tool_name);
        let pod_id = self.tasks.get(&thread_id).map(|t| t.pod_id.clone());
        let cancel = self.cancel_token_or_default(&thread_id);
        let (pod_modify_ceiling, behaviors_ceiling) = match pod_id.as_deref() {
            Some(id) => match self.pods.get(id) {
                Some(pod) => (
                    pod.config.allow.caps.pod_modify,
                    pod.config.allow.caps.behaviors,
                ),
                None => (
                    crate::permission::PodModifyCap::None,
                    crate::permission::BehaviorOpsCap::None,
                ),
            },
            None => (
                crate::permission::PodModifyCap::None,
                crate::permission::BehaviorOpsCap::None,
            ),
        };
        match route {
            Some(crate::runtime::scheduler::ToolRoute::Builtin { pod_id }) => {
                let snapshot = self.pod_snapshot(&pod_id);
                Box::pin(async move {
                    let snapshot = match snapshot {
                        Some(s) => s,
                        None => {
                            return SchedulerCompletion::SudoInner(SudoInnerCompletion {
                                function_id,
                                thread_id,
                                decision,
                                result: Err(format!("sudo: pod `{pod_id}` vanished")),
                                pod_update: None,
                                scheduler_command: None,
                            });
                        }
                    };
                    let dispatch_fut = crate::tools::builtin_tools::dispatch(
                        snapshot.pod_dir,
                        snapshot.config,
                        snapshot.behavior_ids,
                        pod_modify_ceiling,
                        behaviors_ceiling,
                        &tool_name,
                        args,
                    );
                    let outcome = tokio::select! {
                        o = dispatch_fut => o,
                        _ = cancel.cancelled() => {
                            return SchedulerCompletion::SudoInner(SudoInnerCompletion {
                                function_id,
                                thread_id,
                                decision,
                                result: Err("cancelled".into()),
                                pod_update: None,
                                scheduler_command: None,
                            });
                        }
                    };
                    SchedulerCompletion::SudoInner(SudoInnerCompletion {
                        function_id,
                        thread_id,
                        decision,
                        result: Ok(outcome.result),
                        pod_update: outcome.pod_update,
                        scheduler_command: outcome.scheduler_command,
                    })
                })
            }
            Some(crate::runtime::scheduler::ToolRoute::Mcp {
                session, real_name, ..
            }) => Box::pin(async move {
                use futures::StreamExt;
                let result = match session.invoke(&real_name, args, &cancel).await {
                    Ok(mut stream) => {
                        let mut last = None;
                        while let Some(event) = stream.next().await {
                            if let crate::tools::mcp::ToolEvent::Completed(r) = event {
                                last = Some(r);
                            }
                        }
                        match last {
                            Some(r) => Ok(r),
                            None => Err("no Completed event in tool stream".to_string()),
                        }
                    }
                    Err(e) => Err(e.to_string()),
                };
                SchedulerCompletion::SudoInner(SudoInnerCompletion {
                    function_id,
                    thread_id,
                    decision,
                    result,
                    pod_update: None,
                    scheduler_command: None,
                })
            }),
            None => Box::pin(async move {
                SchedulerCompletion::SudoInner(SudoInnerCompletion {
                    function_id,
                    thread_id,
                    decision,
                    result: Err(format!("sudo: no bound host advertises `{tool_name}`")),
                    pod_update: None,
                    scheduler_command: None,
                })
            }),
        }
    }

    /// Convert a `SchedulerCompletion::SudoInner` into the Function
    /// outcome: wrap an Ok CallToolResult as `Sudo(SudoTerminal)` with
    /// the inner text (inherited `is_error` bit drives the parent's
    /// tool_result flag via the parked oneshot); wrap an Err as a
    /// Function error so the parent's oneshot fires Err.
    pub(crate) fn apply_sudo_inner_completion(
        &mut self,
        done: crate::runtime::io_dispatch::SudoInnerCompletion,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        use crate::runtime::io_dispatch::SudoInnerCompletion;
        let SudoInnerCompletion {
            function_id,
            thread_id,
            decision,
            result,
            pod_update,
            scheduler_command,
        } = done;
        // Capture audit context from the Function entry BEFORE
        // complete_function (below) removes it. For the Sudo variant
        // we know it's populated; if the entry is gone (racey
        // double-complete) we skip the audit rather than panic.
        let audit_ctx = self.active_functions.get(&function_id).and_then(|e| {
            if let Function::Sudo {
                tool_name,
                args,
                reason,
                ..
            } = &e.spec
            {
                Some((tool_name.clone(), args.clone(), reason.clone()))
            } else {
                None
            }
        });
        match result {
            Ok(call_result) => {
                // Mirror the normal tool-call path: apply side effects
                // (pod config refresh, behavior register, run/enable
                // commands) BEFORE firing the Function terminal so the
                // parent thread — and any subsequent sudo in the same
                // authoring sequence — observes the updated scheduler
                // state. Skipped when the inner tool reported an error.
                if !call_result.is_error {
                    if let Some(update) = pod_update {
                        self.apply_pod_update(&thread_id, update);
                    }
                    if let Some(command) = scheduler_command {
                        self.apply_scheduler_command(&thread_id, command, pending_io);
                    }
                }
                let text = crate::tools::mcp::mcp_blocks_text_preview(&call_result.content);
                if let Some((tool_name, args, reason)) = audit_ctx.clone() {
                    let inner_outcome = crate::server::thread_router::SudoInnerOutcome::Ok {
                        is_error: call_result.is_error,
                    };
                    self.router.write_sudo_audit(
                        &thread_id,
                        &tool_name,
                        args,
                        reason,
                        sudo_decision_tag(decision),
                        None,
                        Some(inner_outcome),
                    );
                }
                if call_result.is_error {
                    // Inner tool ran but failed — surface the error text
                    // on the parent's oneshot as Err so the model's
                    // tool_result is flagged as an error.
                    self.complete_function(
                        function_id,
                        FunctionOutcome::Error(crate::functions::FunctionError {
                            kind: crate::functions::FunctionErrorKind::Execution,
                            detail: text,
                        }),
                        pending_io,
                    );
                } else {
                    self.complete_function(
                        function_id,
                        FunctionOutcome::Success(crate::functions::FunctionTerminal::Sudo(
                            crate::functions::SudoTerminal {
                                decision,
                                tool_result_text: text,
                            },
                        )),
                        pending_io,
                    );
                }
            }
            Err(detail) => {
                if let Some((tool_name, args, reason)) = audit_ctx {
                    let inner_outcome = crate::server::thread_router::SudoInnerOutcome::Failed {
                        message: detail.clone(),
                    };
                    self.router.write_sudo_audit(
                        &thread_id,
                        &tool_name,
                        args,
                        reason,
                        sudo_decision_tag(decision),
                        None,
                        Some(inner_outcome),
                    );
                }
                self.complete_function(
                    function_id,
                    FunctionOutcome::Error(crate::functions::FunctionError {
                        kind: crate::functions::FunctionErrorKind::Execution,
                        detail,
                    }),
                    pending_io,
                );
            }
        }
    }

    /// Sweep the task map and the active-functions registry for any
    /// state tied to the dropped conn. Threads whose scope's
    /// interactive channel was this conn flip to autonomous;
    /// `RequestEscalation` Functions targeting those threads complete
    /// with a channel-gone error so their parked tool calls resume.
    pub(crate) fn revoke_escalation_channel(
        &mut self,
        dropped: ConnId,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let affected_threads: Vec<String> = self
            .tasks
            .iter_mut()
            .filter_map(|(id, task)| match task.scope.escalation {
                crate::permission::Escalation::Interactive { via_conn } if via_conn == dropped => {
                    task.scope.escalation = crate::permission::Escalation::None;
                    Some(id.clone())
                }
                _ => None,
            })
            .collect();
        if affected_threads.is_empty() {
            return;
        }
        let pending_sudo: Vec<crate::functions::FunctionId> = self
            .active_functions
            .iter()
            .filter_map(|(id, entry)| match &entry.spec {
                Function::Sudo { thread_id, .. }
                    if affected_threads.iter().any(|t| t == thread_id) =>
                {
                    Some(*id)
                }
                _ => None,
            })
            .collect();
        for fn_id in pending_sudo {
            self.complete_function(
                fn_id,
                FunctionOutcome::Error(crate::functions::FunctionError {
                    kind: crate::functions::FunctionErrorKind::Execution,
                    detail: "sudo approver closed before the user could approve; \
                         request abandoned"
                        .into(),
                }),
                pending_io,
            );
        }
        for thread_id in &affected_threads {
            self.mark_dirty(thread_id);
        }
    }

    /// If `thread_id` is a user-owned top-level thread whose
    /// escalation channel is currently `None`, re-attach it to
    /// `conn_id`. Called from the `SendUserMessage` handler so that a
    /// user who reconnects after their original conn dropped (page
    /// reload, WS blip, server restart sanitize) regains `sudo`
    /// without having to recreate the thread.
    ///
    /// Conservative on two axes:
    /// - Only touches `None`. An existing `Interactive{via_conn: M}`
    ///   is left alone even if `M != conn_id` — don't steal a live
    ///   channel out from under another client.
    /// - Only touches threads with no behavior `origin` and no
    ///   `dispatched_by`. Behavior-fired / dispatched threads get
    ///   their escalation at creation and should not gain one from a
    ///   stray client message.
    pub(crate) fn rebind_escalation_if_orphaned(&mut self, thread_id: &str, conn_id: ConnId) {
        let Some(task) = self.tasks.get_mut(thread_id) else {
            return;
        };
        if task.origin.is_some() || task.dispatched_by.is_some() {
            return;
        }
        if !matches!(task.scope.escalation, crate::permission::Escalation::None) {
            return;
        }
        task.scope.escalation = crate::permission::Escalation::Interactive { via_conn: conn_id };
        self.mark_dirty(thread_id);
    }

    /// Look up the effective tool disposition for `thread_id` + tool
    /// `name`. Returns `None` if the thread is unknown — callers pass
    /// the request through unchecked in that case so upstream
    /// route_tool surfaces the error. Consults the thread's snapshot
    /// `Scope.tools`; pod-file edits don't retroactively change a
    /// thread's active scope.
    pub(super) fn tool_disposition(
        &self,
        thread_id: &str,
        name: &str,
    ) -> Option<crate::permission::Disposition> {
        let task = self.tasks.get(thread_id)?;
        Some(task.scope.tools.disposition(&name.to_string()))
    }

    /// Complete the tool-call Function for `(thread_id, tool_use_id)`
    /// with `result`. No-op if no such Function is registered — e.g.,
    /// the dispatch-thread intercept path doesn't register a Function
    /// at this layer (CreateThread Function lands in Phase 5).
    pub(super) fn complete_tool_function(
        &mut self,
        thread_id: &str,
        tool_use_id: &str,
        result: &Result<crate::tools::mcp::CallToolResult, String>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(fn_id) = self.find_tool_function_for(thread_id, tool_use_id) else {
            return;
        };
        let outcome = match result {
            Ok(call) => {
                let terminal_payload = crate::functions::ToolResult {
                    // Phase 4b: terminal payload is minimal — the
                    // actual tool-result content is already integrated
                    // into the thread's conversation via
                    // `apply_io_result`. Phase 4d fills this out when
                    // streaming + caller-link routing to non-thread
                    // callers (lua, WS-direct) needs the payload here.
                    content: Vec::new(),
                    is_error: call.is_error,
                };
                // `find_tool_function_for` already filtered to
                // BuiltinToolCall / McpToolUse specs, so this match is
                // exhaustive for reachable inputs. A non-tool spec
                // here is a bug in upstream filtering; debug_assert.
                let terminal = match self.active_functions.get(&fn_id).map(|e| &e.spec) {
                    Some(Function::BuiltinToolCall { .. }) => {
                        FunctionTerminal::BuiltinToolCall(terminal_payload)
                    }
                    Some(Function::McpToolUse { .. }) => {
                        FunctionTerminal::McpToolUse(terminal_payload)
                    }
                    other => {
                        debug_assert!(
                            false,
                            "complete_tool_function matched non-tool Function spec: {other:?}"
                        );
                        FunctionTerminal::BuiltinToolCall(terminal_payload)
                    }
                };
                FunctionOutcome::Success(terminal)
            }
            Err(e) => FunctionOutcome::Error(crate::functions::FunctionError {
                kind: crate::functions::FunctionErrorKind::Execution,
                detail: e.clone(),
            }),
        };
        self.complete_function(fn_id, outcome, pending_io);
    }

    /// Launch body for `Function::CreateThread`. Creates the thread via
    /// `create_task`, seeds the initial message, steps the thread. For
    /// `WaitMode::ThreadCreated` the Function completes immediately with
    /// a `CreateThreadTerminal { thread_id, final_result: None }` — the
    /// caller only cared that the thread exists. For
    /// `WaitMode::ThreadTerminal` the Function stays in the registry
    /// with `awaiting_child_thread_id = Some(child)`; the scheduler's
    /// thread-terminal hook fires `complete_function` once the child
    /// reaches Completed/Failed/Cancelled.
    #[allow(clippy::too_many_arguments)]
    fn launch_create_thread(
        &mut self,
        id: FunctionId,
        pod_id: Option<crate::permission::PodId>,
        initial_message: Option<String>,
        initial_attachments: Vec<whisper_agent_protocol::Attachment>,
        parent: Option<crate::functions::ParentLink>,
        wait_mode: crate::functions::WaitMode,
        config_override: Option<whisper_agent_protocol::ThreadConfigOverride>,
        bindings_request: Option<whisper_agent_protocol::ThreadBindingsRequest>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        // Pull requester + correlation_id from the caller-link so
        // `create_task` routes `ThreadCreated` broadcasts the same way
        // the pre-migration interactive handler did.
        let (requester, correlation_id) = match self.active_functions.get(&id) {
            Some(entry) => match &entry.caller {
                CallerLink::WsClient {
                    conn_id,
                    correlation_id,
                } => (Some(*conn_id), correlation_id.clone()),
                _ => (None, None),
            },
            None => (None, None),
        };
        // Parent lineage is derivable from the spec + parent thread
        // state: depth = parent.dispatch_depth + 1 (capped by the
        // precondition check).
        let dispatch_lineage = match &parent {
            Some(link) => self
                .tasks
                .get(&link.thread_id)
                .map(|t| (link.thread_id.clone(), t.dispatch_depth)),
            None => None,
        };
        let create_result = self.create_task(
            requester,
            correlation_id,
            pod_id,
            config_override,
            bindings_request,
            None,
            dispatch_lineage,
            None,
            None,
            pending_io,
        );
        let thread_id = match create_result {
            Ok(id) => id,
            Err(e) => {
                self.complete_function(
                    id,
                    FunctionOutcome::Error(crate::functions::FunctionError {
                        kind: crate::functions::FunctionErrorKind::Execution,
                        detail: e,
                    }),
                    pending_io,
                );
                return;
            }
        };
        self.mark_dirty(&thread_id);
        if let Some(text) = initial_message {
            self.send_user_message(&thread_id, text, initial_attachments, pending_io);
        } else if !initial_attachments.is_empty() {
            // Attachments-without-text is a legitimate compose path (user
            // dropped an image into a fresh thread without typing anything).
            // Send with empty text so the Image block(s) still land on the
            // first user turn.
            self.send_user_message(&thread_id, String::new(), initial_attachments, pending_io);
        }
        self.step_until_blocked(&thread_id, pending_io);

        match wait_mode {
            crate::functions::WaitMode::ThreadCreated => {
                self.complete_function(
                    id,
                    FunctionOutcome::Success(FunctionTerminal::CreateThread(
                        crate::functions::CreateThreadTerminal {
                            thread_id,
                            final_result: None,
                        },
                    )),
                    pending_io,
                );
            }
            crate::functions::WaitMode::ThreadTerminal => {
                // Stash the child id; the thread-terminal hook will
                // call `complete_function` when the child finishes.
                if let Some(entry) = self.active_functions.get_mut(&id) {
                    entry.awaiting_child_thread_id = Some(thread_id.clone());
                }
                // Edge case: step_until_blocked may have driven the
                // child all the way to terminal already (zero-turn
                // response, immediate error, etc.). Fire the terminal
                // hook now so we don't park indefinitely waiting for a
                // signal that has already passed.
                let terminal = self
                    .tasks
                    .get(&thread_id)
                    .map(|t| {
                        matches!(
                            t.public_state(),
                            ThreadStateLabel::Completed
                                | ThreadStateLabel::Failed
                                | ThreadStateLabel::Cancelled
                        )
                    })
                    .unwrap_or(false);
                if terminal {
                    self.complete_functions_awaiting_thread(&thread_id, pending_io);
                }
            }
        }
    }

    /// Thread-terminal hook that closes out any Function waiting on
    /// this thread as its "await child" completion signal. Walks the
    /// active_functions registry; for each entry with
    /// `awaiting_child_thread_id == Some(thread_id)`, builds the
    /// terminal payload from the thread's final state and calls
    /// `complete_function`. The thread's state drives Success vs
    /// Error vs Cancelled.
    pub(super) fn complete_functions_awaiting_thread(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let matches: Vec<FunctionId> = self
            .active_functions
            .iter()
            .filter_map(
                |(fn_id, entry)| match entry.awaiting_child_thread_id.as_deref() {
                    Some(t) if t == thread_id => Some(*fn_id),
                    _ => None,
                },
            )
            .collect();
        if matches.is_empty() {
            return;
        }
        let Some(task) = self.tasks.get(thread_id) else {
            // Thread vanished — complete each Function with a
            // CallerGone cancel so the caller-link routing still
            // delivers something.
            for fn_id in matches {
                self.complete_function(
                    fn_id,
                    FunctionOutcome::Cancelled(crate::functions::CancelReason::CallerGone),
                    pending_io,
                );
            }
            return;
        };
        let state = task.public_state();
        let outcome = match state {
            ThreadStateLabel::Completed => {
                let text = crate::runtime::scheduler::compaction::extract_last_assistant_text(task);
                FunctionOutcome::Success(FunctionTerminal::CreateThread(
                    crate::functions::CreateThreadTerminal {
                        thread_id: thread_id.to_string(),
                        final_result: Some(crate::functions::ThreadTerminalSummary {
                            state: "completed".into(),
                            final_text: Some(text),
                        }),
                    },
                ))
            }
            ThreadStateLabel::Failed => {
                let detail = task
                    .failure_detail()
                    .unwrap_or_else(|| "child thread failed".into());
                FunctionOutcome::Error(crate::functions::FunctionError {
                    kind: crate::functions::FunctionErrorKind::Execution,
                    detail,
                })
            }
            ThreadStateLabel::Cancelled => {
                FunctionOutcome::Cancelled(crate::functions::CancelReason::ExplicitCancel)
            }
            // Not terminal — shouldn't happen (caller only invokes this
            // on terminal transitions), but be defensive.
            _ => return,
        };
        for fn_id in matches {
            self.complete_function(fn_id, outcome.clone(), pending_io);
        }
    }

    /// Walk `active_functions` for entries whose `CallerLink` targets
    /// this terminating parent thread via `ThreadToolCall` and cancel
    /// them. For Functions awaiting a child (sync dispatch), also
    /// cancel the child thread — its result has no consumer anymore.
    /// Recursively processes the cancelled child's post-terminal
    /// lifecycle so nested dispatches cascade transitively.
    pub(super) fn cascade_cancel_caller_gone(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(parent_task) = self.tasks.get(thread_id) else {
            return;
        };
        let parent_dead = matches!(
            parent_task.public_state(),
            ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
        );
        if !parent_dead {
            return;
        }
        let targeted: Vec<(FunctionId, Option<String>)> = self
            .active_functions
            .iter()
            .filter_map(|(fn_id, entry)| match &entry.caller {
                CallerLink::ThreadToolCall {
                    thread_id: parent, ..
                } if parent == thread_id => Some((*fn_id, entry.awaiting_child_thread_id.clone())),
                _ => None,
            })
            .collect();
        for (fn_id, awaiting_child) in targeted {
            // Cancel the awaited child (if any) so its result has no
            // dangling consumer, and run the child's own post-terminal
            // lifecycle so nested dispatches cascade transitively.
            let newly_cancelled_child = if let Some(child_id) = &awaiting_child
                && let Some(child) = self.tasks.get_mut(child_id)
            {
                let already_terminal = matches!(
                    child.public_state(),
                    ThreadStateLabel::Completed
                        | ThreadStateLabel::Failed
                        | ThreadStateLabel::Cancelled
                );
                if already_terminal {
                    None
                } else {
                    let mut cancel_events = Vec::new();
                    child.cancel(&mut cancel_events);
                    if let Some(token) = self.cancel_tokens.get(child_id) {
                        token.cancel();
                    }
                    self.router.dispatch_events(child_id, cancel_events);
                    self.mark_dirty(child_id);
                    self.router
                        .broadcast_task_list(ServerToClient::ThreadStateChanged {
                            thread_id: child_id.clone(),
                            state: ThreadStateLabel::Cancelled,
                        });
                    warn!(
                        parent = %thread_id, child = %child_id,
                        "parent terminated; cascading cancel to dispatched child"
                    );
                    self.teardown_host_env_if_terminal(child_id);
                    self.on_behavior_thread_terminal(child_id, pending_io);
                    Some(child_id.clone())
                }
            } else {
                None
            };
            // Clear the awaiting link before completing so the child's
            // own terminal processing (below) doesn't re-fire this
            // Function after we've cancelled it.
            if let Some(entry) = self.active_functions.get_mut(&fn_id) {
                entry.awaiting_child_thread_id = None;
            }
            self.complete_function(
                fn_id,
                FunctionOutcome::Cancelled(crate::functions::CancelReason::CallerGone),
                pending_io,
            );
            // Transitive cascade: the now-cancelled child may itself
            // be a parent of further dispatched children / awaiting
            // Functions. Recursion terminates at threads with no
            // registered Functions targeting them.
            if let Some(child_id) = newly_cancelled_child {
                self.complete_functions_awaiting_thread(&child_id, pending_io);
                self.cascade_cancel_caller_gone(&child_id, pending_io);
            }
        }
    }

    /// Cancel a thread by id: flip its state, broadcast, and run the
    /// existing lifecycle cascade (host-env teardown, behavior terminal
    /// hook, dispatch resolution, child cancellation). Equivalent to the
    /// pre-migration `ClientToServer::CancelThread` handler body; called
    /// by `Function::CancelThread`'s launch path and by server-config
    /// updates that need to cancel users of a removed/changed backend.
    pub(super) fn execute_cancel_thread(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(task) = self.tasks.get_mut(thread_id) else {
            // Precondition already checked thread existence, but the
            // scheduler's single-writer discipline means we re-check
            // defensively — a concurrent teardown could have removed it.
            return;
        };
        let mut cancel_events = Vec::new();
        task.cancel(&mut cancel_events);
        // Fire the per-thread cancel signal so any in-flight
        // model/tool/MCP HTTP request aborts at the wire instead of
        // running to completion and having its result discarded.
        if let Some(token) = self.cancel_tokens.get(thread_id) {
            token.cancel();
        }
        // Dispatch any synthesized ToolCallEnd events so live
        // subscribers see the interrupted tool-call rows close out.
        self.router.dispatch_events(thread_id, cancel_events);
        self.mark_dirty(thread_id);
        self.router
            .broadcast_task_list(ServerToClient::ThreadStateChanged {
                thread_id: thread_id.to_string(),
                state: ThreadStateLabel::Cancelled,
            });
        self.teardown_host_env_if_terminal(thread_id);
        self.on_behavior_thread_terminal(thread_id, pending_io);
        // Function-registry lifecycle: if this cancelled thread is a
        // child awaited by a `Function::CreateThread{ThreadTerminal}`,
        // fire that Function's terminal. If this thread is itself a
        // parent with registered Functions pointing back at it, cancel
        // them via caller-gone cascade.
        self.complete_functions_awaiting_thread(thread_id, pending_io);
        self.cascade_cancel_caller_gone(thread_id, pending_io);
    }
}

/// Map the wire-level [`crate::permission::SudoDecision`] to the
/// audit-log [`crate::runtime::audit::SudoDecisionTag`]. They're the
/// same three variants; the indirection keeps the audit module from
/// importing the protocol crate's permission types directly.
fn sudo_decision_tag(
    decision: crate::permission::SudoDecision,
) -> crate::runtime::audit::SudoDecisionTag {
    match decision {
        crate::permission::SudoDecision::ApproveOnce => {
            crate::runtime::audit::SudoDecisionTag::ApproveOnce
        }
        crate::permission::SudoDecision::ApproveRemember => {
            crate::runtime::audit::SudoDecisionTag::ApproveRemember
        }
        crate::permission::SudoDecision::Reject => crate::runtime::audit::SudoDecisionTag::Reject,
    }
}

/// Convert a Function terminal outcome to the Result payload a sync
/// `dispatch_thread` caller expects: `Ok(final_text)` for a successful
/// CreateThread terminal; `Err(message)` for error or cancelled.
fn outcome_to_sync_tool_result(outcome: &FunctionOutcome) -> Result<String, String> {
    match outcome {
        FunctionOutcome::Success(FunctionTerminal::CreateThread(term)) => {
            let text = term
                .final_result
                .as_ref()
                .and_then(|s| s.final_text.clone())
                .unwrap_or_default();
            Ok(text)
        }
        FunctionOutcome::Success(FunctionTerminal::Sudo(term)) => Ok(term.tool_result_text.clone()),
        FunctionOutcome::Success(_) => {
            Err("tool-parked Function terminated with non-tool-result payload".to_string())
        }
        FunctionOutcome::Error(err) => Err(err.detail.clone()),
        FunctionOutcome::Cancelled(reason) => Err(format!("tool call cancelled ({:?})", reason)),
    }
}

/// Build a future that immediately fires with a tool-error
/// `IoCompletion`. Used for dispatch_thread arg-parse errors and
/// register_function rejections — the thread integrates the error as
/// a tool_result and continues its turn.
fn immediate_tool_error(
    parent_thread_id: String,
    op_id: crate::runtime::thread::OpId,
    tool_use_id: String,
    message: String,
) -> SchedulerFuture {
    Box::pin(async move {
        crate::runtime::io_dispatch::SchedulerCompletion::Io(
            crate::runtime::io_dispatch::IoCompletion {
                thread_id: parent_thread_id,
                op_id,
                result: crate::runtime::thread::IoResult::ToolCall {
                    tool_use_id,
                    result: Err(message),
                },
                pod_update: None,
                scheduler_command: None,
                host_env_lost: None,
            },
        )
    })
}

/// Build a future that immediately fires with a successful tool
/// completion carrying the given text body. Used by the synchronous
/// scheduler-intercepted tools (`describe_tool`, `find_tool`) where
/// the result is computable directly from scheduler state — no I/O,
/// no Function, no waiting.
fn immediate_tool_success(
    parent_thread_id: String,
    op_id: crate::runtime::thread::OpId,
    tool_use_id: String,
    text: String,
) -> SchedulerFuture {
    Box::pin(async move {
        crate::runtime::io_dispatch::SchedulerCompletion::Io(
            crate::runtime::io_dispatch::IoCompletion {
                thread_id: parent_thread_id,
                op_id,
                result: crate::runtime::thread::IoResult::ToolCall {
                    tool_use_id,
                    result: Ok(crate::tools::mcp::CallToolResult {
                        content: vec![crate::tools::mcp::McpContentBlock::Text { text }],
                        is_error: false,
                    }),
                },
                pod_update: None,
                scheduler_command: None,
                host_env_lost: None,
            },
        )
    })
}

/// Like `immediate_tool_success` but produces the inner
/// `SchedulerCompletion` directly — for use from inside an `async
/// move` block already pushed onto `pending_io` as a `Box::pin`. The
/// outer future returns this completion as its terminal value.
fn tool_success_completion(
    parent_thread_id: String,
    op_id: crate::runtime::thread::OpId,
    tool_use_id: String,
    text: String,
) -> crate::runtime::io_dispatch::SchedulerCompletion {
    crate::runtime::io_dispatch::SchedulerCompletion::Io(
        crate::runtime::io_dispatch::IoCompletion {
            thread_id: parent_thread_id,
            op_id,
            result: crate::runtime::thread::IoResult::ToolCall {
                tool_use_id,
                result: Ok(crate::tools::mcp::CallToolResult {
                    content: vec![crate::tools::mcp::McpContentBlock::Text { text }],
                    is_error: false,
                }),
            },
            pod_update: None,
            scheduler_command: None,
            host_env_lost: None,
        },
    )
}

/// Like `immediate_tool_error` but produces the inner
/// `SchedulerCompletion` directly. See `tool_success_completion`.
fn tool_error_completion(
    parent_thread_id: String,
    op_id: crate::runtime::thread::OpId,
    tool_use_id: String,
    message: String,
) -> crate::runtime::io_dispatch::SchedulerCompletion {
    crate::runtime::io_dispatch::SchedulerCompletion::Io(
        crate::runtime::io_dispatch::IoCompletion {
            thread_id: parent_thread_id,
            op_id,
            result: crate::runtime::thread::IoResult::ToolCall {
                tool_use_id,
                result: Err(message),
            },
            pod_update: None,
            scheduler_command: None,
            host_env_lost: None,
        },
    )
}

/// Format reranked hits for the `knowledge_query` tool response. One
/// header line then per-hit: source title, retrieval-path tag,
/// scores, locator (if any), then a snippet of the chunk text capped
/// at `SNIPPET_CHARS` to keep the per-call response bounded.
fn format_hits(
    query: &str,
    queried_buckets: &[String],
    hits: &[crate::knowledge::RerankedCandidate],
) -> String {
    use crate::knowledge::SearchPath;

    let bucket_list = queried_buckets
        .iter()
        .map(|b| format!("`{b}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut out = format!(
        "Query: {query:?}\nQueried buckets: {bucket_list}\nHits: {}\n\n",
        hits.len()
    );
    if hits.is_empty() {
        out.push_str("(no hits — try rephrasing or widening the query)\n");
        return out;
    }
    for (i, h) in hits.iter().enumerate() {
        let path = match h.source_path {
            SearchPath::Dense => "dense",
            SearchPath::Sparse => "sparse",
        };
        let locator = h
            .source_ref
            .locator
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(|s| format!(" • {s}"))
            .unwrap_or_default();
        let title = if h.source_ref.source_id.is_empty() {
            format!(
                "(no source id) — chunk {}",
                short_id(&h.chunk_id.to_string())
            )
        } else {
            h.source_ref.source_id.clone()
        };
        // Send the full chunk text. The chunker is the natural bound on
        // per-hit size — chunks default to ~500 tokens (~2000 chars),
        // so even top_k=20 stays under ~10k tokens of tool output.
        // Truncating here just hid the answer when it landed past the
        // first paragraph (e.g. "melting point of lead" landed in the
        // second half of the lede chunk and wasn't visible to the model).
        out.push_str(&format!(
            "{rank}. [{bucket}] {title}\n   via {path}, rerank={rerank:.3}, src={src:.3}{locator}\n\n{text}\n\n",
            rank = i + 1,
            bucket = h.bucket_id,
            rerank = h.rerank_score,
            src = h.source_score,
            text = h.chunk_text,
        ));
    }
    out
}

fn short_id(s: &str) -> String {
    if s.len() > 12 {
        format!("{}…", &s[..12])
    } else {
        s.to_string()
    }
}

/// Truncate a multi-line/multi-sentence description for `find_tool`'s
/// result lines. Same spirit as `tool_listing::one_line_description`
/// but kept local to keep the scheduler module free of wiring into
/// the tool-listing module's private helper.
fn one_line(desc: &str) -> String {
    let first = desc.split(['\n', '.']).next().unwrap_or(desc).trim();
    if first.len() > 120 {
        let mut t = first.chars().take(117).collect::<String>();
        t.push_str("...");
        t
    } else {
        first.to_string()
    }
}

/// Build a future that immediately fires with a tool-denial completion.
/// Used when scope evaluation returns `Deny` and the thread needs to
/// see the denial as a regular tool-result error.
fn make_denial_future(
    thread_id: String,
    tool_use_id: String,
    op_id: crate::runtime::thread::OpId,
    tool_name: String,
) -> SchedulerFuture {
    Box::pin(async move {
        crate::runtime::io_dispatch::SchedulerCompletion::Io(
            crate::runtime::io_dispatch::IoCompletion {
                thread_id,
                op_id,
                result: crate::runtime::thread::IoResult::ToolCall {
                    tool_use_id,
                    result: Err(format!(
                        "tool `{tool_name}` denied by scope (disposition: deny)"
                    )),
                },
                pod_update: None,
                scheduler_command: None,
                host_env_lost: None,
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::{FunctionError, FunctionErrorKind};

    #[test]
    fn outcome_to_sync_tool_result_forwards_sudo_text_on_approve() {
        use crate::functions::SudoTerminal;
        use crate::permission::SudoDecision;
        let outcome = FunctionOutcome::Success(FunctionTerminal::Sudo(SudoTerminal {
            decision: SudoDecision::ApproveOnce,
            tool_result_text: "wrote 42 bytes to system_prompt.md".into(),
        }));
        let r = outcome_to_sync_tool_result(&outcome).expect("approved sudo is Ok");
        assert!(r.contains("wrote 42 bytes"));
    }

    #[test]
    fn outcome_to_sync_tool_result_forwards_sudo_error_on_inner_fail() {
        // ApproveOnce + inner failure rides the Error outcome so the
        // parent's parked oneshot fires Err(detail) — the model's
        // tool_result shows is_error: true with the inner tool's text.
        let outcome = FunctionOutcome::Error(FunctionError {
            kind: FunctionErrorKind::Execution,
            detail: "pod_write_file: path `.archived/x` is not in the allowlist".into(),
        });
        let e = outcome_to_sync_tool_result(&outcome).expect_err("inner failure is Err");
        assert!(e.contains("allowlist"));
    }
}
