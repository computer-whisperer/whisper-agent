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
            Function::RequestEscalation {
                thread_id, request, ..
            } => {
                let task =
                    self.tasks
                        .get(thread_id)
                        .ok_or_else(|| RejectReason::PreconditionFailed {
                            detail: format!("unknown thread {thread_id}"),
                        })?;
                if !task.scope.escalation.is_interactive() {
                    return Err(RejectReason::ScopeDenied {
                        detail: "thread has no interactive escalation channel".into(),
                    });
                }
                let pod_ceiling = self.pod_scope_ceiling(&task.pod_id);
                task.scope
                    .check_escalation(request, &pod_ceiling)
                    .map_err(|detail| RejectReason::ScopeDenied { detail })
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
                parent,
                wait_mode,
                config_override,
                bindings_request,
            } => {
                self.launch_create_thread(
                    id,
                    pod_id,
                    initial_message,
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
            Function::RequestEscalation {
                thread_id,
                request,
                reason,
            } => {
                // Emit the wire-side approval request to the thread's
                // interactive channel. The Function stays admitted
                // until `apply_client_message` routes a matching
                // `ClientToServer::ResolveEscalation` through
                // `resolve_escalation`, which either applies the
                // widening and completes successfully or completes
                // with an error.
                let via_conn = match self.tasks.get(&thread_id).map(|t| t.scope.escalation) {
                    Some(crate::permission::Escalation::Interactive { via_conn }) => via_conn,
                    _ => {
                        // Precondition says this can't happen, but be
                        // defensive: if the escalation channel vanished
                        // between registration and launch, complete the
                        // Function with a denial rather than leaving it
                        // pending with no route back.
                        self.complete_function(
                            id,
                            FunctionOutcome::Error(crate::functions::FunctionError {
                                kind: crate::functions::FunctionErrorKind::Execution,
                                detail: "interactive escalation channel unavailable".into(),
                            }),
                            pending_io,
                        );
                        return;
                    }
                };
                self.router.send_to_client(
                    via_conn,
                    ServerToClient::EscalationRequested {
                        function_id: id,
                        thread_id: thread_id.clone(),
                        request: request.clone(),
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

        // `request_escalation` is intercepted the same way: register
        // a `Function::RequestEscalation`, emit the wire event to the
        // interactive channel, park the tool call on the Function's
        // terminal via a `FunctionDelivery::ToolResultChannel`.
        if name == crate::tools::builtin_tools::REQUEST_ESCALATION {
            self.register_request_escalation_tool(
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
                    "available via escalation"
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
        let matched: Vec<_> = all
            .into_iter()
            .filter(|t| {
                if !args.include_escalation && t.requires_escalation {
                    return false;
                }
                if let Some(cat) = &args.category
                    && cat != t.category.coarse()
                {
                    return false;
                }
                regex.is_match(&t.name) || regex.is_match(&t.description)
            })
            .collect();
        let total = matched.len();
        let page: Vec<_> = matched.into_iter().skip(offset).take(limit).collect();
        let shown = page.len();
        let mut out = String::new();
        out.push_str(&format!(
            "Found {total} tool(s) matching `{pattern}`; showing {shown} (offset {offset}, limit {limit}).\n\n",
            pattern = args.pattern
        ));
        for t in page {
            let tag = if t.requires_escalation {
                " [escalation]"
            } else {
                ""
            };
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

    /// `request_escalation`-specific registration path. Parses the
    /// tool args, refuses at the tool layer if the thread's scope lacks
    /// an interactive channel, otherwise registers the Function (which
    /// fires the wire event on launch) and parks the tool call on its
    /// terminal via `FunctionDelivery::ToolResultChannel`. The parent's
    /// turn resumes when the user resolves the escalation.
    fn register_request_escalation_tool(
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
                crate::tools::builtin_tools::REQUEST_ESCALATION.to_string(),
            ));
            return;
        }
        // Hard gate: without an interactive channel, there's no
        // approver — `request_escalation` is a no-op.
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
                "request_escalation denied: thread has no interactive approver \
                 (scope.escalation = None). Autonomous threads cannot widen \
                 their own scope."
                    .into(),
            ));
            return;
        }
        let args = match crate::tools::builtin_tools::request_escalation::parse_args(input) {
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
        let spec = Function::RequestEscalation {
            thread_id: thread_id.to_string(),
            request: args.request,
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
                    format!(
                        "request_escalation: {}",
                        super::client_messages::reject_reason_detail(&e)
                    ),
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
                    // On reject / channel-drop we return is_error=true
                    // but keep the message inline so the model's tool
                    // result carries the denial text for course
                    // correction.
                    content: vec![crate::tools::mcp::McpContentBlock::Text { text: msg }],
                    is_error: true,
                }),
                Err(_) => Err(
                    "request_escalation: Function terminated without delivering a result".into(),
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
    }

    /// Resolve an in-flight `RequestEscalation` Function. On approve,
    /// widens the thread's scope in place and broadcasts a refreshed
    /// `ThreadSnapshot` so subscribers see the new scope; on reject,
    /// surfaces the user's reason through the tool result. Either path
    /// completes the Function, firing the parked tool-result oneshot
    /// and unblocking the thread's turn.
    ///
    /// `resolver_conn` is the connection that sent the resolution —
    /// must match the thread's interactive channel. A mismatch returns
    /// an error event to the resolver and leaves the Function pending.
    pub(crate) fn resolve_escalation(
        &mut self,
        resolver_conn: ConnId,
        function_id: crate::functions::FunctionId,
        decision: crate::permission::EscalationDecision,
        reason: Option<String>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let entry = match self.active_functions.get(&function_id) {
            Some(e) => e,
            None => {
                warn!(
                    function_id,
                    resolver_conn, "resolve_escalation: no such Function"
                );
                self.router.send_to_client(
                    resolver_conn,
                    ServerToClient::Error {
                        correlation_id: None,
                        thread_id: None,
                        message: format!("resolve_escalation: unknown function_id {function_id}"),
                    },
                );
                return;
            }
        };
        let Function::RequestEscalation {
            thread_id, request, ..
        } = entry.spec.clone()
        else {
            self.router.send_to_client(
                resolver_conn,
                ServerToClient::Error {
                    correlation_id: None,
                    thread_id: entry.spec.primary_thread_id().map(str::to_string),
                    message: format!(
                        "resolve_escalation: function {function_id} is not a RequestEscalation"
                    ),
                },
            );
            return;
        };
        // The resolver must own the thread's interactive channel —
        // arbitrary third parties can't grant each other's widenings.
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
                    message: "resolve_escalation: caller is not the thread's interactive channel"
                        .to_string(),
                },
            );
            return;
        }
        match decision {
            crate::permission::EscalationDecision::Approve => {
                // Re-check the widening against the current scope +
                // pod ceiling at grant time. Mid-flight pod edits or
                // other grants could have shifted the baseline — if
                // the check fails now, surface the reason rather than
                // silently succeeding on a stale request.
                let grant_text = {
                    let pod_id = self.tasks.get(&thread_id).map(|t| t.pod_id.clone());
                    let pod_ceiling = pod_id.as_deref().map(|id| self.pod_scope_ceiling(id));
                    let check = self.tasks.get(&thread_id).and_then(|t| {
                        pod_ceiling
                            .as_ref()
                            .map(|c| t.scope.check_escalation(&request, c))
                    });
                    match check {
                        Some(Ok(())) => Ok(()),
                        Some(Err(detail)) => Err(detail),
                        None => Err("thread gone".into()),
                    }
                };
                match grant_text {
                    Ok(()) => {
                        let snapshot = if let Some(task) = self.tasks.get_mut(&thread_id) {
                            task.scope.apply_escalation(&request);
                            Some(task.snapshot())
                        } else {
                            None
                        };
                        let blurb = format!(
                            "Escalation granted: {}. The capability is now available \
                             for this thread.",
                            describe_request(&request)
                        );
                        self.mark_dirty(&thread_id);
                        if let Some(snapshot) = snapshot {
                            self.router.broadcast_to_subscribers(
                                &thread_id,
                                ServerToClient::ThreadSnapshot {
                                    thread_id: thread_id.clone(),
                                    snapshot,
                                },
                            );
                        }
                        self.router
                            .broadcast_task_list(ServerToClient::EscalationResolved {
                                function_id,
                                thread_id: thread_id.clone(),
                                decision,
                            });
                        self.router.write_escalation_audit(
                            &thread_id,
                            request.clone(),
                            crate::server::thread_router::EscalationResolutionOwned::Approved,
                        );
                        self.complete_function(
                            function_id,
                            FunctionOutcome::Success(
                                crate::functions::FunctionTerminal::RequestEscalation(
                                    crate::functions::RequestEscalationTerminal {
                                        decision,
                                        tool_result_text: blurb,
                                    },
                                ),
                            ),
                            pending_io,
                        );
                    }
                    Err(detail) => {
                        self.router
                            .broadcast_task_list(ServerToClient::EscalationResolved {
                                function_id,
                                thread_id: thread_id.clone(),
                                decision: crate::permission::EscalationDecision::Reject,
                            });
                        self.router.write_escalation_audit(
                            &thread_id,
                            request.clone(),
                            crate::server::thread_router::EscalationResolutionOwned::RecheckFailed {
                                detail: detail.clone(),
                            },
                        );
                        self.complete_function(
                            function_id,
                            FunctionOutcome::Error(crate::functions::FunctionError {
                                kind: crate::functions::FunctionErrorKind::Execution,
                                detail: format!("escalation grant re-check failed: {detail}"),
                            }),
                            pending_io,
                        );
                    }
                }
            }
            crate::permission::EscalationDecision::Reject => {
                let denial_text = match &reason {
                    Some(r) if !r.trim().is_empty() => format!("Escalation rejected: {r}"),
                    _ => "Escalation rejected by user.".to_string(),
                };
                self.router
                    .broadcast_task_list(ServerToClient::EscalationResolved {
                        function_id,
                        thread_id: thread_id.clone(),
                        decision,
                    });
                self.router.write_escalation_audit(
                    &thread_id,
                    request.clone(),
                    crate::server::thread_router::EscalationResolutionOwned::Rejected {
                        reason: reason.as_ref().and_then(|r| {
                            let trimmed = r.trim();
                            if trimmed.is_empty() {
                                None
                            } else {
                                Some(trimmed.to_string())
                            }
                        }),
                    },
                );
                // Rejection rides the Error outcome so
                // `outcome_to_sync_tool_result` returns `Err(detail)`,
                // which the parked oneshot in
                // `register_request_escalation_tool` converts into a
                // `CallToolResult { is_error: true }`. The model sees
                // the denial framed as a tool-result error and can
                // course-correct.
                self.complete_function(
                    function_id,
                    FunctionOutcome::Error(crate::functions::FunctionError {
                        kind: crate::functions::FunctionErrorKind::Execution,
                        detail: denial_text,
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
        let pending_fns: Vec<(
            crate::functions::FunctionId,
            String,
            crate::permission::EscalationRequest,
        )> = self
            .active_functions
            .iter()
            .filter_map(|(id, entry)| match &entry.spec {
                Function::RequestEscalation {
                    thread_id, request, ..
                } if affected_threads.iter().any(|t| t == thread_id) => {
                    Some((*id, thread_id.clone(), request.clone()))
                }
                _ => None,
            })
            .collect();
        for (fn_id, thread_id, request) in pending_fns {
            self.router.write_escalation_audit(
                &thread_id,
                request,
                crate::server::thread_router::EscalationResolutionOwned::ChannelDropped,
            );
            self.complete_function(
                fn_id,
                FunctionOutcome::Error(crate::functions::FunctionError {
                    kind: crate::functions::FunctionErrorKind::Execution,
                    detail: "escalation channel closed before the user could approve; \
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
            self.send_user_message(&thread_id, text, pending_io);
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
                    child.cancel();
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
    /// by `Function::CancelThread`'s launch path.
    fn execute_cancel_thread(
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
        task.cancel();
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
        FunctionOutcome::Success(FunctionTerminal::RequestEscalation(term)) => {
            Ok(term.tool_result_text.clone())
        }
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

/// Truncate a multi-line/multi-sentence description for `find_tool`'s
/// result lines. Same spirit as `tool_listing::one_line_description`
/// but kept local to keep the scheduler module free of wiring into
/// the tool-listing module's private helper.
fn one_line(desc: &str) -> String {
    let first = desc
        .split(['\n', '.'])
        .next()
        .unwrap_or(desc)
        .trim();
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

/// Human-readable one-liner describing what an `EscalationRequest`
/// would widen. Used in the grant tool-result text.
fn describe_request(req: &crate::permission::EscalationRequest) -> String {
    use crate::permission::EscalationRequest::*;
    match req {
        AddTool { name } => format!("tool `{name}` added to scope"),
        RaisePodModify { target } => format!("pod_modify raised to {target:?}"),
        RaiseBehaviors { target } => format!("behaviors raised to {target:?}"),
        RaiseDispatch { target } => format!("dispatch raised to {target:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::{FunctionError, FunctionErrorKind, RequestEscalationTerminal};
    use crate::permission::{BehaviorOpsCap, DispatchCap, EscalationDecision, PodModifyCap};

    #[test]
    fn describe_request_covers_every_variant() {
        use crate::permission::EscalationRequest::*;
        assert_eq!(
            describe_request(&AddTool {
                name: "write_file".into()
            }),
            "tool `write_file` added to scope"
        );
        assert!(
            describe_request(&RaisePodModify {
                target: PodModifyCap::ModifyAllow,
            })
            .contains("pod_modify raised to"),
        );
        assert!(
            describe_request(&RaiseBehaviors {
                target: BehaviorOpsCap::AuthorAny,
            })
            .contains("behaviors raised to"),
        );
        assert!(
            describe_request(&RaiseDispatch {
                target: DispatchCap::WithinScope,
            })
            .contains("dispatch raised to"),
        );
    }

    #[test]
    fn outcome_to_sync_tool_result_returns_ok_for_approved_escalation() {
        let outcome = FunctionOutcome::Success(FunctionTerminal::RequestEscalation(
            RequestEscalationTerminal {
                decision: EscalationDecision::Approve,
                tool_result_text: "Escalation granted: tool `exec` added to scope.".into(),
            },
        ));
        let r = outcome_to_sync_tool_result(&outcome).expect("approved grant is Ok");
        assert!(r.contains("Escalation granted"));
    }

    #[test]
    fn outcome_to_sync_tool_result_returns_err_for_rejected_escalation() {
        // Rejection rides the Error outcome (see note in resolve_escalation):
        // the parked oneshot converts Err(detail) into a tool-result with
        // is_error: true, so the model reads it as a denial and recovers.
        let outcome = FunctionOutcome::Error(FunctionError {
            kind: FunctionErrorKind::Execution,
            detail: "Escalation rejected: not now.".into(),
        });
        let e = outcome_to_sync_tool_result(&outcome).expect_err("rejection surfaces as Err");
        assert!(e.contains("rejected"));
    }

    #[test]
    fn outcome_to_sync_tool_result_reports_recheck_failure_as_err() {
        // Approve-path re-check failure is also surfaced as Err so the
        // calling tool sees an error result rather than a silent success.
        let outcome = FunctionOutcome::Error(FunctionError {
            kind: FunctionErrorKind::Execution,
            detail: "escalation grant re-check failed: would exceed pod ceiling".into(),
        });
        let e = outcome_to_sync_tool_result(&outcome).expect_err("recheck failure is Err");
        assert!(e.contains("re-check failed"));
    }
}
