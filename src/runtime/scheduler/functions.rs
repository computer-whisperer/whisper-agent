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

use futures::stream::FuturesUnordered;
use tokio::sync::oneshot;
use tracing::{debug, warn};
use whisper_agent_protocol::{ServerToClient, ThreadStateLabel};

use super::Scheduler;
use crate::functions::{
    CallerLink, Function, FunctionId, FunctionOutcome, FunctionTerminal, InFlightOps,
    RejectReason,
};
use crate::permission::{BehaviorOp, PermissionScope, ThreadOp};
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
    pub scope: PermissionScope,
    pub caller: CallerLink,
    /// For tool-call Functions whose scope disposition is
    /// `AllowWithPrompt`: the IoRequest is buffered here while we
    /// wait for the user's approval. When the `ApprovalDecision`
    /// arrives, the scheduler rebuilds the future from this request
    /// (for approve) or synthesizes a denial (for reject).
    ///
    /// `None` for Functions that aren't pending approval (synchronous
    /// variants, already-approved tool calls, etc.).
    pub pending_approval_io: Option<crate::runtime::thread::IoRequest>,
    /// For `Function::CreateThread { wait_mode: ThreadTerminal, .. }`:
    /// the child thread whose terminal completes this Function. The
    /// scheduler's thread-terminal hook reads this to decide which
    /// Functions to fire.
    pub awaiting_child_thread_id: Option<String>,
    /// How the terminal is delivered to the caller. Set at
    /// registration; consumed by `complete_function`.
    pub delivery: FunctionDelivery,
}

/// How the terminal of an in-flight Function gets delivered back to
/// its caller beyond the default wire-Error routing in
/// `complete_function`. Most variants use `None`; dispatch-driven
/// tool calls populate one of the tool-result variants so the
/// parent's tool_use_id eventually sees the child's result.
pub enum FunctionDelivery {
    /// Default — no variant-specific delivery. Errors for `WsClient`
    /// callers still produce a wire `Error`; successful terminals
    /// rely on variant-level broadcasts (`ThreadCompacted`,
    /// `ThreadBindingsChanged`, etc.).
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

    /// Full-trust scope for WS clients. Hardcoded for Phase 2 — see the
    /// "scope plumbing" commitment in `docs/design_functions.md`. Becomes
    /// per-identity when we ship Pattern-3 authz.
    pub(super) fn ws_client_scope(&self) -> PermissionScope {
        PermissionScope::allow_all()
    }

    /// Full-trust scope for scheduler-internal callers (auto-compact,
    /// cron fires). These originate from the server's own timers /
    /// state machine; there's no caller identity to narrow against.
    pub(super) fn internal_scope(&self) -> PermissionScope {
        PermissionScope::allow_all()
    }

    /// Synchronous accept-or-deny registration. On accept the
    /// `ActiveFunctionEntry` enters the registry and its `FunctionId` is
    /// returned. Anything that requires awaiting I/O happens later, in
    /// the variant's launch path.
    pub(super) fn register_function(
        &mut self,
        spec: Function,
        scope: PermissionScope,
        caller: CallerLink,
    ) -> Result<FunctionId, RejectReason> {
        self.register_function_with_delivery(spec, scope, caller, FunctionDelivery::None)
    }

    /// Full registration shape used by callers that need to attach a
    /// non-default terminal delivery (dispatch_thread's sync oneshot or
    /// async follow-up). Most callers use `register_function` and get
    /// `FunctionDelivery::None`.
    pub(super) fn register_function_with_delivery(
        &mut self,
        spec: Function,
        scope: PermissionScope,
        caller: CallerLink,
        delivery: FunctionDelivery,
    ) -> Result<FunctionId, RejectReason> {
        self.variant_precondition_check(&spec, &scope)?;
        let id = self.next_function_id();
        let entry = ActiveFunctionEntry {
            id,
            spec,
            scope,
            caller,
            pending_approval_io: None,
            awaiting_child_thread_id: None,
            delivery,
        };
        debug!(
            function_id = id,
            caller = %entry.caller.audit_tag(),
            "Function registered"
        );
        self.active_functions.insert(id, entry);
        Ok(id)
    }

    /// Variant-specific scope + precondition check. Called synchronously
    /// from `register_function` before admission.
    fn variant_precondition_check(
        &self,
        spec: &Function,
        scope: &PermissionScope,
    ) -> Result<(), RejectReason> {
        match spec {
            Function::CancelThread { thread_id } => {
                let task = self.tasks.get(thread_id).ok_or_else(|| {
                    RejectReason::PreconditionFailed {
                        detail: format!("unknown thread {thread_id}"),
                    }
                })?;
                admission_check(scope.thread_op(&task.pod_id, ThreadOp::Cancel), || {
                    format!("thread {thread_id} cancel denied by scope")
                })
            }
            Function::CompactThread { thread_id } => {
                let task = self.tasks.get(thread_id).ok_or_else(|| {
                    RejectReason::PreconditionFailed {
                        detail: format!("unknown thread {thread_id}"),
                    }
                })?;
                admission_check(scope.thread_op(&task.pod_id, ThreadOp::Compact), || {
                    format!("thread {thread_id} compact denied by scope")
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
            Function::CreateThread {
                pod_id, parent, ..
            } => {
                let effective_pod = pod_id.clone().unwrap_or_else(|| self.default_pod_id.clone());
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
                admission_check(scope.thread_op(&effective_pod, ThreadOp::Create), || {
                    format!("thread create in pod {effective_pod} denied by scope")
                })
            }
            Function::RunBehavior {
                pod_id,
                behavior_id,
                ..
            } => {
                let pod = self.pods.get(pod_id).ok_or_else(|| {
                    RejectReason::PreconditionFailed {
                        detail: format!("unknown pod `{pod_id}`"),
                    }
                })?;
                let behavior = pod.behaviors.get(behavior_id).ok_or_else(|| {
                    RejectReason::PreconditionFailed {
                        detail: format!(
                            "unknown behavior `{behavior_id}` under pod `{pod_id}`"
                        ),
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
                admission_check(scope.behavior_op(pod_id, BehaviorOp::Run), || {
                    format!("behavior {behavior_id} run denied by scope")
                })
            }
            Function::RebindThread { thread_id, .. } => {
                let task = self.tasks.get(thread_id).ok_or_else(|| {
                    RejectReason::PreconditionFailed {
                        detail: format!("unknown thread {thread_id}"),
                    }
                })?;
                admission_check(scope.thread_op(&task.pod_id, ThreadOp::Rebind), || {
                    format!("thread {thread_id} rebind denied by scope")
                })
                // Full validation (pod allowlist, MCP-host existence)
                // happens inside `apply_rebind` at launch; errors there
                // surface as a Function Error terminal routed through
                // the caller-link.
            }
            Function::BuiltinToolCall { name, .. } => {
                // Scope rubber-stamps for Phase 4b: the thread's
                // `tools_scope`-based evaluate path in `Thread::step`
                // already gated this call. Phase 4c moves that
                // evaluation here and removes the thread-side check.
                admission_check(scope.tool(name), || format!("tool {name} denied by scope"))
            }
            Function::McpToolUse { name, .. } => {
                admission_check(scope.tool(name), || format!("tool {name} denied by scope"))
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
            Function::RebindThread { thread_id, patch } => {
                // Pull correlation_id from the caller-link so the
                // ThreadBindingsChanged broadcast still tags its
                // originator. WS caller is the only current source;
                // other variants pass `None`.
                let correlation_id = match self.active_functions.get(&id) {
                    Some(entry) => match &entry.caller {
                        CallerLink::WsClient { correlation_id, .. } => correlation_id.clone(),
                        _ => None,
                    },
                    None => None,
                };
                match self.apply_rebind(&thread_id, patch, correlation_id, pending_io) {
                    Ok(()) => self.complete_function(
                        id,
                        FunctionOutcome::Success(FunctionTerminal::RebindThread(
                            crate::functions::RebindThreadTerminal::default(),
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
        let wire_error_for_ws_client =
            if let (FunctionOutcome::Error(err), CallerLink::WsClient { conn_id, correlation_id }) =
                (&outcome, &entry.caller)
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
        self.active_functions.iter().find_map(|(id, entry)| {
            match &entry.spec {
                Function::CompactThread { thread_id: t } if t == thread_id => Some(*id),
                _ => None,
            }
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
            {
                if t == thread_id && u == tool_use_id {
                    return Some(*id);
                }
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

        let Some(scope) = self.thread_scope(thread_id) else {
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
        let disposition = scope.tool(&name);

        // `dispatch_thread` is a model-facing alias for
        // `Function::CreateThread`. Deny stops at the tool layer;
        // Allow / AllowWithPrompt proceed with the CreateThread spec.
        if name == crate::tools::builtin_tools::DISPATCH_THREAD {
            self.register_dispatch_thread_tool(
                thread_id,
                op_id,
                tool_use_id,
                input,
                scope,
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
                let _ = self.register_function(spec, scope, caller);
                let fut = crate::runtime::io_dispatch::build_io_future(
                    self,
                    thread_id.to_string(),
                    io_request,
                );
                pending_io.push(fut);
            }
            crate::permission::Disposition::AllowWithPrompt => {
                // Register the Function with the IoRequest buffered.
                // Emit a PendingApproval event so the UI can prompt.
                // Approval resolution later rebuilds the future from
                // the buffered IoRequest.
                match self.register_function(spec, scope, caller) {
                    Ok(fn_id) => {
                        if let Some(entry) = self.active_functions.get_mut(&fn_id) {
                            entry.pending_approval_io = Some(io_request);
                        }
                        let approval_id = format!("ap-{tool_use_id}");
                        self.emit_pending_approval(
                            thread_id,
                            &approval_id,
                            &tool_use_id,
                            &name,
                            &input,
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            %thread_id, %tool_use_id, %name, error = ?e,
                            "AllowWithPrompt tool registration unexpectedly rejected"
                        );
                        let fut = crate::runtime::io_dispatch::build_io_future(
                            self,
                            thread_id.to_string(),
                            io_request,
                        );
                        pending_io.push(fut);
                    }
                }
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
    /// or immediate ack for async). `AllowWithPrompt` for
    /// `dispatch_thread` is not yet supported — treated as Allow with
    /// a warn log. The common case today is `Allow`; approval
    /// integration for Function-spawning tool aliases can land when
    /// there's a concrete use case.
    #[allow(clippy::too_many_arguments)]
    fn register_dispatch_thread_tool(
        &mut self,
        parent_thread_id: &str,
        op_id: crate::runtime::thread::OpId,
        tool_use_id: String,
        input: serde_json::Value,
        scope: PermissionScope,
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
        if matches!(disposition, crate::permission::Disposition::AllowWithPrompt) {
            tracing::warn!(
                parent = %parent_thread_id, %tool_use_id,
                "dispatch_thread has AllowWithPrompt disposition; approval integration \
                 for Function-spawning aliases is not yet implemented — proceeding as Allow"
            );
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
            let fn_id = match self.register_function_with_delivery(spec, scope, caller, delivery)
            {
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
                    },
                )
            }));
        } else {
            let delivery = FunctionDelivery::ToolResultFollowup {
                parent_thread_id: parent_thread_id.to_string(),
                parent_tool_use_id: tool_use_id.clone(),
            };
            let fn_id = match self.register_function_with_delivery(spec, scope, caller, delivery)
            {
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
                    },
                )
            }));
        }
    }

    /// Emit a `ThreadEvent::PendingApproval` into the thread's event
    /// stream. Called from the scheduler-side approval path, replacing
    /// the old thread-side emission (which happened inside
    /// `Thread::step` before the approval move).
    fn emit_pending_approval(
        &mut self,
        thread_id: &str,
        approval_id: &str,
        tool_use_id: &str,
        name: &str,
        input: &serde_json::Value,
    ) {
        let annotations = self.annotations_for(thread_id);
        let empty = crate::tools::mcp::ToolAnnotations::default();
        let ann = annotations.get(name).unwrap_or(&empty);
        let args_preview = {
            let s = serde_json::to_string(input).unwrap_or_default();
            if s.len() > 200 {
                let mut i = 200;
                while !s.is_char_boundary(i) && i > 0 {
                    i -= 1;
                }
                format!("{}…", &s[..i])
            } else {
                s
            }
        };
        let ev = crate::runtime::thread::ThreadEvent::PendingApproval {
            approval_id: approval_id.to_string(),
            tool_use_id: tool_use_id.to_string(),
            name: name.to_string(),
            args_preview,
            destructive: ann.is_destructive(),
            read_only: ann.is_read_only(),
        };
        self.router.dispatch_events(thread_id, vec![ev]);
    }

    /// Resolve a pending-approval Function given the client's
    /// `ApprovalDecision`. Routes the decision to the Function whose
    /// caller-link matches `(thread_id, approval_id's tool_use_id)`.
    ///
    /// - Approve: build and push the buffered IO future; narrow the
    ///   thread's `tools_scope` to Allow for this tool if `remember`.
    /// - Reject: complete the Function with `Cancelled(UserDenied)`
    ///   and push a synthetic denial so the thread integrates a
    ///   denied tool_result and continues.
    pub(super) fn resolve_tool_approval(
        &mut self,
        thread_id: &str,
        approval_id: &str,
        decision: whisper_agent_protocol::ApprovalChoice,
        remember: bool,
        pending_io: &mut futures::stream::FuturesUnordered<SchedulerFuture>,
    ) -> bool {
        // approval_id format is "ap-{tool_use_id}" — see emit_pending_approval.
        let tool_use_id = approval_id.strip_prefix("ap-").unwrap_or(approval_id);
        let Some(fn_id) = self.find_tool_function_for(thread_id, tool_use_id) else {
            return false;
        };
        let Some(entry) = self.active_functions.get_mut(&fn_id) else {
            return false;
        };
        let Some(io_request) = entry.pending_approval_io.take() else {
            // Function exists but isn't pending approval — caller sent
            // a stale decision. Ignore.
            return false;
        };
        let tool_name = match &entry.spec {
            Function::BuiltinToolCall { name, .. } | Function::McpToolUse { name, .. } => {
                name.clone()
            }
            _ => String::new(),
        };

        // Emit the ApprovalResolved event into the thread's event stream.
        let resolved_ev = crate::runtime::thread::ThreadEvent::ApprovalResolved {
            approval_id: approval_id.to_string(),
            decision,
            decided_by_conn: None,
        };
        self.router.dispatch_events(thread_id, vec![resolved_ev]);

        match decision {
            whisper_agent_protocol::ApprovalChoice::Approve => {
                // "Remember this approval" narrows the thread's
                // tools_scope so future calls to this tool skip the
                // prompt.
                if remember {
                    if let Some(task) = self.tasks.get_mut(thread_id) {
                        task.tools_scope.set_allow(tool_name.clone());
                        task.tool_allowlist.insert(tool_name.clone());
                        let allowlist_ev =
                            crate::runtime::thread::ThreadEvent::AllowlistChanged {
                                allowlist: task.tool_allowlist.iter().cloned().collect(),
                            };
                        self.router.dispatch_events(thread_id, vec![allowlist_ev]);
                    }
                }
                let fut = crate::runtime::io_dispatch::build_io_future(
                    self,
                    thread_id.to_string(),
                    io_request,
                );
                pending_io.push(fut);
                true
            }
            whisper_agent_protocol::ApprovalChoice::Reject => {
                // Synthesize a denial and complete the Function.
                let crate::runtime::thread::IoRequest::ToolCall {
                    op_id,
                    tool_use_id,
                    name,
                    ..
                } = io_request
                else {
                    return false;
                };
                let synthetic = make_denial_future(
                    thread_id.to_string(),
                    tool_use_id.clone(),
                    op_id,
                    name,
                );
                pending_io.push(synthetic);
                self.complete_function(
                    fn_id,
                    FunctionOutcome::Cancelled(crate::functions::CancelReason::UserDenied),
                    pending_io,
                );
                true
            }
        }
    }

    /// Build a best-effort PermissionScope from the thread's current
    /// state. Phase 4b: only `tools_scope` is populated meaningfully;
    /// other fields rubber-stamp. Future commits extend this to cover
    /// bindings and pod-level operations as they start being gated at
    /// registration.
    pub(super) fn thread_scope(&self, thread_id: &str) -> Option<PermissionScope> {
        let task = self.tasks.get(thread_id)?;
        let mut scope = PermissionScope::allow_all();
        scope.tools = task.tools_scope.clone();
        Some(scope)
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
            .filter_map(|(fn_id, entry)| {
                match entry.awaiting_child_thread_id.as_deref() {
                    Some(t) if t == thread_id => Some(*fn_id),
                    _ => None,
                }
            })
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
                let text =
                    crate::runtime::scheduler::compaction::extract_last_assistant_text(task);
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
            ThreadStateLabel::Cancelled => FunctionOutcome::Cancelled(
                crate::functions::CancelReason::ExplicitCancel,
            ),
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
                } if parent == thread_id => {
                    Some((*fn_id, entry.awaiting_child_thread_id.clone()))
                }
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
                    self.router.broadcast_task_list(
                        ServerToClient::ThreadStateChanged {
                            thread_id: child_id.clone(),
                            state: ThreadStateLabel::Cancelled,
                        },
                    );
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
        FunctionOutcome::Success(_) => Err(
            "dispatch_thread Function terminated with non-CreateThread payload".to_string(),
        ),
        FunctionOutcome::Error(err) => Err(format!(
            "dispatched child failed: {}",
            err.detail
        )),
        FunctionOutcome::Cancelled(reason) => Err(format!(
            "dispatched child cancelled ({:?})",
            reason
        )),
    }
}

/// Translate a scope-check `Disposition` into a `Result`. `Deny` rejects;
/// `Allow` and `AllowWithPrompt` both admit. The prompt dimension is
/// handled per-variant by the caller (tool Functions buffer the IO
/// request and emit a PendingApproval event; other variants treat
/// AllowWithPrompt as admit-and-proceed since they have no prompt UX).
fn admission_check(
    disp: crate::permission::Disposition,
    detail: impl FnOnce() -> String,
) -> Result<(), RejectReason> {
    if disp.admits() {
        Ok(())
    } else {
        Err(RejectReason::ScopeDenied { detail: detail() })
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
            },
        )
    })
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
            },
        )
    })
}
