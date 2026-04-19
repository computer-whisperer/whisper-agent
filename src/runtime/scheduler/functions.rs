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
use tracing::{debug, warn};
use whisper_agent_protocol::{ServerToClient, ThreadStateLabel};

use super::Scheduler;
use crate::functions::{
    CallerLink, Function, FunctionId, FunctionOutcome, FunctionTerminal, InFlightOps,
    RejectReason,
};
use crate::permission::{PermissionScope, ThreadOp};
use crate::runtime::io_dispatch::SchedulerFuture;

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
        self.variant_precondition_check(&spec, &scope)?;
        let id = self.next_function_id();
        let entry = ActiveFunctionEntry {
            id,
            spec,
            scope,
            caller,
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
            // Other variants are declared but not yet consumed; they'll
            // get their precondition logic as they're migrated in later
            // commits.
            _ => Err(RejectReason::InvalidSpec {
                detail: "Function variant not yet implemented".into(),
            }),
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
                );
            }
            Function::CompactThread { thread_id } => {
                self.launch_compact_thread(&thread_id, pending_io);
                // Function stays in the registry until
                // `finalize_pending_compaction` calls `complete_function`.
            }
            _ => {
                // Variants that passed registration but aren't launchable
                // yet — shouldn't hit this path because
                // `variant_precondition_check` rejects them, but belt &
                // suspenders.
                self.complete_function(
                    id,
                    FunctionOutcome::Error(crate::functions::FunctionError {
                        kind: crate::functions::FunctionErrorKind::Execution,
                        detail: "variant not yet implemented".into(),
                    }),
                );
            }
        }
    }

    /// Remove the entry from `active_functions` and log/audit the
    /// terminal. Caller-link routing of the terminal payload is a no-op
    /// for Phase-2/3 operations whose caller-surfaces don't require a
    /// direct reply (CancelThread emits nothing; CompactThread's client
    /// UX is covered by the existing `ThreadCompacted` broadcast).
    /// Terminal routing over the WS wire lands with the tool-call
    /// variants in Commit 4.
    pub(super) fn complete_function(&mut self, id: FunctionId, outcome: FunctionOutcome) {
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
        // If the cancelled thread is either a dispatched child (parent
        // is waiting on its final text) or a parent of still-running
        // dispatched children, run the dispatch-lifecycle hooks. These
        // are idempotent no-ops otherwise.
        self.resolve_pending_dispatch(thread_id, pending_io);
        self.cascade_cancel_dispatched_children(thread_id, pending_io);
    }
}

/// Translate a scope-check `Disposition` into a `Result`. Phase-2/3
/// callers are happy with "admit ⇒ run, deny ⇒ reject"; the prompt flow
/// required for `AllowWithPrompt` lands in Commit 4 alongside the
/// tool-call variants that actually have a prompt UX.
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
