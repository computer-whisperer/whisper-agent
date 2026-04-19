//! Function registry on the scheduler — Phase 2's first concrete
//! consumer of the types from `crate::functions`.
//!
//! The registry is the scheduler's source of truth for in-flight
//! caller-initiated operations. Each entry is an [`ActiveFunction`] —
//! spec + scope + caller-link. `register_function` is the synchronous
//! accept/deny path; per-variant `execute_*` methods run the actual work
//! and produce the terminal.
//!
//! Today the only variant wired through is `Function::CancelThread`,
//! which is synchronous (state flip + cascade hooks). Subsequent commits
//! will add `CompactThread` (async, uses in-flight flags),
//! `BuiltinToolCall` / `McpToolUse` (async, approval path), and the
//! thread-lifecycle variants.

use futures::stream::FuturesUnordered;
use tracing::{debug, warn};
use whisper_agent_protocol::{ServerToClient, ThreadStateLabel};

use super::Scheduler;
use crate::functions::{
    CallerLink, Function, FunctionId, FunctionOutcome, FunctionTerminal, InternalOriginator,
    RejectReason,
};
use crate::permission::{PermissionScope, ThreadOp};
use crate::runtime::io_dispatch::SchedulerFuture;

/// Server-owned runtime state of an in-flight Function.
///
/// Phase 2 shape: minimal — enough to support the synchronous
/// `CancelThread` variant. Async Functions will grow progress/terminal
/// channels and a cancel handle as they land.
#[derive(Debug)]
// Fields are written at registration and read by later variants + the
// cancel-by-thread sweep; silence early dead-code until those consumers
// land in Commits 3+.
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

    /// Synchronous accept-or-deny registration. On accept the `ActiveFunctionEntry`
    /// enters the registry in Pending state and its `FunctionId` is returned.
    ///
    /// This is the one place scope + variant preconditions are checked.
    /// Anything that requires awaiting I/O happens later, in the variant's
    /// execute path, through the caller-link delivery channel.
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

    /// Variant-specific scope + precondition check. Called synchronously from
    /// `register_function` before admission.
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
                let disp = scope.thread_op(&task.pod_id, ThreadOp::Cancel);
                if !disp.admits() {
                    return Err(RejectReason::ScopeDenied {
                        detail: format!("thread {thread_id} cancel denied by scope"),
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

    /// Pop an admitted Function from the registry and execute it. For
    /// Phase 2's synchronous `CancelThread`, execution runs inline during
    /// this call — terminal fires before we return. Async variants
    /// landing later will keep the Function in the registry until their
    /// future completes.
    pub(super) fn execute_function(
        &mut self,
        id: FunctionId,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let entry = match self.active_functions.remove(&id) {
            Some(e) => e,
            None => {
                warn!(function_id = id, "execute_function: no such entry");
                return;
            }
        };
        let caller_tag = entry.caller.audit_tag();
        let outcome = match entry.spec {
            Function::CancelThread { thread_id } => {
                self.execute_cancel_thread(&thread_id, pending_io);
                FunctionOutcome::Success(FunctionTerminal::CancelThread)
            }
            _ => {
                // Variants that passed registration but aren't executable
                // yet — shouldn't hit this path because
                // `variant_precondition_check` rejects them, but belt &
                // suspenders.
                FunctionOutcome::Error(crate::functions::FunctionError {
                    kind: crate::functions::FunctionErrorKind::Execution,
                    detail: "variant not yet implemented".into(),
                })
            }
        };
        debug!(
            function_id = id,
            caller = %caller_tag,
            outcome = ?outcome,
            "Function terminal"
        );
        // Caller-link routing of the terminal is a no-op for Phase 2's
        // CancelThread (no wire-level correlation_id, no delivery
        // surface). Real routing lands with variants whose callers need
        // direct replies.
    }

    /// Cancel a thread by id: flip its state, broadcast, and run the
    /// existing lifecycle cascade (host-env teardown, behavior terminal
    /// hook, dispatch resolution, child cancellation). Equivalent to the
    /// pre-migration `ClientToServer::CancelThread` handler body; called
    /// by `Function::CancelThread`'s execution path.
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
        // If the cancelled thread is either a dispatched child (parent is
        // waiting on its final text) or a parent of still-running
        // dispatched children, run the dispatch-lifecycle hooks. These
        // are idempotent no-ops otherwise.
        self.resolve_pending_dispatch(thread_id, pending_io);
        self.cascade_cancel_dispatched_children(thread_id, pending_io);
    }
}

// `_` prefix silences the unused warning for `InternalOriginator` — it's
// imported so later variants can construct scheduler-internal caller-links
// without touching this file.
#[allow(dead_code)]
fn _keep_internal_originator_in_scope(_x: InternalOriginator) {}
