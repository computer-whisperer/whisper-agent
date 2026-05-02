//! Per-session worker registry.
//!
//! Owns the [`Worker`] children spawned in response to each
//! [`whisper_agent_host_proto::Frame::OpenSession`]. Lookup is by
//! [`SessionId`]; closing a session drops the [`Worker`], which kills
//! the child via `kill_on_drop`.
//!
//! No locking on the registry itself in this phase: all access is
//! from the single connection-task event loop. If we later parallelize
//! tool dispatch (e.g. spawn each `InvokeTool` to its own task) we'll
//! either pin the registry to that task and use a channel, or move
//! to an `Arc<Mutex<_>>` — but the data the consumer needs from the
//! registry to dispatch a tool call is small and cheap to clone.

use std::collections::HashMap;
use std::sync::Arc;

use whisper_agent_host_proto::{SessionId, ThreadContext, ThreadContextDelta};

use crate::worker::Worker;

/// State the daemon retains per live session: the worker handle and
/// the session-level [`ThreadContext`] (denylist, runas, env, etc.).
/// Later phases will accrete background-task state and active call
/// cancellations.
pub struct Session {
    pub worker: Arc<Worker>,
    pub context: ThreadContext,
}

#[derive(Default)]
pub struct SessionRegistry {
    by_id: HashMap<SessionId, Session>,
}

/// What the dispatcher needs to make one tool call against a session
/// without holding a registry borrow across the await: a clone of the
/// worker handle (cheap — just an `Arc` + an `mpsc::Sender`) plus a
/// snapshot of the session's [`ThreadContext`] so the dispatch task
/// can apply denylist / argument-overrides without re-locking the
/// registry.
#[derive(Clone)]
pub struct DispatchTarget {
    pub worker: Arc<Worker>,
    pub context: ThreadContext,
}

impl SessionRegistry {
    pub fn insert(&mut self, id: SessionId, session: Session) {
        self.by_id.insert(id, session);
    }

    pub fn remove(&mut self, id: &SessionId) -> Option<Session> {
        self.by_id.remove(id)
    }

    pub fn dispatch_target(&self, id: &SessionId) -> Option<DispatchTarget> {
        self.by_id.get(id).map(|s| DispatchTarget {
            worker: s.worker.clone(),
            context: s.context.clone(),
        })
    }

    /// Apply a [`ThreadContextDelta`] to the session's stored context.
    /// Returns `false` if no session by that id exists (caller decides
    /// whether that's a warn or a no-op).
    pub fn apply_context_delta(&mut self, id: &SessionId, delta: &ThreadContextDelta) -> bool {
        match self.by_id.get_mut(id) {
            Some(s) => {
                s.context.apply_delta(delta);
                true
            }
            None => false,
        }
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Drain every session, returning the workers in arbitrary order.
    /// Used at shutdown so the caller can await each worker's exit
    /// (or just drop them and rely on `kill_on_drop`).
    pub fn drain(&mut self) -> impl Iterator<Item = (SessionId, Session)> + '_ {
        self.by_id.drain()
    }
}
