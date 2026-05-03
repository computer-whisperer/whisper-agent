//! Host-env protocol v2 — server side. Daemons dial in to
//! `/v1/host_env_link` over HTTPS and the connection multiplexes
//! every per-thread session that names this daemon as a provider.
//!
//! See `docs/design_host_env_protocol.md` for the architecture and
//! the [`whisper_agent_host_proto`] crate for the wire types.
//!
//! This module owns:
//! - [`auth::DaemonAuthState`] / [`auth::require_daemon_auth`] —
//!   bearer admission against `[[auth.daemons]]`.
//! - [`endpoint::link_handler`] — the axum WebSocket upgrade.
//! - [`LiveDaemonRegistry`] — name → live connection handle plus the
//!   admitted-offline set (the names a v2_ws catalog entry knows
//!   about, regardless of current connection state).
//! - [`LiveDaemonHandle`] / [`SessionHandle`] — what consumers hold
//!   to drive sessions over the wire.
//!
//! Phase 2b scope: the wire is end-to-end functional but no scheduler
//! code path yet binds threads to v2 daemons. That's phase 4. Tests
//! exercise the wire directly via an in-process daemon.

pub mod auth;
mod connection;
pub mod endpoint;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use thiserror::Error;
use tokio::sync::{Notify, mpsc, oneshot};
use whisper_agent_host_proto::{
    CallId, CallToolResult, DaemonCapabilities, HostEnvSpec, ProvisionPhase, SessionId,
    ThreadContext,
};

pub use auth::{AdmittedDaemon, DaemonAuthState};
pub use endpoint::link_handler;

use connection::Command;

/// Errors surfaced through the v2 host-env link to consumers.
#[derive(Debug, Error)]
pub enum LinkError {
    /// No daemon by that name is currently connected. Includes the
    /// "not admitted" case (catalog has no v2_ws entry by that name)
    /// and the "admitted but offline" case (no live WebSocket).
    #[error("daemon `{0}` not connected")]
    NotConnected(String),
    /// Daemon disconnected mid-operation.
    #[error("daemon `{0}` disconnected")]
    Disconnected(String),
    /// Daemon refused to provision the session. Carries the phase
    /// the failure was reported in plus the daemon's free-text reason.
    #[error("session provision failed at {phase:?}: {message}")]
    ProvisionFailed {
        phase: ProvisionPhase,
        message: String,
    },
    /// Daemon's tool call returned an error or the session is gone.
    #[error("call failed: {0}")]
    CallFailed(String),
}

/// Server-side registry of v2 host-env daemons.
///
/// Two membership tiers, distinct by design:
/// - **`admitted`** — daemon names admitted by `[[auth.daemons]]`.
///   Stable across restarts; reflects what the operator has configured.
/// - **`connected`** — daemon names currently holding a live WS to
///   this scheduler. Comes and goes as daemons dial in / drop out.
///
/// A pod's `[[allow.host_env]]` references a name that must be in
/// `admitted`. Whether a thread can use it *right now* depends on
/// `connected`; that's the per-thread disconnect-policy surface
/// (phase 4 will wire it in).
#[derive(Default)]
pub struct LiveDaemonRegistry {
    inner: Mutex<RegistryInner>,
    /// Fires every time a `connected` insertion or removal happens.
    /// Lets callers wait for a daemon to dial in (or vacate a name)
    /// without polling. Notify supports many waiters; each notify_one
    /// wakes one waiter, so consumers should re-check state in a loop.
    change: Notify,
}

#[derive(Default)]
struct RegistryInner {
    admitted: HashSet<String>,
    connected: HashMap<String, Arc<LiveDaemonHandle>>,
}

impl LiveDaemonRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed the admitted set from the configured `[[auth.daemons]]`
    /// names. Idempotent. Called at startup; runtime mutation lands
    /// when v2 catalog edits become a thing.
    pub fn admit_names<I: IntoIterator<Item = String>>(&self, names: I) {
        let mut inner = self.inner.lock().expect("registry mutex poisoned");
        inner.admitted.extend(names);
    }

    /// Return the connection handle for a daemon if currently online.
    pub fn handle(&self, name: &str) -> Option<Arc<LiveDaemonHandle>> {
        self.inner
            .lock()
            .expect("registry mutex poisoned")
            .connected
            .get(name)
            .cloned()
    }

    pub fn is_admitted(&self, name: &str) -> bool {
        self.inner
            .lock()
            .expect("registry mutex poisoned")
            .admitted
            .contains(name)
    }

    pub fn is_connected(&self, name: &str) -> bool {
        self.inner
            .lock()
            .expect("registry mutex poisoned")
            .connected
            .contains_key(name)
    }

    /// Snapshot of (admitted, connected) name lists. Sorted output for
    /// deterministic logging / tests.
    pub fn snapshot(&self) -> RegistrySnapshot {
        let inner = self.inner.lock().expect("registry mutex poisoned");
        let mut admitted: Vec<String> = inner.admitted.iter().cloned().collect();
        admitted.sort();
        let mut connected: Vec<String> = inner.connected.keys().cloned().collect();
        connected.sort();
        RegistrySnapshot {
            admitted,
            connected,
        }
    }

    /// Block until `name` has a live connection. Returns the handle
    /// once available. Used by tests; also useful for "wait for daemon
    /// to come back" patterns the disconnect policy will need.
    ///
    /// Cancellation-safe: drop the future and no state changes.
    pub async fn wait_for_connection(&self, name: &str) -> Arc<LiveDaemonHandle> {
        loop {
            // Subscribe before checking state so we can't miss the
            // notify that fires between check and await.
            let notified = self.change.notified();
            if let Some(h) = self.handle(name) {
                return h;
            }
            notified.await;
        }
    }

    /// Insert a connected handle. Returns `false` if a handle with
    /// the same name is already registered (caller should send
    /// `Goodbye::NameAlreadyConnected` and close).
    fn try_insert_connected(&self, name: String, handle: Arc<LiveDaemonHandle>) -> bool {
        let mut inner = self.inner.lock().expect("registry mutex poisoned");
        if inner.connected.contains_key(&name) {
            return false;
        }
        inner.connected.insert(name, handle);
        self.change.notify_waiters();
        true
    }

    /// Remove a connected handle on disconnect. No-op if absent.
    fn remove_connected(&self, name: &str) {
        let mut inner = self.inner.lock().expect("registry mutex poisoned");
        if inner.connected.remove(name).is_some() {
            self.change.notify_waiters();
        }
    }

    /// Test-only: insert a handle directly. Production code routes
    /// through `try_insert_connected` from the connection task. Tests
    /// for the dispatcher's reconnect retry loop need to swap in fresh
    /// handles without standing up the full WS handshake.
    #[cfg(test)]
    pub(crate) fn insert_for_test(&self, name: String, handle: Arc<LiveDaemonHandle>) {
        let mut inner = self.inner.lock().expect("registry mutex poisoned");
        inner.connected.insert(name, handle);
        self.change.notify_waiters();
    }

    /// Test-only: drop a handle directly. Pair with `insert_for_test`
    /// for tests that simulate disconnect.
    #[cfg(test)]
    pub(crate) fn remove_for_test(&self, name: &str) {
        self.remove_connected(name);
    }
}

/// Sorted snapshot of registry state for logging / introspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrySnapshot {
    pub admitted: Vec<String>,
    pub connected: Vec<String>,
}

/// Handle on a connected v2 daemon. Cheap to clone (`Arc`).
pub struct LiveDaemonHandle {
    name: String,
    capabilities: DaemonCapabilities,
    cmd_tx: mpsc::Sender<Command>,
    next_session_seq: AtomicU64,
}

impl fmt::Debug for LiveDaemonHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiveDaemonHandle")
            .field("name", &self.name)
            .field("tool_count", &self.capabilities.tools.len())
            .finish()
    }
}

impl LiveDaemonHandle {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn capabilities(&self) -> &DaemonCapabilities {
        &self.capabilities
    }

    /// Open a new session against this daemon. Returns once the daemon
    /// replies with [`whisper_agent_host_proto::Frame::SessionReady`]
    /// (success) or [`whisper_agent_host_proto::Frame::SessionFailed`]
    /// (error).
    ///
    /// `thread_id` is for daemon-side logging only — the scheduler's
    /// thread id is opaque to the daemon, but surfacing it in the
    /// daemon's logs makes correlated debugging vastly easier.
    pub async fn open_session(
        self: &Arc<Self>,
        thread_id: impl Into<String>,
        spec: HostEnvSpec,
        context: ThreadContext,
    ) -> Result<SessionHandle, LinkError> {
        let session_id = self.allocate_session_id();
        let (ready_tx, ready_rx) = oneshot::channel();
        let cmd = Command::OpenSession {
            session_id: session_id.clone(),
            thread_id: thread_id.into(),
            spec,
            context,
            ready: ready_tx,
        };
        if self.cmd_tx.send(cmd).await.is_err() {
            return Err(LinkError::Disconnected(self.name.clone()));
        }
        match ready_rx.await {
            Ok(Ok(())) => Ok(SessionHandle {
                daemon_name: self.name.clone(),
                session_id,
                cmd_tx: self.cmd_tx.clone(),
                next_call_seq: AtomicU64::new(1),
            }),
            Ok(Err((phase, message))) => Err(LinkError::ProvisionFailed { phase, message }),
            Err(_) => Err(LinkError::Disconnected(self.name.clone())),
        }
    }

    fn allocate_session_id(&self) -> SessionId {
        let n = self.next_session_seq.fetch_add(1, Ordering::Relaxed);
        // Daemon name + monotonic counter. Scheduler-assigned per the
        // protocol contract; the format is opaque to the daemon.
        SessionId::new(format!("{}-{n:08x}", self.name))
    }
}

/// A live session against a daemon. Drops cleanly via
/// [`Drop`] — the session is told to close in a fire-and-forget send.
///
/// Phase 2b's API is minimal: open a session, invoke a tool, await
/// the final result. Streaming chunks, mid-flight `update_context`,
/// and background-task interaction land in later phases.
pub struct SessionHandle {
    daemon_name: String,
    session_id: SessionId,
    cmd_tx: mpsc::Sender<Command>,
    next_call_seq: AtomicU64,
}

impl fmt::Debug for SessionHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionHandle")
            .field("daemon", &self.daemon_name)
            .field("session_id", &self.session_id)
            .finish()
    }
}

impl SessionHandle {
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    /// Invoke a tool on this session. Returns the terminal
    /// [`CallToolResult`] when the daemon emits `ToolFinal`.
    /// Streaming chunks are silently discarded in phase 2b.
    pub async fn invoke_tool(
        &self,
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, LinkError> {
        let call_id = CallId(self.next_call_seq.fetch_add(1, Ordering::Relaxed));
        let (result_tx, result_rx) = oneshot::channel();
        let cmd = Command::InvokeTool {
            session_id: self.session_id.clone(),
            call_id,
            tool_name: tool_name.into(),
            arguments,
            result: result_tx,
        };
        if self.cmd_tx.send(cmd).await.is_err() {
            return Err(LinkError::Disconnected(self.daemon_name.clone()));
        }
        match result_rx.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(message)) => Err(LinkError::CallFailed(message)),
            Err(_) => Err(LinkError::Disconnected(self.daemon_name.clone())),
        }
    }
}

impl SessionHandle {
    /// Push a [`whisper_agent_host_proto::ThreadContextDelta`] to the
    /// daemon. Fire-and-forget: the daemon doesn't ack, and the
    /// protocol's failure surface is the next tool call against this
    /// session (which would either succeed under the new context or
    /// surface the failure via its own ToolFinal).
    ///
    /// Returns [`LinkError::Disconnected`] if the channel to the
    /// connection task is closed (daemon already torn down) so the
    /// caller can decide whether to log or retry.
    pub fn update_context(
        &self,
        delta: whisper_agent_host_proto::ThreadContextDelta,
    ) -> Result<(), LinkError> {
        let cmd = Command::UpdateSession {
            session_id: self.session_id.clone(),
            context_delta: delta,
        };
        self.cmd_tx
            .try_send(cmd)
            .map_err(|_| LinkError::Disconnected(self.daemon_name.clone()))
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        // Best-effort fire-and-forget: tell the connection task to
        // close the session on the daemon side. If the channel is
        // full or closed we drop silently — the connection task's
        // own teardown also tears sessions down.
        let _ = self.cmd_tx.try_send(Command::CloseSession {
            session_id: self.session_id.clone(),
        });
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod test_fixtures;
