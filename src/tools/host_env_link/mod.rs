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
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use thiserror::Error;
use tokio::sync::{Notify, mpsc, oneshot};
use whisper_agent_host_proto::{
    CallId, CallToolResult, ContentBlock, DaemonCapabilities, GoodbyeReason, HostEnvSpec,
    HostEnvSpecKind, ProvisionPhase, SessionId, ThreadContext,
};
use whisper_agent_protocol::{HostEnvConfigurableSummary, HostEnvDaemonSummary};

pub use auth::{AdmittedDaemon, DaemonAuthState};
pub use endpoint::link_handler;

use connection::Command;

/// Combined state the `/v1/host_env_link` endpoint needs at upgrade
/// time. Bundled into one type so the axum router stores a single
/// `Arc` rather than two parallel `State`s — also makes it easy to
/// thread test fixtures (a stub inbox + a live registry) through
/// the upgrade path.
pub struct HostEnvLinkState {
    pub registry: Arc<LiveDaemonRegistry>,
    /// Scheduler inbox. The connection task forwards
    /// `Frame::PublishCredential` here as
    /// `SchedulerMsg::DaemonPublishCredential` so the scheduler can
    /// authorize and dispatch through the same `update_codex_auth`
    /// path the admin paste rotation uses.
    pub scheduler_inbox: mpsc::UnboundedSender<crate::runtime::scheduler::SchedulerMsg>,
}

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

    /// Host-env daemon rows for the client settings UI. Includes the
    /// union of admitted names and connected handles so an unexpected
    /// live handle is visible even if admission state changes later.
    pub fn daemon_summaries(&self) -> Vec<HostEnvDaemonSummary> {
        let now_ms = unix_millis_now();
        let inner = self.inner.lock().expect("registry mutex poisoned");
        let mut names: Vec<String> = inner
            .admitted
            .iter()
            .chain(inner.connected.keys())
            .cloned()
            .collect();
        names.sort();
        names.dedup();

        names
            .into_iter()
            .map(|name| {
                let admitted = inner.admitted.contains(&name);
                let handle = inner.connected.get(&name);
                match handle {
                    Some(handle) => {
                        let mut tools: Vec<String> = handle
                            .capabilities
                            .tools
                            .iter()
                            .map(|tool| tool.name.clone())
                            .collect();
                        tools.sort();
                        HostEnvDaemonSummary {
                            name,
                            admitted,
                            connected: true,
                            daemon_version: Some(handle.daemon_version.clone()),
                            protocol_version: Some(handle.protocol_version),
                            spec_kinds: handle
                                .capabilities
                                .spec_kinds
                                .iter()
                                .map(host_env_spec_kind_label)
                                .collect(),
                            tools,
                            max_concurrent_sessions: handle.capabilities.max_concurrent_sessions,
                            supports_background_tasks: handle
                                .capabilities
                                .supports_background_tasks,
                            session_configurables: handle
                                .capabilities
                                .session_configurables
                                .iter()
                                .map(|c| HostEnvConfigurableSummary {
                                    spec: c.spec.clone(),
                                    lifecycle: session_configurable_lifecycle_label(c.lifecycle)
                                        .into(),
                                })
                                .collect(),
                            last_active_ms_ago: Some(
                                (now_ms - handle.last_active_at_ms.load(Ordering::Acquire)).max(0)
                                    as u64,
                            ),
                        }
                    }
                    None => HostEnvDaemonSummary {
                        name,
                        admitted,
                        connected: false,
                        daemon_version: None,
                        protocol_version: None,
                        spec_kinds: Vec::new(),
                        tools: Vec::new(),
                        max_concurrent_sessions: None,
                        supports_background_tasks: false,
                        session_configurables: Vec::new(),
                        last_active_ms_ago: None,
                    },
                }
            })
            .collect()
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

    /// Insert a connected handle, evicting a stale prior connection if
    /// one is squatting on the name.
    ///
    /// Conflict policy (per `docs/design_host_env_protocol.md`
    /// §"Conflict policy"):
    /// - Slot empty → insert and return [`InsertOutcome::Inserted`].
    /// - Slot occupied and the existing handle has shown liveness
    ///   within `fresh_window` → return [`InsertOutcome::Rejected`].
    ///   Caller should send `Goodbye::NameAlreadyConnected`.
    /// - Slot occupied and the existing handle is stale → ask the
    ///   prior connection to shut down (so it sends `Goodbye::Superseded`
    ///   and tears down the WS), replace the registry entry with the
    ///   new handle, and return [`InsertOutcome::Inserted`].
    ///
    /// The "stale" path is what un-deadlocks the half-open case: the
    /// scheduler-side TCP can't tell a network-blip drop from a still-
    /// alive idle daemon, so we lean on the heartbeat (the connection
    /// task updates `last_active_at_ms` on every inbound frame and
    /// pings every `HEARTBEAT_INTERVAL`). Old enough = dead.
    fn insert_or_supersede(
        &self,
        name: String,
        handle: Arc<LiveDaemonHandle>,
        fresh_window: Duration,
    ) -> InsertOutcome {
        let evicted = {
            let mut inner = self.inner.lock().expect("registry mutex poisoned");
            match inner.connected.get(&name) {
                None => {
                    inner.connected.insert(name, handle);
                    self.change.notify_waiters();
                    return InsertOutcome::Inserted;
                }
                Some(existing) => {
                    let now = unix_millis_now();
                    let last = existing.last_active_at_ms.load(Ordering::Acquire);
                    let age_ms = (now - last).max(0) as u128;
                    if age_ms < fresh_window.as_millis() {
                        return InsertOutcome::Rejected {
                            existing_age_ms: age_ms.min(u64::MAX as u128) as u64,
                        };
                    }
                    // Stale. Replace under the same lock so a third
                    // concurrent connection sees the new entry, not a
                    // briefly-empty slot.
                    let evicted = inner.connected.insert(name, handle);
                    self.change.notify_waiters();
                    evicted
                }
            }
        };
        // Best-effort tell the evicted task to send `Goodbye::Superseded`
        // and exit. If its command channel is closed (already torn down)
        // or full, we don't care — the new handle is already in the
        // registry, and the evicted task's eventual `remove_connected_if`
        // is a no-op against the new Arc.
        if let Some(old) = evicted {
            old.try_send_supersede(
                GoodbyeReason::Superseded,
                "evicted: prior connection failed liveness check".into(),
            );
        }
        InsertOutcome::Inserted
    }

    /// Remove the registered handle for `name` *only if* it points at
    /// the same Arc as `handle`. Used by a connection task on exit so
    /// a task that was previously superseded doesn't yank out the new
    /// handle that took its slot.
    fn remove_connected_if(&self, name: &str, handle: &Arc<LiveDaemonHandle>) {
        let mut inner = self.inner.lock().expect("registry mutex poisoned");
        if let Some(current) = inner.connected.get(name)
            && Arc::ptr_eq(current, handle)
        {
            inner.connected.remove(name);
            self.change.notify_waiters();
        }
    }

    /// Test-only: insert a handle directly. Production code routes
    /// through `insert_or_supersede` from the connection task. Tests
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
        let mut inner = self.inner.lock().expect("registry mutex poisoned");
        if inner.connected.remove(name).is_some() {
            self.change.notify_waiters();
        }
    }
}

/// Result of [`LiveDaemonRegistry::insert_or_supersede`].
#[derive(Debug)]
pub(super) enum InsertOutcome {
    /// The new handle is now the registered entry. If a stale prior
    /// entry was evicted, it's already been told to shut down.
    Inserted,
    /// The slot is occupied by a connection that's still showing
    /// liveness; the caller should send `Goodbye::NameAlreadyConnected`
    /// to the freshly-arrived connection. `existing_age_ms` is for
    /// log context — how recently the existing connection last
    /// reported activity.
    Rejected { existing_age_ms: u64 },
}

pub(super) fn unix_millis_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn session_configurable_lifecycle_label(
    lifecycle: whisper_agent_host_proto::SessionConfigurableLifecycle,
) -> &'static str {
    match lifecycle {
        whisper_agent_host_proto::SessionConfigurableLifecycle::SpawnOnly => "spawn_only",
        whisper_agent_host_proto::SessionConfigurableLifecycle::LiveUpdate => "live_update",
    }
}

fn host_env_spec_kind_label(kind: &HostEnvSpecKind) -> String {
    match kind {
        HostEnvSpecKind::Landlock => "landlock",
        HostEnvSpecKind::Container => "container",
    }
    .to_string()
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
    daemon_version: String,
    protocol_version: u32,
    capabilities: DaemonCapabilities,
    cmd_tx: mpsc::Sender<Command>,
    next_session_seq: AtomicU64,
    /// Millis-since-Unix-epoch of the most recent inbound activity on
    /// this connection (any frame, including WS-level Pong). Read by
    /// the registry's conflict-resolution path to decide whether a
    /// colliding new connection should evict this one.
    last_active_at_ms: AtomicI64,
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

    /// Mark this handle as having seen inbound activity now. Called
    /// from the connection task on every received frame.
    pub(super) fn touch_active(&self) {
        self.last_active_at_ms
            .store(unix_millis_now(), Ordering::Release);
    }

    /// Best-effort send of [`Command::Supersede`] so the connection
    /// task knows to send a `Goodbye::Superseded` and exit. We use
    /// `try_send` because: (a) we're already holding the registry
    /// lock when we call this, so we can't `await`; (b) if the channel
    /// is full or closed the task is either already shutting down or
    /// can't keep up — either way the new handle is already registered,
    /// so eventual cleanup is correct without our help.
    pub(super) fn try_send_supersede(&self, reason: GoodbyeReason, message: String) {
        let _ = self.cmd_tx.try_send(Command::Supersede { reason, message });
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
    ///
    /// `attachments` is a sidecar of content blocks the scheduler
    /// resolved upstream from conversation state — populated only
    /// for tools whose input schema declares an `x-content-ref`
    /// parameter. Pass `vec![]` for tools that don't opt in.
    pub async fn invoke_tool(
        &self,
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
        attachments: Vec<ContentBlock>,
    ) -> Result<CallToolResult, LinkError> {
        let call_id = CallId(self.next_call_seq.fetch_add(1, Ordering::Relaxed));
        let (result_tx, result_rx) = oneshot::channel();
        let cmd = Command::InvokeTool {
            session_id: self.session_id.clone(),
            call_id,
            tool_name: tool_name.into(),
            arguments,
            attachments,
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
