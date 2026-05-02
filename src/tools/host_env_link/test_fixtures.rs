//! Test-only helpers for exercising consumers of [`LiveDaemonHandle`]
//! without standing up the full WS round-trip.
//!
//! Spawns a fake-daemon task that owns the receiver side of the
//! handle's command channel and replies to OpenSession / InvokeTool
//! per a caller-supplied factory. Counters on a shared [`FakeDaemonState`]
//! let the caller assert how many of each command flowed through.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use tokio::sync::{Mutex, mpsc, oneshot};
use whisper_agent_host_proto::{
    CallToolResult, DaemonCapabilities, ThreadContext, ThreadContextDelta,
};

use super::LiveDaemonHandle;
use super::connection::Command;

/// Outcome of a fake daemon's response to a single InvokeTool.
pub(crate) enum InvokeOutcome {
    /// Reply immediately with this result.
    Reply(Result<CallToolResult, String>),
    /// Stash the reply oneshot in `pending_invokes` and never reply.
    /// Lets a test exercise cancellation or disconnect mid-call.
    Stash,
}

#[derive(Default)]
pub(crate) struct FakeDaemonState {
    pub opens: AtomicUsize,
    pub invokes: AtomicUsize,
    pub closes: AtomicUsize,
    pub pending_invokes: Mutex<Vec<oneshot::Sender<Result<CallToolResult, String>>>>,
    /// `ThreadContext` carried by each received `OpenSession`, in
    /// arrival order. Lets phase 5b+ tests assert that the dispatcher
    /// is sending the right context per `(thread, binding)` pair.
    pub opened_contexts: Mutex<Vec<ThreadContext>>,
    /// `ThreadContextDelta` carried by each received `UpdateSession`,
    /// in arrival order. Lets phase 5c tests assert that mid-session
    /// edits flow through the wire.
    pub updates: Mutex<Vec<ThreadContextDelta>>,
}

pub(crate) struct FakeDaemonRig {
    pub handle: Arc<LiveDaemonHandle>,
    pub state: Arc<FakeDaemonState>,
    pub task: tokio::task::JoinHandle<()>,
}

/// Spawn a fake-daemon task driving a fresh command channel.
///
/// `result_factory(tool_name, arguments)` is called for every
/// InvokeTool; its return value decides how to reply. OpenSession
/// always succeeds. CloseSession is just counted.
///
/// The task exits when the last clone of `handle` is dropped (which
/// closes the command channel and wakes `recv` with `None`).
pub(crate) fn spawn_fake_daemon<F>(
    name: String,
    capabilities: DaemonCapabilities,
    mut result_factory: F,
) -> FakeDaemonRig
where
    F: FnMut(&str, &serde_json::Value) -> InvokeOutcome + Send + 'static,
{
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<Command>(64);
    let handle = Arc::new(LiveDaemonHandle::__for_test(name, capabilities, cmd_tx));
    let state = Arc::new(FakeDaemonState::default());
    let state_for_task = state.clone();
    let task = tokio::spawn(async move {
        while let Some(cmd) = cmd_rx.recv().await {
            match cmd {
                Command::OpenSession { ready, context, .. } => {
                    state_for_task.opens.fetch_add(1, Ordering::Relaxed);
                    state_for_task.opened_contexts.lock().await.push(context);
                    let _ = ready.send(Ok(()));
                }
                Command::InvokeTool {
                    tool_name,
                    arguments,
                    result,
                    ..
                } => {
                    state_for_task.invokes.fetch_add(1, Ordering::Relaxed);
                    match result_factory(&tool_name, &arguments) {
                        InvokeOutcome::Reply(r) => {
                            let _ = result.send(r);
                        }
                        InvokeOutcome::Stash => {
                            state_for_task.pending_invokes.lock().await.push(result);
                        }
                    }
                }
                Command::UpdateSession { context_delta, .. } => {
                    state_for_task.updates.lock().await.push(context_delta);
                }
                Command::CloseSession { .. } => {
                    state_for_task.closes.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });
    FakeDaemonRig {
        handle,
        state,
        task,
    }
}

impl LiveDaemonHandle {
    /// Test-only ctor that bypasses the WS handshake. Scoped to
    /// `host_env_link` (where `Command` is defined) so the test-fixture
    /// helpers in this module can call it without leaking `Command`
    /// outside the module tree.
    #[doc(hidden)]
    pub(in crate::tools::host_env_link) fn __for_test(
        name: String,
        capabilities: DaemonCapabilities,
        cmd_tx: mpsc::Sender<Command>,
    ) -> Self {
        Self {
            name,
            capabilities,
            cmd_tx,
            next_session_seq: AtomicU64::new(1),
        }
    }
}
