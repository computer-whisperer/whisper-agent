//! Per-WebSocket connection task.
//!
//! Owns the WebSocket and a command channel from the
//! [`crate::tools::host_env_link::LiveDaemonHandle`]. Runs the
//! Hello/Welcome handshake, then a select! loop multiplexing
//! commands-out (consumer → wire) with frames-in (wire → consumer)
//! across many concurrent sessions and per-session calls.
//!
//! Demux state is per-session: each [`SessionState`] holds the
//! pending OpenSession reply channel and a map of in-flight call
//! reply channels. When a frame arrives, we look up the right
//! one-shot and forward the result.
//!
//! On exit (any reason): unregister from the
//! [`crate::tools::host_env_link::LiveDaemonRegistry`] so future
//! `handle()` lookups for this name return `None`. Pending
//! oneshots get dropped, which surfaces as `LinkError::Disconnected`
//! to consumers waiting on them.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Context, anyhow, bail};
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt, stream::SplitSink, stream::SplitStream};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};
use whisper_agent_host_proto::{
    CallId, CallToolResult, DaemonCapabilities, Frame, GoodbyeReason, HostEnvSpec,
    PROTOCOL_VERSION, ProvisionPhase, SessionId, ThreadContext,
};

use super::{LiveDaemonHandle, LiveDaemonRegistry};

/// Bound on the consumer-→-task command channel. Small because
/// commands are control-plane (open/invoke/close), not bulk data.
/// A consumer that floods the channel deserves backpressure.
const COMMAND_CHANNEL_BOUND: usize = 64;

/// Commands the consumer-side handles send to the connection task.
/// The task is the sole writer to the WebSocket; consumers never
/// touch frame bytes.
pub(super) enum Command {
    OpenSession {
        session_id: SessionId,
        thread_id: String,
        spec: HostEnvSpec,
        context: ThreadContext,
        ready: oneshot::Sender<Result<(), (ProvisionPhase, String)>>,
    },
    InvokeTool {
        session_id: SessionId,
        call_id: CallId,
        tool_name: String,
        arguments: serde_json::Value,
        result: oneshot::Sender<Result<CallToolResult, String>>,
    },
    /// Apply a [`whisper_agent_host_proto::ThreadContextDelta`] to a
    /// live session. Fire-and-forget: the daemon doesn't ack — the
    /// scheduler discovers application failures only at the next tool
    /// call (which would either succeed under the new context or
    /// surface the failure via its own ToolFinal).
    UpdateSession {
        session_id: SessionId,
        context_delta: whisper_agent_host_proto::ThreadContextDelta,
    },
    CloseSession {
        session_id: SessionId,
    },
}

/// Per-session demux state held by the connection task.
struct SessionState {
    /// Set while the OpenSession is in flight; consumed by either
    /// SessionReady or SessionFailed.
    pending_open: Option<oneshot::Sender<Result<(), (ProvisionPhase, String)>>>,
    /// In-flight tool calls awaiting their terminal `ToolFinal`.
    /// Cleared on `ToolFinal` arrival or on session termination.
    pending_calls: HashMap<CallId, oneshot::Sender<Result<CallToolResult, String>>>,
}

/// Drive a connection from initial handshake through disconnect.
///
/// Caller (the WS upgrade handler) provides:
/// - `name`: daemon identity from the bearer-resolved
///   [`super::AdmittedDaemon`].
/// - `socket`: the freshly-upgraded WebSocket.
/// - `registry`: the [`LiveDaemonRegistry`] this connection should
///   register itself in on Hello-success and unregister from on exit.
///
/// The function consumes everything and returns once the connection
/// is fully torn down.
pub(super) async fn run_connection(
    name: String,
    socket: WebSocket,
    registry: Arc<LiveDaemonRegistry>,
) {
    let (mut sink, mut stream) = socket.split();
    let capabilities = match handshake(&name, &mut sink, &mut stream).await {
        Ok(c) => c,
        Err(e) => {
            warn!(daemon = %name, error = %e, "host-env link handshake failed");
            return;
        }
    };

    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>(COMMAND_CHANNEL_BOUND);
    let handle = Arc::new(LiveDaemonHandle {
        name: name.clone(),
        capabilities,
        cmd_tx,
        next_session_seq: AtomicU64::new(1),
    });

    if !registry.try_insert_connected(name.clone(), handle.clone()) {
        // A daemon with this name is already registered. Reject the
        // newcomer with `Goodbye` so the operator notices and don't
        // touch the existing slot. (The existing-handle eviction
        // policy described in the design doc is phase-3 work.)
        warn!(
            daemon = %name,
            "rejecting v2 daemon connection: name already connected"
        );
        let _ = send_frame(
            &mut sink,
            Frame::Goodbye {
                reason: GoodbyeReason::NameAlreadyConnected,
                message: format!("daemon `{name}` already connected"),
            },
        )
        .await;
        return;
    }

    info!(
        daemon = %name,
        tools = handle.capabilities.tools.len(),
        spec_kinds = ?handle.capabilities.spec_kinds,
        background_tasks = handle.capabilities.supports_background_tasks,
        "v2 host-env daemon connected"
    );

    // Drop our local handle reference so once `event_loop` finishes
    // and we leave this scope, the registry's clone is the only one
    // left — letting it drop naturally on `remove_connected` below.
    drop(handle);

    event_loop(&name, &mut sink, &mut stream, cmd_rx).await;

    registry.remove_connected(&name);
    info!(daemon = %name, "v2 host-env daemon disconnected");
}

/// Wait for `Hello`, validate it, send `Welcome`. Errors send the
/// matching `Goodbye` before returning so the daemon sees a clean
/// reason rather than a bare socket close.
async fn handshake(
    name: &str,
    sink: &mut SplitSink<WebSocket, Message>,
    stream: &mut SplitStream<WebSocket>,
) -> anyhow::Result<DaemonCapabilities> {
    let frame = recv_frame(stream)
        .await?
        .ok_or_else(|| anyhow!("daemon closed before sending Hello"))?;
    let (daemon_version, daemon_proto_version, capabilities) = match frame {
        Frame::Hello {
            daemon_version,
            protocol_version,
            capabilities,
        } => (daemon_version, protocol_version, capabilities),
        other => {
            let kind = frame_kind(&other);
            let _ = send_frame(
                sink,
                Frame::Goodbye {
                    reason: GoodbyeReason::Other,
                    message: format!("expected Hello, got {kind:?}"),
                },
            )
            .await;
            bail!("expected Hello, got {kind:?}");
        }
    };
    if daemon_proto_version != PROTOCOL_VERSION {
        let _ = send_frame(
            sink,
            Frame::Goodbye {
                reason: GoodbyeReason::ProtocolMismatch,
                message: format!(
                    "scheduler protocol_version={PROTOCOL_VERSION}, daemon protocol_version={daemon_proto_version}",
                ),
            },
        )
        .await;
        bail!(
            "protocol_version mismatch: daemon={daemon_proto_version}, scheduler={PROTOCOL_VERSION}",
        );
    }
    send_frame(
        sink,
        Frame::Welcome {
            scheduler_version: env!("CARGO_PKG_VERSION").into(),
            protocol_version: PROTOCOL_VERSION,
        },
    )
    .await
    .context("send Welcome")?;
    info!(
        daemon = name,
        daemon_version,
        protocol_version = daemon_proto_version,
        "v2 host-env daemon authenticated"
    );
    Ok(capabilities)
}

async fn event_loop(
    name: &str,
    sink: &mut SplitSink<WebSocket, Message>,
    stream: &mut SplitStream<WebSocket>,
    mut cmd_rx: mpsc::Receiver<Command>,
) {
    let mut sessions: HashMap<SessionId, SessionState> = HashMap::new();
    loop {
        tokio::select! {
            cmd = cmd_rx.recv() => match cmd {
                Some(c) => {
                    if let Err(e) = handle_command(c, &mut sessions, sink).await {
                        warn!(daemon = %name, error = %e, "send failed; closing connection");
                        return;
                    }
                }
                None => {
                    // All consumer handles dropped — the daemon is no
                    // longer referenced by anyone. Tell it goodbye and
                    // close. (In practice this won't happen until
                    // the LiveDaemonRegistry is itself dropped — which
                    // means server shutdown.)
                    let _ = send_frame(sink, Frame::Goodbye {
                        reason: GoodbyeReason::ServerShutdown,
                        message: "scheduler closing connection".into(),
                    }).await;
                    return;
                }
            },
            msg = stream.next() => match msg {
                Some(Ok(Message::Binary(bytes))) => match Frame::decode_cbor(&bytes) {
                    Ok(frame) => handle_frame(name, frame, &mut sessions),
                    Err(e) => {
                        warn!(daemon = %name, error = %e, "frame decode failed; closing");
                        let _ = send_frame(sink, Frame::Goodbye {
                            reason: GoodbyeReason::Other,
                            message: format!("frame decode failed: {e}"),
                        }).await;
                        return;
                    }
                },
                Some(Ok(Message::Close(_))) => {
                    debug!(daemon = %name, "daemon sent close frame");
                    return;
                }
                Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => {
                    // axum auto-replies pongs; nothing to do.
                }
                Some(Ok(Message::Text(_))) => {
                    warn!(daemon = %name, "unexpected text frame on host-env link; closing");
                    let _ = send_frame(sink, Frame::Goodbye {
                        reason: GoodbyeReason::Other,
                        message: "text frames not accepted on host-env link".into(),
                    }).await;
                    return;
                }
                Some(Err(e)) => {
                    warn!(daemon = %name, error = %e, "ws receive error; closing");
                    return;
                }
                None => return,
            }
        }
    }
}

async fn handle_command(
    cmd: Command,
    sessions: &mut HashMap<SessionId, SessionState>,
    sink: &mut SplitSink<WebSocket, Message>,
) -> anyhow::Result<()> {
    match cmd {
        Command::OpenSession {
            session_id,
            thread_id,
            spec,
            context,
            ready,
        } => {
            sessions.insert(
                session_id.clone(),
                SessionState {
                    pending_open: Some(ready),
                    pending_calls: HashMap::new(),
                },
            );
            send_frame(
                sink,
                Frame::OpenSession {
                    session_id,
                    thread_id,
                    spec,
                    context,
                },
            )
            .await
        }
        Command::InvokeTool {
            session_id,
            call_id,
            tool_name,
            arguments,
            result,
        } => {
            match sessions.get_mut(&session_id) {
                Some(state) => {
                    state.pending_calls.insert(call_id, result);
                }
                None => {
                    let _ = result.send(Err(format!("no such session `{session_id}`")));
                    return Ok(());
                }
            }
            send_frame(
                sink,
                Frame::InvokeTool {
                    session_id,
                    call_id,
                    tool_name,
                    arguments,
                },
            )
            .await
        }
        Command::UpdateSession {
            session_id,
            context_delta,
        } => {
            // No demux state to update — the scheduler-side store is
            // the source of truth, and the daemon-side application is
            // fire-and-forget. We do skip sending if the session isn't
            // known to this connection (likely already torn down) so a
            // stale UpdateSession after CloseSession doesn't reach the
            // daemon.
            if !sessions.contains_key(&session_id) {
                return Ok(());
            }
            send_frame(
                sink,
                Frame::UpdateSession {
                    session_id,
                    context_delta,
                },
            )
            .await
        }
        Command::CloseSession { session_id } => {
            // Drop any pending call senders so consumers see the
            // session as gone immediately, even before the daemon's
            // SessionClosed arrives.
            if let Some(state) = sessions.remove(&session_id) {
                drop(state);
            } else {
                // Session never opened or already closed — nothing
                // to send.
                return Ok(());
            }
            send_frame(sink, Frame::CloseSession { session_id }).await
        }
    }
}

fn handle_frame(name: &str, frame: Frame, sessions: &mut HashMap<SessionId, SessionState>) {
    match frame {
        Frame::SessionReady { session_id } => {
            if let Some(state) = sessions.get_mut(&session_id) {
                if let Some(tx) = state.pending_open.take() {
                    let _ = tx.send(Ok(()));
                }
            } else {
                warn!(daemon = %name, %session_id, "SessionReady for unknown session");
            }
        }
        Frame::SessionFailed {
            session_id,
            phase,
            message,
        } => {
            if let Some(mut state) = sessions.remove(&session_id) {
                if let Some(tx) = state.pending_open.take() {
                    let _ = tx.send(Err((phase, message.clone())));
                }
                // Pending calls (none expected pre-Ready, but guard) get
                // dropped → consumers see Disconnected.
                drop(state);
            } else {
                warn!(daemon = %name, %session_id, "SessionFailed for unknown session");
            }
        }
        Frame::SessionClosed { session_id, reason } => {
            if let Some(state) = sessions.remove(&session_id) {
                debug!(daemon = %name, %session_id, ?reason, "session closed");
                drop(state);
            }
        }
        Frame::ToolFinal {
            session_id,
            call_id,
            result,
        } => {
            if let Some(state) = sessions.get_mut(&session_id) {
                if let Some(tx) = state.pending_calls.remove(&call_id) {
                    let _ = tx.send(Ok(result));
                } else {
                    warn!(daemon = %name, %session_id, %call_id, "ToolFinal for unknown call");
                }
            } else {
                warn!(daemon = %name, %session_id, "ToolFinal for unknown session");
            }
        }
        Frame::ToolChunk { .. } => {
            // Phase 2b doesn't surface streaming chunks. They're not
            // dropped on purpose — the model still gets the full result
            // via ToolFinal. Streaming consumer API lands later.
        }
        Frame::HookEvent { .. }
        | Frame::BackgroundTaskUpdate { .. }
        | Frame::BackgroundTaskList { .. } => {
            // Background tasks and hooks are part of phases 5 / 6.
            // Acknowledge via debug log so daemons sending them
            // ahead of schedule don't get silently dropped.
            debug!(daemon = %name, frame = ?frame_kind(&frame), "frame received but not yet consumed");
        }
        Frame::Goodbye { reason, message } => {
            info!(daemon = %name, ?reason, %message, "daemon sent goodbye");
            // Caller will see the next ws.recv() return None and
            // exit the loop.
        }
        Frame::Hello { .. } | Frame::Welcome { .. } => {
            warn!(
                daemon = %name,
                "unexpected post-handshake frame: {:?}",
                frame_kind(&frame)
            );
        }
        Frame::OpenSession { .. }
        | Frame::UpdateSession { .. }
        | Frame::CloseSession { .. }
        | Frame::InvokeTool { .. }
        | Frame::CancelCall { .. }
        | Frame::ListBackgroundTasks { .. } => {
            warn!(
                daemon = %name,
                "unexpected scheduler-bound frame from daemon: {:?}",
                frame_kind(&frame)
            );
        }
    }
}

async fn send_frame(sink: &mut SplitSink<WebSocket, Message>, frame: Frame) -> anyhow::Result<()> {
    let bytes = frame
        .encode_cbor()
        .map_err(|e| anyhow!("encode frame: {e}"))?;
    sink.send(Message::Binary(bytes.into()))
        .await
        .map_err(|e| anyhow!("ws send: {e}"))
}

async fn recv_frame(stream: &mut SplitStream<WebSocket>) -> anyhow::Result<Option<Frame>> {
    loop {
        match stream.next().await {
            Some(Ok(Message::Binary(bytes))) => {
                return Frame::decode_cbor(&bytes)
                    .map(Some)
                    .map_err(|e| anyhow!("frame decode: {e}"));
            }
            Some(Ok(Message::Text(_))) => {
                bail!("text frames not accepted on host-env link");
            }
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(Message::Close(_))) | None => return Ok(None),
            Some(Err(e)) => bail!("ws recv: {e}"),
        }
    }
}

/// Cheap textual tag for log messages — the full Debug of a Frame
/// includes spec/context bodies that bloat logs.
fn frame_kind(f: &Frame) -> &'static str {
    match f {
        Frame::Hello { .. } => "Hello",
        Frame::Welcome { .. } => "Welcome",
        Frame::Goodbye { .. } => "Goodbye",
        Frame::OpenSession { .. } => "OpenSession",
        Frame::UpdateSession { .. } => "UpdateSession",
        Frame::CloseSession { .. } => "CloseSession",
        Frame::SessionReady { .. } => "SessionReady",
        Frame::SessionFailed { .. } => "SessionFailed",
        Frame::SessionClosed { .. } => "SessionClosed",
        Frame::InvokeTool { .. } => "InvokeTool",
        Frame::CancelCall { .. } => "CancelCall",
        Frame::ToolChunk { .. } => "ToolChunk",
        Frame::ToolFinal { .. } => "ToolFinal",
        Frame::BackgroundTaskUpdate { .. } => "BackgroundTaskUpdate",
        Frame::ListBackgroundTasks { .. } => "ListBackgroundTasks",
        Frame::BackgroundTaskList { .. } => "BackgroundTaskList",
        Frame::HookEvent { .. } => "HookEvent",
    }
}
