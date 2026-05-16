//! WebSocket connection task — daemon side.
//!
//! Mirror of `src/tools/host_env_link/connection.rs` on the
//! whisper-agent side. The daemon is the WS *client*: it dials,
//! presents `Authorization: Bearer <token>` on the upgrade, sends
//! [`Frame::Hello`], waits for [`Frame::Welcome`], then runs an event
//! loop dispatching incoming frames against a [`SessionRegistry`].
//!
//! One [`run_connection`] call = one dial. Reconnect is the binary's
//! responsibility — the library returns a clean [`ConnectError`] on
//! anything terminal so the operator-visible logging happens at one
//! call site.

use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{
    Connector, MaybeTlsStream, WebSocketStream, connect_async_tls_with_config,
};
use tracing::{debug, info, warn};
use whisper_agent_host_proto::{
    CallId, CallToolResult, ContentBlock, DaemonCapabilities, Frame, GoodbyeReason, HostEnvSpec,
    PROTOCOL_VERSION, ProvisionPhase, SessionEndReason, SessionId, ThreadContext,
};

use crate::sessions::{DispatchTarget, Session, SessionRegistry};
use crate::worker::{self, Worker, WorkerError};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Bound on the inbound work-item channel from worker tasks back into
/// the connection event loop. Modest because the work items are
/// control-plane (one per concluded tool call / spawned worker), not
/// bulk data.
const WORK_CHANNEL_BOUND: usize = 64;

/// Bound on the outbound-frame channel used by the credential
/// publisher tasks to push frames at the scheduler. One per active
/// publisher per refresh is the steady state — token rotations are
/// ~hourly, so the channel never approaches capacity in practice.
const OUTBOUND_CHANNEL_BOUND: usize = 16;

/// Configuration the binary hands to [`run_connection`]. Keeps the
/// library entry point free of CLI-specific knobs.
pub struct ConnectionConfig {
    /// Full WS URL including scheme (`ws://` or `wss://`) and path
    /// (`/v1/host_env_link`).
    pub server_url: String,
    /// Bearer token presented at upgrade.
    pub token: String,
    /// Daemon version string, sent in [`Frame::Hello`]. Defaults to
    /// the daemon binary's `CARGO_PKG_VERSION` when launched from
    /// `main.rs`; tests set their own.
    pub daemon_version: String,
    /// Capability snapshot to advertise. Built once at daemon startup
    /// (typically by [`crate::catalog::probe_tool_catalog`]) and
    /// reused across reconnects.
    pub capabilities: DaemonCapabilities,
    /// Path to the worker binary (`whisper-agent-mcp-host`) the
    /// daemon spawns per-session. Resolved against `$PATH` if not
    /// absolute.
    pub mcp_host_bin: String,
    /// Optional rustls connector. `None` = use system roots
    /// (tokio-tungstenite's default for `wss://`). Tests pass `Some`
    /// with a self-signed-trust connector when dialing local TLS.
    pub tls_connector: Option<Connector>,
    /// Credential files this daemon publishes to the scheduler. One
    /// publisher task is spawned per entry after the handshake. Empty
    /// list (the typical case) → no publishers, no behavior change
    /// from a pre-publication-feature daemon.
    pub publish_credentials: Vec<crate::config::PublishCredentialConfig>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    #[error("dial failed: {0}")]
    Dial(String),
    #[error("scheduler closed before sending Welcome")]
    NoWelcome,
    #[error("protocol_version mismatch: daemon={daemon}, scheduler={scheduler}")]
    ProtocolMismatch { daemon: u32, scheduler: u32 },
    #[error("scheduler sent Goodbye: {reason:?} — {message}")]
    Goodbye {
        reason: GoodbyeReason,
        message: String,
    },
    #[error("scheduler sent unexpected frame at handshake: {0}")]
    UnexpectedHandshakeFrame(String),
    #[error("ws transport error: {0}")]
    Transport(String),
    #[error("frame codec error: {0}")]
    Codec(String),
}

/// Dial the scheduler, handshake, run the event loop until either
/// side closes. Returns `Ok(())` on a clean close (Goodbye::ServerShutdown,
/// daemon-initiated CloseSession exhaustion); errors are reserved for
/// connection-level failures the caller should log and back off on.
pub async fn run_connection(config: ConnectionConfig) -> Result<(), ConnectError> {
    let mut req = config
        .server_url
        .as_str()
        .into_client_request()
        .map_err(|e| ConnectError::Dial(format!("build request: {e}")))?;
    req.headers_mut().insert(
        http::header::AUTHORIZATION,
        http::HeaderValue::from_str(&format!("Bearer {}", config.token))
            .map_err(|e| ConnectError::Dial(format!("bearer header: {e}")))?,
    );
    debug!(url = %config.server_url, "dialing scheduler");
    let (mut socket, _resp) =
        connect_async_tls_with_config(req, None, false, config.tls_connector.clone())
            .await
            .map_err(|e| ConnectError::Dial(e.to_string()))?;

    handshake(
        &mut socket,
        &config.daemon_version,
        config.capabilities.clone(),
    )
    .await?;
    info!(url = %config.server_url, "v2 host-env link established");

    let runtime = Arc::new(Runtime {
        mcp_host_bin: config.mcp_host_bin.clone(),
    });
    event_loop(socket, runtime, config.publish_credentials).await
}

struct Runtime {
    mcp_host_bin: String,
}

/// Inbound work items the event loop processes alongside WS frames.
/// Used to thread spawn-completion / tool-call-completion back into
/// the loop without blocking the loop thread on the actual work.
enum Work {
    /// A worker spawn finished. Either insert the session and emit
    /// [`Frame::SessionReady`], or emit [`Frame::SessionFailed`].
    /// Boxed because `Session` carries the full `Worker` (with its
    /// `Child`) and a `ThreadContext`; without the box the enum is
    /// dominated by this variant and the dispatch path's `ToolDone`
    /// inflates with it (clippy::large_enum_variant).
    SpawnDone {
        session_id: SessionId,
        result: Box<Result<Session, WorkerError>>,
    },
    /// A tool call finished. Emit [`Frame::ToolFinal`] back to the
    /// scheduler. `Err` paths fold into a `CallToolResult` with
    /// `is_error = Some(true)` rather than a separate frame variant
    /// — the wire only has the one terminal.
    ToolDone {
        session_id: SessionId,
        call_id: CallId,
        result: Result<CallToolResult, String>,
    },
}

async fn handshake(
    socket: &mut WsStream,
    daemon_version: &str,
    capabilities: DaemonCapabilities,
) -> Result<(), ConnectError> {
    send_frame(
        socket,
        Frame::Hello {
            daemon_version: daemon_version.into(),
            protocol_version: PROTOCOL_VERSION,
            capabilities,
        },
    )
    .await?;
    let welcome = recv_frame(socket).await?.ok_or(ConnectError::NoWelcome)?;
    match welcome {
        Frame::Welcome {
            protocol_version,
            scheduler_version,
        } => {
            if protocol_version != PROTOCOL_VERSION {
                return Err(ConnectError::ProtocolMismatch {
                    daemon: PROTOCOL_VERSION,
                    scheduler: protocol_version,
                });
            }
            info!(scheduler_version, protocol_version, "handshake complete");
            Ok(())
        }
        Frame::Goodbye { reason, message } => Err(ConnectError::Goodbye { reason, message }),
        other => Err(ConnectError::UnexpectedHandshakeFrame(
            frame_kind(&other).into(),
        )),
    }
}

async fn event_loop(
    mut socket: WsStream,
    runtime: Arc<Runtime>,
    publish_credentials: Vec<crate::config::PublishCredentialConfig>,
) -> Result<(), ConnectError> {
    let (work_tx, mut work_rx) = mpsc::channel::<Work>(WORK_CHANNEL_BOUND);
    let (outbound_tx, mut outbound_rx) = mpsc::channel::<Frame>(OUTBOUND_CHANNEL_BOUND);
    let mut sessions = SessionRegistry::default();

    // Spawn credential publishers (if any). Handles drop on event-
    // loop exit, aborting their tasks at the next await point.
    // Build a dedicated reqwest client so the publishers' refresh
    // calls are independent of any HTTP client the worker layer
    // might add later.
    let _publishers = if publish_credentials.is_empty() {
        crate::credentials::PublisherHandles::default()
    } else {
        crate::credentials::spawn_all(
            &publish_credentials,
            outbound_tx.clone(),
            reqwest::Client::new(),
        )
    };
    // Drop our own send end so `publishers` see the channel close
    // when the event loop exits and the receiver drops below.
    drop(outbound_tx);

    loop {
        tokio::select! {
            msg = socket.next() => match msg {
                Some(Ok(Message::Binary(bytes))) => match Frame::decode_cbor(&bytes) {
                    Ok(frame) => {
                        if let ControlFlow::Stop(reason) = handle_inbound(
                            frame,
                            &mut sessions,
                            runtime.clone(),
                            &work_tx,
                            &mut socket,
                        ).await? {
                            info!(?reason, "scheduler closed link");
                            return Ok(());
                        }
                    }
                    Err(e) => return Err(ConnectError::Codec(e.to_string())),
                },
                Some(Ok(Message::Close(_))) | None => {
                    info!("scheduler closed websocket");
                    return Ok(());
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {
                    // tokio-tungstenite handles WS-level keepalive
                    // (auto-pong on Ping). Pong frames received here
                    // are just liveness signals — drop.
                }
                Some(Ok(Message::Text(_))) => {
                    return Err(ConnectError::Transport(
                        "text frames not accepted on host-env link".into(),
                    ));
                }
                Some(Ok(Message::Frame(_))) => {
                    // Raw frame — tungstenite emits this only when the
                    // user explicitly asks; we don't, so this is a wire
                    // surprise. Treat as an error.
                    return Err(ConnectError::Transport(
                        "unexpected raw frame on host-env link".into(),
                    ));
                }
                Some(Err(e)) => return Err(ConnectError::Transport(e.to_string())),
            },
            Some(work) = work_rx.recv() => {
                handle_work(work, &mut sessions, &mut socket).await?;
            }
            Some(frame) = outbound_rx.recv() => {
                send_frame(&mut socket, frame).await?;
            }
        }
    }
}

enum ControlFlow {
    Continue,
    Stop(GoodbyeReason),
}

async fn handle_inbound(
    frame: Frame,
    sessions: &mut SessionRegistry,
    runtime: Arc<Runtime>,
    work_tx: &mpsc::Sender<Work>,
    socket: &mut WsStream,
) -> Result<ControlFlow, ConnectError> {
    match frame {
        Frame::OpenSession {
            session_id,
            thread_id,
            spec,
            context,
        } => {
            spawn_open_session(
                session_id,
                thread_id,
                spec,
                context,
                runtime,
                work_tx.clone(),
            );
            Ok(ControlFlow::Continue)
        }
        Frame::InvokeTool {
            session_id,
            call_id,
            tool_name,
            arguments,
            attachments,
        } => {
            match sessions.dispatch_target(&session_id) {
                Some(target) => {
                    // Defense-in-depth: the scheduler also enforces the
                    // allowlist before dispatching, but the denylist is
                    // a per-session veto the scheduler may not know about
                    // (e.g. the user just edited it). Refusing here costs
                    // nothing if the scheduler already filtered.
                    if target.context.tool_denylist.contains(&tool_name) {
                        warn!(
                            %session_id,
                            %call_id,
                            %tool_name,
                            "tool denied by session denylist",
                        );
                        send_frame(
                            socket,
                            Frame::ToolFinal {
                                session_id,
                                call_id,
                                result: CallToolResult::error_text(format!(
                                    "tool `{tool_name}` denied by session denylist"
                                )),
                            },
                        )
                        .await?;
                        return Ok(ControlFlow::Continue);
                    }
                    spawn_invoke_tool(
                        session_id,
                        call_id,
                        tool_name,
                        arguments,
                        attachments,
                        target,
                        work_tx.clone(),
                    );
                }
                None => {
                    // No such session — return a synthetic ToolFinal
                    // marked is_error so the scheduler sees a terminal
                    // for this call. The scheduler should not have sent
                    // an InvokeTool for an absent session, but we don't
                    // want to silently swallow it.
                    warn!(
                        %session_id,
                        %call_id,
                        %tool_name,
                        "InvokeTool against unknown session — surfacing as error"
                    );
                    send_frame(
                        socket,
                        Frame::ToolFinal {
                            session_id,
                            call_id,
                            result: CallToolResult::error_text("no such session on this daemon"),
                        },
                    )
                    .await?;
                }
            }
            Ok(ControlFlow::Continue)
        }
        Frame::CloseSession { session_id } => {
            // Drop the worker → child killed via `kill_on_drop`. We
            // emit SessionClosed even if we never knew the session, so
            // the scheduler always gets a terminal for its CloseSession.
            let _ = sessions.remove(&session_id);
            send_frame(
                socket,
                Frame::SessionClosed {
                    session_id,
                    reason: SessionEndReason::RequestedByScheduler,
                },
            )
            .await?;
            Ok(ControlFlow::Continue)
        }
        Frame::UpdateSession {
            session_id,
            context_delta,
        } => {
            // Apply the delta to the session's stored ThreadContext.
            // Subsequent tool calls run under the new values
            // (denylist, runas, env, bash_timeout). workspace_root in
            // a delta is recorded but cannot move the worker — the
            // daemon doesn't tear down and reprovision; only the next
            // OpenSession would re-landlock. We log a warning when
            // the field shows up so operators see the limitation.
            if context_delta.workspace_root.is_some() {
                warn!(
                    %session_id,
                    "UpdateSession.workspace_root cannot move a live worker — \
                     value recorded but ineffective until next OpenSession",
                );
            }
            if !sessions.apply_context_delta(&session_id, &context_delta) {
                warn!(
                    %session_id,
                    "UpdateSession against unknown session — ignoring",
                );
            }
            Ok(ControlFlow::Continue)
        }
        Frame::CancelCall {
            session_id,
            call_id,
        } => {
            // Phase 5/6. We accept the frame and log; the call will
            // still run to completion and emit its ToolFinal. No
            // correctness break — the scheduler treats the eventual
            // ToolFinal as authoritative per the design doc.
            warn!(%session_id, %call_id, "CancelCall received but not yet implemented");
            Ok(ControlFlow::Continue)
        }
        Frame::ListBackgroundTasks { session_id } => {
            // Phase 6 placeholder. Reply with an empty list so the
            // scheduler can complete its resync flow without waiting.
            send_frame(
                socket,
                Frame::BackgroundTaskList {
                    session_id,
                    tasks: vec![],
                },
            )
            .await?;
            Ok(ControlFlow::Continue)
        }
        Frame::Goodbye { reason, message } => {
            info!(?reason, %message, "scheduler sent goodbye");
            Ok(ControlFlow::Stop(reason))
        }
        Frame::CredentialAck { backend, result } => {
            use whisper_agent_host_proto::CredentialAckResult;
            match result {
                CredentialAckResult::Accepted => {
                    info!(backend, "credential publish accepted by scheduler");
                }
                CredentialAckResult::Rejected { reason } => {
                    // Surface loudly: a rejection means the bilateral
                    // pairing is misconfigured (wrong backend name,
                    // wrong manager on the scheduler) or the payload
                    // doesn't parse. The publisher will republish on
                    // its next change/refresh; if the misconfiguration
                    // persists, the rejections will too.
                    warn!(
                        backend,
                        reason,
                        "credential publish rejected by scheduler — check backend/manager pairing"
                    );
                }
            }
            Ok(ControlFlow::Continue)
        }
        // Frames the scheduler should never send to the daemon.
        Frame::Hello { .. }
        | Frame::Welcome { .. }
        | Frame::SessionReady { .. }
        | Frame::SessionFailed { .. }
        | Frame::SessionClosed { .. }
        | Frame::ToolChunk { .. }
        | Frame::ToolFinal { .. }
        | Frame::BackgroundTaskUpdate { .. }
        | Frame::BackgroundTaskList { .. }
        | Frame::HookEvent { .. }
        | Frame::PublishCredential { .. } => {
            warn!(
                kind = frame_kind(&frame),
                "unexpected daemon-bound frame from scheduler — ignoring"
            );
            Ok(ControlFlow::Continue)
        }
    }
}

async fn handle_work(
    work: Work,
    sessions: &mut SessionRegistry,
    socket: &mut WsStream,
) -> Result<(), ConnectError> {
    match work {
        Work::SpawnDone { session_id, result } => match *result {
            Ok(session) => {
                sessions.insert(session_id.clone(), session);
                send_frame(socket, Frame::SessionReady { session_id }).await?;
            }
            Err(e) => {
                let phase = phase_of(&e);
                send_frame(
                    socket,
                    Frame::SessionFailed {
                        session_id,
                        phase,
                        message: e.to_string(),
                    },
                )
                .await?;
            }
        },
        Work::ToolDone {
            session_id,
            call_id,
            result,
        } => {
            let result = result.unwrap_or_else(CallToolResult::error_text);
            send_frame(
                socket,
                Frame::ToolFinal {
                    session_id,
                    call_id,
                    result,
                },
            )
            .await?;
        }
    }
    Ok(())
}

fn spawn_open_session(
    session_id: SessionId,
    thread_id: String,
    spec: HostEnvSpec,
    context: ThreadContext,
    runtime: Arc<Runtime>,
    work_tx: mpsc::Sender<Work>,
) {
    let _ = thread_id; // daemon-side logging only; the worker doesn't see it
    tokio::spawn(async move {
        let workspace_root_override = context.workspace_root.clone();
        let runas_override = context.runas.clone();
        let bash_timeout_secs = context.bash_timeout_secs;
        let result = worker::spawn(
            &spec,
            &runtime.mcp_host_bin,
            workspace_root_override.as_deref(),
            runas_override.as_deref(),
            bash_timeout_secs,
        )
        .await
        .map(|w| Session {
            worker: Arc::new(w),
            context,
        });
        let _ = work_tx
            .send(Work::SpawnDone {
                session_id,
                result: Box::new(result),
            })
            .await;
    });
}

fn spawn_invoke_tool(
    session_id: SessionId,
    call_id: CallId,
    tool_name: String,
    arguments: serde_json::Value,
    attachments: Vec<ContentBlock>,
    target: DispatchTarget,
    work_tx: mpsc::Sender<Work>,
) {
    tokio::spawn(async move {
        let result = invoke_tool(&target.worker, call_id, tool_name, arguments, attachments)
            .await
            .map_err(|e| e.to_string());
        let _ = work_tx
            .send(Work::ToolDone {
                session_id,
                call_id,
                result,
            })
            .await;
    });
}

/// Run one tool call against the session's worker over the daemon ↔
/// worker IPC. Returns the worker's terminal [`CallToolResult`]
/// (which itself may carry `is_error = Some(true)` for tool-level
/// failures), or an [`Err`] for IPC-level failures (worker died,
/// socket dropped). Errors fold into the scheduler-bound `ToolFinal`
/// at the call site, so the scheduler always sees a terminal frame
/// per call_id.
async fn invoke_tool(
    worker: &Worker,
    call_id: CallId,
    tool_name: String,
    arguments: serde_json::Value,
    attachments: Vec<ContentBlock>,
) -> Result<CallToolResult, WorkerError> {
    worker
        .invoke(call_id, tool_name, arguments, attachments)
        .await
}

/// Best guess at which [`ProvisionPhase`] a [`WorkerError`] occurred
/// in. The protocol's `phase` is informational for the scheduler /
/// operator — there's no "retry on phase X" logic that depends on it.
fn phase_of(err: &WorkerError) -> ProvisionPhase {
    match err {
        WorkerError::NoWorkspaceRoot
        | WorkerError::WorkspaceRootNotInSpec { .. }
        | WorkerError::Unsupported(_)
        | WorkerError::PreExec(_) => ProvisionPhase::PreExec,
        WorkerError::Spawn(_) => ProvisionPhase::WorkerSpawn,
        WorkerError::ChildExited { .. }
        | WorkerError::StartupTimeout { .. }
        | WorkerError::ProtocolMismatch { .. }
        | WorkerError::Disconnected => ProvisionPhase::WorkerHandshake,
    }
}

async fn send_frame(socket: &mut WsStream, frame: Frame) -> Result<(), ConnectError> {
    let bytes = frame
        .encode_cbor()
        .map_err(|e| ConnectError::Codec(e.to_string()))?;
    socket
        .send(Message::Binary(bytes.into()))
        .await
        .map_err(|e| ConnectError::Transport(e.to_string()))
}

async fn recv_frame(socket: &mut WsStream) -> Result<Option<Frame>, ConnectError> {
    loop {
        match socket.next().await {
            Some(Ok(Message::Binary(bytes))) => {
                return Frame::decode_cbor(&bytes)
                    .map(Some)
                    .map_err(|e| ConnectError::Codec(e.to_string()));
            }
            Some(Ok(Message::Text(_))) => {
                return Err(ConnectError::Transport(
                    "text frames not accepted on host-env link".into(),
                ));
            }
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(Message::Frame(_))) => {
                return Err(ConnectError::Transport(
                    "unexpected raw frame on host-env link".into(),
                ));
            }
            Some(Ok(Message::Close(_))) | None => return Ok(None),
            Some(Err(e)) => return Err(ConnectError::Transport(e.to_string())),
        }
    }
}

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
        Frame::PublishCredential { .. } => "PublishCredential",
        Frame::CredentialAck { .. } => "CredentialAck",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_of_classifies_errors() {
        use ProvisionPhase::*;
        assert_eq!(phase_of(&WorkerError::NoWorkspaceRoot), PreExec);
        assert_eq!(
            phase_of(&WorkerError::WorkspaceRootNotInSpec {
                override_path: "/nope".into()
            }),
            PreExec
        );
        assert_eq!(
            phase_of(&WorkerError::Unsupported("container".into())),
            PreExec
        );
        assert_eq!(phase_of(&WorkerError::Spawn("execvp".into())), WorkerSpawn);
        assert_eq!(
            phase_of(&WorkerError::ChildExited {
                code: Some(1),
                stderr_tail: String::new()
            }),
            WorkerHandshake
        );
        assert_eq!(
            phase_of(&WorkerError::StartupTimeout {
                seconds: 10,
                stderr_tail: String::new()
            }),
            WorkerHandshake
        );
        assert_eq!(
            phase_of(&WorkerError::ProtocolMismatch {
                worker: 999,
                daemon: 1,
                stderr_tail: String::new()
            }),
            WorkerHandshake
        );
        assert_eq!(phase_of(&WorkerError::Disconnected), WorkerHandshake);
        assert_eq!(
            phase_of(&WorkerError::PreExec("setuid: EPERM".into())),
            PreExec,
        );
    }
}
