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

use std::net::IpAddr;
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
    CallId, CallToolResult, ContentBlock, DaemonCapabilities, EmbeddedResource, Frame,
    GoodbyeReason, HostEnvSpec, PROTOCOL_VERSION, ProvisionPhase, SessionEndReason, SessionId,
    ThreadContext,
};

use crate::sessions::{DispatchTarget, Session, SessionRegistry};
use crate::worker::{self, WorkerError};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Bound on the inbound work-item channel from worker tasks back into
/// the connection event loop. Modest because the work items are
/// control-plane (one per concluded tool call / spawned worker), not
/// bulk data.
const WORK_CHANNEL_BOUND: usize = 64;

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
    /// Loopback IP the spawned worker binds. Same convention as
    /// `whisper-agent-sandbox`: `127.0.0.1` to keep workers
    /// loopback-only (the common case), `[::]` for dual-stack dev.
    pub bind_ip: IpAddr,
    /// Optional rustls connector. `None` = use system roots
    /// (tokio-tungstenite's default for `wss://`). Tests pass `Some`
    /// with a self-signed-trust connector when dialing local TLS.
    pub tls_connector: Option<Connector>,
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
        bind_ip: config.bind_ip,
    });
    event_loop(socket, runtime).await
}

struct Runtime {
    mcp_host_bin: String,
    bind_ip: IpAddr,
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

async fn event_loop(mut socket: WsStream, runtime: Arc<Runtime>) -> Result<(), ConnectError> {
    let (work_tx, mut work_rx) = mpsc::channel::<Work>(WORK_CHANNEL_BOUND);
    let mut sessions = SessionRegistry::default();

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
                    let arguments = apply_context_overrides(&tool_name, arguments, &target.context);
                    spawn_invoke_tool(
                        session_id,
                        call_id,
                        tool_name,
                        arguments,
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
        | Frame::HookEvent { .. } => {
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
        let result = worker::spawn(
            &spec,
            &runtime.mcp_host_bin,
            runtime.bind_ip,
            workspace_root_override.as_deref(),
        )
        .await
        .map(|w| Session { worker: w, context });
        let _ = work_tx
            .send(Work::SpawnDone {
                session_id,
                result: Box::new(result),
            })
            .await;
    });
}

/// Inject session-level [`ThreadContext`] knobs into the per-call
/// `arguments` blob. Today only `bash` has corresponding tool args
/// (`run_as`, `timeout_seconds`); other tools are passed through
/// unchanged. The session's value wins over whatever the model put
/// in the call — that's the point of the daemon-side enforcement.
///
/// `env` and `output_byte_cap` need worker-side support and aren't
/// applied yet; the session record retains them so a later sub-phase
/// can wire them without another round-trip.
pub(crate) fn apply_context_overrides(
    tool_name: &str,
    mut arguments: serde_json::Value,
    context: &ThreadContext,
) -> serde_json::Value {
    if tool_name != "bash" {
        return arguments;
    }
    let Some(obj) = arguments.as_object_mut() else {
        return arguments;
    };
    if let Some(runas) = &context.runas {
        obj.insert(
            "run_as".to_string(),
            serde_json::Value::String(runas.clone()),
        );
    }
    if let Some(secs) = context.bash_timeout_secs {
        obj.insert(
            "timeout_seconds".to_string(),
            serde_json::Value::Number(serde_json::Number::from(secs)),
        );
    }
    arguments
}

fn spawn_invoke_tool(
    session_id: SessionId,
    call_id: CallId,
    tool_name: String,
    arguments: serde_json::Value,
    target: DispatchTarget,
    work_tx: mpsc::Sender<Work>,
) {
    tokio::spawn(async move {
        let result = invoke_tool(&target, &tool_name, arguments).await;
        let _ = work_tx
            .send(Work::ToolDone {
                session_id,
                call_id,
                result,
            })
            .await;
    });
}

/// One MCP `tools/call` against the session's worker. Returns either
/// the parsed [`CallToolResult`] or a string carrying enough detail
/// for the scheduler-side error path. Network/HTTP/JSON errors all
/// fold into the `Err` arm; tool-level errors (file not found, bad
/// args) come back as an `Ok(CallToolResult)` with `is_error = Some(true)`
/// — same convention as MCP.
async fn invoke_tool(
    target: &DispatchTarget,
    tool_name: &str,
    arguments: serde_json::Value,
) -> Result<CallToolResult, String> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": tool_name, "arguments": arguments },
    });
    let resp = reqwest::Client::new()
        .post(&target.mcp_url)
        .bearer_auth(&target.mcp_token)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("MCP request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("MCP returned HTTP {}", resp.status()));
    }
    let value: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("MCP response decode: {e}"))?;
    if let Some(err) = value.get("error") {
        return Err(format!("MCP error: {err}"));
    }
    let result = value
        .get("result")
        .ok_or_else(|| "MCP response missing `result`".to_string())?;
    parse_call_result(result).map_err(|e| format!("MCP result parse: {e}"))
}

/// Translate the MCP `tools/call` result body (camelCase) into a
/// host-proto [`CallToolResult`] (snake_case).
fn parse_call_result(value: &serde_json::Value) -> Result<CallToolResult, String> {
    let content = value
        .get("content")
        .and_then(|v| v.as_array())
        .ok_or("missing `content` array")?;
    let blocks = content
        .iter()
        .map(parse_content_block)
        .collect::<Result<Vec<_>, _>>()?;
    let is_error = value.get("isError").and_then(|v| v.as_bool());
    Ok(CallToolResult {
        content: blocks,
        is_error,
    })
}

fn parse_content_block(value: &serde_json::Value) -> Result<ContentBlock, String> {
    let ty = value
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or("content block missing `type`")?;
    match ty {
        "text" => {
            let text = value
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or("text block missing `text`")?
                .to_string();
            Ok(ContentBlock::Text { text })
        }
        "image" => {
            let data = value
                .get("data")
                .and_then(|v| v.as_str())
                .ok_or("image block missing `data`")?
                .to_string();
            let mime_type = value
                .get("mimeType")
                .and_then(|v| v.as_str())
                .ok_or("image block missing `mimeType`")?
                .to_string();
            Ok(ContentBlock::Image { data, mime_type })
        }
        "resource" => {
            let resource = value
                .get("resource")
                .ok_or("resource block missing `resource`")?;
            let uri = resource
                .get("uri")
                .and_then(|v| v.as_str())
                .ok_or("resource missing `uri`")?
                .to_string();
            let mime_type = resource
                .get("mimeType")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let text = resource
                .get("text")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let blob = resource
                .get("blob")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            Ok(ContentBlock::Resource {
                resource: EmbeddedResource {
                    uri,
                    mime_type,
                    text,
                    blob,
                },
            })
        }
        other => Err(format!("unknown content block type: {other}")),
    }
}

/// Best guess at which [`ProvisionPhase`] a [`WorkerError`] occurred
/// in. The protocol's `phase` is informational for the scheduler /
/// operator — there's no "retry on phase X" logic that depends on it.
fn phase_of(err: &WorkerError) -> ProvisionPhase {
    match err {
        WorkerError::NoWorkspaceRoot
        | WorkerError::WorkspaceRootNotInSpec { .. }
        | WorkerError::Unsupported(_) => ProvisionPhase::PreExec,
        WorkerError::Spawn(_) => ProvisionPhase::WorkerSpawn,
        WorkerError::ChildExited { .. } | WorkerError::StartupTimeout { .. } => {
            ProvisionPhase::WorkerHandshake
        }
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_call_result_text_block() {
        let v = json!({
            "content": [{ "type": "text", "text": "hello" }],
        });
        let r = parse_call_result(&v).unwrap();
        assert_eq!(r.content.len(), 1);
        assert!(matches!(&r.content[0], ContentBlock::Text { text } if text == "hello"));
        assert_eq!(r.is_error, None);
    }

    #[test]
    fn parse_call_result_is_error_propagates() {
        let v = json!({
            "content": [{ "type": "text", "text": "boom" }],
            "isError": true,
        });
        let r = parse_call_result(&v).unwrap();
        assert_eq!(r.is_error, Some(true));
    }

    #[test]
    fn parse_call_result_image_block() {
        let v = json!({
            "content": [{
                "type": "image",
                "data": "AAAA",
                "mimeType": "image/png",
            }],
        });
        let r = parse_call_result(&v).unwrap();
        match &r.content[0] {
            ContentBlock::Image { data, mime_type } => {
                assert_eq!(data, "AAAA");
                assert_eq!(mime_type, "image/png");
            }
            other => panic!("unexpected block: {other:?}"),
        }
    }

    #[test]
    fn parse_call_result_resource_blob_block() {
        let v = json!({
            "content": [{
                "type": "resource",
                "resource": {
                    "uri": "file:///tmp/x.pdf",
                    "mimeType": "application/pdf",
                    "blob": "BBBB",
                }
            }],
        });
        let r = parse_call_result(&v).unwrap();
        match &r.content[0] {
            ContentBlock::Resource { resource } => {
                assert_eq!(resource.uri, "file:///tmp/x.pdf");
                assert_eq!(resource.mime_type.as_deref(), Some("application/pdf"));
                assert_eq!(resource.blob.as_deref(), Some("BBBB"));
                assert_eq!(resource.text, None);
            }
            other => panic!("unexpected block: {other:?}"),
        }
    }

    #[test]
    fn parse_call_result_rejects_missing_content() {
        let v = json!({});
        assert!(parse_call_result(&v).is_err());
    }

    #[test]
    fn parse_call_result_rejects_unknown_type() {
        let v = json!({
            "content": [{ "type": "video", "url": "x" }],
        });
        let e = parse_call_result(&v).unwrap_err();
        assert!(e.contains("video"), "got: {e}");
    }

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
    }

    fn ctx_with_runas(name: &str) -> ThreadContext {
        ThreadContext {
            runas: Some(name.to_string()),
            ..ThreadContext::default()
        }
    }

    fn ctx_with_bash_timeout(secs: u32) -> ThreadContext {
        ThreadContext {
            bash_timeout_secs: Some(secs),
            ..ThreadContext::default()
        }
    }

    #[test]
    fn apply_context_overrides_passes_through_non_bash_tools() {
        let ctx = ctx_with_runas("worker");
        let args = json!({"path": "/etc/hostname"});
        let out = apply_context_overrides("read_file", args.clone(), &ctx);
        assert_eq!(out, args, "non-bash tools must pass through unchanged");
    }

    #[test]
    fn apply_context_overrides_injects_runas_into_bash() {
        let ctx = ctx_with_runas("worker");
        let args = json!({"command": "id"});
        let out = apply_context_overrides("bash", args, &ctx);
        assert_eq!(out["run_as"], json!("worker"));
        assert_eq!(out["command"], json!("id"));
    }

    #[test]
    fn apply_context_overrides_runas_wins_over_model_supplied_value() {
        // Defense-in-depth: the session's runas overrides whatever the
        // model put in the call. A compromised model can't escape by
        // supplying its own run_as.
        let ctx = ctx_with_runas("worker");
        let args = json!({"command": "id", "run_as": "root"});
        let out = apply_context_overrides("bash", args, &ctx);
        assert_eq!(out["run_as"], json!("worker"));
    }

    #[test]
    fn apply_context_overrides_injects_bash_timeout() {
        let ctx = ctx_with_bash_timeout(30);
        let args = json!({"command": "sleep 60"});
        let out = apply_context_overrides("bash", args, &ctx);
        assert_eq!(out["timeout_seconds"], json!(30));
    }

    #[test]
    fn apply_context_overrides_bash_timeout_wins_over_model_value() {
        let ctx = ctx_with_bash_timeout(30);
        let args = json!({"command": "sleep 60", "timeout_seconds": 600});
        let out = apply_context_overrides("bash", args, &ctx);
        assert_eq!(out["timeout_seconds"], json!(30));
    }

    #[test]
    fn apply_context_overrides_default_context_does_not_mutate() {
        let ctx = ThreadContext::default();
        let args = json!({"command": "id", "run_as": "root", "timeout_seconds": 60});
        let out = apply_context_overrides("bash", args.clone(), &ctx);
        assert_eq!(out, args);
    }

    #[test]
    fn apply_context_overrides_handles_non_object_arguments() {
        // Pathological: scheduler shouldn't send non-object args, but
        // we shouldn't panic if it does — just pass through.
        let ctx = ctx_with_runas("worker");
        let args = json!("not an object");
        let out = apply_context_overrides("bash", args.clone(), &ctx);
        assert_eq!(out, args);
    }
}
