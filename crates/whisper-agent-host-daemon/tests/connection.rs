//! Integration tests for the daemon-side WS connection.
//!
//! Stand up a loopback axum server with a hand-written scheduler-side
//! script per test, then call [`whisper_agent_host_daemon::run_connection`]
//! and assert on what comes back. The actual worker spawn (which would
//! require building `whisper-agent-mcp-host`) is out of scope here —
//! these tests cover the wire/handshake/dispatch behavior the daemon
//! adds on top of `whisper-agent-host-proto`. Worker spawn behavior is
//! covered by the unit tests in `worker.rs` (parse helpers) and by the
//! existing `whisper-agent-sandbox` integration tests against the same
//! landlock machinery.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use axum::Router;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::routing::get;
use futures::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use whisper_agent_host_daemon::{ConnectError, connection};
use whisper_agent_host_proto::{
    CallId, ContentBlock, DaemonCapabilities, Frame, GoodbyeReason, HostEnvSpecKind,
    PROTOCOL_VERSION, SessionId,
};

/// Per-test server state: a boxed async script + a one-shot the
/// script signals when it's done with its WS interactions.
///
/// We wrap the script in `Mutex<Option<...>>` because axum's `State`
/// extractor requires `Clone`, and we want `take()` semantics: the
/// first (and only) connection runs the script, subsequent connects
/// see `None` and panic the test as a configuration error.
type Script =
    Box<dyn FnOnce(WebSocket) -> futures::future::BoxFuture<'static, ()> + Send + Sync + 'static>;

#[derive(Clone)]
struct TestSync {
    script: Arc<Mutex<Option<Script>>>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<TestSync>,
) -> axum::response::Response {
    let script = state
        .script
        .lock()
        .unwrap()
        .take()
        .expect("ws_handler called twice — test misconfigured");
    ws.on_upgrade(move |socket| async move {
        script(socket).await;
    })
}

/// Bind a fresh loopback port, mount the supplied script, return
/// `(port, shutdown_tx, join)`. Test signals `shutdown_tx` when done;
/// `join` waits for the server to exit.
async fn spawn_server(script: Script) -> (u16, oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .expect("bind");
    let port = listener.local_addr().unwrap().port();
    let state = TestSync {
        script: Arc::new(Mutex::new(Some(script))),
    };
    let router = Router::new()
        .route("/v1/host_env_link", get(ws_handler))
        .with_state(state);
    let (tx, rx) = oneshot::channel::<()>();
    let join = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await
            .expect("server error");
    });
    (port, tx, join)
}

fn cap() -> DaemonCapabilities {
    DaemonCapabilities {
        tools: vec![],
        spec_kinds: vec![HostEnvSpecKind::Landlock],
        max_concurrent_sessions: None,
        supports_background_tasks: false,
        session_configurables: Vec::new(),
    }
}

fn config(port: u16) -> connection::ConnectionConfig {
    connection::ConnectionConfig {
        server_url: format!("ws://127.0.0.1:{port}/v1/host_env_link"),
        token: "ignored-by-fake-server".into(),
        daemon_version: "test-0".into(),
        capabilities: cap(),
        mcp_host_bin: "whisper-agent-mcp-host".into(),
        tls_connector: None,
        publish_credentials: Vec::new(),
    }
}

async fn send_frame(ws: &mut WebSocket, frame: Frame) {
    let bytes = frame.encode_cbor().expect("encode");
    ws.send(Message::Binary(bytes.into())).await.expect("send");
}

async fn recv_frame(ws: &mut WebSocket) -> Frame {
    loop {
        match ws.next().await.expect("ws closed").expect("ws error") {
            Message::Binary(bytes) => return Frame::decode_cbor(&bytes).expect("decode"),
            Message::Ping(_) | Message::Pong(_) => continue,
            other => panic!("unexpected message: {other:?}"),
        }
    }
}

async fn welcome(ws: &mut WebSocket) {
    send_frame(
        ws,
        Frame::Welcome {
            scheduler_version: "fake".into(),
            protocol_version: PROTOCOL_VERSION,
        },
    )
    .await;
}

async fn close_clean(ws: &mut WebSocket) {
    send_frame(
        ws,
        Frame::Goodbye {
            reason: GoodbyeReason::ServerShutdown,
            message: "test done".into(),
        },
    )
    .await;
    let _ = ws.send(Message::Close(None)).await;
}

#[tokio::test]
async fn handshake_completes_and_link_closes_cleanly() {
    let script: Script = Box::new(|mut socket| {
        Box::pin(async move {
            let frame = recv_frame(&mut socket).await;
            assert!(
                matches!(
                    &frame,
                    Frame::Hello { protocol_version, .. } if *protocol_version == PROTOCOL_VERSION
                ),
                "expected Hello, got {frame:?}"
            );
            welcome(&mut socket).await;
            close_clean(&mut socket).await;
        })
    });
    let (port, shutdown, join) = spawn_server(script).await;

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        connection::run_connection(config(port)),
    )
    .await
    .expect("daemon hung");

    assert!(matches!(result, Ok(())), "expected Ok, got: {result:?}");
    let _ = shutdown.send(());
    let _ = join.await;
}

#[tokio::test]
async fn handshake_protocol_mismatch_returns_error() {
    let script: Script = Box::new(|mut socket| {
        Box::pin(async move {
            let _ = recv_frame(&mut socket).await; // Hello
            send_frame(
                &mut socket,
                Frame::Welcome {
                    scheduler_version: "fake".into(),
                    protocol_version: PROTOCOL_VERSION + 1,
                },
            )
            .await;
            let _ = socket.send(Message::Close(None)).await;
        })
    });
    let (port, shutdown, join) = spawn_server(script).await;

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        connection::run_connection(config(port)),
    )
    .await
    .expect("daemon hung");

    match result {
        Err(ConnectError::ProtocolMismatch { daemon, scheduler }) => {
            assert_eq!(daemon, PROTOCOL_VERSION);
            assert_eq!(scheduler, PROTOCOL_VERSION + 1);
        }
        other => panic!("expected ProtocolMismatch, got: {other:?}"),
    }
    let _ = shutdown.send(());
    let _ = join.await;
}

#[tokio::test]
async fn handshake_goodbye_at_welcome_returns_error() {
    let script: Script = Box::new(|mut socket| {
        Box::pin(async move {
            let _ = recv_frame(&mut socket).await;
            send_frame(
                &mut socket,
                Frame::Goodbye {
                    reason: GoodbyeReason::Unauthorized,
                    message: "bad token".into(),
                },
            )
            .await;
            let _ = socket.send(Message::Close(None)).await;
        })
    });
    let (port, shutdown, join) = spawn_server(script).await;

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        connection::run_connection(config(port)),
    )
    .await
    .expect("daemon hung");

    match result {
        Err(ConnectError::Goodbye { reason, message }) => {
            assert_eq!(reason, GoodbyeReason::Unauthorized);
            assert_eq!(message, "bad token");
        }
        other => panic!("expected Goodbye, got: {other:?}"),
    }
    let _ = shutdown.send(());
    let _ = join.await;
}

#[tokio::test]
async fn invoke_tool_for_unknown_session_returns_error_final() {
    let script: Script = Box::new(|mut socket| {
        Box::pin(async move {
            let _ = recv_frame(&mut socket).await;
            welcome(&mut socket).await;
            let session_id = SessionId::new("never-opened");
            send_frame(
                &mut socket,
                Frame::InvokeTool {
                    session_id: session_id.clone(),
                    call_id: CallId(1),
                    tool_name: "read_file".into(),
                    arguments: serde_json::json!({"path": "/etc/hostname"}),
                    attachments: vec![],
                },
            )
            .await;
            match recv_frame(&mut socket).await {
                Frame::ToolFinal {
                    session_id: sid,
                    call_id,
                    result,
                } => {
                    assert_eq!(sid, session_id);
                    assert_eq!(call_id, CallId(1));
                    assert_eq!(result.is_error, Some(true));
                    assert!(matches!(
                        result.content.first(),
                        Some(ContentBlock::Text { text }) if text.contains("no such session")
                    ));
                }
                other => panic!("expected ToolFinal, got {other:?}"),
            }
            close_clean(&mut socket).await;
        })
    });
    let (port, shutdown, join) = spawn_server(script).await;

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        connection::run_connection(config(port)),
    )
    .await
    .expect("daemon hung");
    assert!(matches!(result, Ok(())));
    let _ = shutdown.send(());
    let _ = join.await;
}

#[tokio::test]
async fn close_session_for_unknown_session_returns_session_closed() {
    // Even when the daemon never knew about the session, it must reply
    // to CloseSession with SessionClosed so the scheduler doesn't hang
    // waiting for a terminal frame.
    let script: Script = Box::new(|mut socket| {
        Box::pin(async move {
            let _ = recv_frame(&mut socket).await;
            welcome(&mut socket).await;
            let session_id = SessionId::new("ghost");
            send_frame(
                &mut socket,
                Frame::CloseSession {
                    session_id: session_id.clone(),
                },
            )
            .await;
            match recv_frame(&mut socket).await {
                Frame::SessionClosed {
                    session_id: sid, ..
                } => {
                    assert_eq!(sid, session_id);
                }
                other => panic!("expected SessionClosed, got {other:?}"),
            }
            close_clean(&mut socket).await;
        })
    });
    let (port, shutdown, join) = spawn_server(script).await;

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        connection::run_connection(config(port)),
    )
    .await
    .expect("daemon hung");
    assert!(matches!(result, Ok(())));
    let _ = shutdown.send(());
    let _ = join.await;
}

#[tokio::test]
async fn list_background_tasks_returns_empty_list() {
    let script: Script = Box::new(|mut socket| {
        Box::pin(async move {
            let _ = recv_frame(&mut socket).await;
            welcome(&mut socket).await;
            let session_id = SessionId::new("anything");
            send_frame(
                &mut socket,
                Frame::ListBackgroundTasks {
                    session_id: session_id.clone(),
                },
            )
            .await;
            match recv_frame(&mut socket).await {
                Frame::BackgroundTaskList {
                    session_id: sid,
                    tasks,
                } => {
                    assert_eq!(sid, session_id);
                    assert!(tasks.is_empty());
                }
                other => panic!("expected BackgroundTaskList, got {other:?}"),
            }
            close_clean(&mut socket).await;
        })
    });
    let (port, shutdown, join) = spawn_server(script).await;

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        connection::run_connection(config(port)),
    )
    .await
    .expect("daemon hung");
    assert!(matches!(result, Ok(())));
    let _ = shutdown.send(());
    let _ = join.await;
}
