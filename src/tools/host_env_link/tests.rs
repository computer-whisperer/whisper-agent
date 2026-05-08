//! Integration test for the v2 host-env protocol's wire layer.
//!
//! Exercises the full path the daemon side will use in production:
//! axum-mounted WS endpoint → bearer middleware → connection task →
//! [`LiveDaemonRegistry`] → consumer-facing handle → tool round-trip.
//! The "daemon" in this test is a tokio task speaking the wire
//! directly; once the new daemon binary lands in phase 3 it'll
//! replace this fixture.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::middleware;
use axum::routing::get;
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as WsClientMessage;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use whisper_agent_host_proto::{
    CallToolResult, ContentBlock, DaemonCapabilities, Frame, HostEnvSpec, HostEnvSpecKind,
    PROTOCOL_VERSION, ProvisionPhase, SessionEndReason, ThreadContext, ToolDescriptor,
};
use whisper_agent_protocol::sandbox::{NetworkPolicy, PathAccess};

use crate::pod::config::AuthDaemon;

use super::auth::{DaemonAuthState, require_daemon_auth};
use super::{LinkError, LiveDaemonRegistry, link_handler};

/// Build the axum app the test mounts: just `/v1/host_env_link` plus
/// the daemon-auth middleware. No webui assets, no scheduler — only
/// the wire surface we're testing.
fn build_app(registry: Arc<LiveDaemonRegistry>, auth: Arc<DaemonAuthState>) -> Router {
    Router::new()
        .route("/v1/host_env_link", get(link_handler))
        .route_layer(middleware::from_fn_with_state(
            auth.clone(),
            require_daemon_auth,
        ))
        .with_state(registry)
}

/// Bind to 127.0.0.1:0, return the bound address + a shutdown signal
/// the caller fires to cleanly stop the server.
async fn spawn_server(app: Router) -> (SocketAddr, oneshot::Sender<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    (addr, tx)
}

/// Build a sample [`DaemonCapabilities`] the test "daemon" will
/// advertise. Keep it minimal — we're testing the wire, not the
/// catalog content.
fn sample_capabilities() -> DaemonCapabilities {
    DaemonCapabilities {
        tools: vec![ToolDescriptor {
            name: "echo".into(),
            description: "echo a message".into(),
            input_schema: serde_json::json!({"type":"object"}),
            annotations: Default::default(),
        }],
        spec_kinds: vec![HostEnvSpecKind::Landlock],
        max_concurrent_sessions: None,
        supports_background_tasks: false,
    }
}

fn sample_spec() -> HostEnvSpec {
    HostEnvSpec::Landlock {
        allowed_paths: vec![PathAccess::read_write("/tmp/test-ws")],
        network: NetworkPolicy::Isolated,
    }
}

async fn send_frame(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    frame: Frame,
) {
    let bytes = frame.encode_cbor().unwrap();
    ws.send(WsClientMessage::Binary(bytes.into()))
        .await
        .unwrap();
}

async fn recv_frame(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Frame {
    loop {
        match ws.next().await.expect("ws closed").expect("ws error") {
            WsClientMessage::Binary(bytes) => return Frame::decode_cbor(&bytes).unwrap(),
            WsClientMessage::Ping(_) | WsClientMessage::Pong(_) => continue,
            WsClientMessage::Close(_) => panic!("ws closed unexpectedly"),
            other => panic!("unexpected non-binary frame: {other:?}"),
        }
    }
}

/// Build a Bearer-authed WS client request. tokio-tungstenite
/// `connect_async` doesn't take headers directly — we have to build
/// a `Request` and add them.
fn ws_request_with_token(addr: SocketAddr, token: &str) -> http::Request<()> {
    let mut req = format!("ws://{addr}/v1/host_env_link")
        .into_client_request()
        .unwrap();
    req.headers_mut().insert(
        "Authorization",
        http::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
    );
    req
}

#[tokio::test]
async fn round_trip_open_session_invoke_tool() {
    // ─── server side ───
    let registry = Arc::new(LiveDaemonRegistry::new());
    registry.admit_names(["alpha".to_string()]);
    let auth = Arc::new(DaemonAuthState::new(vec![AuthDaemon {
        name: "alpha".into(),
        token: "tok-alpha".into(),
    }]));
    let (addr, shutdown) = spawn_server(build_app(registry.clone(), auth)).await;

    // ─── daemon side: connect with valid token, complete handshake ───
    let req = ws_request_with_token(addr, "tok-alpha");
    let (mut ws, _) = connect_async(req).await.expect("ws connect");

    send_frame(
        &mut ws,
        Frame::Hello {
            daemon_version: "test-0.1".into(),
            protocol_version: PROTOCOL_VERSION,
            capabilities: sample_capabilities(),
        },
    )
    .await;

    match recv_frame(&mut ws).await {
        Frame::Welcome {
            scheduler_version,
            protocol_version,
        } => {
            assert!(!scheduler_version.is_empty());
            assert_eq!(protocol_version, PROTOCOL_VERSION);
        }
        other => panic!("expected Welcome, got {other:?}"),
    }

    // ─── scheduler side: wait for the daemon to register, open a session ───
    let handle = tokio::time::timeout(
        Duration::from_secs(2),
        registry.wait_for_connection("alpha"),
    )
    .await
    .expect("daemon failed to register");
    assert_eq!(handle.name(), "alpha");
    assert_eq!(handle.capabilities().tools.len(), 1);
    assert_eq!(handle.capabilities().tools[0].name, "echo");

    // Drive the daemon side from a tokio task: receive OpenSession,
    // reply with SessionReady, receive InvokeTool, reply with ToolFinal.
    let daemon_task = tokio::spawn(async move {
        // ─ OpenSession arrives.
        let session_id = match recv_frame(&mut ws).await {
            Frame::OpenSession {
                session_id,
                thread_id,
                spec,
                context,
            } => {
                assert_eq!(thread_id, "thread-1");
                assert!(matches!(spec, HostEnvSpec::Landlock { .. }));
                assert!(context.env.is_empty());
                session_id
            }
            other => panic!("expected OpenSession, got {other:?}"),
        };
        send_frame(
            &mut ws,
            Frame::SessionReady {
                session_id: session_id.clone(),
            },
        )
        .await;

        // ─ InvokeTool arrives.
        let (call_id, args) = match recv_frame(&mut ws).await {
            Frame::InvokeTool {
                session_id: sid,
                call_id,
                tool_name,
                arguments,
                ..
            } => {
                assert_eq!(sid, session_id);
                assert_eq!(tool_name, "echo");
                (call_id, arguments)
            }
            other => panic!("expected InvokeTool, got {other:?}"),
        };
        send_frame(
            &mut ws,
            Frame::ToolFinal {
                session_id: session_id.clone(),
                call_id,
                result: CallToolResult::text(format!("echoed: {args}")),
            },
        )
        .await;

        // ─ Drop on close: scheduler-side SessionHandle::Drop fires
        // CloseSession; we ack with SessionClosed.
        match recv_frame(&mut ws).await {
            Frame::CloseSession { session_id: sid } => {
                assert_eq!(sid, session_id);
                send_frame(
                    &mut ws,
                    Frame::SessionClosed {
                        session_id: sid,
                        reason: SessionEndReason::RequestedByScheduler,
                    },
                )
                .await;
            }
            other => panic!("expected CloseSession, got {other:?}"),
        }
        // Daemon stays open for the test driver.
        ws
    });

    // Open the session.
    let session = handle
        .open_session("thread-1", sample_spec(), ThreadContext::default())
        .await
        .expect("open_session failed");
    assert!(session.session_id().to_string().starts_with("alpha-"));

    // Invoke the tool.
    let result = session
        .invoke_tool("echo", serde_json::json!({"msg":"hi"}), vec![])
        .await
        .expect("invoke_tool failed");
    assert_eq!(result.is_error, None);
    assert_eq!(result.content.len(), 1);
    match &result.content[0] {
        ContentBlock::Text { text } => assert!(text.starts_with("echoed: ")),
        other => panic!("expected Text content, got {other:?}"),
    }

    // Drop the session — fires CloseSession via Drop. The daemon
    // task's recv covers that path; await it.
    drop(session);
    let _ws = tokio::time::timeout(Duration::from_secs(2), daemon_task)
        .await
        .expect("daemon task hung")
        .expect("daemon task panicked");

    // Tear down.
    let _ = shutdown.send(());
}

#[tokio::test]
async fn rejects_invalid_bearer() {
    let registry = Arc::new(LiveDaemonRegistry::new());
    let auth = Arc::new(DaemonAuthState::new(vec![AuthDaemon {
        name: "alpha".into(),
        token: "tok-alpha".into(),
    }]));
    let (addr, shutdown) = spawn_server(build_app(registry, auth)).await;

    let req = ws_request_with_token(addr, "wrong-token");
    let err = connect_async(req).await.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("401") || msg.contains("Unauthorized"),
        "expected 401, got: {msg}",
    );
    let _ = shutdown.send(());
}

#[tokio::test]
async fn rejects_protocol_mismatch() {
    let registry = Arc::new(LiveDaemonRegistry::new());
    let auth = Arc::new(DaemonAuthState::new(vec![AuthDaemon {
        name: "alpha".into(),
        token: "tok-alpha".into(),
    }]));
    let (addr, shutdown) = spawn_server(build_app(registry.clone(), auth)).await;

    let req = ws_request_with_token(addr, "tok-alpha");
    let (mut ws, _) = connect_async(req).await.unwrap();

    // Send a Hello with a deliberately-wrong protocol_version.
    send_frame(
        &mut ws,
        Frame::Hello {
            daemon_version: "test".into(),
            protocol_version: PROTOCOL_VERSION + 99,
            capabilities: sample_capabilities(),
        },
    )
    .await;

    // Expect a Goodbye with ProtocolMismatch and a closed socket
    // shortly after.
    let goodbye = recv_frame(&mut ws).await;
    match goodbye {
        Frame::Goodbye { reason, .. } => {
            use whisper_agent_host_proto::GoodbyeReason;
            assert!(matches!(reason, GoodbyeReason::ProtocolMismatch));
        }
        other => panic!("expected Goodbye, got {other:?}"),
    }
    // Daemon should never have appeared in the registry.
    assert!(!registry.is_connected("alpha"));

    let _ = shutdown.send(());
}

#[tokio::test]
async fn rejects_duplicate_name() {
    let registry = Arc::new(LiveDaemonRegistry::new());
    registry.admit_names(["alpha".to_string()]);
    let auth = Arc::new(DaemonAuthState::new(vec![AuthDaemon {
        name: "alpha".into(),
        token: "tok-alpha".into(),
    }]));
    let (addr, shutdown) = spawn_server(build_app(registry.clone(), auth)).await;

    // First daemon: complete handshake, become live.
    let req1 = ws_request_with_token(addr, "tok-alpha");
    let (mut ws1, _) = connect_async(req1).await.unwrap();
    send_frame(
        &mut ws1,
        Frame::Hello {
            daemon_version: "first".into(),
            protocol_version: PROTOCOL_VERSION,
            capabilities: sample_capabilities(),
        },
    )
    .await;
    let _ = recv_frame(&mut ws1).await; // Welcome
    tokio::time::timeout(
        Duration::from_secs(2),
        registry.wait_for_connection("alpha"),
    )
    .await
    .unwrap();

    // Second daemon: same name → expect Goodbye::NameAlreadyConnected.
    let req2 = ws_request_with_token(addr, "tok-alpha");
    let (mut ws2, _) = connect_async(req2).await.unwrap();
    send_frame(
        &mut ws2,
        Frame::Hello {
            daemon_version: "second".into(),
            protocol_version: PROTOCOL_VERSION,
            capabilities: sample_capabilities(),
        },
    )
    .await;
    let _ = recv_frame(&mut ws2).await; // Welcome (handshake completes before duplicate check)
    let goodbye = recv_frame(&mut ws2).await;
    match goodbye {
        Frame::Goodbye { reason, .. } => {
            use whisper_agent_host_proto::GoodbyeReason;
            assert!(matches!(reason, GoodbyeReason::NameAlreadyConnected));
        }
        other => panic!("expected Goodbye, got {other:?}"),
    }
    // Original daemon still connected.
    assert!(registry.is_connected("alpha"));
    let _ = shutdown.send(());
}

#[tokio::test]
async fn evicts_stale_existing_on_collision() {
    // Connect a first daemon, force its registry handle to look stale
    // (last_active_at backdated past CONFLICT_STALE_WINDOW), then
    // connect a second daemon under the same name. The second should
    // win the slot; the first should receive `Goodbye::Superseded`.
    let registry = Arc::new(LiveDaemonRegistry::new());
    registry.admit_names(["alpha".to_string()]);
    let auth = Arc::new(DaemonAuthState::new(vec![AuthDaemon {
        name: "alpha".into(),
        token: "tok-alpha".into(),
    }]));
    let (addr, shutdown) = spawn_server(build_app(registry.clone(), auth)).await;

    // First daemon: complete handshake and register.
    let req1 = ws_request_with_token(addr, "tok-alpha");
    let (mut ws1, _) = connect_async(req1).await.unwrap();
    send_frame(
        &mut ws1,
        Frame::Hello {
            daemon_version: "first".into(),
            protocol_version: PROTOCOL_VERSION,
            capabilities: sample_capabilities(),
        },
    )
    .await;
    let _ = recv_frame(&mut ws1).await; // Welcome
    let first_handle = tokio::time::timeout(
        Duration::from_secs(2),
        registry.wait_for_connection("alpha"),
    )
    .await
    .unwrap();

    // Backdate the existing handle's liveness timestamp so the
    // registry sees it as stale on the next collision.
    first_handle.__set_last_active_for_test(0);

    // Second daemon: same name. Should be admitted.
    let req2 = ws_request_with_token(addr, "tok-alpha");
    let (mut ws2, _) = connect_async(req2).await.unwrap();
    send_frame(
        &mut ws2,
        Frame::Hello {
            daemon_version: "second".into(),
            protocol_version: PROTOCOL_VERSION,
            capabilities: sample_capabilities(),
        },
    )
    .await;
    match recv_frame(&mut ws2).await {
        Frame::Welcome { .. } => {}
        other => panic!("expected Welcome on second connection, got {other:?}"),
    }

    // First daemon receives `Goodbye::Superseded`.
    match recv_frame(&mut ws1).await {
        Frame::Goodbye { reason, .. } => {
            use whisper_agent_host_proto::GoodbyeReason;
            assert!(
                matches!(reason, GoodbyeReason::Superseded),
                "expected Superseded, got {reason:?}"
            );
        }
        other => panic!("expected Goodbye on first connection, got {other:?}"),
    }

    // Registry now points at the new handle, not the backdated one.
    let current = tokio::time::timeout(
        Duration::from_secs(2),
        registry.wait_for_connection("alpha"),
    )
    .await
    .unwrap();
    assert!(
        !std::sync::Arc::ptr_eq(&current, &first_handle),
        "registry should hold the replacement handle, not the superseded one",
    );

    let _ = shutdown.send(());
}

#[tokio::test]
async fn surfaces_session_failed_to_consumer() {
    let registry = Arc::new(LiveDaemonRegistry::new());
    registry.admit_names(["alpha".to_string()]);
    let auth = Arc::new(DaemonAuthState::new(vec![AuthDaemon {
        name: "alpha".into(),
        token: "tok-alpha".into(),
    }]));
    let (addr, shutdown) = spawn_server(build_app(registry.clone(), auth)).await;

    let req = ws_request_with_token(addr, "tok-alpha");
    let (mut ws, _) = connect_async(req).await.unwrap();
    send_frame(
        &mut ws,
        Frame::Hello {
            daemon_version: "t".into(),
            protocol_version: PROTOCOL_VERSION,
            capabilities: sample_capabilities(),
        },
    )
    .await;
    let _ = recv_frame(&mut ws).await; // Welcome

    let handle = tokio::time::timeout(
        Duration::from_secs(2),
        registry.wait_for_connection("alpha"),
    )
    .await
    .unwrap();

    let daemon_task = tokio::spawn(async move {
        let session_id = match recv_frame(&mut ws).await {
            Frame::OpenSession { session_id, .. } => session_id,
            other => panic!("expected OpenSession, got {other:?}"),
        };
        send_frame(
            &mut ws,
            Frame::SessionFailed {
                session_id,
                phase: ProvisionPhase::Landlock,
                message: "ABI mismatch".into(),
            },
        )
        .await;
        ws
    });

    let err = handle
        .open_session("t", sample_spec(), ThreadContext::default())
        .await
        .unwrap_err();
    match err {
        LinkError::ProvisionFailed { phase, message } => {
            assert!(matches!(phase, ProvisionPhase::Landlock));
            assert_eq!(message, "ABI mismatch");
        }
        other => panic!("expected ProvisionFailed, got {other:?}"),
    }

    let _ = tokio::time::timeout(Duration::from_secs(2), daemon_task).await;
    let _ = shutdown.send(());
}

#[tokio::test]
async fn registry_admits_offline_names_separately_from_connected() {
    let registry = LiveDaemonRegistry::new();
    registry.admit_names(["alpha".to_string(), "beta".to_string()]);
    let snap = registry.snapshot();
    assert_eq!(snap.admitted, vec!["alpha".to_string(), "beta".to_string()]);
    assert!(snap.connected.is_empty());
    assert!(registry.is_admitted("alpha"));
    assert!(!registry.is_connected("alpha"));
    assert!(!registry.is_admitted("gamma"));
}
