//! HTTP server: hosts the webui assets and the multiplexed WebSocket protocol.
//!
//! Mirrors whisper-tensor-server's pattern: `ServeDir`-mounted `/pkg` (wasm-pack output)
//! and `/assets` (static index.html etc.).
//!
//! Per-WebSocket state is just a subscription set (client registration lives on the
//! [`TaskManager`]). The manager runs independently of any connection — tasks outlive
//! their WebSockets and can be observed by any client.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use whisper_agent_protocol::{ServerToClient, TaskConfig, decode_from_client, encode_to_client};

use crate::anthropic::AnthropicClient;
use crate::audit::AuditLog;
use crate::task_manager::TaskManager;

const WEBUI_PKG_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/crates/whisper-agent-webui/pkg"
);
const WEBUI_ASSETS_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/crates/whisper-agent-webui/assets"
);
const WEBUI_INDEX: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/crates/whisper-agent-webui/assets/index.html"
);

/// Server-wide configuration. Consumed at startup to build the shared [`TaskManager`].
pub struct ServerConfig {
    pub anthropic_api_key: String,
    pub default_task_config: TaskConfig,
    pub audit_log_path: PathBuf,
    pub host_id: String,
}

pub async fn serve(listen: SocketAddr, config: ServerConfig) -> anyhow::Result<()> {
    let pkg_path = PathBuf::from(WEBUI_PKG_DIR);
    if !pkg_path.exists() {
        warn!(
            path = %pkg_path.display(),
            "webui pkg/ dir does not exist; build it with: \
             RUSTFLAGS='--cfg getrandom_backend=\"wasm_js\"' \
             wasm-pack build crates/whisper-agent-webui --target web"
        );
    }

    let audit = AuditLog::open(config.audit_log_path.clone())
        .await
        .with_context(|| format!("open audit log {}", config.audit_log_path.display()))?;
    info!(audit_log = %audit.path().display(), "audit log open");

    let anthropic = Arc::new(AnthropicClient::new(config.anthropic_api_key));
    let manager = Arc::new(TaskManager::new(
        config.default_task_config,
        config.host_id,
        anthropic,
        audit,
    ));

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/ws", get(ws_handler))
        .nest_service("/pkg", ServeDir::new(WEBUI_PKG_DIR))
        .nest_service("/assets", ServeDir::new(WEBUI_ASSETS_DIR))
        .route_service("/index.html", ServeFile::new(WEBUI_INDEX))
        .route_service("/", ServeFile::new(WEBUI_INDEX))
        .layer(TraceLayer::new_for_http())
        .with_state(manager);

    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind {listen}"))?;
    info!(addr = %listen, "whisper-agent server listening (open http://{listen}/ in a browser)");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ws_handler(
    State(manager): State<Arc<TaskManager>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_session(socket, manager))
}

/// Drive one WebSocket connection: register as a client, forward manager events to the
/// socket, decode inbound frames, hand them to the manager. The connection is stateless
/// beyond "which client id am I"; tasks live on the manager.
async fn handle_ws_session(socket: WebSocket, manager: Arc<TaskManager>) {
    let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<ServerToClient>();
    let conn_id = manager.register_client(outbound_tx);
    info!(conn_id, "ws session opened");

    let (mut sink, mut stream) = socket.split();

    let writer = tokio::spawn(async move {
        while let Some(event) = outbound_rx.recv().await {
            match encode_to_client(&event) {
                Ok(bytes) => {
                    if sink.send(Message::Binary(bytes.into())).await.is_err() {
                        break;
                    }
                }
                Err(e) => error!("encode_to_client failed: {e}"),
            }
        }
        let _ = sink.close().await;
    });

    while let Some(msg) = stream.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!(conn_id, error = %e, "ws receive error");
                break;
            }
        };
        match msg {
            Message::Binary(bytes) => match decode_from_client(&bytes) {
                Ok(parsed) => {
                    manager.handle_client_message(conn_id, parsed).await;
                }
                Err(e) => {
                    warn!(conn_id, error = %e, "decode failed");
                    // Surface decode errors inline — no task context.
                    manager.unregister_client(conn_id);
                    break;
                }
            },
            Message::Text(t) => {
                warn!(conn_id, "received text frame (expected binary CBOR): {t}");
            }
            Message::Close(_) => {
                info!(conn_id, "ws close received");
                break;
            }
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }

    manager.unregister_client(conn_id);
    let _ = writer.await;
    info!(conn_id, "ws session closed");
}
