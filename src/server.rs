//! HTTP server: hosts the webui assets and the WebSocket protocol.
//!
//! Mirrors whisper-tensor-server's pattern: ServeDir-mounted `/pkg` (wasm-pack output)
//! and `/assets` (static index.html etc.). The WebSocket at `/ws` runs one chat session
//! per connection — see [`handle_ws_session`].
//!
//! The webui's pkg/ and assets/ directories are resolved relative to this binary's
//! Cargo manifest so the server can find them in a development checkout. For deployed
//! builds we'll either embed via `include_dir!` or take a runtime path — that's a
//! future decision; not blocking MVP.

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
use whisper_agent_protocol::{
    ClientToServer, SessionState, ServerToClient, decode_from_client, encode_to_client,
};

use crate::agent_loop::{self, LoopConfig};
use crate::anthropic::AnthropicClient;
use crate::audit::AuditLog;
use crate::conversation::Conversation;
use crate::mcp::McpSession;

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

/// Server-wide configuration shared across all WebSocket sessions.
#[derive(Clone)]
pub struct ServerConfig {
    pub anthropic_api_key: Arc<String>,
    pub mcp_host_url: Arc<String>,
    pub model: Arc<String>,
    pub system_prompt: Arc<String>,
    pub max_tokens: u32,
    pub max_turns: u32,
    pub audit_log_path: PathBuf,
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

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/ws", get(ws_handler))
        .nest_service("/pkg", ServeDir::new(WEBUI_PKG_DIR))
        .nest_service("/assets", ServeDir::new(WEBUI_ASSETS_DIR))
        .route_service("/index.html", ServeFile::new(WEBUI_INDEX))
        .route_service("/", ServeFile::new(WEBUI_INDEX))
        .layer(TraceLayer::new_for_http())
        .with_state(config);

    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind {listen}"))?;
    info!(addr = %listen, "whisper-agent server listening (open http://{listen}/ in a browser)");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ws_handler(State(config): State<ServerConfig>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_session(socket, config))
}

/// One chat session per WebSocket connection. Conversation persists across multiple user
/// messages within the connection. No multi-session concurrency yet — the inbound reader
/// blocks on the agent loop while it runs (cancellation is a v0.2 concern).
async fn handle_ws_session(socket: WebSocket, config: ServerConfig) {
    let session_id = format!(
        "ses-{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    );
    info!(session_id, "ws session opened");

    let (mut sink, mut stream) = socket.split();
    let (events_tx, mut events_rx) = mpsc::unbounded_channel::<ServerToClient>();

    // Drain outbound events to the WebSocket. Runs alongside the inbound loop until either
    // side closes the channel.
    let outbound_task = tokio::spawn(async move {
        while let Some(event) = events_rx.recv().await {
            match encode_to_client(&event) {
                Ok(bytes) => {
                    if sink.send(Message::Binary(bytes.into())).await.is_err() {
                        break;
                    }
                }
                Err(e) => error!("encode_to_client failed: {e}"),
            }
        }
    });

    let send = |e: ServerToClient| {
        let _ = events_tx.send(e);
    };

    // Initial handshake: tell the client we're up and (lazily) connecting to the upstream.
    send(ServerToClient::Status {
        state: SessionState::Connected,
        detail: None,
    });

    // Open Anthropic + MCP + audit log lazily on first user message? Or eagerly here?
    // Eagerly is simpler and surfaces config errors immediately.
    let anthropic = AnthropicClient::new((*config.anthropic_api_key).clone());
    let mcp = match McpSession::connect(config.mcp_host_url.as_str()).await {
        Ok(s) => s,
        Err(e) => {
            send(ServerToClient::Error {
                message: format!("mcp connect failed: {e}"),
            });
            send(ServerToClient::Status { state: SessionState::Error, detail: None });
            drop(events_tx);
            let _ = outbound_task.await;
            return;
        }
    };
    let audit = match AuditLog::open(config.audit_log_path.clone()).await {
        Ok(a) => a,
        Err(e) => {
            send(ServerToClient::Error {
                message: format!("audit log open failed: {e}"),
            });
            send(ServerToClient::Status { state: SessionState::Error, detail: None });
            drop(events_tx);
            let _ = outbound_task.await;
            return;
        }
    };

    send(ServerToClient::Status {
        state: SessionState::Ready,
        detail: None,
    });

    let cfg = LoopConfig {
        model: (*config.model).clone(),
        max_tokens: config.max_tokens,
        system_prompt: (*config.system_prompt).clone(),
        session_id: session_id.clone(),
        host_id: "default".into(),
        max_turns: config.max_turns,
    };

    let mut conversation = Conversation::new();

    while let Some(msg) = stream.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!(session_id, error = %e, "ws receive error");
                break;
            }
        };
        match msg {
            Message::Binary(bytes) => match decode_from_client(&bytes) {
                Ok(ClientToServer::UserMessage { text }) => {
                    info!(session_id, len = text.len(), "user message");
                    send(ServerToClient::Status {
                        state: SessionState::Working,
                        detail: None,
                    });
                    let result = agent_loop::run(
                        &cfg,
                        &anthropic,
                        &mcp,
                        &audit,
                        &mut conversation,
                        text,
                        Some(events_tx.clone()),
                    )
                    .await;
                    match result {
                        Ok(outcome) => {
                            info!(
                                session_id,
                                turns = outcome.turns,
                                stop_reason = ?outcome.final_stop_reason,
                                "loop done"
                            );
                            send(ServerToClient::Status {
                                state: SessionState::Ready,
                                detail: None,
                            });
                        }
                        Err(e) => {
                            error!(session_id, error = %e, "loop failed");
                            send(ServerToClient::Error {
                                message: format!("loop failed: {e}"),
                            });
                            send(ServerToClient::Status {
                                state: SessionState::Error,
                                detail: None,
                            });
                        }
                    }
                }
                Err(e) => {
                    warn!(session_id, error = %e, "decode failed");
                    send(ServerToClient::Error {
                        message: format!("decode failed: {e}"),
                    });
                }
            },
            Message::Text(t) => {
                warn!(session_id, "received text frame (expected binary CBOR): {t}");
            }
            Message::Close(_) => {
                info!(session_id, "ws close received");
                break;
            }
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }

    drop(events_tx);
    let _ = outbound_task.await;
    info!(session_id, "ws session closed");
}
