//! HTTP server that hosts the webui and (eventually) the WebSocket protocol.
//!
//! Mirrors whisper-tensor-server's pattern: ServeDir-mounted `/pkg` (wasm-pack output)
//! and `/assets` (static index.html etc.), plus `/` and `/index.html` resolving to the
//! UI's index. WebSocket endpoint at `/ws` is a placeholder until step 6.
//!
//! The webui's pkg/ and assets/ directories are resolved relative to this binary's
//! Cargo manifest so the server can find them in a development checkout. For deployed
//! builds we'll either embed via `include_dir!` or take a runtime path — that's a
//! future decision; not blocking MVP.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context;
use axum::{
    Router,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use tokio::net::TcpListener;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

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

pub async fn serve(listen: SocketAddr) -> anyhow::Result<()> {
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
        .layer(TraceLayer::new_for_http());

    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind {listen}"))?;
    info!(addr = %listen, "whisper-agent server listening (open http://{listen}/ in a browser)");
    axum::serve(listener, app).await?;
    Ok(())
}

/// WebSocket placeholder. Echoes one frame and closes — real protocol lands in step 6.
async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_ws_placeholder)
}

async fn handle_ws_placeholder(mut socket: WebSocket) {
    info!("ws placeholder: connection opened");
    let _ = socket
        .send(Message::Text(
            "whisper-agent ws placeholder — protocol not yet implemented (step 6)".into(),
        ))
        .await;
    // Connection drops on return.
}
