//! HTTP server: hosts the webui assets and the multiplexed WebSocket protocol.
//!
//! The server itself holds no task state. Every WebSocket connection:
//!   1. Generates a ConnId.
//!   2. Sends [`SchedulerMsg::RegisterClient`] into the scheduler inbox with an mpsc
//!      outbound channel the scheduler will push wire events to.
//!   3. Forwards decoded [`ClientToServer`] frames into the scheduler inbox as
//!      [`SchedulerMsg::ClientMessage`].
//!   4. On close, sends [`SchedulerMsg::UnregisterClient`].
//!
//! The scheduler (single tokio task, `scheduler::run`) is the only code path that
//! mutates tasks or broadcasts events.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
use whisper_agent_protocol::{ServerToClient, ThreadConfig, decode_from_client, encode_to_client};

use crate::audit::AuditLog;
use crate::persist::Persister;
use crate::pod::{Pod, PodId};
use crate::scheduler::{
    BackendEntry, ConnId, Scheduler, SchedulerMsg, SharedHostConfig, build_default_pod_config,
};

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

pub struct ServerConfig {
    /// Named backends the scheduler can dispatch model calls to.
    pub backends: std::collections::HashMap<String, BackendEntry>,
    /// Fallback backend for tasks that don't specify one. Must be a key in `backends`.
    pub default_backend: String,
    pub default_task_config: ThreadConfig,
    pub audit_log_path: PathBuf,
    pub host_id: String,
    /// Pods root directory. If `None`, persistence is disabled.
    pub pods_root: Option<PathBuf>,
    pub sandbox_provider: std::sync::Arc<dyn crate::sandbox::SandboxProvider>,
    /// Catalog of shared (singleton) MCP hosts the scheduler connects to at
    /// startup. Tasks opt in by name via `ThreadConfig.shared_mcp_hosts`.
    pub shared_mcp_hosts: Vec<SharedHostConfig>,
}

#[derive(Clone)]
struct AppState {
    inbox: mpsc::UnboundedSender<SchedulerMsg>,
    next_conn_id: Arc<AtomicU64>,
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

    // Synthesize the default pod from the server-level config. The actual
    // on-disk pod.toml is materialized below (only when persistence is
    // enabled); the in-memory entry is what the scheduler routes
    // no-pod-specified `CreateThread` requests to.
    const DEFAULT_POD_ID: &str = "default";
    let default_pod_id: PodId = DEFAULT_POD_ID.into();
    let backend_names: Vec<String> = config.backends.keys().cloned().collect();
    let shared_host_names: Vec<String> = config.shared_mcp_hosts.iter().map(|h| h.name.clone()).collect();
    let default_pod_config = build_default_pod_config(
        &default_pod_id,
        &config.default_task_config,
        &backend_names,
        &shared_host_names,
    );
    let default_pod_dir = config
        .pods_root
        .as_ref()
        .map(|root| root.join(&default_pod_id))
        .unwrap_or_else(|| PathBuf::from(format!("./{default_pod_id}")));
    let default_mcp_host_url = config.default_task_config.mcp_host_url.clone();
    let default_system_prompt = config.default_task_config.system_prompt.clone();
    let raw_toml = crate::pod::to_toml(&default_pod_config)
        .context("encode default pod.toml for in-memory bootstrap")?;
    let default_pod = Pod::new(
        default_pod_id.clone(),
        default_pod_dir.clone(),
        default_pod_config.clone(),
        raw_toml,
        default_system_prompt.clone(),
    );

    let mut scheduler = Scheduler::new(
        default_mcp_host_url,
        default_pod,
        config.host_id,
        config.backends,
        config.default_backend,
        audit,
        config.sandbox_provider,
        config.shared_mcp_hosts,
    )
    .await
    .context("scheduler init")?;

    if let Some(pods_root) = config.pods_root {
        let persister = Persister::new(pods_root.clone())
            .await
            .with_context(|| format!("open pods_root {}", pods_root.display()))?;
        // Materialize the default pod on disk so future server starts find
        // it via `load_all`. ensure_pod_toml is a no-op when a pod.toml
        // already exists, so hand-edits survive.
        persister
            .ensure_pod_toml(&default_pod_id, &default_pod_config)
            .await
            .context("write default pod.toml")?;
        let prompt_path = pods_root
            .join(&default_pod_id)
            .join(&default_pod_config.thread_defaults.system_prompt_file);
        if !default_system_prompt.is_empty()
            && !tokio::fs::try_exists(&prompt_path).await.unwrap_or(false)
        {
            tokio::fs::write(&prompt_path, &default_system_prompt)
                .await
                .with_context(|| format!("write {}", prompt_path.display()))?;
        }
        let loaded = persister
            .load_all()
            .await
            .with_context(|| "load persisted state")?;
        scheduler.load_state(loaded);
        scheduler = scheduler.with_persister(persister);
    }

    let (inbox_tx, inbox_rx) = mpsc::unbounded_channel::<SchedulerMsg>();
    let scheduler_handle = tokio::spawn(crate::scheduler::run(scheduler, inbox_rx));

    let state = AppState {
        inbox: inbox_tx.clone(),
        next_conn_id: Arc::new(AtomicU64::new(1)),
    };

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/ws", get(ws_handler))
        .nest_service("/pkg", ServeDir::new(WEBUI_PKG_DIR))
        .nest_service("/assets", ServeDir::new(WEBUI_ASSETS_DIR))
        .route_service("/index.html", ServeFile::new(WEBUI_INDEX))
        .route_service("/", ServeFile::new(WEBUI_INDEX))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind {listen}"))?;
    info!(addr = %listen, "whisper-agent server listening (open http://{listen}/ in a browser)");
    let serve_result = axum::serve(listener, app).await;

    // Shutdown: drop the inbox sender so the scheduler exits, then await it.
    drop(inbox_tx);
    let _ = scheduler_handle.await;
    serve_result?;
    Ok(())
}

async fn ws_handler(State(state): State<AppState>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_session(socket, state))
}

async fn handle_ws_session(socket: WebSocket, state: AppState) {
    let conn_id: ConnId = state.next_conn_id.fetch_add(1, Ordering::Relaxed);
    let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<ServerToClient>();

    // Register with the scheduler.
    if state
        .inbox
        .send(SchedulerMsg::RegisterClient {
            conn_id,
            outbound: outbound_tx,
        })
        .is_err()
    {
        error!("scheduler inbox closed; cannot register client");
        return;
    }
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
                    if state
                        .inbox
                        .send(SchedulerMsg::ClientMessage {
                            conn_id,
                            msg: parsed,
                        })
                        .is_err()
                    {
                        error!(conn_id, "scheduler inbox closed; dropping message");
                        break;
                    }
                }
                Err(e) => {
                    warn!(conn_id, error = %e, "decode failed");
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

    let _ = state.inbox.send(SchedulerMsg::UnregisterClient { conn_id });
    let _ = writer.await;
    info!(conn_id, "ws session closed");
}
