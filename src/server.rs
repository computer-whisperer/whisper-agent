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
//!
//! [`thread_router`] owns the connection registry, subscriptions, and the
//! `ThreadEvent → ServerToClient` translation that the scheduler hands off
//! after each step.

pub mod thread_router;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context;
use axum::{
    Router,
    body::Bytes,
    extract::{
        Path, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use whisper_agent_protocol::{
    HostEnvSpec, ServerToClient, ThreadConfig, decode_from_client, encode_to_client,
};

use crate::pod::persist::Persister;
use crate::pod::{Pod, PodId};
use crate::runtime::audit::AuditLog;
use crate::runtime::scheduler::{
    BackendEntry, ConnId, Scheduler, SchedulerMsg, SharedHostConfig, TriggerFireError,
    build_default_pod_config,
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
    /// Plain config (model, limits, policy) the synthesized default
    /// pod's `thread_defaults` table is built from.
    pub default_task_config: ThreadConfig,
    /// Seed text for the default pod's `system_prompt.md`. Populated
    /// from the `--system-prompt` CLI flag; an empty string means the
    /// default pod starts with no system prompt file (threads inside
    /// it run under an empty system message).
    pub default_system_prompt: String,
    /// Host-env spec + provider pairing for the synthesized default
    /// pod's single `[[allow.host_env]]` entry. `None` means "no host
    /// env" — the default pod has an empty allow.host_env and threads
    /// inside it run with no host-env MCP connection.
    pub default_host_env: Option<(String, HostEnvSpec)>,
    /// Names of shared MCP hosts that should appear in the synthesized
    /// default pod's `[allow].mcp_hosts` and `thread_defaults.mcp_hosts`.
    /// Typically every host configured via `shared_mcp_hosts`.
    pub default_shared_host_names: Vec<String>,
    pub audit_log_path: PathBuf,
    pub host_id: String,
    /// Pods root directory. If `None`, persistence is disabled.
    pub pods_root: Option<PathBuf>,
    /// Host-env provider catalog. Empty registry is a valid config —
    /// threads in such a server just have no host-env MCP connection.
    pub host_env_registry: crate::tools::sandbox::HostEnvRegistry,
    /// Catalog of shared (singleton) MCP hosts the scheduler connects to at
    /// startup. Pods opt in by name via `[allow].mcp_hosts`.
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
    let default_pod_config = build_default_pod_config(
        &default_pod_id,
        &config.default_task_config,
        &config.default_backend,
        config.default_host_env.clone(),
        &backend_names,
        &config.default_shared_host_names,
    );
    let default_pod_dir = config
        .pods_root
        .as_ref()
        .map(|root| root.join(&default_pod_id))
        .unwrap_or_else(|| PathBuf::from(format!("./{default_pod_id}")));
    let default_system_prompt = config.default_system_prompt.clone();
    let raw_toml = crate::pod::to_toml(&default_pod_config)
        .context("encode default pod.toml for in-memory bootstrap")?;
    let default_pod = Pod::new(
        default_pod_id.clone(),
        default_pod_dir.clone(),
        default_pod_config.clone(),
        raw_toml,
        default_system_prompt.clone(),
    );

    let (mut scheduler, stream_rx) = Scheduler::new(
        default_pod,
        config.host_id,
        config.backends,
        config.default_backend,
        audit,
        config.host_env_registry,
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
    let scheduler_handle = tokio::spawn(crate::runtime::scheduler::run(
        scheduler, inbox_rx, stream_rx,
    ));

    let state = AppState {
        inbox: inbox_tx.clone(),
        next_conn_id: Arc::new(AtomicU64::new(1)),
    };

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/ws", get(ws_handler))
        .route(
            "/triggers/{pod_id}/{behavior_id}",
            post(webhook_trigger_handler),
        )
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

/// Webhook trigger delivery. `POST /triggers/{pod_id}/{behavior_id}`
/// with a JSON body (or empty body → Null payload). Minimal v1:
/// no auth, no signature verification — suitable only for loopback
/// or trusted-network use. When remote exposure becomes a real
/// requirement, add HMAC validation here before pushing to the
/// scheduler (the design sketch in `docs/design_behaviors.md` lists
/// a `webhook_secret` shape we can bolt onto the `Webhook` variant).
///
/// Response semantics:
///   - 202 Accepted: trigger dispatched (thread spawned OR payload
///     queued OR skipped per overlap policy — all three are
///     successful deliveries from the sender's perspective).
///   - 400 Bad Request: body is non-empty and not valid JSON.
///   - 404 Not Found: pod or behavior doesn't exist.
///   - 409 Conflict: behavior exists but isn't webhook-triggered, or
///     has a `behavior.toml` load error.
///   - 503 Service Unavailable: scheduler inbox closed (shutting down).
async fn webhook_trigger_handler(
    State(state): State<AppState>,
    Path((pod_id, behavior_id)): Path<(String, String)>,
    body: Bytes,
) -> axum::response::Response {
    let payload = if body.is_empty() {
        serde_json::Value::Null
    } else {
        match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("body must be valid JSON: {e}"),
                )
                    .into_response();
            }
        }
    };

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    let msg = SchedulerMsg::TriggerFired {
        pod_id,
        behavior_id,
        payload,
        reply: reply_tx,
    };
    if state.inbox.send(msg).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "scheduler inbox closed").into_response();
    }

    match reply_rx.await {
        Ok(Ok(())) => StatusCode::ACCEPTED.into_response(),
        Ok(Err(err)) => {
            let status = match &err {
                TriggerFireError::UnknownPod | TriggerFireError::UnknownBehavior => {
                    StatusCode::NOT_FOUND
                }
                TriggerFireError::NotWebhookTrigger(_) | TriggerFireError::BehaviorLoadError(_) => {
                    StatusCode::CONFLICT
                }
                TriggerFireError::Paused => StatusCode::SERVICE_UNAVAILABLE,
            };
            (status, err.to_string()).into_response()
        }
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler dropped trigger request",
        )
            .into_response(),
    }
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
