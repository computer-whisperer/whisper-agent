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

mod auth;
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
        ConnectInfo, Path, Query, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode, header},
    middleware,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};

use crate::pod::config::AuthClient;
use crate::server::auth::AuthState;
use futures::{SinkExt, StreamExt};
use rust_embed::RustEmbed;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use whisper_agent_protocol::{
    HostEnvSpec, ServerToClient, ThreadConfig, decode_from_client, encode_to_client,
};

use crate::pod::persist::Persister;
use crate::pod::{Pod, PodId};
use crate::runtime::audit::AuditLog;
use crate::runtime::scheduler::{
    BackendEntry, ConnId, Scheduler, SchedulerMsg, SharedHostOverlay, TriggerFireError,
    build_default_pod_config,
};

// Webui assets baked into the release binary by `rust-embed`. In debug
// builds the macro reads from disk on each request (so `cargo run`
// picks up edits to assets/ and fresh wasm-pack output without a
// rebuild); in release builds the file bodies are embedded at compile
// time. The asset folders must exist when cargo compiles this crate —
// `scripts/dev.sh` and the Dockerfile both run wasm-pack before the
// release cargo build, populating `crates/whisper-agent-webui/pkg/`.
#[derive(RustEmbed)]
#[folder = "crates/whisper-agent-webui/pkg/"]
struct WebuiPkg;

#[derive(RustEmbed)]
#[folder = "crates/whisper-agent-webui/assets/"]
struct WebuiAssets;

/// Look up `path` in an embedded asset folder and return it with a
/// content-type guessed from its extension. 404s if the file isn't in
/// the embed (in debug builds, that includes "the file isn't on disk
/// at the configured folder right now").
fn serve_embedded<E: RustEmbed>(path: &str) -> Response {
    match E::get(path) {
        Some(file) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            (
                [(header::CONTENT_TYPE, mime.as_ref().to_string())],
                file.data.into_owned(),
            )
                .into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

pub struct ServerConfig {
    /// Named backends the scheduler can dispatch model calls to.
    pub backends: std::collections::HashMap<String, BackendEntry>,
    /// Named embedding providers from `[embedding_providers.*]`. Empty
    /// when the server boots without any — knowledge buckets become
    /// inaccessible until at least one is registered.
    pub embedding_providers:
        std::collections::HashMap<String, crate::runtime::scheduler::EmbeddingProviderEntry>,
    /// Named rerank providers from `[rerank_providers.*]`.
    pub rerank_providers:
        std::collections::HashMap<String, crate::runtime::scheduler::RerankProviderEntry>,
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
    /// Knowledge buckets root directory. Sibling of `pods_root`. When
    /// `None`, the server boots with an empty bucket registry — chat
    /// threads still work, knowledge-bucket operations don't.
    pub buckets_root: Option<PathBuf>,
    /// Host-env provider catalog. Empty registry is a valid config —
    /// threads in such a server just have no host-env MCP connection.
    pub host_env_registry: crate::tools::sandbox::HostEnvRegistry,
    /// Durable backing store for `host_env_registry`. The scheduler
    /// owns it; every runtime add/update/remove command mutates the
    /// registry and writes the store through. A fresh server with no
    /// prior state starts with an empty store sitting at
    /// `<pods_root>/../host_env_providers.toml`.
    pub host_env_catalog: crate::tools::host_env_catalog::CatalogStore,
    /// Durable catalog of shared MCP hosts. Add/Update/Remove wire
    /// commands mutate this on the scheduler thread.
    pub shared_mcp_catalog: crate::tools::shared_mcp_catalog::CatalogStore,
    /// CLI `--shared-mcp-host name=url` entries the server run was
    /// launched with. These are anonymous, non-persistent, and
    /// shadowed by any catalog entry with the same name.
    pub shared_mcp_overlays: Vec<SharedHostOverlay>,
    /// When set, the server speaks HTTPS instead of plain HTTP on the
    /// listen address. The cert and key are PEM-encoded files; cert
    /// chains with intermediates are supported.
    pub tls: Option<TlsConfig>,
    /// Configured client-auth tokens. Empty means "loopback only" — see
    /// `auth::require_auth` for the gating policy. The list contains full
    /// `AuthClient` entries (name + token) so we can log which device
    /// names are registered at startup; the runtime gate only consults
    /// the tokens.
    pub auth_clients: Vec<AuthClient>,
    /// Configured admin-auth tokens. Admin tokens also grant chat access
    /// (every `matches_client` check is a superset), but additionally
    /// unlock settings-mutation handlers like `UpdateCodexAuth`.
    pub auth_admins: Vec<AuthClient>,
    /// Path to the `whisper-agent.toml` the server was loaded from. `None`
    /// when the server was started without `--config` (env-key-only
    /// fallback) — runtime config-edit endpoints reject in that mode
    /// since there's no on-disk file to round-trip through.
    pub server_config_path: Option<PathBuf>,
}

/// Paths to the PEM-encoded cert chain and private key. Loaded from
/// disk at server startup; rotation requires a process restart (the
/// server does not currently watch the files for changes).
pub struct TlsConfig {
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
}

#[derive(Clone)]
struct AppState {
    inbox: mpsc::UnboundedSender<SchedulerMsg>,
    next_conn_id: Arc<AtomicU64>,
    /// Shared auth state — used by the ws upgrade handler to re-read
    /// the presented token and decide whether the connection has admin
    /// capability. The `require_auth` middleware has already verified
    /// that the token is valid; this probe only refines "plain client"
    /// vs "admin" from the same token bytes.
    auth: Arc<AuthState>,
}

pub async fn serve(listen: SocketAddr, config: ServerConfig) -> anyhow::Result<()> {
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
    // Sorted so the synthesized pod's `thread_defaults.backend` (which
    // picks the first entry) is stable across runs — `config.backends`
    // is a HashMap, so iteration order alone is not.
    let mut backend_names: Vec<String> = config.backends.keys().cloned().collect();
    backend_names.sort();
    let default_pod_config = build_default_pod_config(
        &default_pod_id,
        &config.default_task_config,
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

    let bucket_registry = match config.buckets_root.clone() {
        Some(root) => crate::knowledge::BucketRegistry::load(root.clone())
            .await
            .with_context(|| format!("load buckets_root {}", root.display()))?,
        None => crate::knowledge::BucketRegistry::default(),
    };

    let (mut scheduler, stream_rx) = Scheduler::new(
        default_pod,
        config.host_id,
        config.backends,
        config.embedding_providers,
        config.rerank_providers,
        bucket_registry,
        audit,
        config.host_env_registry,
        config.host_env_catalog,
        config.shared_mcp_catalog,
        config.shared_mcp_overlays,
        config.server_config_path,
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

    let auth_state = Arc::new(AuthState::new(
        config
            .auth_clients
            .iter()
            .map(|c| c.token.clone())
            .collect(),
        config.auth_admins.iter().map(|c| c.token.clone()).collect(),
    ));
    if !auth_state.any_configured() {
        info!(
            "client auth disabled (no [[auth.clients]] or [[auth.admins]] in config) — non-loopback connections will be rejected"
        );
    } else {
        let client_names: Vec<&str> = config
            .auth_clients
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        let admin_names: Vec<&str> = config.auth_admins.iter().map(|c| c.name.as_str()).collect();
        info!(clients = ?client_names, admins = ?admin_names, "client auth enabled");
    }

    let state = AppState {
        inbox: inbox_tx.clone(),
        next_conn_id: Arc::new(AtomicU64::new(1)),
        auth: auth_state.clone(),
    };

    // route_layer applies only to routes registered before it. The /ws and
    // /auth/check routes go through the gate; /auth/login (the way you
    // acquire a session) stays open, as do /health (k8s probe), the
    // webhook trigger surface (separate per-trigger-secret story TBD),
    // and the static webui assets.
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/auth/check", get(auth::handle_check))
        .route_layer(middleware::from_fn_with_state(
            auth_state.clone(),
            auth::require_auth,
        ))
        .route("/health", get(|| async { "ok" }))
        .route(
            "/auth/login",
            post(auth::handle_login).with_state(auth_state),
        )
        .route(
            "/triggers/{pod_id}/{behavior_id}",
            post(webhook_trigger_handler),
        )
        .route("/oauth/callback", get(oauth_callback_handler))
        .route(
            "/pkg/{*path}",
            get(|Path(p): Path<String>| async move { serve_embedded::<WebuiPkg>(&p) }),
        )
        .route(
            "/assets/{*path}",
            get(|Path(p): Path<String>| async move { serve_embedded::<WebuiAssets>(&p) }),
        )
        .route(
            "/",
            get(|| async { serve_embedded::<WebuiAssets>("index.html") }),
        )
        .route(
            "/index.html",
            get(|| async { serve_embedded::<WebuiAssets>("index.html") }),
        )
        .route(
            "/login.html",
            get(|| async { serve_embedded::<WebuiAssets>("login.html") }),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let serve_result = match config.tls {
        Some(tls) => {
            let rustls =
                axum_server::tls_rustls::RustlsConfig::from_pem_file(&tls.cert_path, &tls.key_path)
                    .await
                    .with_context(|| {
                        format!(
                            "load tls cert={} key={}",
                            tls.cert_path.display(),
                            tls.key_path.display()
                        )
                    })?;
            info!(
                version = env!("CARGO_PKG_VERSION"),
                addr = %listen,
                cert = %tls.cert_path.display(),
                "whisper-agent server listening (open https://{listen}/ in a browser)"
            );
            axum_server::bind_rustls(listen, rustls)
                .serve(app.into_make_service_with_connect_info::<SocketAddr>())
                .await
        }
        None => {
            let listener = TcpListener::bind(listen)
                .await
                .with_context(|| format!("bind {listen}"))?;
            info!(
                version = env!("CARGO_PKG_VERSION"),
                addr = %listen,
                "whisper-agent server listening (open http://{listen}/ in a browser)"
            );
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
        }
    };

    // Shutdown: drop the inbox sender so the scheduler exits, then await it.
    drop(inbox_tx);
    let _ = scheduler_handle.await;
    serve_result?;
    Ok(())
}

/// Query params on `GET /oauth/callback`. The authorization server
/// redirects the user's browser here after consent; we forward the
/// `state` + `code` (or `error`) to the scheduler via the inbox and
/// serve a static page telling the user they can close the tab. The
/// final success/failure lands on the webui WS connection that
/// started the flow.
#[derive(serde::Deserialize, Debug)]
struct OauthCallbackQuery {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    error_description: Option<String>,
}

/// `GET /oauth/callback` — unauthenticated; lives after
/// `route_layer(require_auth)` so the OAuth provider's browser
/// redirect doesn't need to carry our bearer. The `state` parameter
/// (CSRF-bound + minted server-side) is our identity check.
async fn oauth_callback_handler(
    State(state): State<AppState>,
    Query(q): Query<OauthCallbackQuery>,
) -> impl IntoResponse {
    let Some(state_param) = q.state else {
        return (StatusCode::BAD_REQUEST, "missing `state` query parameter").into_response();
    };
    // Combine `error` + `error_description` into a single readable
    // string so the scheduler's reply to the webui surfaces both.
    let combined_error = match (q.error.as_deref(), q.error_description.as_deref()) {
        (Some(e), Some(d)) => Some(format!("{e}: {d}")),
        (Some(e), None) => Some(e.to_string()),
        (None, Some(d)) => Some(d.to_string()),
        (None, None) => None,
    };
    let msg = SchedulerMsg::OauthCallback {
        state: state_param,
        code: q.code,
        error: combined_error,
    };
    if state.inbox.send(msg).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "scheduler inbox closed").into_response();
    }
    // Minimal "you can close this tab" page. We don't try to
    // auto-close — many browsers block script-driven close on tabs
    // the script didn't open — so a manual close with a clear
    // explanation is the least-surprising UX.
    Html(
        r#"<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Authorization complete</title></head>
  <body style="font-family: system-ui, sans-serif; max-width: 32rem; margin: 3rem auto; line-height: 1.5;">
    <h1>Authorization complete</h1>
    <p>You can close this tab and return to whisper-agent.</p>
  </body>
</html>"#,
    )
    .into_response()
}

async fn ws_handler(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    // The `require_auth` middleware has already approved the request.
    // Re-read the presented token here to decide whether the connection
    // is admin-capable. Loopback peers are promoted to admin
    // unconditionally — they can already read the config from disk, so
    // the gate would be theatre.
    let is_admin = if addr.ip().is_loopback() {
        true
    } else {
        auth::bearer_or_cookie(&headers)
            .map(|t| state.auth.matches_admin(t.as_bytes()))
            .unwrap_or(false)
    };
    ws.on_upgrade(move |socket| handle_ws_session(socket, state, is_admin))
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

async fn handle_ws_session(socket: WebSocket, state: AppState, is_admin: bool) {
    let conn_id: ConnId = state.next_conn_id.fetch_add(1, Ordering::Relaxed);
    let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<ServerToClient>();

    // Register with the scheduler.
    if state
        .inbox
        .send(SchedulerMsg::RegisterClient {
            conn_id,
            outbound: outbound_tx,
            is_admin,
        })
        .is_err()
    {
        error!("scheduler inbox closed; cannot register client");
        return;
    }
    info!(conn_id, is_admin, "ws session opened");

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
