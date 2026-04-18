//! Sandbox provisioning daemon.
//!
//! Listens for provision/teardown requests from the whisper-agent scheduler and
//! manages isolated execution environments. Each provisioned sandbox runs an
//! instance of `whisper-agent-mcp-host` inside it — the scheduler connects MCP
//! to the returned URL with the per-sandbox bearer token included in the
//! response.
//!
//! Auth: the control plane requires `Authorization: Bearer <token>` on
//! `/provision` and `/teardown`. The expected token is loaded from
//! `--control-token-file` at startup (a static per-dispatcher secret the
//! operator manages). `/health` is always open for liveness probing.
//! `--no-auth` exists as an explicit dev opt-in so a missing token never
//! silently leaves the daemon unauthenticated.
//!
//! Supported backends:
//!   - **Landlock**: lightweight Linux-native filesystem + network sandboxing.
//!   - **Podman/Docker** (planned): full OCI container isolation.

mod provision;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use clap::Parser;
use subtle::ConstantTimeEq;
use tokio::process::Child;
use tokio::signal;
use tokio::sync::Mutex;
use tracing::{error, info, warn};
use whisper_agent_protocol::sandbox::{
    HostEnvSpec, NetworkPolicy, ProvisionRequest, ProvisionResponse, TeardownRequest,
};

#[derive(Parser, Debug)]
#[command(name = "whisper-agent-sandbox", about = "Sandbox provisioning daemon")]
struct Args {
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:9810")]
    listen: SocketAddr,

    /// Path to the whisper-agent-mcp-host binary. The daemon starts this
    /// inside each provisioned sandbox.
    #[arg(long, default_value = "whisper-agent-mcp-host")]
    mcp_host_bin: String,

    /// Path to a file containing the control-plane bearer token (single
    /// line, whitespace trimmed). Every `/provision` and `/teardown`
    /// request must present it as `Authorization: Bearer <token>`;
    /// `/health` is always open. Rotate by replacing the file and
    /// restarting the daemon.
    #[arg(long, conflicts_with = "no_auth")]
    control_token_file: Option<PathBuf>,

    /// Run without control-plane auth — any local or network caller
    /// that can reach the listen address can provision sandboxes and
    /// drive MCP tools inside them. Explicit opt-in so a missing
    /// `--control-token-file` fails loudly instead of silently opening
    /// a root-equivalent RPC surface.
    #[arg(long, conflicts_with = "control_token_file")]
    no_auth: bool,
}

struct Session {
    thread_id: String,
    child: Child,
}

struct DaemonState {
    mcp_host_bin: String,
    sessions: Mutex<HashMap<String, Session>>,
    /// Expected control-plane bearer. `None` means `--no-auth`.
    control_token: Option<Arc<String>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "whisper_agent_sandbox=info".into()),
        )
        .init();

    let args = Args::parse();

    if args.control_token_file.is_none() && !args.no_auth {
        return Err(anyhow!(
            "must pass exactly one of --control-token-file <path> or --no-auth"
        ));
    }

    let control_token = match &args.control_token_file {
        Some(path) => {
            let raw = tokio::fs::read_to_string(path)
                .await
                .with_context(|| format!("reading control token from {}", path.display()))?;
            let tok = raw.trim().to_string();
            if tok.is_empty() {
                return Err(anyhow!(
                    "control token file {} is empty after trim",
                    path.display()
                ));
            }
            Some(Arc::new(tok))
        }
        None => None,
    };

    let listen = args.listen;
    let state = Arc::new(DaemonState {
        mcp_host_bin: args.mcp_host_bin,
        sessions: Mutex::new(HashMap::new()),
        control_token,
    });

    // /health is deliberately exempt from auth so liveness/readiness
    // probes can stay dumb. Everything destructive lives under the
    // auth layer.
    let control_plane = Router::new()
        .route("/provision", post(handle_provision))
        .route("/teardown", post(handle_teardown))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_bearer,
        ));
    let app = Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .merge(control_plane)
        .with_state(state.clone());

    if state.control_token.is_some() {
        info!(%listen, "sandbox daemon starting (bearer auth)");
    } else {
        warn!(%listen, "sandbox daemon starting (NO AUTH — --no-auth was set)");
    }

    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .expect("bind failed");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    // Clean up all sessions on shutdown.
    let mut sessions = state.sessions.lock().await;
    for (id, mut session) in sessions.drain() {
        info!(session_id = %id, thread_id = %session.thread_id, "killing orphaned session");
        let _ = session.child.kill().await;
    }

    info!("sandbox daemon stopped");
    Ok(())
}

/// Axum middleware: require `Authorization: Bearer <token>` matching
/// the daemon's configured control token. Pass-through when no token
/// is configured (i.e. `--no-auth`).
async fn require_bearer(
    State(state): State<Arc<DaemonState>>,
    headers: HeaderMap,
    req: axum::extract::Request,
    next: Next,
) -> axum::response::Response {
    if let Some(expected) = &state.control_token {
        let presented = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .unwrap_or("");
        if presented.as_bytes().ct_eq(expected.as_bytes()).unwrap_u8() != 1 {
            return (StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response();
        }
    }
    next.run(req).await
}

async fn handle_provision(
    State(state): State<Arc<DaemonState>>,
    Json(req): Json<ProvisionRequest>,
) -> impl IntoResponse {
    info!(thread_id = %req.thread_id, "provision request");

    match &req.spec {
        HostEnvSpec::Container { image, .. } => {
            info!(thread_id = %req.thread_id, %image, "container provisioning not yet implemented");
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({
                    "error": "container provisioning not yet implemented"
                })),
            )
                .into_response();
        }
        HostEnvSpec::Landlock { network, .. } => {
            if matches!(network, NetworkPolicy::AllowList { .. }) {
                warn!(
                    thread_id = %req.thread_id,
                    "landlock cannot do per-host network filtering; \
                     AllowList will be treated as Isolated"
                );
            }
        }
    }

    match provision::provision(&req.spec, &state.mcp_host_bin).await {
        Ok(session) => {
            let session_id = req.thread_id.clone();
            let mcp_url = session.mcp_url.clone();
            let mcp_token = session.mcp_token.clone();

            let mut sessions = state.sessions.lock().await;
            sessions.insert(
                session_id.clone(),
                Session {
                    thread_id: req.thread_id,
                    child: session.child,
                },
            );

            (
                StatusCode::OK,
                Json(
                    serde_json::to_value(ProvisionResponse {
                        session_id,
                        mcp_url,
                        mcp_token,
                    })
                    .unwrap(),
                ),
            )
                .into_response()
        }
        Err(e) => {
            error!(thread_id = %req.thread_id, error = %e, "provision failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

async fn handle_teardown(
    State(state): State<Arc<DaemonState>>,
    Json(req): Json<TeardownRequest>,
) -> impl IntoResponse {
    info!(session_id = %req.session_id, "teardown request");

    let mut sessions = state.sessions.lock().await;
    match sessions.remove(&req.session_id) {
        Some(mut session) => {
            info!(
                session_id = %req.session_id,
                thread_id = %session.thread_id,
                "tearing down session"
            );
            if let Err(e) = session.child.kill().await {
                warn!(
                    session_id = %req.session_id,
                    error = %e,
                    "failed to kill child (may have already exited)"
                );
            }
            StatusCode::OK.into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no session with id {}", req.session_id)
            })),
        )
            .into_response(),
    }
}

async fn shutdown_signal() {
    let ctrl_c = async { signal::ctrl_c().await.expect("ctrl+c handler") };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => info!("received ctrl+c"),
        () = terminate => info!("received SIGTERM"),
    }
}
