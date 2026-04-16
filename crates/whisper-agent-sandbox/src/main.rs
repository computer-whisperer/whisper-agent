//! Sandbox provisioning daemon.
//!
//! Listens for provision/teardown requests from the whisper-agent scheduler and
//! manages isolated execution environments. Each provisioned sandbox runs an
//! instance of `whisper-agent-mcp-host` inside it — the scheduler connects MCP
//! to the returned URL.
//!
//! Supported backends (planned):
//!   - **Podman/Docker**: full OCI container isolation.
//!   - **Landlock**: lightweight Linux-native filesystem sandboxing.
//!
//! Currently a scaffold — provision requests are accepted but return
//! "not implemented" until a backend is wired up.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use clap::Parser;
use tokio::signal;
use tokio::sync::Mutex;
use tracing::{info, warn};
use whisper_agent_protocol::sandbox::{ProvisionRequest, SandboxSpec, TeardownRequest};

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
}

/// A live sandbox session.
#[allow(dead_code)]
struct Session {
    task_id: String,
    spec: SandboxSpec,
    mcp_url: String,
    // Future: container ID, child PID, etc.
}

struct DaemonState {
    #[allow(dead_code)]
    args: Args,
    sessions: Mutex<HashMap<String, Session>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "whisper_agent_sandbox=info".into()),
        )
        .init();

    let args = Args::parse();
    let listen = args.listen;

    let state = Arc::new(DaemonState {
        args,
        sessions: Mutex::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/provision", post(handle_provision))
        .route("/teardown", post(handle_teardown))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .with_state(state);

    info!(%listen, "sandbox daemon starting");

    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .expect("bind failed");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    info!("sandbox daemon stopped");
}

async fn handle_provision(
    State(_state): State<Arc<DaemonState>>,
    Json(req): Json<ProvisionRequest>,
) -> impl IntoResponse {
    info!(task_id = %req.task_id, "provision request");

    match &req.spec {
        SandboxSpec::None => {
            // No sandbox needed — shouldn't normally reach the daemon, but
            // handle gracefully.
            warn!(task_id = %req.task_id, "provision called with SandboxSpec::None");
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "SandboxSpec::None does not require provisioning"
                })),
            )
                .into_response()
        }
        SandboxSpec::Container { image, .. } => {
            info!(task_id = %req.task_id, %image, "container provisioning not yet implemented");
            // TODO: podman run --rm -d --name <session_id>
            //       bind-mount mounts, apply network policy, start mcp-host
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({
                    "error": "container provisioning not yet implemented"
                })),
            )
                .into_response()
        }
        SandboxSpec::Landlock { .. } => {
            info!(task_id = %req.task_id, "landlock provisioning not yet implemented");
            // TODO: fork, apply landlock rules, exec mcp-host
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({
                    "error": "landlock provisioning not yet implemented"
                })),
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
        Some(session) => {
            info!(
                session_id = %req.session_id,
                task_id = %session.task_id,
                "session torn down"
            );
            // TODO: podman stop / kill child process
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
