//! Sandbox provisioning daemon.
//!
//! Listens for provision/teardown requests from the whisper-agent scheduler and
//! manages isolated execution environments. Each provisioned sandbox runs an
//! instance of `whisper-agent-mcp-host` inside it — the scheduler connects MCP
//! to the returned URL.
//!
//! Supported backends:
//!   - **Landlock**: lightweight Linux-native filesystem + network sandboxing.
//!   - **Podman/Docker** (planned): full OCI container isolation.

mod provision;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use clap::Parser;
use tokio::process::Child;
use tokio::signal;
use tokio::sync::Mutex;
use tracing::{error, info, warn};
use whisper_agent_protocol::sandbox::{
    NetworkPolicy, ProvisionRequest, ProvisionResponse, HostEnvSpec, TeardownRequest,
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
}

struct Session {
    thread_id: String,
    child: Child,
}

struct DaemonState {
    mcp_host_bin: String,
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
        mcp_host_bin: args.mcp_host_bin,
        sessions: Mutex::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/provision", post(handle_provision))
        .route("/teardown", post(handle_teardown))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .with_state(state.clone());

    info!(%listen, "sandbox daemon starting");

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
