//! axum WebSocket upgrade handler for `/v1/host_env_link`.
//!
//! Mounted in the server's main router behind
//! [`crate::tools::host_env_link::auth::require_daemon_auth`] so the
//! upgrade only fires once the bearer has resolved to an
//! [`crate::tools::host_env_link::AdmittedDaemon`].

use std::sync::Arc;

use axum::extract::{Extension, State, WebSocketUpgrade};
use axum::response::IntoResponse;

use super::HostEnvLinkState;
use super::auth::AdmittedDaemon;
use super::connection;

/// `GET /v1/host_env_link` — upgrade to WebSocket and hand the connection
/// off to the per-daemon task. Returns 101 Switching Protocols on
/// success; the auth middleware has already returned 401 / 503 if the
/// bearer didn't resolve.
pub async fn link_handler(
    State(state): State<Arc<HostEnvLinkState>>,
    Extension(daemon): Extension<AdmittedDaemon>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        connection::run_connection(
            daemon.name,
            socket,
            state.registry.clone(),
            state.scheduler_inbox.clone(),
        )
        .await;
    })
}
