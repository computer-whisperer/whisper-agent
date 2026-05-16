//! Tool-catalog discovery.
//!
//! On daemon startup we spawn one short-lived probe worker, take its
//! [`whisper_agent_worker_proto::WorkerFrame::Hello`] (which carries
//! the worker's tool catalog directly), and cache it for use in
//! [`whisper_agent_host_proto::Frame::Hello`]. The probe worker is
//! then killed (its `kill_on_drop` does the work).
//!
//! Why probe instead of baking a `const` in the daemon binary? The
//! design doc says "tools change rarely (compiled into the daemon)" —
//! both true here, since the daemon ships in lock-step with the
//! worker. But the worker binary is the actual source of truth, so
//! probing eliminates a second copy of the schema that could drift
//! from the worker's behavior. The probe runs once at startup so it's
//! not on the hot path; if the worker can't be spawned, daemon
//! startup fails loudly — which is what we want.

use std::path::Path;

use tracing::{debug, info};
use whisper_agent_host_proto::ToolDescriptor;
use whisper_agent_protocol::sandbox::{NetworkPolicy, PathAccess};

use crate::worker::{WorkerError, spawn};

/// Errors from [`probe_tool_catalog`]. Today this is just spawn
/// failures; the post-spawn handshake error is folded into
/// [`WorkerError`] (the worker's protocol mismatch / startup timeout
/// / child exit cases all surface there before we ever look at the
/// catalog).
#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("probe worker failed to spawn: {0}")]
    Spawn(#[from] WorkerError),
}

/// Spawn a probe worker against `probe_workspace`, take its Hello,
/// return the advertised tool catalog. The probe worker is killed on
/// return (via `kill_on_drop` when the [`crate::worker::Worker`]
/// drops at the end of this scope).
///
/// `probe_workspace` is an arbitrary directory the daemon can grant
/// the probe read+write access to. The actual filesystem doesn't
/// matter — the probe never reads or writes a file, it only completes
/// the handshake.
pub async fn probe_tool_catalog(
    mcp_host_bin: &str,
    probe_workspace: &str,
) -> Result<Vec<ToolDescriptor>, CatalogError> {
    let spec = whisper_agent_host_proto::HostEnvSpec::Landlock {
        allowed_paths: vec![PathAccess::read_write(probe_workspace)],
        network: NetworkPolicy::Isolated,
    };
    debug!(
        %probe_workspace,
        "spawning probe worker for tool-catalog discovery"
    );
    // The probe runs without a `ThreadContext` (no scheduler attached
    // yet), so it has to supply its own `workspace_root` override —
    // since 9887b84 retired the daemon's spec-fallback heuristic. The
    // override has to live under one of the spec's RW paths, which
    // it does by construction (we grant it RW just above).
    // Probe runs as the daemon's own uid — there's no session, so
    // there's no per-binding `runas` to honor.
    let worker = spawn(
        &spec,
        mcp_host_bin,
        Some(Path::new(probe_workspace)),
        None,
        None,
    )
    .await?;
    info!(count = worker.tools.len(), "probed worker tool catalog");
    // `worker` drops at end of scope → child gets killed via
    // `kill_on_drop`. The frame loop task finishes when the socket
    // EOFs.
    Ok(worker.tools)
}
