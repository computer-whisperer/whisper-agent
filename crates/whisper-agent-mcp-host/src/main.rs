//! `whisper-agent-mcp-host` worker binary.
//!
//! Spawned by `whisper-agent-host-daemon` once per session, inside a
//! landlock sandbox. Communicates with its parent over a Unix
//! `socketpair` the daemon dup'd to FD 3 at spawn — see
//! `whisper-agent-worker-proto` for the wire and `crate::ipc` for
//! the local end of it.
//!
//! No HTTP server, no listening port, no bearer token: the worker is
//! strictly daemon-private and the OS authenticates the channel
//! (you have FD 3, you're trusted). The crate name still carries
//! "mcp-host" for hysterical-raisins reasons; see
//! `docs/design_host_env_protocol.md` §Phase 7 for the rename plan.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

mod ipc;
mod tools;
mod workspace;

use crate::workspace::Workspace;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Per-session worker for whisper-agent-host-daemon. Speaks worker-proto over FD 3."
)]
struct Args {
    /// Workspace root. All file operations are confined to this
    /// directory. Daemon-supplied per spawn.
    #[arg(long)]
    workspace_root: PathBuf,

    /// Daemon-supplied default + ceiling for the bash tool's
    /// `timeout_seconds`. `None` ⇒ bash falls back to its built-in
    /// 120 s default. Comes from the scheduler's
    /// `ThreadContext.bash_timeout_secs`, plumbed through the daemon
    /// at session spawn.
    #[arg(long)]
    default_bash_timeout_secs: Option<u32>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        // Worker logs go to stderr; the daemon captures and tail-buffers
        // them for failure reporting (see `worker.rs::drain_stderr`).
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    let workspace = Workspace::new(&args.workspace_root)
        .with_context(|| format!("invalid workspace root {:?}", args.workspace_root))?
        .with_default_bash_timeout_secs(args.default_bash_timeout_secs);
    info!(
        version = env!("CARGO_PKG_VERSION"),
        root = %args.workspace_root.display(),
        default_bash_timeout_secs = ?args.default_bash_timeout_secs,
        "whisper-agent-mcp-host starting"
    );

    ipc::run(Arc::new(workspace)).await
}
