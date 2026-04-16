use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::EnvFilter;

mod server;
mod tools;
mod workspace;

use crate::workspace::Workspace;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "MCP server exposing read_file, write_file, and bash within a workspace root."
)]
struct Args {
    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8800")]
    listen: SocketAddr,

    /// Workspace root. All file operations are confined to this directory.
    #[arg(long)]
    workspace_root: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let workspace = Workspace::new(&args.workspace_root)
        .with_context(|| format!("invalid workspace root {:?}", args.workspace_root))?;

    let state = server::AppState {
        workspace: Arc::new(workspace),
    };
    let app = server::router(state);

    let listener = TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("failed to bind {}", args.listen))?;
    info!(addr = %args.listen, root = %args.workspace_root.display(), "whisper-agent-mcp-host listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server failed")?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal received");
}
