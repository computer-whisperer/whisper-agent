use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use clap::Parser;
use tokio::io::{AsyncBufReadExt, BufReader};
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
    /// HTTP listen address. Defaults to dual-stack (`[::]:8800`) when run
    /// standalone. Production use is via `whisper-agent-sandbox`, which
    /// always passes an explicit `--listen 127.0.0.1:<port>` (one mcp-host
    /// per landlock-isolated thread, loopback-only by design).
    #[arg(long, default_value = "[::]:8800")]
    listen: SocketAddr,

    /// Workspace root. All file operations are confined to this directory.
    #[arg(long)]
    workspace_root: PathBuf,

    /// Read a bearer token from a single line on stdin at startup; every
    /// request to `/mcp` must then present it as `Authorization: Bearer
    /// <token>`. The launching process (typically
    /// whisper-agent-sandbox) writes the token once and closes stdin.
    /// Stdin is used so the token never appears in argv or the
    /// environment, both of which are readable via `/proc/<pid>/...`
    /// under the same uid.
    #[arg(long, conflicts_with = "no_auth")]
    token_stdin: bool,

    /// Run with no authentication — any caller that can reach the
    /// listen address can invoke tools. Explicit opt-in so a
    /// misconfiguration cannot silently leave the server unauthenticated.
    /// Intended only for one-off local dev; real deployments use
    /// `--token-stdin`.
    #[arg(long, conflicts_with = "token_stdin")]
    no_auth: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    if !args.token_stdin && !args.no_auth {
        return Err(anyhow!(
            "must pass exactly one of --token-stdin or --no-auth"
        ));
    }

    let expected_token = if args.token_stdin {
        let mut reader = BufReader::new(tokio::io::stdin());
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .await
            .context("reading token from stdin")?;
        if n == 0 {
            return Err(anyhow!("stdin closed before a token was read"));
        }
        let token = line.trim().to_string();
        if token.is_empty() {
            return Err(anyhow!("stdin supplied an empty token"));
        }
        Some(token)
    } else {
        None
    };

    let workspace = Workspace::new(&args.workspace_root)
        .with_context(|| format!("invalid workspace root {:?}", args.workspace_root))?;

    let state = server::AppState {
        workspace: Arc::new(workspace),
        expected_token: expected_token.map(Arc::new),
    };
    let app = server::router(state);

    let listener = TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("failed to bind {}", args.listen))?;
    if args.no_auth {
        info!(version = env!("CARGO_PKG_VERSION"), addr = %args.listen, root = %args.workspace_root.display(), "whisper-agent-mcp-host listening (NO AUTH)");
    } else {
        info!(version = env!("CARGO_PKG_VERSION"), addr = %args.listen, root = %args.workspace_root.display(), "whisper-agent-mcp-host listening (bearer auth)");
    }

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
