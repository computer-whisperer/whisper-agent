//! MCP server exposing a single `web_fetch` tool. Singleton daemon — one process
//! shared across whisper-agent tasks, opted into per-task via the scheduler's
//! shared-host allowlist.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::EnvFilter;

mod server;
mod tools;

#[derive(Parser, Debug)]
#[command(version, about = "MCP server exposing a guarded web_fetch tool.")]
struct Args {
    /// HTTP listen address. Defaults to dual-stack (`[::]:9830`); accepts
    /// both IPv6 and IPv4 (via v4-mapped). Override to `127.0.0.1:9830`
    /// for v4-only loopback.
    #[arg(long, default_value = "[::]:9830")]
    listen: SocketAddr,

    /// Hard cap on response body size, in bytes.
    #[arg(long, default_value_t = 5_000_000)]
    max_response_bytes: usize,

    /// Hard cap on request duration (connect + read), in seconds.
    #[arg(long, default_value_t = 30)]
    request_timeout_seconds: u64,

    /// Maximum redirects to follow before giving up.
    #[arg(long, default_value_t = 5)]
    max_redirects: usize,

    /// Allow fetches that resolve to private / loopback / link-local addresses.
    /// Off by default — SSRF mitigation. Turn on only for local-network testing.
    #[arg(long)]
    allow_private_addresses: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let http = reqwest::Client::builder()
        // We enforce redirect count manually so we can re-check SSRF on each hop.
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(args.request_timeout_seconds))
        .user_agent(concat!(
            env!("CARGO_PKG_NAME"),
            "/",
            env!("CARGO_PKG_VERSION")
        ))
        .build()
        .context("build reqwest client")?;

    let state = server::AppState {
        cfg: Arc::new(tools::FetchConfig {
            http,
            max_response_bytes: args.max_response_bytes,
            max_redirects: args.max_redirects,
            allow_private_addresses: args.allow_private_addresses,
        }),
    };

    let app = server::router(state);

    let listener = TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("failed to bind {}", args.listen))?;
    info!(
        addr = %args.listen,
        max_response_bytes = args.max_response_bytes,
        timeout_s = args.request_timeout_seconds,
        max_redirects = args.max_redirects,
        allow_private = args.allow_private_addresses,
        "whisper-agent-mcp-fetch listening"
    );

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
