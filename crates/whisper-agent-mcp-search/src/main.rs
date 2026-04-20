//! MCP server exposing a single `web_search` tool backed by the Brave Search
//! API. Singleton daemon — one process shared across whisper-agent tasks,
//! opted into per-task via the scheduler's shared-host allowlist.

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
#[command(
    version,
    about = "MCP server exposing the web_search tool (Brave Search API)."
)]
struct Args {
    /// HTTP listen address. Defaults to dual-stack (`[::]:9831`); accepts
    /// both IPv6 and IPv4 (via v4-mapped). Override to `127.0.0.1:9831`
    /// for v4-only loopback.
    #[arg(long, default_value = "[::]:9831")]
    listen: SocketAddr,

    /// Brave Search API key. Required — the daemon refuses to start without
    /// one because every tool call needs it.
    #[arg(long, env = "BRAVE_API_KEY", hide_env_values = true)]
    api_key: String,

    /// Brave Search API base URL. Override only if you front the API behind a
    /// proxy / mock for testing.
    #[arg(long, default_value = "https://api.search.brave.com/res/v1")]
    api_base: String,

    /// Hard cap on request duration to the upstream search API, in seconds.
    #[arg(long, default_value_t = 15)]
    request_timeout_seconds: u64,

    /// Default `count` used when a tool call doesn't specify one. Brave
    /// caps at 20 per request.
    #[arg(long, default_value_t = 10)]
    default_count: u32,
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
        .timeout(Duration::from_secs(args.request_timeout_seconds))
        .user_agent(concat!(
            env!("CARGO_PKG_NAME"),
            "/",
            env!("CARGO_PKG_VERSION")
        ))
        .build()
        .context("build reqwest client")?;

    let state = server::AppState {
        cfg: Arc::new(tools::SearchConfig {
            http,
            api_key: args.api_key,
            api_base: args.api_base.trim_end_matches('/').to_string(),
            default_count: args.default_count,
        }),
    };

    let app = server::router(state);

    let listener = TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("failed to bind {}", args.listen))?;
    info!(
        version = env!("CARGO_PKG_VERSION"),
        addr = %args.listen,
        timeout_s = args.request_timeout_seconds,
        default_count = args.default_count,
        "whisper-agent-mcp-search listening"
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
