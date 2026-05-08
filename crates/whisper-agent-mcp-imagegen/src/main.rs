//! MCP server exposing a single `image_generate` tool backed by the OpenAI
//! Images API. Singleton daemon — one process shared across whisper-agent
//! tasks, opted into per-task via the scheduler's shared-host allowlist.
//!
//! Auth is read from the same `whisper-agent.toml` the main daemon uses:
//! pass `--backend <name>` to pick which `[backends.X]` entry's auth to use.
//! Both `api_key` and `chatgpt_subscription` (Codex OAuth) modes are
//! supported. The config file is watched live — credential rotations and
//! backend swaps take effect on the next outbound request.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use clap::Parser;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

mod codex;
mod config;
mod server;
mod tools;
mod watcher;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "MCP server exposing the image_generate tool (OpenAI Images API)."
)]
struct Args {
    /// HTTP listen address. Defaults to dual-stack (`[::]:9832`); accepts
    /// both IPv6 and IPv4 (via v4-mapped). Override to `127.0.0.1:9832`
    /// for v4-only loopback.
    #[arg(long, default_value = "[::]:9832")]
    listen: SocketAddr,

    /// Path to `whisper-agent.toml`. If unset, search `$XDG_CONFIG_HOME`,
    /// `$HOME/.config`, and `./` for the same file the main daemon uses.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Which `[backends.X]` entry from `whisper-agent.toml` provides the
    /// OpenAI auth. Must be `kind = "openai_responses"`. Required.
    #[arg(long)]
    backend: String,

    /// Default size (e.g. `1024x1024`, `1536x1024`, `auto`). Passed to the
    /// API verbatim when the tool call doesn't specify one. Daemon-level —
    /// not in the config file.
    #[arg(long, default_value = "auto")]
    default_size: String,

    /// Default quality (`low`, `medium`, `high`, `auto`). Daemon-level.
    #[arg(long, default_value = "auto")]
    default_quality: String,

    /// Hard cap on request duration to the upstream Images API, in seconds.
    /// Image generation is slow on `high` quality and large sizes — defaults
    /// generous.
    #[arg(long, default_value_t = 180)]
    request_timeout_seconds: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // reqwest 0.13's `rustls-no-provider` requires the application to
    // install a CryptoProvider before any Client is built. `.ok()`
    // swallows "already installed" if some library beat us to it.
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_seconds))
        .user_agent(concat!(
            env!("CARGO_PKG_NAME"),
            "/",
            env!("CARGO_PKG_VERSION")
        ))
        .build()
        .context("build reqwest client")?;

    let config_path = config::discover_config_path(args.config)?;
    let mut initial = config::resolve(&config_path, &args.backend)
        .with_context(|| format!("initial resolve from {}", config_path.display()))?;
    config::refresh_chat_model(&http, &mut initial).await;
    let auth_mode = match &initial.auth {
        whisper_agent_auth::ClientAuth::ApiKey(_) => "api_key",
        whisper_agent_auth::ClientAuth::Codex(_) => "chatgpt_subscription",
    };
    info!(
        config = %config_path.display(),
        backend = %args.backend,
        auth_mode,
        api_base = %initial.api_base,
        image_model = %initial.image_model,
        chat_model = %initial.chat_model,
        "resolved initial config"
    );

    let resolved = Arc::new(ArcSwap::from_pointee(initial));

    let cfg = Arc::new(tools::ImageGenConfig {
        http: http.clone(),
        resolved: Arc::clone(&resolved),
        default_size: args.default_size,
        default_quality: args.default_quality,
    });

    // Spawn the watcher before the HTTP server so reload errors during
    // startup land in the same log stream.
    let watcher_path = config_path.clone();
    let watcher_backend = args.backend.clone();
    let watcher_resolved = Arc::clone(&resolved);
    let watcher_http = http.clone();
    tokio::spawn(async move {
        if let Err(e) = watcher::watch(
            watcher_path,
            watcher_backend,
            watcher_resolved,
            watcher_http,
        )
        .await
        {
            error!(error = %e, "config watcher exited");
        }
    });

    let app = server::router(server::AppState {
        cfg: Arc::clone(&cfg),
    });

    let listener = TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("failed to bind {}", args.listen))?;
    info!(
        version = env!("CARGO_PKG_VERSION"),
        addr = %args.listen,
        timeout_s = args.request_timeout_seconds,
        default_size = %cfg.default_size,
        default_quality = %cfg.default_quality,
        "whisper-agent-mcp-imagegen listening"
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
