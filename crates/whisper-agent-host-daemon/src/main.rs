//! `whisper-agent-host-daemon` binary.
//!
//! Long-lived process that dials whisper-agent over an authenticated
//! WebSocket and provisions per-session sandboxes for incoming
//! tool calls. Reconnect-with-backoff is at this layer; the library
//! gives us a clean error per dial, we log it, sleep, retry.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, anyhow};
use clap::Parser;
use rand::RngExt;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use whisper_agent_host_daemon::{ConnectError, catalog, config::DaemonConfig, connection};
use whisper_agent_host_proto::{DaemonCapabilities, HostEnvSpecKind};

/// Default location of the daemon's TOML config — matches the
/// existing `host-daemon.env` location used by the AUR systemd unit.
/// An absent default-path file is a soft no-op; an explicit
/// `--config` pointing at a missing file errors out.
const DEFAULT_CONFIG_PATH: &str = "/etc/whisper-agent/host-daemon.toml";

#[derive(Parser, Debug)]
#[command(
    name = "whisper-agent-host-daemon",
    version,
    about = "Dial-in host daemon for the v2 host-env protocol"
)]
struct Args {
    /// Full WebSocket URL of the scheduler endpoint, including scheme
    /// and path. `wss://` for production (TLS via system roots),
    /// `ws://` for loopback dev.
    #[arg(long, env = "WHISPER_AGENT_SERVER_URL")]
    server_url: String,

    /// Path to the bearer-token file the daemon presents at WS
    /// upgrade. Single line, whitespace-trimmed. Must match a
    /// `[[auth.daemons]]` entry on the scheduler.
    #[arg(long, env = "WHISPER_AGENT_TOKEN_FILE")]
    token_file: PathBuf,

    /// Path to the `whisper-agent-mcp-host` binary the daemon spawns
    /// for each session. Resolved against `$PATH` if not absolute.
    /// The daemon ↔ worker hop is a Unix socketpair (handed to the
    /// child as FD 3 at spawn) — there is no port allocation, so no
    /// bind-IP knob.
    #[arg(long, default_value = "whisper-agent-mcp-host")]
    mcp_host_bin: String,

    /// Optional override for the workspace handed to the startup
    /// probe worker. Defaults to a fresh tempdir under `$TMPDIR` —
    /// override only when probing in a writable-elsewhere setup.
    #[arg(long)]
    probe_workspace: Option<PathBuf>,

    /// Reconnect backoff ceiling. The dial loop uses exponential
    /// backoff with jitter, capped at this many seconds.
    #[arg(long, default_value_t = 60)]
    reconnect_max_secs: u64,

    /// Daemon-side TOML config. Today's only knob is the list of
    /// credential files to manage and publish (e.g. the user's
    /// `~/.codex/auth.json`). Optional: if the default path doesn't
    /// exist the daemon runs without publishing credentials; an
    /// explicitly-named missing file errors out.
    #[arg(long, env = "WHISPER_AGENT_HOST_DAEMON_CONFIG")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("whisper_agent_host_daemon=info")),
        )
        .init();

    let args = Args::parse();

    // rustls 0.23 panics on the first TLS use unless a provider is
    // installed as the process default. Match the pattern used by the
    // other binary crates in this workspace. `.ok()` swallows
    // "already installed" if a future dep beats us to it.
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let token = tokio::fs::read_to_string(&args.token_file)
        .await
        .with_context(|| format!("reading token from {}", args.token_file.display()))?;
    let token = token.trim().to_string();
    if token.is_empty() {
        return Err(anyhow!(
            "token file {} is empty after trim",
            args.token_file.display()
        ));
    }

    // Probe the worker once to learn what it advertises. Failure here
    // is fatal — if we can't talk to the worker locally, we have no
    // business announcing capabilities to the scheduler. The probe
    // workspace must outlive the probe call; we hold it in this scope
    // and let it drop after the catalog is built.
    let _probe_dir_guard;
    let probe_ws_path: String = match args.probe_workspace.as_ref() {
        Some(p) => p.to_string_lossy().into_owned(),
        None => {
            let dir = tempfile::tempdir().context("creating probe-workspace tempdir")?;
            let path = dir.path().to_string_lossy().into_owned();
            _probe_dir_guard = dir;
            path
        }
    };
    let tools = catalog::probe_tool_catalog(&args.mcp_host_bin, &probe_ws_path)
        .await
        .context("probing worker tool catalog")?;
    let capabilities = DaemonCapabilities {
        tools,
        spec_kinds: vec![HostEnvSpecKind::Landlock],
        max_concurrent_sessions: None,
        supports_background_tasks: false,
    };

    // Load the daemon TOML config. The default path is treated as
    // soft-missing (an unconfigured daemon still works); an explicit
    // `--config` against a missing file is fatal — the operator
    // clearly meant something.
    let (config_path, allow_missing) = match args.config.clone() {
        Some(p) => (p, false),
        None => (PathBuf::from(DEFAULT_CONFIG_PATH), true),
    };
    let daemon_config = DaemonConfig::load(&config_path, allow_missing)
        .with_context(|| format!("load daemon config {}", config_path.display()))?;
    if !daemon_config.publish_credential.is_empty() {
        info!(
            entries = daemon_config.publish_credential.len(),
            path = %config_path.display(),
            "credential publishers configured"
        );
    }

    run_loop(args, token, capabilities, daemon_config).await
}

async fn run_loop(
    args: Args,
    token: String,
    capabilities: DaemonCapabilities,
    daemon_config: DaemonConfig,
) -> anyhow::Result<()> {
    let mut backoff_secs = 1u64;
    info!(
        url = %args.server_url,
        tools = capabilities.tools.len(),
        publishers = daemon_config.publish_credential.len(),
        "host-env daemon starting"
    );
    loop {
        let config = connection::ConnectionConfig {
            server_url: args.server_url.clone(),
            token: token.clone(),
            daemon_version: env!("CARGO_PKG_VERSION").into(),
            capabilities: capabilities.clone(),
            mcp_host_bin: args.mcp_host_bin.clone(),
            tls_connector: None,
            publish_credentials: daemon_config.publish_credential.clone(),
        };
        match connection::run_connection(config).await {
            Ok(()) => {
                info!("link closed cleanly, reconnecting after backoff");
                backoff_secs = 1; // reset on clean close
            }
            Err(e) => match &e {
                ConnectError::ProtocolMismatch { .. } => {
                    error!(error = %e, "fatal: protocol mismatch — exiting");
                    return Err(anyhow!(e.to_string()));
                }
                ConnectError::Goodbye { reason, .. } => {
                    use whisper_agent_host_proto::GoodbyeReason::*;
                    match reason {
                        ProtocolMismatch | Unauthorized | NameAlreadyConnected => {
                            error!(error = %e, "fatal: scheduler refused connection — exiting");
                            return Err(anyhow!(e.to_string()));
                        }
                        // `Superseded` means the scheduler treated our
                        // prior connection as dead (heartbeat timeout)
                        // and admitted a fresh one in its place. From
                        // this daemon's view, the relevant connection
                        // just ended — retry with backoff to make sure
                        // we have a live socket again.
                        Superseded | ServerShutdown | DaemonShutdown | Other => {
                            warn!(error = %e, "scheduler said goodbye, will reconnect");
                        }
                    }
                }
                _ => warn!(error = %e, "connection failed, will reconnect"),
            },
        }
        let jitter = rand::rng().random_range(0..1000);
        let sleep_ms = backoff_secs * 1000 + jitter;
        info!(sleep_ms, "reconnect backoff");
        tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
        backoff_secs = (backoff_secs * 2).min(args.reconnect_max_secs);
    }
}
