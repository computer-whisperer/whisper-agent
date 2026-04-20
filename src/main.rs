use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;
use whisper_agent_protocol::ThreadConfig;
use whisper_agent_protocol::sandbox::{HostEnvSpec, NetworkPolicy, PathAccess};

use whisper_agent::pod::config::Config;
use whisper_agent::providers::anthropic::AnthropicClient;
use whisper_agent::runtime::scheduler::{BackendEntry, SharedHostConfig};
use whisper_agent::server::{self, ServerConfig};
use whisper_agent::tools::sandbox::{HostEnvProviderEntry, HostEnvRegistry};

// Seed for a fresh default pod's `system_prompt.md`. A neutral
// general-assistant baseline — domain-specific pods should clone the
// default pod and rewrite this file, or override per-thread via the
// `system_prompt` field on `ThreadConfigOverride` / the [thread]
// table in behavior.toml. Deliberately free of software-engineering
// assumptions: the old "You are a software engineering agent" default
// silently biased behavior-spawned threads toward build-tools / edit
// actions even when the behavior's prompt.md said something else.
const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a helpful AI assistant. Use the available tools to investigate, reason about, and \
complete the user's requests.

Communicate concisely. Before your first tool call, say in one short sentence what you're \
about to do. Give brief updates when you find something important, change direction, or hit \
a blocker. End your turn with a one-to-two-sentence summary of what changed and what's next.

Cite sources you reference — for code, use `path:line` so the user can jump to it.
";
const DEFAULT_BACKEND_NAME: &str = "anthropic";

#[derive(Parser, Debug)]
#[command(
    version,
    about = "whisper-agent: headless agent loop with embedded webui server."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
// Serve holds the full ServeArgs (~288 B) while Config is tiny (~24 B).
// The enum is constructed once during CLI parsing and discarded after
// the match, so boxing the large variant would add an alloc without
// saving memory anywhere that matters.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Start the HTTP server: hosts the webui assets and drives the task scheduler.
    /// Clients subscribe to individual tasks via the multiplexed WebSocket protocol.
    Serve(ServeArgs),
    /// Introspect the resolved config (secrets export, path discovery, ...).
    Config(ConfigCmd),
}

#[derive(Parser, Debug)]
struct ConfigCmd {
    #[command(subcommand)]
    action: ConfigAction,
}

#[derive(Subcommand, Debug)]
enum ConfigAction {
    /// Emit `export KEY='VALUE'` lines for every entry in the [secrets] table.
    /// Intended for shell scripts that launch sibling daemons:
    ///   `eval "$(whisper-agent config env)"`
    Env(ConfigEnvArgs),
}

#[derive(Parser, Debug)]
struct ConfigEnvArgs {
    /// Override the config path. Same precedence as `serve --config`.
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct ServeArgs {
    /// HTTP listen address. Defaults to dual-stack (`[::]:8080`) so the
    /// server is reachable on both IPv6 and IPv4 (via v4-mapped) without
    /// extra config. Override to `127.0.0.1:8080` to bind v4-only.
    #[arg(long, default_value = "[::]:8080")]
    listen: SocketAddr,

    /// Path to a PEM-encoded TLS certificate chain. When set together
    /// with `--tls-key`, the server speaks HTTPS instead of HTTP on
    /// the same listen address. Both flags must be set together;
    /// passing only one is a configuration error.
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<PathBuf>,

    /// Path to a PEM-encoded TLS private key. Pairs with `--tls-cert`.
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<PathBuf>,

    /// Path to a TOML config file describing the model-backend catalog. If omitted,
    /// the server searches (in order): `$XDG_CONFIG_HOME/whisper-agent/whisper-agent.toml`,
    /// `$HOME/.config/whisper-agent/whisper-agent.toml`, then `./whisper-agent.toml`.
    /// If none is found, it falls back to a single-backend "anthropic" config built
    /// from `--anthropic-api-key` + `--model`.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Anthropic API key (used only when `--config` is not provided).
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    anthropic_api_key: Option<String>,

    /// Anthropic model ID (used only when `--config` is not provided).
    #[arg(long, default_value = "claude-sonnet-4-6")]
    model: String,

    /// Path to the audit log file.
    #[arg(long, default_value = "audit.jsonl")]
    audit_log: PathBuf,

    /// Pods root directory. Each pod is a subdirectory containing a
    /// `pod.toml` and a `threads/` folder of per-thread JSON. Pass an
    /// empty string to disable persistence.
    #[arg(long, default_value = "pods")]
    pods_root: PathBuf,

    /// System prompt to send to the model.
    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system_prompt: String,

    /// Maximum number of model turns per user message.
    #[arg(long, default_value_t = 30)]
    max_turns: u32,

    /// max_tokens parameter passed to the model backend. Generous by default
    /// because a large `write_file` or multi-edit can easily exceed 4k tokens,
    /// and the model truncates mid-tool-call if it runs out of budget.
    #[arg(long, default_value_t = 16384)]
    max_tokens: u32,

    /// Deprecated — the `--prompt-destructive` flag previously selected
    /// an `ApprovalPolicy` preset. Approval policy is now expressed per-pod
    /// as `[allow.tools]` in pod.toml (default disposition + per-tool
    /// overrides). Set this flag on the command line and it's ignored.
    #[arg(long, hide = true)]
    #[allow(dead_code)]
    prompt_destructive: bool,

    /// Register a host-env provider from the CLI: `name=url`. Each
    /// catalog entry is a `whisper-agent-sandbox`-shaped daemon.
    /// Repeatable. Also configurable in TOML:
    /// `[[host_env_providers]] name = "...", url = "..."`. A server
    /// with zero providers is valid — threads in it just have no
    /// host-env MCP connection.
    #[arg(long = "host-env-provider", value_parser = parse_host_env_provider_arg)]
    host_env_providers: Vec<(String, String)>,

    /// Attach a control-plane bearer token file to a CLI-registered
    /// provider. Format: `name=path`. Repeatable. Pairs with
    /// `--host-env-provider` by name. TOML-registered providers use
    /// the `token_file` key on their `[[host_env_providers]]` entry
    /// instead.
    #[arg(long = "host-env-provider-token", value_parser = parse_host_env_provider_token_arg)]
    host_env_provider_tokens: Vec<(String, PathBuf)>,

    /// Name a provider from the catalog for the synthesized default
    /// pod's single `[[allow.host_env]]` entry. Paired with
    /// `--default-host-env-workspace`. When omitted, the default pod
    /// has no host_env configured and fresh threads inside it run
    /// without a host-env MCP (tool catalog is shared-only).
    #[arg(long)]
    default_host_env_provider: Option<String>,

    /// Default the synthesized pod to a Landlock host env rooted at
    /// this workspace. Auto-includes ~/.cargo and ~/.rustup read-only
    /// if present. Requires --default-host-env-provider.
    #[arg(long)]
    default_host_env_workspace: Option<PathBuf>,

    /// Register a singleton MCP host the scheduler should connect to at
    /// startup. Format: `name=url`. Repeatable. Tasks opt in by listing the
    /// names in their `shared_mcp_hosts` field; the server's
    /// `default_task_config` includes every name configured here, so fresh
    /// tasks default to using all of them.
    ///
    /// Example: `--shared-mcp-host fetch=http://127.0.0.1:9830/mcp`.
    /// Also configurable in TOML: `[shared_mcp_hosts] fetch = "http://..."`.
    #[arg(long = "shared-mcp-host", value_parser = parse_shared_host_arg)]
    shared_mcp_hosts: Vec<(String, String)>,
}

/// Parse a `name=url` pair for `--host-env-provider`. Same shape as
/// `--shared-mcp-host`.
fn parse_host_env_provider_arg(s: &str) -> Result<(String, String), String> {
    parse_shared_host_arg(s)
}

/// Parse a `name=path` pair for `--host-env-provider-token`. The path
/// is intentionally not checked for existence here — the token file
/// is read at registry construction, which is where the clearer error
/// lives.
fn parse_host_env_provider_token_arg(s: &str) -> Result<(String, PathBuf), String> {
    let (name, path) = s
        .split_once('=')
        .ok_or_else(|| "expected `name=path`".to_string())?;
    if name.is_empty() {
        return Err("name must be non-empty".into());
    }
    if path.is_empty() {
        return Err("path must be non-empty".into());
    }
    Ok((name.to_string(), PathBuf::from(path)))
}

/// Resolve which TOML config file to load. Precedence:
///   1. `--config <path>` — explicit, errors if the path is missing.
///   2. `$XDG_CONFIG_HOME/whisper-agent/whisper-agent.toml`
///   3. `$HOME/.config/whisper-agent/whisper-agent.toml`
///   4. `./whisper-agent.toml` (legacy dev-loop fallback).
///
/// Returns `None` only when no flag was given and none of the default paths
/// exist — callers then fall through to the single-backend CLI-arg path.
fn resolve_config_path(explicit: Option<PathBuf>) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        if !path.exists() {
            return Err(anyhow!("--config {} does not exist", path.display()));
        }
        return Ok(Some(path));
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        candidates.push(PathBuf::from(xdg).join("whisper-agent/whisper-agent.toml"));
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        candidates.push(PathBuf::from(home).join(".config/whisper-agent/whisper-agent.toml"));
    }
    candidates.push(PathBuf::from("whisper-agent.toml"));

    // Dedup — when XDG_CONFIG_HOME is unset or equal to $HOME/.config the XDG
    // and HOME paths collapse to the same file.
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    for p in candidates {
        if !seen.insert(p.clone()) {
            continue;
        }
        if p.exists() {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

/// Parse a `name=url` pair for `--shared-mcp-host`. Names must be non-empty
/// and free of `=` so future syntax extensions stay possible.
fn parse_shared_host_arg(s: &str) -> Result<(String, String), String> {
    let (name, url) = s
        .split_once('=')
        .ok_or_else(|| "expected `name=url`".to_string())?;
    if name.is_empty() {
        return Err("name must be non-empty".into());
    }
    if url.is_empty() {
        return Err("url must be non-empty".into());
    }
    Ok((name.to_string(), url.to_string()))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,whisper_agent=info,tower_http=info")),
        )
        .init();

    // Pin rustls's process-level CryptoProvider before any TLS code
    // runs. Both `ring` and `aws-lc-rs` arrive transitively, so rustls
    // refuses to auto-pick. `.ok()` swallows the "already installed"
    // error if some library beat us to it.
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => run_serve(args).await,
        Command::Config(cmd) => run_config(cmd).await,
    }
}

async fn run_config(cmd: ConfigCmd) -> Result<()> {
    match cmd.action {
        ConfigAction::Env(args) => run_config_env(args).await,
    }
}

async fn run_config_env(args: ConfigEnvArgs) -> Result<()> {
    let path = resolve_config_path(args.config)?
        .ok_or_else(|| anyhow!("no config file found (searched --config, XDG, ~/.config, cwd)"))?;
    let cfg = Config::load(&path).await?;
    for (key, value) in &cfg.secrets {
        println!("export {}={}", key, shell_single_quote(value));
    }
    Ok(())
}

/// Wrap `s` in single quotes for safe shell eval, escaping any embedded
/// single quotes via the standard `'\''` idiom.
fn shell_single_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

async fn run_serve(args: ServeArgs) -> Result<()> {
    let pods_root = if args.pods_root.as_os_str().is_empty() {
        None
    } else {
        Some(args.pods_root)
    };

    // Resolve the backend catalog and shared-host map. Either source can come
    // from a TOML config; CLI --shared-mcp-host flags layer on top (CLI wins
    // on name conflict).
    let resolved_config = resolve_config_path(args.config.clone())?;
    let (
        backends,
        default_backend,
        default_model,
        mut shared_host_map,
        toml_provider_entries,
        auth_clients,
    ) = match &resolved_config {
        Some(path) => {
            info!(config = %path.display(), "loading backend catalog");
            let cfg = Config::load(path).await?;
            let default_model = cfg
                .backends
                .get(&cfg.default_backend)
                .ok_or_else(|| anyhow!("config: default_backend missing entry"))?
                .default_model()
                .map(|s| s.to_string())
                .unwrap_or_default();
            let mut map = HashMap::new();
            for (name, bcfg) in &cfg.backends {
                let provider = bcfg
                    .build()
                    .with_context(|| format!("build backend `{name}`"))?;
                map.insert(
                    name.clone(),
                    BackendEntry {
                        provider,
                        kind: bcfg.kind().into(),
                        default_model: bcfg.default_model().map(|s| s.to_string()),
                    },
                );
            }
            (
                map,
                cfg.default_backend,
                default_model,
                cfg.shared_mcp_hosts.into_iter().collect::<HashMap<_, _>>(),
                cfg.host_env_providers,
                cfg.auth.clients,
            )
        }
        None => {
            let key = args.anthropic_api_key.clone().ok_or_else(|| {
                anyhow!("no --config provided and ANTHROPIC_API_KEY / --anthropic-api-key is unset")
            })?;
            let mut map = HashMap::new();
            map.insert(
                DEFAULT_BACKEND_NAME.into(),
                BackendEntry {
                    provider: std::sync::Arc::new(AnthropicClient::new(key)),
                    kind: "anthropic".into(),
                    default_model: Some(args.model.clone()),
                },
            );
            (
                map,
                DEFAULT_BACKEND_NAME.into(),
                args.model.clone(),
                HashMap::new(),
                Vec::new(),
                Vec::new(),
            )
        }
    };
    for (name, url) in args.shared_mcp_hosts {
        shared_host_map.insert(name, url);
    }
    let mut shared_mcp_hosts: Vec<SharedHostConfig> = shared_host_map
        .into_iter()
        .map(|(name, url)| SharedHostConfig { name, url })
        .collect();
    // Stable order so log output and the default_task_config list don't
    // depend on HashMap iteration order.
    shared_mcp_hosts.sort_by(|a, b| a.name.cmp(&b.name));
    let default_shared_host_names: Vec<String> =
        shared_mcp_hosts.iter().map(|h| h.name.clone()).collect();

    let cli_token_files: HashMap<String, PathBuf> =
        args.host_env_provider_tokens.iter().cloned().collect();
    let mut provider_entries: Vec<HostEnvProviderEntry> = toml_provider_entries
        .into_iter()
        .map(|p| HostEnvProviderEntry {
            name: p.name,
            url: p.url,
            token_file: p.token_file,
        })
        .collect();
    for (name, url) in &args.host_env_providers {
        provider_entries.push(HostEnvProviderEntry {
            name: name.clone(),
            url: url.clone(),
            token_file: cli_token_files.get(name).cloned(),
        });
    }
    // Warn about CLI token flags that don't match any registered provider,
    // so typos don't silently disable auth.
    for name in cli_token_files.keys() {
        let known_from_cli = args.host_env_providers.iter().any(|(n, _)| n == name);
        let known_from_toml = provider_entries.iter().any(|e| &e.name == name);
        if !known_from_cli && !known_from_toml {
            anyhow::bail!(
                "--host-env-provider-token names provider `{name}` but no provider by that name is registered"
            );
        }
    }
    let host_env_registry =
        HostEnvRegistry::new(provider_entries).context("build host_env provider catalog")?;

    let default_host_env = build_default_host_env(
        &args.default_host_env_provider,
        &args.default_host_env_workspace,
    );

    // clap's `requires` attr already enforces the cert+key pair, so by
    // the time we get here the two are either both Some or both None.
    let tls = match (args.tls_cert, args.tls_key) {
        (Some(cert), Some(key)) => Some(server::TlsConfig {
            cert_path: cert,
            key_path: key,
        }),
        _ => None,
    };

    // If the listen address is reachable beyond loopback and tokens are
    // configured, refuse to start without TLS. Auth secrets across the
    // wire in cleartext is a footgun we'd rather hard-fail on than warn
    // about. Wildcard binds (0.0.0.0, ::) are conservatively treated as
    // non-loopback because they accept both.
    if !auth_clients.is_empty() && !args.listen.ip().is_loopback() && tls.is_none() {
        return Err(anyhow!(
            "[[auth.clients]] is configured and --listen {} is not loopback, but no --tls-cert/--tls-key was provided; \
             refusing to send tokens over plain HTTP",
            args.listen
        ));
    }

    let server_config = ServerConfig {
        backends,
        default_backend,
        default_task_config: ThreadConfig {
            model: default_model,
            max_tokens: args.max_tokens,
            max_turns: args.max_turns,
            compaction: Default::default(),
        },
        default_system_prompt: args.system_prompt,
        default_host_env,
        default_shared_host_names,
        audit_log_path: args.audit_log,
        host_id: "default".into(),
        pods_root,
        host_env_registry,
        shared_mcp_hosts,
        tls,
        auth_clients,
    };
    server::serve(args.listen, server_config).await
}

/// Resolve the synthesized default pod's `(provider, spec)` pair from
/// CLI args. Returns `None` when no provider is named — the default
/// pod then has no host env configured, and threads inside it run
/// with no host-env MCP connection (only shared MCP hosts apply).
fn build_default_host_env(
    provider: &Option<String>,
    workspace: &Option<PathBuf>,
) -> Option<(String, HostEnvSpec)> {
    let provider = provider.as_deref()?;
    let Some(ws) = workspace else {
        // Provider named without a workspace: use a no-paths landlock
        // shell — the user can edit pod.toml to flesh it out.
        return Some((
            provider.to_string(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: NetworkPolicy::Unrestricted,
            },
        ));
    };
    let mut paths = vec![PathAccess::read_write(ws.to_string_lossy().into_owned())];
    let home = std::env::var("HOME").unwrap_or_default();
    if !home.is_empty() {
        for subdir in [".cargo", ".rustup"] {
            let p = format!("{home}/{subdir}");
            if std::path::Path::new(&p).exists() {
                paths.push(PathAccess::read_only(p));
            }
        }
    }
    info!(
        provider = %provider,
        workspace = %paths[0].path,
        extra_ro = paths.len() - 1,
        "default host env: landlock"
    );
    Some((
        provider.to_string(),
        HostEnvSpec::Landlock {
            allowed_paths: paths,
            network: NetworkPolicy::Unrestricted,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::shell_single_quote;

    #[test]
    fn quotes_plain_value() {
        assert_eq!(shell_single_quote("abc123"), "'abc123'");
    }

    #[test]
    fn quotes_special_characters_without_expansion() {
        assert_eq!(
            shell_single_quote("a $b `c` \"d\" \\e"),
            "'a $b `c` \"d\" \\e'"
        );
    }

    #[test]
    fn escapes_embedded_single_quote() {
        // The idiom: close, escaped single-quote, reopen.
        assert_eq!(shell_single_quote("it's"), "'it'\\''s'");
    }
}
