use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;
use whisper_agent_protocol::sandbox::{HostEnvSpec, NetworkPolicy, PathAccess};
use whisper_agent_protocol::{ApprovalPolicy, ThreadConfig};

use whisper_agent::anthropic::AnthropicClient;
use whisper_agent::config::Config;
use whisper_agent::sandbox::{HostEnvProviderEntry, HostEnvRegistry};
use whisper_agent::scheduler::{BackendEntry, SharedHostConfig};
use whisper_agent::server::{self, ServerConfig};

const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a software engineering agent working in a workspace on the user's machine. Use the \
available tools to investigate and change code to complete the user's request.

Tool selection:
- Prefer dedicated tools over `bash` when one fits: `read_file`, `edit_file`, `write_file`, \
`glob`, `grep`, `list_dir`. Reserve `bash` for shell-only work — builds, tests, git, running \
scripts.
- For searches: use `grep` for file contents and `glob` for filenames. Do NOT run \
`grep`/`rg`/`find`/`ls`/`cat`/`head`/`tail`/`sed` via `bash`.
- For edits: `edit_file` is for targeted substring edits and is almost always what you want. \
Use `write_file` only to create a new file or fully rewrite one. Read the file before editing \
it so your `old_string` matches what is actually on disk.
- When tool calls are independent, issue them in parallel in a single response.

Output style:
- The user sees your text output, not your tool calls or reasoning. Before your first tool \
call, say in one short sentence what you're about to do. Give brief updates when you find \
something important, change direction, or hit a blocker — one sentence is almost always \
enough.
- Do not narrate deliberation or restate what tool results already show.
- End-of-turn summary: one or two sentences on what changed and what's next. Nothing else.
- Reference code as `path:line` so the user can jump to it.
- Default to writing no comments in code unless the WHY is non-obvious. No emojis unless the \
user asks.

Scope:
- Don't add features, refactors, or error handling beyond what the task requires.
- Prefer editing existing files over creating new ones. Never create Markdown (*.md) or \
README files unless the user explicitly asks.
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
    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

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

    /// Prompt the user before running non-read-only tool calls. Default is to
    /// auto-approve everything — the sandbox layer is the safety boundary.
    #[arg(long)]
    prompt_destructive: bool,

    /// Register a host-env provider from the CLI: `name=url`. Each
    /// catalog entry is a `whisper-agent-sandbox`-shaped daemon.
    /// Repeatable. Also configurable in TOML:
    /// `[[host_env_providers]] name = "...", url = "..."`. A server
    /// with zero providers is valid — threads in it just have no
    /// host-env MCP connection.
    #[arg(long = "host-env-provider", value_parser = parse_host_env_provider_arg)]
    host_env_providers: Vec<(String, String)>,

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
    let (backends, default_backend, default_model, mut shared_host_map, toml_provider_entries) =
        match &resolved_config {
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
                )
            }
            None => {
                let key = args.anthropic_api_key.clone().ok_or_else(|| {
                    anyhow!(
                        "no --config provided and ANTHROPIC_API_KEY / --anthropic-api-key is unset"
                    )
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

    let mut provider_entries: Vec<HostEnvProviderEntry> = toml_provider_entries
        .into_iter()
        .map(|p| HostEnvProviderEntry {
            name: p.name,
            url: p.url,
        })
        .collect();
    for (name, url) in &args.host_env_providers {
        provider_entries.push(HostEnvProviderEntry {
            name: name.clone(),
            url: url.clone(),
        });
    }
    let host_env_registry =
        HostEnvRegistry::new(provider_entries).context("build host_env provider catalog")?;

    let default_host_env = build_default_host_env(
        &args.default_host_env_provider,
        &args.default_host_env_workspace,
    );

    let server_config = ServerConfig {
        backends,
        default_backend,
        default_task_config: ThreadConfig {
            model: default_model,
            system_prompt: args.system_prompt,
            max_tokens: args.max_tokens,
            max_turns: args.max_turns,
            approval_policy: if args.prompt_destructive {
                ApprovalPolicy::PromptDestructive
            } else {
                ApprovalPolicy::PromptPodModify
            },
        },
        default_host_env,
        default_shared_host_names,
        audit_log_path: args.audit_log,
        host_id: "default".into(),
        pods_root,
        host_env_registry,
        shared_mcp_hosts,
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
