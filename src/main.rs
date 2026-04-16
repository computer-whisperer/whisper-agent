use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use tokio::sync::mpsc;
use tracing::info;
use tracing_subscriber::EnvFilter;
use whisper_agent_protocol::sandbox::{AccessMode, NetworkPolicy, PathAccess, SandboxSpec};
use whisper_agent_protocol::{
    ApprovalPolicy, ClientToServer, Role, ServerToClient, ThreadConfig, ThreadStateLabel,
};

use whisper_agent::anthropic::AnthropicClient;
use whisper_agent::audit::AuditLog;
use whisper_agent::config::Config;
use whisper_agent::sandbox::{BareMetal, DaemonClient};
use whisper_agent::scheduler::{
    self, BackendEntry, ConnId, Scheduler, SchedulerMsg, SharedHostConfig,
};
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
enum Command {
    /// Start the HTTP server: hosts the webui assets and drives the task scheduler.
    /// Clients subscribe to individual tasks via the multiplexed WebSocket protocol.
    Serve(ServeArgs),
    /// One-shot CLI: drive a single task through the scheduler against a
    /// hard-coded prompt and exit.
    Run(RunArgs),
}

#[derive(Parser, Debug)]
struct ServeArgs {
    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

    /// Path to a TOML config file describing the model-backend catalog. If omitted,
    /// the server falls back to a single-backend "anthropic" config built from
    /// `--anthropic-api-key` + `--model`.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Anthropic API key (used only when `--config` is not provided).
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    anthropic_api_key: Option<String>,

    /// Anthropic model ID (used only when `--config` is not provided).
    #[arg(long, default_value = "claude-sonnet-4-6")]
    model: String,

    /// MCP host URL (the streamable-HTTP /mcp endpoint).
    #[arg(long, default_value = "http://127.0.0.1:8800/mcp")]
    mcp_host_url: String,

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

    /// URL of the sandbox provisioning daemon. When set, tasks with non-None
    /// SandboxSpec will be provisioned via this daemon. When omitted, only
    /// SandboxSpec::None is supported (bare-metal execution).
    #[arg(long, env = "SANDBOX_DAEMON_URL")]
    sandbox_daemon_url: Option<String>,

    /// Convenience: default all tasks to a Landlock sandbox with this directory
    /// as the read-write workspace. Automatically adds ~/.cargo and ~/.rustup
    /// as read-only if they exist. Requires --sandbox-daemon-url.
    #[arg(long)]
    sandbox_workspace: Option<PathBuf>,

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

#[derive(Parser, Debug)]
struct RunArgs {
    /// Anthropic API key.
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    anthropic_api_key: String,

    /// Anthropic model ID.
    #[arg(long, default_value = "claude-sonnet-4-6")]
    model: String,

    /// MCP host URL (the streamable-HTTP /mcp endpoint).
    #[arg(long, default_value = "http://127.0.0.1:8800/mcp")]
    mcp_host_url: String,

    /// Path to the audit log file.
    #[arg(long, default_value = "audit.jsonl")]
    audit_log: PathBuf,

    /// Path to dump the final conversation as JSON.
    #[arg(long)]
    dump_conversation: Option<PathBuf>,

    /// System prompt to send to the model.
    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system_prompt: String,

    /// Maximum number of model turns before halting.
    #[arg(long, default_value_t = 30)]
    max_turns: u32,

    /// max_tokens parameter passed to Anthropic.
    #[arg(long, default_value_t = 16384)]
    max_tokens: u32,

    /// The user prompt that drives the loop.
    prompt: String,
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
        Command::Run(args) => run_one_shot(args).await,
    }
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
    let (backends, default_backend, default_model, mut shared_host_map) = match &args.config {
        Some(path) => {
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

    let server_config = ServerConfig {
        backends,
        default_backend,
        default_task_config: ThreadConfig {
            backend: String::new(), // empty → scheduler uses default_backend
            model: default_model,
            system_prompt: args.system_prompt,
            mcp_host_url: args.mcp_host_url,
            max_tokens: args.max_tokens,
            max_turns: args.max_turns,
            approval_policy: if args.prompt_destructive {
                ApprovalPolicy::PromptDestructive
            } else {
                ApprovalPolicy::AutoApproveAll
            },
            sandbox: build_default_sandbox(&args.sandbox_workspace),
            shared_mcp_hosts: default_shared_host_names,
        },
        audit_log_path: args.audit_log,
        host_id: "default".into(),
        pods_root,
        sandbox_provider: match args.sandbox_daemon_url {
            Some(url) => Arc::new(DaemonClient::new(url)),
            None => Arc::new(BareMetal),
        },
        shared_mcp_hosts,
    };
    server::serve(args.listen, server_config).await
}

fn build_default_sandbox(workspace: &Option<PathBuf>) -> SandboxSpec {
    let Some(ws) = workspace else {
        return SandboxSpec::None;
    };
    let ws_str = ws.to_string_lossy().into_owned();
    let mut paths = vec![PathAccess {
        path: ws_str,
        mode: AccessMode::ReadWrite,
    }];
    let home = std::env::var("HOME").unwrap_or_default();
    if !home.is_empty() {
        for subdir in [".cargo", ".rustup"] {
            let p = format!("{home}/{subdir}");
            if std::path::Path::new(&p).exists() {
                paths.push(PathAccess {
                    path: p,
                    mode: AccessMode::ReadOnly,
                });
            }
        }
    }
    info!(
        workspace = %paths[0].path,
        extra_ro = paths.len() - 1,
        "default sandbox: landlock"
    );
    SandboxSpec::Landlock {
        allowed_paths: paths,
        network: NetworkPolicy::Unrestricted,
    }
}

async fn run_one_shot(args: RunArgs) -> Result<()> {
    info!(model = %args.model, host = %args.mcp_host_url, "starting whisper-agent (one-shot)");

    let backend_name = DEFAULT_BACKEND_NAME.to_string();
    let mut backends = HashMap::new();
    backends.insert(
        backend_name.clone(),
        BackendEntry {
            provider: Arc::new(AnthropicClient::new(args.anthropic_api_key)),
            kind: "anthropic".into(),
            default_model: Some(args.model.clone()),
        },
    );

    let audit = AuditLog::open(args.audit_log.clone())
        .await
        .with_context(|| format!("open audit log {}", args.audit_log.display()))?;
    info!(audit_log = %audit.path().display(), "audit log open");

    let task_config = ThreadConfig {
        backend: backend_name.clone(),
        model: args.model.clone(),
        system_prompt: args.system_prompt,
        mcp_host_url: args.mcp_host_url,
        max_tokens: args.max_tokens,
        max_turns: args.max_turns,
        approval_policy: ApprovalPolicy::AutoApproveAll,
        sandbox: SandboxSpec::None,
        shared_mcp_hosts: Vec::new(),
    };

    let scheduler = Scheduler::new(
        task_config,
        "default".into(),
        backends,
        backend_name,
        audit,
        Arc::new(BareMetal),
        Vec::new(),
    )
    .await?;

    let (inbox_tx, inbox_rx) = mpsc::unbounded_channel::<SchedulerMsg>();
    let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<ServerToClient>();
    let scheduler_handle = tokio::spawn(scheduler::run(scheduler, inbox_rx));

    const CONN_ID: ConnId = 1;
    inbox_tx
        .send(SchedulerMsg::RegisterClient {
            conn_id: CONN_ID,
            outbound: outbound_tx,
        })
        .map_err(|_| anyhow!("scheduler inbox closed before registration"))?;
    inbox_tx
        .send(SchedulerMsg::ClientMessage {
            conn_id: CONN_ID,
            msg: ClientToServer::CreateThread {
                correlation_id: Some("cli-one-shot".into()),
                initial_message: args.prompt,
                config_override: None,
            },
        })
        .map_err(|_| anyhow!("scheduler inbox closed before create_task"))?;

    let mut thread_id: Option<String> = None;
    let mut last_stop_reason: Option<String> = None;
    let mut final_snapshot = None;
    let mut last_error: Option<String> = None;

    while let Some(event) = outbound_rx.recv().await {
        match event {
            ServerToClient::ThreadCreated { thread_id: tid, .. } => {
                thread_id = Some(tid.clone());
                inbox_tx
                    .send(SchedulerMsg::ClientMessage {
                        conn_id: CONN_ID,
                        msg: ClientToServer::SubscribeToThread { thread_id: tid },
                    })
                    .map_err(|_| anyhow!("scheduler inbox closed"))?;
            }
            ServerToClient::ThreadAssistantEnd { stop_reason, .. } => {
                last_stop_reason = stop_reason;
            }
            ServerToClient::Error { message, .. } => {
                last_error = Some(message);
            }
            ServerToClient::ThreadStateChanged { state, .. } => {
                if matches!(
                    state,
                    ThreadStateLabel::Completed | ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
                ) {
                    // Re-subscribe to get a fresh snapshot containing the
                    // final conversation + total_usage. SubscribeToThread is
                    // idempotent — the scheduler always replies with a fresh
                    // ThreadSnapshot regardless of existing membership.
                    if let Some(tid) = &thread_id {
                        inbox_tx
                            .send(SchedulerMsg::ClientMessage {
                                conn_id: CONN_ID,
                                msg: ClientToServer::SubscribeToThread {
                                    thread_id: tid.clone(),
                                },
                            })
                            .map_err(|_| anyhow!("scheduler inbox closed"))?;
                    }
                }
            }
            ServerToClient::ThreadSnapshot { snapshot, .. } => {
                if matches!(
                    snapshot.state,
                    ThreadStateLabel::Completed | ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
                ) {
                    final_snapshot = Some(snapshot);
                    break;
                }
                // Initial snapshot from our first subscribe — task still
                // working, so wait for the terminal-state re-subscribe.
            }
            _ => {}
        }
    }

    drop(inbox_tx);
    let _ = scheduler_handle.await;

    let snap = final_snapshot.ok_or_else(|| anyhow!("scheduler exited before final snapshot"))?;
    let turns = snap
        .conversation
        .messages()
        .iter()
        .filter(|m| matches!(m.role, Role::Assistant))
        .count();

    println!();
    println!("=== loop finished ===");
    println!("state: {:?}", snap.state);
    println!("turns: {turns}");
    println!("stop_reason: {last_stop_reason:?}");
    println!(
        "tokens: input={} output={} cache_read={} cache_create={}",
        snap.total_usage.input_tokens,
        snap.total_usage.output_tokens,
        snap.total_usage.cache_read_input_tokens,
        snap.total_usage.cache_creation_input_tokens
    );
    if let Some(detail) = &snap.failure {
        println!("failure: {detail}");
    } else if let Some(msg) = &last_error {
        println!("last_error: {msg}");
    }

    if let Some(path) = args.dump_conversation {
        let serialized = serde_json::to_string_pretty(snap.conversation.messages())?;
        tokio::fs::write(&path, serialized).await?;
        println!("conversation dumped to {}", path.display());
    }

    if matches!(snap.state, ThreadStateLabel::Failed) {
        return Err(anyhow!(
            "task failed: {}",
            snap.failure.unwrap_or_else(|| "unknown".into())
        ));
    }
    Ok(())
}
