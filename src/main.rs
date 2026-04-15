use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;

use whisper_agent::agent_loop::{LoopConfig, run};
use whisper_agent::anthropic::AnthropicClient;
use whisper_agent::audit::AuditLog;
use whisper_agent::conversation::Conversation;
use whisper_agent::mcp::McpSession;
use whisper_agent::server::{self, ServerConfig};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a software engineering agent. You have access to a set of tools that operate on a workspace directory on the user's machine. Use the tools as needed to complete the user's request. Be concise. When you have finished, summarize what you did.";

#[derive(Parser, Debug)]
#[command(version, about = "whisper-agent: headless agent loop with embedded webui server.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Start the HTTP server: hosts the webui assets and runs one chat session per
    /// WebSocket connection.
    Serve(ServeArgs),
    /// One-shot CLI: run the agent loop against a single prompt and exit. Used for
    /// development and the CLI-only smoke test in docs/design_mvp.md (step 4).
    Run(RunArgs),
}

#[derive(Parser, Debug)]
struct ServeArgs {
    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

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

    /// System prompt to send to the model.
    #[arg(long, default_value = DEFAULT_SYSTEM_PROMPT)]
    system_prompt: String,

    /// Maximum number of model turns per user message.
    #[arg(long, default_value_t = 30)]
    max_turns: u32,

    /// max_tokens parameter passed to Anthropic.
    #[arg(long, default_value_t = 4096)]
    max_tokens: u32,
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
    #[arg(long, default_value_t = 4096)]
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
    let config = ServerConfig {
        anthropic_api_key: Arc::new(args.anthropic_api_key),
        mcp_host_url: Arc::new(args.mcp_host_url),
        model: Arc::new(args.model),
        system_prompt: Arc::new(args.system_prompt),
        max_tokens: args.max_tokens,
        max_turns: args.max_turns,
        audit_log_path: args.audit_log,
    };
    server::serve(args.listen, config).await
}

async fn run_one_shot(args: RunArgs) -> Result<()> {
    info!(model = %args.model, host = %args.mcp_host_url, "starting whisper-agent (one-shot)");

    let session_id = pseudo_session_id();
    let host_id = "default";

    let anthropic = AnthropicClient::new(args.anthropic_api_key);
    let mcp = McpSession::connect(&args.mcp_host_url)
        .await
        .with_context(|| format!("connect to MCP host {}", args.mcp_host_url))?;
    let audit = AuditLog::open(args.audit_log.clone())
        .await
        .with_context(|| format!("open audit log {}", args.audit_log.display()))?;
    info!(audit_log = %audit.path().display(), session_id = %session_id, "audit log open");

    let cfg = LoopConfig {
        model: args.model,
        max_tokens: args.max_tokens,
        system_prompt: args.system_prompt,
        session_id,
        host_id: host_id.to_string(),
        max_turns: args.max_turns,
    };

    let mut conversation = Conversation::new();
    let outcome = run(&cfg, &anthropic, &mcp, &audit, &mut conversation, args.prompt, None).await?;

    println!();
    println!("=== loop finished ===");
    println!("turns: {}", outcome.turns);
    println!("stop_reason: {:?}", outcome.final_stop_reason);
    println!(
        "tokens: input={} output={} cache_read={} cache_create={}",
        outcome.usage_total.input_tokens,
        outcome.usage_total.output_tokens,
        outcome.usage_total.cache_read_input_tokens,
        outcome.usage_total.cache_creation_input_tokens
    );

    if let Some(path) = args.dump_conversation {
        let serialized = serde_json::to_string_pretty(conversation.messages())?;
        tokio::fs::write(&path, serialized).await?;
        println!("conversation dumped to {}", path.display());
    }

    Ok(())
}

fn pseudo_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!("ses-{:016x}-{:08x}", nanos as u64, std::process::id())
}
