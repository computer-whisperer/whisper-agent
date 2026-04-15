use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use whisper_agent::agent_loop::{LoopConfig, run};
use whisper_agent::anthropic::AnthropicClient;
use whisper_agent::audit::AuditLog;
use whisper_agent::mcp::McpSession;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a software engineering agent. You have access to a set of tools that operate on a workspace directory on the user's machine. Use the tools as needed to complete the user's request. Be concise. When you have finished, summarize what you did.";

#[derive(Parser, Debug)]
#[command(version, about = "Headless agent loop driving Anthropic Messages + MCP host tools.")]
struct Args {
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
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,whisper_agent=info")))
        .init();

    let args = Args::parse();
    info!(model = %args.model, host = %args.mcp_host_url, "starting whisper-agent");

    let session_id = uuid_v4_pseudo();
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

    let outcome = run(cfg, anthropic, mcp, audit, args.prompt).await?;

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
        let serialized = serde_json::to_string_pretty(outcome.conversation.messages())?;
        tokio::fs::write(&path, serialized).await?;
        println!("conversation dumped to {}", path.display());
    }

    Ok(())
}

/// Tiny v4-shaped pseudo-random session ID. Real UUIDs come when we have a real reason.
fn uuid_v4_pseudo() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!(
        "ses-{:016x}-{:08x}",
        nanos as u64,
        std::process::id()
    )
}
