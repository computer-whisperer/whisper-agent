//! Pure turn-running logic.
//!
//! Appends a user message to the conversation, then drives the model → MCP tool →
//! `tool_result` cycle until the assistant emits a non-tool stop reason or `max_turns`
//! is hit. Emits [`TurnEvent`]s via an optional sink so callers can stream them to
//! UIs or logs without this module needing to know about the wire protocol.
//!
//! Used only by the CLI one-shot path now that the scheduler is authoritative.

use anyhow::{Context, Result};
use futures::StreamExt;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{info, warn};
use whisper_agent_protocol::{ContentBlock, Conversation, Message, ToolResultContent, Usage};

use crate::audit::{AuditLog, ToolCallEntry, ToolCallOutcome};
use crate::mcp::{McpSession, ToolDescriptor as McpTool, ToolEvent};
use crate::model::{ModelProvider, ModelRequest, ToolSpec, default_cache_policy};

/// Per-turn events emitted during the loop.
#[derive(Debug, Clone)]
pub enum TurnEvent {
    AssistantBegin { turn: u32 },
    AssistantText { text: String },
    ToolCallBegin {
        tool_use_id: String,
        name: String,
        args_preview: String,
    },
    ToolCallEnd {
        tool_use_id: String,
        result_preview: String,
        is_error: bool,
    },
    AssistantEnd {
        stop_reason: Option<String>,
        usage: Usage,
    },
    LoopComplete,
}

pub struct TurnConfig<'a> {
    pub model: &'a str,
    pub max_tokens: u32,
    pub system_prompt: &'a str,
    pub task_id: &'a str,
    pub host_id: &'a str,
    pub max_turns: u32,
}

pub struct TurnOutcome {
    pub turns: u32,
    pub usage_total: Usage,
    pub final_stop_reason: Option<String>,
}

/// Append `new_user_message` to the conversation and iterate the model/tool loop.
pub async fn run(
    cfg: &TurnConfig<'_>,
    model: &dyn ModelProvider,
    mcp: &McpSession,
    audit: &AuditLog,
    conversation: &mut Conversation,
    new_user_message: String,
    events: Option<UnboundedSender<TurnEvent>>,
) -> Result<TurnOutcome> {
    let emit = |e: TurnEvent| {
        if let Some(s) = &events {
            let _ = s.send(e);
        }
    };

    conversation.push(Message::user_text(new_user_message));

    let mcp_tools = mcp.list_tools().await.context("mcp list_tools failed")?;
    let tools: Vec<ToolSpec> = mcp_tools.iter().map(mcp_to_tool_spec).collect();
    info!(count = tools.len(), "advertising tools to model");

    let mut turns = 0u32;
    let mut total = Usage::default();
    let mut last_stop_reason = None;

    loop {
        if turns >= cfg.max_turns {
            warn!(max_turns = cfg.max_turns, "max_turns reached, halting");
            break;
        }

        turns += 1;
        emit(TurnEvent::AssistantBegin { turn: turns });

        let breakpoints = default_cache_policy(conversation.messages(), 2);
        let req = ModelRequest {
            model: cfg.model,
            max_tokens: cfg.max_tokens,
            system_prompt: cfg.system_prompt,
            tools: &tools,
            messages: conversation.messages(),
            cache_breakpoints: &breakpoints,
        };
        let response = model
            .create_message(&req)
            .await
            .map_err(|e| anyhow::anyhow!("model create_message failed: {e}"))?;
        total.add(&response.usage);
        last_stop_reason = response.stop_reason.clone();
        info!(
            turn = turns,
            stop_reason = ?response.stop_reason,
            input = response.usage.input_tokens,
            output = response.usage.output_tokens,
            cache_read = response.usage.cache_read_input_tokens,
            cache_create = response.usage.cache_creation_input_tokens,
            "model response"
        );

        let assistant_blocks = response.content;

        for block in &assistant_blocks {
            if let ContentBlock::Text { text } = block {
                emit(TurnEvent::AssistantText { text: text.clone() });
            }
        }

        let tool_uses: Vec<(String, String, serde_json::Value)> = assistant_blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
            .collect();

        emit(TurnEvent::AssistantEnd {
            stop_reason: response.stop_reason.clone(),
            usage: response.usage,
        });

        conversation.push(Message::assistant_blocks(assistant_blocks));

        if tool_uses.is_empty() {
            break;
        }

        let mut tool_result_blocks = Vec::with_capacity(tool_uses.len());
        for (use_id, name, input) in tool_uses {
            info!(tool = %name, "invoking");
            emit(TurnEvent::ToolCallBegin {
                tool_use_id: use_id.clone(),
                name: name.clone(),
                args_preview: truncate(serde_json::to_string(&input).unwrap_or_default(), 200),
            });

            let outcome = invoke_and_collect(mcp, &name, input.clone()).await;

            let err_msg;
            let (text, is_error, audit_outcome) = match &outcome {
                Ok(result) => (
                    join_text(&result.content),
                    result.is_error,
                    ToolCallOutcome::Ok { is_error: result.is_error },
                ),
                Err(e) => {
                    err_msg = format!("tool invocation failed: {e}");
                    (err_msg.clone(), true, ToolCallOutcome::Failed { message: &err_msg })
                }
            };

            audit
                .write(&ToolCallEntry {
                    timestamp: chrono::Utc::now(),
                    task_id: cfg.task_id,
                    host_id: cfg.host_id,
                    tool_name: &name,
                    args: input,
                    decision: "auto-approve",
                    who_decided: "stub",
                    outcome: audit_outcome,
                })
                .await
                .context("audit write failed")?;

            emit(TurnEvent::ToolCallEnd {
                tool_use_id: use_id.clone(),
                result_preview: truncate(text.clone(), 200),
                is_error,
            });

            tool_result_blocks.push(ContentBlock::ToolResult {
                tool_use_id: use_id,
                content: ToolResultContent::Text(text),
                is_error,
            });
        }

        conversation.push(Message::user_blocks(tool_result_blocks));
    }

    emit(TurnEvent::LoopComplete);

    Ok(TurnOutcome { turns, usage_total: total, final_stop_reason: last_stop_reason })
}

async fn invoke_and_collect(
    mcp: &McpSession,
    name: &str,
    input: serde_json::Value,
) -> Result<crate::mcp::CallToolResult, crate::mcp::McpError> {
    let mut stream = mcp.invoke(name, input).await?;
    let mut last = None;
    while let Some(event) = stream.next().await {
        match event {
            ToolEvent::Completed(r) => last = Some(r),
        }
    }
    last.ok_or_else(|| crate::mcp::McpError::Malformed("no Completed event in stream".into()))
}

fn mcp_to_tool_spec(t: &McpTool) -> ToolSpec {
    ToolSpec {
        name: t.name.clone(),
        description: t.description.clone(),
        input_schema: t.input_schema.clone(),
    }
}

fn join_text(blocks: &[crate::mcp::McpContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        match b {
            crate::mcp::McpContentBlock::Text { text } => out.push_str(text),
        }
    }
    out
}

fn truncate(mut s: String, max: usize) -> String {
    if s.len() > max {
        let mut cut = max;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push('…');
    }
    s
}
