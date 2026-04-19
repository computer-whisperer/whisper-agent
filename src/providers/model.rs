//! Provider-agnostic model interface.
//!
//! [`ModelProvider`] is the object-safe trait every backend (Anthropic, OpenAI-compatible
//! Chat Completions, …) implements. The scheduler holds `Arc<dyn ModelProvider>` and
//! never touches a provider-specific type. Each impl converts between its own wire
//! shape and the normalized types here at the boundary.
//!
//! Normalized shapes intentionally re-use our canonical [`ContentBlock`] / [`Message`]
//! types, which are Anthropic-content-block-shaped. Translating OpenAI-style messages
//! into / out of this shape is the OpenAI backend's job, not the scheduler's.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use futures::stream::{Stream, StreamExt};
use serde_json::Value;
use thiserror::Error;
use whisper_agent_protocol::{ContentBlock, Message, Usage};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

pub struct ModelRequest<'a> {
    pub model: &'a str,
    pub max_tokens: u32,
    pub system_prompt: &'a str,
    pub tools: &'a [ToolSpec],
    pub messages: &'a [Message],
    /// Positions at which the backend should mark prompt-cache checkpoints.
    /// Providers that don't support caching ignore this list. Anthropic caps the
    /// list at 4 markers per request — the scheduler is responsible for staying
    /// under that limit.
    pub cache_breakpoints: &'a [CacheBreakpoint],
}

/// Logical positions at which a cache checkpoint can be attached. Translated into
/// provider-specific markers by each [`ModelProvider`] implementation — or ignored
/// by providers without cache support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBreakpoint {
    /// Cache the system prompt (everything up to and including `system`).
    AfterSystem,
    /// Cache the tool declarations (everything up to and including `tools`).
    AfterTools,
    /// Cache through `messages[index]` (inclusive).
    AfterMessage(usize),
}

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug)]
pub struct ModelResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: Option<String>,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("api error {status}: {body}")]
    Api { status: u16, body: String },
    /// Server signalled a short-term capacity / quota exhaustion (typically HTTP
    /// 429). Carries the server-suggested wait before retrying along with the
    /// raw body so the caller can show a useful message. The dispatch layer may
    /// retry once transparently when `retry_after` is small; longer waits
    /// surface as failures with the wait in the message.
    #[error("rate limited: retry after {retry_after:?}: {body}")]
    RateLimited { retry_after: Duration, body: String },
}

/// Incremental event emitted from a streaming model call. Intended for live UI
/// updates; the canonical, assembled turn content lands in the terminal
/// [`ModelEvent::Completed`] event.
///
/// Streams are `Stream<Item = Result<ModelEvent, ModelError>>`. A terminal
/// error ends the stream without a `Completed`. On success, `Completed` is
/// always the last event — the scheduler persists from its `content`, not
/// from reassembling deltas.
///
/// Tool calls arrive as a single `ToolCall` event with fully-assembled args,
/// not a begin/delta/end trio — live partial-JSON rendering isn't worth the
/// webui complexity. Text and thinking *are* emitted as deltas because
/// character-by-character streaming is the user-visible win.
#[derive(Debug, Clone)]
pub enum ModelEvent {
    /// Append to the current assistant text block (open a new one if the last
    /// emitted event wasn't also a `TextDelta`).
    TextDelta { text: String },
    /// Append to the current assistant thinking block.
    ThinkingDelta { text: String },
    /// A fully-formed tool call. Emitted once per call, after the model has
    /// streamed its full argument JSON and the adapter has parsed it.
    ToolCall {
        id: String,
        name: String,
        input: Value,
    },
    /// Terminal event. `content` is the assistant-turn content block list the
    /// scheduler should persist. Deltas emitted before this are strictly for
    /// broadcast; the canonical state comes from here.
    Completed {
        content: Vec<ContentBlock>,
        stop_reason: Option<String>,
        usage: Usage,
    },
}

/// Provider-agnostic model backend. Object-safe via explicit pinned-boxed futures
/// (native `async fn in trait` isn't dyn-safe with `Send` bound without extra crates).
pub trait ModelProvider: Send + Sync {
    fn create_message<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
    ) -> BoxFuture<'a, Result<ModelResponse, ModelError>>;

    /// Streaming variant. Returns a stream of [`ModelEvent`]s ending in a
    /// single `Completed` event. The default implementation buffers on top of
    /// `create_message` and synthesizes one delta per text/thinking block
    /// plus one `ToolCall` per tool_use, so consumers only need to handle
    /// the streaming shape. Adapters override to emit real SSE deltas as
    /// bytes arrive.
    fn create_message_streaming<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        Box::pin(async_stream::try_stream! {
            let resp = self.create_message(req).await?;
            // Synthesize per-block events so a non-streaming backend
            // flows through the same consumer code path as a streaming
            // one. Callers that only care about Completed still get the
            // canonical assembled content there.
            for block in &resp.content {
                match block {
                    ContentBlock::Text { text } if !text.is_empty() => {
                        yield ModelEvent::TextDelta { text: text.clone() };
                    }
                    ContentBlock::Thinking { thinking, .. } if !thinking.is_empty() => {
                        yield ModelEvent::ThinkingDelta { text: thinking.clone() };
                    }
                    ContentBlock::ToolUse { id, name, input, .. } => {
                        yield ModelEvent::ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        };
                    }
                    _ => {}
                }
            }
            yield ModelEvent::Completed {
                content: resp.content,
                stop_reason: resp.stop_reason,
                usage: resp.usage,
            };
        })
    }

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<ModelInfo>, ModelError>>;
}

/// Drain a `create_message_streaming` stream into a single [`ModelResponse`]
/// — use this when a caller needs batch semantics on top of a streaming
/// provider. Returns the first error if the stream terminates early.
pub async fn buffer_stream<'a>(
    mut stream: BoxStream<'a, Result<ModelEvent, ModelError>>,
) -> Result<ModelResponse, ModelError> {
    while let Some(ev) = stream.next().await {
        match ev? {
            ModelEvent::Completed {
                content,
                stop_reason,
                usage,
            } => {
                return Ok(ModelResponse {
                    content,
                    stop_reason,
                    usage,
                });
            }
            // Deltas are discarded — the canonical content arrives in
            // `Completed`. A caller that wants them should consume the
            // stream directly instead of using this helper.
            _ => continue,
        }
    }
    Err(ModelError::Transport(
        "model stream ended without Completed event".into(),
    ))
}

/// Default cache policy — mirrors Claude Code's observed wire behavior: one
/// breakpoint on the system prompt (stable) plus one rolling breakpoint on the
/// most recent user-side message (where fresh input lands each turn). Tools sit
/// between the two and ride the prefix implicitly — the server extends the cached
/// prefix forward on each turn as long as tool declarations are byte-stable.
///
/// Keeping the total at two markers is intentional: extra breakpoints each
/// create separate cache entries with their own write multipliers (1.25× for 5m,
/// 2× for 1h), and the redundancy only pays off if the longer entry TTLs out
/// mid-session — unlikely at 1h. Two markers is also well under Anthropic's 4-cap.
///
/// "User-side" is `Role::User` or `Role::ToolResult` — anything non-assistant.
/// Tool results are where fresh input lands on tool-loop turns; filtering them
/// out stalls the rolling checkpoint on the last human-typed message.
pub fn default_cache_policy(messages: &[Message]) -> Vec<CacheBreakpoint> {
    let mut out = vec![CacheBreakpoint::AfterSystem];
    if let Some(last) = messages
        .iter()
        .enumerate()
        .rfind(|(_, m)| !matches!(m.role, whisper_agent_protocol::Role::Assistant))
        .map(|(i, _)| i)
    {
        out.push(CacheBreakpoint::AfterMessage(last));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::{ContentBlock, Message, ToolResultContent};

    fn tr(id: &str, text: &str) -> Message {
        Message::tool_result_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: id.into(),
            content: ToolResultContent::Text(text.into()),
            is_error: false,
        }])
    }

    #[test]
    fn rolling_breakpoint_tracks_last_tool_result() {
        // Agentic loop: one user prompt followed by several assistant/tool_result
        // cycles. The rolling breakpoint must land on the last user-side message
        // (the trailing tool_result at index 6), not stall on the initial user
        // text at index 0.
        let msgs = vec![
            Message::user_text("start"),
            Message::assistant_blocks(vec![ContentBlock::Text { text: "a".into() }]),
            tr("t1", "r1"),
            Message::assistant_blocks(vec![ContentBlock::Text { text: "b".into() }]),
            tr("t2", "r2"),
            Message::assistant_blocks(vec![ContentBlock::Text { text: "c".into() }]),
            tr("t3", "r3"),
        ];
        let bps = default_cache_policy(&msgs);
        assert_eq!(
            bps,
            vec![
                CacheBreakpoint::AfterSystem,
                CacheBreakpoint::AfterMessage(6)
            ]
        );
    }

    #[test]
    fn empty_conversation_yields_system_marker_only() {
        // No messages (ScheduleWakeup from a fresh thread, seeded systems, etc.) —
        // nothing to anchor a rolling breakpoint on, so only the system marker.
        let bps = default_cache_policy(&[]);
        assert_eq!(bps, vec![CacheBreakpoint::AfterSystem]);
    }

    #[test]
    fn trailing_assistant_message_is_not_a_breakpoint() {
        // Trailing assistant message (shouldn't normally happen at dispatch
        // time, but the policy must handle it gracefully): walk back past it
        // to the nearest user-side anchor.
        let msgs = vec![
            Message::user_text("hi"),
            Message::assistant_blocks(vec![ContentBlock::Text { text: "hey".into() }]),
        ];
        let bps = default_cache_policy(&msgs);
        assert_eq!(
            bps,
            vec![
                CacheBreakpoint::AfterSystem,
                CacheBreakpoint::AfterMessage(0)
            ]
        );
    }
}
