//! OpenAI Chat Completions backend — the de-facto open source standard.
//!
//! Speaks the OpenAI `/v1/chat/completions` wire shape, which Ollama, LM Studio, vLLM,
//! llama.cpp server, LiteLLM, Groq, Fireworks, Together, OpenRouter and plain OpenAI
//! all accept. `base_url` is whatever endpoint you want the `/chat/completions` and
//! `/models` paths appended to (e.g. `https://api.openai.com/v1`,
//! `http://localhost:11434/v1`).
//!
//! Message conversion flattens our content-block form down to OpenAI's shape:
//!   - [`ContentBlock::Text`] blocks in an assistant/user message → merged `content` string.
//!   - [`ContentBlock::ToolUse`] → assistant `tool_calls[]`.
//!   - [`ContentBlock::ToolResult`] → a separate `role: "tool"` message with
//!     `tool_call_id`. (A single user message with mixed text + tool_results becomes
//!     multiple OpenAI messages.)
//!   - [`ContentBlock::Thinking`] → emitted on the assistant `OaMessage` as the
//!     non-standard `reasoning_content` field (concatenated when there are several).
//!     Servers that don't recognize the field ignore it; servers that do (vLLM,
//!     recent Ollama, llama.cpp w/ reasoning models) replay it as the model's
//!     prior chain-of-thought, which keeps multi-turn reasoning coherent.
//!
//! Inbound parsing recovers reasoning from two places:
//!   - Top-level `reasoning_content` on the response message (the canonical form).
//!   - Inline `<think>...</think>` spans inside the response `content` text
//!     (what llama.cpp typically passes through unmodified).
//!
//! Both paths produce [`ContentBlock::Thinking`].

use async_stream::try_stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use whisper_agent_protocol::{ContentBlock, Message, Role, ToolResultContent, Usage};

use crate::providers::model::{
    BoxFuture, BoxStream, ModelError, ModelEvent, ModelInfo, ModelProvider, ModelRequest,
    ModelResponse, ToolSpec,
};

pub struct OpenAiChatClient {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl OpenAiChatClient {
    /// `base_url` is the prefix under which `/chat/completions` and `/models` live.
    /// `api_key` is omitted for endpoints that don't require auth (Ollama, LM Studio).
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            http: reqwest::Client::new(),
            base_url,
            api_key,
        }
    }

    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(k) => builder.bearer_auth(k),
            None => builder,
        }
    }

    async fn do_create_message(&self, req: &ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        let mut messages: Vec<OaMessage> = Vec::with_capacity(req.messages.len() + 1);
        if !req.system_prompt.is_empty() {
            messages.push(OaMessage {
                role: "system".into(),
                content: Some(OaContent::Text(req.system_prompt.to_string())),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
        for m in req.messages {
            convert_message(m, &mut messages);
        }

        let tools: Vec<OaTool> = req.tools.iter().map(spec_to_oa_tool).collect();

        let body = OaRequest {
            model: req.model,
            max_tokens: req.max_tokens,
            messages,
            tools: if tools.is_empty() { None } else { Some(tools) },
            stream: None,
            stream_options: None,
        };

        let url = format!("{}/chat/completions", self.base_url);
        let builder = self.http.post(&url).json(&body);
        let resp = self
            .apply_auth(builder)
            .send()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ModelError::Api {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: OaResponse = resp
            .json()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ModelError::Transport("response had no choices[]".into()))?;

        let mut content: Vec<ContentBlock> = Vec::new();
        // Reasoning first if present — preserves the natural "thought → reply
        // → tool call" order in the conversation log.
        push_reasoning_content(choice.message.reasoning_content, &mut content);
        if let Some(text) = choice.message.content.and_then(content_as_text)
            && !text.is_empty()
        {
            push_text_with_inline_thinking(&text, &mut content);
        }
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let input: Value = if tc.function.arguments.is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&tc.function.arguments).map_err(|e| {
                        ModelError::Transport(format!("tool_call arguments not valid JSON: {e}"))
                    })?
                };
                content.push(ContentBlock::ToolUse {
                    id: tc.id,
                    name: tc.function.name,
                    input,
                    replay: None,
                });
            }
        }

        let usage = parsed
            .usage
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cache_read_input_tokens: u
                    .prompt_tokens_details
                    .and_then(|d| d.cached_tokens)
                    .unwrap_or(0),
                cache_creation_input_tokens: 0,
            })
            .unwrap_or_default();

        Ok(ModelResponse {
            content,
            stop_reason: choice.finish_reason.map(normalize_finish_reason),
            usage,
        })
    }

    /// Open the streaming `/chat/completions` connection and drive
    /// [`ChatStreamState`] across the SSE frames. Delta frames become
    /// `TextDelta` / `ThinkingDelta`; accumulated tool calls fire as
    /// `ToolCall` when `finish_reason` arrives; the `[DONE]` sentinel
    /// (or natural stream close) triggers the terminal `Completed` with
    /// the full reassembled content.
    fn do_create_message_streaming<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        Box::pin(try_stream! {
            let mut messages: Vec<OaMessage> = Vec::with_capacity(req.messages.len() + 1);
            if !req.system_prompt.is_empty() {
                messages.push(OaMessage {
                    role: "system".into(),
                    content: Some(OaContent::Text(req.system_prompt.to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            for m in req.messages {
                convert_message(m, &mut messages);
            }
            let tools: Vec<OaTool> = req.tools.iter().map(spec_to_oa_tool).collect();
            let body = OaRequest {
                model: req.model,
                max_tokens: req.max_tokens,
                messages,
                tools: if tools.is_empty() { None } else { Some(tools) },
                stream: Some(true),
                stream_options: Some(OaStreamOptions { include_usage: true }),
            };

            let url = format!("{}/chat/completions", self.base_url);
            let builder = self.http.post(&url).json(&body);
            let resp = self
                .apply_auth(builder)
                .send()
                .await
                .map_err(|e| ModelError::Transport(e.to_string()))?;
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                Err(ModelError::Api { status: status.as_u16(), body })?;
                return;
            }

            let mut bytes = resp.bytes_stream();
            let mut sse_buf: Vec<u8> = Vec::new();
            let mut state = ChatStreamState::default();
            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|e| ModelError::Transport(e.to_string()))?;
                sse_buf.extend_from_slice(&chunk);
                while let Some(event_payload) = take_sse_event(&mut sse_buf) {
                    let Some(raw) = parse_sse_data(&event_payload) else {
                        continue;
                    };
                    if raw == "[DONE]" {
                        yield state.finalize()?;
                        return;
                    }
                    let ev: OaStreamEvent = match serde_json::from_str(&raw) {
                        Ok(e) => e,
                        // Unknown/extra fields — skip rather than fail. Different
                        // compat servers inject different metadata.
                        Err(_) => continue,
                    };
                    for out in state.consume(ev)? {
                        yield out;
                    }
                }
            }
            // Stream closed without a `[DONE]` sentinel — some compat servers
            // (older Ollama builds) drop the marker. Synthesize the terminal
            // event from whatever we accumulated.
            yield state.finalize()?;
        })
    }

    async fn do_list_models(&self) -> Result<Vec<ModelInfo>, ModelError> {
        let url = format!("{}/models", self.base_url);
        let builder = self.http.get(&url);
        let resp = self
            .apply_auth(builder)
            .send()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ModelError::Api {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: OaListModelsResponse = resp
            .json()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        Ok(parsed
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                display_name: None,
            })
            .collect())
    }
}

impl ModelProvider for OpenAiChatClient {
    fn create_message<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
    ) -> BoxFuture<'a, Result<ModelResponse, ModelError>> {
        Box::pin(self.do_create_message(req))
    }

    fn create_message_streaming<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        self.do_create_message_streaming(req)
    }

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<ModelInfo>, ModelError>> {
        Box::pin(self.do_list_models())
    }
}

// ---------- Message conversion ----------

fn convert_message(m: &Message, out: &mut Vec<OaMessage>) {
    match m.role {
        // OpenAI chat splits user and tool-result into separate wire
        // messages with different roles (`user` vs `tool`). Both of
        // our in-memory roles currently go through the same builder —
        // `convert_user_message` inspects each content block and
        // emits `role: "user"` for text blocks and `role: "tool"`
        // for tool_result blocks, so a `Role::ToolResult` message
        // with only tool_result blocks produces only `tool` messages
        // on the wire, which is exactly what OpenAI wants.
        Role::User | Role::ToolResult => convert_user_message(&m.content, out),
        Role::Assistant => convert_assistant_message(&m.content, out),
        // Setup-prefix messages (system prompt, tool manifest) are
        // filtered out by `build_model_request` and lifted into the
        // system-role preamble / `tools` wire field. Defensive no-op
        // if one slips through.
        Role::System | Role::Tools => {}
    }
}

/// A user turn can be a plain text prompt OR a bundle of tool results. OpenAI
/// represents tool results as standalone `role: "tool"` messages, so we may emit
/// multiple output messages for one input.
fn convert_user_message(blocks: &[ContentBlock], out: &mut Vec<OaMessage>) {
    let mut text_accum = String::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                if !text_accum.is_empty() {
                    text_accum.push('\n');
                }
                text_accum.push_str(text);
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                // Flush any accumulated plain text first so ordering survives.
                if !text_accum.is_empty() {
                    out.push(OaMessage {
                        role: "user".into(),
                        content: Some(OaContent::Text(std::mem::take(&mut text_accum))),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                }
                let text = tool_result_as_text(content, *is_error);
                out.push(OaMessage {
                    role: "tool".into(),
                    content: Some(OaContent::Text(text)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                    reasoning_content: None,
                });
            }
            ContentBlock::ToolUse { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::ToolSchema { .. } => {
                // None make sense in a user message. Drop silently.
            }
        }
    }
    if !text_accum.is_empty() {
        out.push(OaMessage {
            role: "user".into(),
            content: Some(OaContent::Text(text_accum)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
}

fn convert_assistant_message(blocks: &[ContentBlock], out: &mut Vec<OaMessage>) {
    let mut text_accum = String::new();
    let mut reasoning_accum = String::new();
    let mut tool_calls: Vec<OaToolCall> = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                if !text_accum.is_empty() {
                    text_accum.push('\n');
                }
                text_accum.push_str(text);
            }
            ContentBlock::Thinking { thinking, .. } => {
                // Concatenate multi-block reasoning with a blank line between
                // so paragraph structure survives the round-trip.
                if !reasoning_accum.is_empty() {
                    reasoning_accum.push_str("\n\n");
                }
                reasoning_accum.push_str(thinking);
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(OaToolCall {
                    id: id.clone(),
                    kind: "function".into(),
                    function: OaFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                    },
                });
            }
            ContentBlock::ToolResult { .. } | ContentBlock::ToolSchema { .. } => {
                // Neither belongs on an assistant message.
            }
        }
    }
    // Wire contract: an assistant message must carry `content` OR `tool_calls`
    // (OpenAI and llama.cpp's compat endpoint both reject a message that
    // carries neither with a 400). When the model returned truly-empty output
    // — no text and no tools — we still have to emit a valid shape, so fall
    // back to an empty-string content. A tool-call-only turn correctly omits
    // content; a text turn carries the accumulated text.
    let has_tool_calls = !tool_calls.is_empty();
    let content = if text_accum.is_empty() {
        if has_tool_calls {
            None
        } else {
            Some(OaContent::Text(String::new()))
        }
    } else {
        Some(OaContent::Text(text_accum))
    };
    out.push(OaMessage {
        role: "assistant".into(),
        content,
        tool_calls: if has_tool_calls {
            Some(tool_calls)
        } else {
            None
        },
        tool_call_id: None,
        reasoning_content: if reasoning_accum.is_empty() {
            None
        } else {
            Some(reasoning_accum)
        },
    });
}

fn tool_result_as_text(content: &ToolResultContent, is_error: bool) -> String {
    let base = match content {
        ToolResultContent::Text(s) => s.clone(),
        ToolResultContent::Blocks(blocks) => {
            let mut out = String::new();
            for b in blocks {
                if let ContentBlock::Text { text } = b {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(text);
                }
            }
            out
        }
    };
    if is_error && !base.starts_with("ERROR:") {
        format!("ERROR: {base}")
    } else {
        base
    }
}

fn spec_to_oa_tool(t: &ToolSpec) -> OaTool {
    OaTool {
        kind: "function".into(),
        function: OaFunctionSpec {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.input_schema.clone(),
        },
    }
}

fn content_as_text(c: OaContent) -> Option<String> {
    match c {
        OaContent::Text(s) => Some(s),
        OaContent::Null => None,
    }
}

/// Push a Thinking block from the response's `reasoning_content` field. No-op
/// when the field is absent or whitespace-only — empty Thinking blocks would
/// just be noise in the conversation log.
fn push_reasoning_content(reasoning: Option<String>, out: &mut Vec<ContentBlock>) {
    if let Some(r) = reasoning
        && !r.trim().is_empty()
    {
        out.push(ContentBlock::Thinking {
            replay: None,
            thinking: r,
        });
    }
}

/// Walk `text` and push alternating Text / Thinking blocks for any inline
/// `<think>...</think>` spans. Used for endpoints (typically llama.cpp) that
/// pass through the model's raw chain-of-thought instead of moving it into a
/// separate `reasoning_content` field.
///
/// Edge cases:
/// - Unclosed `<think>` (model truncated mid-reasoning): everything after the
///   tag becomes a Thinking block. Better than losing the partial reasoning.
/// - Empty `<think></think>`: skipped.
/// - Whitespace-only Text spans: skipped (avoids cluttering the log with
///   blank lines around think tags).
/// - No tags at all: emits a single Text block with the input verbatim.
/// - Tags are matched literally as `<think>` / `</think>` — case-sensitive
///   lowercase, no attribute support. Models that emit other variants get
///   their reasoning preserved as plain text, which is acceptable.
fn push_text_with_inline_thinking(text: &str, out: &mut Vec<ContentBlock>) {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut rest = text;
    loop {
        match rest.find(OPEN) {
            None => {
                // No more think tags — flush remaining as text.
                push_text_if_nonblank(rest, out);
                return;
            }
            Some(open_idx) => {
                // Text before the tag.
                push_text_if_nonblank(&rest[..open_idx], out);
                let after_open = &rest[open_idx + OPEN.len()..];
                match after_open.find(CLOSE) {
                    Some(close_idx) => {
                        push_thinking_if_nonblank(&after_open[..close_idx], out);
                        rest = &after_open[close_idx + CLOSE.len()..];
                    }
                    None => {
                        // Unclosed: the rest is all reasoning.
                        push_thinking_if_nonblank(after_open, out);
                        return;
                    }
                }
            }
        }
    }
}

fn push_text_if_nonblank(s: &str, out: &mut Vec<ContentBlock>) {
    if !s.trim().is_empty() {
        out.push(ContentBlock::Text {
            text: s.to_string(),
        });
    }
}

fn push_thinking_if_nonblank(s: &str, out: &mut Vec<ContentBlock>) {
    if !s.trim().is_empty() {
        out.push(ContentBlock::Thinking {
            replay: None,
            thinking: s.to_string(),
        });
    }
}

/// Normalize OpenAI `finish_reason` values to the Anthropic-style labels we use
/// elsewhere so the UI can render a single vocabulary.
fn normalize_finish_reason(s: String) -> String {
    match s.as_str() {
        "stop" => "end_turn".into(),
        "length" => "max_tokens".into(),
        "tool_calls" | "function_call" => "tool_use".into(),
        _ => s,
    }
}

// ---------- Wire types (private to this module) ----------

#[derive(Serialize, Debug)]
struct OaRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<OaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OaTool>>,
    /// `true` switches the response to `text/event-stream`. Omitted on
    /// the batch path so servers that don't accept explicit `false`
    /// still work.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// When streaming, ask the server to emit a final `usage` chunk.
    /// Unsupported servers ignore the field; supported servers (OpenAI,
    /// vLLM, llama.cpp compat) send one extra `data:` frame with an
    /// empty `choices` and a populated `usage` object.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OaStreamOptions>,
}

#[derive(Serialize, Debug)]
struct OaStreamOptions {
    include_usage: bool,
}

#[derive(Serialize, Debug)]
struct OaMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OaContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OaToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    /// Replays the assistant's prior reasoning. Recognized by vLLM, recent
    /// Ollama, and llama.cpp w/ reasoning templates; ignored as an unknown
    /// field by stricter servers. Skipped entirely when empty so we don't
    /// poke servers that haven't seen the field.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

/// Emitted as either a plain string or explicit JSON null. OpenAI requires `content`
/// to be present on assistant messages even when only `tool_calls` are set, and
/// various servers disagree on whether `null` or the field being absent is accepted.
/// We emit `null` to match the OpenAI spec; `skip_serializing_if` on the field still
/// omits it entirely when we want to (not currently used).
#[derive(Debug)]
enum OaContent {
    Text(String),
    Null,
}

impl Serialize for OaContent {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            OaContent::Text(t) => s.serialize_str(t),
            OaContent::Null => s.serialize_none(),
        }
    }
}

impl<'de> Deserialize<'de> for OaContent {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(d)?;
        Ok(match opt {
            Some(s) => OaContent::Text(s),
            None => OaContent::Null,
        })
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct OaToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: OaFunctionCall,
}

#[derive(Serialize, Deserialize, Debug)]
struct OaFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug)]
struct OaTool {
    #[serde(rename = "type")]
    kind: String,
    function: OaFunctionSpec,
}

#[derive(Serialize, Debug)]
struct OaFunctionSpec {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Deserialize, Debug)]
struct OaResponse {
    choices: Vec<OaChoice>,
    #[serde(default)]
    usage: Option<OaUsage>,
}

#[derive(Deserialize, Debug)]
struct OaChoice {
    message: OaResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OaResponseMessage {
    #[serde(default)]
    content: Option<OaContent>,
    #[serde(default)]
    tool_calls: Option<Vec<OaToolCall>>,
    /// Non-standard but widely used field for reasoning models. vLLM, recent
    /// Ollama, and llama.cpp w/ reasoning model templates emit it; OpenAI's
    /// own reasoning models keep their reasoning encrypted. We capture it
    /// when present and skip it cleanly when not.
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OaUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<OaPromptTokensDetails>,
}

#[derive(Deserialize, Debug)]
struct OaPromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct OaListModelsResponse {
    data: Vec<OaModelInfo>,
}

#[derive(Deserialize, Debug)]
struct OaModelInfo {
    id: String,
}

// ---------- Streaming wire types + state machine ----------

/// One decoded SSE payload from `/chat/completions?stream=true`. Both the
/// text/tool deltas and the terminal `usage` frame land in this shape —
/// the `choices` array is empty on the usage frame.
#[derive(Deserialize, Debug)]
struct OaStreamEvent {
    #[serde(default)]
    choices: Vec<OaStreamChoice>,
    #[serde(default)]
    usage: Option<OaUsage>,
}

#[derive(Deserialize, Debug)]
struct OaStreamChoice {
    #[serde(default)]
    delta: Option<OaDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct OaDelta {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning fragment on compat servers that stream it as a dedicated
    /// field (vLLM, recent Ollama, llama.cpp w/ reasoning templates).
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OaStreamToolCallDelta>>,
}

/// A single tool-call delta. OpenAI streams tool calls in pieces keyed
/// by `index`: the first frame for a given index carries `id` + `name`
/// + opening `arguments`, subsequent frames append more `arguments`.
#[derive(Deserialize, Debug)]
struct OaStreamToolCallDelta {
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OaStreamFunctionCallDelta>,
}

#[derive(Deserialize, Debug, Default)]
struct OaStreamFunctionCallDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Running state for a single streaming `/chat/completions` response.
/// Accumulates text, reasoning, and per-index tool-call arguments; the
/// terminal `finalize` method reassembles the canonical content block
/// list for `ModelEvent::Completed`.
#[derive(Default)]
struct ChatStreamState {
    content_accum: String,
    reasoning_accum: String,
    /// Indexed by OpenAI's `tool_calls[].index` so fragments with
    /// interleaved indices still concatenate into the right call.
    tool_calls: Vec<Option<StreamToolCall>>,
    stop_reason: Option<String>,
    usage: Option<OaUsage>,
}

/// Per-index tool call being assembled during streaming.
struct StreamToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ChatStreamState {
    /// Fold one decoded SSE frame into running state and emit whatever
    /// user-visible `ModelEvent`s it produced. Content + reasoning
    /// become `TextDelta` / `ThinkingDelta` immediately; tool calls are
    /// held until a `finish_reason` arrives so the caller sees a single
    /// fully-assembled `ToolCall` event.
    fn consume(&mut self, ev: OaStreamEvent) -> Result<Vec<ModelEvent>, ModelError> {
        let mut out = Vec::new();
        if let Some(u) = ev.usage {
            self.usage = Some(u);
        }
        for choice in ev.choices {
            if let Some(delta) = choice.delta {
                if let Some(text) = delta.content
                    && !text.is_empty()
                {
                    self.content_accum.push_str(&text);
                    out.push(ModelEvent::TextDelta { text });
                }
                if let Some(text) = delta.reasoning_content
                    && !text.is_empty()
                {
                    self.reasoning_accum.push_str(&text);
                    out.push(ModelEvent::ThinkingDelta { text });
                }
                if let Some(tcs) = delta.tool_calls {
                    for tc in tcs {
                        self.apply_tool_call_delta(tc);
                    }
                }
            }
            if let Some(fr) = choice.finish_reason {
                // Flush all accumulated tool calls now that the model
                // has declared it's done emitting them. Consumers see a
                // single `ToolCall` event per call with the full parsed
                // argument object.
                for tc in self.tool_calls.iter().flatten() {
                    let input: Value = if tc.arguments.is_empty() {
                        Value::Object(Default::default())
                    } else {
                        serde_json::from_str(&tc.arguments).map_err(|e| {
                            ModelError::Transport(format!(
                                "tool_call arguments not valid JSON: {e}"
                            ))
                        })?
                    };
                    out.push(ModelEvent::ToolCall {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input,
                    });
                }
                self.stop_reason = Some(normalize_finish_reason(fr));
            }
        }
        Ok(out)
    }

    fn apply_tool_call_delta(&mut self, tc: OaStreamToolCallDelta) {
        // Without an explicit index, treat fragments as the current open
        // call (index 0). Compat servers that omit `index` ship a single
        // call at a time, so this is safe in practice.
        let idx = tc.index.unwrap_or(0) as usize;
        while self.tool_calls.len() <= idx {
            self.tool_calls.push(None);
        }
        let slot = &mut self.tool_calls[idx];
        let entry = slot.get_or_insert_with(|| StreamToolCall {
            id: String::new(),
            name: String::new(),
            arguments: String::new(),
        });
        if let Some(id) = tc.id
            && !id.is_empty()
        {
            entry.id = id;
        }
        if let Some(fn_delta) = tc.function {
            if let Some(name) = fn_delta.name
                && !name.is_empty()
            {
                entry.name = name;
            }
            if let Some(args) = fn_delta.arguments {
                entry.arguments.push_str(&args);
            }
        }
    }

    /// Reassemble accumulated deltas into the `ModelEvent::Completed`
    /// event. Produces the same canonical content the non-streaming
    /// code path would — reasoning block first, text with inline
    /// `<think>` tags split out, then any tool_use blocks in the order
    /// the server streamed them.
    fn finalize(&mut self) -> Result<ModelEvent, ModelError> {
        let mut content: Vec<ContentBlock> = Vec::new();
        if !self.reasoning_accum.trim().is_empty() {
            content.push(ContentBlock::Thinking {
                replay: None,
                thinking: std::mem::take(&mut self.reasoning_accum),
            });
        }
        if !self.content_accum.is_empty() {
            push_text_with_inline_thinking(&self.content_accum, &mut content);
            self.content_accum.clear();
        }
        for tc in self.tool_calls.drain(..).flatten() {
            let input: Value = if tc.arguments.is_empty() {
                Value::Object(Default::default())
            } else {
                serde_json::from_str(&tc.arguments).map_err(|e| {
                    ModelError::Transport(format!("tool_call arguments not valid JSON: {e}"))
                })?
            };
            content.push(ContentBlock::ToolUse {
                id: tc.id,
                name: tc.name,
                input,
                replay: None,
            });
        }
        let usage = self
            .usage
            .take()
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cache_read_input_tokens: u
                    .prompt_tokens_details
                    .and_then(|d| d.cached_tokens)
                    .unwrap_or(0),
                cache_creation_input_tokens: 0,
            })
            .unwrap_or_default();
        Ok(ModelEvent::Completed {
            content,
            stop_reason: self.stop_reason.take(),
            usage,
        })
    }
}

// ---------- SSE framing helpers ----------

/// Pull the next complete SSE event (bytes up to and including the
/// event-terminating `\n\n` or `\r\n\r\n`) out of `buf`. Returns `None`
/// while the buffer contains only a partial frame.
fn take_sse_event(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    let end = (0..buf.len().saturating_sub(1))
        .find(|&i| &buf[i..i + 2] == b"\n\n")
        .map(|i| i + 2)
        .or_else(|| {
            (0..buf.len().saturating_sub(3))
                .find(|&i| &buf[i..i + 4] == b"\r\n\r\n")
                .map(|i| i + 4)
        })?;
    let event = buf[..end].to_vec();
    buf.drain(..end);
    Some(event)
}

/// Extract the concatenated `data:` payload from a single SSE event.
/// Multiple `data:` lines join with `\n`; the single leading space
/// after the colon is stripped when present.
fn parse_sse_data(event: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(event).ok()?;
    let mut data = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data.is_empty() { None } else { Some(data) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_text_then_tool_uses_converts_to_one_message() {
        let blocks = vec![
            ContentBlock::Text {
                text: "let me check".into(),
            },
            ContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "foo.txt"}),
                replay: None,
            },
        ];
        let mut out = Vec::new();
        convert_assistant_message(&blocks, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "assistant");
        assert!(matches!(out[0].content, Some(OaContent::Text(ref s)) if s == "let me check"));
        let tcs = out[0].tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "toolu_1");
        assert_eq!(tcs[0].function.name, "read_file");
    }

    #[test]
    fn user_with_only_tool_results_emits_tool_role_messages() {
        let blocks = vec![
            ContentBlock::ToolResult {
                tool_use_id: "toolu_1".into(),
                content: ToolResultContent::Text("file contents".into()),
                is_error: false,
            },
            ContentBlock::ToolResult {
                tool_use_id: "toolu_2".into(),
                content: ToolResultContent::Text("more".into()),
                is_error: false,
            },
        ];
        let mut out = Vec::new();
        convert_user_message(&blocks, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "tool");
        assert_eq!(out[0].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(out[1].tool_call_id.as_deref(), Some("toolu_2"));
    }

    #[test]
    fn empty_assistant_blocks_emit_empty_string_content() {
        // Regression: an assistant message with no blocks used to serialize as
        // `{"role":"assistant"}` (both content and tool_calls omitted), which
        // OpenAI-compat servers reject with 400. We now emit `content: ""`.
        let mut out = Vec::new();
        convert_assistant_message(&[], &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "assistant");
        assert!(matches!(out[0].content, Some(OaContent::Text(ref s)) if s.is_empty()));
        assert!(out[0].tool_calls.is_none());
        // And the serialized JSON actually carries the content field.
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["content"], serde_json::Value::String(String::new()));
    }

    #[test]
    fn tool_call_only_assistant_omits_content_field() {
        // Tool-call-only turns should still omit `content` entirely — OpenAI
        // accepts that shape and some strict servers prefer it to a null/empty.
        let blocks = vec![ContentBlock::ToolUse {
            id: "toolu_1".into(),
            name: "bash".into(),
            input: serde_json::json!({"command": "ls"}),
            replay: None,
        }];
        let mut out = Vec::new();
        convert_assistant_message(&blocks, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out[0].content.is_none());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert!(
            v.get("content").is_none(),
            "tool_call-only turn must not carry content field"
        );
    }

    #[test]
    fn is_error_prefixes_tool_result_text() {
        let t = tool_result_as_text(&ToolResultContent::Text("boom".into()), true);
        assert_eq!(t, "ERROR: boom");
    }

    // ---------- reasoning / thinking ----------

    #[test]
    fn reasoning_content_field_becomes_thinking_block() {
        let mut out = Vec::new();
        push_reasoning_content(Some("let me think...\nthe answer is 42".into()), &mut out);
        assert_eq!(out.len(), 1);
        match &out[0] {
            ContentBlock::Thinking { thinking, replay } => {
                assert_eq!(thinking, "let me think...\nthe answer is 42");
                assert!(replay.is_none());
            }
            _ => panic!("expected Thinking block"),
        }
    }

    #[test]
    fn reasoning_content_blank_or_missing_emits_nothing() {
        let mut out = Vec::new();
        push_reasoning_content(None, &mut out);
        push_reasoning_content(Some(String::new()), &mut out);
        push_reasoning_content(Some("   \n\t".into()), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn inline_think_tags_split_into_alternating_blocks() {
        let mut out = Vec::new();
        push_text_with_inline_thinking(
            "before <think>reasoning here</think>after the tag",
            &mut out,
        );
        assert_eq!(out.len(), 3);
        match &out[0] {
            ContentBlock::Text { text } => assert_eq!(text, "before "),
            _ => panic!(),
        }
        match &out[1] {
            ContentBlock::Thinking { thinking, .. } => {
                assert_eq!(thinking, "reasoning here");
            }
            _ => panic!(),
        }
        match &out[2] {
            ContentBlock::Text { text } => assert_eq!(text, "after the tag"),
            _ => panic!(),
        }
    }

    #[test]
    fn inline_think_handles_multiple_blocks() {
        let mut out = Vec::new();
        push_text_with_inline_thinking(
            "<think>first</think>middle<think>second</think>tail",
            &mut out,
        );
        let kinds: Vec<&str> = out
            .iter()
            .map(|b| match b {
                ContentBlock::Text { .. } => "T",
                ContentBlock::Thinking { .. } => "R",
                _ => "?",
            })
            .collect();
        assert_eq!(kinds, vec!["R", "T", "R", "T"]);
    }

    #[test]
    fn unclosed_think_tag_treats_rest_as_thinking() {
        // A model that hits max_tokens partway through reasoning. We'd rather
        // preserve the partial reasoning than lose it entirely.
        let mut out = Vec::new();
        push_text_with_inline_thinking("visible <think>started but never finished", &mut out);
        assert_eq!(out.len(), 2);
        match &out[0] {
            ContentBlock::Text { text } => assert_eq!(text, "visible "),
            _ => panic!(),
        }
        match &out[1] {
            ContentBlock::Thinking { thinking, .. } => {
                assert_eq!(thinking, "started but never finished");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn empty_think_tags_are_skipped() {
        let mut out = Vec::new();
        push_text_with_inline_thinking("hello <think></think>world", &mut out);
        // The empty Thinking is dropped; surrounding text survives. No empty
        // Text spans either.
        assert_eq!(out.len(), 2);
        assert!(matches!(&out[0], ContentBlock::Text { text } if text == "hello "));
        assert!(matches!(&out[1], ContentBlock::Text { text } if text == "world"));
    }

    #[test]
    fn no_tags_emits_single_text_block() {
        let mut out = Vec::new();
        push_text_with_inline_thinking("just some text", &mut out);
        assert_eq!(out.len(), 1);
        assert!(matches!(&out[0], ContentBlock::Text { text } if text == "just some text"));
    }

    #[test]
    fn thinking_block_in_assistant_emits_reasoning_content_field() {
        let blocks = vec![
            ContentBlock::Thinking {
                replay: None,
                thinking: "I should look at foo.txt first".into(),
            },
            ContentBlock::Text {
                text: "let me check".into(),
            },
            ContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "foo.txt"}),
                replay: None,
            },
        ];
        let mut out = Vec::new();
        convert_assistant_message(&blocks, &mut out);
        assert_eq!(out.len(), 1);
        let m = &out[0];
        assert_eq!(m.role, "assistant");
        assert!(matches!(m.content, Some(OaContent::Text(ref s)) if s == "let me check"));
        assert!(m.tool_calls.is_some());
        assert_eq!(
            m.reasoning_content.as_deref(),
            Some("I should look at foo.txt first")
        );
        // Wire check: the field actually appears.
        let v = serde_json::to_value(m).unwrap();
        assert_eq!(v["reasoning_content"], "I should look at foo.txt first");
    }

    #[test]
    fn multiple_thinking_blocks_concatenate_with_blank_line() {
        let blocks = vec![
            ContentBlock::Thinking {
                replay: None,
                thinking: "first thought".into(),
            },
            ContentBlock::Thinking {
                replay: None,
                thinking: "second thought".into(),
            },
            ContentBlock::Text {
                text: "answer".into(),
            },
        ];
        let mut out = Vec::new();
        convert_assistant_message(&blocks, &mut out);
        assert_eq!(
            out[0].reasoning_content.as_deref(),
            Some("first thought\n\nsecond thought")
        );
    }

    #[test]
    fn no_thinking_means_field_is_omitted_from_wire() {
        let blocks = vec![ContentBlock::Text { text: "hi".into() }];
        let mut out = Vec::new();
        convert_assistant_message(&blocks, &mut out);
        let v = serde_json::to_value(&out[0]).unwrap();
        assert!(
            v.get("reasoning_content").is_none(),
            "absent reasoning must not serialize the field"
        );
    }

    // ---------- streaming ----------

    /// Feed a list of raw `data:` payloads through [`ChatStreamState`]
    /// as if they were decoded SSE frames. Returns all emitted events
    /// plus the final `Completed` event.
    fn drive_stream(frames: &[&str]) -> Vec<ModelEvent> {
        let mut state = ChatStreamState::default();
        let mut out = Vec::new();
        for f in frames {
            if *f == "[DONE]" {
                out.push(state.finalize().expect("finalize ok"));
                return out;
            }
            let ev: OaStreamEvent = serde_json::from_str(f).expect("valid json");
            for e in state.consume(ev).expect("consume ok") {
                out.push(e);
            }
        }
        out.push(state.finalize().expect("finalize ok"));
        out
    }

    #[test]
    fn streaming_plain_text_emits_deltas_and_completed() {
        let frames = [
            r#"{"choices":[{"delta":{"role":"assistant"}}]}"#,
            r#"{"choices":[{"delta":{"content":"Hel"}}]}"#,
            r#"{"choices":[{"delta":{"content":"lo"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#,
            "[DONE]",
        ];
        let events = drive_stream(&frames);
        let deltas: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                ModelEvent::TextDelta { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["Hel", "lo"]);
        // Last event is Completed with assembled text and normalized stop_reason.
        let last = events.last().unwrap();
        match last {
            ModelEvent::Completed {
                content,
                stop_reason,
                usage,
            } => {
                assert_eq!(content.len(), 1);
                assert!(matches!(&content[0], ContentBlock::Text { text } if text == "Hello"));
                assert_eq!(stop_reason.as_deref(), Some("end_turn"));
                assert_eq!(usage.input_tokens, 5);
                assert_eq!(usage.output_tokens, 2);
            }
            _ => panic!("expected Completed, got {last:?}"),
        }
    }

    #[test]
    fn streaming_tool_call_arguments_reassemble_in_index_order() {
        let frames = [
            // Tool call appears in pieces — id/name first, then arguments
            // split across three frames (realistic for long argument JSON).
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            "[DONE]",
        ];
        let events = drive_stream(&frames);
        // Exactly one fully-assembled ToolCall event fires on finish_reason.
        let tool_calls: Vec<&ModelEvent> = events
            .iter()
            .filter(|e| matches!(e, ModelEvent::ToolCall { .. }))
            .collect();
        assert_eq!(tool_calls.len(), 1);
        match tool_calls[0] {
            ModelEvent::ToolCall { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "bash");
                assert_eq!(input, &serde_json::json!({"command": "ls"}));
            }
            _ => unreachable!(),
        }
        // The terminal Completed carries the same tool_use block plus
        // the normalized stop_reason.
        match events.last().unwrap() {
            ModelEvent::Completed {
                content,
                stop_reason,
                ..
            } => {
                assert_eq!(content.len(), 1);
                assert!(matches!(
                    &content[0],
                    ContentBlock::ToolUse { name, id, .. }
                    if name == "bash" && id == "call_1"
                ));
                assert_eq!(stop_reason.as_deref(), Some("tool_use"));
            }
            e => panic!("expected Completed, got {e:?}"),
        }
    }

    #[test]
    fn streaming_reasoning_content_fields_become_thinking_deltas() {
        let frames = [
            r#"{"choices":[{"delta":{"reasoning_content":"I should "}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"check foo"}}]}"#,
            r#"{"choices":[{"delta":{"content":"Result"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ];
        let events = drive_stream(&frames);
        let thinking: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                ModelEvent::ThinkingDelta { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking, vec!["I should ", "check foo"]);
        // Completed orders Thinking before Text.
        match events.last().unwrap() {
            ModelEvent::Completed { content, .. } => {
                assert_eq!(content.len(), 2);
                assert!(matches!(
                    &content[0],
                    ContentBlock::Thinking { thinking, .. }
                    if thinking == "I should check foo"
                ));
                assert!(matches!(&content[1], ContentBlock::Text { text } if text == "Result"));
            }
            e => panic!("expected Completed, got {e:?}"),
        }
    }

    #[test]
    fn streaming_inline_think_tags_split_in_completed() {
        // llama.cpp-style inline reasoning: the reasoning arrives in the
        // `content` field wrapped in `<think>...</think>`. Deltas pass
        // through as TextDelta (the UI sees the raw tag briefly), but
        // Completed's canonical content splits the tag into a Thinking
        // block — matching what the non-streaming path produces.
        let frames = [
            r#"{"choices":[{"delta":{"content":"<think>reasoning</think>answer"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ];
        let events = drive_stream(&frames);
        match events.last().unwrap() {
            ModelEvent::Completed { content, .. } => {
                assert_eq!(content.len(), 2);
                assert!(matches!(
                    &content[0],
                    ContentBlock::Thinking { thinking, .. }
                    if thinking == "reasoning"
                ));
                assert!(matches!(&content[1], ContentBlock::Text { text } if text == "answer"));
            }
            e => panic!("expected Completed, got {e:?}"),
        }
    }

    #[test]
    fn streaming_handles_missing_done_sentinel() {
        // Some older Ollama builds close the stream without a `[DONE]`
        // marker. finalize() should still produce a Completed event
        // from whatever was accumulated.
        let mut state = ChatStreamState::default();
        let frames = [
            r#"{"choices":[{"delta":{"content":"hi"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ];
        for f in frames {
            let ev: OaStreamEvent = serde_json::from_str(f).unwrap();
            for _ in state.consume(ev).unwrap() {}
        }
        // Stream closed without [DONE] — finalize still works.
        match state.finalize().unwrap() {
            ModelEvent::Completed {
                content,
                stop_reason,
                ..
            } => {
                assert_eq!(content.len(), 1);
                assert!(matches!(&content[0], ContentBlock::Text { text } if text == "hi"));
                assert_eq!(stop_reason.as_deref(), Some("end_turn"));
            }
            e => panic!("expected Completed, got {e:?}"),
        }
    }

    #[test]
    fn streaming_tool_call_with_empty_arguments_parses_as_empty_object() {
        // A zero-arg tool call: the arguments field can stream as "" or
        // never be updated after the initial frame. It should still
        // produce a valid JSON object in the ToolCall event.
        let frames = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"now","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            "[DONE]",
        ];
        let events = drive_stream(&frames);
        let tc = events
            .iter()
            .find(|e| matches!(e, ModelEvent::ToolCall { .. }))
            .expect("ToolCall emitted");
        match tc {
            ModelEvent::ToolCall { input, .. } => {
                assert_eq!(input, &serde_json::json!({}));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn streaming_finish_reason_length_maps_to_max_tokens() {
        let frames = [
            r#"{"choices":[{"delta":{"content":"truncated"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#,
            "[DONE]",
        ];
        let events = drive_stream(&frames);
        match events.last().unwrap() {
            ModelEvent::Completed { stop_reason, .. } => {
                assert_eq!(stop_reason.as_deref(), Some("max_tokens"));
            }
            e => panic!("expected Completed, got {e:?}"),
        }
    }

    #[test]
    fn take_sse_event_splits_on_double_newline() {
        let mut buf = b"data: a\n\ndata: b\n\npartial".to_vec();
        let a = take_sse_event(&mut buf).unwrap();
        assert_eq!(a, b"data: a\n\n");
        let b = take_sse_event(&mut buf).unwrap();
        assert_eq!(b, b"data: b\n\n");
        assert!(take_sse_event(&mut buf).is_none());
        assert_eq!(buf, b"partial".to_vec());
    }

    #[test]
    fn parse_sse_data_strips_leading_space_and_joins_lines() {
        let payload = b"data: {\"first\":1}\ndata:{\"second\":2}\n\n";
        let data = parse_sse_data(payload).unwrap();
        assert_eq!(data, "{\"first\":1}\n{\"second\":2}");
    }
}
