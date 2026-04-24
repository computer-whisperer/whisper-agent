//! Anthropic Messages + Models API client.
//!
//! Implements [`ModelProvider`] for the `https://api.anthropic.com` endpoints. All
//! Anthropic-specific wire types (system-block cache_control, tool schema shape,
//! `MessageResponse` raw usage) are module-private — external code sees only the
//! normalized [`ModelRequest`] / [`ModelResponse`].
//!
//! Cache breakpoints are driven by [`ModelRequest::cache_breakpoints`]. Each entry
//! is translated into a `cache_control: {"type":"ephemeral","ttl":"1h"}` marker on
//! the corresponding wire element — the system block, the last tool declaration,
//! or the last content block of a message. 1h TTL requires the
//! `extended-cache-ttl-2025-04-11` beta header, which this client always sends.

use std::collections::HashSet;

use async_stream::try_stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use whisper_agent_protocol::{ContentBlock, Message, ProviderReplay, ToolResultContent, Usage};

use crate::providers::model::{
    BoxFuture, BoxStream, CacheBreakpoint, ModelError, ModelEvent, ModelInfo, ModelProvider,
    ModelRequest, ModelResponse, ToolSpec,
};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const MODELS_URL: &str = "https://api.anthropic.com/v1/models";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const EXTENDED_CACHE_BETA: &str = "extended-cache-ttl-2025-04-11";
const CACHE_TTL_1H: &str = "1h";

/// Identifier stored on [`ProviderReplay::provider`] for blobs minted by this
/// backend. Anthropic `Thinking` blocks carry their opaque `signature` here on
/// inbound and have it re-extracted on outbound so the wire stays clean.
const PROVIDER_TAG: &str = "anthropic";

/// Minimum character growth between consecutive `ToolCallStreaming`
/// emissions for the same tool-use block. 32 chars keeps updates
/// perceptible (roughly every few JSON tokens) without one wire event
/// per `input_json_delta` chunk, which can arrive byte-by-byte on
/// long-argument tool calls.
const TOOL_STREAMING_CHAR_STEP: u32 = 32;

pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
}

impl AnthropicClient {
    pub fn new(api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
        }
    }

    async fn do_create_message(
        &self,
        req: &ModelRequest<'_>,
        cancel: &CancellationToken,
    ) -> Result<ModelResponse, ModelError> {
        let body = build_request_body(req);
        let send = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", EXTENDED_CACHE_BETA)
            .header("content-type", "application/json")
            .json(&body)
            .send();
        let resp = tokio::select! {
            r = send => r.map_err(|e| ModelError::Transport(e.to_string()))?,
            _ = cancel.cancelled() => return Err(ModelError::Cancelled),
        };
        let status = resp.status();
        if !status.is_success() {
            let body = tokio::select! {
                b = resp.text() => b.unwrap_or_default(),
                _ = cancel.cancelled() => return Err(ModelError::Cancelled),
            };
            return Err(ModelError::Api {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: MessageResponse = tokio::select! {
            r = resp.json() => r.map_err(|e| ModelError::Transport(e.to_string()))?,
            _ = cancel.cancelled() => return Err(ModelError::Cancelled),
        };
        Ok(ModelResponse {
            content: parsed.content.into_iter().map(wire_to_block).collect(),
            stop_reason: parsed.stop_reason,
            usage: Usage {
                input_tokens: parsed.usage.input_tokens,
                output_tokens: parsed.usage.output_tokens,
                cache_read_input_tokens: parsed.usage.cache_read_input_tokens,
                cache_creation_input_tokens: parsed.usage.cache_creation_input_tokens,
            },
        })
    }

    fn do_create_message_streaming<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        let mut body = build_request_body(req);
        body.stream = true;
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        Box::pin(try_stream! {
            let send = http
                .post(API_URL)
                .header("x-api-key", &api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("anthropic-beta", EXTENDED_CACHE_BETA)
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .json(&body)
                .send();
            let resp: reqwest::Response = tokio::select! {
                r = send => r.map_err(|e| ModelError::Transport(e.to_string())),
                _ = cancel.cancelled() => Err(ModelError::Cancelled),
            }?;
            let status = resp.status();
            if !status.is_success() {
                let err_body = resp.text().await.unwrap_or_default();
                Err(ModelError::Api { status: status.as_u16(), body: err_body })?;
                return;
            }
            let mut bytes = resp.bytes_stream();
            let mut sse_buf: Vec<u8> = Vec::new();
            let mut state = StreamState::default();
            loop {
                // Dropping `bytes` when the cancel branch wins aborts
                // the underlying reqwest stream — the remote sees a
                // dropped connection and stops generating tokens.
                let next = tokio::select! {
                    n = bytes.next() => match n {
                        Some(Ok(b)) => Ok(Some(b)),
                        Some(Err(e)) => Err(ModelError::Transport(e.to_string())),
                        None => Ok(None),
                    },
                    _ = cancel.cancelled() => Err(ModelError::Cancelled),
                };
                let Some(chunk) = next? else { break };
                sse_buf.extend_from_slice(&chunk);
                while let Some(event_payload) = take_sse_event(&mut sse_buf) {
                    let Some(raw) = parse_sse_data(&event_payload) else {
                        continue;
                    };
                    let ev: AnthropicStreamEvent = match serde_json::from_str(&raw) {
                        Ok(ev) => ev,
                        // Skip unknown event shapes rather than terminating
                        // the stream — the server may add new event types.
                        Err(_) => continue,
                    };
                    for out in state.consume(ev)? {
                        yield out;
                    }
                    if state.done {
                        return;
                    }
                }
            }
            // The server closed the stream without sending message_stop —
            // treat as transport failure so the scheduler can decide
            // whether to retry.
            Err(ModelError::Transport(
                "SSE stream ended without message_stop".into(),
            ))?;
        })
    }

    async fn do_list_models(&self) -> Result<Vec<ModelInfo>, ModelError> {
        let resp = self
            .http
            .get(MODELS_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
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
        let parsed: ListModelsResponse = resp
            .json()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        Ok(parsed
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                display_name: Some(m.display_name),
                // Anthropic's /v1/models doesn't publish context or
                // output limits; surface as unknown until we add a
                // model-registry override.
                context_window: None,
                max_output_tokens: None,
            })
            .collect())
    }
}

impl ModelProvider for AnthropicClient {
    fn create_message<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<ModelResponse, ModelError>> {
        Box::pin(self.do_create_message(req, cancel))
    }

    fn create_message_streaming<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        self.do_create_message_streaming(req, cancel)
    }

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<ModelInfo>, ModelError>> {
        Box::pin(self.do_list_models())
    }
}

fn spec_to_anthropic_tool(t: &ToolSpec) -> AnthropicTool {
    AnthropicTool {
        name: t.name.clone(),
        description: t.description.clone(),
        input_schema: t.input_schema.clone(),
        cache_control: None,
    }
}

/// Build the Anthropic wire body from a generic [`ModelRequest`], translating each
/// [`CacheBreakpoint`] into `cache_control` markers on the appropriate element.
fn build_request_body<'a>(req: &'a ModelRequest<'a>) -> CreateMessageRequest<'a> {
    let cache_system = req
        .cache_breakpoints
        .iter()
        .any(|b| matches!(b, CacheBreakpoint::AfterSystem));
    let cache_tools = req
        .cache_breakpoints
        .iter()
        .any(|b| matches!(b, CacheBreakpoint::AfterTools));
    let message_cache_indices: HashSet<usize> = req
        .cache_breakpoints
        .iter()
        .filter_map(|b| match b {
            CacheBreakpoint::AfterMessage(i) => Some(*i),
            _ => None,
        })
        .collect();

    // Empty system prompt → omit the system field entirely. Anthropic
    // accepts this, and it also sidesteps a hard constraint: the API
    // rejects `cache_control` on empty text blocks with a 400. A pod
    // whose `system_prompt_file` is missing (or not-yet-written) lands
    // here with an empty string; silently dropping the block is the
    // right move — there's nothing to cache anyway.
    let system = if req.system_prompt.is_empty() {
        Vec::new()
    } else {
        vec![SystemBlock {
            kind: "text",
            text: req.system_prompt.to_string(),
            cache_control: if cache_system {
                Some(CacheControl::ephemeral_1h())
            } else {
                None
            },
        }]
    };

    let mut tools: Vec<AnthropicTool> = req.tools.iter().map(spec_to_anthropic_tool).collect();
    if cache_tools && let Some(last) = tools.last_mut() {
        last.cache_control = Some(CacheControl::ephemeral_1h());
    }

    let messages: Vec<Value> = req
        .messages
        .iter()
        .enumerate()
        .map(|(i, m)| message_to_value(m, message_cache_indices.contains(&i)))
        .collect();

    CreateMessageRequest {
        model: req.model,
        max_tokens: req.max_tokens,
        system,
        tools,
        messages,
        stream: false,
    }
}

/// Serialize a message and optionally patch `cache_control` onto its final
/// content block.
///
/// Beyond the serde derive, we also fix up three things the wire expects:
/// - `Role::ToolResult` → `user` on the role field (Anthropic only accepts
///   `user` and `assistant`).
/// - `Role::System` (mid-conversation injection, not the thread-prefix
///   system prompt — that one is lifted into the top-level `system` field
///   upstream) → `user` on the role field, with each text block's content
///   wrapped in `<system-reminder>...</system-reminder>` so the model can
///   tell harness guidance from real user speech.
/// - `Thinking.replay` → wire-level `signature` field when the block was
///   minted by this backend; stripped otherwise (a blob tagged for another
///   provider is meaningless and the server 400s on unknown fields).
fn message_to_value(m: &Message, cache_last_block: bool) -> Value {
    let mut v = serde_json::to_value(m).expect("Message is serializable");
    if let Some(role) = v.get_mut("role") {
        match role.as_str() {
            Some("tool_result") => *role = Value::String("user".into()),
            Some("system") => {
                *role = Value::String("user".into());
                wrap_text_blocks_in_system_reminder(&mut v);
            }
            _ => {}
        }
    }
    if let Some(content) = v.get_mut("content").and_then(|c| c.as_array_mut()) {
        for block in content.iter_mut() {
            if let Some(obj) = block.as_object_mut() {
                fold_replay_to_wire(obj);
            }
        }
    }
    if !cache_last_block {
        return v;
    }
    let Some(content) = v.get_mut("content").and_then(|c| c.as_array_mut()) else {
        return v;
    };
    let Some(last_block) = content.last_mut().and_then(|b| b.as_object_mut()) else {
        return v;
    };
    last_block.insert(
        "cache_control".to_string(),
        serde_json::json!({"type": "ephemeral", "ttl": CACHE_TTL_1H}),
    );
    v
}

/// Wrap every text block's content in `<system-reminder>...</system-reminder>`
/// tags. Applied to mid-conversation `Role::System` messages when they're
/// translated to wire `role: "user"` so the model can distinguish harness
/// guidance from real user speech (Anthropic models are trained on this
/// convention; other tag choices would be out-of-distribution). Non-text
/// blocks pass through untouched — we don't expect any in a system
/// injection, but rewriting them blindly would be worse than a no-op.
fn wrap_text_blocks_in_system_reminder(v: &mut Value) {
    let Some(content) = v.get_mut("content").and_then(|c| c.as_array_mut()) else {
        return;
    };
    for block in content.iter_mut() {
        let Some(obj) = block.as_object_mut() else {
            continue;
        };
        if obj.get("type").and_then(Value::as_str) != Some("text") {
            continue;
        }
        let Some(Value::String(text)) = obj.remove("text") else {
            continue;
        };
        obj.insert(
            "text".into(),
            Value::String(format!("<system-reminder>\n{text}\n</system-reminder>")),
        );
    }
}

/// Translate a serialized content block from our normalized shape into
/// Anthropic's wire shape. For `thinking` blocks, extract the `signature`
/// out of `replay.data` when the replay was minted by this backend. For all
/// blocks, drop the `replay` field — Anthropic doesn't accept it on the wire.
fn fold_replay_to_wire(obj: &mut serde_json::Map<String, Value>) {
    let is_thinking = obj.get("type").and_then(Value::as_str) == Some("thinking");
    let replay = obj.remove("replay");
    if !is_thinking {
        return;
    }
    let Some(replay) = replay else { return };
    let Some(provider) = replay.get("provider").and_then(Value::as_str) else {
        return;
    };
    if provider != PROVIDER_TAG {
        return;
    }
    if let Some(sig) = replay.get("data").and_then(|d| d.get("signature")) {
        obj.insert("signature".into(), sig.clone());
    }
}

// ---------- Wire types (private to this module) ----------

#[derive(Serialize, Debug)]
struct CreateMessageRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<SystemBlock>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    messages: Vec<Value>,
    /// Only emitted on the streaming path. The batch path leaves this at
    /// `false` so the server returns a single JSON body.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize, Debug, Clone)]
struct SystemBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Serialize, Debug, Clone)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<&'static str>,
}

impl CacheControl {
    fn ephemeral_1h() -> Self {
        Self {
            kind: "ephemeral",
            ttl: Some(CACHE_TTL_1H),
        }
    }
}

#[derive(Serialize, Debug, Clone)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Deserialize, Debug)]
struct MessageResponse {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    role: String,
    content: Vec<AnthropicWireBlock>,
    #[allow(dead_code)]
    model: String,
    stop_reason: Option<String>,
    #[allow(dead_code)]
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

/// Anthropic's wire-level content block shape. Mirrors the JSON the API emits
/// 1:1 — the `signature` field on `thinking` lives here at the top of the
/// block, not nested inside a `replay` container like our normalized
/// [`ContentBlock`] uses. Translation happens in [`wire_to_block`].
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicWireBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(default)]
        is_error: bool,
    },
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
}

/// Fold the Anthropic-shaped `signature` into our provider-tagged
/// [`ProviderReplay`] so the block can round-trip through the conversation
/// store and other backends (which will strip the replay on outbound since
/// the provider tag won't match).
fn wire_to_block(w: AnthropicWireBlock) -> ContentBlock {
    match w {
        AnthropicWireBlock::Text { text } => ContentBlock::Text { text },
        AnthropicWireBlock::ToolUse { id, name, input } => ContentBlock::ToolUse {
            id,
            name,
            input,
            replay: None,
        },
        AnthropicWireBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        },
        AnthropicWireBlock::Thinking {
            thinking,
            signature,
        } => {
            let replay = signature.map(|s| ProviderReplay {
                provider: PROVIDER_TAG.into(),
                data: serde_json::json!({"signature": s}),
            });
            ContentBlock::Thinking { replay, thinking }
        }
    }
}

#[derive(Deserialize, Debug, Clone, Copy, Default)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    /// Absent on `message_start` in practice — the server fills it in on
    /// the terminal `message_delta`. Default to 0 so streaming parses cleanly.
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct ListModelsResponse {
    data: Vec<AnthropicModelInfo>,
}

#[derive(Deserialize, Debug)]
struct AnthropicModelInfo {
    id: String,
    display_name: String,
}

// ---------- Streaming SSE parse + state machine ----------

/// Pull one complete SSE event (terminated by a blank line, i.e. `\n\n` or
/// `\r\n\r\n`) out of `buf` and return its raw bytes (including the blank
/// line terminator). Returns `None` when no complete event is buffered yet.
/// Implemented as a byte scan so chunk boundaries in the middle of a line
/// don't confuse us.
fn take_sse_event(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    // Look for `\n\n` or `\r\n\r\n`. SSE spec allows either but Anthropic
    // consistently uses `\n\n`.
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

/// Extract the concatenated `data:` payload from a single SSE event. The
/// SSE spec allows multiple `data:` lines per event to be joined with
/// newlines; we honour that even though Anthropic sends one per event.
/// `event:` lines are discarded — Anthropic's JSON payload always carries
/// the `type` so they'd be redundant anyway.
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

/// Discriminator for the currently-open content block, so we know how to
/// interpret the deltas that follow its `content_block_start`.
#[derive(Debug)]
enum OpenBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        args_json: String,
        /// Last `args_json.len()` we reported via
        /// [`ModelEvent::ToolCallStreaming`]. Throttles streaming
        /// updates so a rapid-fire `input_json_delta` burst doesn't
        /// spam one wire event per byte — we only re-emit after the
        /// buffer has grown by at least [`TOOL_STREAMING_CHAR_STEP`]
        /// chars since the previous emission.
        last_emitted_chars: u32,
    },
    /// A block type we don't model (e.g. `server_tool_use`). Deltas are
    /// ignored; the block is dropped at `content_block_stop`.
    Other,
}

/// Running state for one Anthropic streaming call. Tracks the currently-open
/// content block, accumulates deltas until each block closes, and collects
/// the final content list for the terminal [`ModelEvent::Completed`].
#[derive(Default)]
struct StreamState {
    open: Option<OpenBlock>,
    content: Vec<ContentBlock>,
    input_tokens: u32,
    output_tokens: u32,
    cache_read_input_tokens: u32,
    cache_creation_input_tokens: u32,
    stop_reason: Option<String>,
    /// Set when we've emitted `Completed`; callers stop pulling the stream
    /// after this so a stray `message_stop` (which we tolerate) doesn't
    /// emit a second Completed.
    done: bool,
}

impl StreamState {
    /// Fold one SSE event into the running state, returning any [`ModelEvent`]s
    /// that should be emitted to the consumer. Most events either yield zero
    /// or one event; `message_stop` yields the terminal `Completed`.
    fn consume(&mut self, event: AnthropicStreamEvent) -> Result<Vec<ModelEvent>, ModelError> {
        let mut out = Vec::new();
        match event {
            AnthropicStreamEvent::MessageStart { message } => {
                self.input_tokens = message.usage.input_tokens;
                self.cache_read_input_tokens = message.usage.cache_read_input_tokens;
                self.cache_creation_input_tokens = message.usage.cache_creation_input_tokens;
            }
            AnthropicStreamEvent::ContentBlockStart { content_block, .. } => {
                self.open = Some(match content_block {
                    AnthropicWireBlock::Text { text } => OpenBlock::Text { text },
                    AnthropicWireBlock::Thinking {
                        thinking,
                        signature,
                    } => OpenBlock::Thinking {
                        thinking,
                        signature,
                    },
                    AnthropicWireBlock::ToolUse { id, name, input: _ } => {
                        // Announce the tool call the moment we know its
                        // name — args follow over many `input_json_delta`
                        // events, and we want the UI to frame the row
                        // immediately rather than stay silent for the
                        // seconds it takes the model to stream a long
                        // args JSON.
                        out.push(ModelEvent::ToolCallStreaming {
                            id: id.clone(),
                            name: name.clone(),
                            args_chars: 0,
                        });
                        OpenBlock::ToolUse {
                            id,
                            name,
                            args_json: String::new(),
                            last_emitted_chars: 0,
                        }
                    }
                    // Tool-result blocks don't appear on assistant turns;
                    // anything else (server_tool_use, etc.) we treat as
                    // opaque and drop.
                    AnthropicWireBlock::ToolResult { .. } => OpenBlock::Other,
                });
            }
            AnthropicStreamEvent::ContentBlockDelta { delta, .. } => {
                match (&mut self.open, delta) {
                    (Some(OpenBlock::Text { text }), AnthropicDelta::TextDelta { text: chunk }) => {
                        text.push_str(&chunk);
                        out.push(ModelEvent::TextDelta { text: chunk });
                    }
                    (
                        Some(OpenBlock::Thinking { thinking, .. }),
                        AnthropicDelta::ThinkingDelta { thinking: chunk },
                    ) => {
                        thinking.push_str(&chunk);
                        out.push(ModelEvent::ThinkingDelta { text: chunk });
                    }
                    (
                        Some(OpenBlock::Thinking { signature, .. }),
                        AnthropicDelta::SignatureDelta {
                            signature: sig_chunk,
                        },
                    ) => {
                        // Signature arrives as a separate delta type at the
                        // end of a thinking block. Accumulate silently — the
                        // consumer sees it when `Completed` lands.
                        signature
                            .get_or_insert_with(String::new)
                            .push_str(&sig_chunk);
                    }
                    (
                        Some(OpenBlock::ToolUse {
                            id,
                            name,
                            args_json,
                            last_emitted_chars,
                        }),
                        AnthropicDelta::InputJsonDelta {
                            partial_json: chunk,
                        },
                    ) => {
                        args_json.push_str(&chunk);
                        // No live *parsed* tool-arg streaming — we can't
                        // partial-parse JSON reliably. But we can surface
                        // a cumulative char count so the UI's placeholder
                        // chip shows progress. Throttled by char-step to
                        // keep the wire quiet.
                        let total = args_json.len() as u32;
                        if total >= last_emitted_chars.saturating_add(TOOL_STREAMING_CHAR_STEP) {
                            *last_emitted_chars = total;
                            out.push(ModelEvent::ToolCallStreaming {
                                id: id.clone(),
                                name: name.clone(),
                                args_chars: total,
                            });
                        }
                    }
                    _ => {} // mismatched delta for current block — drop.
                }
            }
            AnthropicStreamEvent::ContentBlockStop { .. } => match self.open.take() {
                Some(OpenBlock::Text { text }) => {
                    self.content.push(ContentBlock::Text { text });
                }
                Some(OpenBlock::Thinking {
                    thinking,
                    signature,
                }) => {
                    let replay = signature.map(|s| ProviderReplay {
                        provider: PROVIDER_TAG.into(),
                        data: serde_json::json!({"signature": s}),
                    });
                    self.content
                        .push(ContentBlock::Thinking { replay, thinking });
                }
                Some(OpenBlock::ToolUse {
                    id,
                    name,
                    args_json,
                    last_emitted_chars: _,
                }) => {
                    let input: Value = if args_json.is_empty() {
                        Value::Object(Default::default())
                    } else {
                        serde_json::from_str(&args_json).map_err(|e| {
                            ModelError::Transport(format!(
                                "tool_use input_json_delta didn't parse: {e}"
                            ))
                        })?
                    };
                    out.push(ModelEvent::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    });
                    self.content.push(ContentBlock::ToolUse {
                        id,
                        name,
                        input,
                        replay: None,
                    });
                }
                Some(OpenBlock::Other) | None => {}
            },
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                if let Some(sr) = delta.stop_reason {
                    self.stop_reason = Some(sr);
                }
                // message_delta carries the final output_tokens on
                // `usage.output_tokens`; input_tokens were set at
                // message_start and don't change.
                if let Some(u) = usage
                    && let Some(ot) = u.output_tokens
                {
                    self.output_tokens = ot;
                }
            }
            AnthropicStreamEvent::MessageStop => {
                out.push(ModelEvent::Completed {
                    content: std::mem::take(&mut self.content),
                    stop_reason: self.stop_reason.take(),
                    usage: Usage {
                        input_tokens: self.input_tokens,
                        output_tokens: self.output_tokens,
                        cache_read_input_tokens: self.cache_read_input_tokens,
                        cache_creation_input_tokens: self.cache_creation_input_tokens,
                    },
                });
                self.done = true;
            }
            AnthropicStreamEvent::Ping | AnthropicStreamEvent::Other => {}
            AnthropicStreamEvent::Error { error } => {
                Err(ModelError::Api {
                    status: 500,
                    body: error.message,
                })?;
            }
        }
        Ok(out)
    }
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamEvent {
    MessageStart {
        message: MessageStartEnvelope,
    },
    ContentBlockStart {
        #[allow(dead_code)]
        index: usize,
        content_block: AnthropicWireBlock,
    },
    ContentBlockDelta {
        #[allow(dead_code)]
        index: usize,
        delta: AnthropicDelta,
    },
    ContentBlockStop {
        #[allow(dead_code)]
        index: usize,
    },
    MessageDelta {
        delta: MessageDeltaPayload,
        #[serde(default)]
        usage: Option<MessageDeltaUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicStreamError,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Debug)]
struct MessageStartEnvelope {
    usage: AnthropicUsage,
}

// Variant names mirror the wire tags Anthropic emits (text_delta,
// input_json_delta, etc.) — the "Delta" postfix is intentional parity
// with the on-the-wire names, not a naming smell.
#[allow(clippy::enum_variant_names)]
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Deserialize, Debug)]
struct MessageDeltaPayload {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct MessageDeltaUsage {
    #[serde(default)]
    output_tokens: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct AnthropicStreamError {
    #[serde(default)]
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use whisper_agent_protocol::{ContentBlock, Message, ToolResultContent};

    fn make_tools() -> Vec<ToolSpec> {
        vec![
            ToolSpec {
                name: "bash".into(),
                description: "run a shell command".into(),
                input_schema: json!({"type":"object"}),
            },
            ToolSpec {
                name: "read_file".into(),
                description: "read a file".into(),
                input_schema: json!({"type":"object"}),
            },
        ]
    }

    fn make_messages() -> Vec<Message> {
        vec![
            Message::user_text("hello"),
            Message::assistant_blocks(vec![ContentBlock::Text {
                text: "hi back".into(),
            }]),
            Message::user_blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_1".into(),
                content: ToolResultContent::Text("ok".into()),
                is_error: false,
            }]),
        ]
    }

    #[test]
    fn no_breakpoints_leaves_wire_body_cache_free() {
        let tools = make_tools();
        let messages = make_messages();
        let req = ModelRequest {
            model: "claude-opus-4-6",
            max_tokens: 1024,
            system_prompt: "you are helpful",
            tools: &tools,
            messages: &messages,
            cache_breakpoints: &[],
        };
        let body = build_request_body(&req);
        let v = serde_json::to_value(&body).unwrap();
        // No cache_control anywhere.
        assert!(!v.to_string().contains("cache_control"));
    }

    #[test]
    fn empty_system_prompt_omits_system_field_and_cache_control() {
        // Regression: a pod whose system_prompt_file is missing loads
        // with an empty system prompt. The scheduler still emits the
        // AfterSystem breakpoint by default; Anthropic 400s on
        // cache_control over an empty text block. Adapter must drop
        // the system block entirely in that case.
        let tools = make_tools();
        let messages = make_messages();
        let req = ModelRequest {
            model: "claude-opus-4-6",
            max_tokens: 1024,
            system_prompt: "",
            tools: &tools,
            messages: &messages,
            cache_breakpoints: &[CacheBreakpoint::AfterSystem],
        };
        let body = build_request_body(&req);
        let v = serde_json::to_value(&body).unwrap();
        // `system` should be omitted entirely (skip_serializing_if).
        assert!(
            v.get("system").is_none(),
            "expected no `system` field, got {}",
            v
        );
        // And definitely no cache_control anywhere.
        assert!(!v.to_string().contains("cache_control"));
    }

    #[test]
    fn after_system_caches_system_block_only() {
        let tools = make_tools();
        let messages = make_messages();
        let req = ModelRequest {
            model: "claude-opus-4-6",
            max_tokens: 1024,
            system_prompt: "you are helpful",
            tools: &tools,
            messages: &messages,
            cache_breakpoints: &[CacheBreakpoint::AfterSystem],
        };
        let body = build_request_body(&req);
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(
            v["system"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        // No cache_control on tools or messages.
        for t in v["tools"].as_array().unwrap() {
            assert!(t.get("cache_control").is_none());
        }
        for m in v["messages"].as_array().unwrap() {
            for b in m["content"].as_array().unwrap() {
                assert!(b.get("cache_control").is_none());
            }
        }
    }

    #[test]
    fn after_tools_caches_last_tool_only() {
        let tools = make_tools();
        let messages = make_messages();
        let req = ModelRequest {
            model: "claude-opus-4-6",
            max_tokens: 1024,
            system_prompt: "you are helpful",
            tools: &tools,
            messages: &messages,
            cache_breakpoints: &[CacheBreakpoint::AfterTools],
        };
        let body = build_request_body(&req);
        let v = serde_json::to_value(&body).unwrap();
        let tools_arr = v["tools"].as_array().unwrap();
        assert!(tools_arr[0].get("cache_control").is_none());
        assert_eq!(
            tools_arr[1]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
    }

    #[test]
    fn after_message_caches_last_content_block_of_that_message() {
        let tools = make_tools();
        let messages = make_messages();
        let req = ModelRequest {
            model: "claude-opus-4-6",
            max_tokens: 1024,
            system_prompt: "you are helpful",
            tools: &tools,
            messages: &messages,
            cache_breakpoints: &[CacheBreakpoint::AfterMessage(2)],
        };
        let body = build_request_body(&req);
        let v = serde_json::to_value(&body).unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert!(msgs[0]["content"][0].get("cache_control").is_none());
        assert!(msgs[1]["content"][0].get("cache_control").is_none());
        assert_eq!(
            msgs[2]["content"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
    }

    #[test]
    fn full_policy_caches_system_and_last_user_side_message_only() {
        // Mirrors Claude Code: one cache_control on the system block, one on the
        // last user-side message. Tools and earlier messages ride the implicit
        // prefix extension — no cache_control on them.
        let tools = make_tools();
        let messages = make_messages();
        let breakpoints = crate::providers::model::default_cache_policy(&messages);
        let req = ModelRequest {
            model: "claude-opus-4-6",
            max_tokens: 1024,
            system_prompt: "you are helpful",
            tools: &tools,
            messages: &messages,
            cache_breakpoints: &breakpoints,
        };
        let body = build_request_body(&req);
        let v = serde_json::to_value(&body).unwrap();
        assert!(v["system"][0]["cache_control"].is_object());
        for t in v["tools"].as_array().unwrap() {
            assert!(t.get("cache_control").is_none());
        }
        let msgs = v["messages"].as_array().unwrap();
        assert!(msgs[0]["content"][0].get("cache_control").is_none());
        assert!(msgs[1]["content"][0].get("cache_control").is_none());
        assert!(msgs[2]["content"][0]["cache_control"].is_object());
    }

    // ---------- Reasoning replay ----------

    #[test]
    fn anthropic_thinking_inbound_captures_signature_as_replay() {
        let wire = json!({
            "type": "thinking",
            "thinking": "let me think",
            "signature": "sig-blob-base64"
        });
        let parsed: AnthropicWireBlock = serde_json::from_value(wire).unwrap();
        let block = wire_to_block(parsed);
        match block {
            ContentBlock::Thinking { replay, thinking } => {
                assert_eq!(thinking, "let me think");
                let r = replay.expect("signature should surface as replay");
                assert_eq!(r.provider, "anthropic");
                assert_eq!(r.data, json!({"signature": "sig-blob-base64"}));
            }
            _ => panic!("expected Thinking"),
        }
    }

    #[test]
    fn anthropic_outbound_puts_replay_back_as_signature() {
        // Round-trip: an anthropic-tagged Thinking block should serialize as
        // the wire shape with `signature` at the top of the block, and no
        // stray `replay` leaking through.
        let msg = Message::assistant_blocks(vec![ContentBlock::Thinking {
            replay: Some(ProviderReplay {
                provider: "anthropic".into(),
                data: json!({"signature": "sig-blob"}),
            }),
            thinking: "let me think".into(),
        }]);
        let v = message_to_value(&msg, false);
        let block = &v["content"][0];
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "let me think");
        assert_eq!(block["signature"], "sig-blob");
        assert!(block.get("replay").is_none());
    }

    #[test]
    fn anthropic_outbound_strips_foreign_provider_replay() {
        // A Thinking block minted by a different backend (user switched
        // models mid-thread) must not leak its opaque blob to Anthropic —
        // both the `replay` field and any pretend-`signature` must be
        // absent from the wire body.
        let msg = Message::assistant_blocks(vec![ContentBlock::Thinking {
            replay: Some(ProviderReplay {
                provider: "openai_responses".into(),
                data: json!({"encrypted_content": "opaque"}),
            }),
            thinking: "prior reasoning".into(),
        }]);
        let v = message_to_value(&msg, false);
        let block = &v["content"][0];
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "prior reasoning");
        assert!(block.get("replay").is_none());
        assert!(block.get("signature").is_none());
    }

    #[test]
    fn mid_conversation_system_becomes_user_with_wrapped_text() {
        // An injected Role::System message (not the thread-prefix prompt —
        // that one is lifted into the top-level `system` field upstream)
        // must land on the wire as role:"user" with each text block wrapped
        // in <system-reminder> tags so the model can distinguish harness
        // guidance from real user speech.
        let msg = Message::system_text("remember: check memory/ for prior context");
        let v = message_to_value(&msg, false);
        assert_eq!(v["role"], "user");
        let blocks = v["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(
            blocks[0]["text"],
            "<system-reminder>\nremember: check memory/ for prior context\n</system-reminder>"
        );
    }

    #[test]
    fn system_message_is_still_eligible_for_cache_control_marker() {
        // cache_last_block applies after the role/content rewrite, so an
        // AfterMessage breakpoint landing on an injected system message
        // still attaches cache_control to the last (now-wrapped) text block.
        let msg = Message::system_text("pinned harness note");
        let v = message_to_value(&msg, true);
        assert_eq!(v["role"], "user");
        let blocks = v["content"].as_array().unwrap();
        assert_eq!(
            blocks[0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
    }

    #[test]
    fn anthropic_outbound_strips_replay_from_tool_use() {
        // Replay on ToolUse (Gemini puts its thoughtSignature there) never
        // maps to anything on Anthropic's wire. Drop silently.
        let msg = Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: "toolu_1".into(),
            name: "bash".into(),
            input: json!({"cmd": "ls"}),
            replay: Some(ProviderReplay {
                provider: "gemini".into(),
                data: json!({"thought_signature": "gemini-blob"}),
            }),
        }]);
        let v = message_to_value(&msg, false);
        let block = &v["content"][0];
        assert_eq!(block["type"], "tool_use");
        assert!(block.get("replay").is_none());
        assert!(block.get("thought_signature").is_none());
    }

    // ---------- SSE parsing ----------

    #[test]
    fn take_sse_event_yields_whole_events_only() {
        // Half an event waiting for more bytes — should not yield.
        let mut buf = b"event: ping\ndata: {\"type\":\"ping\"}".to_vec();
        assert!(take_sse_event(&mut buf).is_none());
        // Complete the terminator — now yield exactly the first event
        // (buffer preserved for subsequent calls).
        buf.extend_from_slice(b"\n\nevent: done\ndata: {\"type\":\"message_stop\"}\n\n");
        let first = take_sse_event(&mut buf).expect("first event");
        assert!(std::str::from_utf8(&first).unwrap().contains("ping"));
        let second = take_sse_event(&mut buf).expect("second event");
        assert!(
            std::str::from_utf8(&second)
                .unwrap()
                .contains("message_stop")
        );
        assert!(take_sse_event(&mut buf).is_none());
    }

    #[test]
    fn parse_sse_data_joins_multi_line_data_fields() {
        let ev = b"event: x\ndata: line one\ndata: line two\n\n";
        assert_eq!(parse_sse_data(ev).as_deref(), Some("line one\nline two"));
    }

    // ---------- Streaming state machine ----------

    fn feed_events(events: &[&str]) -> (Vec<ModelEvent>, StreamState) {
        let mut state = StreamState::default();
        let mut out = Vec::new();
        for ev_json in events {
            let ev: AnthropicStreamEvent = serde_json::from_str(ev_json).unwrap();
            out.extend(state.consume(ev).unwrap());
        }
        (out, state)
    }

    #[test]
    fn text_block_emits_deltas_and_final_completed() {
        let events = [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":0}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let (evs, _) = feed_events(&events);
        let deltas: Vec<&str> = evs
            .iter()
            .filter_map(|e| match e {
                ModelEvent::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec!["hello ", "world"]);
        // Completed is last, with full content and usage.
        match evs.last().unwrap() {
            ModelEvent::Completed {
                content,
                stop_reason,
                usage,
            } => {
                assert_eq!(content.len(), 1);
                assert!(
                    matches!(&content[0], ContentBlock::Text { text } if text == "hello world")
                );
                assert_eq!(stop_reason.as_deref(), Some("end_turn"));
                assert_eq!(usage.input_tokens, 5);
                assert_eq!(usage.output_tokens, 2);
            }
            _ => panic!("last event must be Completed"),
        }
    }

    #[test]
    fn tool_use_emits_streaming_start_then_toolcall_after_args_assembled() {
        // Args stream in as fragments; a `ToolCallStreaming` fires
        // immediately with args_chars=0 (so the UI can frame the row
        // before the args finish). The fully-parsed `ToolCall` still
        // lands once at `content_block_stop`.
        let events = [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":1,"output_tokens":0}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"ls\"}"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let (evs, _) = feed_events(&events);
        // Expect: ToolCallStreaming(0) → ToolCall → Completed.
        // The two args fragments here total 15 chars, under the
        // throttling step, so no intermediate streaming events.
        match &evs[0] {
            ModelEvent::ToolCallStreaming {
                id,
                name,
                args_chars,
            } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "bash");
                assert_eq!(*args_chars, 0);
            }
            other => panic!("expected ToolCallStreaming first, got {other:?}"),
        }
        match &evs[1] {
            ModelEvent::ToolCall { id, name, input } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "bash");
                assert_eq!(input, &json!({"cmd": "ls"}));
            }
            other => panic!("expected ToolCall second, got {other:?}"),
        }
        match evs.last().unwrap() {
            ModelEvent::Completed {
                content,
                stop_reason,
                ..
            } => {
                assert_eq!(content.len(), 1);
                assert!(
                    matches!(&content[0], ContentBlock::ToolUse { name, .. } if name == "bash")
                );
                assert_eq!(stop_reason.as_deref(), Some("tool_use"));
            }
            _ => panic!("expected Completed last"),
        }
    }

    #[test]
    fn tool_use_emits_throttled_streaming_updates_for_long_args() {
        // Long args JSON should produce multiple `ToolCallStreaming`
        // emissions, one every ~`TOOL_STREAMING_CHAR_STEP` chars.
        let big_arg = "a".repeat(200);
        let fragment = format!(r#"{{"data":"{big_arg}"}}"#);
        let partial = fragment.replace('"', "\\\"");
        let delta = format!(
            r#"{{"type":"content_block_delta","index":0,"delta":{{"type":"input_json_delta","partial_json":"{partial}"}}}}"#,
        );
        let events = [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_big","name":"bash","input":{}}}"#,
            &delta,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let (evs, _) = feed_events(&events);
        let streaming_counts: Vec<u32> = evs
            .iter()
            .filter_map(|e| match e {
                ModelEvent::ToolCallStreaming { args_chars, .. } => Some(*args_chars),
                _ => None,
            })
            .collect();
        // First emission at 0, then at least one more after the big
        // delta pushes past the char step.
        assert!(streaming_counts.len() >= 2, "got {streaming_counts:?}");
        assert_eq!(streaming_counts[0], 0);
        assert!(streaming_counts.last().unwrap() > &32);
    }

    #[test]
    fn thinking_block_captures_signature_into_replay_at_block_stop() {
        let events = [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-blob"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let (evs, _) = feed_events(&events);
        match evs.last().unwrap() {
            ModelEvent::Completed { content, .. } => match &content[0] {
                ContentBlock::Thinking { replay, thinking } => {
                    assert_eq!(thinking, "hmm");
                    let r = replay.as_ref().expect("signature should fold into replay");
                    assert_eq!(r.provider, "anthropic");
                    assert_eq!(r.data, json!({"signature": "sig-blob"}));
                }
                _ => panic!("expected Thinking block"),
            },
            _ => panic!("expected Completed"),
        }
    }
}
