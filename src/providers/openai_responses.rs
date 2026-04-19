//! OpenAI Responses API backend.
//!
//! Speaks the `/v1/responses` wire shape — OpenAI's current production primitive,
//! recommended for all new integrations. Differs from Chat Completions in two
//! structurally important ways:
//!
//! - Input is a flat array of typed *items* (`message`, `function_call`,
//!   `function_call_output`, `reasoning`, …) rather than role-tagged messages.
//!   Tool calls and their outputs are first-class siblings correlated by
//!   `call_id`, which matches our content-block shape naturally.
//! - The system prompt is lifted out of `input` into a dedicated `instructions`
//!   field.
//!
//! This provider runs in *stateless* mode: it always sends the full input array
//! on every turn (no `previous_response_id`) and sets `store: false`. That
//! matches the scheduler's replay model — the scheduler is the source of truth
//! for conversation state.
//!
//! Reasoning replay: requests `include: ["reasoning.encrypted_content"]` so
//! the server returns `encrypted_content` on each `reasoning` item. Inbound,
//! the reasoning item's `id`, `encrypted_content`, and `summary` parts get
//! stashed into [`ContentBlock::Thinking::replay`] (tagged `openai_responses`).
//! Outbound, assistant `Thinking` blocks whose replay came from this backend
//! are echoed back as full `reasoning` items in `input` so the server can
//! resume its chain-of-thought. Blocks tagged for a different backend are
//! dropped — their opaque blob is meaningless here.

use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;
use whisper_agent_protocol::{
    ContentBlock, Message, ProviderReplay, Role, ToolResultContent, Usage,
};

use crate::providers::codex_auth::CodexAuth;
use crate::providers::model::{
    BoxFuture, BoxStream, ModelError, ModelEvent, ModelInfo, ModelProvider, ModelRequest,
    ModelResponse, ToolSpec,
};

pub const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// Identifier stored on [`ProviderReplay::provider`] for blobs minted by this
/// backend. Used on outbound to filter out replay data tagged for a different
/// provider.
const PROVIDER_TAG: &str = "openai_responses";
/// ChatGPT subscription route: reachable by ChatGPT Plus/Pro OAuth access tokens,
/// not by plain API keys. Serves the same Responses API surface at `/responses`.
pub const CHATGPT_CODEX_BASE: &str = "https://chatgpt.com/backend-api/codex";

/// `client_version` query param advertised on the Codex-route `/models` call.
/// The ChatGPT backend gates visibility of newer models behind a minimum Codex
/// client version (e.g. gpt-5.4 requires >= 0.98.0) so we lie in this
/// direction deliberately — we aren't the Codex CLI, but we implement the
/// same wire contract, and anything below the highest observed gate hides
/// frontier models from the picker.
const CODEX_CLIENT_VERSION: &str = "99.0.0";

/// Runtime auth material for a Responses-API client. The client itself is
/// cheap and shared; [`ClientAuth`] is what differs between an API-key session
/// and a ChatGPT-subscription session.
pub enum ClientAuth {
    ApiKey(String),
    /// ChatGPT OAuth tokens, typically loaded from `~/.codex/auth.json`.
    /// Wrapped in a mutex because refresh mutates the tokens, and the client
    /// may be called from multiple scheduler tasks concurrently.
    Codex(Arc<Mutex<CodexAuth>>),
}

pub struct OpenAiResponsesClient {
    http: reqwest::Client,
    base_url: String,
    auth: ClientAuth,
}

impl OpenAiResponsesClient {
    /// API-key flow — Bearer the key against `base_url` (typically
    /// `https://api.openai.com/v1`).
    pub fn with_api_key(base_url: String, api_key: String) -> Self {
        Self::new(base_url, ClientAuth::ApiKey(api_key))
    }

    /// ChatGPT-subscription flow — Bearer the access_token from
    /// `auth` plus a `chatgpt-account-id` header, against `base_url`
    /// (typically [`CHATGPT_CODEX_BASE`]).
    pub fn with_codex_auth(base_url: String, auth: CodexAuth) -> Self {
        Self::new(base_url, ClientAuth::Codex(Arc::new(Mutex::new(auth))))
    }

    fn new(base_url: String, auth: ClientAuth) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            http: reqwest::Client::new(),
            base_url,
            auth,
        }
    }

    /// Returns Bearer token + any extra headers (e.g. chatgpt-account-id)
    /// the current auth mode requires. For Codex auth, refreshes the token
    /// if it's near expiry before returning.
    async fn prepare_headers(&self) -> Result<(String, Vec<(&'static str, String)>), ModelError> {
        match &self.auth {
            ClientAuth::ApiKey(key) => Ok((key.clone(), Vec::new())),
            ClientAuth::Codex(m) => {
                let mut guard = m.lock().await;
                guard
                    .ensure_fresh(&self.http)
                    .await
                    .map_err(|e| ModelError::Transport(format!("codex auth refresh: {e}")))?;
                let mut extras = Vec::new();
                if let Some(acc) = guard.chatgpt_account_id() {
                    extras.push(("chatgpt-account-id", acc.to_string()));
                }
                Ok((guard.access_token().to_string(), extras))
            }
        }
    }

    /// Build + send the Responses request, returning the live HTTP response
    /// for the caller to consume either as a buffered body or an SSE byte
    /// stream. Consolidates body construction, header/auth setup, and the
    /// error-status early return so both the streaming and buffered paths
    /// share one code path up through "first byte".
    async fn send_request(&self, req: &ModelRequest<'_>) -> Result<reqwest::Response, ModelError> {
        let mut input: Vec<RspItem> = Vec::new();
        for m in req.messages {
            convert_message(m, &mut input);
        }

        let tools: Vec<RspTool> = req.tools.iter().map(spec_to_rsp_tool).collect();

        let body = RspRequest {
            model: req.model,
            instructions: if req.system_prompt.is_empty() {
                None
            } else {
                Some(req.system_prompt)
            },
            input,
            tools: if tools.is_empty() { None } else { Some(tools) },
            include: REQUEST_INCLUDES,
            store: false,
            // The ChatGPT-subscription backend refuses stream:false with
            // "Stream must be set to true". api.openai.com accepts either; we
            // always stream and reassemble so both routes share one code path.
            stream: true,
        };
        // req.max_tokens is ignored: the ChatGPT-subscription route rejects
        // `max_output_tokens` outright, and Codex's own client doesn't send it
        // on either route. The backend picks an output cap per model; the
        // scheduler's `max_turns` still bounds the overall loop.
        let _ = req.max_tokens;

        let url = format!("{}/responses", self.base_url);
        let (bearer, extra_headers) = self.prepare_headers().await?;
        let mut builder = self
            .http
            .post(&url)
            .bearer_auth(&bearer)
            .header("accept", "text/event-stream")
            .json(&body);
        for (k, v) in &extra_headers {
            builder = builder.header(*k, v);
        }
        let resp = builder
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
        Ok(resp)
    }

    async fn do_create_message(&self, req: &ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        let resp = self.send_request(req).await?;
        // Buffer the full SSE stream and dig out the `response.completed` event
        // — its `response` field has the status / usage / incomplete_details
        // bits we still need even when using the streaming assembly.
        let raw = resp
            .text()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        let parsed: RspResponse = extract_completed_response(&raw)?;
        let (content, stop_reason, usage) = finalize_response(
            parsed.output,
            parsed.status,
            parsed.incomplete_details,
            parsed.usage,
        )?;
        Ok(ModelResponse {
            content,
            stop_reason,
            usage,
        })
    }

    fn do_create_message_streaming<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        Box::pin(try_stream! {
            let resp = self.send_request(req).await?;
            let mut bytes = resp.bytes_stream();
            let mut sse_buf: Vec<u8> = Vec::new();
            let mut state = RspStreamState::default();
            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|e| ModelError::Transport(e.to_string()))?;
                sse_buf.extend_from_slice(&chunk);
                while let Some(event_payload) = take_sse_event(&mut sse_buf) {
                    let Some(raw) = parse_sse_data(&event_payload) else {
                        continue;
                    };
                    if raw == "[DONE]" {
                        continue;
                    }
                    let ev: RspStreamEvent = match serde_json::from_str(&raw) {
                        Ok(ev) => ev,
                        // Unknown event shape — skip rather than fail. New
                        // event types land in the Responses API from time to
                        // time; we don't want to hard-error on them.
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
            Err(ModelError::Transport(
                "SSE stream ended without response.completed".into(),
            ))?;
        })
    }

    async fn do_list_models(&self) -> Result<Vec<ModelInfo>, ModelError> {
        // Both routes expose `/models`, but the response shape differs:
        //   - api.openai.com/v1/models → `{ data: [{ id }] }`       (OpenAI standard)
        //   - chatgpt.com/backend-api/codex/models → `{ models: [...] }` with
        //     per-model `visibility`, `supported_in_api`, `display_name`, etc.
        // The Codex endpoint also requires a `client_version` query param so the
        // backend can gate version-locked models.
        let codex_mode = matches!(self.auth, ClientAuth::Codex(_));
        let url = if codex_mode {
            format!(
                "{}/models?client_version={}",
                self.base_url, CODEX_CLIENT_VERSION
            )
        } else {
            format!("{}/models", self.base_url)
        };

        let (bearer, extra_headers) = self.prepare_headers().await?;
        let mut builder = self.http.get(&url).bearer_auth(&bearer);
        for (k, v) in &extra_headers {
            builder = builder.header(*k, v);
        }
        let resp = builder
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

        if codex_mode {
            let parsed: RspCodexModelsResponse = resp
                .json()
                .await
                .map_err(|e| ModelError::Transport(e.to_string()))?;
            Ok(parsed
                .models
                .into_iter()
                // Mirror Codex's own picker filter: only models the backend
                // marks as user-visible AND usable via the Responses API.
                .filter(|m| m.supported_in_api && m.visibility.as_deref() == Some("list"))
                .map(|m| ModelInfo {
                    id: m.slug,
                    display_name: m.display_name,
                })
                .collect())
        } else {
            let parsed: RspListModelsResponse = resp
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
}

impl ModelProvider for OpenAiResponsesClient {
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

// ---------- Message → Item conversion ----------

fn convert_message(m: &Message, out: &mut Vec<RspItem>) {
    match m.role {
        // Responses wants tool results as standalone
        // `function_call_output` items, which `convert_user_message`
        // already emits per-block. A `Role::ToolResult` message with
        // only tool_result blocks routes through the same path and
        // produces only function_call_output items — exactly what
        // Responses expects.
        Role::User | Role::ToolResult => convert_user_message(&m.content, out),
        Role::Assistant => convert_assistant_message(&m.content, out),
    }
}

/// A user turn is a mix of plain text and tool results. Responses wants tool
/// results as standalone `function_call_output` items, so we emit those
/// inline and flush accumulated text as a `message` item at the boundary.
fn convert_user_message(blocks: &[ContentBlock], out: &mut Vec<RspItem>) {
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
                flush_user_text(&mut text_accum, out);
                out.push(RspItem::FunctionCallOutput {
                    call_id: tool_use_id.clone(),
                    output: tool_result_as_text(content, *is_error),
                });
            }
            ContentBlock::ToolUse { .. } | ContentBlock::Thinking { .. } => {
                // Neither belongs on a user message. Drop silently.
            }
        }
    }
    flush_user_text(&mut text_accum, out);
}

fn flush_user_text(text_accum: &mut String, out: &mut Vec<RspItem>) {
    if text_accum.is_empty() {
        return;
    }
    out.push(RspItem::Message {
        role: "user",
        content: vec![RspInputMessagePart::InputText {
            text: std::mem::take(text_accum),
        }],
    });
}

fn convert_assistant_message(blocks: &[ContentBlock], out: &mut Vec<RspItem>) {
    let mut text_accum = String::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                if !text_accum.is_empty() {
                    text_accum.push('\n');
                }
                text_accum.push_str(text);
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                if !text_accum.is_empty() {
                    out.push(RspItem::Message {
                        role: "assistant",
                        content: vec![RspInputMessagePart::OutputText {
                            text: std::mem::take(&mut text_accum),
                        }],
                    });
                }
                out.push(RspItem::FunctionCall {
                    call_id: id.clone(),
                    name: name.clone(),
                    arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                });
            }
            ContentBlock::Thinking {
                thinking: _,
                replay,
            } => {
                // Reasoning replay: if the block was minted by *this*
                // backend, echo the reasoning item back into `input` so
                // the server resumes its chain-of-thought. A block tagged
                // for a different provider is dropped — its opaque blob
                // is meaningless here (and potentially error-inducing).
                // Without replay data there's nothing the server can do
                // with our textual summary alone, so dropping is also the
                // right move.
                if let Some(item) = thinking_to_reasoning_item(replay.as_ref()) {
                    if !text_accum.is_empty() {
                        out.push(RspItem::Message {
                            role: "assistant",
                            content: vec![RspInputMessagePart::OutputText {
                                text: std::mem::take(&mut text_accum),
                            }],
                        });
                    }
                    out.push(item);
                }
            }
            ContentBlock::ToolResult { .. } => {
                // ToolResult doesn't belong on an assistant message.
            }
        }
    }
    if !text_accum.is_empty() {
        out.push(RspItem::Message {
            role: "assistant",
            content: vec![RspInputMessagePart::OutputText { text: text_accum }],
        });
    }
}

/// Build an outbound `reasoning` input item from a stored [`ProviderReplay`].
/// Returns `None` when the blob was minted by a different provider or lacks
/// any replay-relevant fields — in either case there's nothing to echo.
fn thinking_to_reasoning_item(replay: Option<&ProviderReplay>) -> Option<RspItem> {
    let r = replay?;
    if r.provider != PROVIDER_TAG {
        return None;
    }
    let obj = r.data.as_object()?;
    let id = obj.get("id").and_then(|v| v.as_str()).map(String::from);
    let encrypted_content = obj
        .get("encrypted_content")
        .and_then(|v| v.as_str())
        .map(String::from);
    let summary = obj
        .get("summary")
        .and_then(|v| v.as_array())
        .cloned()
        .map(Value::Array)
        .and_then(|v| serde_json::from_value::<Vec<RspReasoningPart>>(v).ok())
        .unwrap_or_default();
    if id.is_none() && encrypted_content.is_none() && summary.is_empty() {
        return None;
    }
    Some(RspItem::Reasoning {
        id,
        encrypted_content,
        summary,
    })
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

fn spec_to_rsp_tool(t: &ToolSpec) -> RspTool {
    RspTool::Function {
        name: t.name.clone(),
        description: t.description.clone(),
        parameters: t.input_schema.clone(),
        strict: false,
    }
}

/// Reassemble the final response object from the SSE stream body.
///
/// Items arrive incrementally as `response.output_item.done` events — each
/// carrying one fully-formed message / function_call / reasoning item. The
/// final `response.completed` event supplies the status, usage, and
/// incomplete_details, but its embedded `response.output` array is often
/// empty on the ChatGPT-subscription route (the server relies on the client
/// having accumulated the items already). We merge both: completed-event
/// metadata + accumulated items as the output.
///
/// `response.failed` short-circuits to [`ModelError::Api`] so terminal
/// server-side errors surface cleanly to the caller.
fn extract_completed_response(body: &str) -> Result<RspResponse, ModelError> {
    let mut items: Vec<RspOutputItem> = Vec::new();
    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim_start();
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let event: SseEvent = match serde_json::from_str(payload) {
            Ok(ev) => ev,
            Err(_) => continue, // non-event data lines are skippable
        };
        match event {
            SseEvent::OutputItemDone { item } => items.push(item),
            SseEvent::ResponseCompleted { mut response } => {
                // If the server included the items inline (api.openai.com does),
                // trust that — otherwise reuse what we accumulated.
                if response.output.is_empty() {
                    response.output = items;
                }
                return Ok(response);
            }
            SseEvent::ResponseFailed { response, error } => {
                let detail = error
                    .as_ref()
                    .map(|e| e.message.clone())
                    .unwrap_or_else(|| "response.failed".to_string());
                let status = response
                    .as_ref()
                    .and_then(|r| r.status.clone())
                    .unwrap_or_else(|| "failed".to_string());
                return Err(ModelError::Api {
                    status: 500,
                    body: format!("{status}: {detail}"),
                });
            }
            SseEvent::Other => {}
        }
    }
    Err(ModelError::Transport(
        "SSE stream ended without response.completed".into(),
    ))
}

/// Collapse Responses' `status` + `incomplete_details.reason` + whether the
/// output included a function_call into the Anthropic-style stop_reason labels
/// the rest of the stack consumes.
fn derive_stop_reason(
    status: Option<&str>,
    incomplete: Option<&RspIncompleteDetails>,
    saw_function_call: bool,
) -> Option<String> {
    if saw_function_call {
        return Some("tool_use".into());
    }
    match status {
        Some("completed") => Some("end_turn".into()),
        Some("incomplete") => match incomplete.and_then(|d| d.reason.as_deref()) {
            Some("max_output_tokens") => Some("max_tokens".into()),
            Some(other) => Some(other.to_string()),
            None => Some("incomplete".into()),
        },
        Some(other) => Some(other.to_string()),
        None => None,
    }
}

// ---------- Streaming SSE parse + state machine ----------

/// Pull one complete SSE event (terminated by a blank line) out of `buf`.
/// Returns `None` until enough bytes are buffered. Mirrors the Anthropic
/// adapter's helper — same SSE framing, so we could consolidate later, but
/// each adapter has its own rules for how to interpret the data payload.
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

/// Responses SSE events we actually consume. Delta events drive live
/// broadcasting; `output_item.done` accumulates fully-formed items for the
/// final `Completed`; `response.completed` closes the stream with usage +
/// status. Unknown events collapse to `Other` so new server-side event
/// types don't break the stream.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspStreamEvent {
    /// Live text fragment. Emitted repeatedly during a message block.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        #[serde(default)]
        delta: String,
    },
    /// Live reasoning-summary fragment (the default visible prose on
    /// reasoning models).
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        #[serde(default)]
        delta: String,
    },
    /// Live reasoning-text fragment (older path; some model variants emit
    /// this instead of the summary track).
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        #[serde(default)]
        delta: String,
    },
    /// A fully-formed output item — message, function_call, reasoning.
    /// Accumulated into the final Completed's `content`.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone { item: RspOutputItem },
    /// Terminal success. Carries `usage` and `status`. The embedded
    /// `response.output` array may be empty (ChatGPT-subscription path),
    /// in which case we rely on accumulated `output_item.done`s.
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: RspResponse },
    /// Terminal failure.
    #[serde(rename = "response.failed")]
    ResponseFailed {
        #[serde(default)]
        response: Option<RspResponse>,
        #[serde(default)]
        error: Option<SseError>,
    },
    #[serde(other)]
    Other,
}

/// Running state for one Responses streaming call — accumulates full
/// content blocks for the terminal `Completed` event while delta events
/// fly through to the consumer live.
#[derive(Default)]
struct RspStreamState {
    items: Vec<RspOutputItem>,
    done: bool,
}

impl RspStreamState {
    fn consume(&mut self, event: RspStreamEvent) -> Result<Vec<ModelEvent>, ModelError> {
        let mut out = Vec::new();
        match event {
            RspStreamEvent::OutputTextDelta { delta } => {
                if !delta.is_empty() {
                    out.push(ModelEvent::TextDelta { text: delta });
                }
            }
            RspStreamEvent::ReasoningSummaryTextDelta { delta }
            | RspStreamEvent::ReasoningTextDelta { delta } => {
                if !delta.is_empty() {
                    out.push(ModelEvent::ThinkingDelta { text: delta });
                }
            }
            RspStreamEvent::OutputItemDone { item } => {
                // Emit a ToolCall event for function_call items so the
                // downstream consumer (the scheduler) has symmetric
                // coverage with the Anthropic adapter. Message /
                // reasoning items already streamed as deltas.
                if let RspOutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                } = &item
                {
                    let input: Value = if arguments.is_empty() {
                        Value::Object(Default::default())
                    } else {
                        serde_json::from_str(arguments).map_err(|e| {
                            ModelError::Transport(format!(
                                "function_call arguments not valid JSON: {e}"
                            ))
                        })?
                    };
                    out.push(ModelEvent::ToolCall {
                        id: call_id.clone(),
                        name: name.clone(),
                        input,
                    });
                }
                self.items.push(item);
            }
            RspStreamEvent::ResponseCompleted { response } => {
                // The server sometimes inlines the full output here (api.openai.com)
                // and sometimes sends it piecemeal via output_item.done
                // (ChatGPT-subscription). Prefer the inline list when present.
                let items = if response.output.is_empty() {
                    std::mem::take(&mut self.items)
                } else {
                    response.output
                };
                let (content, stop_reason, usage) = finalize_response(
                    items,
                    response.status,
                    response.incomplete_details,
                    response.usage,
                )?;
                out.push(ModelEvent::Completed {
                    content,
                    stop_reason,
                    usage,
                });
                self.done = true;
            }
            RspStreamEvent::ResponseFailed { response, error } => {
                let detail = error
                    .as_ref()
                    .map(|e| e.message.clone())
                    .unwrap_or_else(|| "response.failed".to_string());
                let status = response
                    .as_ref()
                    .and_then(|r| r.status.clone())
                    .unwrap_or_else(|| "failed".to_string());
                return Err(ModelError::Api {
                    status: 500,
                    body: format!("{status}: {detail}"),
                });
            }
            RspStreamEvent::Other => {}
        }
        Ok(out)
    }
}

/// Translate a finished item list + terminal metadata into the normalized
/// content blocks + stop_reason + usage that `ModelEvent::Completed` and
/// `do_create_message` both want.
fn finalize_response(
    items: Vec<RspOutputItem>,
    status: Option<String>,
    incomplete_details: Option<RspIncompleteDetails>,
    usage: Option<RspUsage>,
) -> Result<(Vec<ContentBlock>, Option<String>, Usage), ModelError> {
    let mut content: Vec<ContentBlock> = Vec::new();
    let mut saw_function_call = false;
    for item in items {
        match item {
            RspOutputItem::Reasoning {
                id,
                encrypted_content,
                summary,
                content: inline_content,
            } => {
                let text = summary
                    .iter()
                    .chain(inline_content.iter())
                    .map(|p| match p {
                        RspReasoningPart::ReasoningText { text }
                        | RspReasoningPart::SummaryText { text } => text.as_str(),
                    })
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                let mut replay_data = serde_json::Map::new();
                if let Some(id) = &id {
                    replay_data.insert("id".into(), Value::String(id.clone()));
                }
                if let Some(ec) = &encrypted_content {
                    replay_data.insert("encrypted_content".into(), Value::String(ec.clone()));
                }
                if !summary.is_empty() {
                    replay_data.insert(
                        "summary".into(),
                        serde_json::to_value(&summary).unwrap_or(Value::Null),
                    );
                }
                let replay = (!replay_data.is_empty()).then(|| ProviderReplay {
                    provider: PROVIDER_TAG.into(),
                    data: Value::Object(replay_data),
                });
                if replay.is_some() || !text.trim().is_empty() {
                    content.push(ContentBlock::Thinking {
                        replay,
                        thinking: text,
                    });
                }
            }
            RspOutputItem::Message { content: parts, .. } => {
                for part in parts {
                    match part {
                        RspOutputMessagePart::OutputText { text } => {
                            if !text.is_empty() {
                                content.push(ContentBlock::Text { text });
                            }
                        }
                        RspOutputMessagePart::Refusal { refusal } => {
                            content.push(ContentBlock::Text {
                                text: format!("[refusal] {refusal}"),
                            });
                        }
                    }
                }
            }
            RspOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                saw_function_call = true;
                let input_val: Value = if arguments.is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&arguments).map_err(|e| {
                        ModelError::Transport(format!(
                            "function_call arguments not valid JSON: {e}"
                        ))
                    })?
                };
                content.push(ContentBlock::ToolUse {
                    id: call_id,
                    name,
                    input: input_val,
                    replay: None,
                });
            }
            RspOutputItem::Other => {}
        }
    }
    let stop_reason = derive_stop_reason(
        status.as_deref(),
        incomplete_details.as_ref(),
        saw_function_call,
    );
    let usage = usage
        .map(|u| Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_input_tokens: u
                .input_tokens_details
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0),
            cache_creation_input_tokens: 0,
        })
        .unwrap_or_default();
    Ok((content, stop_reason, usage))
}

// ---------- Wire types (private to this module) ----------

#[derive(Serialize, Debug)]
struct RspRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
    input: Vec<RspItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<RspTool>>,
    /// Extra fields to request back from the server. We set
    /// `reasoning.encrypted_content` so reasoning items come back with the
    /// opaque blob we need to echo them on subsequent turns.
    #[serde(skip_serializing_if = "<[&str]>::is_empty")]
    include: &'static [&'static str],
    store: bool,
    stream: bool,
}

/// Wire value for [`RspRequest::include`]. Constant because we always want the
/// same set — the overhead of echoing the blob back is negligible vs. the
/// alternative of re-reasoning from scratch every turn.
const REQUEST_INCLUDES: &[&str] = &["reasoning.encrypted_content"];

/// One element of the `input` array. Outbound only — serialization shape must
/// exactly match what the API expects.
#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspItem {
    Message {
        role: &'static str,
        content: Vec<RspInputMessagePart>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        /// JSON-encoded argument object (string, not a parsed Value — the API
        /// expects the stringified form).
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
    /// Replay item for chain-of-thought continuity. Echoed into `input` on
    /// turns that follow a reasoning-model response so the server can
    /// resume its internal reasoning state.
    Reasoning {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        summary: Vec<RspReasoningPart>,
    },
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspInputMessagePart {
    InputText { text: String },
    OutputText { text: String },
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspTool {
    Function {
        name: String,
        description: String,
        parameters: Value,
        strict: bool,
    },
}

/// SSE frames we care about on the streaming `/responses` route.
///
/// `output_item.done` carries one fully-formed item (message, function_call,
/// reasoning, …) — we accumulate those. `response.completed` carries the
/// terminal status + usage; `response.failed` carries a terminal error. Every
/// other event (`.delta`, `.in_progress`, `.created`, …) is useful for live
/// UI streaming but we don't expose that at the provider trait, so they
/// collapse into `Other` and get discarded.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SseEvent {
    #[serde(rename = "response.output_item.done")]
    OutputItemDone { item: RspOutputItem },
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: RspResponse },
    #[serde(rename = "response.failed")]
    ResponseFailed {
        #[serde(default)]
        response: Option<RspResponse>,
        #[serde(default)]
        error: Option<SseError>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Debug)]
struct SseError {
    #[serde(default)]
    message: String,
}

#[derive(Deserialize, Debug)]
struct RspResponse {
    #[serde(default)]
    output: Vec<RspOutputItem>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    incomplete_details: Option<RspIncompleteDetails>,
    #[serde(default)]
    usage: Option<RspUsage>,
}

#[derive(Deserialize, Debug)]
struct RspIncompleteDetails {
    #[serde(default)]
    reason: Option<String>,
}

/// Items returned in `response.output`. The API emits other types we don't yet
/// consume (web_search_call, code_interpreter_call, computer_call, …); they
/// collapse into the `Other` variant and are ignored.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspOutputItem {
    Message {
        #[serde(default)]
        content: Vec<RspOutputMessagePart>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        encrypted_content: Option<String>,
        /// Returned by default. Short gist the API surfaces to clients.
        #[serde(default)]
        summary: Vec<RspReasoningPart>,
        /// Legacy / model-variant path for reasoning prose. Older turns
        /// surface text here instead of `summary`; both can be present.
        #[serde(default)]
        content: Vec<RspReasoningPart>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspOutputMessagePart {
    OutputText { text: String },
    Refusal { refusal: String },
}

/// Reasoning prose carrier. Appears inbound as [`RspOutputItem::Reasoning`]'s
/// `summary` / `content` fields and outbound inside [`RspItem::Reasoning`]'s
/// `summary` field when we echo a stored reasoning item back into `input`.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspReasoningPart {
    ReasoningText { text: String },
    SummaryText { text: String },
}

#[derive(Deserialize, Debug)]
struct RspUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    input_tokens_details: Option<RspInputTokensDetails>,
}

#[derive(Deserialize, Debug)]
struct RspInputTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct RspListModelsResponse {
    data: Vec<RspModel>,
}

#[derive(Deserialize, Debug)]
struct RspModel {
    id: String,
}

/// `GET chatgpt.com/backend-api/codex/models` shape. Carries a much richer per-model
/// metadata block than api.openai.com — we grab just enough for the picker.
#[derive(Deserialize, Debug)]
struct RspCodexModelsResponse {
    #[serde(default)]
    models: Vec<RspCodexModel>,
}

#[derive(Deserialize, Debug)]
struct RspCodexModel {
    slug: String,
    #[serde(default)]
    display_name: Option<String>,
    /// One of "list" / "hide" / (possibly others). Only `"list"` models are
    /// user-facing; hidden entries are legacy or internal.
    #[serde(default)]
    visibility: Option<String>,
    /// `false` on entries the backend exposes for telemetry but doesn't accept
    /// create-message calls for.
    #[serde(default)]
    supported_in_api: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::Message;

    #[test]
    fn serializes_simple_user_message() {
        let mut items = Vec::new();
        convert_message(&Message::user_text("hello"), &mut items);
        let json = serde_json::to_value(&items).unwrap();
        assert_eq!(
            json,
            serde_json::json!([
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
            ])
        );
    }

    #[test]
    fn serializes_assistant_tool_use_plus_text() {
        let msg = Message::assistant_blocks(vec![
            ContentBlock::Text { text: "ok".into() },
            ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "shell".into(),
                input: serde_json::json!({"cmd": "ls"}),
                replay: None,
            },
        ]);
        let mut items = Vec::new();
        convert_message(&msg, &mut items);
        let json = serde_json::to_value(&items).unwrap();
        assert_eq!(
            json,
            serde_json::json!([
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
                {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
            ])
        );
    }

    #[test]
    fn serializes_tool_result_on_user_turn() {
        let msg = Message::user_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "call_1".into(),
            content: ToolResultContent::Text("output".into()),
            is_error: false,
        }]);
        let mut items = Vec::new();
        convert_message(&msg, &mut items);
        let json = serde_json::to_value(&items).unwrap();
        assert_eq!(
            json,
            serde_json::json!([
                {"type": "function_call_output", "call_id": "call_1", "output": "output"}
            ])
        );
    }

    #[test]
    fn parses_response_with_function_call() {
        let body = r#"{
            "status": "completed",
            "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "think"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
                {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp: RspResponse = serde_json::from_str(body).unwrap();
        let saw_fc = resp
            .output
            .iter()
            .any(|i| matches!(i, RspOutputItem::FunctionCall { .. }));
        assert!(saw_fc);
        assert_eq!(
            derive_stop_reason(
                resp.status.as_deref(),
                resp.incomplete_details.as_ref(),
                saw_fc,
            ),
            Some("tool_use".into())
        );
    }

    #[test]
    fn extracts_final_response_from_sse_stream_with_inline_output() {
        // api.openai.com inlines the full output array in response.completed.
        let stream = r#"event: response.created
data: {"type":"response.created","response":{"id":"r_1","status":"in_progress","output":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"hi"}

event: response.completed
data: {"type":"response.completed","response":{"id":"r_1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":3,"output_tokens":1}}}

"#;
        let r = extract_completed_response(stream).unwrap();
        assert_eq!(r.status.as_deref(), Some("completed"));
        assert_eq!(r.usage.as_ref().unwrap().output_tokens, 1);
        assert_eq!(r.output.len(), 1);
    }

    #[test]
    fn reconstructs_output_from_output_item_done_events() {
        // ChatGPT-subscription path: response.completed carries an empty
        // output array; the items arrive on response.output_item.done.
        let stream = r#"event: response.created
data: {"type":"response.created","response":{"id":"r_1"}}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"c1","name":"shell","arguments":"{\"cmd\":\"ls\"}"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"r_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}

"#;
        let r = extract_completed_response(stream).unwrap();
        assert_eq!(r.output.len(), 2);
        assert!(matches!(&r.output[0], RspOutputItem::Message { .. }));
        assert!(matches!(&r.output[1], RspOutputItem::FunctionCall { .. }));
    }

    #[test]
    fn sse_stream_without_completed_errors() {
        let stream = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n";
        let err = extract_completed_response(stream).unwrap_err();
        assert!(matches!(err, ModelError::Transport(_)));
    }

    #[test]
    fn sse_response_failed_surfaces_as_api_error() {
        let stream = r#"event: response.failed
data: {"type":"response.failed","response":{"status":"failed"},"error":{"message":"rate limited"}}

"#;
        let err = extract_completed_response(stream).unwrap_err();
        match err {
            ModelError::Api { status, body } => {
                assert_eq!(status, 500);
                assert!(body.contains("rate limited"), "body was: {body}");
            }
            _ => panic!("expected Api error"),
        }
    }

    #[test]
    fn codex_models_response_filters_to_user_visible_api_supported() {
        let body = r#"{
            "models": [
                {"slug": "gpt-5.4",             "display_name": "GPT-5.4",        "visibility": "list", "supported_in_api": true},
                {"slug": "gpt-5.3-codex",       "display_name": "GPT-5.3 Codex",  "visibility": "list", "supported_in_api": true},
                {"slug": "gpt-5",               "display_name": "GPT-5 (legacy)", "visibility": "hide", "supported_in_api": true},
                {"slug": "internal-eval-only",  "display_name": null,             "visibility": "list", "supported_in_api": false}
            ]
        }"#;
        let parsed: RspCodexModelsResponse = serde_json::from_str(body).unwrap();
        let kept: Vec<String> = parsed
            .models
            .into_iter()
            .filter(|m| m.supported_in_api && m.visibility.as_deref() == Some("list"))
            .map(|m| m.slug)
            .collect();
        assert_eq!(
            kept,
            vec!["gpt-5.4".to_string(), "gpt-5.3-codex".to_string()]
        );
    }

    #[test]
    fn stop_reason_max_tokens() {
        let body = r#"{
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }"#;
        let resp: RspResponse = serde_json::from_str(body).unwrap();
        assert_eq!(
            derive_stop_reason(
                resp.status.as_deref(),
                resp.incomplete_details.as_ref(),
                false
            ),
            Some("max_tokens".into())
        );
    }

    // ---------- Reasoning replay ----------

    #[test]
    fn request_body_includes_encrypted_content_flag() {
        // `include: ["reasoning.encrypted_content"]` is what tells the server
        // to return the opaque blob we need to echo back on later turns.
        // Verify it's always present on the wire body.
        let body = RspRequest {
            model: "gpt-5",
            instructions: None,
            input: Vec::new(),
            tools: None,
            include: REQUEST_INCLUDES,
            store: false,
            stream: true,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(
            v["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn thinking_to_reasoning_item_only_echoes_matching_provider() {
        // Foreign-provider replay → dropped (can't ship an Anthropic blob
        // to OpenAI).
        let foreign = ProviderReplay {
            provider: "anthropic".into(),
            data: serde_json::json!({"signature": "sig"}),
        };
        assert!(thinking_to_reasoning_item(Some(&foreign)).is_none());

        // Matching provider → emitted as Reasoning item carrying id +
        // encrypted_content (matches what the server originally sent).
        let native = ProviderReplay {
            provider: "openai_responses".into(),
            data: serde_json::json!({
                "id": "rs_1",
                "encrypted_content": "ec-blob",
                "summary": [{"type": "summary_text", "text": "short gist"}]
            }),
        };
        let item = thinking_to_reasoning_item(Some(&native)).expect("should echo");
        let v = serde_json::to_value(&item).unwrap();
        assert_eq!(v["type"], "reasoning");
        assert_eq!(v["id"], "rs_1");
        assert_eq!(v["encrypted_content"], "ec-blob");
        assert_eq!(v["summary"][0]["type"], "summary_text");
    }

    #[test]
    fn thinking_to_reasoning_item_no_data_returns_none() {
        // Replay with empty data object has nothing to echo — drop.
        let native = ProviderReplay {
            provider: "openai_responses".into(),
            data: serde_json::json!({}),
        };
        assert!(thinking_to_reasoning_item(Some(&native)).is_none());
    }

    #[test]
    fn parses_reasoning_item_and_preserves_replay() {
        // Inbound: a reasoning item with id + encrypted_content + summary
        // parses into a Thinking block whose replay carries all three so
        // the next turn can echo the full item back.
        let wire = serde_json::json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "ec-blob",
            "summary": [{"type": "summary_text", "text": "hi"}]
        });
        let item: RspOutputItem = serde_json::from_value(wire).unwrap();
        match item {
            RspOutputItem::Reasoning {
                id,
                encrypted_content,
                summary,
                ..
            } => {
                assert_eq!(id.as_deref(), Some("rs_1"));
                assert_eq!(encrypted_content.as_deref(), Some("ec-blob"));
                assert_eq!(summary.len(), 1);
            }
            _ => panic!("expected Reasoning"),
        }
    }

    // ---------- Streaming state machine ----------

    fn feed_events(events: &[&str]) -> (Vec<ModelEvent>, RspStreamState) {
        let mut state = RspStreamState::default();
        let mut out = Vec::new();
        for ev_json in events {
            let ev: RspStreamEvent = serde_json::from_str(ev_json).unwrap();
            out.extend(state.consume(ev).unwrap());
        }
        (out, state)
    }

    #[test]
    fn output_text_delta_emits_textdelta() {
        let events = [
            r#"{"type":"response.output_text.delta","delta":"hello "}"#,
            r#"{"type":"response.output_text.delta","delta":"world"}"#,
            r#"{"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"hello world"}]}}"#,
            r#"{"type":"response.completed","response":{"status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":2}}}"#,
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
                assert_eq!(usage.input_tokens, 3);
                assert_eq!(usage.output_tokens, 2);
            }
            _ => panic!("last event must be Completed"),
        }
    }

    #[test]
    fn reasoning_summary_delta_emits_thinkingdelta_and_replay_survives() {
        let events = [
            r#"{"type":"response.reasoning_summary_text.delta","delta":"planning"}"#,
            r#"{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","encrypted_content":"ec","summary":[{"type":"summary_text","text":"planning"}]}}"#,
            r#"{"type":"response.completed","response":{"status":"completed","output":[]}}"#,
        ];
        let (evs, _) = feed_events(&events);
        assert!(matches!(&evs[0], ModelEvent::ThinkingDelta { text } if text == "planning"));
        match evs.last().unwrap() {
            ModelEvent::Completed { content, .. } => match &content[0] {
                ContentBlock::Thinking { replay, thinking } => {
                    assert_eq!(thinking, "planning");
                    let r = replay.as_ref().expect("reasoning must carry replay");
                    assert_eq!(r.provider, "openai_responses");
                    assert_eq!(r.data["encrypted_content"], "ec");
                    assert_eq!(r.data["id"], "rs_1");
                }
                _ => panic!("expected Thinking"),
            },
            _ => panic!("expected Completed"),
        }
    }

    #[test]
    fn function_call_item_emits_toolcall_once_assembled() {
        // OpenAI's function_call item arrives fully-formed as
        // output_item.done — no partial-JSON streaming exposed.
        let events = [
            r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"c1","name":"shell","arguments":"{\"cmd\":\"ls\"}"}}"#,
            r#"{"type":"response.completed","response":{"status":"completed","output":[]}}"#,
        ];
        let (evs, _) = feed_events(&events);
        let tool_calls: Vec<_> = evs
            .iter()
            .filter_map(|e| match e {
                ModelEvent::ToolCall { name, input, .. } => Some((name.as_str(), input.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].0, "shell");
        assert_eq!(tool_calls[0].1, serde_json::json!({"cmd": "ls"}));
        match evs.last().unwrap() {
            ModelEvent::Completed { stop_reason, .. } => {
                assert_eq!(stop_reason.as_deref(), Some("tool_use"));
            }
            _ => panic!("expected Completed"),
        }
    }
}
