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

use std::collections::HashMap;
use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use whisper_agent_auth::{ClientAuth, CodexAuth, CodexAuthSlot, SlotUpdateError};
use whisper_agent_protocol::{
    ContentBlock, Message, ProviderReplay, Role, ToolResultContent, TunableEnumVariant,
    TunableKind, TunableSpec, TunableValue, Usage,
};

use crate::providers::model::{
    BoxFuture, BoxStream, ForensicBuf, ForensicContext, ModelError, ModelEvent, ModelInfo,
    ModelProvider, ModelRequest, ModelResponse, ProviderErrorDetail, ToolSpec, UpdateAuthError,
};

pub const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// Identifier stored on [`ProviderReplay::provider`] for blobs minted by this
/// backend. Used on outbound to filter out replay data tagged for a different
/// provider.
const PROVIDER_TAG: &str = "openai_responses";
/// ChatGPT subscription route: reachable by ChatGPT Plus/Pro OAuth access tokens,
/// not by plain API keys. Serves the same Responses API surface at `/responses`.
pub const CHATGPT_CODEX_BASE: &str = "https://chatgpt.com/backend-api/codex";

/// ChatGPT subscription usage / quota endpoint. Sibling to
/// [`CHATGPT_CODEX_BASE`] — note it lives under `/backend-api/wham/`,
/// *not* under `/backend-api/codex/`. Reuses the same OAuth access
/// token and `chatgpt-account-id` header as the `/responses` route.
/// Hardcoded rather than derived from the configured `base_url`
/// because any non-default `base_url` (test fakegateways, etc.) would
/// not have a matching `wham` surface anyway.
pub const CHATGPT_WHAM_USAGE_URL: &str = "https://chatgpt.com/backend-api/wham/usage";

/// `client_version` query param advertised on the Codex-route `/models` call.
/// The ChatGPT backend gates visibility of newer models behind a minimum Codex
/// client version (e.g. gpt-5.4 requires >= 0.98.0) so we lie in this
/// direction deliberately — we aren't the Codex CLI, but we implement the
/// same wire contract, and anything below the highest observed gate hides
/// frontier models from the picker.
const CODEX_CLIENT_VERSION: &str = "99.0.0";

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
        Self::new(
            base_url,
            ClientAuth::Codex(Arc::new(CodexAuthSlot::loaded(auth))),
        )
    }

    /// Pre-built shared slot variant. Used by call sites that need
    /// the slot to outlive the constructor (e.g. wiring the same
    /// slot into a daemon-driven publication path *and* the provider
    /// at startup).
    pub fn with_codex_slot(base_url: String, slot: Arc<CodexAuthSlot>) -> Self {
        Self::new(base_url, ClientAuth::Codex(slot))
    }

    fn new(base_url: String, auth: ClientAuth) -> Self {
        crate::ensure_default_crypto_provider();
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
        self.auth
            .prepare_headers(&self.http)
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))
    }

    /// Build + send the Responses request, returning the live HTTP response
    /// for the caller to consume either as a buffered body or an SSE byte
    /// stream. Consolidates body construction, header/auth setup, and the
    /// error-status early return so both the streaming and buffered paths
    /// share one code path up through "first byte".
    async fn send_request(
        &self,
        req: &ModelRequest<'_>,
        cancel: &CancellationToken,
    ) -> Result<(reqwest::Response, ForensicContext), ModelError> {
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
            service_tier: pick_service_tier(req.tunables),
            prompt_cache_key: req.request_cache_key,
        };
        // req.max_tokens is ignored: the ChatGPT-subscription route rejects
        // `max_output_tokens` outright, and Codex's own client doesn't send it
        // on either route. The backend picks an output cap per model; the
        // scheduler's `max_turns` still bounds the overall loop.
        let _ = req.max_tokens;

        // Serialize once, ourselves, so the forensic context can hold a
        // copy of the exact body that went on the wire. `.json(&body)`
        // would serialize internally and discard it before we get a
        // chance to grab a reference.
        let body_str = serde_json::to_string(&body)
            .map_err(|e| ModelError::Transport(format!("serialize request body: {e}")))?;
        let forensic = ForensicContext {
            backend: PROVIDER_TAG.to_string(),
            model: req.model.to_string(),
            request_body: body_str.clone(),
            request_content_type: "application/json",
        };

        let url = format!("{}/responses", self.base_url);
        let prepare = self.prepare_headers();
        let (bearer, extra_headers) = tokio::select! {
            r = prepare => r?,
            _ = cancel.cancelled() => return Err(ModelError::Cancelled),
        };
        let mut builder = self
            .http
            .post(&url)
            .bearer_auth(&bearer)
            .header("accept", "text/event-stream")
            .header("content-type", "application/json")
            .body(body_str);
        for (k, v) in &extra_headers {
            builder = builder.header(*k, v);
        }
        let send = builder.send();
        let resp = tokio::select! {
            r = send => r.map_err(|e| ModelError::Transport(e.to_string()))?,
            _ = cancel.cancelled() => return Err(ModelError::Cancelled),
        };
        let status = resp.status();
        if !status.is_success() {
            let body_text = tokio::select! {
                b = resp.text() => b.unwrap_or_default(),
                _ = cancel.cancelled() => return Err(ModelError::Cancelled),
            };
            return Err(http_error_to_model_error(
                status.as_u16(),
                body_text,
                forensic,
            ));
        }
        Ok((resp, forensic))
    }

    async fn do_create_message(
        &self,
        req: &ModelRequest<'_>,
        cancel: &CancellationToken,
    ) -> Result<ModelResponse, ModelError> {
        let (resp, forensic) = self.send_request(req, cancel).await?;
        // Buffer the full SSE stream and dig out the `response.completed` event
        // — its `response` field has the status / usage / incomplete_details
        // bits we still need even when using the streaming assembly.
        let raw = tokio::select! {
            r = resp.text() => r.map_err(|e| ModelError::Transport(e.to_string()))?,
            _ = cancel.cancelled() => return Err(ModelError::Cancelled),
        };
        let parsed: RspResponse = extract_completed_response(&raw, Some(&forensic))?;
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
        cancel: &'a CancellationToken,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        Box::pin(try_stream! {
            let (resp, forensic) = self.send_request(req, cancel).await?;
            // Extract Codex-route usage headers before the response is
            // consumed by `bytes_stream` — these are the per-account
            // 5h/weekly utilization counters the ChatGPT backend ships
            // on every reply. Returns `None` on the API-key route,
            // where the headers don't exist; the snapshot ride is
            // skipped silently in that case.
            if let Some(snapshot) =
                crate::providers::codex_usage::parse_codex_headers(resp.headers())
            {
                yield ModelEvent::ProviderUsage { snapshot };
            }
            let mut bytes = resp.bytes_stream();
            let mut sse_buf: Vec<u8> = Vec::new();
            let mut forensic_buf = ForensicBuf::default();
            let mut state = RspStreamState::default();
            loop {
                let next = tokio::select! {
                    n = bytes.next() => match n {
                        Some(Ok(b)) => Ok(Some(b)),
                        Some(Err(e)) => Err(ModelError::Transport(e.to_string())),
                        None => Ok(None),
                    },
                    _ = cancel.cancelled() => Err(ModelError::Cancelled),
                };
                let Some(chunk) = next? else { break };
                forensic_buf.push(&chunk);
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
                    for out in state.consume(ev, &forensic, &forensic_buf)? {
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
            return Err(ModelError::api(status.as_u16(), body));
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
                .map(|m| {
                    let capabilities = crate::providers::model::openai_vision_capabilities(&m.slug);
                    let tunables = build_service_tier_tunable(&m)
                        .map(|t| vec![t])
                        .unwrap_or_default();
                    ModelInfo {
                        id: m.slug,
                        display_name: m.display_name,
                        // Codex's /models payload carries display + visibility
                        // but no numeric context/output caps; leave unknown.
                        context_window: None,
                        max_output_tokens: None,
                        capabilities,
                        tunables,
                    }
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
                .map(|m| {
                    let capabilities = crate::providers::model::openai_vision_capabilities(&m.id);
                    ModelInfo {
                        id: m.id,
                        display_name: None,
                        context_window: None,
                        max_output_tokens: None,
                        capabilities,
                        tunables: Vec::new(),
                    }
                })
                .collect())
        }
    }
}

impl ModelProvider for OpenAiResponsesClient {
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

    fn capabilities_for(&self, model_id: &str) -> whisper_agent_protocol::ContentCapabilities {
        crate::providers::model::openai_vision_capabilities(model_id)
    }

    /// Only meaningful for `ClientAuth::Codex` — the API-key route
    /// has no equivalent OAuth-only usage endpoint and returns
    /// `Ok(None)`. Hits [`CHATGPT_WHAM_USAGE_URL`] with the same
    /// bearer + `chatgpt-account-id` header used for `/responses`.
    fn fetch_usage<'a>(
        &'a self,
    ) -> BoxFuture<'a, Result<Option<whisper_agent_protocol::BackendUsage>, ModelError>> {
        Box::pin(async move {
            let slot = match &self.auth {
                ClientAuth::ApiKey(_) => return Ok(None),
                ClientAuth::Codex(s) => s,
            };
            let (bearer, extras) = slot
                .prepare(&self.http)
                .await
                .map_err(|e| ModelError::Transport(e.to_string()))?;
            let mut builder = self.http.get(CHATGPT_WHAM_USAGE_URL).bearer_auth(&bearer);
            for (k, v) in &extras {
                builder = builder.header(*k, v);
            }
            let resp = builder
                .send()
                .await
                .map_err(|e| ModelError::Transport(e.to_string()))?;
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(ModelError::api(status.as_u16(), body));
            }
            let body = resp
                .text()
                .await
                .map_err(|e| ModelError::Transport(e.to_string()))?;
            crate::providers::codex_usage::parse_wham_usage_response(&body)
                .map_err(|e| ModelError::Transport(format!("wham/usage parse: {e}")))
        })
    }

    /// Only meaningful for `ClientAuth::Codex` — API-key clients have
    /// nothing to rotate via this path and return `NotSupported`.
    /// Used by both the admin `UpdateCodexAuth` paste path and the
    /// daemon-driven `PublishCredential` host-env-link path, so the
    /// two channels can't diverge.
    fn update_codex_auth<'a>(
        &'a self,
        contents: &'a str,
    ) -> BoxFuture<'a, Result<(), UpdateAuthError>> {
        Box::pin(async move {
            let slot = match &self.auth {
                ClientAuth::ApiKey(_) => return Err(UpdateAuthError::NotSupported),
                ClientAuth::Codex(s) => s,
            };
            slot.update_from_contents(contents)
                .await
                .map_err(|e| match e {
                    SlotUpdateError::Invalid(m) => UpdateAuthError::Invalid(m),
                    SlotUpdateError::Io(m) => UpdateAuthError::Io(m),
                })
        })
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
        // Mid-conversation harness injection. The thread-prefix
        // system prompt is lifted into `instructions` upstream and
        // stripped from `req.messages` by `build_model_request`, so
        // anything reaching here is a harness injection. Emitted as
        // `role:"developer"` — the Responses API's canonical
        // instructions role. `role:"system"` used to be accepted but
        // `chatgpt.com/backend-api/codex` now rejects it with
        // `{"detail":"System messages are not allowed"}`.
        Role::System => convert_system_message(&m.content, out),
        // Tool-manifest setup messages are filtered out upstream and
        // lifted into the `tools` wire field. Defensive no-op if one
        // slips through.
        Role::Tools => {}
    }
}

/// Emit a mid-conversation `Role::System` injection as a
/// `role:"developer"` input item. Text blocks concatenate
/// (newline-joined) into a single message; non-text blocks
/// (unexpected in a harness injection) are dropped silently.
fn convert_system_message(blocks: &[ContentBlock], out: &mut Vec<RspItem>) {
    let mut text_accum = String::new();
    for block in blocks {
        if let ContentBlock::Text { text } = block {
            if !text_accum.is_empty() {
                text_accum.push('\n');
            }
            text_accum.push_str(text);
        }
    }
    if text_accum.is_empty() {
        return;
    }
    out.push(RspItem::Message {
        role: "developer",
        content: vec![RspInputMessagePart::InputText { text: text_accum }],
    });
}

/// A user turn is a mix of plain text, tool results, and multimodal
/// attachments. Responses wants tool results as standalone
/// `function_call_output` items, so those come out inline; text and
/// images land together on a single `message` item with a
/// `content: [{InputText}, {InputImage}, ...]` parts vector.
///
/// The parts buffer is built eagerly when we hit the first image — we
/// need it whether or not text shows up after. Pure text turns still
/// emit one `InputText` part in their message (same shape they already
/// had).
fn convert_user_message(blocks: &[ContentBlock], out: &mut Vec<RspItem>) {
    let mut parts: Vec<RspInputMessagePart> = Vec::new();
    let mut text_accum = String::new();
    let fold_text = |parts: &mut Vec<RspInputMessagePart>, text_accum: &mut String| {
        if !text_accum.is_empty() {
            parts.push(RspInputMessagePart::InputText {
                text: std::mem::take(text_accum),
            });
        }
    };
    let flush_message =
        |parts: &mut Vec<RspInputMessagePart>, text_accum: &mut String, out: &mut Vec<RspItem>| {
            fold_text(parts, text_accum);
            if !parts.is_empty() {
                out.push(RspItem::Message {
                    role: "user",
                    content: std::mem::take(parts),
                });
            }
        };
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                if !text_accum.is_empty() {
                    text_accum.push('\n');
                }
                text_accum.push_str(text);
            }
            ContentBlock::Image { source, .. } => {
                fold_text(&mut parts, &mut text_accum);
                parts.push(image_source_to_input_part(source));
            }
            ContentBlock::Document { source } => {
                fold_text(&mut parts, &mut text_accum);
                parts.push(document_source_to_input_part(source));
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                flush_message(&mut parts, &mut text_accum, out);
                out.push(RspItem::FunctionCallOutput {
                    call_id: tool_use_id.clone(),
                    output: tool_result_as_text(content, *is_error),
                });
                // Responses' `function_call_output` carries only a
                // plain `output: string` — image and document
                // attachments ride in a follow-up `role:user` message
                // with `input_image` / `input_file` parts, matching
                // the workaround used on Chat Completions' text-only
                // `role:tool` message.
                let images = content.image_sources();
                let documents = content.document_sources();
                if !images.is_empty() || !documents.is_empty() {
                    let mut media_parts: Vec<RspInputMessagePart> =
                        Vec::with_capacity(images.len() + documents.len() + 1);
                    media_parts.push(RspInputMessagePart::InputText {
                        text: format!("[media output from tool call {tool_use_id} follows]"),
                    });
                    for src in images {
                        media_parts.push(image_source_to_input_part(src));
                    }
                    for src in documents {
                        media_parts.push(document_source_to_input_part(src));
                    }
                    out.push(RspItem::Message {
                        role: "user",
                        content: media_parts,
                    });
                }
            }
            ContentBlock::ToolUse { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::ToolSchema { .. } => {
                // None belong on a user message. Drop silently.
            }
        }
    }
    flush_message(&mut parts, &mut text_accum, out);
}

/// Lower an image source to the Responses API's `input_image` part.
/// Bytes → base64 data URL; Url → https passthrough. File-id references
/// aren't v1 scope — Files API integration is deferred until we hit a
/// cost/size constraint that makes inline bytes impractical.
fn image_source_to_input_part(src: &whisper_agent_protocol::ImageSource) -> RspInputMessagePart {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use whisper_agent_protocol::ImageSource;
    let url = match src {
        ImageSource::Bytes { media_type, data } => format!(
            "data:{};base64,{}",
            media_type.as_mime_str(),
            STANDARD.encode(data)
        ),
        ImageSource::Url { url } => url.clone(),
    };
    RspInputMessagePart::InputImage {
        image_url: Some(url),
    }
}

/// Lower a document source to the Responses API's `input_file` part.
/// Bytes → base64 data URL with a synthetic filename; Url → file_url
/// passthrough. file_id (Files API) is reserved.
fn document_source_to_input_part(
    src: &whisper_agent_protocol::DocumentSource,
) -> RspInputMessagePart {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use whisper_agent_protocol::{DocumentMime, DocumentSource};
    match src {
        DocumentSource::Bytes { media_type, data } => RspInputMessagePart::InputFile {
            filename: Some(format!(
                "document.{}",
                match media_type {
                    DocumentMime::Pdf => "pdf",
                }
            )),
            file_data: Some(format!(
                "data:{};base64,{}",
                media_type.as_mime_str(),
                STANDARD.encode(data)
            )),
            file_url: None,
        },
        DocumentSource::Url { url } => RspInputMessagePart::InputFile {
            filename: None,
            file_data: None,
            file_url: Some(url.clone()),
        },
    }
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
            ContentBlock::ToolResult { .. } | ContentBlock::ToolSchema { .. } => {
                // Neither belongs on an assistant message.
            }
            ContentBlock::Image { source, replay } => {
                // Built-in image_generation echo: if the block was
                // minted by *this* backend, re-emit as an
                // image_generation_call input item so the server sees
                // the prior image and can edit it on follow-up turns.
                // Blocks tagged for a different provider (or untagged
                // user uploads) don't belong on an assistant turn —
                // drop. Mirrors the Thinking-replay treatment above.
                if let Some(item) = image_to_image_generation_call_item(source, replay.as_ref()) {
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
            ContentBlock::Document { .. } => {
                // Documents are never model-emitted. Drop silently.
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

/// Build an outbound `image_generation_call` input item from a stored
/// assistant `ContentBlock::Image`. Returns `None` if the replay blob was
/// minted by a different provider (we can't echo someone else's id) or
/// if the source isn't inline bytes (URL-source images don't carry the
/// base64 the API expects on the input side).
///
/// Output-only fields (`action`, `background`, `output_format`,
/// `quality`, `size`, `revised_prompt`) are deliberately omitted — the
/// API 400s with `Unknown parameter: input[N].<field>` if they're
/// present. Empirical reference: `scripts/probe_responses_api.sh`.
fn image_to_image_generation_call_item(
    source: &whisper_agent_protocol::ImageSource,
    replay: Option<&ProviderReplay>,
) -> Option<RspItem> {
    let r = replay?;
    if r.provider != PROVIDER_TAG {
        return None;
    }
    let bytes = match source {
        whisper_agent_protocol::ImageSource::Bytes { data, .. } => data,
        whisper_agent_protocol::ImageSource::Url { .. } => return None,
    };
    use base64::Engine;
    let result = base64::engine::general_purpose::STANDARD.encode(bytes);
    let obj = r.data.as_object();
    let id = obj
        .and_then(|o| o.get("id"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let status = obj
        .and_then(|o| o.get("status"))
        .and_then(|v| v.as_str())
        .map(String::from);
    Some(RspItem::ImageGenerationCall { id, status, result })
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
    if let whisper_agent_protocol::ToolKind::ProviderBuiltin = t.kind {
        // Provider-built-in tools dispatch by reserved name; the
        // synthesizer in scheduler.rs and this match must agree.
        // Future built-ins (file_search, code_interpreter, …) slot in
        // alongside.
        match t.name.as_str() {
            "image_generation" => {
                return RspTool::ImageGeneration {
                    size: "auto",
                    quality: "auto",
                };
            }
            "web_search" => return RspTool::WebSearch,
            _ => {}
        }
    }
    RspTool::Function {
        name: t.name.clone(),
        description: t.description.clone(),
        parameters: t.input_schema_value(),
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
/// Cross-module test hook: surface the parser to `runtime::forensics`
/// so it can drive an end-to-end test (response.failed body →
/// ModelError → ForensicSink dump). The Ok variant is dropped because
/// the test only cares about the error path; staying behind `cfg(test)`
/// keeps the success type private.
#[cfg(test)]
pub fn extract_completed_response_for_testing(
    body: &str,
    forensic: Option<&ForensicContext>,
) -> Result<(), ModelError> {
    extract_completed_response(body, forensic).map(|_| ())
}

/// `response.failed` short-circuits to [`ModelError::Api`] so terminal
/// server-side errors surface cleanly to the caller. When `forensic`
/// is `Some`, the assembled body bytes ride along on the error for the
/// forensic dump path.
fn extract_completed_response(
    body: &str,
    forensic: Option<&ForensicContext>,
) -> Result<RspResponse, ModelError> {
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
                let (status, body) = response_failed_body(&response, &error);
                let detail = response_failed_detail(response.as_ref(), error.as_ref());
                let artifact = forensic
                    .cloned()
                    .map(|ctx| ctx.into_http_error_artifact(body.as_bytes().to_vec()));
                return Err(ModelError::Api {
                    status: 500,
                    body: format!("{status}: {body}"),
                    detail: Some(Box::new(detail)),
                    forensic: artifact.map(std::sync::Arc::new),
                });
            }
            SseEvent::Other => {}
        }
    }
    Err(ModelError::Transport(
        "SSE stream ended without response.completed".into(),
    ))
}

/// Pull `(status_label, human_message)` out of a `response.failed`
/// payload for the user-visible `ModelError::Api.body` string. Status
/// is the parent `response.status`; message prefers the most-specific
/// nested error.message over the top-level one (the codex backend
/// puts the real detail on `response.error.message`).
fn response_failed_body(
    response: &Option<RspResponse>,
    error: &Option<SseError>,
) -> (String, String) {
    let nested = response.as_ref().and_then(|r| r.error.as_ref());
    let msg = nested
        .map(|e| e.message.clone())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            error
                .as_ref()
                .map(|e| e.message.clone())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "response.failed".into());
    let status = response
        .as_ref()
        .and_then(|r| r.status.clone())
        .unwrap_or_else(|| "failed".into());
    (status, msg)
}

/// Build a [`ProviderErrorDetail`] from a `response.failed` payload.
/// Reads both the top-level `error` and the nested `response.error`
/// since chatgpt.com/codex routes the useful fields through the
/// nested envelope, while api.openai.com uses the top level. Prefers
/// the nested values when populated.
fn response_failed_detail(
    response: Option<&RspResponse>,
    top_error: Option<&SseError>,
) -> ProviderErrorDetail {
    let nested = response.and_then(|r| r.error.as_ref());
    let pick = |nested: Option<&String>, top: Option<&String>| -> Option<String> {
        nested
            .filter(|s| !s.is_empty())
            .or(top.filter(|s| !s.is_empty()))
            .cloned()
    };
    ProviderErrorDetail {
        code: pick(
            nested.and_then(|e| e.code.as_ref()),
            top_error.and_then(|e| e.code.as_ref()),
        ),
        kind: pick(
            nested.and_then(|e| e.kind.as_ref()),
            top_error.and_then(|e| e.kind.as_ref()),
        ),
        response_id: response.and_then(|r| r.id.clone()).or_else(|| {
            // Fallback: some payloads put the request id on the error
            // envelope instead of the parent response.
            top_error
                .and_then(|e| e.request_id.clone())
                .or_else(|| nested.and_then(|e| e.request_id.clone()))
        }),
        message: pick(nested.map(|e| &e.message), top_error.map(|e| &e.message)),
    }
}

/// Build a [`ModelError`] from an HTTP non-2xx response body. Tries to
/// deserialize the body as the standard OpenAI `{ "error": { ... } }`
/// envelope; falls back to all-None detail when the body isn't JSON
/// (e.g. a Cloudflare HTML page in front of a real outage). Always
/// attaches the forensic context — the body IS the response excerpt.
fn http_error_to_model_error(status: u16, body: String, forensic: ForensicContext) -> ModelError {
    #[derive(Deserialize)]
    struct Envelope {
        error: SseError,
    }
    let detail = serde_json::from_str::<Envelope>(&body)
        .ok()
        .map(|env| ProviderErrorDetail {
            code: env.error.code.filter(|s| !s.is_empty()),
            kind: env.error.kind.filter(|s| !s.is_empty()),
            response_id: env.error.request_id.filter(|s| !s.is_empty()),
            message: Some(env.error.message).filter(|s| !s.is_empty()),
        });
    let artifact = forensic.into_http_error_artifact(body.as_bytes().to_vec());
    ModelError::Api {
        status,
        body,
        detail: detail.map(Box::new),
        forensic: Some(std::sync::Arc::new(artifact)),
    }
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
    /// A new output item is starting. For `function_call` items this is
    /// the earliest point we know the tool name — before any arguments
    /// have streamed — so we can emit a placeholder for the UI.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded { item: RspOutputItem },
    /// Streaming fragment of a function_call's arguments JSON. We use
    /// it only to drive throttled `ToolCallStreaming` char counts; the
    /// fully-parsed args still come through `OutputItemDone`.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        item_id: String,
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
    /// Pending function_call items seen via `response.output_item.added`
    /// but not yet through `output_item.done`. Keyed by the item id
    /// that `function_call_arguments.delta` events reference. Each entry
    /// carries the `call_id` (the identifier surfaced on the wire in
    /// subsequent `ToolCall` / `ThreadToolCallBegin` events) and the
    /// running args-char count for throttled streaming updates.
    streaming_tool_calls: HashMap<String, ResponsesStreamingCall>,
    done: bool,
}

/// Book-keeping for one in-flight `function_call` item during streaming.
/// Lives in [`RspStreamState::streaming_tool_calls`] between the
/// `output_item.added` that opened it and the `output_item.done` that
/// finalizes it.
struct ResponsesStreamingCall {
    call_id: String,
    name: String,
    args_chars: u32,
    last_emitted_chars: u32,
}

/// Matches the Anthropic driver's identical constant. Throttles the
/// per-char-progress emission rate during streaming.
const TOOL_STREAMING_CHAR_STEP: u32 = 32;

impl RspStreamState {
    fn consume(
        &mut self,
        event: RspStreamEvent,
        forensic: &ForensicContext,
        forensic_buf: &ForensicBuf,
    ) -> Result<Vec<ModelEvent>, ModelError> {
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
            RspStreamEvent::OutputItemAdded { item } => {
                // Surface the function_call placeholder the moment its
                // name is known — the args may take seconds to stream
                // through `function_call_arguments.delta`, and we'd
                // rather the UI frame the row now than stay silent.
                if let RspOutputItem::FunctionCall {
                    id: Some(item_id),
                    call_id,
                    name,
                    arguments,
                } = &item
                {
                    let args_chars = arguments.len() as u32;
                    self.streaming_tool_calls.insert(
                        item_id.clone(),
                        ResponsesStreamingCall {
                            call_id: call_id.clone(),
                            name: name.clone(),
                            args_chars,
                            last_emitted_chars: args_chars,
                        },
                    );
                    out.push(ModelEvent::ToolCallStreaming {
                        id: call_id.clone(),
                        name: name.clone(),
                        args_chars,
                    });
                }
            }
            RspStreamEvent::FunctionCallArgumentsDelta { item_id, delta } => {
                if let Some(pending) = self.streaming_tool_calls.get_mut(&item_id) {
                    pending.args_chars = pending.args_chars.saturating_add(delta.len() as u32);
                    if pending.args_chars
                        >= pending
                            .last_emitted_chars
                            .saturating_add(TOOL_STREAMING_CHAR_STEP)
                    {
                        pending.last_emitted_chars = pending.args_chars;
                        out.push(ModelEvent::ToolCallStreaming {
                            id: pending.call_id.clone(),
                            name: pending.name.clone(),
                            args_chars: pending.args_chars,
                        });
                    }
                }
            }
            RspStreamEvent::OutputItemDone { item } => {
                // Retire any pending streaming entry for this item — the
                // subsequent `ToolCall` event supersedes every
                // `ToolCallStreaming` we emitted for the same call.
                if let RspOutputItem::FunctionCall { id: Some(id), .. } = &item {
                    self.streaming_tool_calls.remove(id);
                }
                // Emit a ToolCall event for function_call items so the
                // downstream consumer (the scheduler) has symmetric
                // coverage with the Anthropic adapter. Message /
                // reasoning items already streamed as deltas.
                if let RspOutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
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
                // image_generation_call items also need a live event —
                // without one the webui never receives ThreadAssistantImage
                // and the image only appears after the next snapshot
                // refresh. Decode the base64 result here and emit an
                // ImageBlock so the picture lights up immediately on
                // turn completion. The persisted ContentBlock::Image
                // (with ProviderReplay metadata for round-trip) still
                // rides on the final Completed event via finalize_response.
                if let RspOutputItem::ImageGenerationCall {
                    result: Some(b64), ..
                } = &item
                {
                    use base64::Engine;
                    match base64::engine::general_purpose::STANDARD.decode(b64.as_bytes()) {
                        Ok(bytes) => {
                            out.push(ModelEvent::ImageBlock {
                                source: whisper_agent_protocol::ImageSource::Bytes {
                                    media_type: whisper_agent_protocol::ImageMime::Png,
                                    data: bytes,
                                },
                            });
                        }
                        Err(e) => {
                            return Err(ModelError::Transport(format!(
                                "image_generation_call result not valid base64: {e}"
                            )));
                        }
                    }
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
                // If the turn ended `incomplete` for any reason other
                // than hitting max_output_tokens, surface a warning so
                // an operator sees content_filter / safety / unknown
                // truncations in `kubectl logs` rather than only as a
                // mystery `stop_reason`.
                if response.status.as_deref() == Some("incomplete") {
                    let reason = response
                        .incomplete_details
                        .as_ref()
                        .and_then(|d| d.reason.as_deref())
                        .unwrap_or("unspecified");
                    if reason != "max_output_tokens" {
                        out.push(ModelEvent::ProviderWarning {
                            code: format!("incomplete:{reason}"),
                            message: format!(
                                "Responses API returned incomplete status (reason={reason})"
                            ),
                        });
                    }
                }
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
                let (status, body) = response_failed_body(&response, &error);
                let detail = response_failed_detail(response.as_ref(), error.as_ref());
                let artifact = forensic.clone().into_artifact_with_response(ForensicBuf {
                    bytes: forensic_buf.bytes.clone(),
                    truncated: forensic_buf.truncated,
                });
                return Err(ModelError::Api {
                    status: 500,
                    body: format!("{status}: {body}"),
                    detail: Some(Box::new(detail)),
                    forensic: Some(std::sync::Arc::new(artifact)),
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
            RspOutputItem::ImageGenerationCall {
                id,
                status,
                result,
                revised_prompt,
            } => {
                let Some(b64) = result else {
                    // Some intermediate variants of this item arrive
                    // mid-stream with status="generating" and no result;
                    // the terminal item carries the bytes. Defensive:
                    // skip a result-less item rather than fail the turn.
                    continue;
                };
                use base64::Engine;
                let bytes = match base64::engine::general_purpose::STANDARD.decode(b64.as_bytes()) {
                    Ok(b) => b,
                    Err(e) => {
                        return Err(ModelError::Transport(format!(
                            "image_generation_call result not valid base64: {e}"
                        )));
                    }
                };
                let mut replay_data = serde_json::Map::new();
                if let Some(id) = &id {
                    replay_data.insert("id".into(), Value::String(id.clone()));
                }
                if let Some(s) = &status {
                    replay_data.insert("status".into(), Value::String(s.clone()));
                }
                if let Some(rp) = &revised_prompt {
                    // Display-only; never sent back. Preserved on the
                    // replay blob so the UI can surface "what the model
                    // decided to draw" without us re-deriving it.
                    replay_data.insert("revised_prompt".into(), Value::String(rp.clone()));
                }
                let replay = (!replay_data.is_empty()).then(|| ProviderReplay {
                    provider: PROVIDER_TAG.into(),
                    data: Value::Object(replay_data),
                });
                content.push(ContentBlock::Image {
                    source: whisper_agent_protocol::ImageSource::Bytes {
                        media_type: whisper_agent_protocol::ImageMime::Png,
                        data: bytes,
                    },
                    replay,
                });
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
            RspOutputItem::WebSearchCall => {
                // Intentional drop — see the doc comment on the variant.
                // The model's text reply (next item) carries the search
                // results inline with citations.
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
    /// Codex service-tier override (e.g. `"priority"` for Fast). Omitted
    /// from the wire when `None` so default-tier requests stay
    /// byte-identical to pre-tunable bodies — the chatgpt.com backend
    /// 400s on unknown tiers, so we only send values the model
    /// advertised in `/models`.
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a str>,
    /// Stable per-thread routing-affinity hint. Combined with the prefix
    /// hash on the server side so successive requests for the same
    /// thread land on the same cache shard. Without it the load
    /// balancer routes by prefix-hash alone and adjacent same-prefix
    /// requests inside one tool loop can be shuffled across cache-cold
    /// nodes, blowing the prompt cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
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
    ///
    /// `summary` is always serialized — even when empty — because the
    /// Responses API requires the field to be present on every
    /// reasoning item (the server 400s with "Missing required
    /// parameter: 'input[N].summary'" if the field is absent). The
    /// captured value is empty when the prior response was a pure
    /// reasoning turn that surfaced no visible summary text but still
    /// carried `id` + `encrypted_content` for replay.
    Reasoning {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        summary: Vec<RspReasoningPart>,
    },
    /// Replay item for a previously-generated image, echoed into `input`
    /// so the model sees the prior bytes and can edit them on a follow-up
    /// turn. Output-only fields (action / background / output_format /
    /// quality / size / revised_prompt) are stripped — the API rejects
    /// them on input (`Unknown parameter: input[N].action`). Empirical
    /// reference: `scripts/probe_responses_api.sh`.
    ImageGenerationCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        result: String,
    },
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RspInputMessagePart {
    InputText {
        text: String,
    },
    OutputText {
        text: String,
    },
    /// Responses-API user-input image. Accepts either an `image_url`
    /// (https URL or `data:image/...;base64,...` data URL) or a
    /// `file_id` from the Files API. V1 uses `image_url` for both
    /// inline bytes and user-supplied URLs; Files-API integration is
    /// deferred until v2.
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
    },
    /// Responses-API user-input file (PDF). Mirrors `InputImage`:
    /// either `file_data` (a `data:application/pdf;base64,...` URL,
    /// usually with a `filename` companion field) or `file_url`
    /// (https URL) or `file_id` (Files API). V1 inlines via
    /// file_data; the other forms are reserved for later.
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
    },
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
    /// Provider-side built-in: model invokes it inside `/responses` and
    /// the result rides back as an output item, no separate tool_use →
    /// tool_result roundtrip. v1 keeps the size/quality knobs at their
    /// `auto` defaults so the API picks per-prompt; future iterations
    /// could plumb operator-set defaults.
    ImageGeneration {
        size: &'static str,
        quality: &'static str,
    },
    /// Provider-side web search. Output items (`web_search_call`) are
    /// metadata about the search the model ran; the actual results
    /// flow into the model's text reply as inline markdown citations,
    /// which already round-trip through the normal assistant-message
    /// path. We don't echo `web_search_call` items back next turn.
    WebSearch,
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

/// Top-level `error` payload on a `response.failed` SSE event, or the
/// error envelope returned in an HTTP non-2xx JSON body. OpenAI ships
/// the same shape both places.
#[derive(Deserialize, Debug, Default)]
struct SseError {
    #[serde(default)]
    message: String,
    /// Provider-shaped code: `context_length_exceeded`,
    /// `invalid_api_key`, etc. Optional because some payloads only
    /// carry a `type`.
    #[serde(default)]
    code: Option<String>,
    /// Coarser category: `invalid_request_error`, `server_error`,
    /// `rate_limit_error`.
    #[serde(default, rename = "type")]
    kind: Option<String>,
    /// OpenAI request id. Sometimes lives here on the API-key route;
    /// the subscription route puts it on the parent envelope instead.
    #[serde(default)]
    request_id: Option<String>,
}

#[derive(Deserialize, Debug)]
struct RspResponse {
    /// Response id (the `resp_...` token the API mints for this turn).
    /// Empty on the older subscription-path replies that omit it.
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    output: Vec<RspOutputItem>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    incomplete_details: Option<RspIncompleteDetails>,
    #[serde(default)]
    usage: Option<RspUsage>,
    /// Nested error envelope — populated by the chatgpt.com/codex route
    /// on `response.failed` instead of (or in addition to) the
    /// top-level `error` field. The api.openai.com route ships info
    /// on top-level; subscription is inconsistent. Read both, prefer
    /// whichever has the most filled fields.
    #[serde(default)]
    error: Option<SseError>,
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
        /// Item id, distinct from `call_id`. `function_call_arguments.delta`
        /// events reference this field, not `call_id`. Optional because
        /// some payloads (notably the `response.output` array on
        /// `response.completed`) omit it; only `output_item.added` is
        /// guaranteed to carry it.
        #[serde(default)]
        id: Option<String>,
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
    /// Result of an `image_generation` built-in tool invocation. The
    /// model decided to invoke the tool and the server ran it inline;
    /// `result` is base64-encoded PNG bytes. Other fields (action,
    /// background, output_format, quality, size, revised_prompt) are
    /// output-only metadata — see the probe findings under
    /// `scripts/probe_responses_api.sh` for the input-shape constraints
    /// when echoing this back next turn.
    ImageGenerationCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        result: Option<String>,
        #[serde(default)]
        revised_prompt: Option<String>,
    },
    /// Result of a `web_search` built-in invocation. The user-visible
    /// search results flow through as inline markdown citations on the
    /// subsequent `Message` item — this item just captures metadata
    /// (id + the queries the model issued). We accept it explicitly
    /// so unknown shapes still loud-fail via `Other`, but we don't
    /// surface it to the conversation log: the Message already has
    /// everything the user needs, and there's no round-trip benefit
    /// (the model's text reply already carries the search context
    /// forward). Fields are deliberately omitted — none are read
    /// today; serde silently accepts the rest of the wire payload.
    WebSearchCall,
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
    /// Service tiers this model can run with — each tier becomes a
    /// variant on the `service_tier` tunable advertised to the UI.
    /// Empty for models that don't support any tier override.
    #[serde(default)]
    service_tiers: Vec<RspCodexServiceTier>,
    /// Legacy shape for fast-mode-only models that pre-date
    /// `service_tiers`. Codex's own `supports_fast_mode()` falls back
    /// to checking this for `"fast"`; we mirror that so models on the
    /// older metadata still pick up a Fast variant.
    #[serde(default)]
    additional_speed_tiers: Vec<String>,
}

/// One entry in the codex `/models` response's `service_tiers` array.
/// `id` is the wire value sent back as the request body's `service_tier`
/// field; `name` and `description` are user-facing strings.
#[derive(Deserialize, Debug)]
struct RspCodexServiceTier {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

/// Tunable key surfaced on every codex-mode model that supports a
/// service tier. Stable identifier — UIs may persist user choices by
/// this string. Matches the wire field name for clarity.
const TUNABLE_KEY_SERVICE_TIER: &str = "service_tier";

/// Build the `service_tier` tunable advertised for one codex model, or
/// `None` when the model's metadata says no override is available.
///
/// Variants are pulled directly from the API: `service_tiers` for the
/// modern shape (`id`, `name`, `description`), with a legacy
/// `additional_speed_tiers: ["fast"]` fallback that maps to the same
/// wire value (`"priority"`) so older metadata still surfaces the
/// Fast option. A synthetic "Default" variant with empty `value`
/// represents "omit the field entirely" on the wire.
fn build_service_tier_tunable(m: &RspCodexModel) -> Option<TunableSpec> {
    let mut variants: Vec<TunableEnumVariant> = vec![TunableEnumVariant {
        value: String::new(),
        label: Some("Default".into()),
        description: Some("No service tier override.".into()),
    }];
    for tier in &m.service_tiers {
        if tier.id.is_empty() || variants.iter().any(|v| v.value == tier.id) {
            continue;
        }
        variants.push(TunableEnumVariant {
            value: tier.id.clone(),
            label: tier.name.clone(),
            description: tier.description.clone(),
        });
    }
    // `additional_speed_tiers: ["fast"]` is the older shape; codex
    // itself maps that to the modern Fast tier whose wire id is
    // "priority". Only synthesize a variant if the modern entry
    // wasn't already pulled in above.
    for legacy in &m.additional_speed_tiers {
        if legacy.eq_ignore_ascii_case("fast") && !variants.iter().any(|v| v.value == "priority") {
            variants.push(TunableEnumVariant {
                value: "priority".into(),
                label: Some("Fast".into()),
                description: Some("Priority queue inference.".into()),
            });
        }
    }
    // Only the synthetic Default — model effectively has no choice,
    // so don't render a control at all.
    if variants.len() <= 1 {
        return None;
    }
    Some(TunableSpec {
        key: TUNABLE_KEY_SERVICE_TIER.into(),
        label: Some("Service tier".into()),
        description: Some("Inference queue for this model.".into()),
        kind: TunableKind::Enum {
            variants,
            default: String::new(),
        },
    })
}

/// Translate the picked `service_tier` value into the wire field. Empty
/// strings (the synthetic Default) and unset tunables both elide the
/// field — the API treats both as "no override."
fn pick_service_tier(tunables: &std::collections::BTreeMap<String, TunableValue>) -> Option<&str> {
    match tunables.get(TUNABLE_KEY_SERVICE_TIER) {
        Some(TunableValue::Enum(v)) if !v.is_empty() => Some(v.as_str()),
        _ => None,
    }
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
    fn serializes_user_message_with_image_bytes() {
        use whisper_agent_protocol::{ImageMime, ImageSource};
        let msg = Message::user_blocks(vec![
            ContentBlock::Text {
                text: "what's here?".into(),
            },
            ContentBlock::Image {
                source: ImageSource::Bytes {
                    media_type: ImageMime::Jpeg,
                    data: vec![0xff, 0xd8, 0xff, 0xe0],
                },
                replay: None,
            },
        ]);
        let mut items = Vec::new();
        convert_message(&msg, &mut items);
        let json = serde_json::to_value(&items).unwrap();
        let content = &json[0]["content"];
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "what's here?");
        assert_eq!(content[1]["type"], "input_image");
        let url = content[1]["image_url"].as_str().unwrap();
        assert!(url.starts_with("data:image/jpeg;base64,"), "url={url}");
    }

    #[test]
    fn serializes_user_message_with_image_url() {
        use whisper_agent_protocol::ImageSource;
        let msg = Message::user_blocks(vec![ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.com/a.png".into(),
            },
            replay: None,
        }]);
        let mut items = Vec::new();
        convert_message(&msg, &mut items);
        let json = serde_json::to_value(&items).unwrap();
        assert_eq!(
            json[0]["content"][0]["image_url"],
            "https://example.com/a.png"
        );
    }

    #[test]
    fn tool_result_with_image_emits_followup_user_input_image() {
        // function_call_output is `output: string`. Image tool results
        // ride in a follow-up Message item with input_image parts.
        use whisper_agent_protocol::{ImageMime, ImageSource, ToolResultContent};
        let msg = Message::tool_result_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "call_42".into(),
            content: ToolResultContent::Blocks(vec![
                ContentBlock::Text {
                    text: "got it".into(),
                },
                ContentBlock::Image {
                    source: ImageSource::Bytes {
                        media_type: ImageMime::Png,
                        data: vec![137, 80, 78, 71],
                    },
                    replay: None,
                },
            ]),
            is_error: false,
        }]);
        let mut items = Vec::new();
        convert_message(&msg, &mut items);
        let json = serde_json::to_value(&items).unwrap();
        let arr = json.as_array().unwrap();
        // [function_call_output, message{role:user, input_image}]
        assert!(arr.len() >= 2, "got {arr:?}");
        assert_eq!(arr[0]["type"], "function_call_output");
        assert_eq!(arr[0]["call_id"], "call_42");
        assert_eq!(arr[1]["role"], "user");
        let parts = arr[1]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "input_text");
        assert_eq!(parts[1]["type"], "input_image");
        let url = parts[1]["image_url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "url={url}");
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
    fn mid_conversation_system_emits_role_developer_input_item() {
        // Injected Role::System (not the thread-prefix prompt — that one
        // is lifted into `instructions` upstream) should appear as a
        // role:"developer" input item with the wrapped text. The Codex
        // backend rejects `role:"system"`; `developer` is the modern
        // Responses-API name for the same slot.
        let msg = Message::system_text("check memory/ before answering");
        let mut items = Vec::new();
        convert_message(&msg, &mut items);
        let json = serde_json::to_value(&items).unwrap();
        assert_eq!(
            json,
            serde_json::json!([
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "check memory/ before answering"}]}
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
        let r = extract_completed_response(stream, None).unwrap();
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
        let r = extract_completed_response(stream, None).unwrap();
        assert_eq!(r.output.len(), 2);
        assert!(matches!(&r.output[0], RspOutputItem::Message { .. }));
        assert!(matches!(&r.output[1], RspOutputItem::FunctionCall { .. }));
    }

    #[test]
    fn sse_stream_without_completed_errors() {
        let stream = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n";
        let err = extract_completed_response(stream, None).unwrap_err();
        assert!(matches!(err, ModelError::Transport(_)));
    }

    #[test]
    fn sse_response_failed_surfaces_as_api_error() {
        let stream = r#"event: response.failed
data: {"type":"response.failed","response":{"status":"failed"},"error":{"message":"rate limited"}}

"#;
        let err = extract_completed_response(stream, None).unwrap_err();
        match err {
            ModelError::Api { status, body, .. } => {
                assert_eq!(status, 500);
                assert!(body.contains("rate limited"), "body was: {body}");
            }
            _ => panic!("expected Api error"),
        }
    }

    #[test]
    fn response_failed_with_nested_response_error_extracts_detail() {
        // chatgpt.com/codex shape: error payload lives on
        // `response.error.{code,message}` not on the top-level
        // `error`. Verifies we mine the nested envelope into
        // ProviderErrorDetail so io_dispatch can log the actual code.
        let stream = r#"event: response.failed
data: {"type":"response.failed","response":{"id":"resp_abc123","status":"failed","error":{"code":"server_error","type":"server_error","message":"Sorry, something went wrong"}}}

"#;
        let err = extract_completed_response(stream, None).unwrap_err();
        let detail = err.provider_detail().cloned().expect("detail attached");
        assert_eq!(detail.code.as_deref(), Some("server_error"));
        assert_eq!(detail.kind.as_deref(), Some("server_error"));
        assert_eq!(detail.response_id.as_deref(), Some("resp_abc123"));
        assert_eq!(
            detail.message.as_deref(),
            Some("Sorry, something went wrong")
        );
        match err {
            ModelError::Api { status, body, .. } => {
                assert_eq!(status, 500);
                assert!(
                    body.contains("Sorry, something went wrong"),
                    "body was: {body}"
                );
            }
            _ => panic!("expected Api error"),
        }
    }

    #[test]
    fn response_failed_prefers_nested_when_both_envelopes_present() {
        // Pathological-but-real shape from the chatgpt subscription
        // route: both top-level `error.message` and `response.error`
        // populated. The nested one carries the specific code, so it
        // wins.
        let stream = r#"event: response.failed
data: {"type":"response.failed","response":{"id":"resp_xyz","status":"failed","error":{"code":"context_length_exceeded","message":"Request exceeds context window"}},"error":{"message":"top-level fallback"}}

"#;
        let err = extract_completed_response(stream, None).unwrap_err();
        let detail = err.provider_detail().cloned().unwrap();
        assert_eq!(detail.code.as_deref(), Some("context_length_exceeded"));
        assert_eq!(
            detail.message.as_deref(),
            Some("Request exceeds context window")
        );
        assert_eq!(detail.response_id.as_deref(), Some("resp_xyz"));
    }

    #[test]
    fn response_failed_with_forensic_attaches_artifact() {
        // Verify the forensic context flows through to the error so
        // io_dispatch's dump-to-disk path has the request body to
        // record alongside the failed response.
        let stream = r#"event: response.failed
data: {"type":"response.failed","response":{"status":"failed"},"error":{"message":"boom"}}

"#;
        let ctx = ForensicContext {
            backend: "openai_responses".into(),
            model: "gpt-5".into(),
            request_body: r#"{"hello":"world"}"#.into(),
            request_content_type: "application/json",
        };
        let err = extract_completed_response(stream, Some(&ctx)).unwrap_err();
        let artifact = err.forensic().expect("artifact attached");
        assert_eq!(artifact.backend, "openai_responses");
        assert_eq!(artifact.model, "gpt-5");
        assert_eq!(artifact.request_body, r#"{"hello":"world"}"#);
        assert!(
            std::str::from_utf8(&artifact.response_excerpt)
                .unwrap()
                .contains("boom"),
            "excerpt should include the failure body"
        );
        assert!(!artifact.response_truncated);
    }

    #[test]
    fn http_error_envelope_parses_into_provider_detail() {
        // Standard OpenAI 4xx body shape — verify we mine code/type/
        // message into ProviderErrorDetail so a 400 from a malformed
        // request shows up as `error.code = invalid_request_error`
        // in the logs instead of just a raw body.
        let body = r#"{"error":{"message":"Invalid model","type":"invalid_request_error","code":"model_not_found"}}"#;
        let ctx = ForensicContext {
            backend: PROVIDER_TAG.to_string(),
            model: "bad-model".into(),
            request_body: "{}".into(),
            request_content_type: "application/json",
        };
        let err = http_error_to_model_error(400, body.to_string(), ctx);
        let detail = err.provider_detail().cloned().expect("detail extracted");
        assert_eq!(detail.code.as_deref(), Some("model_not_found"));
        assert_eq!(detail.kind.as_deref(), Some("invalid_request_error"));
        assert_eq!(detail.message.as_deref(), Some("Invalid model"));
        let artifact = err.forensic().expect("artifact attached");
        assert_eq!(artifact.request_body, "{}");
        assert!(
            std::str::from_utf8(&artifact.response_excerpt)
                .unwrap()
                .contains("Invalid model")
        );
    }

    #[test]
    fn http_error_with_non_json_body_still_attaches_forensic_without_detail() {
        // Cloudflare HTML page or similar — body isn't JSON, so detail
        // stays None but the forensic capture still records the bytes
        // so an operator can see what the upstream actually returned.
        let body = "<html><body>502 Bad Gateway</body></html>";
        let ctx = ForensicContext {
            backend: PROVIDER_TAG.to_string(),
            model: "gpt-5".into(),
            request_body: "{}".into(),
            request_content_type: "application/json",
        };
        let err = http_error_to_model_error(502, body.to_string(), ctx);
        assert!(err.provider_detail().is_none(), "no structured detail");
        let artifact = err.forensic().expect("artifact still attached");
        assert!(
            std::str::from_utf8(&artifact.response_excerpt)
                .unwrap()
                .contains("502 Bad Gateway")
        );
    }

    #[test]
    fn build_service_tier_tunable_from_modern_payload() {
        // Modern shape: each tier carries id + name + description. The
        // synthetic Default variant is prepended so omitting the wire
        // field stays expressible from the UI.
        let body = r#"{
            "models": [{
                "slug": "gpt-5.4",
                "display_name": "GPT-5.4",
                "visibility": "list",
                "supported_in_api": true,
                "service_tiers": [
                    {"id": "priority", "name": "Fast", "description": "Priority queue."}
                ]
            }]
        }"#;
        let parsed: RspCodexModelsResponse = serde_json::from_str(body).unwrap();
        let m = &parsed.models[0];
        let spec = build_service_tier_tunable(m).expect("model advertises a tier");
        assert_eq!(spec.key, "service_tier");
        match spec.kind {
            TunableKind::Enum { variants, default } => {
                assert_eq!(default, ""); // synthetic Default selected
                assert_eq!(variants.len(), 2);
                assert_eq!(variants[0].value, "");
                assert_eq!(variants[1].value, "priority");
                assert_eq!(variants[1].label.as_deref(), Some("Fast"));
                assert_eq!(variants[1].description.as_deref(), Some("Priority queue."));
            }
            _ => panic!("expected Enum kind"),
        }
    }

    #[test]
    fn build_service_tier_tunable_legacy_additional_speed_tiers() {
        // Older metadata: only `additional_speed_tiers: ["fast"]`.
        // Codex's `supports_fast_mode()` treats this as equivalent to
        // a modern `priority` tier — we mirror that mapping so the UI
        // still surfaces the Fast option on grandfathered models.
        let body = r#"{
            "models": [{
                "slug": "gpt-5-legacy",
                "display_name": "GPT-5 (legacy)",
                "visibility": "list",
                "supported_in_api": true,
                "additional_speed_tiers": ["fast"]
            }]
        }"#;
        let parsed: RspCodexModelsResponse = serde_json::from_str(body).unwrap();
        let m = &parsed.models[0];
        let spec = build_service_tier_tunable(m).expect("legacy model still surfaces a tier");
        match spec.kind {
            TunableKind::Enum { variants, .. } => {
                assert_eq!(variants.len(), 2);
                assert_eq!(variants[1].value, "priority");
                assert_eq!(variants[1].label.as_deref(), Some("Fast"));
            }
            _ => panic!("expected Enum kind"),
        }
    }

    #[test]
    fn build_service_tier_tunable_skips_models_without_tiers() {
        // A model with neither service_tiers nor additional_speed_tiers
        // should produce no tunable — synthesizing a "Default-only"
        // control would be a dead-end UI surface.
        let body = r#"{
            "models": [{
                "slug": "gpt-4.1",
                "display_name": "GPT-4.1",
                "visibility": "list",
                "supported_in_api": true
            }]
        }"#;
        let parsed: RspCodexModelsResponse = serde_json::from_str(body).unwrap();
        let m = &parsed.models[0];
        assert!(build_service_tier_tunable(m).is_none());
    }

    #[test]
    fn pick_service_tier_omits_field_for_empty_and_unset() {
        // The synthetic Default variant lands as `Enum("")`; both that
        // and the absent-entirely case must elide the wire field.
        let mut t = std::collections::BTreeMap::new();
        assert_eq!(pick_service_tier(&t), None);
        t.insert("service_tier".into(), TunableValue::Enum(String::new()));
        assert_eq!(pick_service_tier(&t), None);
        t.insert("service_tier".into(), TunableValue::Enum("priority".into()));
        assert_eq!(pick_service_tier(&t), Some("priority"));
    }

    #[test]
    fn request_body_service_tier_serializes_only_when_present() {
        // Codex backend 400s on unknown tier strings, so the wire field
        // must stay absent for default-tier calls instead of riding as
        // null or empty string.
        let body = RspRequest {
            model: "gpt-5",
            instructions: None,
            input: Vec::new(),
            tools: None,
            include: REQUEST_INCLUDES,
            store: false,
            stream: true,
            service_tier: None,
            prompt_cache_key: None,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert!(v.get("service_tier").is_none(), "absent when None: {v}");

        let body = RspRequest {
            service_tier: Some("priority"),
            ..body
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["service_tier"], serde_json::json!("priority"));
    }

    #[test]
    fn request_body_prompt_cache_key_serializes_only_when_present() {
        // Mirrors the service_tier story: absent on the wire when None
        // so non-routing callers stay byte-stable against pre-field
        // request bodies. Present as a plain string when set.
        let body = RspRequest {
            model: "gpt-5",
            instructions: None,
            input: Vec::new(),
            tools: None,
            include: REQUEST_INCLUDES,
            store: false,
            stream: true,
            service_tier: None,
            prompt_cache_key: None,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert!(v.get("prompt_cache_key").is_none(), "absent when None: {v}");

        let body = RspRequest {
            prompt_cache_key: Some("task-18b03f4cbd87ae97"),
            ..body
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(
            v["prompt_cache_key"],
            serde_json::json!("task-18b03f4cbd87ae97")
        );
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
            service_tier: None,
            prompt_cache_key: None,
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
    fn image_generation_call_outbound_shape_matches_api_constraints() {
        // The Codex subscription API rejects output-only fields
        // (`action`, `background`, `output_format`, `quality`, `size`,
        // `revised_prompt`) when items are echoed back as input. The
        // empirical reference is `scripts/probe_responses_api.sh`. Lock
        // the wire shape so nobody adds these fields without remembering
        // why they're absent.
        let native = ProviderReplay {
            provider: "openai_responses".into(),
            data: serde_json::json!({
                "id": "ig_1",
                "status": "completed",
                "revised_prompt": "Edit the existing image…",
            }),
        };
        let source = whisper_agent_protocol::ImageSource::Bytes {
            media_type: whisper_agent_protocol::ImageMime::Png,
            data: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        let item =
            image_to_image_generation_call_item(&source, Some(&native)).expect("should echo");
        let v = serde_json::to_value(&item).unwrap();
        assert_eq!(v["type"], "image_generation_call");
        assert_eq!(v["id"], "ig_1");
        assert_eq!(v["status"], "completed");
        assert_eq!(
            v["result"], "3q2+7w==",
            "base64(0xDEADBEEF) lands in `result`"
        );
        // Everything output-only must be absent.
        for forbidden in [
            "action",
            "background",
            "output_format",
            "quality",
            "size",
            "revised_prompt",
        ] {
            assert!(
                v.get(forbidden).is_none(),
                "field `{forbidden}` must be stripped — API rejects it on input"
            );
        }
    }

    #[test]
    fn image_generation_call_outbound_drops_foreign_provider_replay() {
        let foreign = ProviderReplay {
            provider: "anthropic".into(),
            data: serde_json::json!({"id": "x"}),
        };
        let source = whisper_agent_protocol::ImageSource::Bytes {
            media_type: whisper_agent_protocol::ImageMime::Png,
            data: vec![1, 2, 3],
        };
        assert!(image_to_image_generation_call_item(&source, Some(&foreign)).is_none());
    }

    #[test]
    fn image_generation_call_outbound_skips_url_sources() {
        // URL-source images don't carry the bytes the API needs to put
        // in `result`; nothing useful to echo back.
        let native = ProviderReplay {
            provider: "openai_responses".into(),
            data: serde_json::json!({"id": "ig_x"}),
        };
        let source = whisper_agent_protocol::ImageSource::Url {
            url: "https://example.com/x.png".into(),
        };
        assert!(image_to_image_generation_call_item(&source, Some(&native)).is_none());
    }

    #[test]
    fn image_generation_call_inbound_decodes_base64_into_bytes() {
        // The full incoming-item path: SSE → finalize_response →
        // ContentBlock::Image with bytes + replay metadata.
        let raw = r#"{
            "type": "image_generation_call",
            "id": "ig_1",
            "status": "completed",
            "result": "3q2+7w==",
            "revised_prompt": "a cat"
        }"#;
        let item: RspOutputItem = serde_json::from_str(raw).expect("parse output item");
        let (content, _stop, _usage) =
            finalize_response(vec![item], Some("completed".into()), None, None).expect("finalize");
        assert_eq!(content.len(), 1);
        let ContentBlock::Image { source, replay } = &content[0] else {
            panic!("expected ContentBlock::Image, got {:?}", content[0]);
        };
        match source {
            whisper_agent_protocol::ImageSource::Bytes { media_type, data } => {
                assert_eq!(*media_type, whisper_agent_protocol::ImageMime::Png);
                assert_eq!(data, &vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("expected Bytes source, got {other:?}"),
        }
        let replay = replay.as_ref().expect("replay should be present");
        assert_eq!(replay.provider, "openai_responses");
        let obj = replay.data.as_object().unwrap();
        assert_eq!(obj.get("id").unwrap(), "ig_1");
        assert_eq!(obj.get("status").unwrap(), "completed");
        assert_eq!(obj.get("revised_prompt").unwrap(), "a cat");
    }

    #[test]
    fn image_generation_call_inbound_skips_partial_items_without_result() {
        // Mid-stream partials arrive with status="generating" and no
        // result yet; finalize_response must skip them rather than
        // failing the turn.
        let raw = r#"{
            "type": "image_generation_call",
            "id": "ig_1",
            "status": "generating"
        }"#;
        let item: RspOutputItem = serde_json::from_str(raw).expect("parse output item");
        let (content, _stop, _usage) =
            finalize_response(vec![item], Some("completed".into()), None, None).expect("finalize");
        assert!(content.is_empty(), "partial item must produce no block");
    }

    #[test]
    fn web_search_tool_renders_to_correct_wire_shape() {
        // Codex specifically rejects `web_search_preview` ("Unsupported
        // tool type"). Empirical reference: probe_responses_api.sh.
        let spec = ToolSpec {
            name: "web_search".into(),
            description: "(unused on the wire for built-ins)".into(),
            params: Vec::new(),
            kind: whisper_agent_protocol::ToolKind::ProviderBuiltin,
        };
        let v = serde_json::to_value(spec_to_rsp_tool(&spec)).unwrap();
        assert_eq!(v["type"], "web_search");
        for forbidden in ["name", "description", "parameters", "strict"] {
            assert!(
                v.get(forbidden).is_none(),
                "built-in tool wire shape must not carry `{forbidden}`"
            );
        }
    }

    #[test]
    fn web_search_call_is_dropped_from_conversation() {
        // The user-visible search results flow as citations in the
        // subsequent Message item; the call item itself adds nothing.
        // Make sure we don't accidentally surface it as a content
        // block (which would clutter the chat log).
        let raw = r#"{
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "queries": ["weather today"]}
        }"#;
        let item: RspOutputItem = serde_json::from_str(raw).expect("parse output item");
        let (content, _stop, _usage) =
            finalize_response(vec![item], Some("completed".into()), None, None).expect("finalize");
        assert!(content.is_empty(), "web_search_call must produce no block");
    }

    #[test]
    fn image_generation_tool_renders_to_correct_wire_shape() {
        let spec = ToolSpec {
            name: "image_generation".into(),
            description: "(unused on the wire for built-ins)".into(),
            params: Vec::new(),
            kind: whisper_agent_protocol::ToolKind::ProviderBuiltin,
        };
        let rendered = spec_to_rsp_tool(&spec);
        let v = serde_json::to_value(&rendered).unwrap();
        assert_eq!(v["type"], "image_generation");
        assert_eq!(v["size"], "auto");
        assert_eq!(v["quality"], "auto");
        // No function-call fields (the API rejects them on built-in tools).
        for forbidden in ["name", "description", "parameters", "strict"] {
            assert!(
                v.get(forbidden).is_none(),
                "built-in tool wire shape must not carry `{forbidden}`"
            );
        }
    }

    #[test]
    fn reasoning_item_serializes_summary_field_even_when_empty() {
        // Captured reasoning replays often carry id + encrypted_content
        // but no summary parts (a thinking turn that didn't surface
        // visible summary text). The Responses API requires `summary`
        // to be present on every reasoning input item — omitting it
        // 400s with "Missing required parameter: 'input[N].summary'."
        // Lock in the empty-array serialization so the field can never
        // disappear from the wire again.
        let native = ProviderReplay {
            provider: "openai_responses".into(),
            data: serde_json::json!({
                "id": "rs_42",
                "encrypted_content": "ec-blob",
            }),
        };
        let item = thinking_to_reasoning_item(Some(&native)).expect("should echo");
        let v = serde_json::to_value(&item).unwrap();
        assert_eq!(v["type"], "reasoning");
        assert_eq!(v["id"], "rs_42");
        assert_eq!(v["encrypted_content"], "ec-blob");
        let summary = v
            .get("summary")
            .expect("summary must be present on the wire");
        assert!(
            summary.is_array(),
            "summary must be an array, got {summary:?}"
        );
        assert_eq!(summary.as_array().unwrap().len(), 0);
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
        let forensic = ForensicContext {
            backend: PROVIDER_TAG.to_string(),
            model: "test".into(),
            request_body: String::new(),
            request_content_type: "application/json",
        };
        let buf = ForensicBuf::default();
        for ev_json in events {
            let ev: RspStreamEvent = serde_json::from_str(ev_json).unwrap();
            out.extend(state.consume(ev, &forensic, &buf).unwrap());
        }
        (out, state)
    }

    #[test]
    fn image_generation_call_done_emits_imageblock_for_live_streaming() {
        // Without a live ImageBlock event, the webui only sees the
        // image after the next snapshot refresh — the user reported a
        // chat row with no picture even though the persisted thread
        // had the bytes. Locking in the live-event emission so the
        // image appears as soon as the model finishes generating it.
        let events = [
            r#"{"type":"response.output_item.done","item":{
                "type":"image_generation_call","id":"ig_1",
                "status":"completed","result":"3q2+7w=="
            }}"#,
            r#"{"type":"response.completed","response":{
                "status":"completed","output":[],
                "usage":{"input_tokens":3,"output_tokens":1}
            }}"#,
        ];
        let (evs, _) = feed_events(&events);
        let images: Vec<&whisper_agent_protocol::ImageSource> = evs
            .iter()
            .filter_map(|e| match e {
                ModelEvent::ImageBlock { source } => Some(source),
                _ => None,
            })
            .collect();
        assert_eq!(images.len(), 1, "exactly one ImageBlock per generation");
        match images[0] {
            whisper_agent_protocol::ImageSource::Bytes { media_type, data } => {
                assert_eq!(*media_type, whisper_agent_protocol::ImageMime::Png);
                assert_eq!(data, &vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("expected Bytes source, got {other:?}"),
        }
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
