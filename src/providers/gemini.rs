//! Google Gemini native backend.
//!
//! Speaks the `generateContent` wire shape used by both
//! `generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
//! (API-key auth) and, later, the Code Assist route at
//! `cloudcode-pa.googleapis.com/v1internal:generateContent` (OAuth-auth).
//!
//! This first cut covers API-key auth only; the OAuth route stacks on top in a
//! follow-up that reuses the same [`build_gemini_request`] /
//! [`ContentGenerationResponse`] conversion layer.
//!
//! Wire shape highlights that differ from the Chat Completions / Responses shapes
//! elsewhere in the codebase:
//!
//! - A `Content` has a `role` ("user" | "model") and a `parts` array. Each part is
//!   one of `{text}`, `{functionCall}`, `{functionResponse}`, `{thought: true,
//!   text}`.
//! - Tool calls are `functionCall` parts (no opaque call_id — we synthesize one
//!   and round-trip it through our normalized [`ContentBlock::ToolUse`]).
//! - Tool results are `functionResponse` parts, keyed *by function name*. The
//!   caller has to remember which name each tool_use_id corresponds to.
//! - The system prompt is carried in a dedicated top-level `systemInstruction`
//!   field, not as a role=system content entry.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_stream::try_stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex, OnceCell};
use tokio_util::sync::CancellationToken;
use whisper_agent_protocol::{
    ContentBlock, Message, ProviderReplay, Role, ToolResultContent, Usage,
};

use crate::providers::gemini_auth::GeminiAuth;
use crate::providers::model::{
    BoxFuture, BoxStream, ModelError, ModelEvent, ModelInfo, ModelProvider, ModelRequest,
    ModelResponse, ToolSpec,
};

pub const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";
/// Code Assist route — reachable via gemini-cli OAuth tokens but not by a
/// plain AI Studio API key. Serves the same `generateContent` primitive
/// wrapped in a `{model, project, request, user_prompt_id}` envelope.
pub const GEMINI_CODE_ASSIST_BASE: &str = "https://cloudcode-pa.googleapis.com/v1internal";

/// Identifier stored on [`ProviderReplay::provider`] for blobs minted by this
/// backend. Used on outbound to filter out replay data tagged for a different
/// provider (e.g. an Anthropic `signature` would be meaningless here).
const PROVIDER_TAG: &str = "gemini";

/// Prefix for the opaque call_id we synthesize for each Gemini `functionCall`
/// part. Gemini doesn't emit an id; the scheduler requires one to pair
/// tool_use / tool_result blocks. Choice of prefix is informational only —
/// whatever we emit gets round-tripped verbatim through the conversation log.
const GENERATED_CALL_ID_PREFIX: &str = "gemini-call-";
static CALL_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_call_id() -> String {
    let n = CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{GENERATED_CALL_ID_PREFIX}{n:x}")
}

/// Runtime auth for a Gemini client. `ApiKey` talks to the public AI Studio
/// endpoint; `GeminiCli` talks to the Code Assist endpoint using tokens from
/// `~/.gemini/oauth_creds.json`, refreshed on demand.
pub enum ClientAuth {
    ApiKey(String),
    GeminiCli {
        auth: Arc<Mutex<GeminiAuth>>,
        /// Project ID discovered via `:loadCodeAssist`. Populated lazily on
        /// first request so construction stays cheap and errors surface only
        /// when the user actually tries to use the provider.
        project: Arc<OnceCell<String>>,
    },
}

pub struct GeminiClient {
    http: reqwest::Client,
    base_url: String,
    auth: ClientAuth,
}

impl GeminiClient {
    pub fn with_api_key(base_url: String, api_key: String) -> Self {
        Self::new(base_url, ClientAuth::ApiKey(api_key))
    }

    pub fn with_gemini_cli_auth(base_url: String, auth: GeminiAuth) -> Self {
        Self::new(
            base_url,
            ClientAuth::GeminiCli {
                auth: Arc::new(Mutex::new(auth)),
                project: Arc::new(OnceCell::new()),
            },
        )
    }

    fn new(base_url: String, auth: ClientAuth) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            http: reqwest::Client::new(),
            base_url,
            auth,
        }
    }

    /// Refresh tokens if needed and return the current bearer for Code Assist
    /// requests. API-key auth never uses this path — the key goes in the URL.
    async fn oauth_bearer(&self) -> Result<String, ModelError> {
        match &self.auth {
            ClientAuth::ApiKey(_) => Err(ModelError::Transport(
                "oauth_bearer called on api-key auth".into(),
            )),
            ClientAuth::GeminiCli { auth, .. } => {
                let mut guard = auth.lock().await;
                guard
                    .ensure_fresh(&self.http)
                    .await
                    .map_err(|e| ModelError::Transport(format!("gemini auth refresh: {e}")))?;
                Ok(guard.access_token().to_string())
            }
        }
    }

    /// Resolve the Code Assist project id, calling `:loadCodeAssist` once and
    /// caching the result. Honours `GOOGLE_CLOUD_PROJECT` so users with a
    /// specific GCP project can route through it without an env-var on every
    /// launch.
    async fn codex_project(&self) -> Result<String, ModelError> {
        let ClientAuth::GeminiCli { project, .. } = &self.auth else {
            return Err(ModelError::Transport(
                "codex_project called on non-oauth auth".into(),
            ));
        };
        project
            .get_or_try_init(|| async {
                if let Ok(env_proj) = std::env::var("GOOGLE_CLOUD_PROJECT")
                    && !env_proj.is_empty()
                {
                    return Ok(env_proj);
                }
                let bearer = self.oauth_bearer().await?;
                let body = LoadCodeAssistRequest {
                    cloudaicompanion_project: None,
                    metadata: LoadCodeAssistMetadata {
                        ide_type: "IDE_UNSPECIFIED",
                        platform: "PLATFORM_UNSPECIFIED",
                        plugin_type: "GEMINI",
                    },
                };
                let url = format!("{}:loadCodeAssist", self.base_url);
                let resp = self
                    .http
                    .post(&url)
                    .bearer_auth(&bearer)
                    .json(&body)
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
                let parsed: LoadCodeAssistResponse = resp
                    .json()
                    .await
                    .map_err(|e| ModelError::Transport(e.to_string()))?;
                parsed.cloudaicompanion_project.ok_or_else(|| {
                    // User hasn't completed onboarding with gemini-cli. We don't
                    // implement the onboardUser flow ourselves; point them at the
                    // tool that does.
                    ModelError::Transport(
                        "Code Assist has no project for this account — run `gemini` once \
                         interactively to complete onboarding, then retry."
                            .into(),
                    )
                })
            })
            .await
            .cloned()
    }

    async fn do_create_message(
        &self,
        req: &ModelRequest<'_>,
        cancel: &CancellationToken,
    ) -> Result<ModelResponse, ModelError> {
        let inner = build_gemini_request(req);
        match &self.auth {
            ClientAuth::ApiKey(key) => {
                let url = format!(
                    "{}/models/{}:generateContent?key={}",
                    self.base_url,
                    req.model,
                    urlencode(key)
                );
                let send = self.http.post(&url).json(&inner).send();
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
                    return Err(classify_http_error(status.as_u16(), body));
                }
                let parsed: GenerateContentResponse = tokio::select! {
                    r = resp.json() => r.map_err(|e| ModelError::Transport(e.to_string()))?,
                    _ = cancel.cancelled() => return Err(ModelError::Cancelled),
                };
                Ok(parsed_to_model_response(parsed))
            }
            ClientAuth::GeminiCli { .. } => {
                let project = tokio::select! {
                    r = self.codex_project() => r?,
                    _ = cancel.cancelled() => return Err(ModelError::Cancelled),
                };
                let bearer = tokio::select! {
                    r = self.oauth_bearer() => r?,
                    _ = cancel.cancelled() => return Err(ModelError::Cancelled),
                };
                let envelope = CaGenerateContentRequest {
                    model: req.model,
                    project,
                    request: inner,
                };
                let url = format!("{}:generateContent", self.base_url);
                let send = self
                    .http
                    .post(&url)
                    .bearer_auth(&bearer)
                    .json(&envelope)
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
                    return Err(classify_http_error(status.as_u16(), body));
                }
                let wrapped: CaGenerateContentResponse = tokio::select! {
                    r = resp.json() => r.map_err(|e| ModelError::Transport(e.to_string()))?,
                    _ = cancel.cancelled() => return Err(ModelError::Cancelled),
                };
                let inner = wrapped.response.ok_or_else(|| {
                    ModelError::Transport(
                        "Code Assist response missing `response` envelope field".into(),
                    )
                })?;
                Ok(parsed_to_model_response(inner))
            }
        }
    }

    fn do_create_message_streaming<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        Box::pin(try_stream! {
            let inner = build_gemini_request(req);
            let resp = match &self.auth {
                ClientAuth::ApiKey(key) => {
                    let url = format!(
                        "{}/models/{}:streamGenerateContent?alt=sse&key={}",
                        self.base_url,
                        req.model,
                        urlencode(key)
                    );
                    let send = self.http.post(&url).json(&inner).send();
                    tokio::select! {
                        r = send => r.map_err(|e| ModelError::Transport(e.to_string())),
                        _ = cancel.cancelled() => Err(ModelError::Cancelled),
                    }?
                }
                ClientAuth::GeminiCli { .. } => {
                    let project = tokio::select! {
                        r = self.codex_project() => r,
                        _ = cancel.cancelled() => Err(ModelError::Cancelled),
                    }?;
                    let bearer = tokio::select! {
                        r = self.oauth_bearer() => r,
                        _ = cancel.cancelled() => Err(ModelError::Cancelled),
                    }?;
                    let envelope = CaGenerateContentRequest {
                        model: req.model,
                        project,
                        request: inner,
                    };
                    // The Code Assist backend uses `alt=sse` as a query arg on
                    // the same `streamGenerateContent` verb — envelope shape
                    // is identical to the unary path.
                    let url = format!("{}:streamGenerateContent?alt=sse", self.base_url);
                    let send = self.http
                        .post(&url)
                        .bearer_auth(&bearer)
                        .json(&envelope)
                        .send();
                    tokio::select! {
                        r = send => r.map_err(|e| ModelError::Transport(e.to_string())),
                        _ = cancel.cancelled() => Err(ModelError::Cancelled),
                    }?
                }
            };
            let status = resp.status();
            if !status.is_success() {
                let err_body = resp.text().await.unwrap_or_default();
                Err(classify_http_error(status.as_u16(), err_body))?;
                return;
            }
            let codex_mode = matches!(self.auth, ClientAuth::GeminiCli { .. });
            let mut bytes = resp.bytes_stream();
            let mut sse_buf: Vec<u8> = Vec::new();
            let mut state = GeminiStreamState::default();
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
                sse_buf.extend_from_slice(&chunk);
                while let Some(event_payload) = take_sse_event(&mut sse_buf) {
                    let Some(raw) = parse_sse_data(&event_payload) else {
                        continue;
                    };
                    // Code Assist wraps each chunk in `{response: {...}}`; the
                    // public API sends the bare GenerateContentResponse.
                    let parsed: GenerateContentResponse = if codex_mode {
                        let wrapped: CaGenerateContentResponse = serde_json::from_str(&raw)
                            .map_err(|e| ModelError::Transport(e.to_string()))?;
                        match wrapped.response {
                            Some(r) => r,
                            None => continue, // empty wrapper — skip
                        }
                    } else {
                        serde_json::from_str(&raw)
                            .map_err(|e| ModelError::Transport(e.to_string()))?
                    };
                    for out in state.consume(parsed) {
                        yield out;
                    }
                }
            }
            // Gemini doesn't emit a dedicated terminator event — the HTTP
            // stream closes once the final chunk is delivered. Flush any
            // remaining accumulated state as the terminal Completed.
            for out in state.finish() {
                yield out;
            }
        })
    }

    async fn do_list_models(&self) -> Result<Vec<ModelInfo>, ModelError> {
        match &self.auth {
            ClientAuth::ApiKey(key) => {
                let url = format!("{}/models?key={}", self.base_url, urlencode(key));
                let resp = self
                    .http
                    .get(&url)
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
                    .models
                    .into_iter()
                    .filter(|m| {
                        // Surface only models that actually support generateContent —
                        // /v1beta/models also lists embedding, tuning, and audio models.
                        m.supported_generation_methods
                            .iter()
                            .any(|s| s == "generateContent")
                    })
                    .map(|m| ModelInfo {
                        // Trim the "models/" prefix so the id matches what users type
                        // into `default_model` in config.
                        id: m
                            .name
                            .strip_prefix("models/")
                            .unwrap_or(&m.name)
                            .to_string(),
                        display_name: m.display_name,
                        context_window: m.input_token_limit,
                        max_output_tokens: m.output_token_limit,
                        capabilities: crate::providers::model::gemini_capabilities_for(
                            m.name.strip_prefix("models/").unwrap_or(&m.name),
                        ),
                    })
                    .collect())
            }
            ClientAuth::GeminiCli { .. } => {
                // Code Assist doesn't expose a `/models` catalog, so we mirror
                // the set gemini-cli ships in VALID_GEMINI_MODELS. Users pick
                // via config; `gemini-3-*-preview` variants require preview
                // access on the account (the server returns a clear 403 if
                // not). Context/output caps unknown on this route — the
                // API-key route reports them but Code Assist doesn't.
                let entry = |id: &str, name: &str| ModelInfo {
                    id: id.into(),
                    display_name: Some(name.into()),
                    context_window: None,
                    max_output_tokens: None,
                    capabilities: crate::providers::model::gemini_capabilities_for(id),
                };
                Ok(vec![
                    entry("gemini-3-pro-preview", "Gemini 3 Pro (preview)"),
                    entry("gemini-3.1-pro-preview", "Gemini 3.1 Pro (preview)"),
                    entry("gemini-3-flash-preview", "Gemini 3 Flash (preview)"),
                    entry(
                        "gemini-3.1-flash-lite-preview",
                        "Gemini 3.1 Flash Lite (preview)",
                    ),
                    entry("gemini-2.5-pro", "Gemini 2.5 Pro"),
                    entry("gemini-2.5-flash", "Gemini 2.5 Flash"),
                    entry("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite"),
                ])
            }
        }
    }
}

impl ModelProvider for GeminiClient {
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

// ---------- Request construction ----------

pub(crate) fn build_gemini_request(req: &ModelRequest<'_>) -> GenerateContentRequest {
    // Gemini identifies tool results by function name, but our ToolResult
    // carries only tool_use_id. Pre-scan the conversation for a tool_use →
    // name map so we can look up the name when we hit the result.
    let id_to_name = build_tool_use_name_index(req.messages);

    let mut contents: Vec<Content> = Vec::with_capacity(req.messages.len());
    for m in req.messages {
        if let Some(c) = convert_message(m, &id_to_name) {
            contents.push(c);
        }
    }

    let system_instruction = if req.system_prompt.is_empty() {
        None
    } else {
        Some(Content {
            role: None,
            parts: vec![Part::Text {
                text: req.system_prompt.to_string(),
                thought: None,
            }],
        })
    };

    let tools = if req.tools.is_empty() {
        None
    } else {
        Some(vec![Tool {
            function_declarations: req.tools.iter().map(spec_to_fn_decl).collect(),
        }])
    };

    GenerateContentRequest {
        contents,
        system_instruction,
        tools,
        generation_config: Some(GenerationConfig {
            max_output_tokens: Some(req.max_tokens),
            // Image-output models (`gemini-2.5-flash-image*`,
            // `gemini-2.0-flash-preview-image-generation`) require
            // `responseModalities: ["TEXT", "IMAGE"]` — calling them
            // without it 400s. We send it whenever the model id
            // matches the heuristic, never otherwise (text-only
            // Gemini models *also* 400 if the field is set to
            // include IMAGE). Other Gemini models leave the field
            // omitted and stay text-only.
            response_modalities: if model_emits_images(req.model) {
                Some(vec!["TEXT".to_string(), "IMAGE".to_string()])
            } else {
                None
            },
        }),
    }
}

/// Whether a Gemini model id is one of the image-output variants.
/// Gemini's `/v1beta/models` response carries no explicit
/// supportsImageOutput flag, so we match by the well-known names.
/// Kept intentionally narrow — false positives here mean
/// `responseModalities` lands on a text-only model and 400s the
/// request, so the safer default is "no" for unfamiliar ids.
pub(crate) fn model_emits_images(model: &str) -> bool {
    // Match the live image-output names. The `-image` suffix and
    // `-image-generation` substring catch both the previewed and the
    // GA-tagged variants without listing every revision.
    model.contains("-image-generation") || model.contains("-image")
}

fn build_tool_use_name_index(messages: &[Message]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for m in messages {
        for block in &m.content {
            if let ContentBlock::ToolUse { id, name, .. } = block {
                map.insert(id.clone(), name.clone());
            }
        }
    }
    map
}

fn convert_message(m: &Message, id_to_name: &HashMap<String, String>) -> Option<Content> {
    let (role, parts) = match m.role {
        // Gemini has no `tool` role on the wire — function responses
        // ride inside a `user`-role Content with FunctionResponse
        // parts. `Role::ToolResult` uses the same part-builder as
        // `Role::User`; `convert_user_parts` already knows how to
        // turn `ContentBlock::ToolResult` into a FunctionResponse.
        Role::User | Role::ToolResult => ("user", convert_user_parts(&m.content, id_to_name)),
        Role::Assistant => ("model", convert_assistant_parts(&m.content, id_to_name)),
        // Mid-conversation harness injection. The thread-prefix system
        // prompt is lifted into `systemInstruction` upstream; anything
        // that reaches here is a later injection (e.g. memory-index
        // block). Gemini has no mid-conversation system role, so emit
        // as a `user` Content with each text part wrapped in
        // `<system-reminder>` tags — the same convention Anthropic
        // uses for the equivalent case.
        Role::System => ("user", convert_system_parts(&m.content)),
        // Tool-manifest setup messages are filtered out upstream and
        // lifted into the `tools` wire field. Dropping a stray one
        // yields no parts → the caller's `None` path skips the entry.
        Role::Tools => ("user", Vec::new()),
    };
    if parts.is_empty() {
        None
    } else {
        Some(Content {
            role: Some(role.to_string()),
            parts,
        })
    }
}

/// Build Gemini `parts` for a mid-conversation `Role::System` injection.
/// Each text block's content is wrapped in `<system-reminder>` tags so the
/// model can distinguish harness guidance from real user speech when the
/// Content lands on the wire as `role: "user"`. Non-text blocks are
/// unexpected in a system injection and dropped.
fn convert_system_parts(blocks: &[ContentBlock]) -> Vec<Part> {
    let mut parts = Vec::new();
    for block in blocks {
        if let ContentBlock::Text { text } = block {
            parts.push(Part::Text {
                text: format!("<system-reminder>\n{text}\n</system-reminder>"),
                thought: None,
            });
        }
    }
    parts
}

fn convert_user_parts(blocks: &[ContentBlock], id_to_name: &HashMap<String, String>) -> Vec<Part> {
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                parts.push(Part::Text {
                    text: text.clone(),
                    thought: None,
                });
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let name = id_to_name
                    .get(tool_use_id)
                    .cloned()
                    .unwrap_or_else(|| "unknown_tool".to_string());
                let text = tool_result_as_text(content, *is_error);
                // Gemini's functionResponse takes a structured object, not a
                // plain string. Wrap under "content" so we preserve arbitrary
                // tool output (including error markers) without forcing the
                // tool output to fit a model-chosen schema.
                parts.push(Part::FunctionResponse {
                    function_response: FunctionResponse {
                        name,
                        response: serde_json::json!({ "content": text }),
                    },
                });
                // Tool-result images ride as additional inline_data
                // parts on the same user turn — Gemini's
                // functionResponse.response field is structured JSON
                // (not multipart), so attachments can't nest inside it.
                // Appending them as sibling parts keeps them attached
                // to the same turn and preserves order.
                for src in content.image_sources() {
                    if let Some(part) = image_source_to_part(src) {
                        parts.push(part);
                    }
                }
            }
            ContentBlock::Image { source } => {
                if let Some(part) = image_source_to_part(source) {
                    parts.push(part);
                }
            }
            ContentBlock::ToolUse { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::ToolSchema { .. } => {
                // Not valid on a user turn — drop silently.
            }
        }
    }
    parts
}

fn convert_assistant_parts(
    blocks: &[ContentBlock],
    _id_to_name: &HashMap<String, String>,
) -> Vec<Part> {
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                parts.push(Part::Text {
                    text: text.clone(),
                    thought: None,
                });
            }
            ContentBlock::ToolUse {
                id: _,
                name,
                input,
                replay,
            } => {
                // We drop the id when serializing: Gemini's wire format has no
                // call_id on functionCall. The id we generated on the inbound
                // side lives in our scheduler log so the eventual tool_result
                // can be paired; Gemini itself pairs by function name.
                //
                // If the stored replay came from a Gemini turn, echo its
                // thought_signature on the functionCall part — this is how
                // Gemini carries chain-of-thought across the tool-result
                // boundary. Blobs tagged for a different backend are dropped
                // to avoid cross-provider signature contamination.
                let thought_signature = replay
                    .as_ref()
                    .filter(|r| r.provider == PROVIDER_TAG)
                    .and_then(|r| r.data.get("thought_signature"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                parts.push(Part::FunctionCall {
                    function_call: FunctionCall {
                        name: name.clone(),
                        args: input.clone(),
                    },
                    thought_signature,
                });
            }
            ContentBlock::Thinking { thinking, replay } => {
                // Thought-summary text goes back as a `thought:true` text part
                // — Gemini accepts its own `thought:true` text parts on
                // assistant turns and uses them (together with the
                // thought_signature on the following functionCall) to resume
                // chain-of-thought. Replay data tagged for a different backend
                // (e.g. Anthropic `signature`) is dropped; the prose is still
                // useful so the text itself stays.
                let _ = replay; // Gemini's signature rides on ToolUse, not here.
                parts.push(Part::Text {
                    text: thinking.clone(),
                    thought: Some(true),
                });
            }
            ContentBlock::ToolResult { .. } | ContentBlock::ToolSchema { .. } => {
                // Neither is valid on an assistant turn.
            }
            ContentBlock::Image { source } => {
                // Re-emit native-image output blocks that were minted
                // by Gemini on a prior turn so a follow-up assistant
                // call sees the image it already produced. URL-source
                // images on assistant turns are unusual but reuse the
                // same lowering as user-side images (Gemini still
                // can't fetch URLs, so they degrade to a text note).
                if let Some(part) = image_source_to_part(source) {
                    parts.push(part);
                }
            }
        }
    }
    parts
}

/// Parse an inbound `inlineData` part into a normalized
/// [`ContentBlock::Image`]. Used by both the non-streaming and
/// streaming response parsers when the model emits a native image
/// (gemini-2.5-flash-image*, gemini-2.0-flash-preview-image-generation).
/// Returns `None` if the MIME isn't in our accepted set or the
/// payload fails to base64-decode — we'd rather drop a malformed
/// attachment than panic the assistant turn.
fn inline_data_to_image_block(
    inline_data: &InlineData,
) -> Option<whisper_agent_protocol::ContentBlock> {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use whisper_agent_protocol::{ContentBlock, ImageMime, ImageSource};
    let media_type = ImageMime::from_mime_str(&inline_data.mime_type).or_else(|| {
        tracing::warn!(
            mime = %inline_data.mime_type,
            "Gemini emitted an image with a MIME we don't carry; dropping"
        );
        None
    })?;
    let bytes = STANDARD
        .decode(&inline_data.data)
        .inspect_err(|e| {
            tracing::warn!(error = %e, "Gemini inlineData base64 decode failed; dropping");
        })
        .ok()?;
    Some(ContentBlock::Image {
        source: ImageSource::Bytes {
            media_type,
            data: bytes,
        },
    })
}

/// Lower one of our image-source values into a Gemini `Part`. Bytes
/// land as `inlineData` with a base64-encoded body; URL sources are
/// dropped with a warning and surfaced to the model as a text part
/// noting the missing attachment, since Gemini's wire format doesn't
/// accept URL passthrough (only inline bytes or Files-API URIs). URL
/// support through a fetch-and-inline pass is a v1.1 follow-up.
fn image_source_to_part(src: &whisper_agent_protocol::ImageSource) -> Option<Part> {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use whisper_agent_protocol::ImageSource;
    match src {
        ImageSource::Bytes { media_type, data } => Some(Part::InlineData {
            inline_data: InlineData {
                mime_type: media_type.as_mime_str().to_string(),
                data: STANDARD.encode(data),
            },
        }),
        ImageSource::Url { url } => {
            tracing::warn!(
                url = %url,
                "Gemini does not support URL-sourced images; dropping attachment and \
                 substituting a text note. Fetch-and-inline will be added post-v1."
            );
            Some(Part::Text {
                text: format!("[Image URL attached but not viewable on this model: {url}]"),
                thought: None,
            })
        }
    }
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

fn spec_to_fn_decl(t: &ToolSpec) -> FunctionDeclaration {
    FunctionDeclaration {
        name: t.name.clone(),
        description: t.description.clone(),
        parameters: t.input_schema.clone(),
    }
}

// ---------- Response conversion ----------

pub(crate) fn parsed_to_model_response(parsed: GenerateContentResponse) -> ModelResponse {
    let mut content: Vec<ContentBlock> = Vec::new();
    let mut saw_function_call = false;
    let candidate = parsed.candidates.into_iter().next();
    let finish_reason = candidate.as_ref().and_then(|c| c.finish_reason.clone());

    if let Some(cand) = candidate
        && let Some(cand_content) = cand.content
    {
        for part in cand_content.parts {
            match part {
                Part::Text {
                    text,
                    thought: Some(true),
                } => {
                    if !text.is_empty() {
                        // Gemini's signature rides on the subsequent
                        // functionCall part, so Thinking blocks don't carry
                        // replay data here — it lands on the ToolUse below.
                        content.push(ContentBlock::Thinking {
                            replay: None,
                            thinking: text,
                        });
                    }
                }
                Part::Text { text, .. } => {
                    if !text.is_empty() {
                        content.push(ContentBlock::Text { text });
                    }
                }
                Part::FunctionCall {
                    function_call,
                    thought_signature,
                } => {
                    saw_function_call = true;
                    let replay = thought_signature.map(|sig| ProviderReplay {
                        provider: PROVIDER_TAG.into(),
                        data: serde_json::json!({"thought_signature": sig}),
                    });
                    content.push(ContentBlock::ToolUse {
                        id: next_call_id(),
                        name: function_call.name,
                        input: function_call.args,
                        replay,
                    });
                }
                Part::FunctionResponse { .. } => {
                    // Models don't emit function responses — ignore if echoed.
                }
                Part::InlineData { inline_data } => {
                    // Native image output (gemini-2.5-flash-image*,
                    // gemini-2.0-flash-preview-image-generation). The
                    // inline_data carries base64 + IANA mime; we
                    // normalize into our `ImageSource::Bytes` so the
                    // assistant turn round-trips through conversation
                    // storage and the UI renderer alongside any text /
                    // tool-use parts emitted in the same response.
                    if let Some(block) = inline_data_to_image_block(&inline_data) {
                        content.push(block);
                    }
                }
            }
        }
    }

    let stop_reason = derive_stop_reason(finish_reason.as_deref(), saw_function_call);

    let usage = parsed
        .usage_metadata
        .map(|u| Usage {
            input_tokens: u.prompt_token_count.unwrap_or(0),
            // Gemini breaks its output into "candidates" (text/tools) and
            // "thoughts" (hidden reasoning). Both count toward billing, so
            // surface the combined total.
            output_tokens: u.candidates_token_count.unwrap_or(0)
                + u.thoughts_token_count.unwrap_or(0),
            cache_read_input_tokens: u.cached_content_token_count.unwrap_or(0),
            cache_creation_input_tokens: 0,
        })
        .unwrap_or_default();

    ModelResponse {
        content,
        stop_reason,
        usage,
    }
}

/// Map Gemini's `finishReason` values onto the Anthropic-style stop_reason
/// labels the rest of the codebase consumes. `tool_use` wins when any
/// function call appeared — Gemini sometimes reports STOP alongside tool
/// calls, which would otherwise hide them from the scheduler loop.
fn derive_stop_reason(finish_reason: Option<&str>, saw_function_call: bool) -> Option<String> {
    if saw_function_call {
        return Some("tool_use".into());
    }
    match finish_reason {
        Some("STOP") => Some("end_turn".into()),
        Some("MAX_TOKENS") => Some("max_tokens".into()),
        Some(other) => Some(other.to_lowercase()),
        None => None,
    }
}

// ---------- Streaming SSE parse + state machine ----------

/// Pull one complete SSE event (terminated by a blank line) out of `buf`.
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

/// Currently-open block in a Gemini stream. Unlike Anthropic's explicit
/// content_block_start/stop, Gemini leaves block boundaries implicit — we
/// infer them by tracking whether the last part was text (and its
/// thought:true flag) vs something else (functionCall, end of stream).
#[derive(Debug)]
enum GeminiOpenBlock {
    Text(String),
    Thinking(String),
}

/// Running state for one Gemini streaming call. Each SSE chunk is a full
/// `GenerateContentResponse` with a handful of new `parts` — the state
/// machine folds them into text / thinking accumulators, emits deltas as
/// they arrive, and assembles the final content list for `Completed`.
#[derive(Default)]
struct GeminiStreamState {
    open: Option<GeminiOpenBlock>,
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    saw_function_call: bool,
    usage: Option<Usage>,
}

impl GeminiStreamState {
    fn consume(&mut self, chunk: GenerateContentResponse) -> Vec<ModelEvent> {
        let mut out = Vec::new();

        // Capture finish_reason + usage opportunistically. They typically
        // arrive on the same terminal chunk, but either can slip earlier
        // (e.g. usage on a progress chunk) — keep the latest value we see.
        if let Some(cand) = chunk.candidates.first()
            && let Some(fr) = cand.finish_reason.as_deref()
        {
            self.stop_reason = Some(fr.to_string());
        }
        if let Some(u) = chunk.usage_metadata {
            self.usage = Some(Usage {
                input_tokens: u.prompt_token_count.unwrap_or(0),
                output_tokens: u.candidates_token_count.unwrap_or(0)
                    + u.thoughts_token_count.unwrap_or(0),
                cache_read_input_tokens: u.cached_content_token_count.unwrap_or(0),
                cache_creation_input_tokens: 0,
            });
        }

        let candidate = chunk.candidates.into_iter().next();
        let Some(cand) = candidate else {
            return out;
        };
        let Some(cand_content) = cand.content else {
            return out;
        };

        for part in cand_content.parts {
            match part {
                Part::Text {
                    text,
                    thought: Some(true),
                } => {
                    if !self.matches_open_thinking() {
                        self.close_open();
                        self.open = Some(GeminiOpenBlock::Thinking(String::new()));
                    }
                    if let Some(GeminiOpenBlock::Thinking(buf)) = &mut self.open {
                        buf.push_str(&text);
                    }
                    if !text.is_empty() {
                        out.push(ModelEvent::ThinkingDelta { text });
                    }
                }
                Part::Text { text, .. } => {
                    if !self.matches_open_text() {
                        self.close_open();
                        self.open = Some(GeminiOpenBlock::Text(String::new()));
                    }
                    if let Some(GeminiOpenBlock::Text(buf)) = &mut self.open {
                        buf.push_str(&text);
                    }
                    if !text.is_empty() {
                        out.push(ModelEvent::TextDelta { text });
                    }
                }
                Part::FunctionCall {
                    function_call,
                    thought_signature,
                } => {
                    self.close_open();
                    self.saw_function_call = true;
                    let replay = thought_signature.map(|sig| ProviderReplay {
                        provider: PROVIDER_TAG.into(),
                        data: serde_json::json!({"thought_signature": sig}),
                    });
                    let id = next_call_id();
                    out.push(ModelEvent::ToolCall {
                        id: id.clone(),
                        name: function_call.name.clone(),
                        input: function_call.args.clone(),
                    });
                    self.content.push(ContentBlock::ToolUse {
                        id,
                        name: function_call.name,
                        input: function_call.args,
                        replay,
                    });
                }
                Part::FunctionResponse { .. } => {
                    // Models don't emit function responses — ignore if
                    // echoed back to us.
                }
                Part::InlineData { inline_data } => {
                    // Native image output (gemini-2.5-flash-image*,
                    // gemini-2.0-flash-preview-image-generation). Close
                    // any open text/thinking block so the image lands
                    // in document order, persist it on the assembled
                    // turn content for the terminal `Completed` event,
                    // and emit a live `ImageBlock` so connected webuis
                    // can render the thumbnail without waiting on a
                    // snapshot rebuild.
                    self.close_open();
                    if let Some(ContentBlock::Image { source }) =
                        inline_data_to_image_block(&inline_data)
                    {
                        out.push(ModelEvent::ImageBlock {
                            source: source.clone(),
                        });
                        self.content.push(ContentBlock::Image { source });
                    }
                }
            }
        }
        out
    }

    fn matches_open_text(&self) -> bool {
        matches!(self.open, Some(GeminiOpenBlock::Text(_)))
    }

    fn matches_open_thinking(&self) -> bool {
        matches!(self.open, Some(GeminiOpenBlock::Thinking(_)))
    }

    fn close_open(&mut self) {
        match self.open.take() {
            Some(GeminiOpenBlock::Text(text)) if !text.is_empty() => {
                self.content.push(ContentBlock::Text { text });
            }
            Some(GeminiOpenBlock::Thinking(text)) if !text.is_empty() => {
                // Gemini's thought_signature rides on the sibling
                // functionCall part, not here — leave replay None.
                self.content.push(ContentBlock::Thinking {
                    replay: None,
                    thinking: text,
                });
            }
            _ => {}
        }
    }

    /// Flush remaining open block and emit the terminal `Completed`. The
    /// Gemini stream has no explicit terminator event — the HTTP body just
    /// closes — so the caller invokes this once `bytes_stream()` drains.
    fn finish(mut self) -> Vec<ModelEvent> {
        self.close_open();
        let stop_reason = derive_stop_reason(self.stop_reason.as_deref(), self.saw_function_call);
        vec![ModelEvent::Completed {
            content: self.content,
            stop_reason,
            usage: self.usage.unwrap_or_default(),
        }]
    }
}

// ---------- Misc helpers ----------

/// Map a non-2xx HTTP response into the right ModelError flavor. 429s from
/// Code Assist include a human-readable "reset after Ns" phrase; we pull that
/// duration out so the dispatch layer can decide whether to wait-and-retry
/// transparently.
pub(crate) fn classify_http_error(status: u16, body: String) -> ModelError {
    if status == 429 {
        let retry_after = parse_retry_after_seconds(&body)
            .map(std::time::Duration::from_secs)
            .unwrap_or_else(|| std::time::Duration::from_secs(10));
        ModelError::RateLimited { retry_after, body }
    } else {
        ModelError::Api { status, body }
    }
}

/// Extract the first `N` from phrases like "Your quota will reset after 6s."
/// — Google doesn't put this in a header, but always includes it in the
/// error detail message. Returns None if no match is found.
fn parse_retry_after_seconds(body: &str) -> Option<u64> {
    const NEEDLE: &str = "reset after ";
    let idx = body.find(NEEDLE)?;
    let tail = &body[idx + NEEDLE.len()..];
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// Minimal URL-encoder for the API key (the only value we embed in a URL).
/// Keeping it inline avoids pulling urlencoding as a direct dep.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => out.push(ch),
            _ => {
                for b in ch.to_string().as_bytes() {
                    out.push_str(&format!("%{b:02X}"));
                }
            }
        }
    }
    out
}

// ---------- Wire types ----------

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerateContentRequest {
    pub(crate) contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) generation_config: Option<GenerationConfig>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Content {
    /// `None` on a `systemInstruction`; `Some("user" | "model")` elsewhere.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) role: Option<String>,
    pub(crate) parts: Vec<Part>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum Part {
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCall,
        /// Server-signed reasoning blob attached to function-call parts on
        /// thinking-capable models. Must be echoed back on the assistant
        /// turn's functionCall for chain-of-thought to resume after the
        /// corresponding tool_result. `camelCase` on the wire.
        #[serde(
            rename = "thoughtSignature",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponse,
    },
    /// Inline base64-encoded image (or other media) data. Gemini's
    /// `inlineData` part accepts user-supplied images as content in a
    /// user turn and, on native-image-output models, also appears in
    /// response parts. V1 uses it only on the outbound (user) side;
    /// inbound parsing isn't wired up yet.
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    /// `thought: Some(true)` marks a reasoning part (Gemini 2.x thinking models).
    /// Serialization skips the field when `None` so we don't poke older models
    /// that don't recognize it.
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        thought: Option<bool>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct InlineData {
    #[serde(rename = "mimeType")]
    pub(crate) mime_type: String,
    /// Base64-encoded bytes. Gemini accepts the raw base64 string
    /// here (no `data:` prefix, unlike OpenAI's data URL form).
    pub(crate) data: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct FunctionCall {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) args: Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct FunctionResponse {
    pub(crate) name: String,
    pub(crate) response: Value,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Tool {
    pub(crate) function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Serialize, Debug)]
pub(crate) struct FunctionDeclaration {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: Value,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<u32>,
    /// Tells Gemini which response kinds the model is allowed to
    /// emit — `["TEXT"]` is the default, image-output models require
    /// `["TEXT", "IMAGE"]`. Omitted when `None` so text-only models
    /// don't see the field at all (passing IMAGE on a model that
    /// doesn't support it is a 400 from the API).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) response_modalities: Option<Vec<String>>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerateContentResponse {
    #[serde(default)]
    pub(crate) candidates: Vec<Candidate>,
    #[serde(default)]
    pub(crate) usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Candidate {
    #[serde(default)]
    pub(crate) content: Option<Content>,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UsageMetadata {
    #[serde(default)]
    pub(crate) prompt_token_count: Option<u32>,
    #[serde(default)]
    pub(crate) candidates_token_count: Option<u32>,
    #[serde(default)]
    pub(crate) thoughts_token_count: Option<u32>,
    #[serde(default)]
    pub(crate) cached_content_token_count: Option<u32>,
}

// ---------- Code Assist (OAuth route) envelope ----------

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CaGenerateContentRequest<'a> {
    model: &'a str,
    project: String,
    request: GenerateContentRequest,
}

#[derive(Deserialize, Debug)]
struct CaGenerateContentResponse {
    #[serde(default)]
    response: Option<GenerateContentResponse>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct LoadCodeAssistRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    cloudaicompanion_project: Option<String>,
    metadata: LoadCodeAssistMetadata,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct LoadCodeAssistMetadata {
    ide_type: &'static str,
    platform: &'static str,
    plugin_type: &'static str,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct LoadCodeAssistResponse {
    #[serde(default)]
    cloudaicompanion_project: Option<String>,
}

// ---------- /v1beta/models (API-key route) ----------

#[derive(Deserialize, Debug)]
struct ListModelsResponse {
    #[serde(default)]
    models: Vec<ListedModel>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct ListedModel {
    name: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
    /// Max input tokens the model accepts. Gemini's `/v1beta/models`
    /// publishes this as `inputTokenLimit`.
    #[serde(default)]
    input_token_limit: Option<u32>,
    /// Max output tokens the model will generate. Published as
    /// `outputTokenLimit`.
    #[serde(default)]
    output_token_limit: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req<'a>(messages: &'a [Message], tools: &'a [ToolSpec], model: &'a str) -> ModelRequest<'a> {
        ModelRequest {
            model,
            max_tokens: 1024,
            system_prompt: "",
            tools,
            messages,
            cache_breakpoints: &[],
        }
    }

    #[test]
    fn user_image_bytes_become_inline_data_part() {
        use whisper_agent_protocol::{ImageMime, ImageSource};
        let msg = Message::user_blocks(vec![
            ContentBlock::Text {
                text: "describe".into(),
            },
            ContentBlock::Image {
                source: ImageSource::Bytes {
                    media_type: ImageMime::Webp,
                    data: vec![0x52, 0x49, 0x46, 0x46],
                },
            },
        ]);
        let id_to_name = HashMap::new();
        let parts = convert_user_parts(&msg.content, &id_to_name);
        assert_eq!(parts.len(), 2);
        let json = serde_json::to_value(&parts).unwrap();
        assert_eq!(json[0]["text"], "describe");
        let inline = &json[1]["inlineData"];
        assert_eq!(inline["mimeType"], "image/webp");
        // 4 bytes "RIFF" → base64 "UklGRg=="
        assert_eq!(inline["data"], "UklGRg==");
    }

    #[test]
    fn tool_result_with_image_appends_inline_data_part() {
        // Gemini's functionResponse.response is structured JSON, not
        // multipart, so tool-result images ride as additional sibling
        // inline_data parts on the same user turn alongside the
        // functionResponse.
        use whisper_agent_protocol::{ImageMime, ImageSource, ToolResultContent};
        let blocks = vec![ContentBlock::ToolResult {
            tool_use_id: "call_7".into(),
            content: ToolResultContent::Blocks(vec![
                ContentBlock::Text {
                    text: "screenshot:".into(),
                },
                ContentBlock::Image {
                    source: ImageSource::Bytes {
                        media_type: ImageMime::Png,
                        data: vec![137, 80, 78, 71],
                    },
                },
            ]),
            is_error: false,
        }];
        let mut id_to_name = HashMap::new();
        id_to_name.insert("call_7".to_string(), "shell".to_string());
        let parts = convert_user_parts(&blocks, &id_to_name);
        // [functionResponse, inlineData]
        assert_eq!(parts.len(), 2);
        let json = serde_json::to_value(&parts).unwrap();
        assert_eq!(json[0]["functionResponse"]["name"], "shell");
        assert_eq!(
            json[0]["functionResponse"]["response"]["content"],
            "screenshot:"
        );
        assert_eq!(json[1]["inlineData"]["mimeType"], "image/png");
    }

    #[test]
    fn inbound_inline_data_lowers_to_assistant_image_block() {
        // Round-trip the wire shape an image-output Gemini model emits:
        // a candidate with one inlineData part (mime image/png, base64
        // body) should land in our assistant content as a normalized
        // ContentBlock::Image { ImageSource::Bytes }.
        use base64::Engine;
        use base64::engine::general_purpose::STANDARD;
        let png = vec![137, 80, 78, 71, 13, 10, 26, 10];
        let parsed = GenerateContentResponse {
            candidates: vec![Candidate {
                content: Some(Content {
                    role: Some("model".into()),
                    parts: vec![
                        Part::Text {
                            text: "here you go:".into(),
                            thought: None,
                        },
                        Part::InlineData {
                            inline_data: InlineData {
                                mime_type: "image/png".into(),
                                data: STANDARD.encode(&png),
                            },
                        },
                    ],
                }),
                finish_reason: None,
            }],
            usage_metadata: None,
        };
        let resp = parsed_to_model_response(parsed);
        assert_eq!(resp.content.len(), 2);
        assert!(matches!(&resp.content[0], ContentBlock::Text { text } if text == "here you go:"));
        let ContentBlock::Image {
            source: whisper_agent_protocol::ImageSource::Bytes { data, media_type },
        } = &resp.content[1]
        else {
            panic!("expected Image block, got {:?}", resp.content[1]);
        };
        assert_eq!(*data, png);
        assert_eq!(*media_type, whisper_agent_protocol::ImageMime::Png);
    }

    #[test]
    fn response_modalities_present_for_image_output_models() {
        // Image-output models require responseModalities:
        // ["TEXT","IMAGE"] in generationConfig — without it the API
        // 400s with "model only supports image output mode."
        let req = req(&[], &[], "gemini-2.5-flash-image");
        let body = build_gemini_request(&req);
        let json = serde_json::to_value(&body).unwrap();
        let modalities = json["generationConfig"]["responseModalities"]
            .as_array()
            .expect("responseModalities should be present");
        let kinds: Vec<&str> = modalities.iter().filter_map(|v| v.as_str()).collect();
        assert!(kinds.contains(&"TEXT"));
        assert!(kinds.contains(&"IMAGE"));
    }

    #[test]
    fn response_modalities_omitted_for_text_only_models() {
        // Setting IMAGE on a non-image model is a 400; field must be
        // omitted entirely so text-only models stay unaffected.
        let r = req(&[], &[], "gemini-2.5-flash");
        let body = build_gemini_request(&r);
        let json = serde_json::to_value(&body).unwrap();
        assert!(json["generationConfig"].get("responseModalities").is_none());
    }

    #[test]
    fn assistant_image_block_round_trips_as_inline_data() {
        // A persisted assistant turn that carries a model-emitted image
        // (e.g. the prior turn of a multi-turn image-edit flow) should
        // re-emit the image as an inlineData part on the wire so the
        // model sees what it produced.
        use whisper_agent_protocol::{ImageMime, ImageSource};
        let blocks = vec![
            ContentBlock::Text {
                text: "here:".into(),
            },
            ContentBlock::Image {
                source: ImageSource::Bytes {
                    media_type: ImageMime::Png,
                    data: vec![137, 80, 78, 71],
                },
            },
        ];
        let parts = convert_assistant_parts(&blocks, &HashMap::new());
        assert_eq!(parts.len(), 2);
        let json = serde_json::to_value(&parts).unwrap();
        assert_eq!(json[0]["text"], "here:");
        assert_eq!(json[1]["inlineData"]["mimeType"], "image/png");
    }

    #[test]
    fn user_image_url_becomes_text_fallback_on_gemini() {
        // Gemini doesn't accept URL passthrough, so URL-sourced images
        // get substituted for a text note until fetch-and-inline lands.
        use whisper_agent_protocol::ImageSource;
        let msg = Message::user_blocks(vec![ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.com/cat.png".into(),
            },
        }]);
        let parts = convert_user_parts(&msg.content, &HashMap::new());
        assert_eq!(parts.len(), 1);
        let Part::Text { text, .. } = &parts[0] else {
            panic!("expected text fallback, got {:?}", parts[0]);
        };
        assert!(text.contains("https://example.com/cat.png"));
    }

    #[test]
    fn list_models_parses_input_and_output_token_limits() {
        // Gemini's /v1beta/models returns camelCase `inputTokenLimit`
        // and `outputTokenLimit` per model. Confirm they land on the
        // ListedModel struct so do_list_models can forward them into
        // ModelInfo.context_window / max_output_tokens.
        let json = r#"{
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                    "inputTokenLimit": 1048576,
                    "outputTokenLimit": 8192
                }
            ]
        }"#;
        let parsed: ListModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.models.len(), 1);
        let m = &parsed.models[0];
        assert_eq!(m.input_token_limit, Some(1048576));
        assert_eq!(m.output_token_limit, Some(8192));
    }

    #[test]
    fn list_models_omits_token_limits_gracefully() {
        // Older / partial responses may lack the limit fields — they
        // must deserialize to None rather than erroring.
        let json = r#"{
            "models": [
                {
                    "name": "models/legacy",
                    "supportedGenerationMethods": ["generateContent"]
                }
            ]
        }"#;
        let parsed: ListModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.models[0].input_token_limit, None);
        assert_eq!(parsed.models[0].output_token_limit, None);
    }

    #[test]
    fn user_text_becomes_single_content() {
        let msgs = vec![Message::user_text("hello")];
        let body = build_gemini_request(&req(&msgs, &[], "gemini-2.5-pro"));
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["contents"],
            serde_json::json!([{"role": "user", "parts": [{"text": "hello"}]}])
        );
    }

    #[test]
    fn system_prompt_goes_to_system_instruction() {
        let msgs = vec![Message::user_text("x")];
        let mut r = req(&msgs, &[], "gemini-2.5-pro");
        r.system_prompt = "be helpful";
        let body = build_gemini_request(&r);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["systemInstruction"],
            serde_json::json!({"parts": [{"text": "be helpful"}]})
        );
    }

    #[test]
    fn mid_conversation_system_becomes_user_with_wrapped_text() {
        // Injected Role::System is harness guidance, not the thread-prefix
        // prompt (that one is lifted into `systemInstruction` upstream).
        // Gemini has no mid-conversation system role, so emit as role:user
        // with each text part wrapped in <system-reminder> tags.
        let msgs = vec![Message::system_text("check memory/ before answering")];
        let body = build_gemini_request(&req(&msgs, &[], "gemini-2.5-pro"));
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["contents"],
            serde_json::json!([{
                "role": "user",
                "parts": [{"text": "<system-reminder>\ncheck memory/ before answering\n</system-reminder>"}]
            }])
        );
    }

    #[test]
    fn assistant_tool_use_serializes_as_function_call() {
        let msgs = vec![Message::assistant_blocks(vec![
            ContentBlock::Text { text: "ok".into() },
            ContentBlock::ToolUse {
                id: "gemini-call-0".into(),
                name: "shell".into(),
                input: serde_json::json!({"cmd": "ls"}),
                replay: None,
            },
        ])];
        let body = build_gemini_request(&req(&msgs, &[], "x"));
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["contents"],
            serde_json::json!([{
                "role": "model",
                "parts": [
                    {"text": "ok"},
                    {"functionCall": {"name": "shell", "args": {"cmd": "ls"}}}
                ]
            }])
        );
    }

    #[test]
    fn tool_result_resolves_name_from_prior_tool_use() {
        let msgs = vec![
            Message::assistant_blocks(vec![ContentBlock::ToolUse {
                id: "gemini-call-0".into(),
                name: "shell".into(),
                input: serde_json::json!({}),
                replay: None,
            }]),
            Message::user_blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "gemini-call-0".into(),
                content: ToolResultContent::Text("done".into()),
                is_error: false,
            }]),
        ];
        let body = build_gemini_request(&req(&msgs, &[], "x"));
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["contents"][1],
            serde_json::json!({
                "role": "user",
                "parts": [{"functionResponse": {"name": "shell", "response": {"content": "done"}}}]
            })
        );
    }

    #[test]
    fn parses_text_plus_function_call_response() {
        let body = r#"{
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "ok"},
                    {"functionCall": {"name": "shell", "args": {"cmd": "ls"}}}
                ]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4}
        }"#;
        let parsed: GenerateContentResponse = serde_json::from_str(body).unwrap();
        let r = parsed_to_model_response(parsed);
        assert_eq!(r.content.len(), 2);
        assert!(matches!(&r.content[0], ContentBlock::Text { text } if text == "ok"));
        assert!(
            matches!(&r.content[1], ContentBlock::ToolUse { name, id, .. }
                if name == "shell" && id.starts_with(GENERATED_CALL_ID_PREFIX))
        );
        assert_eq!(r.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(r.usage.input_tokens, 10);
        assert_eq!(r.usage.output_tokens, 4);
    }

    #[test]
    fn parses_thinking_part() {
        let body = r#"{
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "hmm, let me think", "thought": true},
                    {"text": "answer"}
                ]},
                "finishReason": "STOP"
            }]
        }"#;
        let parsed: GenerateContentResponse = serde_json::from_str(body).unwrap();
        let r = parsed_to_model_response(parsed);
        assert_eq!(r.content.len(), 2);
        assert!(
            matches!(&r.content[0], ContentBlock::Thinking { thinking, .. } if thinking == "hmm, let me think")
        );
        assert!(matches!(&r.content[1], ContentBlock::Text { text } if text == "answer"));
        assert_eq!(r.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn parses_thought_signature_into_toolu_replay() {
        // A functionCall part carries thoughtSignature as a sibling field
        // on Gemini's wire. Inbound, that blob must land on the ToolUse
        // block's replay (tagged "gemini"), not on the Thinking block.
        let body = r#"{
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "planning the shell call", "thought": true},
                    {
                        "functionCall": {"name": "shell", "args": {"cmd": "ls"}},
                        "thoughtSignature": "sig-blob"
                    }
                ]},
                "finishReason": "STOP"
            }]
        }"#;
        let parsed: GenerateContentResponse = serde_json::from_str(body).unwrap();
        let r = parsed_to_model_response(parsed);
        assert_eq!(r.content.len(), 2);
        // Thinking part — signature is NOT here.
        match &r.content[0] {
            ContentBlock::Thinking { replay, thinking } => {
                assert_eq!(thinking, "planning the shell call");
                assert!(replay.is_none());
            }
            _ => panic!("expected Thinking"),
        }
        // ToolUse part — signature lives HERE, tagged "gemini".
        match &r.content[1] {
            ContentBlock::ToolUse { replay, name, .. } => {
                assert_eq!(name, "shell");
                let r = replay.as_ref().expect("signature should populate replay");
                assert_eq!(r.provider, "gemini");
                assert_eq!(r.data, serde_json::json!({"thought_signature": "sig-blob"}));
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn outbound_echoes_gemini_thought_signature_and_strips_foreign() {
        // Round-trip: a ToolUse carrying a gemini-tagged replay must put
        // thoughtSignature back on the wire as a sibling of functionCall.
        // A foreign-provider replay must be dropped silently.
        let msgs = vec![Message::assistant_blocks(vec![
            ContentBlock::ToolUse {
                id: "gemini-call-0".into(),
                name: "shell".into(),
                input: serde_json::json!({"cmd": "ls"}),
                replay: Some(ProviderReplay {
                    provider: "gemini".into(),
                    data: serde_json::json!({"thought_signature": "sig-blob"}),
                }),
            },
            ContentBlock::ToolUse {
                id: "gemini-call-1".into(),
                name: "shell".into(),
                input: serde_json::json!({"cmd": "pwd"}),
                replay: Some(ProviderReplay {
                    provider: "anthropic".into(),
                    data: serde_json::json!({"signature": "anth-blob"}),
                }),
            },
        ])];
        let body = build_gemini_request(&req(&msgs, &[], "x"));
        let v = serde_json::to_value(&body).unwrap();
        let parts = &v["contents"][0]["parts"];
        // Native blob echoed.
        assert_eq!(parts[0]["thoughtSignature"], "sig-blob");
        // Foreign blob stripped.
        assert!(parts[1].get("thoughtSignature").is_none());
    }

    #[test]
    fn stop_reason_max_tokens_mapping() {
        assert_eq!(
            derive_stop_reason(Some("MAX_TOKENS"), false),
            Some("max_tokens".into())
        );
        assert_eq!(
            derive_stop_reason(Some("STOP"), false),
            Some("end_turn".into())
        );
        // Function call wins over any finish reason.
        assert_eq!(
            derive_stop_reason(Some("STOP"), true),
            Some("tool_use".into())
        );
    }

    #[test]
    fn classify_429_carries_parsed_retry_after() {
        let body = r#"{"error":{"code":429,"message":"You have exhausted your capacity on this model. Your quota will reset after 6s.","status":"RESOURCE_EXHAUSTED"}}"#;
        match classify_http_error(429, body.to_string()) {
            ModelError::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, std::time::Duration::from_secs(6));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn classify_429_falls_back_to_default_retry_after_when_message_opaque() {
        let body = r#"{"error":{"code":429,"message":"too many requests"}}"#;
        match classify_http_error(429, body.to_string()) {
            ModelError::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, std::time::Duration::from_secs(10));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn classify_non_429_stays_api_error() {
        match classify_http_error(500, "boom".into()) {
            ModelError::Api { status, body } => {
                assert_eq!(status, 500);
                assert_eq!(body, "boom");
            }
            other => panic!("expected Api, got {other:?}"),
        }
    }

    #[test]
    fn urlencode_preserves_unreserved_and_escapes_specials() {
        assert_eq!(urlencode("abcXYZ-_.~"), "abcXYZ-_.~");
        assert_eq!(urlencode("a b"), "a%20b");
        assert_eq!(urlencode("k=v&q"), "k%3Dv%26q");
    }

    // ---------- Streaming state machine ----------

    fn feed_chunks(chunks: &[&str]) -> Vec<ModelEvent> {
        let mut state = GeminiStreamState::default();
        let mut out = Vec::new();
        for raw in chunks {
            let parsed: GenerateContentResponse = serde_json::from_str(raw).unwrap();
            out.extend(state.consume(parsed));
        }
        out.extend(state.finish());
        out
    }

    #[test]
    fn text_parts_across_chunks_produce_deltas_and_single_text_block() {
        // Each streaming chunk is a full GenerateContentResponse carrying
        // incremental text. Consecutive text parts with the same thought
        // flag join into one block.
        let chunks = [
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"hello "}]}}]}"#,
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"world"}]}}]}"#,
            r#"{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}"#,
        ];
        let evs = feed_chunks(&chunks);
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
                assert_eq!(usage.input_tokens, 5);
                assert_eq!(usage.output_tokens, 2);
            }
            _ => panic!("last event must be Completed"),
        }
    }

    #[test]
    fn thought_then_function_call_produces_distinct_blocks_with_replay() {
        // Thinking prose streams as thought:true text parts; the
        // subsequent functionCall part carries thought_signature and
        // should land on ToolUse.replay (not Thinking).
        let chunks = [
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"planning","thought":true}]}}]}"#,
            r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"shell","args":{"cmd":"ls"}},"thoughtSignature":"sig-blob"}]},"finishReason":"STOP"}]}"#,
        ];
        let evs = feed_chunks(&chunks);
        // First event: ThinkingDelta.
        assert!(matches!(&evs[0], ModelEvent::ThinkingDelta { text } if text == "planning"));
        // Second event: ToolCall (function_call assembles fully in one chunk).
        match &evs[1] {
            ModelEvent::ToolCall { name, input, .. } => {
                assert_eq!(name, "shell");
                assert_eq!(input, &serde_json::json!({"cmd": "ls"}));
            }
            _ => panic!("expected ToolCall second"),
        }
        // Terminal Completed carries both blocks; thought_signature on the
        // ToolUse.replay.
        match evs.last().unwrap() {
            ModelEvent::Completed {
                content,
                stop_reason,
                ..
            } => {
                assert_eq!(content.len(), 2);
                match &content[0] {
                    ContentBlock::Thinking { replay, thinking } => {
                        assert_eq!(thinking, "planning");
                        assert!(replay.is_none()); // signature rides on ToolUse
                    }
                    _ => panic!("expected Thinking first"),
                }
                match &content[1] {
                    ContentBlock::ToolUse { replay, name, .. } => {
                        assert_eq!(name, "shell");
                        let r = replay.as_ref().expect("thought_signature must surface");
                        assert_eq!(r.provider, "gemini");
                        assert_eq!(r.data, serde_json::json!({"thought_signature": "sig-blob"}));
                    }
                    _ => panic!("expected ToolUse second"),
                }
                assert_eq!(stop_reason.as_deref(), Some("tool_use"));
            }
            _ => panic!("expected Completed"),
        }
    }
}
