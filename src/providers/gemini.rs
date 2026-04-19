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

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex, OnceCell};
use whisper_agent_protocol::{
    ContentBlock, Message, ProviderReplay, Role, ToolResultContent, Usage,
};

use crate::providers::gemini_auth::GeminiAuth;
use crate::providers::model::{
    BoxFuture, ModelError, ModelInfo, ModelProvider, ModelRequest, ModelResponse, ToolSpec,
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

    async fn do_create_message(&self, req: &ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        let inner = build_gemini_request(req);
        match &self.auth {
            ClientAuth::ApiKey(key) => {
                // Non-streaming: we don't expose a streaming trait at the provider
                // layer, and the public Gemini endpoint supports the unary variant
                // without the `stream: true` requirement that the ChatGPT backend
                // imposes.
                let url = format!(
                    "{}/models/{}:generateContent?key={}",
                    self.base_url,
                    req.model,
                    urlencode(key)
                );
                let resp = self
                    .http
                    .post(&url)
                    .json(&inner)
                    .send()
                    .await
                    .map_err(|e| ModelError::Transport(e.to_string()))?;
                let status = resp.status();
                if !status.is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(classify_http_error(status.as_u16(), body));
                }
                let parsed: GenerateContentResponse = resp
                    .json()
                    .await
                    .map_err(|e| ModelError::Transport(e.to_string()))?;
                Ok(parsed_to_model_response(parsed))
            }
            ClientAuth::GeminiCli { .. } => {
                let project = self.codex_project().await?;
                let bearer = self.oauth_bearer().await?;
                let envelope = CaGenerateContentRequest {
                    model: req.model,
                    project,
                    request: inner,
                };
                let url = format!("{}:generateContent", self.base_url);
                let resp = self
                    .http
                    .post(&url)
                    .bearer_auth(&bearer)
                    .json(&envelope)
                    .send()
                    .await
                    .map_err(|e| ModelError::Transport(e.to_string()))?;
                let status = resp.status();
                if !status.is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(classify_http_error(status.as_u16(), body));
                }
                let wrapped: CaGenerateContentResponse = resp
                    .json()
                    .await
                    .map_err(|e| ModelError::Transport(e.to_string()))?;
                let inner = wrapped.response.ok_or_else(|| {
                    ModelError::Transport(
                        "Code Assist response missing `response` envelope field".into(),
                    )
                })?;
                Ok(parsed_to_model_response(inner))
            }
        }
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
                    })
                    .collect())
            }
            ClientAuth::GeminiCli { .. } => {
                // Code Assist doesn't expose a `/models` catalog, so we mirror
                // the set gemini-cli ships in VALID_GEMINI_MODELS. Users pick
                // via config; `gemini-3-*-preview` variants require preview
                // access on the account (the server returns a clear 403 if
                // not).
                Ok(vec![
                    ModelInfo {
                        id: "gemini-3-pro-preview".into(),
                        display_name: Some("Gemini 3 Pro (preview)".into()),
                    },
                    ModelInfo {
                        id: "gemini-3.1-pro-preview".into(),
                        display_name: Some("Gemini 3.1 Pro (preview)".into()),
                    },
                    ModelInfo {
                        id: "gemini-3-flash-preview".into(),
                        display_name: Some("Gemini 3 Flash (preview)".into()),
                    },
                    ModelInfo {
                        id: "gemini-3.1-flash-lite-preview".into(),
                        display_name: Some("Gemini 3.1 Flash Lite (preview)".into()),
                    },
                    ModelInfo {
                        id: "gemini-2.5-pro".into(),
                        display_name: Some("Gemini 2.5 Pro".into()),
                    },
                    ModelInfo {
                        id: "gemini-2.5-flash".into(),
                        display_name: Some("Gemini 2.5 Flash".into()),
                    },
                    ModelInfo {
                        id: "gemini-2.5-flash-lite".into(),
                        display_name: Some("Gemini 2.5 Flash Lite".into()),
                    },
                ])
            }
        }
    }
}

impl ModelProvider for GeminiClient {
    fn create_message<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
    ) -> BoxFuture<'a, Result<ModelResponse, ModelError>> {
        Box::pin(self.do_create_message(req))
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
        }),
    }
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
            }
            ContentBlock::ToolUse { .. } | ContentBlock::Thinking { .. } => {
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
            ContentBlock::ToolResult { .. } => {
                // Not valid on an assistant turn.
            }
        }
    }
    parts
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
}
