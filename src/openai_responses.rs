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
//! Reasoning replay is *not* round-tripped in this first cut. Reasoning items
//! returned by the model surface as [`ContentBlock::Thinking`] for the log, but
//! `encrypted_content` is not requested and reasoning items are not emitted on
//! subsequent turns. Adding replay means carrying a provider-tagged opaque blob
//! through our normalized `ContentBlock` shape, which is a separate change.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;
use whisper_agent_protocol::{ContentBlock, Message, Role, ToolResultContent, Usage};

use crate::codex_auth::CodexAuth;
use crate::model::{
    BoxFuture, ModelError, ModelInfo, ModelProvider, ModelRequest, ModelResponse, ToolSpec,
};

pub const OPENAI_API_BASE: &str = "https://api.openai.com/v1";
/// ChatGPT subscription route: reachable by ChatGPT Plus/Pro OAuth access tokens,
/// not by plain API keys. Serves the same Responses API surface at `/responses`.
pub const CHATGPT_CODEX_BASE: &str = "https://chatgpt.com/backend-api/codex";

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

    async fn do_create_message(&self, req: &ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
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
            max_output_tokens: req.max_tokens,
            store: false,
            stream: false,
        };

        let url = format!("{}/responses", self.base_url);
        let (bearer, extra_headers) = self.prepare_headers().await?;
        let mut builder = self.http.post(&url).bearer_auth(&bearer).json(&body);
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
        let parsed: RspResponse = resp
            .json()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;

        let mut content: Vec<ContentBlock> = Vec::new();
        let mut saw_function_call = false;
        for item in parsed.output {
            match item {
                RspOutputItem::Reasoning { content: parts, .. } => {
                    let text = parts
                        .into_iter()
                        .filter_map(|p| match p {
                            RspReasoningPart::ReasoningText { text }
                            | RspReasoningPart::SummaryText { text } => Some(text),
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    if !text.trim().is_empty() {
                        content.push(ContentBlock::Thinking {
                            signature: None,
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
                    });
                }
                RspOutputItem::Other => {}
            }
        }

        let stop_reason = derive_stop_reason(
            parsed.status.as_deref(),
            parsed.incomplete_details.as_ref(),
            saw_function_call,
        );

        let usage = parsed
            .usage
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

        Ok(ModelResponse {
            content,
            stop_reason,
            usage,
        })
    }

    async fn do_list_models(&self) -> Result<Vec<ModelInfo>, ModelError> {
        // ChatGPT-subscription backend doesn't expose /models — return a
        // hardcoded list of models that route known to work under that plan,
        // so the UI has something to pick.
        if matches!(self.auth, ClientAuth::Codex(_)) {
            return Ok(vec![
                ModelInfo { id: "gpt-5".into(), display_name: None },
                ModelInfo { id: "gpt-5-codex".into(), display_name: None },
                ModelInfo { id: "codex-mini-latest".into(), display_name: None },
            ]);
        }

        let url = format!("{}/models", self.base_url);
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

impl ModelProvider for OpenAiResponsesClient {
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

// ---------- Message → Item conversion ----------

fn convert_message(m: &Message, out: &mut Vec<RspItem>) {
    match m.role {
        Role::User => convert_user_message(&m.content, out),
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
            ContentBlock::ToolUse { id, name, input } => {
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
            ContentBlock::Thinking { .. } => {
                // Reasoning replay needs provider-tagged encrypted_content, which
                // we don't carry yet — drop silently on outbound.
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

// ---------- Wire types (private to this module) ----------

#[derive(Serialize, Debug)]
struct RspRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
    input: Vec<RspItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<RspTool>>,
    max_output_tokens: u32,
    store: bool,
    stream: bool,
}

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

#[derive(Deserialize, Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::Message;

    #[test]
    fn serializes_simple_user_message() {
        let mut items = Vec::new();
        convert_message(
            &Message::user_text("hello"),
            &mut items,
        );
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
            ContentBlock::Text {
                text: "ok".into(),
            },
            ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "shell".into(),
                input: serde_json::json!({"cmd": "ls"}),
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
    fn stop_reason_max_tokens() {
        let body = r#"{
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }"#;
        let resp: RspResponse = serde_json::from_str(body).unwrap();
        assert_eq!(
            derive_stop_reason(resp.status.as_deref(), resp.incomplete_details.as_ref(), false),
            Some("max_tokens".into())
        );
    }
}
