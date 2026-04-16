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
//!   - [`ContentBlock::Thinking`] → dropped (not representable).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use whisper_agent_protocol::{ContentBlock, Message, Role, ToolResultContent, Usage};

use crate::model::{
    BoxFuture, ModelError, ModelInfo, ModelProvider, ModelRequest, ModelResponse, ToolSpec,
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
        if let Some(text) = choice.message.content.and_then(content_as_text)
            && !text.is_empty()
        {
            content.push(ContentBlock::Text { text });
        }
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let input: Value = if tc.function.arguments.is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&tc.function.arguments).map_err(|e| {
                        ModelError::Transport(format!(
                            "tool_call arguments not valid JSON: {e}"
                        ))
                    })?
                };
                content.push(ContentBlock::ToolUse {
                    id: tc.id,
                    name: tc.function.name,
                    input,
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

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<ModelInfo>, ModelError>> {
        Box::pin(self.do_list_models())
    }
}

// ---------- Message conversion ----------

fn convert_message(m: &Message, out: &mut Vec<OaMessage>) {
    match m.role {
        Role::User => convert_user_message(&m.content, out),
        Role::Assistant => convert_assistant_message(&m.content, out),
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
            ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                // Flush any accumulated plain text first so ordering survives.
                if !text_accum.is_empty() {
                    out.push(OaMessage {
                        role: "user".into(),
                        content: Some(OaContent::Text(std::mem::take(&mut text_accum))),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                let text = tool_result_as_text(content, *is_error);
                out.push(OaMessage {
                    role: "tool".into(),
                    content: Some(OaContent::Text(text)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                });
            }
            ContentBlock::ToolUse { .. } | ContentBlock::Thinking { .. } => {
                // Neither makes sense in a user message. Drop silently.
            }
        }
    }
    if !text_accum.is_empty() {
        out.push(OaMessage {
            role: "user".into(),
            content: Some(OaContent::Text(text_accum)),
            tool_calls: None,
            tool_call_id: None,
        });
    }
}

fn convert_assistant_message(blocks: &[ContentBlock], out: &mut Vec<OaMessage>) {
    let mut text_accum = String::new();
    let mut tool_calls: Vec<OaToolCall> = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                if !text_accum.is_empty() {
                    text_accum.push('\n');
                }
                text_accum.push_str(text);
            }
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(OaToolCall {
                    id: id.clone(),
                    kind: "function".into(),
                    function: OaFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                    },
                });
            }
            ContentBlock::Thinking { .. } | ContentBlock::ToolResult { .. } => {
                // Thinking: no OpenAI equivalent. ToolResult: doesn't belong on assistant.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_text_then_tool_uses_converts_to_one_message() {
        let blocks = vec![
            ContentBlock::Text { text: "let me check".into() },
            ContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "foo.txt"}),
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
        }];
        let mut out = Vec::new();
        convert_assistant_message(&blocks, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out[0].content.is_none());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert!(v.get("content").is_none(), "tool_call-only turn must not carry content field");
    }

    #[test]
    fn is_error_prefixes_tool_result_text() {
        let t = tool_result_as_text(
            &ToolResultContent::Text("boom".into()),
            true,
        );
        assert_eq!(t, "ERROR: boom");
    }
}
