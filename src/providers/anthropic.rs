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

use serde::{Deserialize, Serialize};
use serde_json::Value;
use whisper_agent_protocol::{ContentBlock, Message, Usage};

use crate::providers::model::{
    BoxFuture, CacheBreakpoint, ModelError, ModelInfo, ModelProvider, ModelRequest, ModelResponse,
    ToolSpec,
};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const MODELS_URL: &str = "https://api.anthropic.com/v1/models";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const EXTENDED_CACHE_BETA: &str = "extended-cache-ttl-2025-04-11";
const CACHE_TTL_1H: &str = "1h";

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

    async fn do_create_message(&self, req: &ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        let body = build_request_body(req);
        let resp = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", EXTENDED_CACHE_BETA)
            .header("content-type", "application/json")
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
        let parsed: MessageResponse = resp
            .json()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        Ok(ModelResponse {
            content: parsed.content,
            stop_reason: parsed.stop_reason,
            usage: Usage {
                input_tokens: parsed.usage.input_tokens,
                output_tokens: parsed.usage.output_tokens,
                cache_read_input_tokens: parsed.usage.cache_read_input_tokens,
                cache_creation_input_tokens: parsed.usage.cache_creation_input_tokens,
            },
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
            })
            .collect())
    }
}

impl ModelProvider for AnthropicClient {
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
    }
}

/// Serialize a message and optionally patch `cache_control` onto its final content
/// block. Using `serde_json::Value` here avoids duplicating the full [`ContentBlock`]
/// hierarchy in this module just to add one optional field.
fn message_to_value(m: &Message, cache_last_block: bool) -> Value {
    let mut v = serde_json::to_value(m).expect("Message is serializable");
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
    content: Vec<ContentBlock>,
    #[allow(dead_code)]
    model: String,
    stop_reason: Option<String>,
    #[allow(dead_code)]
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize, Debug, Clone, Copy, Default)]
struct AnthropicUsage {
    input_tokens: u32,
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
    fn full_policy_caches_system_tools_and_trailing_user_messages() {
        let tools = make_tools();
        let messages = make_messages();
        let breakpoints = crate::providers::model::default_cache_policy(&messages, 2);
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
        // system + tools both cached.
        assert!(v["system"][0]["cache_control"].is_object());
        assert!(v["tools"].as_array().unwrap().last().unwrap()["cache_control"].is_object());
        // The two user messages (indices 0 and 2) both carry cache_control on
        // their final content block.
        let msgs = v["messages"].as_array().unwrap();
        assert!(msgs[0]["content"][0]["cache_control"].is_object());
        assert!(msgs[1]["content"][0].get("cache_control").is_none()); // assistant
        assert!(msgs[2]["content"][0]["cache_control"].is_object());
    }
}
