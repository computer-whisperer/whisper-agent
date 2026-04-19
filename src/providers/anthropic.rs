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
use whisper_agent_protocol::{ContentBlock, Message, ProviderReplay, ToolResultContent, Usage};

use crate::providers::model::{
    BoxFuture, CacheBreakpoint, ModelError, ModelInfo, ModelProvider, ModelRequest, ModelResponse,
    ToolSpec,
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

/// Serialize a message and optionally patch `cache_control` onto its final
/// content block.
///
/// Beyond the serde derive, we also fix up two things the wire expects:
/// - `Role::ToolResult` → `user` on the role field (Anthropic only accepts
///   `user` and `assistant`).
/// - `Thinking.replay` → wire-level `signature` field when the block was
///   minted by this backend; stripped otherwise (a blob tagged for another
///   provider is meaningless and the server 400s on unknown fields).
fn message_to_value(m: &Message, cache_last_block: bool) -> Value {
    let mut v = serde_json::to_value(m).expect("Message is serializable");
    if let Some(role) = v.get_mut("role")
        && role.as_str() == Some("tool_result")
    {
        *role = Value::String("user".into());
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
}
