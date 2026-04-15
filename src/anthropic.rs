//! Anthropic Messages + Models API client.
//!
//! Implements [`ModelProvider`] for the `https://api.anthropic.com` endpoints. All
//! Anthropic-specific wire types (system-block cache_control, tool schema shape,
//! `MessageResponse` raw usage) are module-private — external code sees only the
//! normalized [`ModelRequest`] / [`ModelResponse`].
//!
//! One ephemeral `cache_control` breakpoint is placed on the system block so that
//! `system + tools + initial user message` caches after the first turn.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use whisper_agent_protocol::{ContentBlock, Message, Usage};

use crate::model::{
    BoxFuture, ModelError, ModelInfo, ModelProvider, ModelRequest, ModelResponse, ToolSpec,
};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const MODELS_URL: &str = "https://api.anthropic.com/v1/models";
const ANTHROPIC_VERSION: &str = "2023-06-01";

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
        let tools: Vec<AnthropicTool> = req.tools.iter().map(spec_to_anthropic_tool).collect();
        let system = vec![SystemBlock::cached(req.system_prompt.to_string())];
        let body = CreateMessageRequest {
            model: req.model,
            max_tokens: req.max_tokens,
            system,
            tools,
            messages: req.messages,
        };
        let resp = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
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
    }
}

// ---------- Wire types (private to this module) ----------

#[derive(Serialize, Debug)]
struct CreateMessageRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    system: Vec<SystemBlock>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    messages: &'a [Message],
}

#[derive(Serialize, Debug, Clone)]
struct SystemBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

impl SystemBlock {
    fn cached(text: String) -> Self {
        Self {
            kind: "text",
            text,
            cache_control: Some(CacheControl::ephemeral()),
        }
    }
}

#[derive(Serialize, Debug, Clone)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}

impl CacheControl {
    fn ephemeral() -> Self {
        Self { kind: "ephemeral" }
    }
}

#[derive(Serialize, Debug, Clone)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
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
