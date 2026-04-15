//! Anthropic Messages API client.
//!
//! Non-streaming for MVP — single POST to /v1/messages, single JSON response.
//! Streaming via SSE comes when the WebSocket UI lands.
//!
//! One ephemeral cache_control breakpoint placed on the last system block. That
//! caches `system + tools + initial user message` after the first turn, which is
//! where the bulk of the per-turn token cost lives.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use whisper_agent_protocol::{ContentBlock, Message};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Debug, Error)]
pub enum AnthropicError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("api error {status}: {body}")]
    Api { status: u16, body: String },
}

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

    pub async fn create_message(
        &self,
        request: &CreateMessageRequest<'_>,
    ) -> Result<MessageResponse, AnthropicError> {
        let resp = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(request)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(AnthropicError::Api { status: status.as_u16(), body });
        }
        Ok(resp.json::<MessageResponse>().await?)
    }
}

#[derive(Serialize, Debug)]
pub struct CreateMessageRequest<'a> {
    pub model: &'a str,
    pub max_tokens: u32,
    pub system: Vec<SystemBlock>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDescriptor>,
    pub messages: &'a [Message],
}

#[derive(Serialize, Debug, Clone)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self { kind: "text", text: text.into(), cache_control: None }
    }

    pub fn cached(text: impl Into<String>) -> Self {
        Self {
            kind: "text",
            text: text.into(),
            cache_control: Some(CacheControl::ephemeral()),
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: &'static str,
}

impl CacheControl {
    pub fn ephemeral() -> Self {
        Self { kind: "ephemeral" }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Deserialize, Debug)]
pub struct MessageResponse {
    pub id: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Deserialize, Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}
