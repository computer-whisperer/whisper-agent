//! Provider-agnostic model interface.
//!
//! [`ModelProvider`] is the object-safe trait every backend (Anthropic, OpenAI-compatible
//! Chat Completions, …) implements. The scheduler holds `Arc<dyn ModelProvider>` and
//! never touches a provider-specific type. Each impl converts between its own wire
//! shape and the normalized types here at the boundary.
//!
//! Normalized shapes intentionally re-use our canonical [`ContentBlock`] / [`Message`]
//! types, which are Anthropic-content-block-shaped. Translating OpenAI-style messages
//! into / out of this shape is the OpenAI backend's job, not the scheduler's.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;
use thiserror::Error;
use whisper_agent_protocol::{ContentBlock, Message, Usage};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub struct ModelRequest<'a> {
    pub model: &'a str,
    pub max_tokens: u32,
    pub system_prompt: &'a str,
    pub tools: &'a [ToolSpec],
    pub messages: &'a [Message],
}

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug)]
pub struct ModelResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: Option<String>,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("api error {status}: {body}")]
    Api { status: u16, body: String },
}

/// Provider-agnostic model backend. Object-safe via explicit pinned-boxed futures
/// (native `async fn in trait` isn't dyn-safe with `Send` bound without extra crates).
pub trait ModelProvider: Send + Sync {
    fn create_message<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
    ) -> BoxFuture<'a, Result<ModelResponse, ModelError>>;

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<ModelInfo>, ModelError>>;
}
