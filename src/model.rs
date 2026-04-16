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
    /// Positions at which the backend should mark prompt-cache checkpoints.
    /// Providers that don't support caching ignore this list. Anthropic caps the
    /// list at 4 markers per request — the scheduler is responsible for staying
    /// under that limit.
    pub cache_breakpoints: &'a [CacheBreakpoint],
}

/// Logical positions at which a cache checkpoint can be attached. Translated into
/// provider-specific markers by each [`ModelProvider`] implementation — or ignored
/// by providers without cache support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBreakpoint {
    /// Cache the system prompt (everything up to and including `system`).
    AfterSystem,
    /// Cache the tool declarations (everything up to and including `tools`).
    AfterTools,
    /// Cache through `messages[index]` (inclusive).
    AfterMessage(usize),
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

/// Default cache policy: cache the (byte-stable) system prompt and tool declarations,
/// then cache through the most recent user-role messages — up to `max_trailing_user`
/// of them — so each turn's tool_results become a rolling checkpoint without
/// exceeding Anthropic's 4-marker cap. Pass `max_trailing_user = 2` to leave headroom
/// for the two fixed markers.
pub fn default_cache_policy(
    messages: &[Message],
    max_trailing_user: usize,
) -> Vec<CacheBreakpoint> {
    let mut out = vec![CacheBreakpoint::AfterSystem, CacheBreakpoint::AfterTools];
    if max_trailing_user == 0 {
        return out;
    }
    // Walk back through the messages picking the most recent user-role entries —
    // those are where tool_results live, and each is a natural cache boundary.
    let user_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m.role, whisper_agent_protocol::Role::User))
        .map(|(i, _)| i)
        .collect();
    for &i in user_indices.iter().rev().take(max_trailing_user) {
        out.push(CacheBreakpoint::AfterMessage(i));
    }
    out
}
