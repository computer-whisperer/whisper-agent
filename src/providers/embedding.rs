//! Provider-agnostic embedding interface.
//!
//! [`EmbeddingProvider`] is the object-safe trait every embedding backend
//! (TEI today; OpenAI / Cohere / Voyage tomorrow) implements. The scheduler
//! holds `Arc<dyn EmbeddingProvider>` and never touches a provider-specific
//! type. Each impl converts between its own wire shape and the normalized
//! types here at the boundary.
//!
//! Mirror of [`crate::providers::model::ModelProvider`] in shape: pinned-boxed
//! futures (object-safe `async fn` without extra crates), explicit
//! [`CancellationToken`] on every call, and a flat [`EmbeddingError`] enum
//! with the same transient/terminal classification.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use thiserror::Error;
use tokio_util::sync::CancellationToken;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub struct EmbedRequest<'a> {
    /// Model id. Local TEI deployments serve a single model and ignore the
    /// field; OpenAI-shaped backends route on it. Empty string is fine for
    /// providers that don't need it — the impl decides.
    pub model: &'a str,
    /// Texts to embed. Returned vectors line up positionally with this slice.
    pub inputs: &'a [String],
}

#[derive(Debug)]
pub struct EmbeddingResponse {
    /// One vector per input, in the same order. All vectors share the
    /// model's output dimension (validated on parse — a ragged response
    /// surfaces as [`EmbeddingError::Api`] rather than silently mixing dims).
    pub embeddings: Vec<Vec<f32>>,
    /// Token-usage account when the provider reports it. TEI's native
    /// `/embed` endpoint doesn't; OpenAI's `/embeddings` does.
    pub usage: Option<EmbeddingUsage>,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct EmbeddingModelInfo {
    pub id: String,
    /// Output vector dimension. Always present — bucket-build callers need
    /// to validate dimension compatibility before persisting an index.
    pub dimension: u32,
    /// Maximum input length the model accepts. TEI publishes this on
    /// `/info` as `max_input_length` (token count). `None` when the
    /// upstream doesn't report it.
    pub max_input_tokens: Option<u32>,
}

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("api error {status}: {body}")]
    Api { status: u16, body: String },
    /// Server signalled a short-term capacity / quota exhaustion (HTTP
    /// 429). Carries the server-suggested wait before retrying along with
    /// the raw body.
    #[error("rate limited: retry after {retry_after:?}: {body}")]
    RateLimited { retry_after: Duration, body: String },
    #[error("cancelled")]
    Cancelled,
}

impl EmbeddingError {
    /// True when this error is a transient infrastructure fault worth
    /// retrying transparently — same classification as
    /// [`crate::providers::model::ModelError::is_transient`].
    pub fn is_transient(&self) -> bool {
        match self {
            EmbeddingError::Transport(_) => true,
            EmbeddingError::Api { status, .. } => matches!(*status, 408 | 500..=599),
            EmbeddingError::RateLimited { .. } | EmbeddingError::Cancelled => false,
        }
    }
}

/// Provider-agnostic embedding backend. Object-safe via explicit
/// pinned-boxed futures. Cancellation contract matches
/// [`crate::providers::model::ModelProvider`]: when the token fires the
/// provider must abort the in-flight request and return
/// [`EmbeddingError::Cancelled`].
pub trait EmbeddingProvider: Send + Sync {
    fn embed<'a>(
        &'a self,
        req: &'a EmbedRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<EmbeddingResponse, EmbeddingError>>;

    /// Enumerate the models this provider can serve. Single-model TEI
    /// deployments return a one-element list. Used by the settings panel
    /// to surface the embedding dimension and validate bucket configs.
    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<EmbeddingModelInfo>, EmbeddingError>>;
}
