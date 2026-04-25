//! Provider-agnostic reranking interface.
//!
//! A reranker scores a batch of documents against a query and returns them
//! sorted by relevance. In a typical retrieval pipeline the embedding stage
//! produces a coarse top-N (cheap, vector-similarity), and the reranker
//! refines that to a smaller, more accurate top-K (more expensive,
//! cross-encoder).
//!
//! Mirrors [`crate::providers::embedding::EmbeddingProvider`] in shape so
//! the scheduler can hold both kinds in symmetric registries.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use thiserror::Error;
use tokio_util::sync::CancellationToken;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub struct RerankRequest<'a> {
    /// Model id. Local TEI deployments ignore it (single model per server);
    /// hosted providers route on it.
    pub model: &'a str,
    /// User query the documents are scored against.
    pub query: &'a str,
    /// Candidate documents. Indexes returned in [`RerankResult::index`]
    /// refer back into this slice.
    pub documents: &'a [String],
    /// Optional cap on returned results. When `Some`, the provider may
    /// short-circuit and return only the top-N; when `None`, every
    /// document is scored and returned. TEI accepts this as the
    /// `top_n` request field.
    pub top_n: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct RerankResult {
    /// Position in the original [`RerankRequest::documents`] slice.
    pub index: u32,
    /// Relevance score. Higher = more relevant. Magnitude is provider-
    /// and model-specific — callers should compare scores within a
    /// single response, not across requests or models.
    pub score: f32,
}

#[derive(Debug)]
pub struct RerankResponse {
    /// Sorted descending by `score`. Length is `min(documents.len(),
    /// top_n.unwrap_or(MAX))`.
    pub results: Vec<RerankResult>,
}

#[derive(Debug, Clone)]
pub struct RerankModelInfo {
    pub id: String,
    /// Maximum combined query+document length the model accepts, in
    /// tokens. `None` when not reported.
    pub max_input_tokens: Option<u32>,
}

#[derive(Debug, Error)]
pub enum RerankError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("api error {status}: {body}")]
    Api { status: u16, body: String },
    #[error("rate limited: retry after {retry_after:?}: {body}")]
    RateLimited { retry_after: Duration, body: String },
    #[error("cancelled")]
    Cancelled,
}

impl RerankError {
    pub fn is_transient(&self) -> bool {
        match self {
            RerankError::Transport(_) => true,
            RerankError::Api { status, .. } => matches!(*status, 408 | 500..=599),
            RerankError::RateLimited { .. } | RerankError::Cancelled => false,
        }
    }
}

pub trait RerankProvider: Send + Sync {
    fn rerank<'a>(
        &'a self,
        req: &'a RerankRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<RerankResponse, RerankError>>;

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<RerankModelInfo>, RerankError>>;
}
