//! The [`Bucket`] trait — the abstraction boundary the rest of whisper-agent
//! holds when it needs to query, fetch from, or mutate a knowledge bucket.
//!
//! A bucket hides the slot directory, HNSW + tantivy indexes, RAM-vs-disk
//! serving mode, and concurrency model behind a stable `async`-shaped API.
//! Implementations come later in the slice sequence (see
//! `docs/design_knowledge_db.md`); this module pins the trait signature
//! that downstream code can already program against.

use std::future::Future;
use std::pin::Pin;

use tokio_util::sync::CancellationToken;

use super::types::{BucketError, BucketId, BucketStatus, Candidate, Chunk, ChunkId, NewChunk};

/// `Pin<Box<dyn Future>>` alias matching the providers layer
/// ([`crate::providers::model::ModelProvider`],
/// [`crate::providers::embedding::EmbeddingProvider`]). Object-safety via
/// pinned-boxed futures lets us hold buckets behind `Arc<dyn Bucket>` in
/// the resource registry.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Provider-agnostic bucket abstraction. Object-safe via explicit
/// pinned-boxed futures.
///
/// Buckets with a search path disabled (per
/// [`SearchPathsConfig`](super::config::SearchPathsConfig)) return an
/// empty candidate list from the corresponding `*_search` method rather
/// than returning [`BucketError::PathDisabled`] — query-path callers don't
/// need per-bucket capability awareness.
///
/// Cancellation contract matches the embedding/rerank providers: when the
/// token fires, the call must abort in-flight work and return
/// [`BucketError::Cancelled`].
pub trait Bucket: Send + Sync {
    fn id(&self) -> &BucketId;

    fn status(&self) -> BucketStatus;

    // --- query path ---

    /// Search the dense (HNSW over embeddings) path. Returns candidates
    /// ordered by descending source score (cosine similarity), capped at
    /// `top_k`. Empty if the bucket has dense disabled or has no
    /// `ready` slot.
    fn dense_search<'a>(
        &'a self,
        query_vec: &'a [f32],
        top_k: usize,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<Candidate>, BucketError>>;

    /// Search the sparse (BM25 via tantivy) path. Returns candidates
    /// ordered by descending source score (raw BM25), capped at `top_k`.
    /// Empty if the bucket has sparse disabled or has no `ready` slot.
    fn sparse_search<'a>(
        &'a self,
        query_text: &'a str,
        top_k: usize,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<Candidate>, BucketError>>;

    /// Fetch a chunk by id from the active slot. Used by the
    /// `knowledge_query` builtin tool when the LLM follows up on a
    /// candidate to retrieve full text.
    fn fetch_chunk<'a>(&'a self, chunk_id: ChunkId) -> BoxFuture<'a, Result<Chunk, BucketError>>;

    // --- mutation path ---

    /// Append new chunks to the active slot's delta layer. Embeds them
    /// against the slot's recorded embedder, inserts into both dense and
    /// sparse indexes, returns when durable. Returns the assigned chunk
    /// ids in the same order as the input.
    fn insert<'a>(
        &'a self,
        new_chunks: Vec<NewChunk>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<ChunkId>, BucketError>>;

    /// Mark chunks as deleted. They're filtered from query results
    /// immediately; physical removal happens at compaction time.
    fn tombstone<'a>(&'a self, chunk_ids: Vec<ChunkId>) -> BoxFuture<'a, Result<(), BucketError>>;

    /// Trigger a compaction — rebuild the active slot's base from current
    /// state (base + delta − tombstones) into a fresh slot. Long-running;
    /// completes when the new slot has been promoted to `active`.
    fn compact<'a>(
        &'a self,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<(), BucketError>>;
}
