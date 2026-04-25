//! Knowledge-bucket retrieval layer: dense (HNSW over embeddings) + sparse
//! (BM25 via tantivy) hybrid search over named buckets, with a reranker
//! fusing candidates from all paths and all queried buckets.
//!
//! See `docs/design_knowledge_db.md` for the full architectural background.
//! This module currently exposes only the foundational types and parsers;
//! source adapters, chunking, indexing, and the query path land in
//! subsequent slices.
//!
//! # Layout
//!
//! - [`types`] — small types ([`BucketId`], [`ChunkId`], [`Candidate`], ...)
//!   plus error enums.
//! - [`bucket`] — the [`Bucket`] trait, the abstraction boundary the rest
//!   of whisper-agent talks to.
//! - [`config`] — [`BucketConfig`], the parser for `<bucket>/bucket.toml`.
//! - [`manifest`] — [`SlotManifest`], the parser for
//!   `<bucket>/slots/<slot_id>/manifest.toml`.
//! - [`source`] — [`SourceAdapter`] trait + concrete adapters
//!   ([`MarkdownDir`], future MediaWiki / Arxiv / etc.).
//! - [`chunker`] — [`Chunker`] trait + [`TokenBasedChunker`] for slicing
//!   source records into chunks ready for embedding + indexing.

pub mod bucket;
pub mod chunker;
pub mod config;
pub mod manifest;
pub mod source;
pub mod types;

pub use bucket::Bucket;
pub use chunker::{Chunker, TokenBasedChunker};
pub use config::{
    BucketConfig, ChunkerConfig, CompactionConfig, DefaultsConfig, DensePathConfig, Quantization,
    RescanStrategy, SearchPathsConfig, ServingMode, SourceConfig, SparsePathConfig,
};
pub use manifest::{
    EmbedderSnapshot, ServingSnapshot, SlotLineage, SlotManifest, SlotState, SlotStats,
    SparseSnapshot,
};
pub use source::{MarkdownDir, SourceAdapter, SourceError, SourceRecord};
pub use types::{
    BucketError, BucketId, BucketScope, BucketStatus, BuildProgress, Candidate, Chunk, ChunkId,
    NewChunk, ParseBucketIdError, ParseChunkIdError, SearchPath, SlotId, SourceRef,
};
