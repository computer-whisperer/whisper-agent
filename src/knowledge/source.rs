//! Source adapters: read raw content from external systems and present it
//! as a stream of [`SourceRecord`]s for the chunker to slice.
//!
//! The adapter abstraction is what decouples the bucket layer from any
//! specific content source. v1 ships [`markdown_dir::MarkdownDir`]; future
//! adapters (mediawiki XML, arxiv, generic file glob) plug in through the
//! same [`SourceAdapter`] trait without touching the rest of the pipeline.
//!
//! Adapters are synchronous iterators on purpose — the build pipeline runs
//! on a dedicated OS thread (per the concurrency model in
//! `docs/design_knowledge_db.md`) and sync iteration is the natural fit.
//! If we ever want to overlap source IO with embedding requests, the trait
//! can graduate to a `futures::Stream` without changing call sites that
//! collect into `Vec`.

pub mod feed;
pub mod markdown_dir;
pub mod mediawiki_xml;

pub use feed::{DeltaId, FeedDriver, FeedError, SnapshotId, WikipediaDriver};
pub use markdown_dir::MarkdownDir;
pub use mediawiki_xml::MediaWikiXml;

use thiserror::Error;

/// One unit of source content. The chunker slices a record into one or
/// more [`crate::knowledge::NewChunk`]s.
///
/// `content_hash` is `blake3(text)` — content-only, *not* including
/// `source_id`. Two records with identical text produce identical chunk
/// ids, which is the desired property: a renamed file doesn't invalidate
/// its chunks, only the source_ref metadata changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRecord {
    /// Adapter-specific identifier (file path, article title, paper id,
    /// ...). Carried into [`SourceRef`](super::SourceRef) on emitted
    /// chunks for citation purposes.
    pub source_id: String,

    /// Canonical text of the record. Whatever normalization the adapter
    /// applies (frontmatter strip, whitespace normalization) is baked
    /// into this — `content_hash` is over this exact string.
    pub text: String,

    /// `blake3(text)`. Stable across re-ingests of the same content.
    /// Combined with the chunker's per-record offset to produce stable
    /// chunk ids.
    pub content_hash: [u8; 32],
}

impl SourceRecord {
    /// Construct a record, hashing `text` to populate `content_hash`.
    /// Adapters that have already computed the hash (e.g. for
    /// staleness-detection during rescan) should populate the field
    /// directly instead of going through this constructor.
    pub fn new(source_id: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let content_hash = *blake3::hash(text.as_bytes()).as_bytes();
        Self {
            source_id: source_id.into(),
            text,
            content_hash,
        }
    }
}

/// Provider-agnostic source enumeration. Yields records one at a time;
/// per-record errors surface as `Err` items in the iterator without
/// halting iteration (a single malformed file shouldn't abort the
/// build).
pub trait SourceAdapter: Send + Sync {
    /// Walk the source and emit records. Adapters with a finite source
    /// (markdown directory, wikipedia dump) yield until exhaustion;
    /// continuous-stream adapters (future: arxiv RSS, etc.) are out of
    /// scope here and would model differently.
    fn enumerate(&self) -> Box<dyn Iterator<Item = Result<SourceRecord, SourceError>> + Send + '_>;
}

#[derive(Debug, Error)]
pub enum SourceError {
    #[error("io error reading {path}: {error}")]
    Io {
        path: String,
        #[source]
        error: std::io::Error,
    },

    #[error("invalid utf-8 in {path}")]
    InvalidUtf8 { path: String },

    #[error("source not found: {0}")]
    NotFound(String),

    #[error("other: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_record_hash_is_deterministic() {
        let a = SourceRecord::new("a.md", "hello world");
        let b = SourceRecord::new("b.md", "hello world");
        assert_eq!(
            a.content_hash, b.content_hash,
            "same text → same hash regardless of source_id",
        );

        let c = SourceRecord::new("a.md", "different content");
        assert_ne!(a.content_hash, c.content_hash);
    }

    #[test]
    fn source_record_hash_matches_blake3_directly() {
        let r = SourceRecord::new("foo", "bar");
        assert_eq!(r.content_hash, *blake3::hash(b"bar").as_bytes());
    }
}
