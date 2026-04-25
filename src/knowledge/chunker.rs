//! [`Chunker`] — slice [`SourceRecord`]s into [`NewChunk`]s.
//!
//! v1 ships [`TokenBasedChunker`] — character-based with a configurable
//! `chars_per_token` ratio. The "token" count is approximate; we use it
//! to keep chunks comfortably under the embedder's max-input-tokens limit
//! without needing a model-specific tokenizer at chunk time. A future
//! slice can swap in real BPE tokenization (`tokenizers` crate +
//! per-embedder model files) when measurement justifies the cost.
//!
//! Chunkers must produce *deterministic* output for a given record:
//! re-chunking the same record yields the same chunks in the same order
//! at the same offsets. This is what makes
//! [`ChunkId::from_source`](super::ChunkId::from_source) stable across
//! re-ingests.

use super::config::ChunkerConfig;
use super::source::SourceRecord;
use super::types::{NewChunk, SourceRef};

pub trait Chunker: Send + Sync {
    /// Slice a single record into one or more chunks. Empty/whitespace-only
    /// records yield an empty `Vec`. The returned chunks have populated
    /// `source_record_hash` and `chunk_offset` fields ready for chunk-id
    /// derivation by the bucket layer.
    fn chunk(&self, record: &SourceRecord) -> Vec<NewChunk>;
}

/// Character-based chunker with sliding-window overlap.
///
/// Split the record's text into windows of `chunk_tokens * chars_per_token`
/// characters, advancing by `(chunk_tokens - overlap_tokens) * chars_per_token`
/// each step. Counting at the `char` boundary (not byte) ensures we never
/// split a UTF-8 code point.
///
/// `chars_per_token` is a tuning knob — 4 is conservative for English
/// natural-language content (typical BPE output is ~3.5 chars/token).
/// Code-heavy content tokenizes denser (~2-3 chars/token); set lower for
/// those. For wikipedia-scale we'll measure against a real tokenizer
/// before committing a default.
pub struct TokenBasedChunker {
    pub chunk_tokens: u32,
    pub overlap_tokens: u32,
    pub chars_per_token: u32,
}

impl TokenBasedChunker {
    /// Default ratio for English natural-language content. Conservative —
    /// real BPE typically achieves ~3.5 chars/token, so we under-count
    /// tokens and produce chunks slightly shorter than the budget allows.
    pub const DEFAULT_CHARS_PER_TOKEN: u32 = 4;

    pub fn new(chunk_tokens: u32, overlap_tokens: u32) -> Self {
        Self {
            chunk_tokens,
            overlap_tokens,
            chars_per_token: Self::DEFAULT_CHARS_PER_TOKEN,
        }
    }

    pub fn from_config(config: &ChunkerConfig) -> Self {
        match config {
            ChunkerConfig::TokenBased {
                chunk_tokens,
                overlap_tokens,
            } => Self::new(*chunk_tokens, *overlap_tokens),
        }
    }
}

impl Default for TokenBasedChunker {
    fn default() -> Self {
        Self::new(500, 50)
    }
}

impl Chunker for TokenBasedChunker {
    fn chunk(&self, record: &SourceRecord) -> Vec<NewChunk> {
        if record.text.trim().is_empty() {
            return Vec::new();
        }

        // Collect chars upfront so we can index slices cheaply. For a
        // wikipedia-scale build this could matter (a single very long
        // article might be MBs of text); revisit if benchmarks show
        // chunking dominates the build pipeline.
        let chars: Vec<char> = record.text.chars().collect();

        let chunk_chars = (self.chunk_tokens as usize)
            .saturating_mul(self.chars_per_token as usize)
            .max(1);
        let overlap_chars =
            (self.overlap_tokens as usize).saturating_mul(self.chars_per_token as usize);
        let stride = chunk_chars.saturating_sub(overlap_chars).max(1);

        // Single-chunk fast path — the common case for short notes.
        if chars.len() <= chunk_chars {
            return vec![NewChunk {
                text: record.text.clone(),
                source_ref: SourceRef {
                    source_id: record.source_id.clone(),
                    locator: Some(format!("chars 0-{}", chars.len())),
                },
                source_record_hash: record.content_hash,
                chunk_offset: 0,
            }];
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut offset = 0u64;

        while start < chars.len() {
            let end = (start + chunk_chars).min(chars.len());
            let text: String = chars[start..end].iter().collect();
            chunks.push(NewChunk {
                text,
                source_ref: SourceRef {
                    source_id: record.source_id.clone(),
                    locator: Some(format!("chars {start}-{end}")),
                },
                source_record_hash: record.content_hash,
                chunk_offset: offset,
            });
            offset += 1;

            if end == chars.len() {
                break;
            }
            start += stride;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::ChunkId;

    fn record(text: &str) -> SourceRecord {
        SourceRecord::new("test", text)
    }

    #[test]
    fn empty_record_yields_no_chunks() {
        let chunker = TokenBasedChunker::default();
        assert!(chunker.chunk(&record("")).is_empty());
        assert!(chunker.chunk(&record("   \n  \t  ")).is_empty());
    }

    #[test]
    fn single_short_record_is_one_chunk() {
        let chunker = TokenBasedChunker::new(500, 50); // 500 * 4 = 2000 char window
        let chunks = chunker.chunk(&record("just a few words"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "just a few words");
        assert_eq!(chunks[0].chunk_offset, 0);
    }

    #[test]
    fn longer_record_splits_with_overlap() {
        // 100 tokens × 4 chars = 400-char window; 10 tokens × 4 = 40-char overlap.
        // Stride = 360.
        let chunker = TokenBasedChunker::new(100, 10);
        let text: String = std::iter::repeat_n('x', 1000).collect();
        let chunks = chunker.chunk(&record(&text));

        // 1000 chars / 360 stride = 3 chunks (offsets 0, 360, 720).
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text.len(), 400);
        assert_eq!(chunks[1].text.len(), 400);
        // Last chunk is shorter — only chars 720..1000.
        assert_eq!(chunks[2].text.len(), 280);

        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_offset, i as u64);
        }
    }

    #[test]
    fn chunk_offsets_drive_unique_chunk_ids() {
        let chunker = TokenBasedChunker::new(50, 5);
        let text: String = std::iter::repeat_n('a', 500).collect();
        let chunks = chunker.chunk(&record(&text));

        let ids: Vec<ChunkId> = chunks
            .iter()
            .map(|c| ChunkId::from_source(&c.source_record_hash, c.chunk_offset))
            .collect();

        // All chunk ids are distinct
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "chunk ids must be unique");
    }

    #[test]
    fn chunking_is_deterministic() {
        let chunker = TokenBasedChunker::new(100, 10);
        let text: String = "hello world ".repeat(200);
        let r = record(&text);

        let first = chunker.chunk(&r);
        let second = chunker.chunk(&r);

        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.chunk_offset, b.chunk_offset);
            assert_eq!(a.source_record_hash, b.source_record_hash);
        }
    }

    #[test]
    fn unicode_does_not_split_code_points() {
        // Multi-byte characters; ensure char-based slicing is honored.
        let chunker = TokenBasedChunker::new(2, 0); // 2 tokens × 4 chars = 8-char window, no overlap
        // 12 chars total — split into 2 chunks of 8 + 4
        let text = "αβγδεζηθικλμ"; // 12 Greek letters
        let chunks = chunker.chunk(&record(text));

        assert_eq!(chunks.len(), 2);
        // Reassembling with overlap handling — for no-overlap, just
        // concat and you should get back original.
        let recombined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(recombined, text);
        // Each chunk independently parses as valid UTF-8 (a String already is).
        assert!(!chunks[0].text.is_empty());
        assert!(!chunks[1].text.is_empty());
    }

    #[test]
    fn from_config_parses_chunker_config() {
        let cfg = ChunkerConfig::TokenBased {
            chunk_tokens: 250,
            overlap_tokens: 25,
        };
        let chunker = TokenBasedChunker::from_config(&cfg);
        assert_eq!(chunker.chunk_tokens, 250);
        assert_eq!(chunker.overlap_tokens, 25);
        assert_eq!(
            chunker.chars_per_token,
            TokenBasedChunker::DEFAULT_CHARS_PER_TOKEN,
        );
    }

    #[test]
    fn locator_records_char_range() {
        let chunker = TokenBasedChunker::new(50, 5); // 200-char window, 20-char overlap
        let text: String = std::iter::repeat_n('x', 500).collect();
        let chunks = chunker.chunk(&record(&text));

        assert!(
            chunks[0]
                .source_ref
                .locator
                .as_deref()
                .unwrap()
                .starts_with("chars 0-")
        );
        // Second chunk starts after stride = 200 - 20 = 180
        assert!(
            chunks[1]
                .source_ref
                .locator
                .as_deref()
                .unwrap()
                .starts_with("chars 180-")
        );
    }

    #[test]
    fn zero_overlap_is_supported() {
        let chunker = TokenBasedChunker::new(50, 0);
        let text: String = std::iter::repeat_n('x', 500).collect();
        let chunks = chunker.chunk(&record(&text));

        // 200-char window, 200-char stride — 500/200 = 3 chunks (200+200+100)
        assert_eq!(chunks.len(), 3);
        // Concat with no overlap returns original
        let recombined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(recombined, text);
    }

    #[test]
    fn overlap_larger_than_chunk_falls_back_to_stride_1() {
        // Pathological config: overlap >= chunk_chars. Stride saturates to 1.
        let chunker = TokenBasedChunker {
            chunk_tokens: 5,
            overlap_tokens: 100,
            chars_per_token: 1,
        };
        let chunks = chunker.chunk(&record("0123456789"));
        // chunk_chars = 5, overlap = 100, stride = max(5 - 100, 1) = 1
        // 10 chars / stride 1 = many chunks
        assert!(chunks.len() > 5, "got {} chunks", chunks.len());
    }
}
