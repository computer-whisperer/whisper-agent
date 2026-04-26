//! [`Chunker`] — slice [`SourceRecord`]s into [`NewChunk`]s.
//!
//! [`TokenBasedChunker`] is the only chunker. Internally it dispatches
//! between two strategies:
//!
//! - **BPE**: a real HuggingFace tokenizer ([`tokenizers::Tokenizer`])
//!   loaded from a local `tokenizer.json` or auto-fetched via `hf-hub`.
//!   Token-aware char windowing — token-level offsets from the encoding
//!   tell us where each token starts/ends in the original text, and we
//!   slice the source by char at exact token-count boundaries. Output
//!   text is always a contiguous substring of the source.
//! - **Heuristic**: a `chars_per_token` ratio applied as a sliding char
//!   window. Used as a fallback when the bucket has `tokenizer = "auto"`
//!   and resolution fails (no network, embedder doesn't expose a model
//!   id, etc.), and as a test convenience that doesn't need a tokenizer
//!   file or network. Approximate but deterministic.
//!
//! Resolution of which strategy a slot ends up using happens in
//! [`resolve_chunker`], which takes the user-edited [`ChunkerConfig`]
//! and the embedder's reported model id (when available) and returns a
//! ready-to-chunk [`ResolvedChunker`] plus a [`ChunkerSnapshot`] to
//! freeze into the slot manifest.
//!
//! Chunkers must produce *deterministic* output for a given record:
//! re-chunking the same record yields the same chunks in the same order
//! at the same offsets. This is what makes
//! [`ChunkId::from_source`](super::ChunkId::from_source) stable across
//! re-ingests.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokenizers::Tokenizer;

use super::config::{ChunkerConfig, TokenizerSource};
use super::manifest::{ChunkerSnapshot, TokenizerSnapshot};
use super::source::SourceRecord;
use super::types::{NewChunk, SourceRef};

pub trait Chunker: Send + Sync {
    /// Slice a single record into one or more chunks. Empty/whitespace-only
    /// records yield an empty `Vec`. The returned chunks have populated
    /// `source_record_hash` and `chunk_offset` fields ready for chunk-id
    /// derivation by the bucket layer.
    fn chunk(&self, record: &SourceRecord) -> Vec<NewChunk>;
}

/// Sliding-window chunker. Either does real BPE token counting or falls
/// back to a `chars_per_token` heuristic depending on what
/// [`resolve_chunker`] produced.
pub struct TokenBasedChunker {
    chunk_tokens: u32,
    overlap_tokens: u32,
    strategy: ChunkStrategy,
}

enum ChunkStrategy {
    /// Real BPE — `tokenizer.encode_with_offsets` gives us a `(start, end)`
    /// byte range per token, and we slice the source at the exact byte
    /// boundary corresponding to the chunk's last token.
    Bpe(Arc<Tokenizer>),
    /// Char-window heuristic — `chunk_tokens * chars_per_token` chars per
    /// chunk, advancing by `(chunk_tokens - overlap_tokens) * chars_per_token`.
    Heuristic { chars_per_token: u32 },
}

impl TokenBasedChunker {
    /// Conservative default for the heuristic. Real BPE for English
    /// natural-language content typically lands ~3.5 chars/token; 4 is a
    /// safe under-count that produces chunks slightly under the budget.
    pub const DEFAULT_CHARS_PER_TOKEN: u32 = 4;

    /// Heuristic chunker — char-window, no tokenizer required. Used by
    /// tests and as the resolution fallback when a real tokenizer can't
    /// be loaded.
    pub fn heuristic(chunk_tokens: u32, overlap_tokens: u32) -> Self {
        Self {
            chunk_tokens,
            overlap_tokens,
            strategy: ChunkStrategy::Heuristic {
                chars_per_token: Self::DEFAULT_CHARS_PER_TOKEN,
            },
        }
    }

    /// Heuristic with explicit `chars_per_token`. For code-heavy content
    /// (~2-3 chars/token) the default 4 over-counts; bucket configs can
    /// pin a lower ratio.
    pub fn heuristic_with_ratio(
        chunk_tokens: u32,
        overlap_tokens: u32,
        chars_per_token: u32,
    ) -> Self {
        Self {
            chunk_tokens,
            overlap_tokens,
            strategy: ChunkStrategy::Heuristic {
                chars_per_token: chars_per_token.max(1),
            },
        }
    }

    /// BPE chunker backed by a pre-loaded [`Tokenizer`]. Caller is
    /// responsible for loading + sharing the tokenizer (typically via
    /// [`resolve_chunker`]).
    pub fn from_tokenizer(
        chunk_tokens: u32,
        overlap_tokens: u32,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        Self {
            chunk_tokens,
            overlap_tokens,
            strategy: ChunkStrategy::Bpe(tokenizer),
        }
    }

    /// Backwards-compat constructor — heuristic with default ratio.
    /// Equivalent to [`Self::heuristic`].
    pub fn new(chunk_tokens: u32, overlap_tokens: u32) -> Self {
        Self::heuristic(chunk_tokens, overlap_tokens)
    }

    /// Test/dev helper — produce a heuristic chunker plus a matching
    /// [`ChunkerSnapshot`] from a [`ChunkerConfig`], without touching the
    /// network or hf-hub. Production code paths go through
    /// [`resolve_chunker`] instead, which honors the configured
    /// `tokenizer` source.
    pub fn from_config(config: &ChunkerConfig) -> (Self, ChunkerSnapshot) {
        let ChunkerConfig::TokenBased {
            chunk_tokens,
            overlap_tokens,
            ..
        } = config;
        let chunker = Self::heuristic(*chunk_tokens, *overlap_tokens);
        let snapshot = ChunkerSnapshot {
            strategy: "token_based".to_string(),
            chunk_tokens: *chunk_tokens,
            overlap_tokens: *overlap_tokens,
            tokenizer: TokenizerSnapshot::Heuristic {
                chars_per_token: Self::DEFAULT_CHARS_PER_TOKEN,
            },
        };
        (chunker, snapshot)
    }

    /// True when this chunker is using the heuristic fallback. Useful
    /// for tests asserting which strategy `resolve_chunker` produced.
    pub fn is_heuristic(&self) -> bool {
        matches!(self.strategy, ChunkStrategy::Heuristic { .. })
    }

    fn chunk_heuristic(&self, record: &SourceRecord, chars_per_token: u32) -> Vec<NewChunk> {
        // Collect chars upfront so we can index slices cheaply. For a
        // wikipedia-scale build this could matter (a single very long
        // article might be MBs of text); revisit if benchmarks show
        // chunking dominates the build pipeline.
        let chars: Vec<char> = record.text.chars().collect();

        let chunk_chars = (self.chunk_tokens as usize)
            .saturating_mul(chars_per_token as usize)
            .max(1);
        let overlap_chars = (self.overlap_tokens as usize).saturating_mul(chars_per_token as usize);
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

    fn chunk_bpe(&self, record: &SourceRecord, tokenizer: &Tokenizer) -> Vec<NewChunk> {
        // Encode without special tokens — we want offsets into the original
        // text, not artifacts like `[CLS]` / `[SEP]`. `add_special_tokens =
        // false` keeps the tokenization to the actual content.
        let encoding = match tokenizer.encode(record.text.as_str(), false) {
            Ok(e) => e,
            Err(e) => {
                // Tokenizer failure on a single record is non-fatal for the
                // build — fall through to the heuristic so the build makes
                // progress, and log the record id for diagnosis.
                tracing::warn!(
                    source_id = %record.source_id,
                    error = %e,
                    "BPE encode failed; falling back to heuristic for this record",
                );
                return self.chunk_heuristic(record, Self::DEFAULT_CHARS_PER_TOKEN);
            }
        };

        let offsets = encoding.get_offsets(); // Vec<(usize, usize)> — byte offsets
        let n_tokens = offsets.len();

        if n_tokens == 0 {
            return Vec::new();
        }

        let chunk_tokens = (self.chunk_tokens as usize).max(1);
        let overlap_tokens = (self.overlap_tokens as usize).min(chunk_tokens.saturating_sub(1));
        let stride = chunk_tokens.saturating_sub(overlap_tokens).max(1);

        // Single-chunk fast path: whole record fits.
        if n_tokens <= chunk_tokens {
            return vec![NewChunk {
                text: record.text.clone(),
                source_ref: SourceRef {
                    source_id: record.source_id.clone(),
                    locator: Some(format!("tokens 0-{n_tokens}")),
                },
                source_record_hash: record.content_hash,
                chunk_offset: 0,
            }];
        }

        let mut chunks = Vec::new();
        let mut tok_start = 0usize;
        let mut offset = 0u64;

        while tok_start < n_tokens {
            let tok_end = (tok_start + chunk_tokens).min(n_tokens);

            // Convert token-index window → byte-range window. The first
            // token's start byte and the last token's end byte give us the
            // exact span in the source. BPE offsets are guaranteed to lie
            // on UTF-8 boundaries by construction.
            let byte_start = offsets[tok_start].0;
            let byte_end = offsets[tok_end - 1].1;

            // Defensive: the tokenizer can occasionally produce zero-width
            // offsets (e.g. for byte-fallback tokens on malformed input).
            // Skip empty slices rather than emit a zero-text chunk.
            if byte_end <= byte_start {
                tok_start += stride;
                continue;
            }

            let text = record.text[byte_start..byte_end].to_string();
            chunks.push(NewChunk {
                text,
                source_ref: SourceRef {
                    source_id: record.source_id.clone(),
                    locator: Some(format!("tokens {tok_start}-{tok_end}")),
                },
                source_record_hash: record.content_hash,
                chunk_offset: offset,
            });
            offset += 1;

            if tok_end == n_tokens {
                break;
            }
            tok_start += stride;
        }

        chunks
    }
}

impl Default for TokenBasedChunker {
    fn default() -> Self {
        Self::heuristic(500, 50)
    }
}

impl Chunker for TokenBasedChunker {
    fn chunk(&self, record: &SourceRecord) -> Vec<NewChunk> {
        if record.text.trim().is_empty() {
            return Vec::new();
        }
        match &self.strategy {
            ChunkStrategy::Bpe(t) => self.chunk_bpe(record, t),
            ChunkStrategy::Heuristic { chars_per_token } => {
                self.chunk_heuristic(record, *chars_per_token)
            }
        }
    }
}

/// The output of [`resolve_chunker`] — a ready-to-use chunker plus the
/// snapshot that gets frozen into the slot manifest. The snapshot is what
/// the resume path reads to validate the tokenizer hasn't drifted under
/// an in-progress build.
pub struct ResolvedChunker {
    pub chunker: Box<dyn Chunker + Send + Sync>,
    pub snapshot: ChunkerSnapshot,
}

// Manual Debug — `Box<dyn Chunker>` doesn't implement Debug. The
// snapshot carries the user-visible details anyway (strategy, params,
// tokenizer kind/hash).
impl std::fmt::Debug for ResolvedChunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedChunker")
            .field("snapshot", &self.snapshot)
            .finish_non_exhaustive()
    }
}

/// Errors that can surface during resolution. Auto-fallback to heuristic
/// is the policy for `TokenizerSource::Auto`; explicit `HfModel` /
/// `Path` failures are hard errors.
#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("tokenizer file not found at {0}: {1}")]
    PathNotFound(PathBuf, String),
    #[error("tokenizer.json could not be parsed: {0}")]
    InvalidTokenizer(String),
    #[error("hf-hub fetch failed for `{model_id}`: {error}")]
    HubFetch { model_id: String, error: String },
}

/// Resolve a [`ChunkerConfig`] into a runnable chunker plus its frozen
/// manifest snapshot. `embedder_model_id` is the id reported by the
/// bucket's configured embedder via `EmbeddingProvider::list_models()`;
/// it's only consulted when the config is `TokenizerSource::Auto`.
///
/// Resolution policy:
/// - `Auto` + embedder id → hf-hub fetch; fall back to heuristic on
///   any failure.
/// - `Auto` + no embedder id → heuristic with `tracing::warn`.
/// - `HfModel(id)` → hf-hub fetch; hard error on failure.
/// - `Path(p)` → read file; hard error on failure.
/// - `Heuristic { chars_per_token }` → heuristic, no I/O.
pub fn resolve_chunker(
    config: &ChunkerConfig,
    embedder_model_id: Option<&str>,
) -> Result<ResolvedChunker, ResolveError> {
    let ChunkerConfig::TokenBased {
        chunk_tokens,
        overlap_tokens,
        tokenizer,
    } = config;

    let strategy_name = "token_based".to_string();

    let make_heuristic_snapshot = |chars_per_token: u32| ChunkerSnapshot {
        strategy: strategy_name.clone(),
        chunk_tokens: *chunk_tokens,
        overlap_tokens: *overlap_tokens,
        tokenizer: TokenizerSnapshot::Heuristic { chars_per_token },
    };

    let make_bpe_snapshot = |source: BpeSource, content_hash: String| ChunkerSnapshot {
        strategy: strategy_name.clone(),
        chunk_tokens: *chunk_tokens,
        overlap_tokens: *overlap_tokens,
        tokenizer: match source {
            BpeSource::HfModel(id) => TokenizerSnapshot::HfModel {
                model_id: id,
                content_hash,
            },
            BpeSource::Path(p) => TokenizerSnapshot::Path {
                path: p,
                content_hash,
            },
        },
    };

    let heuristic_fallback = |reason: &str| -> ResolvedChunker {
        tracing::warn!(reason, "chunker using char-window heuristic fallback");
        ResolvedChunker {
            chunker: Box::new(TokenBasedChunker::heuristic(*chunk_tokens, *overlap_tokens)),
            snapshot: make_heuristic_snapshot(TokenBasedChunker::DEFAULT_CHARS_PER_TOKEN),
        }
    };

    match tokenizer {
        TokenizerSource::Auto => {
            let Some(model_id) = embedder_model_id else {
                return Ok(heuristic_fallback(
                    "tokenizer = \"auto\" but embedder didn't report a model id",
                ));
            };
            match fetch_hf_tokenizer(model_id) {
                Ok((tok, hash)) => {
                    tracing::info!(model_id, "resolved chunker tokenizer via hf-hub");
                    Ok(ResolvedChunker {
                        chunker: Box::new(TokenBasedChunker::from_tokenizer(
                            *chunk_tokens,
                            *overlap_tokens,
                            Arc::new(tok),
                        )),
                        snapshot: make_bpe_snapshot(BpeSource::HfModel(model_id.to_string()), hash),
                    })
                }
                Err(e) => {
                    tracing::warn!(
                        model_id,
                        error = %e,
                        "auto tokenizer resolution failed; falling back to heuristic",
                    );
                    Ok(heuristic_fallback("hf-hub fetch failed"))
                }
            }
        }
        TokenizerSource::HfModel { model_id } => {
            let (tok, hash) = fetch_hf_tokenizer(model_id).map_err(|e| ResolveError::HubFetch {
                model_id: model_id.clone(),
                error: e,
            })?;
            Ok(ResolvedChunker {
                chunker: Box::new(TokenBasedChunker::from_tokenizer(
                    *chunk_tokens,
                    *overlap_tokens,
                    Arc::new(tok),
                )),
                snapshot: make_bpe_snapshot(BpeSource::HfModel(model_id.clone()), hash),
            })
        }
        TokenizerSource::Path { path } => {
            let (tok, hash) = load_path_tokenizer(path)?;
            Ok(ResolvedChunker {
                chunker: Box::new(TokenBasedChunker::from_tokenizer(
                    *chunk_tokens,
                    *overlap_tokens,
                    Arc::new(tok),
                )),
                snapshot: make_bpe_snapshot(BpeSource::Path(path.clone()), hash),
            })
        }
        TokenizerSource::Heuristic { chars_per_token } => Ok(ResolvedChunker {
            chunker: Box::new(TokenBasedChunker::heuristic_with_ratio(
                *chunk_tokens,
                *overlap_tokens,
                *chars_per_token,
            )),
            snapshot: make_heuristic_snapshot(*chars_per_token),
        }),
    }
}

enum BpeSource {
    HfModel(String),
    Path(PathBuf),
}

fn fetch_hf_tokenizer(model_id: &str) -> Result<(Tokenizer, String), String> {
    let api = hf_hub::api::sync::Api::new().map_err(|e| format!("hf-hub api init: {e}"))?;
    let repo = api.model(model_id.to_string());
    let path = repo
        .get("tokenizer.json")
        .map_err(|e| format!("hf-hub get tokenizer.json: {e}"))?;
    load_path_tokenizer(&path).map_err(|e| match e {
        ResolveError::PathNotFound(_, msg) => msg,
        ResolveError::InvalidTokenizer(msg) => msg,
        ResolveError::HubFetch { error, .. } => error,
    })
}

fn load_path_tokenizer(path: &Path) -> Result<(Tokenizer, String), ResolveError> {
    let bytes = std::fs::read(path)
        .map_err(|e| ResolveError::PathNotFound(path.to_path_buf(), e.to_string()))?;
    let hash = blake3::hash(&bytes).to_hex().to_string();
    let tokenizer =
        Tokenizer::from_bytes(&bytes).map_err(|e| ResolveError::InvalidTokenizer(e.to_string()))?;
    Ok((tokenizer, hash))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::ChunkId;
    use ahash::AHashMap;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn record(text: &str) -> SourceRecord {
        SourceRecord::new("test", text)
    }

    /// Build a tiny WordLevel tokenizer programmatically for tests. No
    /// fixture files; no network, no training — the vocab is built from
    /// the unique whitespace-split words in `corpus`, plus `[UNK]` at id
    /// 0. WordLevel + Whitespace pre-tokenizer is enough to produce
    /// token-level offsets, which is all the chunker needs.
    fn tiny_word_tokenizer(corpus: &[&str]) -> Tokenizer {
        let mut vocab: AHashMap<String, u32> = AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        let mut next_id: u32 = 1;
        for s in corpus {
            for word in s.split_whitespace() {
                if !vocab.contains_key(word) {
                    vocab.insert(word.to_string(), next_id);
                    next_id += 1;
                }
            }
        }
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        tok
    }

    #[test]
    fn empty_record_yields_no_chunks() {
        let chunker = TokenBasedChunker::heuristic(500, 50);
        assert!(chunker.chunk(&record("")).is_empty());
        assert!(chunker.chunk(&record("   \n  \t  ")).is_empty());
    }

    #[test]
    fn heuristic_single_short_record_is_one_chunk() {
        let chunker = TokenBasedChunker::heuristic(500, 50); // 500 * 4 = 2000 char window
        let chunks = chunker.chunk(&record("just a few words"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "just a few words");
        assert_eq!(chunks[0].chunk_offset, 0);
    }

    #[test]
    fn heuristic_longer_record_splits_with_overlap() {
        // 100 tokens × 4 chars = 400-char window; 10 tokens × 4 = 40-char overlap.
        // Stride = 360.
        let chunker = TokenBasedChunker::heuristic(100, 10);
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
        let chunker = TokenBasedChunker::heuristic(50, 5);
        let text: String = std::iter::repeat_n('a', 500).collect();
        let chunks = chunker.chunk(&record(&text));

        let ids: Vec<ChunkId> = chunks
            .iter()
            .map(|c| ChunkId::from_source(&c.source_record_hash, c.chunk_offset))
            .collect();

        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "chunk ids must be unique");
    }

    #[test]
    fn chunking_is_deterministic() {
        let chunker = TokenBasedChunker::heuristic(100, 10);
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
        let chunker = TokenBasedChunker::heuristic(2, 0); // 2 tokens × 4 chars = 8-char window
        let text = "αβγδεζηθικλμ"; // 12 Greek letters
        let chunks = chunker.chunk(&record(text));

        assert_eq!(chunks.len(), 2);
        let recombined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(recombined, text);
        assert!(!chunks[0].text.is_empty());
        assert!(!chunks[1].text.is_empty());
    }

    #[test]
    fn zero_overlap_is_supported() {
        let chunker = TokenBasedChunker::heuristic(50, 0);
        let text: String = std::iter::repeat_n('x', 500).collect();
        let chunks = chunker.chunk(&record(&text));

        assert_eq!(chunks.len(), 3);
        let recombined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(recombined, text);
    }

    #[test]
    fn overlap_larger_than_chunk_falls_back_to_stride_1() {
        // Pathological config: overlap >= chunk_chars. Stride saturates to 1.
        let chunker = TokenBasedChunker::heuristic_with_ratio(5, 100, 1);
        let chunks = chunker.chunk(&record("0123456789"));
        assert!(chunks.len() > 5, "got {} chunks", chunks.len());
    }

    #[test]
    fn bpe_chunker_splits_at_token_boundaries() {
        // 12 words, chunk_tokens=4, overlap=1 → stride=3, expected token
        // windows [0..4], [3..7], [6..10], [9..12] → 4 chunks.
        let corpus = ["one two three four five six seven eight nine ten eleven twelve"];
        let tokenizer = Arc::new(tiny_word_tokenizer(&corpus));
        let chunker = TokenBasedChunker::from_tokenizer(4, 1, tokenizer);

        let chunks = chunker.chunk(&record(corpus[0]));

        assert_eq!(chunks.len(), 4, "got chunks: {chunks:#?}");
        assert!(chunks[0].text.starts_with("one"));
        assert!(chunks[0].text.ends_with("four"));
        assert!(chunks[1].text.starts_with("four"));
        assert!(chunks[3].text.ends_with("twelve"));
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_offset, i as u64);
        }
    }

    #[test]
    fn bpe_chunker_short_record_is_one_chunk() {
        let corpus = ["hello world foo bar"];
        let tokenizer = Arc::new(tiny_word_tokenizer(&corpus));
        let chunker = TokenBasedChunker::from_tokenizer(100, 10, tokenizer);

        let chunks = chunker.chunk(&record(corpus[0]));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, corpus[0]);
        assert_eq!(chunks[0].chunk_offset, 0);
    }

    #[test]
    fn bpe_chunker_text_is_substring_of_source() {
        // Critical invariant: BPE chunker never re-encodes; chunk text is
        // always a contiguous slice of the source.
        let corpus = ["alpha beta gamma delta epsilon zeta eta theta"];
        let tokenizer = Arc::new(tiny_word_tokenizer(&corpus));
        let chunker = TokenBasedChunker::from_tokenizer(3, 0, tokenizer);

        let chunks = chunker.chunk(&record(corpus[0]));
        for c in &chunks {
            assert!(
                corpus[0].contains(&c.text),
                "chunk text {:?} not found in source",
                c.text,
            );
        }
    }

    #[test]
    fn resolve_explicit_heuristic_skips_io() {
        let cfg = ChunkerConfig::TokenBased {
            chunk_tokens: 100,
            overlap_tokens: 10,
            tokenizer: TokenizerSource::Heuristic { chars_per_token: 3 },
        };
        let resolved = resolve_chunker(&cfg, None).unwrap();
        assert!(matches!(
            resolved.snapshot.tokenizer,
            TokenizerSnapshot::Heuristic { chars_per_token: 3 },
        ));
    }

    #[test]
    fn resolve_auto_falls_back_when_no_embedder_id() {
        let cfg = ChunkerConfig::TokenBased {
            chunk_tokens: 100,
            overlap_tokens: 10,
            tokenizer: TokenizerSource::Auto,
        };
        let resolved = resolve_chunker(&cfg, None).unwrap();
        assert!(matches!(
            resolved.snapshot.tokenizer,
            TokenizerSnapshot::Heuristic { .. },
        ));
    }

    #[test]
    fn resolve_path_loads_local_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        let corpus = ["hello world"];
        let tok = tiny_word_tokenizer(&corpus);
        tok.save(&path, false).unwrap();

        let cfg = ChunkerConfig::TokenBased {
            chunk_tokens: 100,
            overlap_tokens: 10,
            tokenizer: TokenizerSource::Path { path: path.clone() },
        };
        let resolved = resolve_chunker(&cfg, None).unwrap();
        match resolved.snapshot.tokenizer {
            TokenizerSnapshot::Path {
                path: snap_path,
                content_hash,
            } => {
                assert_eq!(snap_path, path);
                assert!(!content_hash.is_empty());
            }
            other => panic!("expected Path snapshot, got {other:?}"),
        }
    }

    #[test]
    fn resolve_path_missing_is_hard_error() {
        let cfg = ChunkerConfig::TokenBased {
            chunk_tokens: 100,
            overlap_tokens: 10,
            tokenizer: TokenizerSource::Path {
                path: PathBuf::from("/nonexistent/tokenizer.json"),
            },
        };
        match resolve_chunker(&cfg, None) {
            Err(ResolveError::PathNotFound(..)) => {}
            other => panic!("expected PathNotFound, got {other:?}"),
        }
    }
}
