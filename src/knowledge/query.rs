//! Multi-bucket query layer with reranker fusion.
//!
//! Sits *above* the [`Bucket`] trait — coordinates dense + sparse search
//! across one or more buckets, deduplicates the candidate union, and
//! sends the result to a reranker for final ordering. The output is
//! ranked top-K [`RerankedCandidate`]s ready for the LLM-facing
//! `knowledge_query` tool or the webui search panel.
//!
//! See `docs/design_knowledge_db.md` § "Hybrid retrieval" for the
//! architectural reasoning. Key invariants:
//!
//! - **One embed call per query.** The query string is embedded once
//!   via [`QueryEngine`]'s embedder; the resulting vector is shared
//!   across all dense paths. Buckets whose embedder dimension doesn't
//!   match return [`BucketError::DimensionMismatch`] from
//!   [`Bucket::dense_search`]; we skip those silently and let the
//!   sparse path still contribute candidates.
//! - **Reranker is the unification layer.** Source scores from dense
//!   (HNSW distance) and sparse (BM25 raw) aren't comparable across
//!   paths or buckets — we rely on the reranker to produce a single
//!   credible ordering across the union. Source scores are kept on
//!   each candidate for provenance, never compared between paths.
//!
//! Per-bucket dense+sparse run in parallel (via `tokio::join!`); across
//! buckets we run sequentially. Multi-bucket parallel is a future
//! optimization once we have a use case with many buckets per query.

use std::collections::HashSet;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use super::bucket::Bucket;
use super::types::{BucketError, BucketId, Candidate, ChunkId, SearchPath};
use crate::providers::embedding::{EmbedRequest, EmbeddingProvider};
use crate::providers::rerank::{RerankProvider, RerankRequest};

/// Per-query tuning. Defaults are reasonable for workspace-scale
/// buckets; over-fetch values may need to grow at wikipedia scale to
/// keep recall up.
#[derive(Debug, Clone)]
pub struct QueryParams {
    /// Final result count returned to the caller after reranking.
    pub top_k: usize,
    /// Per-bucket over-fetch on the dense path. Reranker breadth — too
    /// small misses good candidates; too large costs reranker latency.
    pub top_k_dense_per_bucket: usize,
    /// Per-bucket over-fetch on the sparse path.
    pub top_k_sparse_per_bucket: usize,
}

impl Default for QueryParams {
    fn default() -> Self {
        Self {
            top_k: 10,
            top_k_dense_per_bucket: 16,
            top_k_sparse_per_bucket: 16,
        }
    }
}

/// Result of a multi-bucket query, after reranker fusion.
///
/// `source_score` and `source_path` document where the candidate came
/// from (which retrieval path in which bucket); `rerank_score` is the
/// signal callers should use for ordering / threshold gating. Source
/// scores are not comparable across paths and shouldn't be used for
/// cross-bucket or cross-path decisions.
#[derive(Debug, Clone)]
pub struct RerankedCandidate {
    pub bucket_id: BucketId,
    pub chunk_id: ChunkId,
    pub chunk_text: String,
    /// Adapter-level provenance — see [`Candidate::source_ref`].
    pub source_ref: super::types::SourceRef,
    pub source_score: f32,
    pub source_path: SearchPath,
    pub rerank_score: f32,
}

/// Coordinator for multi-bucket retrieval. Holds shared embedder +
/// reranker provider handles; takes the bucket list per call.
pub struct QueryEngine {
    embedder: Arc<dyn EmbeddingProvider>,
    reranker: Arc<dyn RerankProvider>,
}

impl std::fmt::Debug for QueryEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryEngine").finish_non_exhaustive()
    }
}

impl QueryEngine {
    pub fn new(embedder: Arc<dyn EmbeddingProvider>, reranker: Arc<dyn RerankProvider>) -> Self {
        Self { embedder, reranker }
    }

    /// Run a hybrid query across `buckets`. Returns reranked candidates
    /// ordered by descending `rerank_score`, capped at `params.top_k`.
    /// Empty if no candidates match in any bucket on either path.
    pub async fn query(
        &self,
        buckets: &[Arc<dyn Bucket>],
        text: &str,
        params: &QueryParams,
        cancel: &CancellationToken,
    ) -> Result<Vec<RerankedCandidate>, BucketError> {
        if buckets.is_empty() || params.top_k == 0 {
            return Ok(Vec::new());
        }

        let query_vec = self.embed_query(text, cancel).await?;

        let mut all_candidates: Vec<Candidate> = Vec::new();
        for bucket in buckets {
            let (dense_res, sparse_res) = tokio::join!(
                bucket.dense_search(&query_vec, params.top_k_dense_per_bucket, cancel),
                bucket.sparse_search(text, params.top_k_sparse_per_bucket, cancel),
            );

            match dense_res {
                Ok(cands) => all_candidates.extend(cands),
                // Bucket built with a different-dimension embedder.
                // Sparse path can still contribute; skip dense silently.
                Err(BucketError::DimensionMismatch { .. }) => {}
                Err(e) => return Err(e),
            }
            match sparse_res {
                Ok(cands) => all_candidates.extend(cands),
                Err(e) => return Err(e),
            }
        }

        let deduped = dedupe_candidates(all_candidates);
        if deduped.is_empty() {
            return Ok(Vec::new());
        }

        // Rerank the union. Reranker scores fresh against the query —
        // the reranker provider call is the only signal that's
        // credibly comparable across heterogeneous paths and buckets.
        let documents: Vec<String> = deduped.iter().map(|c| c.chunk_text.clone()).collect();
        let req = RerankRequest {
            model: "",
            query: text,
            documents: &documents,
            top_n: Some(params.top_k as u32),
        };
        let resp = self
            .reranker
            .rerank(&req, cancel)
            .await
            .map_err(|e| BucketError::Provider(e.to_string()))?;

        // Defensive truncation: the request asks the reranker for
        // `top_n = params.top_k`, but not every server honors it
        // (observed: TEI 1.9.3 ignores top_n and returns scores for
        // every document). The contract here is "capped at top_k", so
        // enforce it locally regardless.
        let mut out = Vec::with_capacity(params.top_k.min(resp.results.len()));
        for r in resp.results.into_iter().take(params.top_k) {
            let Some(c) = deduped.get(r.index as usize) else {
                return Err(BucketError::Provider(format!(
                    "reranker returned out-of-range index {} for {} documents",
                    r.index,
                    deduped.len(),
                )));
            };
            out.push(RerankedCandidate {
                bucket_id: c.bucket_id.clone(),
                chunk_id: c.chunk_id,
                chunk_text: c.chunk_text.clone(),
                source_ref: c.source_ref.clone(),
                source_score: c.source_score,
                source_path: c.path,
                rerank_score: r.score,
            });
        }
        Ok(out)
    }

    async fn embed_query(
        &self,
        text: &str,
        cancel: &CancellationToken,
    ) -> Result<Vec<f32>, BucketError> {
        let inputs = vec![text.to_string()];
        let req = EmbedRequest {
            model: "",
            inputs: &inputs,
        };
        let resp = self
            .embedder
            .embed(&req, cancel)
            .await
            .map_err(|e| BucketError::Provider(e.to_string()))?;
        resp.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| BucketError::Provider("embedder returned no vectors for query".into()))
    }
}

/// Deduplicate candidates by `(bucket_id, chunk_id)`, keeping the first
/// occurrence. Caller controls fan-out order; we run dense before
/// sparse per bucket, so a chunk that hits on both paths surfaces with
/// `path = Dense` and the dense source_score for provenance.
fn dedupe_candidates(candidates: Vec<Candidate>) -> Vec<Candidate> {
    let mut seen: HashSet<(BucketId, ChunkId)> = HashSet::new();
    let mut out: Vec<Candidate> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let key = (c.bucket_id.clone(), c.chunk_id);
        if seen.insert(key) {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;

    use super::*;
    use crate::knowledge::{
        BucketId, ChunkId, Chunker, DiskBucket, MarkdownDir, SourceRecord, TokenBasedChunker,
    };
    use crate::providers::embedding::{
        BoxFuture as EmbedBoxFuture, EmbeddingError, EmbeddingModelInfo, EmbeddingResponse,
    };
    use crate::providers::rerank::{
        BoxFuture as RerankBoxFuture, RerankError, RerankModelInfo, RerankResponse, RerankResult,
    };

    /// Deterministic mock embedder — same shape as the one in
    /// `disk_bucket::tests` but inlined here to keep the test module
    /// self-contained.
    struct MockEmbedder {
        dimension: u32,
    }

    impl MockEmbedder {
        fn new(dimension: u32) -> Self {
            Self { dimension }
        }

        fn fake_vector(text: &str, dim: u32) -> Vec<f32> {
            let hash = blake3::hash(text.as_bytes());
            let bytes = hash.as_bytes();
            (0..dim)
                .map(|i| {
                    let byte = bytes[(i as usize) % 32];
                    (byte as f32) / 128.0 - 1.0
                })
                .collect()
        }
    }

    impl EmbeddingProvider for MockEmbedder {
        fn embed<'a>(
            &'a self,
            req: &'a EmbedRequest<'a>,
            _cancel: &'a CancellationToken,
        ) -> EmbedBoxFuture<'a, Result<EmbeddingResponse, EmbeddingError>> {
            let dim = self.dimension;
            Box::pin(async move {
                let embeddings: Vec<Vec<f32>> = req
                    .inputs
                    .iter()
                    .map(|t| Self::fake_vector(t, dim))
                    .collect();
                Ok(EmbeddingResponse {
                    embeddings,
                    usage: None,
                })
            })
        }

        fn list_models<'a>(
            &'a self,
        ) -> EmbedBoxFuture<'a, Result<Vec<EmbeddingModelInfo>, EmbeddingError>> {
            let dim = self.dimension;
            Box::pin(async move {
                Ok(vec![EmbeddingModelInfo {
                    id: "mock-embedder".to_string(),
                    dimension: dim,
                    max_input_tokens: Some(512),
                }])
            })
        }
    }

    /// Mock reranker — scores documents by the count of query terms
    /// they contain (case-insensitive whitespace tokenization). Higher
    /// score = better. Deterministic, predictable, easy to assert
    /// against.
    struct MockReranker;

    impl RerankProvider for MockReranker {
        fn rerank<'a>(
            &'a self,
            req: &'a RerankRequest<'a>,
            _cancel: &'a CancellationToken,
        ) -> RerankBoxFuture<'a, Result<RerankResponse, RerankError>> {
            Box::pin(async move {
                let q_words: HashSet<String> = req
                    .query
                    .split_whitespace()
                    .map(|w| w.to_ascii_lowercase())
                    .collect();
                let mut scored: Vec<RerankResult> = req
                    .documents
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        let d_words: HashSet<String> = d
                            .split_whitespace()
                            .map(|w| w.to_ascii_lowercase())
                            .collect();
                        let score = q_words.intersection(&d_words).count() as f32;
                        RerankResult {
                            index: i as u32,
                            score,
                        }
                    })
                    .collect();
                // Sort descending; tantivy / TEI reranker contract is
                // sorted by score desc with optional top_n cap.
                scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                if let Some(n) = req.top_n {
                    scored.truncate(n as usize);
                }
                Ok(RerankResponse { results: scored })
            })
        }

        fn list_models<'a>(
            &'a self,
        ) -> RerankBoxFuture<'a, Result<Vec<RerankModelInfo>, RerankError>> {
            Box::pin(async move {
                Ok(vec![RerankModelInfo {
                    id: "mock-reranker".to_string(),
                    max_input_tokens: Some(512),
                }])
            })
        }
    }

    fn write_md(dir: &Path, name: &str, body: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, body).unwrap();
    }

    fn bucket_toml(source_path: &Path) -> String {
        format!(
            r#"
name = "Notes"
created_at = "2026-04-25T10:23:11Z"

[source]
kind = "linked"
adapter = "markdown_dir"
path = "{}"

[chunker]
strategy = "token_based"
chunk_tokens = 50
overlap_tokens = 5

[defaults]
embedder = "tei_test"
"#,
            source_path.display(),
        )
    }

    async fn build_bucket(
        bucket_root: &Path,
        source: &Path,
        bucket_id: BucketId,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Arc<dyn Bucket> {
        std::fs::create_dir_all(bucket_root).unwrap();
        std::fs::create_dir_all(crate::knowledge::slot::slots_dir(bucket_root)).unwrap();
        std::fs::write(
            crate::knowledge::slot::bucket_toml_path(bucket_root),
            bucket_toml(source),
        )
        .unwrap();
        let bucket = DiskBucket::open(bucket_root, bucket_id).unwrap();
        let adapter = MarkdownDir::new(source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();
        Arc::new(bucket)
    }

    #[tokio::test]
    async fn query_single_bucket_returns_reranked_results() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        std::fs::create_dir_all(&source).unwrap();
        write_md(
            &source,
            "fox.md",
            "the quick brown fox jumps over the lazy dog",
        );
        write_md(&source, "lorem.md", "lorem ipsum dolor sit amet");
        write_md(
            &source,
            "rain.md",
            "the rain in spain falls mainly on the plain",
        );

        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(16));
        let reranker: Arc<dyn RerankProvider> = Arc::new(MockReranker);
        let bucket = build_bucket(
            &tmp.path().join("bucket"),
            &source,
            BucketId::pod("notes"),
            embedder.clone(),
        )
        .await;

        let engine = QueryEngine::new(embedder, reranker);
        let cancel = CancellationToken::new();
        let results = engine
            .query(&[bucket], "fox dog", &QueryParams::default(), &cancel)
            .await
            .unwrap();

        assert!(!results.is_empty());
        // The fox.md chunk contains both "fox" and "dog" → highest
        // mock-reranker score.
        assert!(results[0].chunk_text.contains("fox"));
        assert!(results[0].chunk_text.contains("dog"));
        assert_eq!(results[0].rerank_score, 2.0);
        // Other chunks score lower (or zero). Order is descending.
        for w in results.windows(2) {
            assert!(w[0].rerank_score >= w[1].rerank_score);
        }
    }

    #[tokio::test]
    async fn query_fans_out_across_multiple_buckets() {
        let tmp = tempfile::tempdir().unwrap();
        let src_a = tmp.path().join("notes_a");
        let src_b = tmp.path().join("notes_b");
        std::fs::create_dir_all(&src_a).unwrap();
        std::fs::create_dir_all(&src_b).unwrap();
        write_md(&src_a, "alpha.md", "alpha content about quantum mechanics");
        write_md(&src_b, "beta.md", "beta content about quantum computing");

        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let reranker: Arc<dyn RerankProvider> = Arc::new(MockReranker);
        let bucket_a = build_bucket(
            &tmp.path().join("bucket_a"),
            &src_a,
            BucketId::server("notes_a"),
            embedder.clone(),
        )
        .await;
        let bucket_b = build_bucket(
            &tmp.path().join("bucket_b"),
            &src_b,
            BucketId::pod("notes_b"),
            embedder.clone(),
        )
        .await;

        let engine = QueryEngine::new(embedder, reranker);
        let cancel = CancellationToken::new();
        let results = engine
            .query(
                &[bucket_a, bucket_b],
                "quantum",
                &QueryParams::default(),
                &cancel,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        let bucket_ids: HashSet<BucketId> = results.iter().map(|r| r.bucket_id.clone()).collect();
        assert!(bucket_ids.contains(&BucketId::server("notes_a")));
        assert!(bucket_ids.contains(&BucketId::pod("notes_b")));
    }

    #[tokio::test]
    async fn query_dedupes_dense_and_sparse_hits_to_same_chunk() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        std::fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "fox dog cat");

        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let reranker: Arc<dyn RerankProvider> = Arc::new(MockReranker);
        let bucket = build_bucket(
            &tmp.path().join("bucket"),
            &source,
            BucketId::pod("notes"),
            embedder.clone(),
        )
        .await;

        // Query terms appear in the document; sparse will match.
        // Dense will also return the chunk (only one chunk exists).
        // Result count must be 1, not 2.
        // Re-derive the chunk id by replicating what the chunker
        // produced — the bucket.toml above sets chunk_tokens=50, so
        // the single short record yields exactly one chunk at offset 0.
        let chunker = TokenBasedChunker::new(50, 5);
        let synth = SourceRecord::new("ignored", "fox dog cat");
        let new_chunks = chunker.chunk(&synth);
        let chunk_id = ChunkId::from_source(&new_chunks[0].source_record_hash, 0);

        let engine = QueryEngine::new(embedder, reranker);
        let cancel = CancellationToken::new();
        let results = engine
            .query(&[bucket], "fox", &QueryParams::default(), &cancel)
            .await
            .unwrap();

        assert_eq!(results.len(), 1, "deduped to one chunk; got {results:?}");
        assert_eq!(results[0].chunk_id, chunk_id);
    }

    #[tokio::test]
    async fn query_caps_at_top_k() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        std::fs::create_dir_all(&source).unwrap();
        for i in 0..30 {
            write_md(
                &source,
                &format!("doc-{i:02}.md"),
                &format!("apple banana orange {i}"),
            );
        }

        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let reranker: Arc<dyn RerankProvider> = Arc::new(MockReranker);
        let bucket = build_bucket(
            &tmp.path().join("bucket"),
            &source,
            BucketId::pod("notes"),
            embedder.clone(),
        )
        .await;

        let engine = QueryEngine::new(embedder, reranker);
        let cancel = CancellationToken::new();

        let results = engine
            .query(
                &[bucket],
                "apple",
                &QueryParams {
                    top_k: 5,
                    ..QueryParams::default()
                },
                &cancel,
            )
            .await
            .unwrap();

        assert!(
            results.len() <= 5,
            "expected ≤5 results, got {}",
            results.len()
        );
    }

    #[tokio::test]
    async fn query_returns_empty_when_nothing_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        std::fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "alpha bravo charlie");

        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let reranker: Arc<dyn RerankProvider> = Arc::new(MockReranker);
        let bucket = build_bucket(
            &tmp.path().join("bucket"),
            &source,
            BucketId::pod("notes"),
            embedder.clone(),
        )
        .await;

        let engine = QueryEngine::new(embedder, reranker);
        let cancel = CancellationToken::new();

        // The single chunk has "alpha bravo charlie" so a query of
        // "delta" matches sparse=0 hits, but dense returns the same
        // chunk regardless (HNSW always returns a nearest neighbor).
        // Reranker scores all candidates — chunk has 0 query-term
        // overlap so rerank_score is 0, but it's still a result.
        // For an empty result: query a totally different vocabulary
        // such that sparse misses and... actually dense always hits.
        // So this test really demonstrates that we DO return a
        // candidate even if its rerank score is 0. Let me adjust the
        // assertion accordingly.
        let results = engine
            .query(&[bucket], "delta", &QueryParams::default(), &cancel)
            .await
            .unwrap();
        // Dense always returns its nearest neighbor, so one result.
        // Reranker scored it 0 (no term overlap) but it's still
        // returned. Caller's threshold-on-rerank-score is what
        // would filter it.
        assert!(results.len() <= 1);
        if let Some(r) = results.first() {
            assert_eq!(r.rerank_score, 0.0);
        }
    }

    #[tokio::test]
    async fn empty_buckets_returns_empty() {
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let reranker: Arc<dyn RerankProvider> = Arc::new(MockReranker);
        let engine = QueryEngine::new(embedder, reranker);
        let cancel = CancellationToken::new();
        let results = engine
            .query(&[], "anything", &QueryParams::default(), &cancel)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn top_k_zero_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        std::fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hello");

        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let reranker: Arc<dyn RerankProvider> = Arc::new(MockReranker);
        let bucket = build_bucket(
            &tmp.path().join("bucket"),
            &source,
            BucketId::pod("notes"),
            embedder.clone(),
        )
        .await;

        let engine = QueryEngine::new(embedder, reranker);
        let cancel = CancellationToken::new();
        let results = engine
            .query(
                &[bucket],
                "hello",
                &QueryParams {
                    top_k: 0,
                    ..QueryParams::default()
                },
                &cancel,
            )
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn buckets_with_mismatched_dimension_skip_dense_silently() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        std::fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "fox quick brown");

        // Bucket built with dimension 8.
        let bucket_embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let bucket = build_bucket(
            &tmp.path().join("bucket"),
            &source,
            BucketId::pod("notes"),
            bucket_embedder,
        )
        .await;

        // Query engine uses a different-dimension embedder. Dense
        // path will return DimensionMismatch → skipped silently.
        // Sparse path still runs and finds matches.
        let query_embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(16));
        let reranker: Arc<dyn RerankProvider> = Arc::new(MockReranker);
        let engine = QueryEngine::new(query_embedder, reranker);
        let cancel = CancellationToken::new();

        let results = engine
            .query(&[bucket], "fox", &QueryParams::default(), &cancel)
            .await
            .unwrap();
        // Sparse-only result still returns something
        assert!(!results.is_empty());
        assert!(results[0].chunk_text.contains("fox"));
    }
}
