//! One-off binary: measure dense-path recall@K of one bucket against
//! another. Built for quantization validation — compare an f32
//! baseline against an f16 / int8 candidate with the same source data
//! to bound the recall loss before committing to a multi-day
//! production rebuild.
//!
//! The harness is dense-path only on purpose. Sparse (BM25) is
//! quantization-independent — it operates on tokenized text, not
//! vectors — so including it would dilute the signal. Reranker fusion
//! is also skipped: the reranker scores the union of candidates, and
//! a "candidate that survived f32's top-K but not f16's" is exactly
//! what we want to measure, before fusion smooths it over.
//!
//! Usage:
//!
//!   cargo run --release --bin recall_eval -- \
//!     --baseline-bucket-root /data/buckets \
//!     --baseline-bucket-id wikipedia_simple \
//!     --candidate-bucket-root /data/buckets \
//!     --candidate-bucket-id wikipedia_simple_f16 \
//!     --tei-embedder-url http://10.1.0.198:8080 \
//!     --top-k 10
//!
//! Output: per-query overlap@K, aggregate mean / median / min, and
//! the worst-case query so you can sanity-check it manually.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use tokio_util::sync::CancellationToken;
use whisper_agent::knowledge::{Bucket, BucketId, ChunkId, DiskBucket};
use whisper_agent::providers::embedding::{EmbedRequest, EmbeddingProvider};
use whisper_agent::providers::tei::TeiEmbeddingClient;

#[derive(Parser, Debug)]
#[command(about = "Measure dense recall@K of one bucket against another.")]
struct Args {
    /// Bucket-roots dir containing the baseline bucket subdir.
    #[arg(long)]
    baseline_bucket_root: PathBuf,

    /// Bucket id (subdir name) of the baseline bucket. The baseline is
    /// what the candidate is measured against — typically the f32
    /// build; the candidate is the quantized variant.
    #[arg(long)]
    baseline_bucket_id: String,

    /// Bucket-roots dir containing the candidate bucket subdir.
    #[arg(long)]
    candidate_bucket_root: PathBuf,

    /// Bucket id (subdir name) of the candidate bucket.
    #[arg(long)]
    candidate_bucket_id: String,

    /// TEI `/embed` endpoint. Both buckets must be queryable with
    /// embeddings from this provider (i.e. they were built against
    /// it, or against a model with the same dimension).
    #[arg(long)]
    tei_embedder_url: String,

    /// Optional bearer token for the embedder.
    #[arg(long)]
    tei_embedder_auth: Option<String>,

    /// Top-K to compare. Recall@K = |baseline_topK ∩ candidate_topK| / K.
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// Path to a newline-separated query file. Each non-empty line is
    /// one query. Blank lines and `#`-prefixed comments are skipped.
    /// When omitted, a small built-in simplewiki-flavored list is
    /// used — fine for smoke-testing the harness, but real validation
    /// should use a domain-specific list.
    #[arg(long)]
    queries_file: Option<PathBuf>,

    /// Print the chunk_id and bucket-side score for each candidate
    /// in baseline∖candidate (the missed ones). Helpful for
    /// inspecting whether quantization is dropping near-tied
    /// candidates (recoverable with a slightly larger ef_search) or
    /// genuinely losing distant ones.
    #[arg(long, default_value_t = false)]
    show_misses: bool,
}

/// Default queries — broad coverage of simplewiki topics so the
/// harness produces a useful signal on day one. Replace via
/// `--queries-file` for domain-specific runs.
const DEFAULT_QUERIES: &[&str] = &[
    "albert einstein theory of relativity",
    "world war two pacific theater",
    "python programming language",
    "octopus intelligence and learning",
    "constitution of the united states",
    "photosynthesis chloroplast process",
    "french revolution causes",
    "great wall of china history",
    "human immune system white blood cells",
    "black holes event horizon",
    "shakespeare hamlet plot",
    "mitochondria cellular respiration",
    "earthquakes plate tectonics",
    "renaissance italian art",
    "quantum mechanics wave function",
];

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args = Args::parse();

    let queries: Vec<String> = match &args.queries_file {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("read --queries-file {}", path.display()))?
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(String::from)
            .collect(),
        None => DEFAULT_QUERIES.iter().map(|s| s.to_string()).collect(),
    };
    if queries.is_empty() {
        anyhow::bail!("no queries (file is empty or only comments)");
    }

    let baseline_dir = args.baseline_bucket_root.join(&args.baseline_bucket_id);
    let candidate_dir = args.candidate_bucket_root.join(&args.candidate_bucket_id);
    let baseline = Arc::new(
        DiskBucket::open(
            baseline_dir.clone(),
            BucketId::server(args.baseline_bucket_id.clone()),
        )
        .with_context(|| format!("open baseline {}", baseline_dir.display()))?,
    );
    let candidate = Arc::new(
        DiskBucket::open(
            candidate_dir.clone(),
            BucketId::server(args.candidate_bucket_id.clone()),
        )
        .with_context(|| format!("open candidate {}", candidate_dir.display()))?,
    );

    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(TeiEmbeddingClient::new(
        args.tei_embedder_url.clone(),
        args.tei_embedder_auth.clone(),
    ));

    let cancel = CancellationToken::new();

    println!("baseline:  {}", baseline_dir.display());
    println!("candidate: {}", candidate_dir.display());
    println!("embedder:  {}", args.tei_embedder_url);
    println!("queries:   {}  top_k: {}", queries.len(), args.top_k);
    if baseline_dir == candidate_dir {
        println!();
        println!("note: baseline == candidate — this run measures the HNSW search noise floor.",);
        println!("      hnsw_rs's multi-threaded search isn't deterministic on tied scores; mean",);
        println!("      recall here is the floor below which quantization regressions can't be",);
        println!("      distinguished from search noise.",);
    }
    println!();
    println!("{:<60}  baseline_ms  candidate_ms  recall@k", "query");
    println!("{}", "-".repeat(96));

    let mut overlaps: Vec<f32> = Vec::with_capacity(queries.len());
    let mut total_baseline_ms: u128 = 0;
    let mut total_candidate_ms: u128 = 0;
    let mut worst: Option<(String, f32, Vec<ChunkId>, Vec<ChunkId>)> = None;

    for q in &queries {
        // Single embed call: same vector goes to both buckets so any
        // recall delta comes from the index, not from
        // embedder-call-to-call variance.
        let req = EmbedRequest {
            model: "",
            inputs: std::slice::from_ref(q),
        };
        let resp = embedder
            .embed(&req, &cancel)
            .await
            .with_context(|| format!("embed `{q}`"))?;
        let qvec = resp
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty embedding response for `{q}`"))?;

        let t = Instant::now();
        let b_hits = baseline
            .dense_search(&qvec, args.top_k, &cancel)
            .await
            .with_context(|| format!("baseline.dense_search `{q}`"))?;
        let baseline_ms = t.elapsed().as_millis();
        total_baseline_ms += baseline_ms;

        let t = Instant::now();
        let c_hits = candidate
            .dense_search(&qvec, args.top_k, &cancel)
            .await
            .with_context(|| format!("candidate.dense_search `{q}`"))?;
        let candidate_ms = t.elapsed().as_millis();
        total_candidate_ms += candidate_ms;

        let b_ids: Vec<ChunkId> = b_hits.iter().map(|c| c.chunk_id).collect();
        let c_ids: Vec<ChunkId> = c_hits.iter().map(|c| c.chunk_id).collect();

        let recall = recall_at_k(&b_ids, &c_ids, args.top_k);
        overlaps.push(recall);

        let q_disp: String = if q.chars().count() > 58 {
            let mut s: String = q.chars().take(55).collect();
            s.push_str("...");
            s
        } else {
            q.clone()
        };
        println!(
            "{:<60}  {:>11}  {:>12}  {:>5.2}",
            q_disp, baseline_ms, candidate_ms, recall
        );

        if worst
            .as_ref()
            .map(|(_, r, _, _)| recall < *r)
            .unwrap_or(true)
        {
            worst = Some((q.clone(), recall, b_ids, c_ids));
        }
    }

    let n = overlaps.len() as f32;
    let mean = overlaps.iter().sum::<f32>() / n;
    let mut sorted = overlaps.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let min = sorted.first().copied().unwrap_or(0.0);
    let perfect = overlaps.iter().filter(|r| **r >= 0.999).count();

    println!();
    println!("aggregate:");
    println!("  mean recall@{}: {mean:.3}", args.top_k);
    println!("  median:           {median:.3}");
    println!("  min:              {min:.3}");
    println!(
        "  perfect:          {perfect}/{}  ({:.0}% queries with recall = 1.0)",
        queries.len(),
        100.0 * perfect as f32 / queries.len() as f32
    );
    println!(
        "  total time:       baseline {} ms, candidate {} ms",
        total_baseline_ms, total_candidate_ms,
    );

    if args.show_misses
        && let Some((q, r, b_ids, c_ids)) = worst
    {
        println!();
        println!("worst query (recall {r:.3}): `{q}`");
        let c_set: std::collections::HashSet<&ChunkId> = c_ids.iter().collect();
        let missed: Vec<&ChunkId> = b_ids.iter().filter(|id| !c_set.contains(id)).collect();
        println!("  baseline-only chunks ({}):", missed.len());
        for (i, id) in missed.iter().enumerate() {
            println!("    {i}: {id}");
        }
    }

    Ok(())
}

/// Recall@K = |intersection of baseline_top_k and candidate_top_k| / K.
/// Both inputs must already be capped at K (the dense_search call
/// does this); using actual lengths makes the metric robust if a
/// bucket happened to have fewer than K total hits.
fn recall_at_k(baseline: &[ChunkId], candidate: &[ChunkId], k: usize) -> f32 {
    if baseline.is_empty() {
        // No baseline hits = no measurement signal. Return 1.0 so
        // empty-result queries don't drag the aggregate down — the
        // user should manually inspect those if they're suspicious.
        return 1.0;
    }
    let cset: std::collections::HashSet<&ChunkId> = candidate.iter().collect();
    let inter = baseline.iter().filter(|id| cset.contains(id)).count();
    let denom = baseline.len().min(k) as f32;
    inter as f32 / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> ChunkId {
        ChunkId([byte; 32])
    }

    #[test]
    fn recall_perfect_match_scores_one() {
        let a = vec![id(1), id(2), id(3)];
        let b = vec![id(1), id(2), id(3)];
        assert_eq!(recall_at_k(&a, &b, 3), 1.0);
    }

    #[test]
    fn recall_no_overlap_scores_zero() {
        let a = vec![id(1), id(2), id(3)];
        let b = vec![id(4), id(5), id(6)];
        assert_eq!(recall_at_k(&a, &b, 3), 0.0);
    }

    #[test]
    fn recall_partial_overlap_is_fraction_of_baseline() {
        let a = vec![id(1), id(2), id(3), id(4)];
        let b = vec![id(1), id(2), id(99), id(98)];
        // 2 of baseline's 4 ids appear in candidate → 0.5.
        assert_eq!(recall_at_k(&a, &b, 4), 0.5);
    }

    #[test]
    fn recall_ignores_candidate_only_extras() {
        // Candidate has more hits than the K we measure at — extras
        // beyond the truncation aren't penalized; only baseline-side
        // misses count.
        let a = vec![id(1), id(2)];
        let b = vec![id(1), id(2), id(3), id(4)];
        assert_eq!(recall_at_k(&a, &b, 2), 1.0);
    }

    #[test]
    fn recall_empty_baseline_returns_one() {
        let a: Vec<ChunkId> = Vec::new();
        let b = vec![id(1)];
        assert_eq!(recall_at_k(&a, &b, 10), 1.0);
    }
}
