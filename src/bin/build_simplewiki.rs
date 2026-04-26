//! One-off binary: build a knowledge bucket from a Simple English
//! Wikipedia dump (or any MediaWiki XML export) end-to-end.
//!
//! This is intentionally NOT part of the long-term API surface — it
//! exists to drive the bucket pipeline at non-toy scale before the
//! WebUI lifecycle slice (8c) lands. Once the WebUI can trigger
//! builds, this binary becomes obsolete.
//!
//! Usage examples:
//!
//!   # Build with the deterministic mock embedder (offline, no TEI).
//!   cargo run --release --bin build_simplewiki -- \
//!     --archive /tmp/simplewiki/simplewiki-latest-pages-articles.xml.bz2 \
//!     --bucket-root /tmp/simplewiki/buckets \
//!     --max-pages 5000
//!
//!   # Same archive, then run a sample query against the freshly-built bucket.
//!   cargo run --release --bin build_simplewiki -- \
//!     --archive /tmp/simplewiki/simplewiki-latest-pages-articles.xml.bz2 \
//!     --bucket-root /tmp/simplewiki/buckets \
//!     --max-pages 5000 \
//!     --query "albert einstein"
//!
//! The mock embedder is the only path supported today. When a TEI
//! endpoint shows up in the test environment, swap it in via
//! `TeiEmbeddingClient::new(url, None)` — the rest of the pipeline is
//! provider-agnostic.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use tokio_util::sync::CancellationToken;
use whisper_agent::knowledge::{
    Bucket, BucketId, DiskBucket, MediaWikiXml, QueryEngine, QueryParams, SourceAdapter,
    SourceError, SourceRecord, TokenBasedChunker,
};
use whisper_agent::providers::embedding::{
    BoxFuture as EmbedBoxFuture, EmbedRequest, EmbeddingError, EmbeddingModelInfo,
    EmbeddingProvider, EmbeddingResponse,
};
use whisper_agent::providers::rerank::{
    BoxFuture as RerankBoxFuture, RerankError, RerankModelInfo, RerankProvider, RerankRequest,
    RerankResponse, RerankResult,
};
use whisper_agent::providers::tei::{TeiEmbeddingClient, TeiRerankClient};

#[derive(Parser, Debug)]
#[command(about = "Build a knowledge bucket from a MediaWiki XML dump.")]
struct Args {
    /// Path to the dump file. `.xml` or `.xml.bz2` (extension is the
    /// only signal — content sniffing isn't done).
    #[arg(long)]
    archive: PathBuf,

    /// Directory under which the bucket lives. The bucket's directory
    /// is `<bucket_root>/<bucket_id>/`.
    #[arg(long)]
    bucket_root: PathBuf,

    /// Bucket id (== directory name). Must be ASCII alphanumeric +
    /// `_`/`-`/`.` per the registry's name validation.
    #[arg(long, default_value = "simplewiki")]
    bucket_id: String,

    /// Stop after this many kept pages flow through the adapter. `0`
    /// = no limit. Useful for smoke runs that finish in seconds
    /// without enumerating the whole 250k-article dump.
    #[arg(long, default_value_t = 0)]
    max_pages: u64,

    /// Embedding dimension for the mock embedder. Picks a number
    /// representative of small open-source embedders; not load-bearing
    /// since the dense path is meaningless without a real embedder.
    /// Ignored when --tei-embedder-url is set.
    #[arg(long, default_value_t = 384)]
    mock_embedder_dim: u32,

    /// TEI `/embed` endpoint URL. When set, replaces the deterministic
    /// mock embedder with a real call-out to TEI for every chunk batch.
    /// E.g. `http://10.1.0.198:8080`. Required to validate the dense
    /// retrieval path; the mock embedder produces semantically random
    /// vectors so dense hits look meaningless.
    #[arg(long)]
    tei_embedder_url: Option<String>,

    /// TEI `/rerank` endpoint URL. When set, replaces the whitespace-
    /// overlap mock reranker with a real call-out to TEI for the
    /// reranker fusion stage. Independent of `--tei-embedder-url` —
    /// can mix-and-match for ablation runs.
    #[arg(long)]
    tei_reranker_url: Option<String>,

    /// Sample query to run after the build. Not a flag-takes-vec because
    /// quoting multi-arg queries from the shell is simpler with one
    /// long string per `--query` repetition.
    #[arg(long)]
    query: Vec<String>,

    /// How many results to surface per query.
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Drop and recreate the bucket directory before building. Off by
    /// default so accidental reruns don't nuke a long build.
    #[arg(long, default_value_t = false)]
    fresh: bool,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let bucket_dir = args.bucket_root.join(&args.bucket_id);

    if args.fresh && bucket_dir.exists() {
        eprintln!(
            "--fresh: removing existing bucket dir {}",
            bucket_dir.display()
        );
        std::fs::remove_dir_all(&bucket_dir).context("remove existing bucket dir")?;
    }
    std::fs::create_dir_all(&bucket_dir).context("create bucket dir")?;

    write_bucket_toml(&bucket_dir, &args.bucket_id)?;

    let bucket = DiskBucket::open(bucket_dir.clone(), BucketId::server(args.bucket_id.clone()))
        .context("DiskBucket::open")?;

    let cancel = CancellationToken::new();
    let mock_embedder = Arc::new(MockEmbedder::new(args.mock_embedder_dim));
    // Pick the real or mock embedder. We keep `mock_embedder` around
    // for its `embed_calls` counter even when not used as the bucket
    // embedder — it just stays at 0 in that case.
    let embedder: Arc<dyn EmbeddingProvider> = match &args.tei_embedder_url {
        Some(url) => {
            eprintln!("using TEI embedder at {url}");
            Arc::new(TeiEmbeddingClient::new(url.clone(), None))
        }
        None => {
            eprintln!(
                "using mock embedder (dim={}); dense path will be semantically random",
                args.mock_embedder_dim,
            );
            mock_embedder.clone()
        }
    };
    let chunker = TokenBasedChunker::new(500, 50);
    let source = LoggingSource::new(MediaWikiXml::new(&args.archive), args.max_pages);

    eprintln!(
        "building bucket id={} from archive {} (max_pages={})",
        args.bucket_id,
        args.archive.display(),
        if args.max_pages == 0 {
            "unlimited".to_string()
        } else {
            args.max_pages.to_string()
        }
    );
    let t0 = Instant::now();
    let slot_id = bucket
        .build_slot(&source, &chunker, embedder.clone(), &cancel)
        .await
        .context("build_slot")?;
    let build_dur = t0.elapsed();

    let pages_emitted = source.kept.load(Ordering::SeqCst);
    let pages_skipped = source.skipped.load(Ordering::SeqCst);
    let read_errors = source.errors.load(Ordering::SeqCst);
    let embed_calls = mock_embedder.embed_calls.load(Ordering::SeqCst);
    let bucket_disk_size = dir_size(&bucket_dir).unwrap_or(0);

    eprintln!();
    eprintln!("=== build summary ===");
    eprintln!("slot_id              {slot_id}");
    eprintln!("elapsed              {:.1}s", build_dur.as_secs_f64());
    eprintln!("pages emitted        {pages_emitted}");
    eprintln!("pages skipped (filt) {pages_skipped}");
    eprintln!("source read errors   {read_errors}");
    eprintln!("embed batches        {embed_calls}");
    eprintln!(
        "bucket disk size     {:.1} MiB",
        bucket_disk_size as f64 / (1024.0 * 1024.0)
    );

    if !args.query.is_empty() {
        eprintln!();
        eprintln!("=== sample queries ===");
        let reranker: Arc<dyn RerankProvider> = match &args.tei_reranker_url {
            Some(url) => {
                eprintln!("using TEI reranker at {url}");
                Arc::new(TeiRerankClient::new(url.clone(), None))
            }
            None => {
                eprintln!("using mock reranker (whitespace-token overlap)");
                Arc::new(MockReranker)
            }
        };
        let engine = QueryEngine::new(embedder.clone(), reranker);
        let buckets: Vec<Arc<dyn Bucket>> = vec![Arc::new(bucket)];
        for q in &args.query {
            run_query(&engine, &buckets, q, args.top_k, &cancel).await?;
        }
    }

    Ok(())
}

fn write_bucket_toml(bucket_dir: &std::path::Path, bucket_id: &str) -> Result<()> {
    let path = bucket_dir.join("bucket.toml");
    if path.exists() {
        return Ok(());
    }
    let toml = format!(
        r#"name = "{bucket_id}"
description = "Simple English Wikipedia, built via build_simplewiki"
created_at = "{now}"

[source]
kind = "stored"
adapter = "mediawiki_xml"
archive_path = ""

[chunker]
strategy = "token_based"
chunk_tokens = 500
overlap_tokens = 50

[search_paths.dense]
enabled = true
[search_paths.sparse]
enabled = true
tokenizer = "default"

[defaults]
embedder = "mock-embedder"
serving_mode = "ram"
quantization = "f32"
"#,
        bucket_id = bucket_id,
        now = chrono::Utc::now().to_rfc3339(),
    );
    std::fs::write(&path, toml).context("write bucket.toml")?;
    Ok(())
}

async fn run_query(
    engine: &QueryEngine,
    buckets: &[Arc<dyn Bucket>],
    query_text: &str,
    top_k: usize,
    cancel: &CancellationToken,
) -> Result<()> {
    let params = QueryParams {
        top_k,
        ..Default::default()
    };
    let t0 = Instant::now();
    let hits = engine
        .query(buckets, query_text, &params, cancel)
        .await
        .context("query")?;
    let dur = t0.elapsed();
    eprintln!();
    eprintln!("query {:?}  ({:.0}ms)", query_text, dur.as_millis());
    if hits.is_empty() {
        eprintln!("  (no hits)");
        return Ok(());
    }
    for (i, hit) in hits.iter().enumerate() {
        let snippet: String = hit
            .chunk_text
            .chars()
            .take(120)
            .collect::<String>()
            .replace('\n', " ");
        eprintln!(
            "  {:>2}. [{}] {} via={:?} rerank={:.3} src={:.3}",
            i + 1,
            hit.bucket_id,
            short_chunk_id(&hit.chunk_id.to_string()),
            hit.source_path,
            hit.rerank_score,
            hit.source_score,
        );
        eprintln!("       {snippet}");
    }
    Ok(())
}

fn short_chunk_id(s: &str) -> &str {
    if s.len() > 12 { &s[..12] } else { s }
}

fn dir_size(dir: &std::path::Path) -> std::io::Result<u64> {
    let mut total: u64 = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(p) = stack.pop() {
        for entry in std::fs::read_dir(&p)? {
            let entry = entry?;
            let md = entry.metadata()?;
            if md.is_dir() {
                stack.push(entry.path());
            } else {
                total += md.len();
            }
        }
    }
    Ok(total)
}

// ---------- Logging source wrapper ----------

/// Wraps a [`SourceAdapter`] with periodic stderr progress reporting
/// and an optional max-page cap. The wrapped adapter still drives the
/// actual XML walk; this layer only counts and times.
struct LoggingSource<A: SourceAdapter> {
    inner: A,
    max_pages: u64,
    kept: AtomicU64,
    skipped: AtomicU64,
    errors: AtomicU64,
}

impl<A: SourceAdapter> LoggingSource<A> {
    fn new(inner: A, max_pages: u64) -> Self {
        Self {
            inner,
            max_pages,
            kept: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }
}

impl<A: SourceAdapter> SourceAdapter for LoggingSource<A> {
    fn enumerate(&self) -> Box<dyn Iterator<Item = Result<SourceRecord, SourceError>> + Send + '_> {
        let max = self.max_pages;
        let kept = AtomicCounter(&self.kept);
        let skipped = AtomicCounter(&self.skipped);
        let errors = AtomicCounter(&self.errors);
        let started = Instant::now();
        let inner = self.inner.enumerate();
        Box::new(inner.scan(0u64, move |seen, item| {
            *seen += 1;
            if max > 0 && kept.get() >= max {
                // Stop the iterator — `take_while`-style early exit.
                return None;
            }
            match &item {
                Ok(_) => kept.inc(),
                Err(_) => errors.inc(),
            }
            // The wrapped MediaWikiXml only emits records for pages
            // that pass its filter (ns=0, not redirect, non-empty).
            // We don't see the dropped pages here; `skipped` stays 0
            // unless / until we add pre-filter visibility.
            let _ = &skipped;
            if *seen.max(&mut 1) % 1_000 == 0 {
                let elapsed = started.elapsed().as_secs_f64().max(0.001);
                let rate = kept.get() as f64 / elapsed;
                eprintln!(
                    "  …{} pages emitted, {} errors  [{:.0} pages/s]",
                    kept.get(),
                    errors.get(),
                    rate
                );
            }
            Some(item)
        }))
    }
}

/// Tiny `&AtomicU64` newtype with `inc` / `get` shorthand. Lifetime-
/// scoped so the closures returned from `enumerate` can hold a
/// shared reference into `LoggingSource`'s atomics.
struct AtomicCounter<'a>(&'a AtomicU64);

impl AtomicCounter<'_> {
    fn inc(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
    fn get(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

// ---------- Mock embedder + reranker ----------

/// Deterministic content-hash-derived "embedder" — same shape used in
/// `disk_bucket`'s test module, lifted here so this binary doesn't
/// depend on a TEI server. Vectors carry no semantic meaning; the
/// dense path's results are essentially random under this embedder.
struct MockEmbedder {
    dimension: u32,
    embed_calls: AtomicUsize,
}

impl MockEmbedder {
    fn new(dimension: u32) -> Self {
        Self {
            dimension,
            embed_calls: AtomicUsize::new(0),
        }
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
        self.embed_calls.fetch_add(1, Ordering::SeqCst);
        let dim = self.dimension;
        Box::pin(async move {
            let embeddings = req
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
                id: "mock-embedder".into(),
                dimension: dim,
                max_input_tokens: Some(512),
            }])
        })
    }
}

/// Whitespace-token-overlap reranker — same shape as `query.rs`'s
/// test mock, lifted here. Score = (overlapping unique tokens between
/// query and document), so it produces a sensible-ish ranking on real
/// wikitext keywords without any model.
struct MockReranker;

impl RerankProvider for MockReranker {
    fn rerank<'a>(
        &'a self,
        req: &'a RerankRequest<'a>,
        _cancel: &'a CancellationToken,
    ) -> RerankBoxFuture<'a, Result<RerankResponse, RerankError>> {
        let q_tokens: std::collections::HashSet<String> = req
            .query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .collect();
        let mut scored: Vec<RerankResult> = req
            .documents
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let d_tokens: std::collections::HashSet<String> =
                    doc.split_whitespace().map(|t| t.to_lowercase()).collect();
                let overlap = q_tokens.intersection(&d_tokens).count() as f32;
                RerankResult {
                    index: i as u32,
                    score: overlap,
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(top) = req.top_n {
            scored.truncate(top as usize);
        }
        Box::pin(async move { Ok(RerankResponse { results: scored }) })
    }

    fn list_models<'a>(&'a self) -> RerankBoxFuture<'a, Result<Vec<RerankModelInfo>, RerankError>> {
        Box::pin(async move {
            Ok(vec![RerankModelInfo {
                id: "mock-reranker".into(),
                max_input_tokens: Some(512),
            }])
        })
    }
}
