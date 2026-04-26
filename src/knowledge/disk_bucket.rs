//! [`DiskBucket`] — disk-backed implementation of the [`Bucket`] trait.
//!
//! Owns a bucket directory on disk plus the loaded state for the active
//! slot (if any). Slice 6 lights up the sparse (BM25 via tantivy) path
//! alongside the dense (HNSW) path from slice 5. Both indexes are built
//! during `build_slot` and opened on slot load; both `dense_search` and
//! `sparse_search` produce real candidates. Mutation methods still
//! error pending the slot-mutation primitives slice.
//!
//! Concurrency model: queries take an [`Arc`]-shared snapshot of the
//! active slot's [`LoadedSlot`] from the bucket's [`RwLock`]. Slot
//! rotation (when build_slot finishes) replaces the inner `Arc` under
//! the write lock; in-flight queries that captured the prior `Arc`
//! complete against the old slot and drop their reference, freeing
//! the loaded state when the last query finishes. See
//! `docs/design_knowledge_db.md` § "Concurrency model" for the full
//! shape this scales toward.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use tokio_util::sync::CancellationToken;

use super::bucket::{BoxFuture, Bucket};
use super::chunker::Chunker;
use super::chunks::{ChunkStoreReader, ChunkStoreWriter};
use super::config::{BucketConfig, ChunkerConfig};
use super::dense::{DenseIndex, HnswParams};
use super::manifest::{
    EmbedderSnapshot, ServingSnapshot, SlotManifest, SlotState, SlotStats, SparseSnapshot,
};
use super::slot;
use super::source::SourceAdapter;
use super::sparse::{SparseIndex, SparseIndexBuilder};
use super::types::{
    BucketError, BucketId, BucketStatus, Candidate, Chunk, ChunkId, NewChunk, SearchPath, SlotId,
};
use super::vectors::{VectorStoreReader, VectorStoreWriter};
use crate::providers::embedding::{EmbedRequest, EmbeddingProvider};

/// Number of chunks per `EmbeddingProvider::embed` call. Bumped to 128
/// after the 5k simplewiki TEI smoke showed per-batch HTTP overhead
/// dominating end-to-end embed throughput at 32; with 128 (still under
/// TEI's `max_batch_tokens=65536` ÷ ~500-token chunks ≈ 130-chunk
/// per-batch budget) the round-trip count drops 4× without obviously
/// hurting per-batch latency on the test endpoint. Configurable per-call
/// eventually if measurement justifies finer control; for now it's a
/// const at module level.
const EMBED_BATCH_SIZE: usize = 128;

/// Coarse build-stage tag passed to [`BuildObserver::on_phase`]. Mirrors
/// the wire-side `BucketBuildPhase` so the scheduler can pass these
/// through unmodified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildPhase {
    /// Streaming source → chunks → embeds → writers. The long-tail of
    /// `Indexing` is dominated by embedder round-trips on real provider
    /// runs, by chunker work on the mock embedder.
    Indexing,
    /// HNSW graph construction over the just-written vectors. Single
    /// call, runs on a `spawn_blocking` thread so the runtime worker
    /// stays free; counters don't tick during this phase.
    BuildingDense,
    /// Open readers, write final manifest, atomic active-slot pointer
    /// flip. Sub-second on every build measured so far.
    Finalizing,
}

/// Optional progress hook for `DiskBucket::build_slot`. The build calls
/// `on_phase` exactly once per phase transition and `on_progress` after
/// every chunk-batch flush during the `Indexing` phase. Implementations
/// must be cheap and non-blocking — the observer runs on the same task
/// driving the build, so a slow callback directly slows the build.
///
/// `None` for the observer parameter means "no progress hook"; the
/// build runs identically to the pre-observer code path.
pub trait BuildObserver: Send + Sync {
    fn on_phase(&self, phase: BuildPhase);
    /// Cumulative counters since build start. Both values monotonically
    /// increase. Called at end of each chunk-batch flush.
    fn on_progress(&self, source_records: u64, chunks: u64);
}

#[derive(Debug)]
pub struct DiskBucket {
    id: BucketId,
    root: PathBuf,
    config: BucketConfig,
    /// Loaded state for the active slot, if any.
    ///
    /// Outer `RwLock` guards rotation (write) vs active-slot reads;
    /// inner `Arc` lets each query take a snapshot it can read from
    /// without holding the lock for the duration of the search.
    active: RwLock<Option<Arc<LoadedSlot>>>,
}

/// Loaded state for a single slot — everything a query needs to hit the
/// active slot without going back to disk for index metadata. Slice 6
/// rounds out the dense+sparse pair: chunks + vectors + dense (HNSW) +
/// sparse (tantivy). `sparse` is `None` when the bucket has the sparse
/// path disabled or the slot was built before the sparse path landed.
#[derive(Debug)]
struct LoadedSlot {
    manifest: SlotManifest,
    chunks: ChunkStoreReader,
    vectors: VectorStoreReader,
    dense: DenseIndex,
    sparse: Option<SparseIndex>,
}

impl DiskBucket {
    /// Open an existing bucket. The directory must contain a valid
    /// `bucket.toml`. If a `slots/active` pointer exists, the active
    /// slot is loaded; otherwise the bucket is opened with no active
    /// slot ([`BucketStatus::NoActiveSlot`]).
    pub fn open(root: impl Into<PathBuf>, id: BucketId) -> Result<Self, BucketError> {
        let root = root.into();
        let config = read_bucket_config(&root)?;
        let active = match slot::read_active_slot(&root)? {
            Some(slot_id) => Some(Arc::new(load_slot(&root, &slot_id)?)),
            None => None,
        };
        Ok(Self {
            id,
            root,
            config,
            active: RwLock::new(active),
        })
    }

    /// Build a new slot from a source adapter + chunker + embedder. The
    /// slot is created, populated with chunks and their embeddings, and
    /// promoted to active in one call. Returns the new slot id.
    ///
    /// The embedder snapshot recorded in the manifest is derived from
    /// `embedder.list_models()` at build start — its first model is the
    /// one used. For multi-model providers (future OpenAI hookup), the
    /// caller will eventually supply an explicit model id; for TEI's
    /// single-model deployments, the first-and-only model is the right
    /// answer.
    pub async fn build_slot(
        &self,
        adapter: &dyn SourceAdapter,
        chunker: &dyn Chunker,
        embedder: Arc<dyn EmbeddingProvider>,
        observer: Option<&dyn BuildObserver>,
        cancel: &CancellationToken,
    ) -> Result<SlotId, BucketError> {
        let embedder_snapshot =
            derive_embedder_snapshot(&self.config.defaults.embedder, embedder.as_ref(), cancel)
                .await?;

        let slot_id = slot::generate_slot_id();
        let slot_path = slot::slot_dir(&self.root, &slot_id);
        fs::create_dir_all(&slot_path).map_err(BucketError::Io)?;

        // Sparse path enabled? Recorded in the manifest so slot load
        // can skip opening the tantivy directory when it doesn't exist.
        let sparse_enabled = self.config.search_paths.sparse.enabled;
        let sparse_snapshot = if sparse_enabled {
            Some(SparseSnapshot {
                tokenizer: self.config.search_paths.sparse.tokenizer.clone(),
            })
        } else {
            None
        };

        // Initial manifest with state=Building. Persisted before any
        // heavy work so a crash mid-build leaves a slot directory we
        // can identify and clean up on next open.
        let now = chrono::Utc::now();
        let mut manifest = SlotManifest {
            slot_id: slot_id.clone(),
            state: SlotState::Building,
            created_at: now,
            build_started_at: Some(now),
            build_completed_at: None,
            chunker_snapshot: chunker_snapshot_from_config(&self.config.chunker),
            embedder: embedder_snapshot.clone(),
            sparse: sparse_snapshot,
            serving: ServingSnapshot {
                mode: self.config.defaults.serving_mode,
                quantization: self.config.defaults.quantization,
            },
            stats: SlotStats::default(),
            lineage: None,
        };
        write_manifest(&slot_path, &manifest)?;

        // Open writers.
        let chunks_bin = slot::chunks_bin_path(&slot_path);
        let chunks_idx = slot::chunks_idx_path(&slot_path);
        let vectors_bin = slot::vectors_bin_path(&slot_path);
        let vectors_idx = slot::vectors_idx_path(&slot_path);
        let tantivy_dir = slot::tantivy_dir(&slot_path);

        let mut chunk_writer =
            ChunkStoreWriter::create(&chunks_bin, &chunks_idx).map_err(BucketError::Io)?;
        let mut vector_writer =
            VectorStoreWriter::create(&vectors_bin, &vectors_idx, embedder_snapshot.dimension)
                .map_err(BucketError::Io)?;
        let mut sparse_builder = if sparse_enabled {
            Some(SparseIndexBuilder::create(&tantivy_dir)?)
        } else {
            None
        };

        // Streaming pipeline: source → chunker → batched embed → all
        // three writers (chunks, vectors, tantivy). We batch chunks (not
        // records) because record sizes vary; batching at the chunk
        // level keeps embed calls predictable.
        let mut pending: Vec<NewChunk> = Vec::with_capacity(EMBED_BATCH_SIZE);
        let mut source_records: u64 = 0;
        let mut chunks_emitted: u64 = 0;

        if let Some(obs) = observer {
            obs.on_phase(BuildPhase::Indexing);
        }

        for record in adapter.enumerate() {
            if cancel.is_cancelled() {
                return Err(BucketError::Cancelled);
            }
            let record = record.map_err(|e| BucketError::Other(format!("source error: {e}")))?;
            source_records += 1;

            for new_chunk in chunker.chunk(&record) {
                pending.push(new_chunk);
                if pending.len() >= EMBED_BATCH_SIZE {
                    let flushed = pending.len() as u64;
                    flush_batch(
                        &mut pending,
                        &mut chunk_writer,
                        &mut vector_writer,
                        sparse_builder.as_mut(),
                        embedder.as_ref(),
                        cancel,
                    )
                    .await?;
                    chunks_emitted += flushed;
                    if let Some(obs) = observer {
                        obs.on_progress(source_records, chunks_emitted);
                    }
                }
            }
        }

        // Final partial batch.
        if !pending.is_empty() {
            let flushed = pending.len() as u64;
            flush_batch(
                &mut pending,
                &mut chunk_writer,
                &mut vector_writer,
                sparse_builder.as_mut(),
                embedder.as_ref(),
                cancel,
            )
            .await?;
            chunks_emitted += flushed;
            if let Some(obs) = observer {
                obs.on_progress(source_records, chunks_emitted);
            }
        }

        let chunk_count = chunk_writer.finalize().map_err(BucketError::Io)? as u64;
        let vector_count = vector_writer.finalize().map_err(BucketError::Io)? as u64;
        if chunk_count != vector_count {
            // Should never happen — flush_batch writes one vector per
            // chunk in lockstep — but a structural divergence here
            // would be a real bug worth surfacing.
            return Err(BucketError::Other(format!(
                "chunk/vector count mismatch: {chunk_count} chunks, {vector_count} vectors",
            )));
        }
        if let Some(builder) = sparse_builder.take() {
            let sparse_count = builder.finalize()? as u64;
            if sparse_count != chunk_count {
                return Err(BucketError::Other(format!(
                    "chunk/sparse count mismatch: {chunk_count} chunks, {sparse_count} tantivy docs",
                )));
            }
        }

        // Open readers for the new slot before we promote — promotion
        // should never make a slot active that we can't immediately
        // serve queries from.
        let chunks = ChunkStoreReader::open(&chunks_bin, &chunks_idx).map_err(BucketError::Io)?;
        let vectors =
            VectorStoreReader::open(&vectors_bin, &vectors_idx).map_err(BucketError::Io)?;

        // Build the dense index from the just-written vectors and
        // dump it to disk so the next slot open can skip the rebuild.
        // The HNSW build is the dominant cost on real-scale runs (~7
        // min for 50k vectors, multi-hour for full wikipedia) and is
        // pure CPU, so push it onto a `spawn_blocking` thread to keep
        // the runtime worker free for other tasks. Failure to dump
        // is non-fatal — the slot is still usable (load_slot's
        // fallback rebuilds), so we log and continue rather than
        // tearing down the build.
        if let Some(obs) = observer {
            obs.on_phase(BuildPhase::BuildingDense);
        }
        let slot_path_for_dump = slot_path.clone();
        let slot_id_for_warn = slot_id.clone();
        let (dense, vectors) = tokio::task::spawn_blocking(move || {
            let dense = DenseIndex::build(&vectors, HnswParams::default())?;
            if let Err(e) = dense.dump_to(&slot_path_for_dump) {
                tracing::warn!(
                    slot = %slot_id_for_warn,
                    error = %e,
                    "hnsw dump failed; slot is still usable but next open will rebuild from vectors",
                );
            }
            Ok::<_, BucketError>((dense, vectors))
        })
        .await
        .map_err(|e| BucketError::Other(format!("spawn_blocking dense build: {e}")))??;

        if let Some(obs) = observer {
            obs.on_phase(BuildPhase::Finalizing);
        }

        // Open the sparse index reader if we built one.
        let sparse = if sparse_enabled {
            Some(SparseIndex::open(&tantivy_dir)?)
        } else {
            None
        };

        // Update manifest with final stats and promote to ready.
        manifest.state = SlotState::Ready;
        manifest.build_completed_at = Some(chrono::Utc::now());
        manifest.stats.source_records = source_records;
        manifest.stats.chunk_count = chunk_count;
        manifest.stats.vector_count = vector_count;
        manifest.stats.disk_size_bytes = total_dir_size(&slot_path).unwrap_or(0);
        write_manifest(&slot_path, &manifest)?;

        let loaded = Arc::new(LoadedSlot {
            manifest,
            chunks,
            vectors,
            dense,
            sparse,
        });

        // Atomic active-pointer flip; then swap the loaded state under
        // the write lock so subsequent queries see the new slot.
        slot::set_active_slot(&self.root, &slot_id)?;
        {
            let mut active = self.active.write().expect("active slot lock poisoned");
            *active = Some(loaded);
        }

        Ok(slot_id)
    }

    fn active_snapshot(&self) -> Option<Arc<LoadedSlot>> {
        self.active
            .read()
            .expect("active slot lock poisoned")
            .clone()
    }

    /// Test/inspection accessor. Returns the current active slot id, if any.
    pub fn active_slot_id(&self) -> Option<SlotId> {
        self.active_snapshot().map(|s| s.manifest.slot_id.clone())
    }

    /// Test/inspection accessor. Returns the dimension of the active
    /// slot's vectors, if any.
    pub fn active_dimension(&self) -> Option<u32> {
        self.active_snapshot().map(|s| s.vectors.dimension())
    }

    /// Test/inspection accessor. Fetches a vector by chunk id from the
    /// active slot, bypassing the (currently empty) search path. Useful
    /// for exercising the build pipeline before HNSW lands.
    pub fn fetch_vector(&self, chunk_id: ChunkId) -> Result<Option<Vec<f32>>, BucketError> {
        let Some(active) = self.active_snapshot() else {
            return Ok(None);
        };
        active.vectors.fetch(chunk_id).map_err(BucketError::Io)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &BucketConfig {
        &self.config
    }
}

impl Bucket for DiskBucket {
    fn id(&self) -> &BucketId {
        &self.id
    }

    fn status(&self) -> BucketStatus {
        match self.active_snapshot() {
            Some(_) => BucketStatus::Ready,
            None => BucketStatus::NoActiveSlot,
        }
    }

    fn dense_search<'a>(
        &'a self,
        query_vec: &'a [f32],
        top_k: usize,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<Candidate>, BucketError>> {
        Box::pin(async move {
            let Some(active) = self.active_snapshot() else {
                return Ok(Vec::new());
            };
            // Validate query dimension against the slot's recorded
            // embedder dimension. A mismatch means the caller is
            // querying with a vector from a different embedder — fail
            // explicitly rather than return junk.
            let slot_dim = active.vectors.dimension();
            if query_vec.len() as u32 != slot_dim {
                return Err(BucketError::DimensionMismatch {
                    query: query_vec.len() as u32,
                    slot: slot_dim,
                });
            }

            let hits = active.dense.search(query_vec, top_k);
            let mut out = Vec::with_capacity(hits.len());
            for (chunk_id, distance) in hits {
                // Look up chunk text for inclusion in the candidate.
                // A `None` here would mean vectors.idx and chunks.idx
                // disagree on which chunk ids exist — a real
                // structural failure worth surfacing.
                let chunk = active
                    .chunks
                    .fetch(chunk_id)
                    .map_err(BucketError::Io)?
                    .ok_or_else(|| {
                        BucketError::Other(format!(
                            "dense hit {chunk_id} missing from chunks store",
                        ))
                    })?;
                out.push(Candidate {
                    bucket_id: self.id.clone(),
                    chunk_id,
                    chunk_text: chunk.text,
                    source_ref: chunk.source_ref,
                    source_score: distance,
                    path: SearchPath::Dense,
                });
            }
            Ok(out)
        })
    }

    fn sparse_search<'a>(
        &'a self,
        query_text: &'a str,
        top_k: usize,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<Candidate>, BucketError>> {
        Box::pin(async move {
            let Some(active) = self.active_snapshot() else {
                return Ok(Vec::new());
            };
            let Some(sparse) = active.sparse.as_ref() else {
                // Sparse path disabled for this bucket — empty
                // candidate list rather than an error, per the
                // trait contract.
                return Ok(Vec::new());
            };

            let hits = sparse.search(query_text, top_k)?;
            let mut out = Vec::with_capacity(hits.len());
            for (chunk_id, score) in hits {
                let chunk = active
                    .chunks
                    .fetch(chunk_id)
                    .map_err(BucketError::Io)?
                    .ok_or_else(|| {
                        BucketError::Other(format!(
                            "sparse hit {chunk_id} missing from chunks store",
                        ))
                    })?;
                out.push(Candidate {
                    bucket_id: self.id.clone(),
                    chunk_id,
                    chunk_text: chunk.text,
                    source_ref: chunk.source_ref,
                    source_score: score,
                    path: SearchPath::Sparse,
                });
            }
            Ok(out)
        })
    }

    fn fetch_chunk<'a>(&'a self, chunk_id: ChunkId) -> BoxFuture<'a, Result<Chunk, BucketError>> {
        Box::pin(async move {
            let active = self
                .active_snapshot()
                .ok_or_else(|| BucketError::NoActiveSlot(self.id.clone()))?;
            match active.chunks.fetch(chunk_id).map_err(BucketError::Io)? {
                Some(chunk) => Ok(chunk),
                None => Err(BucketError::Other(format!(
                    "chunk {chunk_id} not in active slot",
                ))),
            }
        })
    }

    fn insert<'a>(
        &'a self,
        _new_chunks: Vec<NewChunk>,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<ChunkId>, BucketError>> {
        Box::pin(async move {
            Err(BucketError::Other(
                "insert: deferred until indexes land (slices 5–6)".to_string(),
            ))
        })
    }

    fn tombstone<'a>(&'a self, _chunk_ids: Vec<ChunkId>) -> BoxFuture<'a, Result<(), BucketError>> {
        Box::pin(async move {
            Err(BucketError::Other(
                "tombstone: deferred until indexes land (slices 5–6)".to_string(),
            ))
        })
    }

    fn compact<'a>(
        &'a self,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<(), BucketError>> {
        Box::pin(async move {
            Err(BucketError::Other(
                "compact: deferred until indexes land (slices 5–6)".to_string(),
            ))
        })
    }
}

// --- helpers ---

async fn flush_batch(
    pending: &mut Vec<NewChunk>,
    chunks: &mut ChunkStoreWriter,
    vectors: &mut VectorStoreWriter,
    sparse: Option<&mut SparseIndexBuilder>,
    embedder: &dyn EmbeddingProvider,
    cancel: &CancellationToken,
) -> Result<(), BucketError> {
    let texts: Vec<String> = pending.iter().map(|c| c.text.clone()).collect();
    let req = EmbedRequest {
        // TEI single-model deployments ignore the model field. Future
        // OpenAI-shaped providers route on it; we'll surface model id
        // selection through the bucket config when that lands.
        model: "",
        inputs: &texts,
    };
    let resp = embedder
        .embed(&req, cancel)
        .await
        .map_err(|e| BucketError::Provider(e.to_string()))?;

    if resp.embeddings.len() != pending.len() {
        return Err(BucketError::Provider(format!(
            "embedder returned {} vectors for {} inputs",
            resp.embeddings.len(),
            pending.len(),
        )));
    }

    let mut sparse = sparse;
    for (chunk, vector) in pending.iter().zip(resp.embeddings.iter()) {
        let chunk_id = ChunkId::from_source(&chunk.source_record_hash, chunk.chunk_offset);
        chunks
            .append(chunk_id, &chunk.source_ref, &chunk.text)
            .map_err(BucketError::Io)?;
        vectors.append(chunk_id, vector).map_err(BucketError::Io)?;
        if let Some(builder) = sparse.as_deref_mut() {
            builder.add(chunk_id, &chunk.text)?;
        }
    }
    pending.clear();
    Ok(())
}

async fn derive_embedder_snapshot(
    provider_name: &str,
    embedder: &dyn EmbeddingProvider,
    _cancel: &CancellationToken,
) -> Result<EmbedderSnapshot, BucketError> {
    let models = embedder
        .list_models()
        .await
        .map_err(|e| BucketError::Provider(e.to_string()))?;
    let model = models
        .into_iter()
        .next()
        .ok_or_else(|| BucketError::Provider("embedder reports no available models".to_string()))?;
    Ok(EmbedderSnapshot {
        provider: provider_name.to_string(),
        model_id: model.id,
        dimension: model.dimension,
    })
}

fn read_bucket_config(root: &Path) -> Result<BucketConfig, BucketError> {
    let path = slot::bucket_toml_path(root);
    let text = fs::read_to_string(&path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            BucketError::Config(format!("bucket.toml not found at {}", path.display()))
        }
        _ => BucketError::Io(e),
    })?;
    BucketConfig::from_toml_str(&text)
}

fn load_slot(bucket_root: &Path, slot_id: &str) -> Result<LoadedSlot, BucketError> {
    let slot_path = slot::slot_dir(bucket_root, slot_id);
    let manifest_path = slot::manifest_path(&slot_path);
    let manifest_text = fs::read_to_string(&manifest_path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => BucketError::Config(format!(
            "manifest.toml missing at {}",
            manifest_path.display()
        )),
        _ => BucketError::Io(e),
    })?;
    let manifest = SlotManifest::from_toml_str(&manifest_text)?;

    let chunks_bin = slot::chunks_bin_path(&slot_path);
    let chunks_idx = slot::chunks_idx_path(&slot_path);
    let chunks = ChunkStoreReader::open(&chunks_bin, &chunks_idx).map_err(BucketError::Io)?;

    let vectors_bin = slot::vectors_bin_path(&slot_path);
    let vectors_idx = slot::vectors_idx_path(&slot_path);
    let vectors = VectorStoreReader::open(&vectors_bin, &vectors_idx).map_err(BucketError::Io)?;

    // Sanity check: vector dimension must match the manifest's recorded
    // embedder dimension. A mismatch means the on-disk vectors weren't
    // produced by the manifest's embedder (the file got swapped or the
    // build lied).
    if vectors.dimension() != manifest.embedder.dimension {
        return Err(BucketError::Config(format!(
            "vectors.bin dimension {} does not match manifest embedder dimension {}",
            vectors.dimension(),
            manifest.embedder.dimension,
        )));
    }

    // Try the persisted HNSW dump first; rebuild from vectors as a
    // fallback for slots built before persistence existed (or
    // recovered from a partial write where the dump is missing /
    // count-mismatched).
    let dense = match DenseIndex::load_from(&slot_path, &vectors) {
        Ok(d) => d,
        Err(e) => {
            tracing::info!(
                slot = %slot_id,
                error = %e,
                "hnsw dump unavailable; rebuilding from vectors.bin",
            );
            let rebuilt = DenseIndex::build(&vectors, HnswParams::default())?;
            // Opportunistically write the dump for next time. Don't
            // fail the load if it doesn't stick.
            if let Err(dump_err) = rebuilt.dump_to(&slot_path) {
                tracing::warn!(
                    slot = %slot_id,
                    error = %dump_err,
                    "post-rebuild hnsw dump failed; next open will rebuild again",
                );
            }
            rebuilt
        }
    };

    // Sparse index — open if the slot manifest says it has one.
    // Tantivy persistence is segment-based, so this is a real on-disk
    // open (no rebuild equivalent needed; tantivy reload is fast).
    let sparse = if manifest.sparse.is_some() {
        Some(SparseIndex::open(&slot::tantivy_dir(&slot_path))?)
    } else {
        None
    };

    Ok(LoadedSlot {
        manifest,
        chunks,
        vectors,
        dense,
        sparse,
    })
}

fn write_manifest(slot_path: &Path, manifest: &SlotManifest) -> Result<(), BucketError> {
    let toml = manifest.to_toml_string()?;
    let path = slot::manifest_path(slot_path);
    fs::write(&path, toml).map_err(BucketError::Io)?;
    Ok(())
}

fn chunker_snapshot_from_config(config: &ChunkerConfig) -> ChunkerConfig {
    config.clone()
}

fn total_dir_size(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            total += meta.len();
        } else if meta.is_dir() {
            total += total_dir_size(&entry.path())?;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::knowledge::{MarkdownDir, TokenBasedChunker};
    use crate::providers::embedding::{
        BoxFuture as ProviderBoxFuture, EmbedRequest as ProviderEmbedRequest, EmbeddingError,
        EmbeddingModelInfo, EmbeddingResponse,
    };

    /// Deterministic mock embedder for tests. Produces a vector derived
    /// from `blake3(text)` so the same input always yields the same
    /// vector — useful for round-trip assertions and reproducible build
    /// outputs.
    struct MockEmbedder {
        dimension: u32,
        model_id: String,
        embed_calls: AtomicUsize,
    }

    impl MockEmbedder {
        fn new(dimension: u32) -> Self {
            Self {
                dimension,
                model_id: "mock-embedder".to_string(),
                embed_calls: AtomicUsize::new(0),
            }
        }

        fn fake_vector(text: &str, dim: u32) -> Vec<f32> {
            let hash = blake3::hash(text.as_bytes());
            let bytes = hash.as_bytes();
            (0..dim)
                .map(|i| {
                    let byte = bytes[(i as usize) % 32];
                    // Map [0..256) into [-1, 1) for mildly realistic spread.
                    (byte as f32) / 128.0 - 1.0
                })
                .collect()
        }
    }

    impl EmbeddingProvider for MockEmbedder {
        fn embed<'a>(
            &'a self,
            req: &'a ProviderEmbedRequest<'a>,
            _cancel: &'a CancellationToken,
        ) -> ProviderBoxFuture<'a, Result<EmbeddingResponse, EmbeddingError>> {
            self.embed_calls.fetch_add(1, Ordering::SeqCst);
            let dim = self.dimension;
            Box::pin(async move {
                let embeddings: Vec<Vec<f32>> = req
                    .inputs
                    .iter()
                    .map(|text| Self::fake_vector(text, dim))
                    .collect();
                Ok(EmbeddingResponse {
                    embeddings,
                    usage: None,
                })
            })
        }

        fn list_models<'a>(
            &'a self,
        ) -> ProviderBoxFuture<'a, Result<Vec<EmbeddingModelInfo>, EmbeddingError>> {
            let dim = self.dimension;
            let id = self.model_id.clone();
            Box::pin(async move {
                Ok(vec![EmbeddingModelInfo {
                    id,
                    dimension: dim,
                    max_input_tokens: Some(512),
                }])
            })
        }
    }

    fn write_md(dir: &Path, name: &str, body: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    fn sample_bucket_toml(source_path: &Path) -> String {
        format!(
            r#"
name = "Test Notes"
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
            source_path.display()
        )
    }

    fn open_test_bucket(bucket_root: &Path, source_path: &Path) -> DiskBucket {
        fs::create_dir_all(bucket_root).unwrap();
        fs::create_dir_all(slot::slots_dir(bucket_root)).unwrap();
        fs::write(
            slot::bucket_toml_path(bucket_root),
            sample_bucket_toml(source_path),
        )
        .unwrap();
        DiskBucket::open(bucket_root, BucketId::pod("notes")).unwrap()
    }

    #[test]
    fn open_loads_bucket_config_with_no_active_slot() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        let bucket_root = tmp.path().join("bucket");

        let bucket = open_test_bucket(&bucket_root, &source);
        assert_eq!(bucket.id().to_string(), "pod:notes");
        assert_eq!(bucket.config().name, "Test Notes");
        assert!(matches!(bucket.status(), BucketStatus::NoActiveSlot));
        assert!(bucket.active_slot_id().is_none());
        assert!(bucket.active_dimension().is_none());
    }

    #[test]
    fn open_missing_bucket_toml_errors() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path()).unwrap();
        let err = DiskBucket::open(tmp.path(), BucketId::pod("notes")).unwrap_err();
        assert!(matches!(err, BucketError::Config(_)), "{err:?}");
    }

    #[tokio::test]
    async fn build_slot_writes_chunks_and_vectors_and_promotes_active() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "alpha.md", "alpha content one");
        write_md(&source, "beta.md", "beta content two");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);

        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(16));
        let cancel = CancellationToken::new();

        let slot_id = bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        // Active pointer flipped + dimension reflects the embedder
        assert_eq!(bucket.active_slot_id().as_deref(), Some(slot_id.as_str()));
        assert_eq!(bucket.active_dimension(), Some(16));
        assert!(matches!(bucket.status(), BucketStatus::Ready));

        // Slot directory exists with chunks AND vectors files
        let slot_path = slot::slot_dir(&bucket_root, &slot_id);
        assert!(slot::chunks_bin_path(&slot_path).is_file());
        assert!(slot::chunks_idx_path(&slot_path).is_file());
        assert!(slot::vectors_bin_path(&slot_path).is_file());
        assert!(slot::vectors_idx_path(&slot_path).is_file());

        // Manifest reflects vector_count + embedder dimension
        let manifest_text = fs::read_to_string(slot::manifest_path(&slot_path)).unwrap();
        let manifest = SlotManifest::from_toml_str(&manifest_text).unwrap();
        assert_eq!(manifest.state, SlotState::Ready);
        assert_eq!(manifest.embedder.model_id, "mock-embedder");
        assert_eq!(manifest.embedder.dimension, 16);
        assert_eq!(manifest.stats.chunk_count, manifest.stats.vector_count);
        assert!(manifest.stats.chunk_count >= 2); // at least one chunk per record
    }

    #[tokio::test]
    async fn fetch_vector_returns_persisted_vector_for_chunk() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(
            &source,
            "doc.md",
            "The quick brown fox jumps over the lazy dog.",
        );

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(32));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        // Synthesize a record matching what the adapter produced and
        // chunk it the same way; the first chunk's id should be in the
        // bucket.
        let record = crate::knowledge::SourceRecord::new(
            "ignored",
            "The quick brown fox jumps over the lazy dog.",
        );
        let new_chunks = chunker.chunk(&record);
        assert!(!new_chunks.is_empty());
        let chunk_id = ChunkId::from_source(&new_chunks[0].source_record_hash, 0);

        let vec = bucket
            .fetch_vector(chunk_id)
            .unwrap()
            .expect("vector present");
        assert_eq!(vec.len(), 32);
        // Mock vectors are deterministic from text, so re-derive and compare.
        let expected = MockEmbedder::fake_vector(&new_chunks[0].text, 32);
        assert_eq!(vec, expected);
    }

    #[tokio::test]
    async fn fetch_chunk_works_alongside_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hi there");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        let record = crate::knowledge::SourceRecord::new("ignored", "hi there");
        let new_chunks = chunker.chunk(&record);
        let chunk_id = ChunkId::from_source(&new_chunks[0].source_record_hash, 0);

        let chunk = bucket.fetch_chunk(chunk_id).await.unwrap();
        assert_eq!(chunk.text, "hi there");

        let vec = bucket.fetch_vector(chunk_id).unwrap().unwrap();
        assert_eq!(vec.len(), 8);
    }

    #[tokio::test]
    async fn fetch_chunk_unknown_id_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hi");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        let bogus = ChunkId::from_source(&[42; 32], 999);
        let err = bucket.fetch_chunk(bogus).await.unwrap_err();
        assert!(matches!(err, BucketError::Other(_)));
    }

    #[tokio::test]
    async fn fetch_chunk_with_no_active_slot_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);

        let id = ChunkId::from_source(&[0; 32], 0);
        let err = bucket.fetch_chunk(id).await.unwrap_err();
        assert!(matches!(err, BucketError::NoActiveSlot(_)));
    }

    #[tokio::test]
    async fn rebuilding_open_finds_active_slot_with_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hello");

        let bucket_root = tmp.path().join("bucket");
        let slot_id = {
            let bucket = open_test_bucket(&bucket_root, &source);
            let adapter = MarkdownDir::new(&source);
            let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
            let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(12));
            let cancel = CancellationToken::new();
            bucket
                .build_slot(&adapter, &chunker, embedder, None, &cancel)
                .await
                .unwrap()
        };

        let reopened = DiskBucket::open(&bucket_root, BucketId::pod("notes")).unwrap();
        assert_eq!(reopened.active_slot_id().as_deref(), Some(slot_id.as_str()));
        assert_eq!(reopened.active_dimension(), Some(12));
        assert!(matches!(reopened.status(), BucketStatus::Ready));
    }

    #[tokio::test]
    async fn dense_search_returns_exact_match_for_exact_query() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "alpha.md", "alpha body");
        write_md(&source, "beta.md", "beta body");
        write_md(&source, "gamma.md", "gamma body");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(16));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        // Query with the same vector that "beta body" would have produced.
        // MockEmbedder is deterministic in its input → output mapping, so
        // we can re-derive the expected vector locally.
        let beta_record = crate::knowledge::SourceRecord::new("ignored", "beta body");
        let beta_chunks = chunker.chunk(&beta_record);
        let query = MockEmbedder::fake_vector(&beta_chunks[0].text, 16);

        let results = bucket.dense_search(&query, 3, &cancel).await.unwrap();
        assert_eq!(results.len(), 3);
        // Top result is the exact-match chunk
        assert!(results[0].chunk_text.contains("beta body"));
        assert!(results[0].source_score < 0.001);
        assert_eq!(results[0].path, SearchPath::Dense);
        assert_eq!(results[0].bucket_id, *bucket.id());

        // Results sorted ascending by distance
        for w in results.windows(2) {
            assert!(w[0].source_score <= w[1].source_score);
        }
    }

    #[tokio::test]
    async fn dense_search_with_no_active_slot_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let cancel = CancellationToken::new();
        let results = bucket.dense_search(&[0.0; 4], 5, &cancel).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn dense_search_rejects_dimension_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hello");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        // Query with the wrong dimension
        let err = bucket
            .dense_search(&[0.0; 16], 5, &cancel)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            BucketError::DimensionMismatch { query: 16, slot: 8 }
        ));
    }

    #[tokio::test]
    async fn sparse_search_returns_bm25_hits() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(
            &source,
            "alpha.md",
            "the quick brown fox jumps over the lazy dog",
        );
        write_md(&source, "beta.md", "lorem ipsum dolor sit amet consectetur");
        write_md(
            &source,
            "gamma.md",
            "the rain in spain falls mainly on the plain",
        );

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        let results = bucket.sparse_search("fox", 5, &cancel).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].chunk_text.contains("fox"));
        assert_eq!(results[0].path, SearchPath::Sparse);
        assert_eq!(results[0].bucket_id, *bucket.id());
        assert!(results[0].source_score > 0.0);
    }

    #[tokio::test]
    async fn sparse_search_with_no_active_slot_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let cancel = CancellationToken::new();
        let results = bucket.sparse_search("anything", 5, &cancel).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn sparse_search_returns_empty_when_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hello world");

        let bucket_root = tmp.path().join("bucket");
        // Bucket config with sparse disabled.
        fs::create_dir_all(&bucket_root).unwrap();
        fs::create_dir_all(slot::slots_dir(&bucket_root)).unwrap();
        let toml = format!(
            r#"
name = "Sparse Off"
created_at = "2026-04-25T10:23:11Z"

[source]
kind = "linked"
adapter = "markdown_dir"
path = "{}"

[chunker]
strategy = "token_based"
chunk_tokens = 50
overlap_tokens = 5

[search_paths.dense]
enabled = true

[search_paths.sparse]
enabled = false
tokenizer = "default"

[defaults]
embedder = "tei_test"
"#,
            source.display(),
        );
        fs::write(slot::bucket_toml_path(&bucket_root), toml).unwrap();
        let bucket = DiskBucket::open(&bucket_root, BucketId::pod("notes")).unwrap();

        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        let results = bucket.sparse_search("hello", 5, &cancel).await.unwrap();
        assert!(results.is_empty());

        // Tantivy directory shouldn't exist for a sparse-disabled slot.
        let slot_id = bucket.active_slot_id().unwrap();
        let tantivy = slot::tantivy_dir(&slot::slot_dir(&bucket_root, &slot_id));
        assert!(
            !tantivy.exists(),
            "sparse-disabled slot must not create tantivy dir"
        );
    }

    #[tokio::test]
    async fn rebuilding_open_finds_sparse_index_too() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "the quick brown fox");

        let bucket_root = tmp.path().join("bucket");
        {
            let bucket = open_test_bucket(&bucket_root, &source);
            let adapter = MarkdownDir::new(&source);
            let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
            let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
            let cancel = CancellationToken::new();
            bucket
                .build_slot(&adapter, &chunker, embedder, None, &cancel)
                .await
                .unwrap();
        }

        // Reopen and verify sparse search still works
        let reopened = DiskBucket::open(&bucket_root, BucketId::pod("notes")).unwrap();
        let cancel = CancellationToken::new();
        let results = reopened.sparse_search("fox", 5, &cancel).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].chunk_text.contains("fox"));
    }

    #[tokio::test]
    async fn mutation_methods_error_until_indexes_land() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let cancel = CancellationToken::new();

        assert!(bucket.insert(Vec::new(), &cancel).await.is_err());
        assert!(bucket.tombstone(Vec::new()).await.is_err());
        assert!(bucket.compact(&cancel).await.is_err());
    }

    #[tokio::test]
    async fn build_slot_cancellation_aborts_with_cancelled_error() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        for i in 0..20 {
            write_md(&source, &format!("doc-{i}.md"), "filler content");
        }
        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, BucketError::Cancelled), "{err:?}");
        assert!(bucket.active_slot_id().is_none());
    }

    #[tokio::test]
    async fn embedder_called_in_batches() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();

        // Create enough markdown files to force at least three embed
        // batches regardless of `EMBED_BATCH_SIZE` — each short file
        // produces exactly one chunk under the 50-token chunker, so
        // file count == chunk count.
        let file_count = super::EMBED_BATCH_SIZE * 2 + 5;
        for i in 0..file_count {
            write_md(
                &source,
                &format!("doc-{i:04}.md"),
                &format!("content number {i}"),
            );
        }
        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);

        let mock = Arc::new(MockEmbedder::new(8));
        let embedder: Arc<dyn EmbeddingProvider> = mock.clone();
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let cancel = CancellationToken::new();

        bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        let calls = mock.embed_calls.load(Ordering::SeqCst);
        // file_count chunks at EMBED_BATCH_SIZE → exactly 3 calls
        // (we wrote 2 full batches + a tail of 5).
        assert_eq!(calls, 3, "expected 3 batched embed calls, got {calls}");
    }

    #[tokio::test]
    async fn open_rejects_dimension_mismatch_between_manifest_and_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hello");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        let slot_id = bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();
        drop(bucket);

        // Hand-edit the manifest to claim a different dimension. Reopening
        // should refuse rather than silently misinterpret vectors.bin.
        let manifest_path = slot::manifest_path(&slot::slot_dir(&bucket_root, &slot_id));
        let text = fs::read_to_string(&manifest_path).unwrap();
        let edited = text.replace("dimension = 8", "dimension = 16");
        fs::write(&manifest_path, edited).unwrap();

        let err = DiskBucket::open(&bucket_root, BucketId::pod("notes")).unwrap_err();
        assert!(matches!(err, BucketError::Config(_)), "{err:?}");
    }
}
