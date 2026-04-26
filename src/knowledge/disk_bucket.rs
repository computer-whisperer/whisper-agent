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
use super::build_state::{BuildStateRecord, BuildStateWriter};
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
///
/// `serde` is derived so the same enum can be embedded in `build.state`
/// records without a parallel definition. Variant names are
/// snake-cased on the wire to match the wire enum's serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildPhase {
    /// Walking the source adapter, hashing each record, and writing
    /// `Planned` entries to `build.state`. Cheap relative to embedding
    /// (minutes for full wikipedia); produces the record-count
    /// denominator that lets the UI estimate completion during
    /// `Indexing`.
    Planning,
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

/// Resume cursor passed into [`DiskBucket::run_after_planning`]. The
/// fresh-build path uses [`ResumePoint::default`] (zeros across the
/// board); the resume path fills it from `build.state`'s last
/// `BatchEmbedded` checkpoint.
#[derive(Default)]
struct ResumePoint {
    /// Number of source records the resumed writers already account
    /// for. The indexing loop skips the first `records_completed`
    /// records from the source iterator before re-engaging the
    /// chunker.
    records_completed: u64,
    /// Cumulative chunk count already on disk in chunks.bin /
    /// vectors.bin / tantivy. Carried into the next BatchEmbedded
    /// record so its `chunks_completed` field stays monotonic.
    chunks_emitted: u64,
    /// Index for the next BatchEmbedded record. On resume, equals
    /// `last_batch_index + 1` so checkpoint indexes don't repeat.
    next_batch_index: u64,
    /// Pre-populated dense index for the resume path — built from
    /// vectors.bin's chunks 0..chunks_emitted before run_after_planning
    /// starts so subsequent inserts continue at the right position.
    /// `None` for fresh builds; the indexing loop creates an empty
    /// one in that case.
    dense: Option<Arc<DenseIndex>>,
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
/// active slot without going back to disk for index metadata.
///
/// All four index components are individually mutable:
/// - `chunks` is replaced per build-batch via [`RwLock`] swap, so a
///   query during an in-progress build sees as-of-last-batch chunk
///   text. Post-finalize, the reader is replaced with a real
///   `chunks.idx`-backed view.
/// - `vectors` is `None` during a build (no `vectors.idx` exists
///   yet) and `Some` once `finalize()` writes the idx. Queries don't
///   read vectors directly; only the test accessor `fetch_vector`
///   does.
/// - `dense` ([`DenseIndex`]) has interior `RwLock`-based mutability
///   on its HNSW graph; build inserts grow the graph in place while
///   queries take read locks.
/// - `sparse` is replaced per build-batch (after each tantivy
///   commit) so the reader sees the latest committed segments.
struct LoadedSlot {
    manifest: SlotManifest,
    chunks: RwLock<Arc<ChunkStoreReader>>,
    vectors: Option<VectorStoreReader>,
    dense: Arc<DenseIndex>,
    sparse: Option<RwLock<Arc<SparseIndex>>>,
}

impl std::fmt::Debug for LoadedSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedSlot")
            .field("slot_id", &self.manifest.slot_id)
            .field("state", &self.manifest.state)
            .field("dense_len", &self.dense.len())
            .finish_non_exhaustive()
    }
}

impl LoadedSlot {
    fn chunks(&self) -> Arc<ChunkStoreReader> {
        self.chunks
            .read()
            .expect("LoadedSlot.chunks lock poisoned")
            .clone()
    }

    fn sparse(&self) -> Option<Arc<SparseIndex>> {
        self.sparse
            .as_ref()
            .map(|lock| lock.read().expect("LoadedSlot.sparse lock poisoned").clone())
    }
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

        // Initial manifest with state=Planning. Persisted before any
        // heavy work so a crash mid-build leaves a slot directory we
        // can identify (and resume from, in Commit B) on next open.
        let now = chrono::Utc::now();
        let mut manifest = SlotManifest {
            slot_id: slot_id.clone(),
            state: SlotState::Planning,
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

        // Append-only resume log. Created here even on first build so
        // the file is in place from the start; resume code reads it
        // back to find the last durable checkpoint.
        let build_state_path = slot_path.join(slot::BUILD_STATE);
        let mut build_state =
            BuildStateWriter::create_or_open(&build_state_path).map_err(BucketError::Io)?;

        // --- Phase: Planning ---
        // Walk the source once to enumerate records and write Planned
        // entries with content_hash. Chunking happens in the next phase.
        // The two-pass design gives us a record-count denominator before
        // any embedding starts, and it gives the resume path a stable
        // record list to drift-check against.
        publish_phase(BuildPhase::Planning, observer, &mut build_state)?;
        let mut source_records: u64 = 0;
        for record in adapter.enumerate() {
            if cancel.is_cancelled() {
                return Err(BucketError::Cancelled);
            }
            let record = record.map_err(|e| BucketError::Other(format!("source error: {e}")))?;
            build_state
                .append(&BuildStateRecord::Planned {
                    source_id: record.source_id.clone(),
                    content_hash: record.content_hash,
                })
                .map_err(BucketError::Io)?;
            source_records += 1;
            if let Some(obs) = observer {
                obs.on_progress(source_records, 0);
            }
        }

        // Promote manifest to Building once planning is done. Writing
        // here (rather than at the very start of build_slot) lets the
        // resume code distinguish "planning was interrupted, redo it"
        // (state=Planning) from "planning finished, embedding was
        // interrupted" (state=Building).
        manifest.state = SlotState::Building;
        write_manifest(&slot_path, &manifest)?;

        // Fresh writers — the slot dir is brand new.
        let chunks_bin = slot::chunks_bin_path(&slot_path);
        let chunks_idx = slot::chunks_idx_path(&slot_path);
        let vectors_bin = slot::vectors_bin_path(&slot_path);
        let vectors_idx = slot::vectors_idx_path(&slot_path);
        let tantivy_dir = slot::tantivy_dir(&slot_path);

        let chunk_writer =
            ChunkStoreWriter::create(&chunks_bin, &chunks_idx).map_err(BucketError::Io)?;
        let vector_writer =
            VectorStoreWriter::create(&vectors_bin, &vectors_idx, embedder_snapshot.dimension)
                .map_err(BucketError::Io)?;
        let sparse_builder = if sparse_enabled {
            Some(SparseIndexBuilder::create(&tantivy_dir)?)
        } else {
            None
        };

        self.run_after_planning(
            &slot_id,
            &slot_path,
            manifest,
            sparse_enabled,
            chunk_writer,
            vector_writer,
            sparse_builder,
            build_state,
            adapter,
            chunker,
            embedder,
            observer,
            cancel,
            ResumePoint::default(),
            source_records,
        )
        .await
    }

    /// Resume an in-progress slot build, starting from the last
    /// durable checkpoint in `build.state`.
    ///
    /// `slot_id` must name an existing slot directory under this
    /// bucket whose manifest is in [`SlotState::Planning`] or
    /// [`SlotState::Building`]. The slot's frozen chunker / embedder
    /// snapshot must match the bucket's current config — a mismatch
    /// means the user changed `bucket.toml` after the first attempt
    /// and the existing slot can no longer be extended; that's a
    /// loud error rather than a silent rebuild from scratch.
    ///
    /// Resume modes:
    /// - `Planning` state: any prior `build.state` is discarded and
    ///   the build restarts from Planning. (Planning is cheap; this
    ///   keeps the resume code simple.)
    /// - `Building` state: replay `build.state` to the last
    ///   `BatchEmbedded` checkpoint, drift-check `Planned` records
    ///   against fresh source content, truncate the chunks/vectors
    ///   files to the checkpoint boundary, and continue from the
    ///   next un-embedded record.
    pub async fn resume_slot(
        &self,
        slot_id: &str,
        adapter: &dyn SourceAdapter,
        chunker: &dyn Chunker,
        embedder: Arc<dyn EmbeddingProvider>,
        observer: Option<&dyn BuildObserver>,
        cancel: &CancellationToken,
    ) -> Result<SlotId, BucketError> {
        let slot_path = slot::slot_dir(&self.root, slot_id);
        let manifest_text = fs::read_to_string(slot::manifest_path(&slot_path))
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => BucketError::Other(format!(
                    "resume_slot: slot {slot_id} has no manifest.toml — slot does not exist?"
                )),
                _ => BucketError::Io(e),
            })?;
        let mut manifest = SlotManifest::from_toml_str(&manifest_text)?;

        match manifest.state {
            SlotState::Planning | SlotState::Building => {}
            other => {
                return Err(BucketError::Other(format!(
                    "resume_slot: slot {slot_id} is in state {other:?}, not resumable"
                )));
            }
        }

        // Validate snapshots against current bucket config. The
        // snapshot is *frozen at slot creation*; if the bucket's
        // config has changed since, this slot's chunk/vector layout
        // is no longer valid for continued building. A loud error
        // beats silently producing a corrupt slot.
        let current_chunker_snapshot = chunker_snapshot_from_config(&self.config.chunker);
        if manifest.chunker_snapshot != current_chunker_snapshot {
            return Err(BucketError::Other(format!(
                "resume_slot: chunker config changed since slot {slot_id} was started; \
                 delete the bucket and rebuild to use the new chunker"
            )));
        }
        let embedder_snapshot =
            derive_embedder_snapshot(&self.config.defaults.embedder, embedder.as_ref(), cancel)
                .await?;
        if manifest.embedder.dimension != embedder_snapshot.dimension {
            return Err(BucketError::Other(format!(
                "resume_slot: embedder dimension changed ({} → {}) since slot {slot_id} was started; \
                 delete the bucket and rebuild",
                manifest.embedder.dimension, embedder_snapshot.dimension,
            )));
        }

        let sparse_enabled = self.config.search_paths.sparse.enabled;
        let chunks_bin = slot::chunks_bin_path(&slot_path);
        let chunks_idx = slot::chunks_idx_path(&slot_path);
        let vectors_bin = slot::vectors_bin_path(&slot_path);
        let vectors_idx = slot::vectors_idx_path(&slot_path);
        let tantivy_dir = slot::tantivy_dir(&slot_path);
        let build_state_path = slot_path.join(slot::BUILD_STATE);

        // Planning state ⇒ discard any partial progress and restart
        // from a clean slate inside the existing slot dir.
        if manifest.state == SlotState::Planning {
            for f in [&chunks_bin, &chunks_idx, &vectors_bin, &vectors_idx] {
                if f.exists() {
                    fs::remove_file(f).map_err(BucketError::Io)?;
                }
            }
            if tantivy_dir.exists() {
                fs::remove_dir_all(&tantivy_dir).map_err(BucketError::Io)?;
            }
            if build_state_path.exists() {
                fs::remove_file(&build_state_path).map_err(BucketError::Io)?;
            }
            // Re-run the same setup the fresh-build path does.
            let mut build_state = BuildStateWriter::create_or_open(&build_state_path)
                .map_err(BucketError::Io)?;

            publish_phase(BuildPhase::Planning, observer, &mut build_state)?;
            let mut source_records: u64 = 0;
            for record in adapter.enumerate() {
                if cancel.is_cancelled() {
                    return Err(BucketError::Cancelled);
                }
                let record =
                    record.map_err(|e| BucketError::Other(format!("source error: {e}")))?;
                build_state
                    .append(&BuildStateRecord::Planned {
                        source_id: record.source_id.clone(),
                        content_hash: record.content_hash,
                    })
                    .map_err(BucketError::Io)?;
                source_records += 1;
                if let Some(obs) = observer {
                    obs.on_progress(source_records, 0);
                }
            }
            manifest.state = SlotState::Building;
            write_manifest(&slot_path, &manifest)?;

            let chunk_writer =
                ChunkStoreWriter::create(&chunks_bin, &chunks_idx).map_err(BucketError::Io)?;
            let vector_writer = VectorStoreWriter::create(
                &vectors_bin,
                &vectors_idx,
                embedder_snapshot.dimension,
            )
            .map_err(BucketError::Io)?;
            let sparse_builder = if sparse_enabled {
                Some(SparseIndexBuilder::create(&tantivy_dir)?)
            } else {
                None
            };
            return self
                .run_after_planning(
                    slot_id,
                    &slot_path,
                    manifest,
                    sparse_enabled,
                    chunk_writer,
                    vector_writer,
                    sparse_builder,
                    build_state,
                    adapter,
                    chunker,
                    embedder,
                    observer,
                    cancel,
                    ResumePoint::default(),
                    source_records,
                )
                .await;
        }

        // Building state ⇒ replay build.state to find the resume point.
        let prior_records = super::build_state::read_all(&build_state_path)
            .map_err(BucketError::Io)?;
        let planned: Vec<(String, [u8; 32])> = prior_records
            .iter()
            .filter_map(|r| match r {
                BuildStateRecord::Planned {
                    source_id,
                    content_hash,
                } => Some((source_id.clone(), *content_hash)),
                _ => None,
            })
            .collect();
        let last_batch = prior_records.iter().rev().find_map(|r| match r {
            BuildStateRecord::BatchEmbedded {
                batch_index,
                source_records_completed,
                chunks_completed,
            } => Some((*batch_index, *source_records_completed, *chunks_completed)),
            _ => None,
        });
        let (last_batch_index, records_done, chunks_done) =
            last_batch.unwrap_or((u64::MAX, 0, 0)); // u64::MAX → 0 below

        // Reopen build.state in append mode (existing entries stay).
        let build_state =
            BuildStateWriter::create_or_open(&build_state_path).map_err(BucketError::Io)?;

        // Re-run chunker on the first `records_done` source records
        // to derive their chunk_ids in insertion order. We feed those
        // into the resumed writers' `entries` vec; the actual chunks
        // are already on disk and we don't re-embed them.
        let mut resumed_chunk_ids: Vec<ChunkId> = Vec::with_capacity(chunks_done as usize);
        let records_to_skip = records_done.min(planned.len() as u64);
        let mut iter = adapter.enumerate();
        for i in 0..records_to_skip as usize {
            if cancel.is_cancelled() {
                return Err(BucketError::Cancelled);
            }
            let record = match iter.next() {
                Some(Ok(r)) => r,
                Some(Err(e)) => return Err(BucketError::Other(format!("source error: {e}"))),
                None => {
                    return Err(BucketError::Other(format!(
                        "resume_slot: source ended after {i} records but build.state expected at least {records_to_skip}"
                    )));
                }
            };
            let (expected_source_id, expected_hash) = &planned[i];
            if record.content_hash != *expected_hash {
                return Err(BucketError::Other(format!(
                    "resume_slot: source drift detected at record `{}` (was `{}` when planned); \
                     delete the bucket and rebuild to ingest the changed source",
                    record.source_id, expected_source_id,
                )));
            }
            if record.source_id != *expected_source_id {
                // Hash matched but source_id differs — a file was
                // renamed without content changes. chunk_ids derive
                // from content_hash alone so this doesn't break
                // resume; just log it.
                tracing::info!(
                    bucket = %self.id,
                    slot = %slot_id,
                    expected = %expected_source_id,
                    got = %record.source_id,
                    "resume: source_id changed but content_hash matches; continuing"
                );
            }
            for new_chunk in chunker.chunk(&record) {
                resumed_chunk_ids.push(ChunkId::from_source(
                    &new_chunk.source_record_hash,
                    new_chunk.chunk_offset,
                ));
            }
        }
        // We don't need `iter` past this point — run_after_planning
        // re-creates a fresh iterator and skips the same prefix.
        drop(iter);
        if (resumed_chunk_ids.len() as u64) != chunks_done {
            return Err(BucketError::Other(format!(
                "resume_slot: re-derived chunk count {} doesn't match build.state checkpoint {}",
                resumed_chunk_ids.len(),
                chunks_done,
            )));
        }

        // Scan chunks.bin to compute byte offsets for the resumed
        // entries; truncate the file at offsets[chunks_done].
        let scan_offsets = if chunks_bin.exists() {
            super::chunks::scan_record_offsets(&chunks_bin).map_err(BucketError::Io)?
        } else {
            vec![0]
        };
        if scan_offsets.len() < (chunks_done + 1) as usize {
            return Err(BucketError::Other(format!(
                "resume_slot: chunks.bin only has {} records, build.state checkpoint expects {}",
                scan_offsets.len().saturating_sub(1),
                chunks_done,
            )));
        }
        let resume_offsets = &scan_offsets[..=(chunks_done as usize)];

        let chunk_writer = if chunks_done == 0 {
            // No checkpoint yet → fresh writers (file may exist but
            // is empty/garbage; truncate it).
            ChunkStoreWriter::create(&chunks_bin, &chunks_idx).map_err(BucketError::Io)?
        } else {
            ChunkStoreWriter::open_resume(
                &chunks_bin,
                &chunks_idx,
                &resumed_chunk_ids,
                resume_offsets,
            )
            .map_err(BucketError::Io)?
        };
        let vector_writer = if chunks_done == 0 {
            VectorStoreWriter::create(&vectors_bin, &vectors_idx, embedder_snapshot.dimension)
                .map_err(BucketError::Io)?
        } else {
            VectorStoreWriter::open_resume(
                &vectors_bin,
                &vectors_idx,
                embedder_snapshot.dimension,
                &resumed_chunk_ids,
            )
            .map_err(BucketError::Io)?
        };
        let sparse_builder = if sparse_enabled {
            if tantivy_dir.exists() {
                Some(SparseIndexBuilder::open_resume(&tantivy_dir)?)
            } else {
                Some(SparseIndexBuilder::create(&tantivy_dir)?)
            }
        } else {
            None
        };

        // Rebuild the in-memory HNSW for the chunks already on disk
        // so subsequent inserts during the resumed Indexing phase
        // continue at position `chunks_done`. We read each vector
        // sequentially from vectors.bin via a position-only reader
        // (no vectors.idx yet — that's written at finalize). This
        // is the same M log N work the original build did up to
        // this point; it's the cost of resume that's roughly
        // proportional to how far the prior attempt got.
        let dense_for_resume: Option<Arc<DenseIndex>> = if chunks_done == 0 {
            None
        } else {
            let dense = DenseIndex::empty(
                HnswParams::default(),
                (planned.len().saturating_mul(8)).max(1024),
            );
            let bin_path_for_thread = vectors_bin.clone();
            let dim = embedder_snapshot.dimension;
            let chunk_ids = resumed_chunk_ids.clone();
            let dense = tokio::task::spawn_blocking(move || -> Result<DenseIndex, BucketError> {
                let mut bin = std::fs::File::open(&bin_path_for_thread).map_err(BucketError::Io)?;
                use std::io::{Read, Seek, SeekFrom};
                for (position, chunk_id) in chunk_ids.iter().enumerate() {
                    let byte_offset = 16u64 + position as u64 * dim as u64 * 4;
                    bin.seek(SeekFrom::Start(byte_offset)).map_err(BucketError::Io)?;
                    let mut bytes = vec![0u8; dim as usize * 4];
                    bin.read_exact(&mut bytes).map_err(BucketError::Io)?;
                    let mut v = Vec::with_capacity(dim as usize);
                    for chunk in bytes.chunks_exact(4) {
                        v.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    dense.insert(*chunk_id, position as u64, &v);
                }
                Ok(dense)
            })
            .await
            .map_err(|e| BucketError::Other(format!("spawn_blocking dense rebuild: {e}")))??;
            Some(Arc::new(dense))
        };

        let resume = ResumePoint {
            records_completed: records_to_skip,
            chunks_emitted: chunks_done,
            next_batch_index: if last_batch_index == u64::MAX {
                0
            } else {
                last_batch_index + 1
            },
            dense: dense_for_resume,
        };

        // The Planning phase already happened; we don't re-publish
        // it. Republish Indexing so the observer sees we're back in
        // the embedding loop on the resume path.
        let source_records_total = planned.len() as u64;
        self.run_after_planning(
            slot_id,
            &slot_path,
            manifest,
            sparse_enabled,
            chunk_writer,
            vector_writer,
            sparse_builder,
            build_state,
            adapter,
            chunker,
            embedder,
            observer,
            cancel,
            resume,
            source_records_total,
        )
        .await
    }

    /// Walk the source from `resume.records_completed` onward,
    /// chunk + embed each record, checkpoint at record-aligned batch
    /// boundaries, then run BuildingDense and Finalizing. Shared
    /// between the fresh-build and resume paths — both call this
    /// with their respective writers and starting positions.
    #[allow(clippy::too_many_arguments)]
    async fn run_after_planning(
        &self,
        slot_id: &str,
        slot_path: &Path,
        mut manifest: SlotManifest,
        sparse_enabled: bool,
        mut chunk_writer: ChunkStoreWriter,
        mut vector_writer: VectorStoreWriter,
        mut sparse_builder: Option<SparseIndexBuilder>,
        mut build_state: BuildStateWriter,
        adapter: &dyn SourceAdapter,
        chunker: &dyn Chunker,
        embedder: Arc<dyn EmbeddingProvider>,
        observer: Option<&dyn BuildObserver>,
        cancel: &CancellationToken,
        resume: ResumePoint,
        source_records: u64,
    ) -> Result<SlotId, BucketError> {
        let chunks_bin = slot::chunks_bin_path(slot_path);
        let chunks_idx = slot::chunks_idx_path(slot_path);
        let vectors_bin = slot::vectors_bin_path(slot_path);
        let vectors_idx = slot::vectors_idx_path(slot_path);
        let tantivy_dir = slot::tantivy_dir(slot_path);

        // --- Phase: Indexing ---
        // Buffer all chunks for one record, then flush at the record
        // boundary if the buffer has reached batch size. flush_batch
        // writes to chunks/vectors/tantivy AND inserts into the
        // dense (HNSW) graph in lockstep. Following the flush we
        // sync chunks/vectors, commit tantivy, optionally update the
        // active LoadedSlot's per-batch-replaceable views, and log
        // BatchEmbedded — that whole sequence is the durability
        // barrier resume relies on.
        publish_phase(BuildPhase::Indexing, observer, &mut build_state)?;
        let mut pending: Vec<NewChunk> = Vec::with_capacity(EMBED_BATCH_SIZE);
        let mut records_completed: u64 = resume.records_completed;
        let mut chunks_emitted: u64 = resume.chunks_emitted;
        let mut batch_index: u64 = resume.next_batch_index;

        // Empty HNSW + per-batch dense inserts. Sized hint is
        // generous — actual chunk count is unknown until the
        // chunker runs (one record can produce many chunks),
        // so we estimate from `source_records` with headroom.
        // Hnsw::new treats this as an allocation hint, not a cap.
        let dense_size_hint = source_records.saturating_mul(8).max(1024) as usize;
        let dense = match resume.dense {
            Some(d) => d,
            None => Arc::new(DenseIndex::empty(HnswParams::default(), dense_size_hint)),
        };

        // Mid-build active install: if the bucket has no active slot
        // (first build, not a rebuild), publish the in-progress slot
        // as the active one after the first batch. Subsequent
        // batches mutate this same Arc<LoadedSlot> in-place via the
        // chunks/sparse RwLocks and the dense interior lock.
        // For rebuilds, the prior active slot keeps serving until
        // we swap at the very end — same as today.
        let install_during_build = !self.has_active_slot();
        let mut active_during_build: Option<Arc<LoadedSlot>> = None;

        let chunks_bin_for_install = chunks_bin.clone();
        let mut iter = adapter.enumerate();
        // Skip records the resume code already accounted for.
        for _ in 0..resume.records_completed {
            match iter.next() {
                Some(Ok(_)) => {}
                Some(Err(e)) => {
                    return Err(BucketError::Other(format!("source error: {e}")))
                }
                None => break,
            }
        }

        for record in iter {
            if cancel.is_cancelled() {
                return Err(BucketError::Cancelled);
            }
            let record = record.map_err(|e| BucketError::Other(format!("source error: {e}")))?;

            for new_chunk in chunker.chunk(&record) {
                pending.push(new_chunk);
            }
            records_completed += 1;

            if pending.len() >= EMBED_BATCH_SIZE {
                let flushed = pending.len() as u64;
                let base_position = chunks_emitted;
                flush_batch(
                    &mut pending,
                    &mut chunk_writer,
                    &mut vector_writer,
                    sparse_builder.as_mut(),
                    dense.as_ref(),
                    base_position,
                    embedder.as_ref(),
                    cancel,
                )
                .await?;
                chunk_writer.flush().map_err(BucketError::Io)?;
                vector_writer.flush().map_err(BucketError::Io)?;
                if let Some(builder) = sparse_builder.as_mut() {
                    builder.commit()?;
                }
                chunks_emitted += flushed;
                self.update_active_during_build(
                    &chunks_bin_for_install,
                    &tantivy_dir,
                    sparse_enabled,
                    &chunk_writer,
                    &dense,
                    &manifest,
                    install_during_build,
                    &mut active_during_build,
                )?;
                build_state
                    .append(&BuildStateRecord::BatchEmbedded {
                        batch_index,
                        source_records_completed: records_completed,
                        chunks_completed: chunks_emitted,
                    })
                    .map_err(BucketError::Io)?;
                batch_index += 1;
                if let Some(obs) = observer {
                    obs.on_progress(records_completed, chunks_emitted);
                }
            }
        }

        // Final partial batch.
        if !pending.is_empty() {
            let flushed = pending.len() as u64;
            let base_position = chunks_emitted;
            flush_batch(
                &mut pending,
                &mut chunk_writer,
                &mut vector_writer,
                sparse_builder.as_mut(),
                dense.as_ref(),
                base_position,
                embedder.as_ref(),
                cancel,
            )
            .await?;
            chunk_writer.flush().map_err(BucketError::Io)?;
            vector_writer.flush().map_err(BucketError::Io)?;
            if let Some(builder) = sparse_builder.as_mut() {
                builder.commit()?;
            }
            chunks_emitted += flushed;
            self.update_active_during_build(
                &chunks_bin_for_install,
                &tantivy_dir,
                sparse_enabled,
                &chunk_writer,
                &dense,
                &manifest,
                install_during_build,
                &mut active_during_build,
            )?;
            build_state
                .append(&BuildStateRecord::BatchEmbedded {
                    batch_index,
                    source_records_completed: records_completed,
                    chunks_completed: chunks_emitted,
                })
                .map_err(BucketError::Io)?;
            if let Some(obs) = observer {
                obs.on_progress(records_completed, chunks_emitted);
            }
        }

        let chunk_count = chunk_writer.finalize().map_err(BucketError::Io)? as u64;
        let vector_count = vector_writer.finalize().map_err(BucketError::Io)? as u64;
        if chunk_count != vector_count {
            return Err(BucketError::Other(format!(
                "chunk/vector count mismatch: {chunk_count} chunks, {vector_count} vectors",
            )));
        }
        if let Some(builder) = sparse_builder.take() {
            let _ = builder.finalize()?;
        }

        let chunks = ChunkStoreReader::open(&chunks_bin, &chunks_idx).map_err(BucketError::Io)?;
        let vectors =
            VectorStoreReader::open(&vectors_bin, &vectors_idx).map_err(BucketError::Io)?;

        // --- Phase: BuildingDense ---
        // HNSW is already built incrementally — this phase is now
        // just the HNSW dump for next-open's benefit. We still emit
        // the phase tag so existing wire consumers / tests don't
        // break, even though counters don't tick.
        publish_phase(BuildPhase::BuildingDense, observer, &mut build_state)?;
        let dense_for_dump = dense.clone();
        let slot_path_for_dump = slot_path.to_path_buf();
        let slot_id_for_warn = slot_id.to_string();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = dense_for_dump.dump_to(&slot_path_for_dump) {
                tracing::warn!(
                    slot = %slot_id_for_warn,
                    error = %e,
                    "hnsw dump failed; slot is still usable but next open will rebuild from vectors",
                );
            }
        })
        .await
        .map_err(|e| BucketError::Other(format!("spawn_blocking dense dump: {e}")))?;

        // --- Phase: Finalizing ---
        publish_phase(BuildPhase::Finalizing, observer, &mut build_state)?;

        let sparse = if sparse_enabled {
            let idx = SparseIndex::open(&tantivy_dir)?;
            if (idx.len() as u64) != chunk_count {
                return Err(BucketError::Other(format!(
                    "tantivy num_docs ({}) does not match chunk_count ({chunk_count})",
                    idx.len(),
                )));
            }
            Some(idx)
        } else {
            None
        };

        manifest.state = SlotState::Ready;
        manifest.build_completed_at = Some(chrono::Utc::now());
        manifest.stats.source_records = source_records;
        manifest.stats.chunk_count = chunk_count;
        manifest.stats.vector_count = vector_count;
        manifest.stats.disk_size_bytes = total_dir_size(slot_path).unwrap_or(0);
        write_manifest(slot_path, &manifest)?;

        let loaded = Arc::new(LoadedSlot {
            manifest,
            chunks: RwLock::new(Arc::new(chunks)),
            vectors: Some(vectors),
            dense,
            sparse: sparse.map(|s| RwLock::new(Arc::new(s))),
        });

        slot::set_active_slot(&self.root, slot_id)?;
        {
            let mut active = self.active.write().expect("active slot lock poisoned");
            *active = Some(loaded);
        }

        build_state
            .append(&BuildStateRecord::BuildCompleted)
            .map_err(BucketError::Io)?;

        Ok(slot_id.to_string())
    }

    /// True iff a slot is currently published as active. Used to
    /// decide whether the in-progress build should publish itself
    /// mid-build (first builds: yes — partial slot is queryable) or
    /// hold off until the new slot is fully built (rebuilds: yes,
    /// keep serving the prior active slot).
    fn has_active_slot(&self) -> bool {
        self.active
            .read()
            .expect("active slot lock poisoned")
            .is_some()
    }

    /// Publish or update the in-progress slot as the bucket's
    /// active slot. Called at every batch boundary; on the first
    /// invocation it constructs and installs an `Arc<LoadedSlot>`
    /// over the in-flight indexes, on subsequent invocations it
    /// swaps the chunks/sparse readers under the LoadedSlot's
    /// per-field locks.
    ///
    /// Skipped entirely (`active_during_build` stays `None`) when
    /// `install_during_build` is false — that's the rebuild-vs-
    /// existing-slot case where the prior active slot must keep
    /// serving until the new slot promotes.
    #[allow(clippy::too_many_arguments)]
    fn update_active_during_build(
        &self,
        chunks_bin: &Path,
        tantivy_dir: &Path,
        sparse_enabled: bool,
        chunk_writer: &ChunkStoreWriter,
        dense: &Arc<DenseIndex>,
        manifest: &SlotManifest,
        install_during_build: bool,
        active_during_build: &mut Option<Arc<LoadedSlot>>,
    ) -> Result<(), BucketError> {
        if !install_during_build {
            return Ok(());
        }
        // Build the live chunks reader from the writer's snapshot.
        let snapshot = chunk_writer.snapshot_index();
        let chunks_reader = Arc::new(
            ChunkStoreReader::open_with_map(chunks_bin, snapshot).map_err(BucketError::Io)?,
        );
        // Reload sparse if the bucket has it — tantivy commits per
        // batch already (Commit B), so re-opening picks up the
        // freshest committed segments.
        let sparse_reader = if sparse_enabled {
            Some(Arc::new(SparseIndex::open(tantivy_dir)?))
        } else {
            None
        };

        match active_during_build {
            // First batch — construct + install.
            None => {
                let slot = Arc::new(LoadedSlot {
                    manifest: manifest.clone(),
                    chunks: RwLock::new(chunks_reader),
                    vectors: None,
                    dense: dense.clone(),
                    sparse: sparse_reader.map(|s| RwLock::new(s)),
                });
                {
                    let mut active = self.active.write().expect("active slot lock poisoned");
                    *active = Some(slot.clone());
                }
                *active_during_build = Some(slot);
            }
            // Subsequent batches — update the existing slot's
            // chunks + sparse readers in place. dense is shared
            // and grows automatically via flush_batch's insert.
            Some(slot) => {
                {
                    let mut chunks_lock =
                        slot.chunks.write().expect("LoadedSlot.chunks lock poisoned");
                    *chunks_lock = chunks_reader;
                }
                if let (Some(new), Some(lock)) = (sparse_reader, slot.sparse.as_ref()) {
                    let mut sparse_lock = lock.write().expect("LoadedSlot.sparse lock poisoned");
                    *sparse_lock = new;
                }
            }
        }
        Ok(())
    }

    /// Find a slot eligible for resume — i.e. one whose manifest is
    /// in `Planning` or `Building` state. If multiple exist, returns
    /// the most recent (slot ids are timestamp-prefixed and sort
    /// chronologically).
    ///
    /// Slots in `Failed` state are *not* returned — those represent
    /// builds that crashed or were rejected and should not be
    /// silently picked up by a fresh "Build" click.
    pub fn find_resumable_slot(&self) -> Result<Option<SlotId>, BucketError> {
        let slots_dir = slot::slots_dir(&self.root);
        if !slots_dir.exists() {
            return Ok(None);
        }
        let mut candidates: Vec<SlotId> = Vec::new();
        for entry in fs::read_dir(&slots_dir).map_err(BucketError::Io)? {
            let entry = entry.map_err(BucketError::Io)?;
            let meta = entry.metadata().map_err(BucketError::Io)?;
            if !meta.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let Some(slot_id) = name.to_str() else { continue };
            let manifest_path = slot::manifest_path(&entry.path());
            let Ok(text) = fs::read_to_string(&manifest_path) else {
                continue;
            };
            let Ok(manifest) = SlotManifest::from_toml_str(&text) else {
                continue;
            };
            if matches!(
                manifest.state,
                SlotState::Planning | SlotState::Building
            ) {
                candidates.push(slot_id.to_string());
            }
        }
        candidates.sort();
        Ok(candidates.into_iter().last())
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
    /// slot's embedder, if any. Reads from the manifest snapshot —
    /// works even during a build when `vectors.idx` doesn't exist
    /// yet (the embedder dimension is frozen at slot creation).
    pub fn active_dimension(&self) -> Option<u32> {
        self.active_snapshot().map(|s| s.manifest.embedder.dimension)
    }

    /// Test/inspection accessor. Fetches a vector by chunk id from the
    /// active slot. Returns `None` when no slot is active or while a
    /// build is in progress (the vectors reader needs `vectors.idx`,
    /// which is only written by `finalize`).
    pub fn fetch_vector(&self, chunk_id: ChunkId) -> Result<Option<Vec<f32>>, BucketError> {
        let Some(active) = self.active_snapshot() else {
            return Ok(None);
        };
        match active.vectors.as_ref() {
            Some(reader) => reader.fetch(chunk_id).map_err(BucketError::Io),
            None => Ok(None),
        }
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
            Some(slot) => match slot.manifest.state {
                SlotState::Ready => BucketStatus::Ready,
                SlotState::Building | SlotState::Planning => BucketStatus::Building {
                    slot_id: slot.manifest.slot_id.clone(),
                    progress: super::types::BuildProgress {
                        // The fine-grained progress lives in
                        // build.state on disk; this status is just
                        // "yes, queries hit a slot that's still
                        // being built." Callers wanting per-batch
                        // counts subscribe to BucketBuildProgress
                        // events.
                        stage: "indexing".to_string(),
                        completed: slot.dense.len() as u64,
                        total: None,
                    },
                },
                SlotState::Failed => BucketStatus::Failed {
                    reason: "slot in failed state".to_string(),
                },
                SlotState::Archived => BucketStatus::NoActiveSlot,
            },
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
            // embedder dimension. Reading from the manifest works
            // both for fully-built slots and slots still in their
            // Indexing phase (vectors.idx may not exist yet).
            let slot_dim = active.manifest.embedder.dimension;
            if query_vec.len() as u32 != slot_dim {
                return Err(BucketError::DimensionMismatch {
                    query: query_vec.len() as u32,
                    slot: slot_dim,
                });
            }

            let hits = active.dense.search(query_vec, top_k);
            let chunks = active.chunks();
            let mut out = Vec::with_capacity(hits.len());
            for (chunk_id, distance) in hits {
                // Look up chunk text for inclusion in the candidate.
                // A miss can happen during a mid-build query if HNSW
                // already has a position whose chunk_id wasn't yet
                // committed to the chunk-store snapshot — skip it
                // rather than erroring; the next query after the
                // missing chunk's batch will see consistent state.
                let Some(chunk) = chunks.fetch(chunk_id).map_err(BucketError::Io)? else {
                    continue;
                };
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
            let Some(sparse) = active.sparse() else {
                // Sparse path disabled for this bucket — empty
                // candidate list rather than an error, per the
                // trait contract.
                return Ok(Vec::new());
            };

            let hits = sparse.search(query_text, top_k)?;
            let chunks = active.chunks();
            let mut out = Vec::with_capacity(hits.len());
            for (chunk_id, score) in hits {
                // Same skip-on-miss as the dense path: a tantivy
                // commit may have outpaced the chunks-store snapshot
                // by one batch.
                let Some(chunk) = chunks.fetch(chunk_id).map_err(BucketError::Io)? else {
                    continue;
                };
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
            match active.chunks().fetch(chunk_id).map_err(BucketError::Io)? {
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

/// Notify the observer of a phase transition and append the matching
/// `PhaseChanged` record to `build.state`. The pair is published
/// together so the live UI signal and the resume log can never disagree
/// on what phase is current.
fn publish_phase(
    phase: BuildPhase,
    observer: Option<&dyn BuildObserver>,
    build_state: &mut BuildStateWriter,
) -> Result<(), BucketError> {
    if let Some(obs) = observer {
        obs.on_phase(phase);
    }
    build_state
        .append(&BuildStateRecord::PhaseChanged { phase })
        .map_err(BucketError::Io)?;
    Ok(())
}

/// Flush a buffered batch through every index. The order matters for
/// the durability barrier:
/// 1. Embed (async network I/O against the provider).
/// 2. For each chunk, write to all four indexes — chunks.bin,
///    vectors.bin, tantivy, dense (HNSW). Position numbers for
///    dense are derived from `base_position + offset_in_batch`.
/// 3. Caller is responsible for `flush()`-ing chunks/vectors and
///    `commit()`-ing tantivy after this returns, before logging the
///    BatchEmbedded checkpoint.
#[allow(clippy::too_many_arguments)]
async fn flush_batch(
    pending: &mut Vec<NewChunk>,
    chunks: &mut ChunkStoreWriter,
    vectors: &mut VectorStoreWriter,
    sparse: Option<&mut SparseIndexBuilder>,
    dense: &DenseIndex,
    base_position: u64,
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
    for (offset, (chunk, vector)) in pending.iter().zip(resp.embeddings.iter()).enumerate() {
        let chunk_id = ChunkId::from_source(&chunk.source_record_hash, chunk.chunk_offset);
        chunks
            .append(chunk_id, &chunk.source_ref, &chunk.text)
            .map_err(BucketError::Io)?;
        vectors.append(chunk_id, vector).map_err(BucketError::Io)?;
        if let Some(builder) = sparse.as_deref_mut() {
            builder.add(chunk_id, &chunk.text)?;
        }
        // Incremental HNSW: insert into the dense graph in lockstep
        // with the other indexes so a query against the in-progress
        // slot finds this chunk.
        dense.insert(chunk_id, base_position + offset as u64, vector);
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
        chunks: RwLock::new(Arc::new(chunks)),
        vectors: Some(vectors),
        dense: Arc::new(dense),
        sparse: sparse.map(|s| RwLock::new(Arc::new(s))),
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
        // HNSW with very small graphs (n=3) and ef_search=64 can
        // occasionally miss a neighbour on the layer traversal —
        // recall is high but not guaranteed-100%. Assert on what the
        // test is actually about: the exact-match chunk surfaces as
        // the top result.
        assert!(!results.is_empty(), "expected at least one hit");
        assert!(results[0].chunk_text.contains("beta body"));
        assert!(results[0].source_score < 0.001);
        assert_eq!(results[0].path, SearchPath::Dense);
        assert_eq!(results[0].bucket_id, *bucket.id());

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
    async fn build_writes_full_build_state_log() {
        // End-to-end verification of the resume log: a successful
        // build emits the full sequence of records (Planning →
        // Planned×N → Indexing → BatchEmbedded×M → BuildingDense →
        // Finalizing → BuildCompleted) in the right order. Resume
        // code (Commit B) consumes this exact format.
        use crate::knowledge::build_state::{self, BuildStateRecord};

        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "alpha.md", "alpha body");
        write_md(&source, "beta.md", "beta body");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        let slot_id = bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();

        let path = slot::slot_dir(&bucket_root, &slot_id).join(slot::BUILD_STATE);
        let records = build_state::read_all(&path).unwrap();

        // Planning + 2 Planned + Indexing + ≥1 BatchEmbedded + BuildingDense + Finalizing + BuildCompleted.
        assert!(
            records.len() >= 8,
            "expected at least 8 records, got {}: {records:#?}",
            records.len()
        );

        // Phases land in the documented order.
        let phases: Vec<BuildPhase> = records
            .iter()
            .filter_map(|r| match r {
                BuildStateRecord::PhaseChanged { phase } => Some(*phase),
                _ => None,
            })
            .collect();
        assert_eq!(
            phases,
            vec![
                BuildPhase::Planning,
                BuildPhase::Indexing,
                BuildPhase::BuildingDense,
                BuildPhase::Finalizing,
            ]
        );

        // Two source records → two Planned entries.
        let planned_count = records
            .iter()
            .filter(|r| matches!(r, BuildStateRecord::Planned { .. }))
            .count();
        assert_eq!(planned_count, 2);

        // Last record must be BuildCompleted.
        assert!(matches!(
            records.last().unwrap(),
            BuildStateRecord::BuildCompleted
        ));

        // BatchEmbedded carries the running record/chunk counts.
        let batches: Vec<&BuildStateRecord> = records
            .iter()
            .filter(|r| matches!(r, BuildStateRecord::BatchEmbedded { .. }))
            .collect();
        assert!(!batches.is_empty(), "at least one batch should be recorded");
        let last = batches.last().unwrap();
        match last {
            BuildStateRecord::BatchEmbedded {
                source_records_completed,
                chunks_completed,
                ..
            } => {
                assert_eq!(*source_records_completed, 2);
                assert!(*chunks_completed >= 2);
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn cancelled_build_leaves_log_without_build_completed() {
        // The Cancelled path must not leave a BuildCompleted record
        // — that's the marker the resume path uses to decide whether
        // to redo work. A cancelled build looks the same as a crashed
        // build to the resume code.
        use crate::knowledge::build_state::{self, BuildStateRecord};

        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hello");
        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let _ = bucket
            .build_slot(&adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap_err();

        // Cancellation fires before the slot is promoted, but the
        // slot directory and build.state already exist. Find them.
        let slots_dir = slot::slots_dir(&bucket_root);
        let entry = fs::read_dir(&slots_dir)
            .unwrap()
            .find_map(|e| {
                let e = e.ok()?;
                let m = e.metadata().ok()?;
                if m.is_dir() { Some(e.path()) } else { None }
            })
            .expect("partial slot dir");
        let path = entry.join(slot::BUILD_STATE);
        let records = build_state::read_all(&path).unwrap();
        assert!(
            !records
                .iter()
                .any(|r| matches!(r, BuildStateRecord::BuildCompleted)),
            "cancelled build must not record BuildCompleted: {records:#?}"
        );
    }

    /// Embedder that fires the cancel token once it has fielded
    /// `cancel_after_calls` embed requests. Lets tests force a
    /// mid-build cancellation deterministically without racing on
    /// timing.
    struct CancellingEmbedder {
        inner: MockEmbedder,
        cancel_after_calls: usize,
        cancel: CancellationToken,
    }

    impl CancellingEmbedder {
        fn new(dim: u32, cancel_after_calls: usize, cancel: CancellationToken) -> Self {
            Self {
                inner: MockEmbedder::new(dim),
                cancel_after_calls,
                cancel,
            }
        }
    }

    impl EmbeddingProvider for CancellingEmbedder {
        fn embed<'a>(
            &'a self,
            req: &'a ProviderEmbedRequest<'a>,
            cancel: &'a CancellationToken,
        ) -> ProviderBoxFuture<'a, Result<EmbeddingResponse, EmbeddingError>> {
            let inner = &self.inner;
            let cancel_after = self.cancel_after_calls;
            let token = self.cancel.clone();
            Box::pin(async move {
                // Run the underlying mock first so the batch lands.
                let result = inner.embed(req, cancel).await;
                let calls = inner.embed_calls.load(Ordering::SeqCst);
                if calls >= cancel_after {
                    token.cancel();
                }
                result
            })
        }

        fn list_models<'a>(
            &'a self,
        ) -> ProviderBoxFuture<'a, Result<Vec<EmbeddingModelInfo>, EmbeddingError>> {
            self.inner.list_models()
        }
    }

    #[tokio::test]
    async fn resume_completes_a_partially_built_slot() {
        // End-to-end resume test:
        // 1. Build a slot, cancel after the first batch lands.
        // 2. Verify the slot is in Building state with a partial
        //    build.state log.
        // 3. Resume: should complete the slot, preserve the slot id,
        //    and produce the same total chunk count as a fresh build.
        // 4. Search the resumed slot to verify it's queryable.
        use crate::knowledge::build_state::{self, BuildStateRecord};

        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();

        // Enough records for at least 2 batches at EMBED_BATCH_SIZE=128.
        let total_records = super::EMBED_BATCH_SIZE * 2 + 30;
        for i in 0..total_records {
            write_md(
                &source,
                &format!("doc-{i:04}.md"),
                &format!("doc-{i:04} body content"),
            );
        }

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);

        // Cancel after exactly 1 batch — leaves ~128 records embedded
        // and ~158 records still to do on resume.
        let cancel = CancellationToken::new();
        let cancelling = Arc::new(CancellingEmbedder::new(8, 1, cancel.clone()));
        let _ = bucket
            .build_slot(&adapter, &chunker, cancelling.clone(), None, &cancel)
            .await
            .unwrap_err();

        // Find the partial slot and verify it's resumable.
        let resumable = bucket.find_resumable_slot().unwrap().expect(
            "find_resumable_slot must return the in-progress slot after cancel",
        );

        let slot_path = slot::slot_dir(&bucket_root, &resumable);
        let manifest =
            SlotManifest::from_toml_str(&fs::read_to_string(slot::manifest_path(&slot_path)).unwrap())
                .unwrap();
        assert_eq!(manifest.state, SlotState::Building);

        let log_pre = build_state::read_all(&slot_path.join(slot::BUILD_STATE)).unwrap();
        let pre_batches = log_pre
            .iter()
            .filter(|r| matches!(r, BuildStateRecord::BatchEmbedded { .. }))
            .count();
        assert!(pre_batches >= 1, "at least one batch should be checkpointed");
        assert!(
            !log_pre
                .iter()
                .any(|r| matches!(r, BuildStateRecord::BuildCompleted)),
            "BuildCompleted must not appear in cancelled log"
        );

        // Resume with a fresh (non-cancelling) embedder.
        let resume_cancel = CancellationToken::new();
        let plain_embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let resumed_slot_id = bucket
            .resume_slot(
                &resumable,
                &adapter,
                &chunker,
                plain_embedder,
                None,
                &resume_cancel,
            )
            .await
            .unwrap();
        assert_eq!(
            resumed_slot_id, resumable,
            "resume must keep the same slot id"
        );
        assert_eq!(bucket.active_slot_id().as_deref(), Some(resumable.as_str()));

        // Verify the resumed slot's stats match a from-scratch
        // build's stats (same chunker + source → same chunk count).
        let manifest =
            SlotManifest::from_toml_str(&fs::read_to_string(slot::manifest_path(&slot_path)).unwrap())
                .unwrap();
        assert_eq!(manifest.state, SlotState::Ready);
        assert_eq!(manifest.stats.source_records, total_records as u64);
        assert_eq!(manifest.stats.chunk_count, total_records as u64);
        assert_eq!(manifest.stats.vector_count, total_records as u64);

        let log_post =
            build_state::read_all(&slot_path.join(slot::BUILD_STATE)).unwrap();
        assert!(matches!(
            log_post.last().unwrap(),
            BuildStateRecord::BuildCompleted
        ));

        // The completed log should contain MORE BatchEmbedded entries
        // than the pre-resume snapshot (resume added more batches).
        let post_batches = log_post
            .iter()
            .filter(|r| matches!(r, BuildStateRecord::BatchEmbedded { .. }))
            .count();
        assert!(post_batches > pre_batches);

        // Resumed slot is queryable on both paths.
        let cancel_q = CancellationToken::new();
        let q_text = "doc-0050 body content";
        let q_record = crate::knowledge::SourceRecord::new("ignored", q_text);
        let q_chunks = chunker.chunk(&q_record);
        let query_vec = MockEmbedder::fake_vector(&q_chunks[0].text, 8);
        let dense_hits = bucket.dense_search(&query_vec, 5, &cancel_q).await.unwrap();
        assert!(!dense_hits.is_empty(), "resumed slot must be queryable on dense");
        let sparse_hits = bucket.sparse_search("doc-0050", 5, &cancel_q).await.unwrap();
        assert!(!sparse_hits.is_empty(), "resumed slot must be queryable on sparse");
    }

    #[tokio::test]
    async fn resume_planning_state_restarts_from_scratch() {
        // If the slot is in Planning state (cancelled before any
        // chunks landed), resume should discard the partial build.state
        // and restart cleanly.
        use crate::knowledge::build_state::{self, BuildStateRecord};

        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "a.md", "alpha");
        write_md(&source, "b.md", "beta");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);

        // Hand-craft a Planning-state slot directory: empty
        // build.state, manifest with state=Planning, no chunks/vectors.
        let slot_id = slot::generate_slot_id();
        let slot_path = slot::slot_dir(&bucket_root, &slot_id);
        fs::create_dir_all(&slot_path).unwrap();
        let now = chrono::Utc::now();
        let manifest = SlotManifest {
            slot_id: slot_id.clone(),
            state: SlotState::Planning,
            created_at: now,
            build_started_at: Some(now),
            build_completed_at: None,
            chunker_snapshot: bucket.config().chunker.clone(),
            embedder: EmbedderSnapshot {
                provider: bucket.config().defaults.embedder.clone(),
                model_id: "mock-embedder".into(),
                dimension: 4,
            },
            sparse: Some(SparseSnapshot {
                tokenizer: bucket.config().search_paths.sparse.tokenizer.clone(),
            }),
            serving: ServingSnapshot {
                mode: bucket.config().defaults.serving_mode,
                quantization: bucket.config().defaults.quantization,
            },
            stats: SlotStats::default(),
            lineage: None,
        };
        write_manifest(&slot_path, &manifest).unwrap();
        // Half-written build.state with one Planned + no PhaseChanged
        // (simulates planning interrupted right after one entry).
        let bs_path = slot_path.join(slot::BUILD_STATE);
        let mut bs = BuildStateWriter::create_or_open(&bs_path).unwrap();
        bs.append(&BuildStateRecord::PhaseChanged {
            phase: BuildPhase::Planning,
        })
        .unwrap();
        bs.append(&BuildStateRecord::Planned {
            source_id: "a.md".into(),
            content_hash: [0xfe; 32], // deliberately wrong hash
        })
        .unwrap();
        drop(bs);

        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        let returned = bucket
            .resume_slot(&slot_id, &adapter, &chunker, embedder, None, &cancel)
            .await
            .unwrap();
        assert_eq!(returned, slot_id);

        // Build completed and the bogus Planned hash got replaced.
        let log = build_state::read_all(&bs_path).unwrap();
        assert!(matches!(
            log.last().unwrap(),
            BuildStateRecord::BuildCompleted
        ));
        let bad_hash = [0xfe; 32];
        let still_has_bad = log.iter().any(|r| {
            matches!(r, BuildStateRecord::Planned { content_hash, .. } if *content_hash == bad_hash)
        });
        assert!(
            !still_has_bad,
            "Planning restart must discard the prior bogus Planned entry"
        );
    }

    #[tokio::test]
    async fn resume_drift_errors_loudly() {
        // Source content changing under a paused build is a hard
        // error — the user must delete and rebuild rather than
        // silently mixing old and new content in one slot.
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        let total_records = super::EMBED_BATCH_SIZE + 5;
        for i in 0..total_records {
            write_md(
                &source,
                &format!("doc-{i:04}.md"),
                &format!("doc-{i:04} body content"),
            );
        }
        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);

        let cancel = CancellationToken::new();
        let cancelling = Arc::new(CancellingEmbedder::new(8, 1, cancel.clone()));
        let _ = bucket
            .build_slot(&adapter, &chunker, cancelling, None, &cancel)
            .await
            .unwrap_err();
        let resumable = bucket.find_resumable_slot().unwrap().unwrap();

        // Mutate one of the source records that should already have
        // been Planned (record 0) so resume sees content drift.
        write_md(&source, "doc-0000.md", "DRIFTED CONTENT");

        let resume_cancel = CancellationToken::new();
        let plain: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let err = bucket
            .resume_slot(&resumable, &adapter, &chunker, plain, None, &resume_cancel)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("drift") || msg.contains("Drift"),
            "expected drift error, got: {msg}"
        );
    }

    /// Build observer that pauses the build right after the first
    /// Indexing batch fully lands + installs in the active slot.
    /// Triggers off `on_progress` *only when the current phase is
    /// Indexing* (not Planning, where `on_progress` ticks per
    /// record before any slot install).
    struct PauseAfterFirstBatch {
        first_batch_landed: tokio::sync::Notify,
        proceed: std::sync::Mutex<bool>,
        proceed_cv: std::sync::Condvar,
        in_indexing: AtomicBool,
        triggered: AtomicBool,
    }

    impl PauseAfterFirstBatch {
        fn new() -> Self {
            Self {
                first_batch_landed: tokio::sync::Notify::new(),
                proceed: std::sync::Mutex::new(false),
                proceed_cv: std::sync::Condvar::new(),
                in_indexing: AtomicBool::new(false),
                triggered: AtomicBool::new(false),
            }
        }

        fn release(&self) {
            *self.proceed.lock().unwrap() = true;
            self.proceed_cv.notify_all();
        }
    }

    impl BuildObserver for PauseAfterFirstBatch {
        fn on_phase(&self, phase: BuildPhase) {
            self.in_indexing
                .store(matches!(phase, BuildPhase::Indexing), Ordering::SeqCst);
        }
        fn on_progress(&self, _source_records: u64, _chunks: u64) {
            if !self.in_indexing.load(Ordering::SeqCst) {
                return;
            }
            // Block only on the first batch of Indexing. After that,
            // run to completion.
            if self
                .triggered
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.first_batch_landed.notify_one();
                let mut guard = self.proceed.lock().unwrap();
                while !*guard {
                    guard = self.proceed_cv.wait(guard).unwrap();
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slot_is_queryable_during_in_progress_build() {
        // While Indexing is mid-stream, queries against the bucket
        // see the chunks already committed by past batches. Both
        // dense and sparse paths return hits without waiting for
        // the build to complete.
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        // Two batches' worth of records: one full batch lands and
        // installs the active slot via update_active_during_build,
        // then the test queries, then the trailing partial batch
        // finishes the build.
        let total = super::EMBED_BATCH_SIZE + 5;
        for i in 0..total {
            write_md(
                &source,
                &format!("doc-{i:04}.md"),
                &format!("doc-{i:04} body content"),
            );
        }

        let bucket_root = tmp.path().join("bucket");
        let bucket = Arc::new(open_test_bucket(&bucket_root, &source));
        let chunker_for_query = TokenBasedChunker::from_config(&bucket.config().chunker);
        let pause = Arc::new(PauseAfterFirstBatch::new());
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));

        let bucket_for_task = bucket.clone();
        let pause_for_task = pause.clone();
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let source_for_task = source.clone();
        let build_task = tokio::spawn(async move {
            let adapter = MarkdownDir::new(&source_for_task);
            let chunker = TokenBasedChunker::from_config(&bucket_for_task.config().chunker);
            let observer: &dyn BuildObserver = &*pause_for_task;
            bucket_for_task
                .build_slot(&adapter, &chunker, embedder, Some(observer), &cancel_for_task)
                .await
        });

        // Wait for the first batch to land and the slot to be
        // published as active.
        pause.first_batch_landed.notified().await;

        // Status reports Building (not Ready) while the build is
        // still mid-stream.
        let st = bucket.status();
        assert!(
            matches!(st, BucketStatus::Building { .. }),
            "expected Building, got {st:?}"
        );

        // Query the partial slot.
        let cancel_q = CancellationToken::new();
        let q_record = crate::knowledge::SourceRecord::new("ignored", "doc-0050 body content");
        let q_chunks = chunker_for_query.chunk(&q_record);
        let query_vec = MockEmbedder::fake_vector(&q_chunks[0].text, 8);
        let dense_hits = bucket.dense_search(&query_vec, 5, &cancel_q).await.unwrap();
        assert!(
            !dense_hits.is_empty(),
            "in-progress slot must serve dense queries after the first batch"
        );
        let sparse_hits = bucket
            .sparse_search("doc-0050", 5, &cancel_q)
            .await
            .unwrap();
        assert!(
            !sparse_hits.is_empty(),
            "in-progress slot must serve sparse queries after the first batch"
        );

        // Release the build to completion.
        pause.release();
        let slot_id = build_task.await.unwrap().unwrap();
        assert_eq!(bucket.active_slot_id().as_deref(), Some(slot_id.as_str()));
        assert!(matches!(bucket.status(), BucketStatus::Ready));

        // After completion the trailing batch is also queryable.
        let q_record2 =
            crate::knowledge::SourceRecord::new("ignored", "doc-0131 body content");
        let q_chunks2 = chunker_for_query.chunk(&q_record2);
        let query_vec2 = MockEmbedder::fake_vector(&q_chunks2[0].text, 8);
        let dense_hits_post =
            bucket.dense_search(&query_vec2, 5, &cancel_q).await.unwrap();
        assert!(!dense_hits_post.is_empty());
        let sparse_hits_post = bucket
            .sparse_search("doc-0131", 5, &cancel_q)
            .await
            .unwrap();
        assert!(!sparse_hits_post.is_empty());
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
