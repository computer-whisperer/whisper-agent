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
use super::chunks::{ChunkStoreReader, ChunkStoreWriter, LoadMode};
use super::config::{BucketConfig, ServingMode};
use super::dense::{DenseIndex, HnswParams};
use super::manifest::{
    ChunkerSnapshot, EmbedderSnapshot, ServingSnapshot, SlotManifest, SlotState, SlotStats,
    SparseSnapshot,
};
use super::slot;
use super::source::{SourceAdapter, SourceError, SourceRecord};
use super::sparse::{SparseIndex, SparseIndexBuilder};
use super::tombstones::Tombstones;
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

/// Dump the in-progress HNSW + sidecar every N batches so resume can
/// pick up the dump instead of rebuilding the graph from vectors.bin
/// (an `O(M N log N)` cost proportional to how far the prior attempt
/// got). The dump itself blocks new appends for its duration since
/// hnsw_rs serializes writes against the read lock — at scales where
/// dumps are expensive, the embedder network roundtrips dominate
/// anyway. Picked 32 batches (= 4096 chunks at EMBED_BATCH_SIZE=128)
/// as a balance: small enough that workspace-scale builds get at
/// least one dump if cancelled past the first few batches, large
/// enough that wiki-scale builds do ~7k dumps over a multi-day build
/// rather than hundreds of thousands.
///
/// Lowered to 1 in tests so the resume tests cover both fast-path
/// (sidecar load) and fallback (rebuild) without needing to feed
/// thousands of records through the build to trigger a real dump.
#[cfg(not(test))]
const DENSE_DUMP_BATCH_INTERVAL: u64 = 32;
#[cfg(test)]
const DENSE_DUMP_BATCH_INTERVAL: u64 = 1;

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

/// Boxed source iterator threaded through the build pipeline. Naming
/// the type once keeps the struct definitions below readable.
type SourceIter<'a> = Box<dyn Iterator<Item = Result<SourceRecord, SourceError>> + Send + 'a>;

/// Mutable owned state driven forward by [`DiskBucket::run_after_planning`]:
/// the manifest, the four index writers, and the build-state log. Bundled
/// so the function signature stays tractable.
struct BuildHandles {
    manifest: SlotManifest,
    chunk_writer: ChunkStoreWriter,
    vector_writer: VectorStoreWriter,
    sparse_builder: Option<SparseIndexBuilder>,
    build_state: BuildStateWriter,
}

/// External services + cancellation token + the active source iterator
/// (already advanced past `resume.records_completed` records when
/// resuming a Building-state slot). Passed by value into
/// [`DiskBucket::run_after_planning`] so the iterator's borrow of the
/// caller's `adapter` stays explicit at the call site.
struct BuildServices<'a> {
    iter: SourceIter<'a>,
    chunker: &'a dyn Chunker,
    embedder: Arc<dyn EmbeddingProvider>,
    observer: Option<&'a dyn BuildObserver>,
    cancel: &'a CancellationToken,
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
    /// Embedding provider matching the active slot's recorded
    /// `embedder.model_id`. Wired by the registry at bucket-open time
    /// (or by tests via `set_embedder`); `Bucket::insert` errors out
    /// when this is `None`.
    embedder: RwLock<Option<Arc<dyn EmbeddingProvider>>>,
    /// Single-writer guard for the mutation triad. `insert` holds
    /// it for its full async span (embed → append → HNSW → tantivy)
    /// so concurrent inserts serialize. `tombstone` mutations remain
    /// independent — they only touch `tombstones.bin`, which has its
    /// own internal serialization.
    mutation_lock: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for DiskBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskBucket")
            .field("id", &self.id)
            .field("root", &self.root)
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
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
/// - `tombstones` ([`Tombstones`]) holds its own internal `RwLock` /
///   `Mutex`; queries take the read lock for `O(log N)` membership
///   tests, [`Bucket::tombstone`] takes the write lock briefly to
///   merge new ids in. The same handle persists across the slot's
///   lifetime.
/// - `delta_chunks` is `None` until the first `Bucket::insert` runs
///   against this slot (or until `load_slot` finds existing delta
///   files on disk). Queries fall back to it after the base reader
///   misses. The HNSW holds delta entries inline (fold-into-base, no
///   separate delta graph in v1) so dense/sparse searches surface
///   delta hits transparently — only the chunk-text fetch needs to
///   consult the second store.
struct LoadedSlot {
    manifest: SlotManifest,
    chunks: RwLock<Arc<ChunkStoreReader>>,
    vectors: Option<VectorStoreReader>,
    dense: Arc<DenseIndex>,
    sparse: Option<RwLock<Arc<SparseIndex>>>,
    tombstones: Arc<Tombstones>,
    delta_chunks: RwLock<Option<Arc<ChunkStoreReader>>>,
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
        self.sparse.as_ref().map(|lock| {
            lock.read()
                .expect("LoadedSlot.sparse lock poisoned")
                .clone()
        })
    }

    fn tombstones(&self) -> Arc<Tombstones> {
        self.tombstones.clone()
    }

    fn delta_chunks(&self) -> Option<Arc<ChunkStoreReader>> {
        self.delta_chunks
            .read()
            .expect("LoadedSlot.delta_chunks lock poisoned")
            .clone()
    }

    /// Look up a chunk by id in base, then delta. Mirrors the lookup
    /// order used by `dense_search` / `sparse_search` candidate
    /// resolution.
    fn fetch_chunk_any(&self, chunk_id: ChunkId) -> std::io::Result<Option<Chunk>> {
        if let Some(c) = self.chunks().fetch(chunk_id)? {
            return Ok(Some(c));
        }
        if let Some(delta) = self.delta_chunks() {
            return delta.fetch(chunk_id);
        }
        Ok(None)
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
            embedder: RwLock::new(None),
            mutation_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// Inject the embedding provider that `Bucket::insert` will call
    /// against. Should match the active slot's recorded
    /// `embedder.model_id`; mismatch is detected at insert time, not
    /// here. Replacing an already-set embedder is allowed (model
    /// rotation) but the caller is responsible for ensuring it matches
    /// the slot's expected dimension.
    pub fn set_embedder(&self, embedder: Arc<dyn EmbeddingProvider>) {
        let mut slot = self.embedder.write().expect("embedder lock poisoned");
        *slot = Some(embedder);
    }

    fn embedder(&self) -> Option<Arc<dyn EmbeddingProvider>> {
        self.embedder
            .read()
            .expect("embedder lock poisoned")
            .clone()
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
        chunker_snapshot: ChunkerSnapshot,
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
            chunker_snapshot,
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

        let source_records = self.do_planning_phase(
            &slot_path,
            &mut manifest,
            adapter,
            &mut build_state,
            observer,
            cancel,
        )?;

        // Fresh writers — the slot dir is brand new.
        let handles = open_fresh_handles(
            &slot_path,
            manifest,
            embedder_snapshot.dimension,
            sparse_enabled,
            build_state,
        )?;

        self.run_after_planning(
            &slot_id,
            &slot_path,
            handles,
            BuildServices {
                iter: adapter.enumerate(),
                chunker,
                embedder,
                observer,
                cancel,
            },
            ResumePoint::default(),
            source_records,
        )
        .await
    }

    /// Walk the source once, append a `Planned` record per source
    /// record with its content hash, sync the log, and promote the
    /// manifest from `Planning` → `Building`. Shared between
    /// fresh-build and resume-from-Planning code paths.
    ///
    /// The single sync at end-of-loop is the resume invariant for
    /// this phase: after this returns, every `Planned` line is
    /// durable. Per-record fsync isn't worth the syscall budget
    /// (Planning crashes just replan).
    fn do_planning_phase(
        &self,
        slot_path: &Path,
        manifest: &mut SlotManifest,
        adapter: &dyn SourceAdapter,
        build_state: &mut BuildStateWriter,
        observer: Option<&dyn BuildObserver>,
        cancel: &CancellationToken,
    ) -> Result<u64, BucketError> {
        publish_phase(BuildPhase::Planning, observer, build_state)?;
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
        build_state.sync().map_err(BucketError::Io)?;

        // Promote manifest to Building once planning is done. Writing
        // here (rather than at the very start of build_slot) lets the
        // resume code distinguish "planning was interrupted, redo it"
        // (state=Planning) from "planning finished, embedding was
        // interrupted" (state=Building).
        manifest.state = SlotState::Building;
        write_manifest(slot_path, manifest)?;
        Ok(source_records)
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
    #[allow(clippy::too_many_arguments)]
    pub async fn resume_slot(
        &self,
        slot_id: &str,
        adapter: &dyn SourceAdapter,
        chunker: &dyn Chunker,
        chunker_snapshot: ChunkerSnapshot,
        embedder: Arc<dyn EmbeddingProvider>,
        observer: Option<&dyn BuildObserver>,
        cancel: &CancellationToken,
    ) -> Result<SlotId, BucketError> {
        let slot_path = slot::slot_dir(&self.root, slot_id);
        let manifest_text =
            fs::read_to_string(slot::manifest_path(&slot_path)).map_err(|e| match e.kind() {
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
        if manifest.chunker_snapshot != chunker_snapshot {
            return Err(BucketError::Other(format!(
                "resume_slot: chunker snapshot changed since slot {slot_id} was started \
                 (params or tokenizer); delete the bucket and rebuild to use the new chunker"
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
        // Same dim, different model = silently mixed vectors. Vectors
        // already on disk came from the prior model; vectors we'd
        // append now would come from the new model. Even if the
        // arithmetic happens to typecheck, the index is corrupt for
        // queries (different models live in different vector spaces).
        if manifest.embedder.model_id != embedder_snapshot.model_id
            || manifest.embedder.provider != embedder_snapshot.provider
        {
            return Err(BucketError::Other(format!(
                "resume_slot: embedder identity changed ({}/{} → {}/{}) since slot {slot_id} was started; \
                 delete the bucket and rebuild",
                manifest.embedder.provider,
                manifest.embedder.model_id,
                embedder_snapshot.provider,
                embedder_snapshot.model_id,
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
            let mut build_state =
                BuildStateWriter::create_or_open(&build_state_path).map_err(BucketError::Io)?;
            let source_records = self.do_planning_phase(
                &slot_path,
                &mut manifest,
                adapter,
                &mut build_state,
                observer,
                cancel,
            )?;
            let handles = open_fresh_handles(
                &slot_path,
                manifest,
                embedder_snapshot.dimension,
                sparse_enabled,
                build_state,
            )?;
            return self
                .run_after_planning(
                    slot_id,
                    &slot_path,
                    handles,
                    BuildServices {
                        iter: adapter.enumerate(),
                        chunker,
                        embedder,
                        observer,
                        cancel,
                    },
                    ResumePoint::default(),
                    source_records,
                )
                .await;
        }

        // Building state ⇒ replay build.state to find the resume point.
        let prior_records =
            super::build_state::read_all(&build_state_path).map_err(BucketError::Io)?;
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
        let (last_batch_index, records_done, chunks_done) = last_batch.unwrap_or((u64::MAX, 0, 0)); // u64::MAX → 0 below

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
        for (i, (expected_source_id, expected_hash)) in
            planned.iter().take(records_to_skip as usize).enumerate()
        {
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
        // `iter` is now positioned at record `records_to_skip`.
        // We hand it off to run_after_planning rather than dropping
        // and re-walking.
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

        // Hydrate the in-memory HNSW for the chunks already on disk
        // so subsequent inserts during the resumed Indexing phase
        // continue at position `chunks_done`. Two paths:
        //
        //  1. Fast path — load the periodic dump if its sidecar
        //     `dense.snapshot` reports exactly `chunks_done` points.
        //     That's an O(file size) read; sub-second at workspace
        //     scale, ~tens of seconds at wiki scale.
        //
        //  2. Fallback — sequentially read vectors.bin and re-insert
        //     each vector into a fresh HNSW. O(M N log N) work
        //     proportional to chunks_done; this is what every resume
        //     paid before the periodic-dump landed, and it remains
        //     the path used when the prior build crashed before any
        //     dump finished, or after a BatchEmbedded for which no
        //     dump has caught up yet.
        //
        // A sidecar value greater than `chunks_done` shouldn't
        // happen (dump always lags BatchEmbedded log entries) but
        // we treat it as "trust the log" and fall back to rebuild
        // rather than load a dump that has more points than the
        // log says are durable.
        let dense_for_resume: Option<Arc<DenseIndex>> = if chunks_done == 0 {
            None
        } else {
            let dump_snapshot =
                super::dense::read_dump_snapshot(&slot_path).map_err(BucketError::Io)?;
            if dump_snapshot == Some(chunks_done) {
                let by_position = resumed_chunk_ids.clone();
                let slot_path_for_thread = slot_path.clone();
                let dense =
                    tokio::task::spawn_blocking(move || -> Result<DenseIndex, BucketError> {
                        DenseIndex::load_from_with_positions(&slot_path_for_thread, by_position)
                    })
                    .await
                    .map_err(|e| BucketError::Other(format!("spawn_blocking dense load: {e}")))??;
                Some(Arc::new(dense))
            } else {
                let dense = DenseIndex::empty(
                    HnswParams::default(),
                    (planned.len().saturating_mul(8)).max(1024),
                );
                let bin_path_for_thread = vectors_bin.clone();
                let dim = embedder_snapshot.dimension;
                let chunk_ids = resumed_chunk_ids.clone();
                let dense =
                    tokio::task::spawn_blocking(move || -> Result<DenseIndex, BucketError> {
                        let mut bin =
                            std::fs::File::open(&bin_path_for_thread).map_err(BucketError::Io)?;
                        use std::io::{Read, Seek, SeekFrom};
                        for (position, chunk_id) in chunk_ids.iter().enumerate() {
                            let byte_offset = 16u64 + position as u64 * dim as u64 * 4;
                            bin.seek(SeekFrom::Start(byte_offset))
                                .map_err(BucketError::Io)?;
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
                    .map_err(|e| {
                        BucketError::Other(format!("spawn_blocking dense rebuild: {e}"))
                    })??;
                Some(Arc::new(dense))
            }
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
        let handles = BuildHandles {
            manifest,
            chunk_writer,
            vector_writer,
            sparse_builder,
            build_state,
        };
        self.run_after_planning(
            slot_id,
            &slot_path,
            handles,
            BuildServices {
                iter,
                chunker,
                embedder,
                observer,
                cancel,
            },
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
    ///
    /// `services.iter` must already be advanced past
    /// `resume.records_completed` records when resuming a Building
    /// slot; for fresh builds (and resume from Planning) the iterator
    /// is fresh and `resume.records_completed == 0`.
    async fn run_after_planning(
        &self,
        slot_id: &str,
        slot_path: &Path,
        handles: BuildHandles,
        services: BuildServices<'_>,
        resume: ResumePoint,
        source_records: u64,
    ) -> Result<SlotId, BucketError> {
        let BuildHandles {
            mut manifest,
            mut chunk_writer,
            mut vector_writer,
            mut sparse_builder,
            mut build_state,
        } = handles;
        let BuildServices {
            iter,
            chunker,
            embedder,
            observer,
            cancel,
        } = services;
        let sparse_enabled = self.config.search_paths.sparse.enabled;

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

        // Periodic-dump cadence: every DENSE_DUMP_BATCH_INTERVAL
        // BatchEmbedded records, dump the HNSW + sidecar so resume
        // can skip the M-N-log-N rebuild from vectors.bin.
        let mut batches_since_dump: u64 = 0;

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
                    slot_path,
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
                // Durability barrier: BatchEmbedded is the resume
                // checkpoint. The chunks/vectors/sparse writers have
                // already flushed + committed above; this fsync makes
                // the log entry stating "everything up to here is
                // durable" itself durable.
                build_state.sync().map_err(BucketError::Io)?;
                batch_index += 1;
                batches_since_dump += 1;
                if batches_since_dump >= DENSE_DUMP_BATCH_INTERVAL {
                    periodic_dump(slot_path, slot_id, &dense).await?;
                    batches_since_dump = 0;
                }
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
                slot_path,
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
            build_state.sync().map_err(BucketError::Io)?;
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

        // Match the slot's serving mode for the post-finalize readers
        // — same dispatch as `load_slot` does on bucket open. The build
        // path runs once at slot completion; readers stay live for the
        // slot's serving lifetime.
        let load_mode = serving_mode_to_load_mode(manifest.serving.mode);
        let chunks = ChunkStoreReader::open_with_mode(&chunks_bin, &chunks_idx, load_mode)
            .map_err(BucketError::Io)?;
        let vectors = VectorStoreReader::open_with_mode(&vectors_bin, &vectors_idx, load_mode)
            .map_err(BucketError::Io)?;

        // --- Phase: BuildingDense ---
        // HNSW is already built incrementally — this phase is now
        // just the HNSW dump for next-open's benefit. We still emit
        // the phase tag so existing wire consumers / tests don't
        // break, even though counters don't tick.
        publish_phase(BuildPhase::BuildingDense, observer, &mut build_state)?;
        periodic_dump(slot_path, slot_id, &dense).await?;

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

        let tombstones =
            Arc::new(Tombstones::open(&slot::tombstones_path(slot_path)).map_err(BucketError::Io)?);
        let loaded = Arc::new(LoadedSlot {
            manifest,
            chunks: RwLock::new(Arc::new(chunks)),
            vectors: Some(vectors),
            dense,
            sparse: sparse.map(|s| RwLock::new(Arc::new(s))),
            tombstones,
            delta_chunks: RwLock::new(None),
        });

        slot::set_active_slot(&self.root, slot_id)?;
        {
            let mut active = self.active.write().expect("active slot lock poisoned");
            *active = Some(loaded);
        }

        build_state
            .append(&BuildStateRecord::BuildCompleted)
            .map_err(BucketError::Io)?;
        build_state.sync().map_err(BucketError::Io)?;

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
    fn update_active_during_build(
        &self,
        slot_path: &Path,
        chunk_writer: &ChunkStoreWriter,
        dense: &Arc<DenseIndex>,
        manifest: &SlotManifest,
        install_during_build: bool,
        active_during_build: &mut Option<Arc<LoadedSlot>>,
    ) -> Result<(), BucketError> {
        if !install_during_build {
            return Ok(());
        }

        // Tantivy doesn't expose a live-segment view, so on every
        // call we re-open the index to pick up the latest committed
        // segments from this batch.
        let sparse_reader = if self.config.search_paths.sparse.enabled {
            let tantivy_dir = slot::tantivy_dir(slot_path);
            Some(Arc::new(SparseIndex::open(&tantivy_dir)?))
        } else {
            None
        };

        match active_during_build {
            // First batch — construct + install. The chunks reader
            // shares the writer's index Arc, so subsequent appends
            // become visible automatically; we never replace it.
            None => {
                let chunks_bin = slot::chunks_bin_path(slot_path);
                let chunks_reader = Arc::new(
                    ChunkStoreReader::open_with_shared_index(
                        &chunks_bin,
                        chunk_writer.share_index(),
                    )
                    .map_err(BucketError::Io)?,
                );
                let tombstones = Arc::new(
                    Tombstones::open(&slot::tombstones_path(slot_path)).map_err(BucketError::Io)?,
                );
                let slot = Arc::new(LoadedSlot {
                    manifest: manifest.clone(),
                    chunks: RwLock::new(chunks_reader),
                    vectors: None,
                    dense: dense.clone(),
                    sparse: sparse_reader.map(RwLock::new),
                    tombstones,
                    delta_chunks: RwLock::new(None),
                });
                {
                    let mut active = self.active.write().expect("active slot lock poisoned");
                    *active = Some(slot.clone());
                }
                *active_during_build = Some(slot);
            }
            // Subsequent batches — chunks reader sees new entries
            // through the shared index, dense graph grows in place,
            // only sparse needs a fresh reader to see the latest
            // committed tantivy segments.
            Some(slot) => {
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
            let Some(slot_id) = name.to_str() else {
                continue;
            };
            let manifest_path = slot::manifest_path(&entry.path());
            let Ok(text) = fs::read_to_string(&manifest_path) else {
                continue;
            };
            let Ok(manifest) = SlotManifest::from_toml_str(&text) else {
                continue;
            };
            if matches!(manifest.state, SlotState::Planning | SlotState::Building) {
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
        self.active_snapshot()
            .map(|s| s.manifest.embedder.dimension)
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
                        // Coarse stage string driven by the manifest
                        // state. Fine BuildPhase progresses via
                        // wire-side BucketBuildProgress events; this
                        // status() result is just "yes, queries hit a
                        // slot that's still being built."
                        stage: match slot.manifest.state {
                            SlotState::Planning => "planning".to_string(),
                            _ => "indexing".to_string(),
                        },
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
            let tombstones = active.tombstones();
            let mut out = Vec::with_capacity(hits.len());
            for (chunk_id, distance) in hits {
                // Drop tombstoned hits before fetching chunk text;
                // physical removal happens at compaction.
                if tombstones.contains(chunk_id) {
                    continue;
                }
                // Resolve chunk text from base, falling back to delta.
                // A miss can happen during a mid-build query if HNSW
                // already has a position whose chunk_id wasn't yet
                // committed to the chunk-store snapshot — skip it
                // rather than erroring; the next query after the
                // missing chunk's batch will see consistent state.
                let Some(chunk) = active.fetch_chunk_any(chunk_id).map_err(BucketError::Io)? else {
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
            let tombstones = active.tombstones();
            let mut out = Vec::with_capacity(hits.len());
            for (chunk_id, score) in hits {
                // Tombstoned chunks may still be in tantivy until the
                // delete-by-term path lands with the insert commit;
                // until then the query-side filter is the source of
                // truth.
                if tombstones.contains(chunk_id) {
                    continue;
                }
                // Same skip-on-miss as the dense path: a tantivy
                // commit may have outpaced the chunks-store snapshot
                // by one batch. Resolve from base, fall back to delta.
                let Some(chunk) = active.fetch_chunk_any(chunk_id).map_err(BucketError::Io)? else {
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
            match active.fetch_chunk_any(chunk_id).map_err(BucketError::Io)? {
                Some(chunk) => Ok(chunk),
                None => Err(BucketError::Other(format!(
                    "chunk {chunk_id} not in active slot",
                ))),
            }
        })
    }

    fn insert<'a>(
        &'a self,
        new_chunks: Vec<NewChunk>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<ChunkId>, BucketError>> {
        Box::pin(async move {
            if new_chunks.is_empty() {
                return Ok(Vec::new());
            }
            let active = self
                .active_snapshot()
                .ok_or_else(|| BucketError::NoActiveSlot(self.id.clone()))?;
            let embedder = self.embedder().ok_or_else(|| {
                BucketError::Other(
                    "insert requires an embedder; call set_embedder first".to_string(),
                )
            })?;

            // Single-writer guard for the whole insert (embed + delta
            // writes + HNSW + tantivy). Concurrent inserts queue here;
            // queries are unaffected (they read the active snapshot).
            let _guard = self.mutation_lock.lock().await;

            // 1. Derive content-addressed chunk ids up front so we can
            //    return them even if the embedder fails partway.
            let chunk_ids: Vec<ChunkId> = new_chunks
                .iter()
                .map(|c| ChunkId::from_source(&c.source_record_hash, c.chunk_offset))
                .collect();

            // 2. Embed (network IO).
            let texts: Vec<String> = new_chunks.iter().map(|c| c.text.clone()).collect();
            let req = EmbedRequest {
                model: "",
                inputs: &texts,
            };
            let resp = embedder
                .embed(&req, cancel)
                .await
                .map_err(|e| BucketError::Provider(e.to_string()))?;
            if resp.embeddings.len() != new_chunks.len() {
                return Err(BucketError::Provider(format!(
                    "embedder returned {} vectors for {} inputs",
                    resp.embeddings.len(),
                    new_chunks.len(),
                )));
            }
            let dim = active.manifest.embedder.dimension;
            for v in &resp.embeddings {
                if v.len() as u32 != dim {
                    return Err(BucketError::DimensionMismatch {
                        query: v.len() as u32,
                        slot: dim,
                    });
                }
            }

            // 3. File IO + HNSW insert + tantivy on a blocking thread.
            //    These are CPU/IO-bound and would otherwise park a
            //    tokio worker.
            let slot_path = slot::slot_dir(&self.root, &active.manifest.slot_id);
            let load_mode = serving_mode_to_load_mode(active.manifest.serving.mode);
            let sparse_enabled = active.sparse.is_some();
            let dense = active.dense.clone();
            let embeddings = resp.embeddings;
            let chunk_ids_clone = chunk_ids.clone();
            let new_delta_reader = tokio::task::spawn_blocking(
                move || -> Result<Arc<ChunkStoreReader>, BucketError> {
                    persist_insert_batch(
                        &slot_path,
                        &new_chunks,
                        &chunk_ids_clone,
                        &embeddings,
                        &dense,
                        sparse_enabled,
                        load_mode,
                        dim,
                    )
                },
            )
            .await
            .map_err(|e| BucketError::Other(format!("insert task: {e}")))??;

            // 4. Swap the active slot's delta-chunks reader so query-
            //    time fetches resolve the just-inserted ids. The HNSW
            //    is shared with `dense` (Arc cloned in step 3) so the
            //    inserts already landed there during the blocking
            //    work.
            {
                let mut slot_delta = active
                    .delta_chunks
                    .write()
                    .expect("delta_chunks lock poisoned");
                *slot_delta = Some(new_delta_reader);
            }

            // 5. Re-open the sparse reader. Tantivy's
            //    `OnCommitWithDelay` reload policy doesn't pick up
            //    a separate writer's commit synchronously enough for
            //    a follow-up query in the same async span; the build
            //    path solves this by re-opening after every commit
            //    and we mirror that here.
            if let Some(sparse_lock) = active.sparse.as_ref() {
                let tantivy_dir =
                    slot::tantivy_dir(&slot::slot_dir(&self.root, &active.manifest.slot_id));
                let new_sparse = Arc::new(SparseIndex::open(&tantivy_dir)?);
                let mut w = sparse_lock.write().expect("sparse lock poisoned");
                *w = new_sparse;
            }

            Ok(chunk_ids)
        })
    }

    fn tombstone<'a>(&'a self, chunk_ids: Vec<ChunkId>) -> BoxFuture<'a, Result<(), BucketError>> {
        Box::pin(async move {
            if chunk_ids.is_empty() {
                return Ok(());
            }
            let active = self
                .active_snapshot()
                .ok_or_else(|| BucketError::NoActiveSlot(self.id.clone()))?;

            // Tombstone now drives a tantivy writer in addition to the
            // tombstones-file append, so it shares the same single-
            // writer guard that `insert` uses. The query-side tombstone
            // filter still applies — defensive against any in-flight
            // queries that captured an Arc<LoadedSlot> snapshot before
            // the sparse swap below — and is cheap enough to keep.
            let _guard = self.mutation_lock.lock().await;

            let tombstones = active.tombstones();
            let slot_path = slot::slot_dir(&self.root, &active.manifest.slot_id);
            let sparse_enabled = active.sparse.is_some();
            let chunk_ids_for_blocking = chunk_ids.clone();
            let slot_path_for_blocking = slot_path.clone();
            tokio::task::spawn_blocking(move || -> Result<(), BucketError> {
                tombstones
                    .append(&chunk_ids_for_blocking)
                    .map_err(BucketError::Io)?;
                if sparse_enabled {
                    let tantivy_dir = slot::tantivy_dir(&slot_path_for_blocking);
                    let mut sparse = SparseIndexBuilder::open_resume(&tantivy_dir)?;
                    sparse.delete_chunks(&chunk_ids_for_blocking);
                    sparse.finalize()?;
                }
                Ok(())
            })
            .await
            .map_err(|e| BucketError::Other(format!("tombstone task: {e}")))??;

            // Re-open the slot's SparseIndex so a follow-up query
            // sees the deletes (mirrors the post-insert swap).
            if let Some(sparse_lock) = active.sparse.as_ref() {
                let tantivy_dir = slot::tantivy_dir(&slot_path);
                let new_sparse = Arc::new(SparseIndex::open(&tantivy_dir)?);
                let mut w = sparse_lock.write().expect("sparse lock poisoned");
                *w = new_sparse;
            }
            Ok(())
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

/// Blocking-thread helper for `Bucket::insert`. Appends one batch of
/// pre-embedded chunks to the slot's delta layer:
/// - opens the delta chunk + vector writers (creating files if absent,
///   resuming an idx-tracked tail if present),
/// - appends each (chunk, vector) to delta files and inserts the
///   vector into the slot's HNSW with a position chosen to extend
///   `dense.len()`,
/// - finalizes both writers (rewrites the .idx so the next call's
///   create_or_open_append picks up the new state),
/// - opens a tantivy writer over the existing index, adds each chunk,
///   commits, and drops it (releasing the index lock),
/// - returns a freshly-opened delta chunk reader for the caller to
///   swap into `LoadedSlot.delta_chunks`.
///
/// Callers must hold the bucket's `mutation_lock` for the duration
/// (this function does not acquire it).
#[allow(clippy::too_many_arguments)]
fn persist_insert_batch(
    slot_path: &Path,
    new_chunks: &[NewChunk],
    chunk_ids: &[ChunkId],
    embeddings: &[Vec<f32>],
    dense: &Arc<DenseIndex>,
    sparse_enabled: bool,
    load_mode: LoadMode,
    dimension: u32,
) -> Result<Arc<ChunkStoreReader>, BucketError> {
    let chunks_bin = slot::delta_chunks_bin_path(slot_path);
    let chunks_idx = slot::delta_chunks_idx_path(slot_path);
    let vectors_bin = slot::delta_vectors_bin_path(slot_path);
    let vectors_idx = slot::delta_vectors_idx_path(slot_path);

    let mut chunk_writer = ChunkStoreWriter::create_or_open_append(&chunks_bin, &chunks_idx)
        .map_err(BucketError::Io)?;
    let mut vector_writer =
        VectorStoreWriter::create_or_open_append(&vectors_bin, &vectors_idx, dimension)
            .map_err(BucketError::Io)?;

    // Position scheme matches `load_slot`: HNSW positions are an
    // opaque address space — the base build occupies `0..base_count`,
    // delta entries extend from `current_len` upward. `current_len`
    // already accounts for any prior delta appends in this slot's
    // lifetime.
    let base_position = dense.len() as u64;
    for (i, ((chunk, &id), vector)) in new_chunks
        .iter()
        .zip(chunk_ids.iter())
        .zip(embeddings.iter())
        .enumerate()
    {
        chunk_writer
            .append(id, &chunk.source_ref, &chunk.text)
            .map_err(BucketError::Io)?;
        vector_writer.append(id, vector).map_err(BucketError::Io)?;
        dense.insert(id, base_position + i as u64, vector);
    }

    // `finalize` flushes the bin and rewrites the .idx — both bins
    // are durable on return so a crash here doesn't desync the
    // chunk + vector stores.
    chunk_writer.flush().map_err(BucketError::Io)?;
    vector_writer.flush().map_err(BucketError::Io)?;
    chunk_writer.finalize().map_err(BucketError::Io)?;
    vector_writer.finalize().map_err(BucketError::Io)?;

    // Tantivy: open writer over existing index, add docs, commit.
    // The slot's existing `SparseIndex` reader uses
    // `OnCommitWithDelay` and picks up the new docs automatically.
    if sparse_enabled {
        let tantivy_dir = slot::tantivy_dir(slot_path);
        let mut sparse = SparseIndexBuilder::open_resume(&tantivy_dir)?;
        for (chunk, &id) in new_chunks.iter().zip(chunk_ids.iter()) {
            sparse.add(id, &chunk.text)?;
        }
        sparse.finalize()?;
    }

    // Re-open the delta chunks reader so the caller's swap surfaces
    // the just-finalized records to subsequent queries.
    let reader = ChunkStoreReader::open_with_mode(&chunks_bin, &chunks_idx, load_mode)
        .map_err(BucketError::Io)?;
    Ok(Arc::new(reader))
}

/// Construct fresh writers + bundle into [`BuildHandles`] for the
/// fresh-build code paths (initial `build_slot` + resume from
/// `Planning` state). The slot dir already exists; this only opens
/// the per-index writers.
fn open_fresh_handles(
    slot_path: &Path,
    manifest: SlotManifest,
    dimension: u32,
    sparse_enabled: bool,
    build_state: BuildStateWriter,
) -> Result<BuildHandles, BucketError> {
    let chunks_bin = slot::chunks_bin_path(slot_path);
    let chunks_idx = slot::chunks_idx_path(slot_path);
    let vectors_bin = slot::vectors_bin_path(slot_path);
    let vectors_idx = slot::vectors_idx_path(slot_path);
    let tantivy_dir = slot::tantivy_dir(slot_path);
    let chunk_writer =
        ChunkStoreWriter::create(&chunks_bin, &chunks_idx).map_err(BucketError::Io)?;
    let vector_writer = VectorStoreWriter::create(&vectors_bin, &vectors_idx, dimension)
        .map_err(BucketError::Io)?;
    let sparse_builder = if sparse_enabled {
        Some(SparseIndexBuilder::create(&tantivy_dir)?)
    } else {
        None
    };
    Ok(BuildHandles {
        manifest,
        chunk_writer,
        vector_writer,
        sparse_builder,
        build_state,
    })
}

/// Dump the in-progress (or just-finished) HNSW + sidecar on a
/// `spawn_blocking` thread. Failures are warn-logged rather than
/// returning an error — a missing dump means resume falls back to a
/// sequential rebuild from vectors.bin, but doesn't lose any data.
///
/// Used both for periodic mid-build dumps (every
/// [`DENSE_DUMP_BATCH_INTERVAL`] batches) and the end-of-build dump
/// in the `BuildingDense` phase.
async fn periodic_dump(
    slot_path: &Path,
    slot_id: &str,
    dense: &Arc<DenseIndex>,
) -> Result<(), BucketError> {
    let dense_for_dump = dense.clone();
    let slot_path_owned = slot_path.to_path_buf();
    let slot_id_for_warn = slot_id.to_string();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = dense_for_dump.dump_to(&slot_path_owned) {
            tracing::warn!(
                slot = %slot_id_for_warn,
                error = %e,
                "hnsw dump failed; slot is still usable but resume will rebuild from vectors",
            );
        }
    })
    .await
    .map_err(|e| BucketError::Other(format!("spawn_blocking dense dump: {e}")))
}

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

    // Translate the manifest's serving mode into the storage layer's
    // load mode. `Ram` → eager-load whole files; `Disk` → mmap. Both
    // expose the same `&[u8]` view of the file for fetches.
    let load_mode = serving_mode_to_load_mode(manifest.serving.mode);

    let chunks_bin = slot::chunks_bin_path(&slot_path);
    let chunks_idx = slot::chunks_idx_path(&slot_path);
    let chunks = ChunkStoreReader::open_with_mode(&chunks_bin, &chunks_idx, load_mode)
        .map_err(BucketError::Io)?;

    let vectors_bin = slot::vectors_bin_path(&slot_path);
    let vectors_idx = slot::vectors_idx_path(&slot_path);
    let vectors = VectorStoreReader::open_with_mode(&vectors_bin, &vectors_idx, load_mode)
        .map_err(BucketError::Io)?;

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

    let tombstones =
        Arc::new(Tombstones::open(&slot::tombstones_path(&slot_path)).map_err(BucketError::Io)?);

    // Stitch the delta layer in: if `delta_vectors.bin` exists, seed
    // its rows into the just-loaded HNSW (positions chosen to extend
    // the base address space — the HNSW position is just an opaque
    // unique key, so `base_count + delta_local_pos` works), and open
    // the matching delta chunks reader so query-time fetches can
    // resolve delta chunk ids.
    //
    // The base HNSW dump is intentionally not refreshed on every
    // insert (would be `O(GB)` per call at scale); reload pays the
    // delta-seed cost instead. Compaction collapses delta back into a
    // fresh base + dump.
    let dense = Arc::new(dense);
    let delta_vectors_bin = slot::delta_vectors_bin_path(&slot_path);
    let delta_vectors_idx = slot::delta_vectors_idx_path(&slot_path);
    let delta_chunks_bin = slot::delta_chunks_bin_path(&slot_path);
    let delta_chunks_idx = slot::delta_chunks_idx_path(&slot_path);

    let delta_chunks = if delta_chunks_bin.exists() && delta_chunks_idx.exists() {
        Some(Arc::new(
            ChunkStoreReader::open_with_mode(&delta_chunks_bin, &delta_chunks_idx, load_mode)
                .map_err(BucketError::Io)?,
        ))
    } else {
        None
    };

    if delta_vectors_bin.exists() && delta_vectors_idx.exists() {
        let delta_vectors =
            VectorStoreReader::open_with_mode(&delta_vectors_bin, &delta_vectors_idx, load_mode)
                .map_err(BucketError::Io)?;
        if delta_vectors.dimension() != manifest.embedder.dimension {
            return Err(BucketError::Config(format!(
                "delta_vectors.bin dimension {} does not match manifest embedder dimension {}",
                delta_vectors.dimension(),
                manifest.embedder.dimension,
            )));
        }
        let base_count = dense.len() as u64;
        let mut pairs: Vec<(u64, ChunkId)> = delta_vectors
            .iter_chunk_positions()
            .map(|(id, pos)| (pos, id))
            .collect();
        pairs.sort_by_key(|(pos, _)| *pos);
        for (delta_pos, chunk_id) in pairs {
            let vector = delta_vectors
                .read_at_position(delta_pos)
                .map_err(BucketError::Io)?;
            dense.insert(chunk_id, base_count + delta_pos, &vector);
        }
    }

    Ok(LoadedSlot {
        manifest,
        chunks: RwLock::new(Arc::new(chunks)),
        vectors: Some(vectors),
        dense,
        sparse: sparse.map(|s| RwLock::new(Arc::new(s))),
        tombstones,
        delta_chunks: RwLock::new(delta_chunks),
    })
}

/// Translate the slot manifest's [`ServingMode`] into the storage
/// layer's [`LoadMode`]. Single source of truth for the dispatch so
/// `build_slot` and `load_slot` agree on what each mode means.
///
/// `ServingMode::Disk` → `LoadMode::Mmap` (lazy paging via OS cache).
/// `ServingMode::Ram` → `LoadMode::Eager` (whole-file `Box<[u8]>`).
fn serving_mode_to_load_mode(mode: ServingMode) -> LoadMode {
    match mode {
        ServingMode::Disk => LoadMode::Mmap,
        ServingMode::Ram => LoadMode::Eager,
    }
}

fn write_manifest(slot_path: &Path, manifest: &SlotManifest) -> Result<(), BucketError> {
    let toml = manifest.to_toml_string()?;
    let path = slot::manifest_path(slot_path);
    fs::write(&path, toml).map_err(BucketError::Io)?;
    Ok(())
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

    /// Variant of [`sample_bucket_toml`] that pins
    /// `[defaults] serving_mode = "disk"` so the build path's slot
    /// manifest records `ServingMode::Disk` and the readers come up in
    /// mmap mode.
    fn sample_bucket_toml_with_serving_mode(source_path: &Path, mode: &str) -> String {
        format!(
            r#"
name = "Test Notes"
created_at = "2026-04-25T10:23:11Z"

[source]
kind = "linked"
adapter = "markdown_dir"
path = "{path}"

[chunker]
strategy = "token_based"
chunk_tokens = 50
overlap_tokens = 5

[defaults]
embedder = "tei_test"
serving_mode = "{mode}"
"#,
            path = source_path.display(),
            mode = mode,
        )
    }

    fn open_test_bucket_with_serving_mode(
        bucket_root: &Path,
        source_path: &Path,
        mode: &str,
    ) -> DiskBucket {
        fs::create_dir_all(bucket_root).unwrap();
        fs::create_dir_all(slot::slots_dir(bucket_root)).unwrap();
        fs::write(
            slot::bucket_toml_path(bucket_root),
            sample_bucket_toml_with_serving_mode(source_path, mode),
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(16));
        let cancel = CancellationToken::new();

        let slot_id = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(32));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
            let (chunker, chunker_snapshot) =
                TokenBasedChunker::from_config(&bucket.config().chunker);
            let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(12));
            let cancel = CancellationToken::new();
            bucket
                .build_slot(
                    &adapter,
                    &chunker,
                    chunker_snapshot.clone(),
                    embedder,
                    None,
                    &cancel,
                )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(16));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
            let (chunker, chunker_snapshot) =
                TokenBasedChunker::from_config(&bucket.config().chunker);
            let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
            let cancel = CancellationToken::new();
            bucket
                .build_slot(
                    &adapter,
                    &chunker,
                    chunker_snapshot.clone(),
                    embedder,
                    None,
                    &cancel,
                )
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
    async fn compact_still_errors_until_indexes_land() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let cancel = CancellationToken::new();
        assert!(bucket.compact(&cancel).await.is_err());
    }

    #[tokio::test]
    async fn insert_empty_is_ok_even_without_slot_or_embedder() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let cancel = CancellationToken::new();
        let ids = bucket.insert(Vec::new(), &cancel).await.unwrap();
        assert!(ids.is_empty());
    }

    #[tokio::test]
    async fn insert_errors_when_no_active_slot() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        bucket.set_embedder(Arc::new(MockEmbedder::new(8)));
        let cancel = CancellationToken::new();
        let err = bucket
            .insert(vec![sample_new_chunk("hello", [1u8; 32], 0)], &cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, BucketError::NoActiveSlot(_)), "{err:?}");
    }

    #[tokio::test]
    async fn insert_errors_when_no_embedder_set() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hello");
        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot,
                embedder,
                None,
                &cancel,
            )
            .await
            .unwrap();
        // Notice: no set_embedder call after build.
        let err = bucket
            .insert(vec![sample_new_chunk("hello", [1u8; 32], 0)], &cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, BucketError::Other(_)), "{err:?}");
    }

    #[tokio::test]
    async fn insert_appears_in_sparse_and_dense_search() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "alpha.md", "alpha body unrelated");
        write_md(&source, "beta.md", "beta body unrelated");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot,
                embedder.clone(),
                None,
                &cancel,
            )
            .await
            .unwrap();
        bucket.set_embedder(embedder);

        let inserted = "fascinating brand new fact about quokkas";
        let ids = bucket
            .insert(vec![sample_new_chunk(inserted, [0xAB; 32], 0)], &cancel)
            .await
            .unwrap();
        assert_eq!(ids.len(), 1);
        let inserted_id = ids[0];

        // Sparse: deterministic — a unique word ("quokkas") is in the
        // newly inserted chunk and nowhere else.
        let hits = bucket.sparse_search("quokkas", 5, &cancel).await.unwrap();
        assert_eq!(hits.len(), 1, "expected the inserted chunk");
        assert_eq!(hits[0].chunk_id, inserted_id);
        assert!(hits[0].chunk_text.contains(inserted));

        // Dense: query with the exact vector that was inserted; the
        // chunk should be the closest hit (distance ~0).
        let query = MockEmbedder::fake_vector(inserted, 8);
        let dense = bucket.dense_search(&query, 3, &cancel).await.unwrap();
        assert!(
            dense.iter().any(|c| c.chunk_id == inserted_id),
            "inserted chunk did not surface in dense search: {dense:?}",
        );
    }

    #[tokio::test]
    async fn insert_persists_across_bucket_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "the only base chunk");

        let bucket_root = tmp.path().join("bucket");
        let inserted_id = {
            let bucket = open_test_bucket(&bucket_root, &source);
            let adapter = MarkdownDir::new(&source);
            let (chunker, chunker_snapshot) =
                TokenBasedChunker::from_config(&bucket.config().chunker);
            let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
            let cancel = CancellationToken::new();
            bucket
                .build_slot(
                    &adapter,
                    &chunker,
                    chunker_snapshot,
                    embedder.clone(),
                    None,
                    &cancel,
                )
                .await
                .unwrap();
            bucket.set_embedder(embedder);
            let ids = bucket
                .insert(
                    vec![sample_new_chunk(
                        "rare token kangaroo-platypus",
                        [0xCD; 32],
                        0,
                    )],
                    &cancel,
                )
                .await
                .unwrap();
            ids[0]
        };

        // Re-open: delta files on disk are stitched into the loaded
        // HNSW + the delta chunks reader, so queries still find the
        // inserted chunk.
        let reopened = DiskBucket::open(&bucket_root, BucketId::pod("notes")).unwrap();
        let cancel = CancellationToken::new();
        let hits = reopened
            .sparse_search("kangaroo-platypus", 5, &cancel)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, inserted_id);

        let query = MockEmbedder::fake_vector("rare token kangaroo-platypus", 8);
        let dense = reopened.dense_search(&query, 3, &cancel).await.unwrap();
        assert!(
            dense.iter().any(|c| c.chunk_id == inserted_id),
            "inserted chunk did not survive reopen in dense search",
        );
    }

    fn sample_new_chunk(text: &str, hash: [u8; 32], offset: u64) -> NewChunk {
        NewChunk {
            text: text.to_string(),
            source_ref: crate::knowledge::SourceRef {
                source_id: "test-source".to_string(),
                locator: None,
            },
            source_record_hash: hash,
            chunk_offset: offset,
        }
    }

    #[tokio::test]
    async fn tombstone_empty_is_ok_even_without_slot() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        // Empty list short-circuits before the active-slot check;
        // there's nothing to durably record.
        bucket.tombstone(Vec::new()).await.unwrap();
    }

    #[tokio::test]
    async fn tombstone_errors_when_no_active_slot() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let err = bucket
            .tombstone(vec![ChunkId([0xAA; 32])])
            .await
            .unwrap_err();
        assert!(matches!(err, BucketError::NoActiveSlot(_)), "{err:?}");
    }

    #[tokio::test]
    async fn tombstone_filters_dense_search_results() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "alpha.md", "alpha body");
        write_md(&source, "beta.md", "beta body");
        write_md(&source, "gamma.md", "gamma body");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(16));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
            .await
            .unwrap();

        // Pull a real chunk id out of a search to avoid duplicating the
        // chunker's id derivation in the test.
        let beta_record = crate::knowledge::SourceRecord::new("ignored", "beta body");
        let beta_chunks = chunker.chunk(&beta_record);
        let query = MockEmbedder::fake_vector(&beta_chunks[0].text, 16);

        let before = bucket.dense_search(&query, 3, &cancel).await.unwrap();
        let target = before
            .iter()
            .find(|c| c.chunk_text.contains("beta body"))
            .expect("expected beta hit before tombstone")
            .chunk_id;

        bucket.tombstone(vec![target]).await.unwrap();

        let after = bucket.dense_search(&query, 3, &cancel).await.unwrap();
        assert!(
            after.iter().all(|c| c.chunk_id != target),
            "tombstoned chunk leaked into dense results: {after:?}",
        );
        assert!(
            after.iter().all(|c| !c.chunk_text.contains("beta body")),
            "tombstoned text leaked into dense results: {after:?}",
        );
    }

    #[tokio::test]
    async fn tombstone_filters_sparse_search_results() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(
            &source,
            "alpha.md",
            "the quick brown fox jumps over the lazy dog",
        );
        write_md(&source, "beta.md", "another fox in another sentence");
        write_md(
            &source,
            "gamma.md",
            "the rain in spain falls mainly on the plain",
        );

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
            .await
            .unwrap();

        let before = bucket.sparse_search("fox", 5, &cancel).await.unwrap();
        assert_eq!(before.len(), 2, "expected both fox docs before tombstone");
        let target = before[0].chunk_id;

        bucket.tombstone(vec![target]).await.unwrap();

        let after = bucket.sparse_search("fox", 5, &cancel).await.unwrap();
        assert!(
            after.iter().all(|c| c.chunk_id != target),
            "tombstoned chunk leaked into sparse results",
        );
        assert_eq!(
            after.len(),
            before.len() - 1,
            "exactly one chunk should drop out",
        );
    }

    #[tokio::test]
    async fn tombstone_persists_across_bucket_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "alpha.md", "the quick brown fox");
        write_md(&source, "beta.md", "another fox sentence");

        let bucket_root = tmp.path().join("bucket");
        let target = {
            let bucket = open_test_bucket(&bucket_root, &source);
            let adapter = MarkdownDir::new(&source);
            let (chunker, chunker_snapshot) =
                TokenBasedChunker::from_config(&bucket.config().chunker);
            let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
            let cancel = CancellationToken::new();
            bucket
                .build_slot(
                    &adapter,
                    &chunker,
                    chunker_snapshot.clone(),
                    embedder,
                    None,
                    &cancel,
                )
                .await
                .unwrap();
            let hits = bucket.sparse_search("fox", 5, &cancel).await.unwrap();
            assert_eq!(hits.len(), 2);
            let target = hits[0].chunk_id;
            bucket.tombstone(vec![target]).await.unwrap();
            target
        };

        // Reopen — tombstones come back from `tombstones.bin`.
        let reopened = DiskBucket::open(&bucket_root, BucketId::pod("notes")).unwrap();
        let cancel = CancellationToken::new();
        let after = reopened.sparse_search("fox", 5, &cancel).await.unwrap();
        assert!(
            after.iter().all(|c| c.chunk_id != target),
            "tombstoned chunk reappeared after reopen",
        );
        assert_eq!(after.len(), 1);
    }

    #[tokio::test]
    async fn tombstone_drops_doc_from_tantivy_index() {
        // Verify the tantivy-side delete actually fires, not just the
        // query-side filter. We open the on-disk SparseIndex directly
        // and check `num_docs` before and after.
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "alpha.md", "alpha solitary phrase");
        write_md(&source, "beta.md", "beta solitary phrase");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot,
                embedder,
                None,
                &cancel,
            )
            .await
            .unwrap();

        let slot_id = bucket.active_slot_id().unwrap();
        let tantivy_dir = slot::tantivy_dir(&slot::slot_dir(&bucket_root, &slot_id));
        let before = SparseIndex::open(&tantivy_dir).unwrap().len();
        assert_eq!(before, 2);

        let hits = bucket.sparse_search("alpha", 5, &cancel).await.unwrap();
        assert_eq!(hits.len(), 1);
        bucket.tombstone(vec![hits[0].chunk_id]).await.unwrap();

        // Re-open the on-disk index from scratch — bypasses both the
        // bucket's reader handle and the query-side tombstone filter.
        let after = SparseIndex::open(&tantivy_dir).unwrap().len();
        assert_eq!(after, 1, "tantivy did not delete the tombstoned doc");
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let cancel = CancellationToken::new();

        bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        let slot_id = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(4));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let _ = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);

        // Cancel after exactly 1 batch — leaves ~128 records embedded
        // and ~158 records still to do on resume.
        let cancel = CancellationToken::new();
        let cancelling = Arc::new(CancellingEmbedder::new(8, 1, cancel.clone()));
        let _ = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                cancelling.clone(),
                None,
                &cancel,
            )
            .await
            .unwrap_err();

        // Find the partial slot and verify it's resumable.
        let resumable = bucket
            .find_resumable_slot()
            .unwrap()
            .expect("find_resumable_slot must return the in-progress slot after cancel");

        let slot_path = slot::slot_dir(&bucket_root, &resumable);
        let manifest = SlotManifest::from_toml_str(
            &fs::read_to_string(slot::manifest_path(&slot_path)).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.state, SlotState::Building);

        let log_pre = build_state::read_all(&slot_path.join(slot::BUILD_STATE)).unwrap();
        let pre_batches = log_pre
            .iter()
            .filter(|r| matches!(r, BuildStateRecord::BatchEmbedded { .. }))
            .count();
        assert!(
            pre_batches >= 1,
            "at least one batch should be checkpointed"
        );
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
                chunker_snapshot.clone(),
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
        let manifest = SlotManifest::from_toml_str(
            &fs::read_to_string(slot::manifest_path(&slot_path)).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.state, SlotState::Ready);
        assert_eq!(manifest.stats.source_records, total_records as u64);
        assert_eq!(manifest.stats.chunk_count, total_records as u64);
        assert_eq!(manifest.stats.vector_count, total_records as u64);

        let log_post = build_state::read_all(&slot_path.join(slot::BUILD_STATE)).unwrap();
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
        assert!(
            !dense_hits.is_empty(),
            "resumed slot must be queryable on dense"
        );
        let sparse_hits = bucket
            .sparse_search("doc-0050", 5, &cancel_q)
            .await
            .unwrap();
        assert!(
            !sparse_hits.is_empty(),
            "resumed slot must be queryable on sparse"
        );
    }

    #[tokio::test]
    async fn resume_loads_periodic_dump_when_sidecar_matches_checkpoint() {
        // The fast resume path: if dense.snapshot's chunk count matches
        // the last BatchEmbedded's chunks_completed, resume should load
        // the dump rather than rebuild the HNSW from vectors.bin. With
        // DENSE_DUMP_BATCH_INTERVAL=1 in tests, every BatchEmbedded
        // fires a dump, so the sidecar always matches the latest log
        // entry on a clean cancel.
        use crate::knowledge::build_state::{self, BuildStateRecord};
        use crate::knowledge::dense::read_dump_snapshot;

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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);

        let cancel = CancellationToken::new();
        let cancelling = Arc::new(CancellingEmbedder::new(8, 1, cancel.clone()));
        let _ = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                cancelling,
                None,
                &cancel,
            )
            .await
            .unwrap_err();

        let resumable = bucket.find_resumable_slot().unwrap().unwrap();
        let slot_path = slot::slot_dir(&bucket_root, &resumable);

        // Sidecar must exist after a single BatchEmbedded and match
        // the checkpoint chunk count.
        let snap = read_dump_snapshot(&slot_path)
            .unwrap()
            .expect("dense.snapshot must exist after a periodic dump");
        let log = build_state::read_all(&slot_path.join(slot::BUILD_STATE)).unwrap();
        let last_batch_chunks = log
            .iter()
            .rev()
            .find_map(|r| match r {
                BuildStateRecord::BatchEmbedded {
                    chunks_completed, ..
                } => Some(*chunks_completed),
                _ => None,
            })
            .expect("at least one BatchEmbedded must be logged");
        assert_eq!(
            snap, last_batch_chunks,
            "sidecar count must match last BatchEmbedded checkpoint",
        );

        // Resume picks the fast path. End-to-end success — same as the
        // sequential-rebuild test, but proves the dump path is also
        // exercised correctly.
        let resume_cancel = CancellationToken::new();
        let plain: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let resumed = bucket
            .resume_slot(
                &resumable,
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                plain,
                None,
                &resume_cancel,
            )
            .await
            .unwrap();
        assert_eq!(resumed, resumable);

        let manifest = SlotManifest::from_toml_str(
            &fs::read_to_string(slot::manifest_path(&slot_path)).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.state, SlotState::Ready);
        assert_eq!(manifest.stats.chunk_count, total_records as u64);

        // Dump still in place after final BuildingDense dump; sidecar
        // must now reflect the full count.
        let final_snap = read_dump_snapshot(&slot_path).unwrap().unwrap();
        assert_eq!(final_snap, total_records as u64);
    }

    #[tokio::test]
    async fn resume_falls_back_to_rebuild_when_sidecar_missing() {
        use crate::knowledge::dense::read_dump_snapshot;
        // Inverse of the fast-path test: if dense.snapshot is absent
        // (simulating a build that crashed before any periodic dump
        // landed, or a corrupted partial dump that left no sidecar),
        // resume must still complete by rebuilding the HNSW from
        // vectors.bin sequentially.
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);

        let cancel = CancellationToken::new();
        let cancelling = Arc::new(CancellingEmbedder::new(8, 1, cancel.clone()));
        let _ = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                cancelling,
                None,
                &cancel,
            )
            .await
            .unwrap_err();

        let resumable = bucket.find_resumable_slot().unwrap().unwrap();
        let slot_path = slot::slot_dir(&bucket_root, &resumable);

        // Force the fallback path by deleting the sidecar (and the
        // hnsw files for good measure — without the sidecar the load
        // wouldn't be attempted anyway, but this also catches a
        // regression where the resume code accidentally calls
        // load_from_with_positions on stale graph files).
        for f in ["dense.snapshot", "dense.hnsw.graph", "dense.hnsw.data"] {
            let p = slot_path.join(f);
            if p.exists() {
                fs::remove_file(&p).unwrap();
            }
        }
        assert!(
            read_dump_snapshot(&slot_path).unwrap().is_none(),
            "sidecar must be cleared",
        );

        let resume_cancel = CancellationToken::new();
        let plain: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let resumed = bucket
            .resume_slot(
                &resumable,
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                plain,
                None,
                &resume_cancel,
            )
            .await
            .unwrap();
        assert_eq!(resumed, resumable);

        let manifest = SlotManifest::from_toml_str(
            &fs::read_to_string(slot::manifest_path(&slot_path)).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.state, SlotState::Ready);
        assert_eq!(manifest.stats.chunk_count, total_records as u64);
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);

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
            chunker_snapshot: chunker_snapshot.clone(),
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
            .resume_slot(
                &slot_id,
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);

        let cancel = CancellationToken::new();
        let cancelling = Arc::new(CancellingEmbedder::new(8, 1, cancel.clone()));
        let _ = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                cancelling,
                None,
                &cancel,
            )
            .await
            .unwrap_err();
        let resumable = bucket.find_resumable_slot().unwrap().unwrap();

        // Mutate one of the source records that should already have
        // been Planned (record 0) so resume sees content drift.
        write_md(&source, "doc-0000.md", "DRIFTED CONTENT");

        let resume_cancel = CancellationToken::new();
        let plain: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let err = bucket
            .resume_slot(
                &resumable,
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                plain,
                None,
                &resume_cancel,
            )
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("drift") || msg.contains("Drift"),
            "expected drift error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn resume_rejects_embedder_identity_change() {
        // Same dimension, different model_id ⇒ silently mixed
        // vectors. Resume must error rather than continue extending
        // a slot whose existing vectors came from a different model.
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);

        let cancel = CancellationToken::new();
        let cancelling = Arc::new(CancellingEmbedder::new(8, 1, cancel.clone()));
        let _ = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                cancelling,
                None,
                &cancel,
            )
            .await
            .unwrap_err();
        let resumable = bucket.find_resumable_slot().unwrap().unwrap();

        // Same dim (8), but a fresh model id.
        let different_model = Arc::new(MockEmbedder {
            dimension: 8,
            model_id: "different-mock".to_string(),
            embed_calls: AtomicUsize::new(0),
        });
        let resume_cancel = CancellationToken::new();
        let err = bucket
            .resume_slot(
                &resumable,
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                different_model,
                None,
                &resume_cancel,
            )
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("embedder identity changed"),
            "expected identity error, got: {msg}"
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
        let (chunker_for_query, _) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let pause = Arc::new(PauseAfterFirstBatch::new());
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));

        let bucket_for_task = bucket.clone();
        let pause_for_task = pause.clone();
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let source_for_task = source.clone();
        let build_task = tokio::spawn(async move {
            let adapter = MarkdownDir::new(&source_for_task);
            let (chunker, chunker_snapshot) =
                TokenBasedChunker::from_config(&bucket_for_task.config().chunker);
            let observer: &dyn BuildObserver = &*pause_for_task;
            bucket_for_task
                .build_slot(
                    &adapter,
                    &chunker,
                    chunker_snapshot,
                    embedder,
                    Some(observer),
                    &cancel_for_task,
                )
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
        let q_record2 = crate::knowledge::SourceRecord::new("ignored", "doc-0131 body content");
        let q_chunks2 = chunker_for_query.chunk(&q_record2);
        let query_vec2 = MockEmbedder::fake_vector(&q_chunks2[0].text, 8);
        let dense_hits_post = bucket
            .dense_search(&query_vec2, 5, &cancel_q)
            .await
            .unwrap();
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
        let (chunker, chunker_snapshot) = TokenBasedChunker::from_config(&bucket.config().chunker);
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
        let cancel = CancellationToken::new();
        let slot_id = bucket
            .build_slot(
                &adapter,
                &chunker,
                chunker_snapshot.clone(),
                embedder,
                None,
                &cancel,
            )
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

    /// `[defaults] serving_mode` flows from `bucket.toml` into the slot
    /// manifest, and a slot built with either `"ram"` or `"disk"`
    /// returns matching dense + chunk results. This is the end-to-end
    /// integration of the storage-layer `LoadMode` dispatch.
    #[tokio::test]
    async fn serving_mode_round_trips_for_both_ram_and_disk_modes() {
        for mode in ["ram", "disk"] {
            let tmp = tempfile::tempdir().unwrap();
            let bucket_root = tmp.path().join("buckets/notes");
            let source = tmp.path().join("source");
            fs::create_dir_all(&source).unwrap();
            for i in 0..10 {
                write_md(
                    &source,
                    &format!("doc-{i:04}.md"),
                    &format!("doc-{i:04} body content"),
                );
            }

            let bucket = open_test_bucket_with_serving_mode(&bucket_root, &source, mode);
            let adapter = MarkdownDir::new(&source);
            let (chunker, chunker_snapshot) =
                TokenBasedChunker::from_config(&bucket.config().chunker);
            let embedder: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbedder::new(8));
            let cancel = CancellationToken::new();

            let slot_id = bucket
                .build_slot(
                    &adapter,
                    &chunker,
                    chunker_snapshot,
                    embedder,
                    None,
                    &cancel,
                )
                .await
                .unwrap();
            assert_eq!(bucket.active_slot_id().as_deref(), Some(slot_id.as_str()));

            // Manifest must record the requested serving mode.
            let manifest_path = slot::manifest_path(&slot::slot_dir(&bucket_root, &slot_id));
            let manifest =
                SlotManifest::from_toml_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
            let expected = match mode {
                "ram" => crate::knowledge::ServingMode::Ram,
                "disk" => crate::knowledge::ServingMode::Disk,
                _ => unreachable!(),
            };
            assert_eq!(manifest.serving.mode, expected, "mode={mode}");

            // Sparse query path works regardless of dense/vector storage.
            let cancel_q = CancellationToken::new();
            let hits = bucket
                .sparse_search("doc-0003", 5, &cancel_q)
                .await
                .unwrap();
            assert!(!hits.is_empty(), "mode={mode} produced no sparse hits");
        }
    }
}
