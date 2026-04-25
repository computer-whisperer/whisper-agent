//! [`DiskBucket`] — disk-backed implementation of the [`Bucket`] trait.
//!
//! Owns a bucket directory on disk plus the loaded state for the active
//! slot (if any). Slice 3 supports the create/open + build + fetch path
//! end-to-end. Search and mutation methods are stubbed pending embedding
//! (slice 4), HNSW (slice 5), and tantivy (slice 6).
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
use super::manifest::{EmbedderSnapshot, ServingSnapshot, SlotManifest, SlotState, SlotStats};
use super::slot;
use super::source::SourceAdapter;
use super::types::{
    BucketError, BucketId, BucketStatus, Candidate, Chunk, ChunkId, NewChunk, SlotId,
};

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
/// active slot without going back to disk for index metadata. Currently
/// just the manifest + chunks reader; future slices add the dense and
/// sparse index handles here.
#[derive(Debug)]
struct LoadedSlot {
    manifest: SlotManifest,
    chunks: ChunkStoreReader,
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

    /// Build a new slot from a source adapter + chunker. Synchronous —
    /// runs on the calling thread. The slot is created, populated with
    /// chunks, and promoted to active in one call. Returns the new
    /// slot id.
    ///
    /// `embedder_snapshot` is recorded into the manifest; for slice 3
    /// no actual embedding happens, so callers pass a snapshot of the
    /// embedder they *intend* to use (the value is only validated when
    /// the dense path comes online in slice 4).
    pub fn build_slot(
        &self,
        adapter: &dyn SourceAdapter,
        chunker: &dyn Chunker,
        embedder_snapshot: EmbedderSnapshot,
        cancel: &CancellationToken,
    ) -> Result<SlotId, BucketError> {
        let slot_id = slot::generate_slot_id();
        let slot_path = slot::slot_dir(&self.root, &slot_id);
        fs::create_dir_all(&slot_path).map_err(BucketError::Io)?;

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
            embedder: embedder_snapshot,
            sparse: None, // sparse path comes online in slice 6
            serving: ServingSnapshot {
                mode: self.config.defaults.serving_mode,
                quantization: self.config.defaults.quantization,
            },
            stats: SlotStats::default(),
            lineage: None,
        };
        write_manifest(&slot_path, &manifest)?;

        // Build the chunk store. We tolerate per-record adapter errors
        // (log + continue) but propagate writer errors (those are
        // structural).
        let bin_path = slot::chunks_bin_path(&slot_path);
        let idx_path = slot::chunks_idx_path(&slot_path);
        let mut writer = ChunkStoreWriter::create(&bin_path, &idx_path).map_err(BucketError::Io)?;

        let mut source_records: u64 = 0;
        for record in adapter.enumerate() {
            if cancel.is_cancelled() {
                return Err(BucketError::Cancelled);
            }
            let record = match record {
                Ok(r) => r,
                Err(e) => {
                    // For slice 3, surface adapter errors as structural
                    // failures. A future slice that wants partial-
                    // recovery will route per-record errors to build.log
                    // and continue — but we shouldn't silently drop
                    // records before that path exists.
                    return Err(BucketError::Other(format!("source error: {e}")));
                }
            };
            source_records += 1;

            for new_chunk in chunker.chunk(&record) {
                let id =
                    ChunkId::from_source(&new_chunk.source_record_hash, new_chunk.chunk_offset);
                writer
                    .append(id, &new_chunk.source_ref, &new_chunk.text)
                    .map_err(BucketError::Io)?;
            }
        }
        let chunk_count = writer.finalize().map_err(BucketError::Io)? as u64;

        // Update manifest with final stats and promote to ready.
        manifest.state = SlotState::Ready;
        manifest.build_completed_at = Some(chrono::Utc::now());
        manifest.stats.source_records = source_records;
        manifest.stats.chunk_count = chunk_count;
        manifest.stats.disk_size_bytes = total_dir_size(&slot_path).unwrap_or(0);
        write_manifest(&slot_path, &manifest)?;

        // Open the chunks reader for the new slot before we promote —
        // promotion should never make a slot active that we can't
        // immediately serve queries from.
        let chunks = ChunkStoreReader::open(&bin_path, &idx_path).map_err(BucketError::Io)?;
        let loaded = Arc::new(LoadedSlot { manifest, chunks });

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
        _query_vec: &'a [f32],
        _top_k: usize,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<Candidate>, BucketError>> {
        // Dense path comes online in slice 4 (embedding) + slice 5 (HNSW).
        // For now: no index, no candidates.
        Box::pin(async move { Ok(Vec::new()) })
    }

    fn sparse_search<'a>(
        &'a self,
        _query_text: &'a str,
        _top_k: usize,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<Candidate>, BucketError>> {
        // Sparse path comes online in slice 6 (tantivy).
        Box::pin(async move { Ok(Vec::new()) })
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
                "insert: deferred until indexes land (slices 4–6)".to_string(),
            ))
        })
    }

    fn tombstone<'a>(&'a self, _chunk_ids: Vec<ChunkId>) -> BoxFuture<'a, Result<(), BucketError>> {
        Box::pin(async move {
            Err(BucketError::Other(
                "tombstone: deferred until indexes land (slices 4–6)".to_string(),
            ))
        })
    }

    fn compact<'a>(
        &'a self,
        _cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<(), BucketError>> {
        Box::pin(async move {
            Err(BucketError::Other(
                "compact: deferred until indexes land (slices 4–6)".to_string(),
            ))
        })
    }
}

// --- helpers ---

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

    let bin_path = slot::chunks_bin_path(&slot_path);
    let idx_path = slot::chunks_idx_path(&slot_path);
    let chunks = ChunkStoreReader::open(&bin_path, &idx_path).map_err(BucketError::Io)?;

    Ok(LoadedSlot { manifest, chunks })
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

    use super::*;
    use crate::knowledge::{MarkdownDir, TokenBasedChunker};

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

    fn sample_embedder_snapshot() -> EmbedderSnapshot {
        EmbedderSnapshot {
            provider: "tei_test".to_string(),
            model_id: "test/embedding".to_string(),
            dimension: 1024,
        }
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
    }

    #[test]
    fn open_missing_bucket_toml_errors() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path()).unwrap();
        let err = DiskBucket::open(tmp.path(), BucketId::pod("notes")).unwrap_err();
        assert!(matches!(err, BucketError::Config(_)), "{err:?}");
    }

    #[test]
    fn build_slot_writes_chunks_and_promotes_active() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "alpha.md", "alpha content one");
        write_md(&source, "beta.md", "beta content two");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);

        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let cancel = CancellationToken::new();

        let slot_id = bucket
            .build_slot(&adapter, &chunker, sample_embedder_snapshot(), &cancel)
            .unwrap();

        // Active pointer flipped
        assert_eq!(bucket.active_slot_id().as_deref(), Some(slot_id.as_str()));
        assert!(matches!(bucket.status(), BucketStatus::Ready));

        // Slot directory exists with the expected files
        let slot_path = slot::slot_dir(&bucket_root, &slot_id);
        assert!(slot_path.is_dir());
        assert!(slot::manifest_path(&slot_path).is_file());
        assert!(slot::chunks_bin_path(&slot_path).is_file());
        assert!(slot::chunks_idx_path(&slot_path).is_file());

        // Manifest reflects ready state with stats populated
        let manifest_text = fs::read_to_string(slot::manifest_path(&slot_path)).unwrap();
        let manifest = SlotManifest::from_toml_str(&manifest_text).unwrap();
        assert_eq!(manifest.state, SlotState::Ready);
        assert!(manifest.build_completed_at.is_some());
        assert_eq!(manifest.stats.source_records, 2);
        assert!(manifest.stats.chunk_count >= 2); // at least one chunk per record
        assert!(manifest.stats.disk_size_bytes > 0);
    }

    #[tokio::test]
    async fn fetch_chunk_returns_persisted_text() {
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
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, sample_embedder_snapshot(), &cancel)
            .unwrap();

        // Find the chunk id by re-running the chunker on a synthesized
        // record matching what the adapter produced.
        let record_text = "The quick brown fox jumps over the lazy dog.";
        let record = crate::knowledge::SourceRecord::new("ignored", record_text);
        let new_chunks = chunker.chunk(&record);
        assert!(!new_chunks.is_empty());
        let chunk_id = ChunkId::from_source(&new_chunks[0].source_record_hash, 0);

        let chunk = bucket.fetch_chunk(chunk_id).await.unwrap();
        assert_eq!(chunk.id, chunk_id);
        assert_eq!(chunk.text, new_chunks[0].text);
        assert!(chunk.source_ref.source_id.ends_with("doc.md"));
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
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, sample_embedder_snapshot(), &cancel)
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

    #[test]
    fn rebuilding_open_finds_active_slot() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hello");

        let bucket_root = tmp.path().join("bucket");
        let slot_id = {
            let bucket = open_test_bucket(&bucket_root, &source);
            let adapter = MarkdownDir::new(&source);
            let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
            let cancel = CancellationToken::new();
            bucket
                .build_slot(&adapter, &chunker, sample_embedder_snapshot(), &cancel)
                .unwrap()
        };

        // Re-open the bucket from disk; the active slot pointer should
        // resolve and the new bucket reports Ready.
        let reopened = DiskBucket::open(&bucket_root, BucketId::pod("notes")).unwrap();
        assert_eq!(reopened.active_slot_id().as_deref(), Some(slot_id.as_str()));
        assert!(matches!(reopened.status(), BucketStatus::Ready));
    }

    #[tokio::test]
    async fn search_methods_return_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("notes");
        fs::create_dir_all(&source).unwrap();
        write_md(&source, "doc.md", "hello world");

        let bucket_root = tmp.path().join("bucket");
        let bucket = open_test_bucket(&bucket_root, &source);
        let adapter = MarkdownDir::new(&source);
        let chunker = TokenBasedChunker::from_config(&bucket.config().chunker);
        let cancel = CancellationToken::new();
        bucket
            .build_slot(&adapter, &chunker, sample_embedder_snapshot(), &cancel)
            .unwrap();

        let dense = bucket
            .dense_search(&[0.0; 1024], 10, &cancel)
            .await
            .unwrap();
        assert!(dense.is_empty());
        let sparse = bucket.sparse_search("anything", 10, &cancel).await.unwrap();
        assert!(sparse.is_empty());
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

    #[test]
    fn build_slot_cancellation_aborts_with_cancelled_error() {
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
        let cancel = CancellationToken::new();
        cancel.cancel(); // pre-cancelled
        let err = bucket
            .build_slot(&adapter, &chunker, sample_embedder_snapshot(), &cancel)
            .unwrap_err();
        assert!(matches!(err, BucketError::Cancelled), "{err:?}");
        // No active slot was promoted
        assert!(bucket.active_slot_id().is_none());
    }
}
