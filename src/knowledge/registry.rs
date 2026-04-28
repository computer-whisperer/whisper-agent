//! Server-side registry of knowledge buckets.
//!
//! Walks `<buckets_root>/*/bucket.toml` at startup and holds an entry per
//! bucket. Each entry carries the parsed [`BucketConfig`] (plus its raw
//! TOML for round-trip editing) and the *manifest* of the bucket's
//! currently-active slot when one exists. Heavy artifacts (chunks,
//! vectors, dense / sparse indexes) are **not** opened here — server
//! startup must not pay HNSW-rebuild cost per bucket on every restart.
//! The query path opens those lazily when a request actually targets the
//! bucket.
//!
//! Pod-scoped buckets live under `<pods_root>/<pod_id>/buckets/` and are
//! tracked in a separate sub-registry per pod. Server boot loads the
//! global server-scope buckets first via [`BucketRegistry::load`] and
//! then folds each loaded pod's buckets in via
//! [`BucketRegistry::register_pod_buckets`]. Two different pods can hold
//! buckets with the same directory name without collision; the
//! registry keys them by `(pod_id, name)`.
//!
//! Malformed buckets (missing `bucket.toml`, parse error, dangling
//! `slots/active` pointer) are skipped with a warning. A single bad
//! bucket directory should not brick the registry — the operator can
//! fix it through the WebUI without rebooting the server.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokio::fs;
use tracing::{info, warn};
use whisper_agent_protocol::{ActiveSlotSummary, BucketSummary, SlotStateLabel};

use super::config::{BucketConfig, SourceConfig};
use super::disk_bucket::DiskBucket;
use super::manifest::{SlotManifest, SlotState};
use super::slot;
use super::types::{BucketError, BucketId, BucketScope};

/// In-memory entry the scheduler holds per bucket on disk.
#[derive(Debug, Clone)]
pub struct BucketEntry {
    /// Bucket directory name. Authoritative within its scope; matches
    /// the dir under `<buckets_root>/` for server scope or under
    /// `<pods_root>/<pod_id>/buckets/` for pod scope. Bucket names
    /// scope-locally — `pod_a` and `pod_b` can both have a `memory`
    /// bucket without colliding.
    pub id: String,
    /// Whether this bucket lives under the server's buckets root or
    /// under a specific pod's `buckets/` directory.
    pub scope: BucketScope,
    /// Owning pod when `scope == BucketScope::Pod`. `None` for server
    /// scope. Used at lookup time to disambiguate pod-scope entries
    /// across pods.
    pub pod_id: Option<String>,
    /// Absolute path to the bucket directory.
    pub dir: PathBuf,
    pub config: BucketConfig,
    /// On-disk text of `bucket.toml`, kept verbatim so the wire can
    /// return it for raw editing without re-serializing through serde
    /// (which would drop comments and reorder fields).
    pub raw_toml: String,
    /// Active slot's manifest, when one exists. `None` for newly-created
    /// buckets (no slot yet) or buckets whose only slot failed before
    /// reaching `Ready`.
    pub active_slot: Option<SlotManifest>,
    /// Disk size of the active slot, computed at registry-load time.
    /// Cached here so summaries don't re-walk the slot directory on
    /// every wire response. `None` mirrors `active_slot`.
    pub active_slot_disk_size_bytes: Option<u64>,
}

/// Knowledge buckets known to the server, partitioned by scope.
///
/// `buckets` holds server-scope entries — buckets directly under
/// `<buckets_root>/`, keyed by directory name. `pod_buckets` holds
/// pod-scope entries — buckets under `<pods_root>/<pod_id>/buckets/`,
/// keyed by `(pod_id, bucket_name)` so two pods can both name a
/// bucket `memory` without colliding.
///
/// `loaded` is a separate cache from the entry maps: the maps hold the
/// cheap config + manifest (loaded eagerly at startup), while `loaded`
/// is the expensive `DiskBucket` view (open the chunk store, mmap
/// vectors, rebuild the HNSW index) — populated lazily on first query
/// and shared across subsequent ones. Pre-populating every bucket at
/// startup would block server boot for minutes per wikipedia-scale
/// bucket; deferring keeps boot fast and pushes the cost to the first
/// query that actually needs it. Today the loaded cache is keyed by
/// the server-scope bucket id only; pod-scope load wiring lands in a
/// follow-up slice alongside the scoped query path.
#[derive(Debug, Clone, Default)]
pub struct BucketRegistry {
    /// `None` when the server runs without a buckets root. The registry
    /// is still constructible (`BucketRegistry::default()`) so the
    /// scheduler doesn't need conditional plumbing.
    pub root: Option<PathBuf>,
    /// Server-scope entries, keyed by bucket directory name.
    pub buckets: BTreeMap<String, BucketEntry>,
    /// Pod-scope entries, keyed by pod id then bucket directory name.
    /// Folded in via [`BucketRegistry::register_pod_buckets`] once the
    /// pod registry is loaded.
    pub pod_buckets: BTreeMap<String, BTreeMap<String, BucketEntry>>,
    /// Lazy `DiskBucket` cache for server-scope buckets, keyed by
    /// bucket name.
    loaded: Arc<Mutex<HashMap<String, Arc<DiskBucket>>>>,
    /// Lazy `DiskBucket` cache for pod-scope buckets, keyed by
    /// `(pod_id, bucket_name)` so two pods with the same bucket name
    /// don't share a cache entry.
    loaded_pod: PodLoadedCache,
}

/// Type alias for the pod-scope `DiskBucket` cache. The
/// `(pod_id, bucket_name)` key keeps two pods' identically-named
/// buckets distinct; the alias avoids the very-complex-type clippy
/// lint at the field site.
type PodLoadedCache = Arc<Mutex<HashMap<(String, String), Arc<DiskBucket>>>>;

impl BucketRegistry {
    /// Walk `root/*/bucket.toml` and build a registry. Missing or
    /// malformed buckets emit a warning and are skipped. The directory
    /// itself is created if absent so a fresh server starts with an
    /// empty registry rather than failing at boot.
    pub async fn load(root: PathBuf) -> Result<Self, BucketError> {
        fs::create_dir_all(&root).await.map_err(BucketError::Io)?;
        let mut buckets: BTreeMap<String, BucketEntry> = BTreeMap::new();

        let mut entries = fs::read_dir(&root).await.map_err(BucketError::Io)?;
        while let Some(entry) = entries.next_entry().await.map_err(BucketError::Io)? {
            let path = entry.path();
            let ft = match entry.file_type().await {
                Ok(ft) => ft,
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "buckets root: skipping unreadable entry");
                    continue;
                }
            };
            if !ft.is_dir() {
                continue;
            }
            let Some(id) = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(str::to_string)
            else {
                warn!(path = %path.display(), "buckets root: skipping non-utf8 directory name");
                continue;
            };
            // Hidden / dotfiles (e.g. lost+found, editor scratch) — skip.
            if id.starts_with('.') {
                continue;
            }
            match load_one(&path, &id, BucketScope::Server, None).await {
                Ok(entry) => {
                    // Sweep orphaned slot directories — partial /
                    // failed / superseded build attempts that no
                    // longer reference the active slot. Failures are
                    // non-fatal: a broken slots dir shouldn't keep
                    // the bucket out of the registry.
                    if let Err(e) = super::gc::gc_orphan_slots(&path, &id) {
                        warn!(
                            bucket = %id,
                            error = %e,
                            "gc: orphan-slot sweep failed; bucket loaded anyway",
                        );
                    }
                    buckets.insert(id, entry);
                }
                Err(e) => {
                    warn!(bucket = %id, error = %e, "skipping malformed bucket");
                }
            }
        }

        info!(
            buckets_root = %root.display(),
            count = buckets.len(),
            "loaded knowledge bucket registry",
        );
        Ok(Self {
            root: Some(root),
            buckets,
            pod_buckets: BTreeMap::new(),
            loaded: Arc::new(Mutex::new(HashMap::new())),
            loaded_pod: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Walk `<pod_dir>/buckets/*/bucket.toml` and register each as a
    /// pod-scope entry under `pod_id`. The `<pod_dir>/buckets` directory
    /// is created on demand so a pod that has never owned a bucket
    /// behaves the same as one that has — both end up with an empty
    /// inner map. Idempotent: re-registering the same pod replaces its
    /// previous entry set.
    ///
    /// Failures while loading a single bucket are logged and skipped
    /// (matching server-scope `load`). The whole call returns `Err` only
    /// if `<pod_dir>/buckets` itself can't be created or read.
    pub async fn register_pod_buckets(
        &mut self,
        pod_id: &str,
        pod_dir: &Path,
    ) -> Result<(), BucketError> {
        let buckets_dir = pod_dir.join("buckets");
        fs::create_dir_all(&buckets_dir)
            .await
            .map_err(BucketError::Io)?;
        let mut inner: BTreeMap<String, BucketEntry> = BTreeMap::new();

        let mut entries = fs::read_dir(&buckets_dir).await.map_err(BucketError::Io)?;
        while let Some(entry) = entries.next_entry().await.map_err(BucketError::Io)? {
            let path = entry.path();
            let ft = match entry.file_type().await {
                Ok(ft) => ft,
                Err(e) => {
                    warn!(
                        path = %path.display(),
                        error = %e,
                        "pod buckets: skipping unreadable entry",
                    );
                    continue;
                }
            };
            if !ft.is_dir() {
                continue;
            }
            let Some(id) = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(str::to_string)
            else {
                warn!(
                    path = %path.display(),
                    "pod buckets: skipping non-utf8 directory name",
                );
                continue;
            };
            if id.starts_with('.') {
                continue;
            }
            match load_one(&path, &id, BucketScope::Pod, Some(pod_id.to_string())).await {
                Ok(entry) => {
                    if let Err(e) = super::gc::gc_orphan_slots(&path, &id) {
                        warn!(
                            pod = %pod_id,
                            bucket = %id,
                            error = %e,
                            "gc: orphan-slot sweep failed; bucket loaded anyway",
                        );
                    }
                    inner.insert(id, entry);
                }
                Err(e) => {
                    warn!(
                        pod = %pod_id,
                        bucket = %id,
                        error = %e,
                        "skipping malformed pod bucket",
                    );
                }
            }
        }

        info!(
            pod = %pod_id,
            buckets_dir = %buckets_dir.display(),
            count = inner.len(),
            "registered pod-scope knowledge buckets",
        );
        self.pod_buckets.insert(pod_id.to_string(), inner);
        Ok(())
    }

    /// Wire snapshot of every bucket the registry knows about. Server-
    /// scope entries come first (BTreeMap order), followed by pod-scope
    /// entries grouped by pod (BTreeMap order on pod_id, then on bucket
    /// name within each pod). Clients re-sort if they want a different
    /// presentation order.
    pub fn summaries(&self) -> Vec<BucketSummary> {
        let mut out: Vec<BucketSummary> = self.buckets.values().map(entry_to_summary).collect();
        for inner in self.pod_buckets.values() {
            out.extend(inner.values().map(entry_to_summary));
        }
        out
    }

    /// Insert a fresh bucket into the registry — caller has already
    /// written `bucket.toml` under `<root>/<id>/` and validated `id`
    /// for filesystem safety. Used by the scheduler's `CreateBucket`
    /// handler; tests construct entries directly.
    ///
    /// Errors if `id` already exists in the registry (the on-disk
    /// directory may pre-date the in-memory entry, so the caller
    /// should also check the filesystem).
    pub fn insert_entry(&mut self, entry: BucketEntry) -> Result<(), BucketError> {
        if self.buckets.contains_key(&entry.id) {
            return Err(BucketError::Other(format!(
                "bucket id `{}` already in registry",
                entry.id,
            )));
        }
        self.buckets.insert(entry.id.clone(), entry);
        Ok(())
    }

    /// Drop the registry's in-memory entry for `id` and evict any
    /// cached `DiskBucket`. The caller is responsible for removing
    /// the on-disk directory afterwards. Returns whether the entry
    /// existed.
    pub fn remove_entry(&mut self, id: &str) -> bool {
        let removed = self.buckets.remove(id).is_some();
        self.evict_loaded(id);
        removed
    }

    /// Drop a cached `DiskBucket` for `id` without touching the
    /// `buckets` map. Used by `refresh_entry` after a rebuild and by
    /// the delete path. Cheap — O(1) hashmap remove under a short-
    /// held mutex.
    pub fn evict_loaded(&self, id: &str) {
        let mut g = self.loaded.lock().expect("loaded mutex poisoned");
        g.remove(id);
    }

    /// Re-read the bucket's `bucket.toml` + active-slot manifest from
    /// disk and replace the in-memory entry. Called after a build
    /// completes so the next `summaries()` reflects the new active
    /// slot. Also evicts the cached `DiskBucket` so the next query
    /// re-opens against the fresh slot.
    pub async fn refresh_entry(&mut self, id: &str) -> Result<(), BucketError> {
        let dir = match self.buckets.get(id) {
            Some(e) => e.dir.clone(),
            None => {
                let root = self.root.as_ref().ok_or_else(|| {
                    BucketError::Other("registry has no root; cannot refresh".to_string())
                })?;
                root.join(id)
            }
        };
        let entry = load_one(&dir, id, BucketScope::Server, None).await?;
        self.buckets.insert(id.to_string(), entry);
        self.evict_loaded(id);
        Ok(())
    }

    /// Return a fully-loaded `DiskBucket` for `id`, populating the
    /// scheduler-side cache on first request. The first call against
    /// an unloaded bucket blocks on slot-load (HNSW rebuild); later
    /// callers reuse the `Arc`.
    ///
    /// Concurrent first-loads of the same bucket are *not*
    /// deduplicated — two callers racing on a cold cache may each
    /// call `DiskBucket::open` and the second's result is discarded
    /// when it tries to insert. Acceptable for v1; if HNSW rebuild
    /// becomes hot path we can graduate to a per-bucket OnceCell.
    pub async fn loaded_bucket(&self, id: &str) -> Result<Arc<DiskBucket>, BucketError> {
        {
            let g = self.loaded.lock().expect("loaded mutex poisoned");
            if let Some(b) = g.get(id) {
                return Ok(b.clone());
            }
        }

        let entry = self
            .buckets
            .get(id)
            .ok_or_else(|| BucketError::Other(format!("unknown bucket id: {id}")))?;
        let dir = entry.dir.clone();
        let id_owned = id.to_string();

        // `DiskBucket::open` is sync but slow (HNSW rebuild). Push it
        // off the runtime so we don't stall the scheduler thread.
        let bucket =
            tokio::task::spawn_blocking(move || DiskBucket::open(dir, BucketId::server(id_owned)))
                .await
                .map_err(|e| BucketError::Other(format!("spawn_blocking: {e}")))??;
        let arc = Arc::new(bucket);

        let mut g = self.loaded.lock().expect("loaded mutex poisoned");
        if let Some(existing) = g.get(id) {
            // Lost the race; throw away our load and use the winner.
            return Ok(existing.clone());
        }
        g.insert(id.to_string(), arc.clone());
        Ok(arc)
    }

    /// Pod-scope counterpart to [`Self::loaded_bucket`]. Looks up
    /// `(pod_id, name)` in the pod sub-registry, opens the bucket
    /// (HNSW load via `spawn_blocking`), and caches the result keyed
    /// by `(pod_id, name)` so two pods with the same bucket name don't
    /// share an entry.
    pub async fn loaded_bucket_pod(
        &self,
        pod_id: &str,
        name: &str,
    ) -> Result<Arc<DiskBucket>, BucketError> {
        let key = (pod_id.to_string(), name.to_string());
        {
            let g = self.loaded_pod.lock().expect("loaded_pod mutex poisoned");
            if let Some(b) = g.get(&key) {
                return Ok(b.clone());
            }
        }

        let entry = self
            .pod_buckets
            .get(pod_id)
            .and_then(|m| m.get(name))
            .ok_or_else(|| {
                BucketError::Other(format!(
                    "unknown pod-scope bucket: pod={pod_id}, name={name}"
                ))
            })?;
        let dir = entry.dir.clone();
        let id_owned = name.to_string();

        let bucket =
            tokio::task::spawn_blocking(move || DiskBucket::open(dir, BucketId::pod(id_owned)))
                .await
                .map_err(|e| BucketError::Other(format!("spawn_blocking: {e}")))??;
        let arc = Arc::new(bucket);

        let mut g = self.loaded_pod.lock().expect("loaded_pod mutex poisoned");
        if let Some(existing) = g.get(&key) {
            return Ok(existing.clone());
        }
        g.insert(key, arc.clone());
        Ok(arc)
    }

    /// Unified lookup for a bucket entry by scope. `pod_id` must be
    /// `Some` for `BucketScope::Pod` and is ignored for
    /// `BucketScope::Server`. Returns `None` if no matching entry
    /// exists in the relevant sub-registry.
    pub fn find_entry(
        &self,
        scope: BucketScope,
        pod_id: Option<&str>,
        name: &str,
    ) -> Option<&BucketEntry> {
        match scope {
            BucketScope::Server => self.buckets.get(name),
            BucketScope::Pod => self.pod_buckets.get(pod_id?).and_then(|m| m.get(name)),
        }
    }
}

/// Public re-export of the registry's per-entry loader. The scheduler
/// uses this from `CreateBucket` after writing a fresh `bucket.toml`
/// to round-trip through the same parser the registry uses at
/// startup. `CreateBucket` only handles server-scope today, so the
/// public form hard-codes that scope.
pub async fn load_one_pub(dir: &Path, id: &str) -> Result<BucketEntry, BucketError> {
    load_one(dir, id, BucketScope::Server, None).await
}

async fn load_one(
    dir: &Path,
    id: &str,
    scope: BucketScope,
    pod_id: Option<String>,
) -> Result<BucketEntry, BucketError> {
    let bucket_toml = slot::bucket_toml_path(dir);
    let raw_toml = fs::read_to_string(&bucket_toml).await.map_err(|e| {
        BucketError::Config(format!(
            "bucket.toml unreadable at {}: {e}",
            bucket_toml.display()
        ))
    })?;
    let config = BucketConfig::from_toml_str(&raw_toml)?;

    // Disk size is read from the manifest's recorded stat rather
    // than recomputed via a recursive directory walk. A simplewiki-
    // sized slot (post-mutation) holds tens of thousands of tantivy
    // segment files; walking them at boot blocked HTTP startup for
    // ~50s on the test sandbox. The manifest stat is set once at
    // build / compaction completion — slightly stale after
    // intervening inserts / tombstones / delta applies, but accurate
    // enough for the UI's "active slot: 9.4 GiB" display. A `Refresh`
    // op (or a future end-of-apply_delta hook) can recompute when
    // freshness matters.
    let (active_slot, active_slot_disk_size_bytes) = match read_active_manifest(dir).await? {
        Some((manifest, _slot_path)) => {
            let size = manifest.stats.disk_size_bytes;
            (Some(manifest), Some(size))
        }
        None => (None, None),
    };

    Ok(BucketEntry {
        id: id.to_string(),
        scope,
        pod_id,
        dir: dir.to_path_buf(),
        config,
        raw_toml,
        active_slot,
        active_slot_disk_size_bytes,
    })
}

/// Returns `(manifest, slot_dir)` for the bucket's active slot, or
/// `None` if there isn't one.
async fn read_active_manifest(
    bucket_dir: &Path,
) -> Result<Option<(SlotManifest, PathBuf)>, BucketError> {
    // `read_active_slot` is sync but cheap (one tiny file read).
    let Some(slot_id) = slot::read_active_slot(bucket_dir)? else {
        return Ok(None);
    };
    let slot_path = slot::slot_dir(bucket_dir, &slot_id);
    let manifest_path = slot::manifest_path(&slot_path);
    let text = match fs::read_to_string(&manifest_path).await {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warn!(
                bucket = %bucket_dir.display(),
                slot = %slot_id,
                "active-slot pointer points at slot with no manifest.toml; treating as no active slot",
            );
            return Ok(None);
        }
        Err(e) => return Err(BucketError::Io(e)),
    };
    let manifest = SlotManifest::from_toml_str(&text)?;
    Ok(Some((manifest, slot_path)))
}

/// Project a registry entry to its wire-summary form. Exposed so
/// scheduler handlers can build a `BucketCreated` / `BucketBuildEnded`
/// payload without re-walking the whole registry.
pub fn entry_to_summary(entry: &BucketEntry) -> BucketSummary {
    let cfg = &entry.config;
    let (source_kind, source_detail) = match &cfg.source {
        SourceConfig::Stored { archive_path, .. } => (
            "stored".to_string(),
            Some(archive_path.display().to_string()),
        ),
        SourceConfig::Linked { path, .. } => {
            ("linked".to_string(), Some(path.display().to_string()))
        }
        SourceConfig::Managed {} => ("managed".to_string(), None),
        SourceConfig::Tracked { driver, .. } => {
            // Surface "wikipedia: en", "wikipedia: de", … so the WebUI
            // bucket list can show what feed each tracked bucket follows
            // without exposing internal cadence / mirror state.
            let detail = match driver {
                super::config::TrackedDriver::Wikipedia { language, .. } => {
                    format!("wikipedia: {language}")
                }
            };
            ("tracked".to_string(), Some(detail))
        }
    };
    BucketSummary {
        id: entry.id.clone(),
        scope: entry.scope.as_str().to_string(),
        pod_id: entry.pod_id.clone(),
        name: cfg.name.clone(),
        description: cfg.description.clone(),
        source_kind,
        source_detail,
        embedder_provider: cfg.defaults.embedder.clone(),
        dense_enabled: cfg.search_paths.dense.enabled,
        sparse_enabled: cfg.search_paths.sparse.enabled,
        created_at: cfg.created_at.to_rfc3339(),
        active_slot: entry.active_slot.as_ref().map(|m| ActiveSlotSummary {
            slot_id: m.slot_id.clone(),
            state: slot_state_label(m.state),
            embedder_model: m.embedder.model_id.clone(),
            dimension: m.embedder.dimension,
            chunk_count: m.stats.chunk_count,
            vector_count: m.stats.vector_count,
            disk_size_bytes: entry.active_slot_disk_size_bytes.unwrap_or(0),
            created_at: m.created_at.to_rfc3339(),
            built_at: m.build_completed_at.map(|t| t.to_rfc3339()),
        }),
    }
}

fn slot_state_label(s: SlotState) -> SlotStateLabel {
    match s {
        SlotState::Planning => SlotStateLabel::Planning,
        SlotState::Building => SlotStateLabel::Building,
        SlotState::Ready => SlotStateLabel::Ready,
        SlotState::Failed => SlotStateLabel::Failed,
        SlotState::Archived => SlotStateLabel::Archived,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(p: &Path, s: &str) {
        std::fs::write(p, s).unwrap();
    }

    fn write_minimal_bucket(root: &Path, name: &str) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        write(
            &dir.join("bucket.toml"),
            r#"
name = "Notes"
created_at = "2026-04-25T10:23:11Z"
[source]
kind = "linked"
adapter = "markdown_dir"
path = "/home/me/notes"
[defaults]
embedder = "tei_local"
"#,
        );
    }

    #[tokio::test]
    async fn empty_root_loads_empty_registry() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = BucketRegistry::load(tmp.path().to_path_buf())
            .await
            .unwrap();
        assert!(reg.buckets.is_empty());
        assert!(reg.summaries().is_empty());
    }

    #[tokio::test]
    async fn missing_root_is_created_and_loads_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("buckets_does_not_exist_yet");
        let reg = BucketRegistry::load(root.clone()).await.unwrap();
        assert!(reg.buckets.is_empty());
        assert!(root.is_dir(), "load should mkdir -p the root");
    }

    #[tokio::test]
    async fn one_bucket_no_active_slot_loads() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_bucket(tmp.path(), "notes");
        let reg = BucketRegistry::load(tmp.path().to_path_buf())
            .await
            .unwrap();
        assert_eq!(reg.buckets.len(), 1);
        let summary = &reg.summaries()[0];
        assert_eq!(summary.id, "notes");
        assert_eq!(summary.scope, "server");
        assert!(summary.pod_id.is_none());
        assert_eq!(summary.name, "Notes");
        assert_eq!(summary.source_kind, "linked");
        assert_eq!(summary.embedder_provider, "tei_local");
        assert!(summary.dense_enabled);
        assert!(summary.sparse_enabled);
        assert!(summary.active_slot.is_none());
    }

    #[tokio::test]
    async fn malformed_bucket_is_skipped_with_warning() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_bucket(tmp.path(), "good");
        let bad = tmp.path().join("bad");
        std::fs::create_dir_all(&bad).unwrap();
        write(&bad.join("bucket.toml"), "this is not valid toml = = =");
        let reg = BucketRegistry::load(tmp.path().to_path_buf())
            .await
            .unwrap();
        assert_eq!(reg.buckets.len(), 1);
        assert!(reg.buckets.contains_key("good"));
        assert!(!reg.buckets.contains_key("bad"));
    }

    #[tokio::test]
    async fn bucket_without_bucket_toml_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("naked")).unwrap();
        write_minimal_bucket(tmp.path(), "good");
        let reg = BucketRegistry::load(tmp.path().to_path_buf())
            .await
            .unwrap();
        assert_eq!(reg.buckets.len(), 1);
        assert!(reg.buckets.contains_key("good"));
    }

    #[tokio::test]
    async fn dotfile_dirs_are_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_bucket(tmp.path(), "good");
        let dot = tmp.path().join(".trash");
        std::fs::create_dir_all(&dot).unwrap();
        write(&dot.join("bucket.toml"), "definitely not valid");
        let reg = BucketRegistry::load(tmp.path().to_path_buf())
            .await
            .unwrap();
        assert_eq!(reg.buckets.len(), 1);
        assert!(reg.buckets.contains_key("good"));
    }

    #[tokio::test]
    async fn bucket_with_active_slot_pointer_but_missing_manifest_falls_back_to_no_slot() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_bucket(tmp.path(), "notes");
        // Create slots/ + an active pointer but no slot dir / manifest.
        let bucket_dir = tmp.path().join("notes");
        let slots = bucket_dir.join("slots");
        std::fs::create_dir_all(&slots).unwrap();
        write(&slots.join("active"), "ghost-slot-id\n");
        let reg = BucketRegistry::load(tmp.path().to_path_buf())
            .await
            .unwrap();
        assert_eq!(reg.buckets.len(), 1);
        assert!(reg.summaries()[0].active_slot.is_none());
    }

    #[tokio::test]
    async fn register_pod_buckets_creates_buckets_subdir_when_absent() {
        // A pod that has never owned a bucket should still register
        // cleanly — the side effect is an empty inner map and a fresh
        // `<pod_dir>/buckets/` on disk.
        let tmp = tempfile::tempdir().unwrap();
        let buckets_root = tmp.path().join("buckets_root");
        let mut reg = BucketRegistry::load(buckets_root).await.unwrap();
        let pod_dir = tmp.path().join("pods").join("alpha");
        std::fs::create_dir_all(&pod_dir).unwrap();
        reg.register_pod_buckets("alpha", &pod_dir).await.unwrap();
        assert!(reg.pod_buckets.contains_key("alpha"));
        assert!(reg.pod_buckets["alpha"].is_empty());
        assert!(pod_dir.join("buckets").is_dir());
    }

    #[tokio::test]
    async fn register_pod_buckets_loads_entries_with_pod_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let buckets_root = tmp.path().join("buckets_root");
        let mut reg = BucketRegistry::load(buckets_root).await.unwrap();

        let pod_dir = tmp.path().join("pods").join("alpha");
        let pod_buckets = pod_dir.join("buckets");
        std::fs::create_dir_all(&pod_buckets).unwrap();
        write_minimal_bucket(&pod_buckets, "memory");

        reg.register_pod_buckets("alpha", &pod_dir).await.unwrap();
        let inner = reg.pod_buckets.get("alpha").expect("pod registered");
        assert_eq!(inner.len(), 1);
        let entry = inner.get("memory").expect("memory bucket loaded");
        assert!(matches!(entry.scope, BucketScope::Pod));
        assert_eq!(entry.pod_id.as_deref(), Some("alpha"));
        assert_eq!(entry.dir, pod_buckets.join("memory"));
    }

    #[tokio::test]
    async fn summaries_include_server_then_pod_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let buckets_root = tmp.path().join("buckets_root");
        std::fs::create_dir_all(&buckets_root).unwrap();
        write_minimal_bucket(&buckets_root, "world");
        let mut reg = BucketRegistry::load(buckets_root).await.unwrap();

        let pod_dir = tmp.path().join("pods").join("alpha");
        let pod_buckets = pod_dir.join("buckets");
        std::fs::create_dir_all(&pod_buckets).unwrap();
        write_minimal_bucket(&pod_buckets, "memory");
        reg.register_pod_buckets("alpha", &pod_dir).await.unwrap();

        let summaries = reg.summaries();
        assert_eq!(summaries.len(), 2);
        // Server scope first.
        assert_eq!(summaries[0].id, "world");
        assert_eq!(summaries[0].scope, "server");
        assert!(summaries[0].pod_id.is_none());
        // Pod scope second, with pod_id surfaced.
        assert_eq!(summaries[1].id, "memory");
        assert_eq!(summaries[1].scope, "pod");
        assert_eq!(summaries[1].pod_id.as_deref(), Some("alpha"));
    }

    #[tokio::test]
    async fn register_pod_buckets_keeps_two_pods_with_same_bucket_name_separate() {
        let tmp = tempfile::tempdir().unwrap();
        let buckets_root = tmp.path().join("buckets_root");
        let mut reg = BucketRegistry::load(buckets_root).await.unwrap();

        for pod in ["alpha", "beta"] {
            let pod_dir = tmp.path().join("pods").join(pod);
            let pod_buckets = pod_dir.join("buckets");
            std::fs::create_dir_all(&pod_buckets).unwrap();
            write_minimal_bucket(&pod_buckets, "memory");
            reg.register_pod_buckets(pod, &pod_dir).await.unwrap();
        }

        assert_eq!(reg.pod_buckets.len(), 2);
        assert_eq!(reg.pod_buckets["alpha"].len(), 1);
        assert_eq!(reg.pod_buckets["beta"].len(), 1);
        assert_eq!(
            reg.pod_buckets["alpha"]["memory"].pod_id.as_deref(),
            Some("alpha"),
        );
        assert_eq!(
            reg.pod_buckets["beta"]["memory"].pod_id.as_deref(),
            Some("beta"),
        );
    }

    #[tokio::test]
    async fn re_registering_a_pod_replaces_its_previous_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let buckets_root = tmp.path().join("buckets_root");
        let mut reg = BucketRegistry::load(buckets_root).await.unwrap();

        let pod_dir = tmp.path().join("pods").join("alpha");
        let pod_buckets = pod_dir.join("buckets");
        std::fs::create_dir_all(&pod_buckets).unwrap();
        write_minimal_bucket(&pod_buckets, "memory");
        reg.register_pod_buckets("alpha", &pod_dir).await.unwrap();
        assert_eq!(reg.pod_buckets["alpha"].len(), 1);

        // Drop the bucket on disk and re-register: the in-memory map
        // should follow disk truth, not accumulate.
        std::fs::remove_dir_all(pod_buckets.join("memory")).unwrap();
        reg.register_pod_buckets("alpha", &pod_dir).await.unwrap();
        assert!(reg.pod_buckets["alpha"].is_empty());
    }
}
