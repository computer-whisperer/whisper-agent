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
//! Today every entry is `BucketScope::Server` — pod-scoped buckets
//! (under `<pods_root>/<pod>/buckets/`) land in a follow-up slice when
//! the per-pod plumbing exists.
//!
//! Malformed buckets (missing `bucket.toml`, parse error, dangling
//! `slots/active` pointer) are skipped with a warning. A single bad
//! bucket directory should not brick the registry — the operator can
//! fix it through the WebUI without rebooting the server.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tokio::fs;
use tracing::{info, warn};
use whisper_agent_protocol::{ActiveSlotSummary, BucketSummary, SlotStateLabel};

use super::config::{BucketConfig, SourceConfig};
use super::manifest::{SlotManifest, SlotState};
use super::slot;
use super::types::BucketError;

/// In-memory entry the scheduler holds per bucket on disk.
#[derive(Debug, Clone)]
pub struct BucketEntry {
    /// Bucket directory name. Authoritative id; matches the dir under
    /// `<buckets_root>/`.
    pub id: String,
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

/// Server-scoped knowledge buckets, keyed by bucket id (== directory name).
#[derive(Debug, Clone, Default)]
pub struct BucketRegistry {
    /// `None` when the server runs without a buckets root. The registry
    /// is still constructible (`BucketRegistry::default()`) so the
    /// scheduler doesn't need conditional plumbing.
    pub root: Option<PathBuf>,
    pub buckets: BTreeMap<String, BucketEntry>,
}

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
            match load_one(&path, &id).await {
                Ok(entry) => {
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
        })
    }

    /// Wire snapshot of every bucket, sorted by id (BTreeMap order).
    pub fn summaries(&self) -> Vec<BucketSummary> {
        self.buckets.values().map(entry_to_summary).collect()
    }
}

async fn load_one(dir: &Path, id: &str) -> Result<BucketEntry, BucketError> {
    let bucket_toml = slot::bucket_toml_path(dir);
    let raw_toml = fs::read_to_string(&bucket_toml).await.map_err(|e| {
        BucketError::Config(format!(
            "bucket.toml unreadable at {}: {e}",
            bucket_toml.display()
        ))
    })?;
    let config = BucketConfig::from_toml_str(&raw_toml)?;

    let (active_slot, active_slot_disk_size_bytes) = match read_active_manifest(dir).await? {
        Some((manifest, slot_path)) => {
            let size = dir_size(&slot_path).await.unwrap_or(0);
            (Some(manifest), Some(size))
        }
        None => (None, None),
    };

    Ok(BucketEntry {
        id: id.to_string(),
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

async fn dir_size(dir: &Path) -> std::io::Result<u64> {
    let mut total: u64 = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(p) = stack.pop() {
        let mut entries = fs::read_dir(&p).await?;
        while let Some(entry) = entries.next_entry().await? {
            let md = entry.metadata().await?;
            if md.is_dir() {
                stack.push(entry.path());
            } else {
                total += md.len();
            }
        }
    }
    Ok(total)
}

fn entry_to_summary(entry: &BucketEntry) -> BucketSummary {
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
    };
    BucketSummary {
        id: entry.id.clone(),
        scope: "server".to_string(),
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
}
