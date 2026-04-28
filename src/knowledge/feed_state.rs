//! `feed-state.toml` — bucket-level cursor for `kind = "tracked"`
//! buckets. Records what base snapshot the active slot was built from
//! and what deltas have been applied on top.
//!
//! Lives at `<bucket>/feed-state.toml`, separate from `bucket.toml` so
//! the user-editable bucket config and the system-managed feed cursor
//! never tangle. `linked` and `stored` buckets do *not* have this file —
//! presence is a sufficient runtime check that a bucket is tracked.
//!
//! Today's schema is intentionally narrow — only what the initial-build
//! path needs (`current_base_snapshot_id`). The fields the per-bucket
//! feed worker needs (`last_applied_delta_id`, `last_check_at`,
//! `last_check_outcome`, `last_error`) land alongside the worker; the
//! layout reserves space for them with `#[serde(default)]` on every
//! field so an old state file forward-loads cleanly.
//!
//! Crash safety: writes go to `<path>.partial` then atomically rename.
//! A reader that observes `<path>` always sees a complete file.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::source::{DeltaId, SnapshotId};

/// Bucket directory subpath where downloaded base + delta archives are
/// staged. Bucket-scoped (not slot-scoped) so a slot rebuild reuses the
/// cached download — important for monthly-resync semantics where the
/// same base feeds multiple slot builds over time.
const SOURCE_CACHE_DIRNAME: &str = "source-cache";
const BASE_SUBDIR: &str = "base";
const DELTAS_SUBDIR: &str = "deltas";
const FEED_STATE_FILENAME: &str = "feed-state.toml";

/// Path of the feed-state file for a bucket.
pub fn feed_state_path(bucket_dir: &Path) -> PathBuf {
    bucket_dir.join(FEED_STATE_FILENAME)
}

/// Top-level source-cache directory for a bucket.
pub fn source_cache_dir(bucket_dir: &Path) -> PathBuf {
    bucket_dir.join(SOURCE_CACHE_DIRNAME)
}

/// Per-snapshot cache directory: `<bucket>/source-cache/base/<id>/`.
/// The base archive itself lives inside this directory under its
/// driver-published filename.
pub fn base_cache_dir(bucket_dir: &Path, id: &SnapshotId) -> PathBuf {
    source_cache_dir(bucket_dir)
        .join(BASE_SUBDIR)
        .join(id.as_str())
}

/// Per-delta cache directory: `<bucket>/source-cache/deltas/<id>/`.
pub fn delta_cache_dir(bucket_dir: &Path, id: &DeltaId) -> PathBuf {
    source_cache_dir(bucket_dir)
        .join(DELTAS_SUBDIR)
        .join(id.as_str())
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedState {
    /// Driver-specific snapshot id (Wikipedia: `YYYYMMDD`) of the base
    /// snapshot the active slot was built from. `None` before the first
    /// base download completes — i.e. an empty state file represents a
    /// freshly-created tracked bucket whose initial build hasn't run.
    #[serde(default)]
    pub current_base_snapshot_id: Option<String>,

    /// Most recent delta id whose content has been applied to the
    /// active slot's index. The per-bucket feed worker advances this
    /// after a successful tombstone+insert pass; queries up to this
    /// delta are guaranteed to reflect Wikipedia's state at that
    /// point.
    ///
    /// In the *current* slice the field is advanced as soon as the
    /// delta has been *downloaded* (the application step lands in a
    /// follow-up); the cursor's purpose during that window is to bound
    /// `list_deltas_since` against an ever-growing directory listing,
    /// not to make a correctness guarantee. The field's name reflects
    /// its post-T4c semantics so the on-disk format doesn't churn when
    /// the application slice lands.
    #[serde(default)]
    pub last_applied_delta_id: Option<String>,

    /// Wall-clock timestamp of the last successful resync — i.e. the
    /// last time the bucket's active slot was rebuilt off the driver's
    /// `latest_base()`. Used by the per-bucket FeedWorker to compute
    /// the next scheduled resync fire-time on spawn (so a server
    /// restart doesn't reset the resync clock; with monthly cadence
    /// and a k8s `Recreate` deploy strategy, that would otherwise
    /// mean resync never fires in practice).
    ///
    /// `None` means "never resynced" — the worker treats this as
    /// "infinitely overdue" and fires on the first cadence tick after
    /// spawn. Initial-build does *not* set this field; only an
    /// explicit Resync run does.
    #[serde(default)]
    pub last_resync_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl FeedState {
    /// Load from `<bucket>/feed-state.toml`. Returns
    /// [`FeedState::default()`] when the file is missing — that's the
    /// expected state for a freshly-created tracked bucket. Other I/O
    /// or parse errors surface as [`FeedStateError`].
    pub fn load(bucket_dir: &Path) -> Result<Self, FeedStateError> {
        let path = feed_state_path(bucket_dir);
        match std::fs::read_to_string(&path) {
            Ok(s) => toml::from_str(&s).map_err(|e| FeedStateError::Parse {
                path: path.clone(),
                error: e.to_string(),
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(FeedStateError::Io {
                path,
                error: e.to_string(),
            }),
        }
    }

    /// Persist to `<bucket>/feed-state.toml` via write-then-rename so a
    /// crash mid-write never leaves a torn file at the canonical path.
    pub fn save_atomic(&self, bucket_dir: &Path) -> Result<(), FeedStateError> {
        let path = feed_state_path(bucket_dir);
        let tmp = path.with_extension("partial");
        let body = toml::to_string_pretty(self).map_err(|e| FeedStateError::Serialize {
            error: e.to_string(),
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| FeedStateError::Io {
                path: parent.to_path_buf(),
                error: e.to_string(),
            })?;
        }
        std::fs::write(&tmp, &body).map_err(|e| FeedStateError::Io {
            path: tmp.clone(),
            error: e.to_string(),
        })?;
        std::fs::rename(&tmp, &path).map_err(|e| FeedStateError::Io {
            path: path.clone(),
            error: e.to_string(),
        })?;
        Ok(())
    }

    /// Convenience: typed accessor for the snapshot id when set.
    pub fn current_base(&self) -> Option<SnapshotId> {
        self.current_base_snapshot_id
            .as_ref()
            .map(|s| SnapshotId::new(s.clone()))
    }

    /// Convenience: typed setter that round-trips through the inner
    /// `String`.
    pub fn set_current_base(&mut self, id: SnapshotId) {
        self.current_base_snapshot_id = Some(id.0);
    }

    /// Typed accessor for the last-applied delta cursor.
    pub fn last_applied_delta(&self) -> Option<DeltaId> {
        self.last_applied_delta_id
            .as_ref()
            .map(|s| DeltaId::new(s.clone()))
    }

    /// Typed setter for the last-applied delta cursor.
    pub fn set_last_applied_delta(&mut self, id: DeltaId) {
        self.last_applied_delta_id = Some(id.0);
    }

    /// Setter for the last-resync timestamp. Callers should use
    /// `chrono::Utc::now()` at the moment a successful Resync slot
    /// rebuild publishes its new active slot.
    pub fn set_last_resync_at(&mut self, at: chrono::DateTime<chrono::Utc>) {
        self.last_resync_at = Some(at);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FeedStateError {
    #[error("io error at {}: {error}", path.display())]
    Io { path: PathBuf, error: String },

    #[error("parse error in {}: {error}", path.display())]
    Parse { path: PathBuf, error: String },

    #[error("serialize error: {error}")]
    Serialize { error: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_path_helpers_use_bucket_scoped_layout() {
        let bucket = Path::new("/data/buckets/wikipedia_en");
        assert_eq!(
            source_cache_dir(bucket),
            Path::new("/data/buckets/wikipedia_en/source-cache")
        );
        assert_eq!(
            base_cache_dir(bucket, &SnapshotId::new("20260401")),
            Path::new("/data/buckets/wikipedia_en/source-cache/base/20260401")
        );
        assert_eq!(
            delta_cache_dir(bucket, &DeltaId::new("20260424")),
            Path::new("/data/buckets/wikipedia_en/source-cache/deltas/20260424")
        );
        assert_eq!(
            feed_state_path(bucket),
            Path::new("/data/buckets/wikipedia_en/feed-state.toml")
        );
    }

    #[test]
    fn load_returns_default_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let state = FeedState::load(tmp.path()).unwrap();
        assert_eq!(state, FeedState::default());
        assert!(state.current_base().is_none());
    }

    #[test]
    fn save_then_load_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let mut state = FeedState::default();
        state.set_current_base(SnapshotId::new("20260401"));
        state.save_atomic(tmp.path()).unwrap();

        let loaded = FeedState::load(tmp.path()).unwrap();
        assert_eq!(loaded.current_base(), Some(SnapshotId::new("20260401")));
    }

    #[test]
    fn save_atomic_writes_via_rename_no_partial_left_on_success() {
        let tmp = tempfile::tempdir().unwrap();
        let mut state = FeedState::default();
        state.set_current_base(SnapshotId::new("20260401"));
        state.save_atomic(tmp.path()).unwrap();

        let canonical = feed_state_path(tmp.path());
        let partial = canonical.with_extension("partial");
        assert!(canonical.exists(), "canonical state file written");
        assert!(!partial.exists(), "partial cleaned up after rename");
    }

    #[test]
    fn empty_state_file_loads_as_default() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(feed_state_path(tmp.path()), "").unwrap();
        let state = FeedState::load(tmp.path()).unwrap();
        assert_eq!(state, FeedState::default());
    }

    #[test]
    fn unknown_fields_are_rejected_to_catch_typos() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            feed_state_path(tmp.path()),
            "current_base_snapshot_id = \"20260401\"\nunknown_field = 42\n",
        )
        .unwrap();
        let err = FeedState::load(tmp.path()).unwrap_err();
        assert!(matches!(err, FeedStateError::Parse { .. }), "got {err:?}");
    }
}
