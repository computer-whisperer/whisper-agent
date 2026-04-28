//! Slot-directory garbage collection.
//!
//! Multi-day enwiki builds get interrupted (deploys, cluster events,
//! manual cancels) and leave `slots/<id>/` directories in `Planning`
//! or `Building` state. Without GC, these accumulate gigabytes of
//! orphaned partial vectors, HNSW dumps, and tantivy indexes that
//! the rest of the system never references again.
//!
//! Policy:
//! - The **active** slot (per `<bucket_dir>/active`) is never removed.
//! - The **most-recent resumable** slot — the latest `Planning` or
//!   `Building` slot, matching [`DiskBucket::find_resumable_slot`]'s
//!   semantics — is preserved so a future "Build" click can resume it.
//! - **Older** `Planning`/`Building` slots (a fresh build was started
//!   while another was already mid-flight) are removed.
//! - **`Failed`** slots are removed. The manifest's
//!   "left in place for diagnosis until explicitly cleaned up" note
//!   doesn't survive contact with multi-day builds — operators can
//!   re-run with `RUST_LOG=...` if they actually want diagnosis.
//! - Slot directories with **no `manifest.toml`** or an **unparseable**
//!   one are removed (crashes during the Planning phase before the
//!   manifest landed).
//! - **`Ready`** and **`Archived`** slots are left alone here. Ready
//!   slots are rotated past versions retained for rollback; Archived
//!   has its own retention window. Both belong to a separate retention
//!   sweep, not this GC pass.
//!
//! GC runs once per bucket inside [`BucketRegistry::load`]. Failures
//! are logged but non-fatal — a broken `slots/` directory shouldn't
//! prevent the bucket from loading.

use std::fs;
use std::path::Path;

use tracing::{debug, info, warn};

use super::manifest::{SlotManifest, SlotState};
use super::slot;
use super::types::{BucketError, SlotId};

/// Outcome of a per-bucket GC pass. Counters surface through the
/// load-time `info!` log so operators see what was reclaimed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct GcReport {
    pub bucket_id: String,
    /// Slot ids that were deleted, in deletion order.
    pub removed: Vec<SlotId>,
    /// Approximate disk space reclaimed by the deletions.
    pub bytes_freed: u64,
    /// The active slot id (never deleted), if any.
    pub kept_active: Option<SlotId>,
    /// The most-recent resumable slot (never deleted), if any.
    pub kept_resumable: Option<SlotId>,
}

/// Sweep `<bucket_dir>/slots/` per the [module docs](self) policy.
/// Returns a [`GcReport`] documenting what was removed and what was
/// kept.
///
/// Pure filesystem operation — no `BucketRegistry`, no `DiskBucket`,
/// no async runtime. Safe to call before the bucket is fully loaded.
pub fn gc_orphan_slots(bucket_dir: &Path, bucket_id: &str) -> Result<GcReport, BucketError> {
    let mut report = GcReport {
        bucket_id: bucket_id.to_string(),
        ..GcReport::default()
    };

    let slots_dir = slot::slots_dir(bucket_dir);
    if !slots_dir.exists() {
        return Ok(report);
    }

    let active = slot::read_active_slot(bucket_dir).unwrap_or(None);
    report.kept_active = active.clone();

    // Walk the slots directory once, classifying each entry.
    let mut entries: Vec<SlotEntry> = Vec::new();
    for entry in fs::read_dir(&slots_dir).map_err(BucketError::Io)? {
        let entry = entry.map_err(BucketError::Io)?;
        let meta = entry.metadata().map_err(BucketError::Io)?;
        if !meta.is_dir() {
            continue;
        }
        let Some(slot_id) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let manifest_path = slot::manifest_path(&entry.path());
        let state = match fs::read_to_string(&manifest_path) {
            Ok(text) => match SlotManifest::from_toml_str(&text) {
                Ok(m) => SlotClassification::Manifest(m.state),
                Err(_) => SlotClassification::UnparseableManifest,
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => SlotClassification::NoManifest,
            Err(e) => {
                // Read error other than "missing" — be conservative
                // and skip. A transient I/O glitch shouldn't trigger
                // a delete.
                warn!(
                    bucket_id = %bucket_id,
                    slot_id = %slot_id,
                    error = %e,
                    "gc: skipping slot whose manifest can't be read",
                );
                continue;
            }
        };
        entries.push(SlotEntry { slot_id, state });
    }

    // Identify the most-recent resumable. Slot ids are timestamp-
    // prefixed (ULID-shaped), so lex-sort = chronological — same
    // discipline as `find_resumable_slot`.
    entries.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    let resumable = entries
        .iter()
        .rev()
        .find(|e| {
            matches!(
                e.state,
                SlotClassification::Manifest(SlotState::Planning | SlotState::Building)
            )
        })
        .map(|e| e.slot_id.clone());
    report.kept_resumable = resumable.clone();

    // Decide per-slot.
    for entry in &entries {
        let is_active = active.as_deref() == Some(entry.slot_id.as_str());
        let is_resumable = resumable.as_deref() == Some(entry.slot_id.as_str());
        if is_active || is_resumable {
            continue;
        }

        let should_remove = match entry.state {
            // Older Planning/Building (a newer resumable supersedes it).
            SlotClassification::Manifest(SlotState::Planning | SlotState::Building) => true,
            // Failed builds.
            SlotClassification::Manifest(SlotState::Failed) => true,
            // Crashed before manifest landed.
            SlotClassification::NoManifest => true,
            SlotClassification::UnparseableManifest => true,
            // Ready / Archived: retention pass owns these.
            SlotClassification::Manifest(SlotState::Ready | SlotState::Archived) => false,
        };
        if !should_remove {
            continue;
        }

        let dir = slot::slot_dir(bucket_dir, &entry.slot_id);
        // Bytes are read from the manifest's recorded stat rather
        // than computed via a recursive walk. Walking a partial
        // build's slot dir blocks the boot path on tens of thousands
        // of tantivy segment files (this GC runs inside
        // `BucketRegistry::load`, before HTTP comes up). Manifest
        // stats are accurate as of build completion; for orphaned
        // mid-build slots they undercount, but `bytes_freed` is a
        // log/observability nicety, not a correctness signal.
        let bytes = match &entry.state {
            SlotClassification::Manifest(_) => fs::read_to_string(slot::manifest_path(&dir))
                .ok()
                .and_then(|t| SlotManifest::from_toml_str(&t).ok())
                .map(|m| m.stats.disk_size_bytes)
                .unwrap_or(0),
            SlotClassification::NoManifest | SlotClassification::UnparseableManifest => 0,
        };
        match fs::remove_dir_all(&dir) {
            Ok(()) => {
                debug!(
                    bucket_id = %bucket_id,
                    slot_id = %entry.slot_id,
                    state = ?entry.state,
                    bytes,
                    "gc: removed orphan slot dir",
                );
                report.removed.push(entry.slot_id.clone());
                report.bytes_freed += bytes;
            }
            Err(e) => {
                // A removal failure on a single slot shouldn't abort
                // the rest of the sweep. Log and keep going.
                warn!(
                    bucket_id = %bucket_id,
                    slot_id = %entry.slot_id,
                    error = %e,
                    "gc: failed to remove orphan slot dir",
                );
            }
        }
    }

    if !report.removed.is_empty() {
        info!(
            bucket_id = %bucket_id,
            removed = report.removed.len(),
            bytes_freed = report.bytes_freed,
            kept_active = ?report.kept_active,
            kept_resumable = ?report.kept_resumable,
            "gc: orphan slots reclaimed",
        );
    }
    Ok(report)
}

#[derive(Debug)]
struct SlotEntry {
    slot_id: String,
    state: SlotClassification,
}

#[derive(Debug, Clone, Copy)]
enum SlotClassification {
    Manifest(SlotState),
    NoManifest,
    UnparseableManifest,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use crate::knowledge::manifest::{
        ChunkerSnapshot, EmbedderSnapshot, ServingSnapshot, SlotStats, TokenizerSnapshot,
    };

    /// Synthetic slot dir with a written manifest. Returns the slot id.
    fn make_slot(bucket_dir: &Path, slot_id: &str, state: SlotState) -> String {
        let dir = slot::slot_dir(bucket_dir, slot_id);
        fs::create_dir_all(&dir).unwrap();
        let manifest = SlotManifest {
            slot_id: slot_id.to_string(),
            state,
            created_at: chrono::Utc::now(),
            build_started_at: None,
            build_completed_at: None,
            chunker_snapshot: ChunkerSnapshot {
                strategy: "token_based".into(),
                chunk_tokens: 500,
                overlap_tokens: 50,
                tokenizer: TokenizerSnapshot::Heuristic { chars_per_token: 4 },
            },
            embedder: EmbedderSnapshot {
                provider: "test".into(),
                model_id: "test-model".into(),
                dimension: 8,
            },
            sparse: None,
            serving: ServingSnapshot {
                mode: crate::knowledge::ServingMode::Disk,
                quantization: crate::knowledge::Quantization::F32,
            },
            stats: SlotStats::default(),
            lineage: None,
        };
        fs::write(
            slot::manifest_path(&dir),
            manifest.to_toml_string().unwrap(),
        )
        .unwrap();
        // Add a non-empty file so bytes_freed is exercised.
        fs::write(dir.join("placeholder.bin"), [0u8; 64]).unwrap();
        slot_id.to_string()
    }

    /// Same as `make_slot` but with a non-zero `disk_size_bytes` in
    /// the manifest's stats — for exercising the byte-counting path
    /// that GC now reads from the manifest rather than the disk.
    fn make_slot_with_disk_size(
        bucket_dir: &Path,
        slot_id: &str,
        state: SlotState,
        disk_size_bytes: u64,
    ) -> String {
        let dir = slot::slot_dir(bucket_dir, slot_id);
        fs::create_dir_all(&dir).unwrap();
        let manifest = SlotManifest {
            slot_id: slot_id.to_string(),
            state,
            created_at: chrono::Utc::now(),
            build_started_at: None,
            build_completed_at: None,
            chunker_snapshot: ChunkerSnapshot {
                strategy: "token_based".into(),
                chunk_tokens: 500,
                overlap_tokens: 50,
                tokenizer: TokenizerSnapshot::Heuristic { chars_per_token: 4 },
            },
            embedder: EmbedderSnapshot {
                provider: "test".into(),
                model_id: "test-model".into(),
                dimension: 8,
            },
            sparse: None,
            serving: ServingSnapshot {
                mode: crate::knowledge::ServingMode::Disk,
                quantization: crate::knowledge::Quantization::F32,
            },
            stats: SlotStats {
                disk_size_bytes,
                ..SlotStats::default()
            },
            lineage: None,
        };
        fs::write(
            slot::manifest_path(&dir),
            manifest.to_toml_string().unwrap(),
        )
        .unwrap();
        slot_id.to_string()
    }

    fn make_orphan_slot(bucket_dir: &Path, slot_id: &str) -> String {
        let dir = slot::slot_dir(bucket_dir, slot_id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("placeholder.bin"), [0u8; 32]).unwrap();
        slot_id.to_string()
    }

    fn set_active(bucket_dir: &Path, slot_id: &str) {
        slot::set_active_slot(bucket_dir, slot_id).unwrap();
    }

    /// Slot ids must lex-sort chronologically — use prefixes that
    /// match production's ULID shape (later prefix = "more recent").
    const SLOT_OLD: &str = "01HV0000000000000000000000";
    const SLOT_MID: &str = "01HV0000000000000000000001";
    const SLOT_NEW: &str = "01HV0000000000000000000002";

    #[test]
    fn no_slots_dir_is_a_clean_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert!(report.removed.is_empty());
        assert_eq!(report.bytes_freed, 0);
        assert_eq!(report.kept_active, None);
    }

    #[test]
    fn active_slot_is_never_removed() {
        let tmp = tempfile::tempdir().unwrap();
        make_slot(tmp.path(), SLOT_OLD, SlotState::Ready);
        set_active(tmp.path(), SLOT_OLD);

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert!(report.removed.is_empty());
        assert_eq!(report.kept_active.as_deref(), Some(SLOT_OLD));
        assert!(slot::slot_dir(tmp.path(), SLOT_OLD).exists());
    }

    #[test]
    fn most_recent_planning_slot_is_kept_as_resumable() {
        let tmp = tempfile::tempdir().unwrap();
        make_slot(tmp.path(), SLOT_OLD, SlotState::Planning);
        make_slot(tmp.path(), SLOT_NEW, SlotState::Planning);

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        // Newer Planning is preserved; older Planning is removed.
        assert_eq!(report.kept_resumable.as_deref(), Some(SLOT_NEW));
        assert_eq!(report.removed, vec![SLOT_OLD.to_string()]);
        assert!(slot::slot_dir(tmp.path(), SLOT_NEW).exists());
        assert!(!slot::slot_dir(tmp.path(), SLOT_OLD).exists());
    }

    #[test]
    fn most_recent_resumable_can_be_building_state_too() {
        // Mixed Planning + Building — the latest of either kind wins.
        let tmp = tempfile::tempdir().unwrap();
        make_slot(tmp.path(), SLOT_OLD, SlotState::Planning);
        make_slot(tmp.path(), SLOT_MID, SlotState::Building);
        make_slot(tmp.path(), SLOT_NEW, SlotState::Planning);

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert_eq!(report.kept_resumable.as_deref(), Some(SLOT_NEW));
        // Both older resumable-state slots removed.
        assert_eq!(
            report.removed,
            vec![SLOT_OLD.to_string(), SLOT_MID.to_string()]
        );
    }

    #[test]
    fn failed_slots_are_removed() {
        let tmp = tempfile::tempdir().unwrap();
        make_slot(tmp.path(), SLOT_OLD, SlotState::Failed);
        // Provide an active so the GC has something to keep — not
        // load-bearing for this test, just making it more realistic.
        make_slot(tmp.path(), SLOT_NEW, SlotState::Ready);
        set_active(tmp.path(), SLOT_NEW);

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert_eq!(report.removed, vec![SLOT_OLD.to_string()]);
        assert!(!slot::slot_dir(tmp.path(), SLOT_OLD).exists());
        assert!(slot::slot_dir(tmp.path(), SLOT_NEW).exists());
    }

    #[test]
    fn ready_and_archived_slots_are_left_alone() {
        // Ready / Archived are the retention sweep's territory, not
        // this GC's. Keep them, even if they're not the active slot.
        let tmp = tempfile::tempdir().unwrap();
        make_slot(tmp.path(), SLOT_OLD, SlotState::Archived);
        make_slot(tmp.path(), SLOT_MID, SlotState::Ready);
        make_slot(tmp.path(), SLOT_NEW, SlotState::Ready);
        set_active(tmp.path(), SLOT_NEW);

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert!(report.removed.is_empty());
        assert!(slot::slot_dir(tmp.path(), SLOT_OLD).exists());
        assert!(slot::slot_dir(tmp.path(), SLOT_MID).exists());
    }

    #[test]
    fn slot_dir_with_no_manifest_is_removed() {
        let tmp = tempfile::tempdir().unwrap();
        make_orphan_slot(tmp.path(), SLOT_OLD);
        // Active slot to ensure `kept_active` works alongside
        // orphan deletion in the same pass.
        make_slot(tmp.path(), SLOT_NEW, SlotState::Ready);
        set_active(tmp.path(), SLOT_NEW);

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert_eq!(report.removed, vec![SLOT_OLD.to_string()]);
    }

    #[test]
    fn unparseable_manifest_is_treated_as_orphan() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = slot::slot_dir(tmp.path(), SLOT_OLD);
        fs::create_dir_all(&dir).unwrap();
        fs::write(slot::manifest_path(&dir), "not valid toml at all !!!").unwrap();

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert_eq!(report.removed, vec![SLOT_OLD.to_string()]);
    }

    #[test]
    fn bytes_freed_uses_manifest_stat_not_walk() {
        // GC reads `bytes_freed` from the slot's manifest stats
        // (set at build time) rather than walking the disk —
        // walking blocked the boot path on tantivy-segment-heavy
        // buckets. A Failed slot with `disk_size_bytes = 12_345`
        // in its manifest should report exactly that, regardless
        // of the actual on-disk size.
        let tmp = tempfile::tempdir().unwrap();
        let slot_id = make_slot_with_disk_size(tmp.path(), SLOT_OLD, SlotState::Failed, 12_345);
        // Sanity: the slot exists and is in the failed state.
        assert_eq!(slot_id, SLOT_OLD);
        // Throw an extra MB of unaccounted data into the dir to
        // demonstrate the report doesn't depend on walking.
        fs::write(
            slot::slot_dir(tmp.path(), SLOT_OLD).join("ignored.bin"),
            vec![0u8; 1_000_000],
        )
        .unwrap();

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert_eq!(report.removed, vec![SLOT_OLD.to_string()]);
        assert_eq!(
            report.bytes_freed, 12_345,
            "should reflect the manifest stat, not the actual on-disk size",
        );
    }

    #[test]
    fn bytes_freed_zero_for_orphan_without_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = slot::slot_dir(tmp.path(), SLOT_OLD);
        fs::create_dir_all(dir.join("nested")).unwrap();
        fs::write(dir.join("nested").join("a.bin"), [0u8; 100]).unwrap();
        fs::write(dir.join("b.bin"), [0u8; 200]).unwrap();

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert_eq!(report.removed, vec![SLOT_OLD.to_string()]);
        // No manifest = no recorded size = 0. Acceptable since
        // bytes_freed is observability, not a correctness signal.
        assert_eq!(report.bytes_freed, 0);
    }

    #[test]
    fn active_slot_in_resumable_state_is_still_kept() {
        // Edge case: the `active` pointer points at a slot that
        // somehow still has Building state (a build that promoted
        // mid-flight, perhaps). Active wins regardless of state.
        let tmp = tempfile::tempdir().unwrap();
        make_slot(tmp.path(), SLOT_OLD, SlotState::Building);
        set_active(tmp.path(), SLOT_OLD);

        let report = gc_orphan_slots(tmp.path(), "test_bucket").unwrap();
        assert!(report.removed.is_empty());
        assert_eq!(report.kept_active.as_deref(), Some(SLOT_OLD));
        // The active slot also satisfies the resumable predicate, so
        // kept_resumable points at the same id — that's fine; either
        // skip path would prevent its removal.
        assert_eq!(report.kept_resumable.as_deref(), Some(SLOT_OLD));
    }
}
