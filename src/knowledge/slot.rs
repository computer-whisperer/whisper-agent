//! Slot directory operations: id generation, layout helpers, active-slot
//! pointer management.
//!
//! A slot's id is a sortable string (timestamp-prefixed + random suffix);
//! the directory `<bucket>/slots/<slot_id>/` holds the slot's persistent
//! state. The active slot for a bucket is recorded in
//! `<bucket>/slots/active` — a tiny text file containing the slot id —
//! rather than a symlink, for cross-platform simplicity.
//!
//! Promotion is "write the new id to a temp file, then rename" —
//! `rename(2)` is atomic on POSIX, so concurrent readers see either the
//! old or new pointer, never a partial write.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use super::types::{BucketError, SlotId};

/// File and directory names. Single source of truth so disk-layout
/// changes don't require touching a dozen call sites.
pub const ACTIVE_FILE: &str = "active";
pub const ACTIVE_TMP_FILE: &str = "active.tmp";
pub const SLOTS_DIR: &str = "slots";
pub const SOURCE_DIR: &str = "source";
pub const BUCKET_TOML: &str = "bucket.toml";
pub const MANIFEST_TOML: &str = "manifest.toml";
pub const CHUNKS_BIN: &str = "chunks.bin";
pub const CHUNKS_IDX: &str = "chunks.idx";
pub const VECTORS_BIN: &str = "vectors.bin";
pub const VECTORS_IDX: &str = "vectors.idx";
pub const INDEX_TANTIVY: &str = "index.tantivy";
pub const BUILD_STATE: &str = "build.state";
pub const BUILD_LOG: &str = "build.log";

pub fn slots_dir(bucket_root: &Path) -> PathBuf {
    bucket_root.join(SLOTS_DIR)
}

pub fn slot_dir(bucket_root: &Path, slot_id: &str) -> PathBuf {
    slots_dir(bucket_root).join(slot_id)
}

pub fn bucket_toml_path(bucket_root: &Path) -> PathBuf {
    bucket_root.join(BUCKET_TOML)
}

pub fn manifest_path(slot_dir: &Path) -> PathBuf {
    slot_dir.join(MANIFEST_TOML)
}

pub fn chunks_bin_path(slot_dir: &Path) -> PathBuf {
    slot_dir.join(CHUNKS_BIN)
}

pub fn chunks_idx_path(slot_dir: &Path) -> PathBuf {
    slot_dir.join(CHUNKS_IDX)
}

pub fn vectors_bin_path(slot_dir: &Path) -> PathBuf {
    slot_dir.join(VECTORS_BIN)
}

pub fn vectors_idx_path(slot_dir: &Path) -> PathBuf {
    slot_dir.join(VECTORS_IDX)
}

pub fn tantivy_dir(slot_dir: &Path) -> PathBuf {
    slot_dir.join(INDEX_TANTIVY)
}

pub fn active_pointer_path(bucket_root: &Path) -> PathBuf {
    slots_dir(bucket_root).join(ACTIVE_FILE)
}

fn active_tmp_path(bucket_root: &Path) -> PathBuf {
    slots_dir(bucket_root).join(ACTIVE_TMP_FILE)
}

/// Generate a fresh slot id. Format: `<unix_millis_hex>-<rand_hex>`,
/// 13 + 1 + 16 = 30 ASCII chars. Sortable lexicographically by creation
/// time; within the same millisecond, the random suffix breaks ties.
///
/// Not a real ULID — those use Crockford base32 and a structured
/// 128-bit layout. We don't need that compat for v1; a typed
/// [`SlotId`](super::SlotId) wrapper can graduate to ULID later if
/// useful.
pub fn generate_slot_id() -> SlotId {
    let now_millis = chrono::Utc::now().timestamp_millis() as u64;
    let rand_bits: u64 = rand::random();
    format!("{now_millis:013x}-{rand_bits:016x}")
}

/// Read the active-slot id, returning `None` if no slot has been
/// promoted yet (file missing or empty), or `Err` on actual IO failure.
pub fn read_active_slot(bucket_root: &Path) -> Result<Option<SlotId>, BucketError> {
    let path = active_pointer_path(bucket_root);
    match File::open(&path) {
        Ok(mut f) => {
            let mut s = String::new();
            f.read_to_string(&mut s).map_err(BucketError::Io)?;
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(BucketError::Io(e)),
    }
}

/// Atomically set the active-slot pointer. Writes to a temp file under
/// the same directory, syncs, then renames over the canonical name.
pub fn set_active_slot(bucket_root: &Path, slot_id: &str) -> Result<(), BucketError> {
    let tmp = active_tmp_path(bucket_root);
    let final_path = active_pointer_path(bucket_root);
    {
        let mut f = File::create(&tmp).map_err(BucketError::Io)?;
        f.write_all(slot_id.as_bytes()).map_err(BucketError::Io)?;
        f.write_all(b"\n").map_err(BucketError::Io)?;
        f.sync_all().map_err(BucketError::Io)?;
    }
    fs::rename(&tmp, &final_path).map_err(BucketError::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_id_is_unique_within_a_run() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            let id = generate_slot_id();
            assert!(seen.insert(id), "slot id collision");
        }
    }

    #[test]
    fn slot_ids_sort_by_creation_time() {
        let earlier = generate_slot_id();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let later = generate_slot_id();
        assert!(earlier < later, "{earlier} should sort before {later}");
    }

    #[test]
    fn slot_id_is_thirty_ascii_chars() {
        let id = generate_slot_id();
        assert_eq!(id.len(), 30);
        assert!(id.is_ascii());
        assert!(id.chars().nth(13) == Some('-'));
    }

    #[test]
    fn read_active_returns_none_on_fresh_bucket() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(slots_dir(tmp.path())).unwrap();
        assert!(read_active_slot(tmp.path()).unwrap().is_none());
    }

    #[test]
    fn set_active_then_read_roundtrips() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(slots_dir(tmp.path())).unwrap();
        let id = generate_slot_id();
        set_active_slot(tmp.path(), &id).unwrap();
        assert_eq!(
            read_active_slot(tmp.path()).unwrap().as_deref(),
            Some(id.as_str())
        );
    }

    #[test]
    fn set_active_overwrites_previous_value() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(slots_dir(tmp.path())).unwrap();
        let first = generate_slot_id();
        set_active_slot(tmp.path(), &first).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(2));
        let second = generate_slot_id();
        set_active_slot(tmp.path(), &second).unwrap();

        assert_eq!(
            read_active_slot(tmp.path()).unwrap().as_deref(),
            Some(second.as_str())
        );
    }

    #[test]
    fn read_active_treats_empty_file_as_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(slots_dir(tmp.path())).unwrap();
        File::create(active_pointer_path(tmp.path())).unwrap(); // empty
        assert!(read_active_slot(tmp.path()).unwrap().is_none());
    }
}
