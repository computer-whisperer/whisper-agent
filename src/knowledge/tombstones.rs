//! Tombstone storage for the slot's mutation layer.
//!
//! `tombstones.bin` is an append-only file of fixed-size 32-byte
//! [`ChunkId`] records. Each [`Bucket::tombstone`](super::Bucket::tombstone)
//! call appends + fsyncs. In RAM we keep a sorted, deduplicated
//! `Vec<ChunkId>`, built once at slot open and updated on each append;
//! query-path filtering hits the in-RAM view via binary search.
//!
//! Tombstone counts stay small — compaction triggers at 10% of total
//! chunks per `docs/design_knowledge_db.md` § "Compaction mechanics" —
//! so sort-on-append + linear-merge is adequate for steady state.

use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Mutex, RwLock};

use super::types::ChunkId;

/// Bytes per on-disk record. Equals `size_of::<ChunkId>()`; the file is
/// a tightly-packed array of these with no header.
const RECORD_SIZE: usize = 32;

#[derive(Debug)]
pub struct Tombstones {
    /// Sorted, deduplicated set of tombstoned ids. Read-locked by
    /// queries (concurrent), write-locked on append (rare).
    sorted: RwLock<Vec<ChunkId>>,
    /// Append-mode file handle held open across calls. The `Mutex`
    /// serializes appends; `O_APPEND` makes each `write_all` atomic at
    /// the end of file even if multiple writers ever raced.
    file: Mutex<std::fs::File>,
}

impl Tombstones {
    /// Open the tombstone file at `path`, creating it if absent. Reads
    /// all existing 32-byte records into a sorted, deduped `Vec`. A
    /// trailing partial record (file size not a multiple of 32) signals
    /// a torn write; we truncate it rather than failing the open.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)?;

        let len = file.metadata()?.len();
        let usable_len = (len / RECORD_SIZE as u64) * RECORD_SIZE as u64;
        if usable_len != len {
            tracing::warn!(
                file = %path.display(),
                file_len = len,
                usable_len,
                "tombstone file has a partial trailing record; truncating",
            );
            file.set_len(usable_len)?;
        }

        let n_records = (usable_len / RECORD_SIZE as u64) as usize;
        let mut buf = Vec::with_capacity(n_records * RECORD_SIZE);
        file.seek(SeekFrom::Start(0))?;
        file.read_to_end(&mut buf)?;

        let mut sorted = Vec::with_capacity(n_records);
        for record in buf.chunks_exact(RECORD_SIZE) {
            let mut id = [0u8; 32];
            id.copy_from_slice(record);
            sorted.push(ChunkId(id));
        }
        sorted.sort_unstable();
        sorted.dedup();

        Ok(Self {
            sorted: RwLock::new(sorted),
            file: Mutex::new(file),
        })
    }

    /// Append `ids` to the tombstone file (with fsync) and merge them
    /// into the in-RAM sorted view. Returns once the data is durable.
    /// Empty input is a no-op (no fsync).
    pub fn append(&self, ids: &[ChunkId]) -> io::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        // Disk first: a caller that observes `Ok(())` is guaranteed the
        // tombstone survives a crash. Build the byte buffer once outside
        // the lock and let `O_APPEND` place it at EOF atomically.
        let mut buf = Vec::with_capacity(ids.len() * RECORD_SIZE);
        for id in ids {
            buf.extend_from_slice(&id.0);
        }
        {
            let mut file = self.file.lock().expect("Tombstones.file lock poisoned");
            file.write_all(&buf)?;
            file.sync_all()?;
        }

        // RAM second: extend + sort + dedup. Sort cost is `O(N log N)`
        // over the whole tombstone set; tolerable because tombstone
        // counts stay bounded by the compaction trigger.
        let mut sorted = self
            .sorted
            .write()
            .expect("Tombstones.sorted lock poisoned");
        sorted.extend_from_slice(ids);
        sorted.sort_unstable();
        sorted.dedup();
        Ok(())
    }

    /// `O(log N)` membership test against the in-RAM sorted set.
    pub fn contains(&self, id: ChunkId) -> bool {
        let sorted = self.sorted.read().expect("Tombstones.sorted lock poisoned");
        sorted.binary_search(&id).is_ok()
    }

    /// Number of unique tombstoned ids.
    pub fn len(&self) -> usize {
        self.sorted
            .read()
            .expect("Tombstones.sorted lock poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> ChunkId {
        ChunkId([byte; 32])
    }

    #[test]
    fn fresh_open_creates_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tombstones.bin");
        let t = Tombstones::open(&path).unwrap();
        assert!(t.is_empty());
        assert!(!t.contains(id(0xAA)));
        assert!(path.exists());
    }

    #[test]
    fn append_then_contains() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tombstones.bin");
        let t = Tombstones::open(&path).unwrap();
        t.append(&[id(1), id(2), id(3)]).unwrap();
        assert_eq!(t.len(), 3);
        assert!(t.contains(id(1)));
        assert!(t.contains(id(2)));
        assert!(t.contains(id(3)));
        assert!(!t.contains(id(4)));
    }

    #[test]
    fn append_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tombstones.bin");
        {
            let t = Tombstones::open(&path).unwrap();
            t.append(&[id(7), id(11), id(2)]).unwrap();
        }
        let t2 = Tombstones::open(&path).unwrap();
        assert_eq!(t2.len(), 3);
        assert!(t2.contains(id(2)));
        assert!(t2.contains(id(7)));
        assert!(t2.contains(id(11)));
    }

    #[test]
    fn duplicates_are_deduped_within_one_call_and_across_calls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tombstones.bin");
        let t = Tombstones::open(&path).unwrap();
        t.append(&[id(1), id(1), id(2)]).unwrap();
        t.append(&[id(2), id(3)]).unwrap();
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn empty_append_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tombstones.bin");
        let t = Tombstones::open(&path).unwrap();
        t.append(&[]).unwrap();
        assert!(t.is_empty());
        // File size still 0.
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
    }

    #[test]
    fn torn_trailing_record_is_truncated_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tombstones.bin");
        // Write one full record + 5 trailing junk bytes.
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[0x42; 32]).unwrap();
            f.write_all(&[0xFF; 5]).unwrap();
        }
        let t = Tombstones::open(&path).unwrap();
        assert_eq!(t.len(), 1);
        assert!(t.contains(id(0x42)));
        // File got truncated to the clean boundary.
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 32);
    }

    #[test]
    fn many_appends_stay_sorted_and_correct() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tombstones.bin");
        let t = Tombstones::open(&path).unwrap();
        // Insert in scrambled order.
        t.append(&[id(50), id(10), id(30)]).unwrap();
        t.append(&[id(20)]).unwrap();
        t.append(&[id(40), id(5)]).unwrap();
        for v in [5u8, 10, 20, 30, 40, 50] {
            assert!(t.contains(id(v)), "missing {v}");
        }
        assert!(!t.contains(id(0)));
        assert!(!t.contains(id(60)));
        assert_eq!(t.len(), 6);
    }
}
