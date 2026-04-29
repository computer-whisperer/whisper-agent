//! Persistent chunk-text storage for a slot.
//!
//! Two files per slot:
//! - `chunks.bin` — append-only sequence of length-prefixed records
//!   carrying chunk text + source-ref metadata.
//! - `chunks.idx` — sorted `(chunk_id, file_offset)` entries, loaded into
//!   an in-memory HashMap on slot open. Random-access fetch is O(1) in
//!   the index plus one disk read per chunk.
//!
//! The split lets us stream chunks during build (append-only is
//! resumable on crash) while keeping query-time lookups fast without
//! scanning the bin file at slot open.
//!
//! Reader storage modes for finalized slots:
//! - `LoadMode::Mmap` — file is mmap'd; concurrent fetches don't
//!   serialize on a Mutex, and the working set lives in the OS page
//!   cache. The right default for disk-backed serving.
//! - `LoadMode::Eager` — file is read in full into a `Box<[u8]>` at
//!   open time. Predictable latency, no page-cache eviction.
//!
//! Live-build readers use [`ChunkStoreReader::open_with_shared_index`]
//! and stay file-based — `chunks.bin` is still being appended to, so a
//! fixed-size mmap snapshot would miss the writer's new records.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use memmap2::Mmap;

use super::types::{Chunk, ChunkId, SourceRef};
pub use super::vectors::LoadMode;

/// Shared id→offset index. The writer mutates it on `append`; readers
/// take read locks on `fetch`. Cloning the [`Arc`] is what gives a
/// live-view reader during a build — no map copy required.
pub type SharedChunkIndex = Arc<RwLock<HashMap<ChunkId, u64>>>;

/// Format-version byte at the head of every record. Bump if the record
/// layout changes; the reader rejects unknown versions explicitly.
const RECORD_VERSION: u8 = 1;

// chunks.bin record format:
//   [4 bytes] body_length (u32 LE) — bytes after this prefix
//   [body_length bytes] body:
//     [1 byte]  version (= RECORD_VERSION)
//     [1 byte]  has_locator flag (0 or 1)
//     [4 bytes] source_id_length (u32 LE)
//     [source_id_length bytes] source_id (UTF-8)
//     if has_locator:
//       [4 bytes] locator_length (u32 LE)
//       [locator_length bytes] locator (UTF-8)
//     [4 bytes] text_length (u32 LE)
//     [text_length bytes] text (UTF-8)
//
// The leading 4-byte prefix lets readers fetch a whole record in two
// pread/read calls (length, then body) without parsing inner fields.
//
// chunks.idx layout:
//   [8 bytes] entry count (u64 LE)
//   [count × 40 bytes] each entry: [32 bytes chunk_id] [8 bytes offset (u64 LE)]
//
// Entries are sorted ascending by chunk_id for deterministic file shape;
// the in-memory HashMap doesn't depend on the order, but stable on-disk
// output makes byte-for-byte comparison useful in tests and audits.

/// Streams chunks into `chunks.bin` during a build, maintains the
/// shared id→offset index, and writes the sorted `chunks.idx` on
/// [`finalize`](Self::finalize).
///
/// Index visibility is gated on [`Self::flush`]: `append` stages new
/// id→offset entries into a `pending` map and only merges them into
/// the shared [`SharedChunkIndex`] after the bin is fsynced. This is
/// what lets the build pipeline batch fsyncs across many `append`s
/// without ever exposing a live reader to an offset whose bytes
/// haven't actually landed on disk — a fetch through the live reader
/// either sees the chunk fully (post-flush) or not at all (still
/// pending).
#[derive(Debug)]
pub struct ChunkStoreWriter {
    bin: BufWriter<File>,
    idx_path: PathBuf,
    index: SharedChunkIndex,
    /// id→offset entries written since the last flush. Promoted into
    /// `index` only after `bin.sync_data()` succeeds in [`Self::flush`].
    pending: HashMap<ChunkId, u64>,
    cursor: u64,
}

impl ChunkStoreWriter {
    pub fn create(bin_path: &Path, idx_path: &Path) -> io::Result<Self> {
        let file = File::create(bin_path)?;
        Ok(Self {
            bin: BufWriter::new(file),
            idx_path: idx_path.to_path_buf(),
            index: Arc::new(RwLock::new(HashMap::new())),
            pending: HashMap::new(),
            cursor: 0,
        })
    }

    /// Reopen an existing `chunks.bin` for resumed appending. Truncates
    /// the file to `record_offsets.last()` (the byte offset just past
    /// the last record we want to keep), seeds the shared index from
    /// `chunk_ids` paired with `record_offsets[..len-1]`, and opens
    /// the file in append mode for further writes.
    ///
    /// Caller invariants:
    /// - `chunk_ids.len() == record_offsets.len() - 1` (one offset per
    ///   record + one trailing EOF offset).
    /// - The chunk_ids are in the same insertion order they were
    ///   originally appended.
    /// - The bin file already contains records aligned with
    ///   `record_offsets` (typically obtained from
    ///   [`scan_record_offsets`]).
    pub fn open_resume(
        bin_path: &Path,
        idx_path: &Path,
        chunk_ids: &[ChunkId],
        record_offsets: &[u64],
    ) -> io::Result<Self> {
        if record_offsets.len() != chunk_ids.len() + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "open_resume: expected {} offsets ({}+1), got {}",
                    chunk_ids.len() + 1,
                    chunk_ids.len(),
                    record_offsets.len(),
                ),
            ));
        }
        let cursor = *record_offsets.last().unwrap();
        // Truncate any trailing partial record, then open in append
        // mode. set_len does both shrink and grow as needed; we only
        // shrink here.
        let truncate = OpenOptions::new().write(true).open(bin_path)?;
        truncate.set_len(cursor)?;
        drop(truncate);

        let file = OpenOptions::new().append(true).open(bin_path)?;
        let mut map = HashMap::with_capacity(chunk_ids.len());
        for (id, off) in chunk_ids
            .iter()
            .copied()
            .zip(record_offsets[..chunk_ids.len()].iter().copied())
        {
            map.insert(id, off);
        }
        Ok(Self {
            bin: BufWriter::new(file),
            idx_path: idx_path.to_path_buf(),
            index: Arc::new(RwLock::new(map)),
            pending: HashMap::new(),
            cursor,
        })
    }

    /// Open an existing `chunks.bin` / `chunks.idx` pair for appending,
    /// or create them fresh if absent. The mutation path uses this to
    /// re-engage the delta writer across `Bucket::insert` calls.
    ///
    /// Source of truth for the in-memory index is `chunks.idx`; we
    /// don't re-scan the bin file. Each call must `finalize()` so the
    /// next call sees an up-to-date `chunks.idx` — same contract as
    /// the build path's per-batch durability barrier.
    pub fn create_or_open_append(bin_path: &Path, idx_path: &Path) -> io::Result<Self> {
        if !bin_path.exists() {
            return Self::create(bin_path, idx_path);
        }
        let cursor = std::fs::metadata(bin_path)?.len();
        let map = if idx_path.exists() {
            read_idx(idx_path)?
        } else {
            HashMap::new()
        };
        let file = OpenOptions::new().append(true).open(bin_path)?;
        Ok(Self {
            bin: BufWriter::new(file),
            idx_path: idx_path.to_path_buf(),
            index: Arc::new(RwLock::new(map)),
            pending: HashMap::new(),
            cursor,
        })
    }

    /// Append a chunk and stage its offset for the next flush. Returns
    /// the offset where the record was written. The id is *not* yet
    /// visible to live readers — see [`Self::flush`].
    pub fn append(
        &mut self,
        chunk_id: ChunkId,
        source_ref: &SourceRef,
        text: &str,
    ) -> io::Result<u64> {
        let offset = self.cursor;
        let written = write_record(&mut self.bin, source_ref, text)?;
        self.cursor += written;
        self.pending.insert(chunk_id, offset);
        Ok(offset)
    }

    /// Flush buffered writes to the underlying file, `sync_data`, and
    /// then promote staged `append` entries into the shared index so
    /// live readers can find them. Holding the bytes-on-disk barrier
    /// before exposing the offsets is what keeps a `Live` reader from
    /// ever seeing an index entry whose record bytes are still in the
    /// `BufWriter` (or the kernel page cache).
    pub fn flush(&mut self) -> io::Result<()> {
        self.bin.flush()?;
        self.bin.get_ref().sync_data()?;
        if !self.pending.is_empty() {
            let mut idx = self
                .index
                .write()
                .expect("ChunkStoreWriter index lock poisoned");
            for (id, off) in self.pending.drain() {
                idx.insert(id, off);
            }
        }
        Ok(())
    }

    /// Clone the shared index handle for a live-view reader. Subsequent
    /// `append` calls grow the same map; the reader sees them as soon
    /// as the write lock is released. No per-batch copy.
    pub fn share_index(&self) -> SharedChunkIndex {
        Arc::clone(&self.index)
    }

    pub fn len(&self) -> usize {
        self.index
            .read()
            .expect("ChunkStoreWriter index lock poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.index
            .read()
            .expect("ChunkStoreWriter index lock poisoned")
            .is_empty()
    }

    /// Flush `chunks.bin` and write `chunks.idx`. Returns the number of
    /// chunks committed.
    pub fn finalize(mut self) -> io::Result<usize> {
        self.bin.flush()?;
        // Drop the BufWriter so the underlying file is closed before we
        // open the next file — helps on Windows; harmless on Unix.
        drop(self.bin);

        // Promote any append-staged entries into the shared index so
        // they land in `chunks.idx` even if the caller never invoked
        // `flush()` between the last append and finalize.
        if !self.pending.is_empty() {
            let mut idx = self
                .index
                .write()
                .expect("ChunkStoreWriter index lock poisoned");
            for (id, off) in self.pending.drain() {
                idx.insert(id, off);
            }
        }

        // Collect into a Vec to sort. Live readers may still be holding
        // an Arc clone; we don't mutate the map here, just iterate.
        let map = self
            .index
            .read()
            .expect("ChunkStoreWriter index lock poisoned");
        let mut entries: Vec<(ChunkId, u64)> = map.iter().map(|(k, v)| (*k, *v)).collect();
        drop(map);
        entries.sort_by_key(|(id, _)| *id);
        let mut idx = BufWriter::new(File::create(&self.idx_path)?);
        idx.write_all(&(entries.len() as u64).to_le_bytes())?;
        for (id, offset) in &entries {
            idx.write_all(&id.0)?;
            idx.write_all(&offset.to_le_bytes())?;
        }
        idx.flush()?;
        Ok(entries.len())
    }
}

fn write_record<W: Write>(w: &mut W, source_ref: &SourceRef, text: &str) -> io::Result<u64> {
    let source_id = source_ref.source_id.as_bytes();
    let locator_bytes = source_ref.locator.as_deref().unwrap_or("").as_bytes();
    let text_bytes = text.as_bytes();
    let has_locator: u8 = u8::from(source_ref.locator.is_some());

    let body_len: usize = 1 // version byte
        + 1 // has_locator flag
        + 4 + source_id.len()
        + if has_locator == 1 { 4 + locator_bytes.len() } else { 0 }
        + 4 + text_bytes.len();

    w.write_all(&(body_len as u32).to_le_bytes())?;
    w.write_all(&[RECORD_VERSION, has_locator])?;
    w.write_all(&(source_id.len() as u32).to_le_bytes())?;
    w.write_all(source_id)?;
    if has_locator == 1 {
        w.write_all(&(locator_bytes.len() as u32).to_le_bytes())?;
        w.write_all(locator_bytes)?;
    }
    w.write_all(&(text_bytes.len() as u32).to_le_bytes())?;
    w.write_all(text_bytes)?;

    Ok(4 + body_len as u64)
}

/// Random-access reader for `chunks.bin` / `chunks.idx`.
///
/// `Send + Sync`. The id→offset index is held behind an [`Arc`] +
/// [`RwLock`] so a reader constructed via
/// [`Self::open_with_shared_index`] sees writer appends live (without
/// a per-batch index copy). For fully-built slots the reader owns its
/// own index, but the type stays the same for uniformity.
///
/// Backed by either an mmap (`LoadMode::Mmap`), an eagerly-loaded
/// buffer (`LoadMode::Eager`), or a `Mutex<File>` for live-build
/// readers (where `chunks.bin` is still being appended to). The first
/// two modes serve concurrent fetches without serialization; the third
/// keeps the existing seek-based behavior.
pub struct ChunkStoreReader {
    storage: ChunkStorage,
    index: SharedChunkIndex,
}

enum ChunkStorage {
    /// Finalized slot, mmap'd. Disk-backed serving mode.
    Mapped(Mmap),
    /// Finalized slot, eagerly-loaded. RAM-resident serving mode.
    Eager(Box<[u8]>),
    /// Live-build reader — file is still being appended by the
    /// writer. Concurrent fetches serialize through this Mutex; tiny
    /// cost compared to the build's overall wall-clock.
    Live(Mutex<File>),
}

impl std::fmt::Debug for ChunkStoreReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.storage {
            ChunkStorage::Mapped(_) => "Mapped",
            ChunkStorage::Eager(_) => "Eager",
            ChunkStorage::Live(_) => "Live",
        };
        f.debug_struct("ChunkStoreReader")
            .field("storage", &kind)
            .finish_non_exhaustive()
    }
}

impl ChunkStoreReader {
    /// Open a finalized slot's chunk store with the default mmap mode.
    /// Equivalent to [`Self::open_with_mode`] with [`LoadMode::Mmap`].
    pub fn open(bin_path: &Path, idx_path: &Path) -> io::Result<Self> {
        Self::open_with_mode(bin_path, idx_path, LoadMode::Mmap)
    }

    /// Open a finalized slot's chunk store with an explicit load mode.
    /// Production slot-open paths translate `manifest.serving.mode`
    /// into a [`LoadMode`] and call this.
    pub fn open_with_mode(bin_path: &Path, idx_path: &Path, mode: LoadMode) -> io::Result<Self> {
        let storage = match mode {
            LoadMode::Mmap => {
                let file = File::open(bin_path)?;
                // SAFETY: same contract as VectorStoreReader — the
                // `Mmap` owns its file handle for its lifetime, and
                // finalized slot files don't get mutated while a slot
                // is active.
                let mmap = unsafe { Mmap::map(&file)? };
                ChunkStorage::Mapped(mmap)
            }
            LoadMode::Eager => {
                let bytes = std::fs::read(bin_path)?;
                ChunkStorage::Eager(bytes.into_boxed_slice())
            }
        };
        let index = read_idx(idx_path)?;
        Ok(Self {
            storage,
            index: Arc::new(RwLock::new(index)),
        })
    }

    /// Open a live-view reader over `chunks.bin` sharing an
    /// in-progress writer's index. Used during a build when
    /// `chunks.idx` doesn't yet exist; the build path obtains the
    /// shared index via [`ChunkStoreWriter::share_index`] and the
    /// reader sees subsequent appends through the same Arc.
    ///
    /// Stays file-based (no mmap): `chunks.bin` is still being
    /// appended to, and a mmap snapshot would not see new records.
    pub fn open_with_shared_index(bin_path: &Path, index: SharedChunkIndex) -> io::Result<Self> {
        let bin = File::open(bin_path)?;
        Ok(Self {
            storage: ChunkStorage::Live(Mutex::new(bin)),
            index,
        })
    }

    pub fn len(&self) -> usize {
        self.index
            .read()
            .expect("ChunkStoreReader index lock poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.index
            .read()
            .expect("ChunkStoreReader index lock poisoned")
            .is_empty()
    }

    pub fn contains(&self, chunk_id: ChunkId) -> bool {
        self.index
            .read()
            .expect("ChunkStoreReader index lock poisoned")
            .contains_key(&chunk_id)
    }

    /// Fetch a chunk by id. Returns `Ok(None)` when the chunk isn't in
    /// the index; `Err` for IO or format errors during the read.
    pub fn fetch(&self, chunk_id: ChunkId) -> io::Result<Option<Chunk>> {
        // Briefly hold the read lock just for the lookup; release
        // before any I/O so concurrent appends aren't blocked on us.
        let offset = {
            let map = self
                .index
                .read()
                .expect("ChunkStoreReader index lock poisoned");
            map.get(&chunk_id).copied()
        };
        let Some(offset) = offset else {
            return Ok(None);
        };

        let body = match &self.storage {
            ChunkStorage::Mapped(m) => fetch_record_from_bytes(m, offset)?,
            ChunkStorage::Eager(v) => fetch_record_from_bytes(v, offset)?,
            ChunkStorage::Live(mu) => {
                let mut bin = mu.lock().expect("chunks.bin mutex poisoned");
                bin.seek(SeekFrom::Start(offset))?;
                let mut len_bytes = [0u8; 4];
                bin.read_exact(&mut len_bytes)?;
                let body_len = u32::from_le_bytes(len_bytes) as usize;
                let mut body = vec![0u8; body_len];
                bin.read_exact(&mut body)?;
                body
            }
        };

        parse_record(chunk_id, &body).map(Some)
    }

    /// Walk every record paired with its `chunk_id`, in `chunks.bin`
    /// insertion order. Returns just `(ChunkId, source_id)` — locator
    /// and chunk text are skipped on parse, so this is materially
    /// cheaper than calling [`fetch`](Self::fetch) per record at slot
    /// open. Used by [`SourceIndex`](super::source_index::SourceIndex)
    /// to seed the in-memory `source_id → Vec<ChunkId>` map.
    pub fn iter_chunk_source_ids(&self) -> io::Result<Vec<(ChunkId, String)>> {
        // Invert idx: offset → chunk_id, sorted ascending by offset so
        // the bin is read sequentially (mmap- and page-cache-friendly).
        let inverse: Vec<(u64, ChunkId)> = {
            let map = self
                .index
                .read()
                .expect("ChunkStoreReader index lock poisoned");
            let mut v: Vec<(u64, ChunkId)> = map.iter().map(|(k, v)| (*v, *k)).collect();
            v.sort_by_key(|(o, _)| *o);
            v
        };

        let mut out: Vec<(ChunkId, String)> = Vec::with_capacity(inverse.len());
        for (offset, chunk_id) in inverse {
            let body = match &self.storage {
                ChunkStorage::Mapped(m) => fetch_record_from_bytes(m, offset)?,
                ChunkStorage::Eager(v) => fetch_record_from_bytes(v, offset)?,
                ChunkStorage::Live(mu) => {
                    let mut bin = mu.lock().expect("chunks.bin mutex poisoned");
                    bin.seek(SeekFrom::Start(offset))?;
                    let mut len_bytes = [0u8; 4];
                    bin.read_exact(&mut len_bytes)?;
                    let body_len = u32::from_le_bytes(len_bytes) as usize;
                    let mut body = vec![0u8; body_len];
                    bin.read_exact(&mut body)?;
                    body
                }
            };
            out.push((chunk_id, parse_source_id_only(&body)?));
        }
        Ok(out)
    }
}

/// Pull one record's body out of a `&[u8]` view of `chunks.bin` at the
/// given offset. The body's `body_len` 4-byte prefix is read from the
/// slice, then the body bytes are copied into a `Vec<u8>` for parsing.
/// (Avoiding the copy would require changing `parse_record` to work
/// against a borrowed slice — a small win, defer until measurements
/// show it.)
fn fetch_record_from_bytes(bytes: &[u8], offset: u64) -> io::Result<Vec<u8>> {
    let off = offset as usize;
    if off + 4 > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "chunks.bin: offset {offset} reads past end ({} bytes)",
                bytes.len(),
            ),
        ));
    }
    let body_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
    let body_start = off + 4;
    let body_end = body_start + body_len;
    if body_end > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "chunks.bin: record body at {offset} reads past end ({body_end} > {})",
                bytes.len(),
            ),
        ));
    }
    Ok(bytes[body_start..body_end].to_vec())
}

/// Walk `chunks.bin` and return the byte offsets of every record's
/// start, in insertion order, with a trailing offset for "just past the
/// last record" (i.e. EOF on a clean file).
///
/// Resume code uses this to truncate the bin to a known boundary
/// without parsing record bodies — only the 4-byte length prefix of
/// each record is consumed. Cost is one sequential read of the file
/// minus the body bytes.
///
/// Stops at the first short read after a record boundary (which is the
/// normal EOF condition); a *partial* trailing record (fewer bytes
/// than its length prefix declares) is reported via
/// `io::ErrorKind::UnexpectedEof` so the resume code can decide to
/// truncate it.
pub fn scan_record_offsets(bin_path: &Path) -> io::Result<Vec<u64>> {
    let mut file = BufReader::new(File::open(bin_path)?);
    let mut offsets = Vec::new();
    let mut cursor: u64 = 0;
    loop {
        offsets.push(cursor);
        let mut len_bytes = [0u8; 4];
        match file.read_exact(&mut len_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // Clean EOF at the start of a record. The last entry
                // we just pushed is the "just past last record" offset.
                return Ok(offsets);
            }
            Err(e) => return Err(e),
        }
        let body_len = u32::from_le_bytes(len_bytes) as u64;
        // Skip the body. seek_relative is a BufReader-friendly forward-only seek.
        file.seek_relative(body_len as i64)?;
        cursor += 4 + body_len;
    }
}

fn read_idx(path: &Path) -> io::Result<HashMap<ChunkId, u64>> {
    let mut idx = BufReader::new(File::open(path)?);
    let mut count_bytes = [0u8; 8];
    idx.read_exact(&mut count_bytes)?;
    let count = u64::from_le_bytes(count_bytes) as usize;

    let mut map = HashMap::with_capacity(count);
    for _ in 0..count {
        let mut id_bytes = [0u8; 32];
        idx.read_exact(&mut id_bytes)?;
        let mut offset_bytes = [0u8; 8];
        idx.read_exact(&mut offset_bytes)?;
        map.insert(ChunkId(id_bytes), u64::from_le_bytes(offset_bytes));
    }
    Ok(map)
}

fn parse_record(chunk_id: ChunkId, body: &[u8]) -> io::Result<Chunk> {
    let mut cur = 0usize;
    let version = take_u8(body, &mut cur)?;
    if version != RECORD_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown chunks.bin record version: {version}"),
        ));
    }
    let has_locator = take_u8(body, &mut cur)? != 0;
    let source_id = take_str(body, &mut cur)?;
    let locator = if has_locator {
        Some(take_str(body, &mut cur)?)
    } else {
        None
    };
    let text = take_str(body, &mut cur)?;

    Ok(Chunk {
        id: chunk_id,
        text,
        source_ref: SourceRef { source_id, locator },
    })
}

/// Cheap variant of [`parse_record`] that consumes only the version,
/// has-locator flag, and source_id fields from a record body — the
/// locator and text bytes are left unread. Used by
/// [`ChunkStoreReader::iter_chunk_source_ids`] for the `SourceIndex`
/// build pass, where the chunk text is irrelevant.
fn parse_source_id_only(body: &[u8]) -> io::Result<String> {
    let mut cur = 0usize;
    let version = take_u8(body, &mut cur)?;
    if version != RECORD_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown chunks.bin record version: {version}"),
        ));
    }
    let _has_locator = take_u8(body, &mut cur)?;
    take_str(body, &mut cur)
}

fn take_u8(body: &[u8], cur: &mut usize) -> io::Result<u8> {
    if *cur >= body.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short record body",
        ));
    }
    let val = body[*cur];
    *cur += 1;
    Ok(val)
}

fn take_str(body: &[u8], cur: &mut usize) -> io::Result<String> {
    if *cur + 4 > body.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short record body",
        ));
    }
    let len = u32::from_le_bytes(body[*cur..*cur + 4].try_into().unwrap()) as usize;
    *cur += 4;
    if *cur + len > body.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short record body",
        ));
    }
    let s = std::str::from_utf8(&body[*cur..*cur + len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        .to_string();
    *cur += len;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sref(source_id: &str, locator: Option<&str>) -> SourceRef {
        SourceRef {
            source_id: source_id.to_string(),
            locator: locator.map(str::to_string),
        }
    }

    #[test]
    fn roundtrip_single_chunk() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        let id = ChunkId::from_source(&[1; 32], 0);

        {
            let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
            w.append(id, &sref("doc.md", Some("chars 0-5")), "hello")
                .unwrap();
            assert_eq!(w.finalize().unwrap(), 1);
        }

        let r = ChunkStoreReader::open(&bin, &idx).unwrap();
        assert_eq!(r.len(), 1);
        let chunk = r.fetch(id).unwrap().unwrap();
        assert_eq!(chunk.id, id);
        assert_eq!(chunk.text, "hello");
        assert_eq!(chunk.source_ref.source_id, "doc.md");
        assert_eq!(chunk.source_ref.locator.as_deref(), Some("chars 0-5"));
    }

    #[test]
    fn roundtrip_many_chunks_with_and_without_locator() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        let inputs: Vec<(ChunkId, &str, Option<&str>, &str)> = (0..50)
            .map(|i| {
                let id = ChunkId::from_source(&[i as u8; 32], i);
                let locator = if i % 2 == 0 {
                    Some(if i % 4 == 0 {
                        "even-zero"
                    } else {
                        "even-other"
                    })
                } else {
                    None
                };
                (
                    id,
                    if i % 3 == 0 { "a.md" } else { "b.md" },
                    locator,
                    if i % 5 == 0 {
                        "short"
                    } else {
                        "longer chunk text"
                    },
                )
            })
            .collect();

        {
            let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
            for (id, src, loc, text) in &inputs {
                w.append(*id, &sref(src, *loc), text).unwrap();
            }
            assert_eq!(w.finalize().unwrap(), inputs.len());
        }

        let r = ChunkStoreReader::open(&bin, &idx).unwrap();
        assert_eq!(r.len(), inputs.len());

        for (id, src, loc, text) in &inputs {
            let chunk = r.fetch(*id).unwrap().expect("present");
            assert_eq!(chunk.text, *text);
            assert_eq!(chunk.source_ref.source_id, *src);
            assert_eq!(chunk.source_ref.locator.as_deref(), *loc);
        }
    }

    #[test]
    fn fetch_unknown_id_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        let id = ChunkId::from_source(&[1; 32], 0);
        {
            let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
            w.append(id, &sref("x", None), "x").unwrap();
            w.finalize().unwrap();
        }
        let r = ChunkStoreReader::open(&bin, &idx).unwrap();
        let bogus = ChunkId::from_source(&[42; 32], 99);
        assert!(r.fetch(bogus).unwrap().is_none());
    }

    #[test]
    fn empty_writer_finalizes_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");
        let w = ChunkStoreWriter::create(&bin, &idx).unwrap();
        assert!(w.is_empty());
        assert_eq!(w.finalize().unwrap(), 0);

        let r = ChunkStoreReader::open(&bin, &idx).unwrap();
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn idx_entries_sorted_by_chunk_id() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        // Insert in reverse-id order; idx file should still be sorted.
        let ids: Vec<ChunkId> = (0..10)
            .map(|i| ChunkId::from_source(&[i; 32], 0))
            .rev()
            .collect();
        {
            let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
            for id in &ids {
                w.append(*id, &sref("x", None), "x").unwrap();
            }
            w.finalize().unwrap();
        }

        // Read idx file directly and verify order.
        let mut idx_file = BufReader::new(File::open(&idx).unwrap());
        let mut count_bytes = [0u8; 8];
        idx_file.read_exact(&mut count_bytes).unwrap();
        let count = u64::from_le_bytes(count_bytes) as usize;
        assert_eq!(count, 10);

        let mut prev: Option<ChunkId> = None;
        for _ in 0..count {
            let mut id_bytes = [0u8; 32];
            idx_file.read_exact(&mut id_bytes).unwrap();
            let mut off_bytes = [0u8; 8];
            idx_file.read_exact(&mut off_bytes).unwrap();
            let id = ChunkId(id_bytes);
            if let Some(p) = prev {
                assert!(p < id, "idx entries must be ascending by chunk_id");
            }
            prev = Some(id);
        }
    }

    #[test]
    fn unicode_text_roundtrips() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        let id = ChunkId::from_source(&[7; 32], 0);
        let text = "αβγδ — emoji: 🦀🎉  CJK: 漢字テスト";
        {
            let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
            w.append(id, &sref("u", Some("loc")), text).unwrap();
            w.finalize().unwrap();
        }
        let r = ChunkStoreReader::open(&bin, &idx).unwrap();
        let chunk = r.fetch(id).unwrap().unwrap();
        assert_eq!(chunk.text, text);
    }

    #[test]
    fn scan_record_offsets_walks_to_eof() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        let texts = ["short", "a slightly longer body", "third"];
        let ids: Vec<ChunkId> = (0..texts.len())
            .map(|i| ChunkId::from_source(&[i as u8; 32], i as u64))
            .collect();
        {
            let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
            for (id, t) in ids.iter().zip(texts.iter()) {
                w.append(*id, &sref("doc", None), t).unwrap();
            }
            w.finalize().unwrap();
        }

        let offsets = scan_record_offsets(&bin).unwrap();
        assert_eq!(offsets.len(), texts.len() + 1);
        assert_eq!(offsets[0], 0);
        // Each subsequent offset is strictly greater than its predecessor.
        for w in offsets.windows(2) {
            assert!(w[0] < w[1]);
        }
        // Last offset matches actual file length.
        let file_len = std::fs::metadata(&bin).unwrap().len();
        assert_eq!(*offsets.last().unwrap(), file_len);
    }

    #[test]
    fn open_resume_truncates_and_continues_appending() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        // First build: write 5 records, simulate-crash before finalize
        // (drop the writer without calling finalize).
        let ids: Vec<ChunkId> = (0..5)
            .map(|i| ChunkId::from_source(&[i as u8; 32], i as u64))
            .collect();
        {
            let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
            for (i, id) in ids.iter().enumerate() {
                w.append(*id, &sref("doc", None), &format!("text-{i}"))
                    .unwrap();
            }
            // Drop without finalize: chunks.bin has data, chunks.idx absent.
        }
        assert!(!idx.exists(), "idx file must not exist pre-finalize");

        // Simulate "an extra trailing record was written but not
        // checkpointed". Resume should truncate it.
        // Here we just claim 3 of 5 are durable; resume truncates 4–5.
        let offsets = scan_record_offsets(&bin).unwrap();
        let keep = 3;
        let resume_offsets = offsets[..=keep].to_vec();
        let resume_ids = ids[..keep].to_vec();
        let mut w =
            ChunkStoreWriter::open_resume(&bin, &idx, &resume_ids, &resume_offsets).unwrap();

        // Append two more (different ids) and finalize.
        let new_ids: Vec<ChunkId> = (10..12)
            .map(|i| ChunkId::from_source(&[i as u8; 32], i as u64))
            .collect();
        for (i, id) in new_ids.iter().enumerate() {
            w.append(*id, &sref("doc", None), &format!("new-{i}"))
                .unwrap();
        }
        let final_count = w.finalize().unwrap();
        assert_eq!(final_count, keep + new_ids.len());

        // All kept ids and new ids are fetchable; truncated ids are not.
        let r = ChunkStoreReader::open(&bin, &idx).unwrap();
        for (i, id) in ids.iter().take(keep).enumerate() {
            let chunk = r.fetch(*id).unwrap().unwrap();
            assert_eq!(chunk.text, format!("text-{i}"));
        }
        for id in &ids[keep..] {
            assert!(r.fetch(*id).unwrap().is_none(), "truncated id reappeared");
        }
        for (i, id) in new_ids.iter().enumerate() {
            let chunk = r.fetch(*id).unwrap().unwrap();
            assert_eq!(chunk.text, format!("new-{i}"));
        }
    }

    #[test]
    fn open_resume_rejects_offset_count_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");
        // Empty file is fine for this check.
        File::create(&bin).unwrap();
        let id = ChunkId::from_source(&[1; 32], 0);
        let err = ChunkStoreWriter::open_resume(&bin, &idx, &[id], &[0]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn shared_index_reader_sees_writer_appends_live() {
        // The whole point of share_index / open_with_shared_index:
        // a reader constructed once at the start of a build observes
        // writer.append() results without any per-batch index copy.
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();

        // Append the first record + flush so its bytes are on disk.
        let id_a = ChunkId::from_source(&[1; 32], 0);
        w.append(id_a, &sref("a.md", None), "alpha").unwrap();
        w.flush().unwrap();

        // Open the live-view reader. Sharing happens here.
        let r = ChunkStoreReader::open_with_shared_index(&bin, w.share_index()).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r.fetch(id_a).unwrap().unwrap().text, "alpha");

        // Append more records and flush. The reader sees them
        // through the same Arc — no re-construction.
        let id_b = ChunkId::from_source(&[2; 32], 0);
        let id_c = ChunkId::from_source(&[3; 32], 0);
        w.append(id_b, &sref("b.md", None), "beta").unwrap();
        w.append(id_c, &sref("c.md", None), "gamma").unwrap();
        w.flush().unwrap();

        assert_eq!(r.len(), 3);
        assert_eq!(r.fetch(id_b).unwrap().unwrap().text, "beta");
        assert_eq!(r.fetch(id_c).unwrap().unwrap().text, "gamma");

        // Finalize doesn't break the live reader (it still holds an
        // Arc into the same map). The map is read-only post-finalize.
        let n = w.finalize().unwrap();
        assert_eq!(n, 3);
        assert_eq!(r.fetch(id_a).unwrap().unwrap().text, "alpha");
    }

    #[test]
    fn open_rejects_unknown_record_version() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        let id = ChunkId::from_source(&[1; 32], 0);

        {
            let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
            w.append(id, &sref("x", None), "x").unwrap();
            w.finalize().unwrap();
        }
        // Tamper with the version byte (offset 4 in the file: 4-byte
        // length prefix, then the version byte).
        let mut buf = std::fs::read(&bin).unwrap();
        buf[4] = 99; // unknown version
        std::fs::write(&bin, buf).unwrap();

        let r = ChunkStoreReader::open(&bin, &idx).unwrap();
        let err = r.fetch(id).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// Both `LoadMode::Mmap` and `LoadMode::Eager` must return
    /// byte-equivalent chunks. The serving-mode dispatch only changes
    /// where the bytes live, not what they decode to.
    #[test]
    fn mmap_and_eager_modes_return_identical_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        let inputs: Vec<(ChunkId, &str, &str)> = (0..8)
            .map(|i| {
                (
                    ChunkId::from_source(&[i as u8; 32], i as u64),
                    "doc.md",
                    "alpha bravo charlie delta echo foxtrot",
                )
            })
            .collect();

        {
            let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
            for (id, src, text) in &inputs {
                w.append(*id, &sref(src, None), text).unwrap();
            }
            w.finalize().unwrap();
        }

        let mmap_reader = ChunkStoreReader::open_with_mode(&bin, &idx, LoadMode::Mmap).unwrap();
        let eager_reader = ChunkStoreReader::open_with_mode(&bin, &idx, LoadMode::Eager).unwrap();

        for (id, _src, text) in &inputs {
            let m = mmap_reader.fetch(*id).unwrap().unwrap();
            let e = eager_reader.fetch(*id).unwrap().unwrap();
            assert_eq!(m.id, e.id);
            assert_eq!(m.text, e.text);
            assert_eq!(m.text, *text);
            assert_eq!(m.source_ref.source_id, e.source_ref.source_id);
        }
    }
}
