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
//! Mutex-guarded sync IO for v1. If concurrent fetch latency becomes a
//! bottleneck, switch to mmap or `pread` without changing the public API.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::types::{Chunk, ChunkId, SourceRef};

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

/// Streams chunks into `chunks.bin` during a build, accumulates the
/// `(chunk_id, offset)` pairs, and writes `chunks.idx` on
/// [`finalize`](Self::finalize).
#[derive(Debug)]
pub struct ChunkStoreWriter {
    bin: BufWriter<File>,
    idx_path: PathBuf,
    entries: Vec<(ChunkId, u64)>,
    cursor: u64,
}

impl ChunkStoreWriter {
    pub fn create(bin_path: &Path, idx_path: &Path) -> io::Result<Self> {
        let file = File::create(bin_path)?;
        Ok(Self {
            bin: BufWriter::new(file),
            idx_path: idx_path.to_path_buf(),
            entries: Vec::new(),
            cursor: 0,
        })
    }

    /// Reopen an existing `chunks.bin` for resumed appending. Truncates
    /// the file to `record_offsets.last()` (the byte offset just past
    /// the last record we want to keep), seeds the in-memory entries
    /// list from `chunk_ids` paired with `record_offsets[..len-1]`, and
    /// opens the file in append mode for further writes.
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
        let entries: Vec<(ChunkId, u64)> = chunk_ids
            .iter()
            .copied()
            .zip(record_offsets[..chunk_ids.len()].iter().copied())
            .collect();
        Ok(Self {
            bin: BufWriter::new(file),
            idx_path: idx_path.to_path_buf(),
            entries,
            cursor,
        })
    }

    /// Append a chunk and record its offset for the index. Returns the
    /// offset where the record was written.
    pub fn append(
        &mut self,
        chunk_id: ChunkId,
        source_ref: &SourceRef,
        text: &str,
    ) -> io::Result<u64> {
        let offset = self.cursor;
        let written = write_record(&mut self.bin, source_ref, text)?;
        self.cursor += written;
        self.entries.push((chunk_id, offset));
        Ok(offset)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Flush `chunks.bin` and write `chunks.idx`. Returns the number of
    /// chunks committed.
    pub fn finalize(mut self) -> io::Result<usize> {
        self.bin.flush()?;
        // Drop the BufWriter so the underlying file is closed before we
        // open the next file — helps on Windows; harmless on Unix.
        drop(self.bin);

        self.entries.sort_by_key(|(id, _)| *id);
        let mut idx = BufWriter::new(File::create(&self.idx_path)?);
        idx.write_all(&(self.entries.len() as u64).to_le_bytes())?;
        for (id, offset) in &self.entries {
            idx.write_all(&id.0)?;
            idx.write_all(&offset.to_le_bytes())?;
        }
        idx.flush()?;
        Ok(self.entries.len())
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
/// `Send + Sync`. Concurrent fetches serialize through an internal Mutex
/// for the file handle (one seek + one read per fetch). Cheap at slice
/// 3's scales; revisit (mmap or `pread`) if measurements show contention.
#[derive(Debug)]
pub struct ChunkStoreReader {
    bin: Mutex<File>,
    index: HashMap<ChunkId, u64>,
}

impl ChunkStoreReader {
    pub fn open(bin_path: &Path, idx_path: &Path) -> io::Result<Self> {
        let bin = File::open(bin_path)?;
        let index = read_idx(idx_path)?;
        Ok(Self {
            bin: Mutex::new(bin),
            index,
        })
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn contains(&self, chunk_id: ChunkId) -> bool {
        self.index.contains_key(&chunk_id)
    }

    /// Fetch a chunk by id. Returns `Ok(None)` when the chunk isn't in
    /// the index; `Err` for IO or format errors during the read.
    pub fn fetch(&self, chunk_id: ChunkId) -> io::Result<Option<Chunk>> {
        let Some(&offset) = self.index.get(&chunk_id) else {
            return Ok(None);
        };

        let mut bin = self.bin.lock().expect("chunks.bin mutex poisoned");
        bin.seek(SeekFrom::Start(offset))?;

        let mut len_bytes = [0u8; 4];
        bin.read_exact(&mut len_bytes)?;
        let body_len = u32::from_le_bytes(len_bytes) as usize;

        let mut body = vec![0u8; body_len];
        bin.read_exact(&mut body)?;
        drop(bin); // release lock before parsing

        parse_record(chunk_id, &body).map(Some)
    }
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
        let err =
            ChunkStoreWriter::open_resume(&bin, &idx, &[id], &[0]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
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
}
