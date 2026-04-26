//! Persistent embedding-vector storage for a slot.
//!
//! Two files per slot, paralleling the chunks layout:
//! - `vectors.bin` — fixed 16-byte header, then `N × dim × 4` bytes of f32
//!   vectors back-to-back. No per-record framing — vectors are
//!   uniform-size for a given slot, so `vectors.bin[16 + pos*dim*4 .. +
//!   dim*4]` is the vector at position `pos`.
//! - `vectors.idx` — sorted `(chunk_id, position)` entries. Position is
//!   the index into the dense vector array, not a byte offset.
//!
//! Two-file split is the same reasoning as chunks: streaming append
//! during build, O(1) random-access fetch by chunk id at query time, and
//! a dense in-memory layout that the HNSW index in slice 5 can wrap
//! directly without a copy.
//!
//! Mutex-guarded sync IO for v1, like chunks. mmap upgrade is the
//! natural follow-up when serving modes (RAM vs disk-backed) get wired
//! end-to-end.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::types::ChunkId;

/// `vectors.bin` magic bytes — `b"VECT"`. Lets us detect a wrong-file or
/// truncated header explicitly rather than getting confusing read errors
/// later.
const MAGIC: &[u8; 4] = b"VECT";

/// Format-version byte. Bump if the file layout changes; reader rejects
/// unknown versions.
const FORMAT_VERSION: u8 = 1;

/// Total header size in bytes. Vectors start at this offset.
const HEADER_SIZE: u64 = 16;

/// Quantization variant tag. Only f32 is supported in slice 4; f16 / int8
/// are reserved values for the future quantization slice.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorQuant {
    F32 = 0,
    // F16 = 1,    // future
    // Int8 = 2,   // future
}

// vectors.bin layout (slice 4):
//   [4 bytes] magic = b"VECT"
//   [1 byte]  format_version (= 1)
//   [1 byte]  quantization (0 = f32)
//   [2 bytes] reserved (zero)
//   [4 bytes] dimension (u32 LE)
//   [4 bytes] reserved (zero, future use)
//   ── header ends at offset 16 ──
//   [N × dim × 4 bytes] vectors back-to-back, native f32 little-endian.
//
// vectors.idx layout:
//   [8 bytes] entry count (u64 LE)
//   [count × 40 bytes] each: [32 bytes chunk_id] [8 bytes position (u64 LE)]
//
// Entries sorted ascending by chunk_id for deterministic file shape.

/// Streams vectors into `vectors.bin` during a build, accumulates the
/// `(chunk_id, position)` pairs, and writes `vectors.idx` on
/// [`finalize`](Self::finalize).
#[derive(Debug)]
pub struct VectorStoreWriter {
    bin: BufWriter<File>,
    idx_path: PathBuf,
    dimension: u32,
    /// Next position to assign to an appended vector.
    next_position: u64,
    entries: Vec<(ChunkId, u64)>,
}

impl VectorStoreWriter {
    pub fn create(bin_path: &Path, idx_path: &Path, dimension: u32) -> io::Result<Self> {
        if dimension == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "dimension must be > 0",
            ));
        }
        let mut bin = BufWriter::new(File::create(bin_path)?);
        bin.write_all(MAGIC)?;
        bin.write_all(&[FORMAT_VERSION, VectorQuant::F32 as u8, 0, 0])?;
        bin.write_all(&dimension.to_le_bytes())?;
        bin.write_all(&[0u8; 4])?; // reserved
        Ok(Self {
            bin,
            idx_path: idx_path.to_path_buf(),
            dimension,
            next_position: 0,
            entries: Vec::new(),
        })
    }

    /// Reopen an existing `vectors.bin` for resumed appending. Verifies
    /// the on-disk header matches `expected_dimension`, truncates the
    /// file to `HEADER_SIZE + chunk_ids.len() * dim * 4` (dropping any
    /// trailing partial vectors), and seeds the in-memory entries with
    /// `chunk_ids` paired with sequential positions `0..len`.
    ///
    /// The caller supplies `chunk_ids` in the same insertion order they
    /// were originally appended; positions are derived from order.
    pub fn open_resume(
        bin_path: &Path,
        idx_path: &Path,
        expected_dimension: u32,
        chunk_ids: &[ChunkId],
    ) -> io::Result<Self> {
        // Verify header dimension matches the embedder we're about to
        // continue building with. Header validation is the same one
        // VectorStoreReader does — bad magic / version / quant all
        // surface as InvalidData.
        let mut probe = File::open(bin_path)?;
        let on_disk_dim = read_header(&mut probe)?;
        drop(probe);
        if on_disk_dim != expected_dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "vectors.bin dimension {on_disk_dim} does not match expected {expected_dimension}"
                ),
            ));
        }

        let len = chunk_ids.len() as u64;
        let target_size = HEADER_SIZE + len * expected_dimension as u64 * 4;
        let truncate = OpenOptions::new().write(true).open(bin_path)?;
        truncate.set_len(target_size)?;
        drop(truncate);

        let file = OpenOptions::new().append(true).open(bin_path)?;
        let entries: Vec<(ChunkId, u64)> =
            chunk_ids.iter().copied().enumerate().map(|(i, id)| (id, i as u64)).collect();
        Ok(Self {
            bin: BufWriter::new(file),
            idx_path: idx_path.to_path_buf(),
            dimension: expected_dimension,
            next_position: len,
            entries,
        })
    }

    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Append a vector and record its position. Returns the position the
    /// vector was assigned.
    pub fn append(&mut self, chunk_id: ChunkId, vector: &[f32]) -> io::Result<u64> {
        if vector.len() != self.dimension as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "vector dimension mismatch: expected {}, got {}",
                    self.dimension,
                    vector.len(),
                ),
            ));
        }
        for value in vector {
            self.bin.write_all(&value.to_le_bytes())?;
        }
        let pos = self.next_position;
        self.next_position += 1;
        self.entries.push((chunk_id, pos));
        Ok(pos)
    }

    /// Flush `vectors.bin` and write `vectors.idx`. Returns the number
    /// of vectors committed.
    pub fn finalize(mut self) -> io::Result<usize> {
        self.bin.flush()?;
        drop(self.bin);

        self.entries.sort_by_key(|(id, _)| *id);
        let mut idx = BufWriter::new(File::create(&self.idx_path)?);
        idx.write_all(&(self.entries.len() as u64).to_le_bytes())?;
        for (id, pos) in &self.entries {
            idx.write_all(&id.0)?;
            idx.write_all(&pos.to_le_bytes())?;
        }
        idx.flush()?;
        Ok(self.entries.len())
    }
}

/// Random-access reader for `vectors.bin` / `vectors.idx`.
///
/// `Send + Sync`. Concurrent fetches serialize through a Mutex on the
/// file handle; the layout is friendly to a future mmap upgrade — the
/// dense vector array starts at byte [`HEADER_SIZE`] and runs to EOF.
#[derive(Debug)]
pub struct VectorStoreReader {
    bin: Mutex<File>,
    dimension: u32,
    index: HashMap<ChunkId, u64>,
}

impl VectorStoreReader {
    pub fn open(bin_path: &Path, idx_path: &Path) -> io::Result<Self> {
        let mut bin = File::open(bin_path)?;
        let dimension = read_header(&mut bin)?;

        let index = read_idx(idx_path)?;
        Ok(Self {
            bin: Mutex::new(bin),
            dimension,
            index,
        })
    }

    pub fn dimension(&self) -> u32 {
        self.dimension
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

    /// Fetch a vector by chunk id. Returns `Ok(None)` if the chunk is
    /// not in the index.
    pub fn fetch(&self, chunk_id: ChunkId) -> io::Result<Option<Vec<f32>>> {
        let Some(&position) = self.index.get(&chunk_id) else {
            return Ok(None);
        };
        Ok(Some(self.read_at_position(position)?))
    }

    /// Read the vector at a specific position (its index in the dense
    /// `vectors.bin` array). Used by the HNSW build path which iterates
    /// vectors in position order rather than by chunk id.
    pub fn read_at_position(&self, position: u64) -> io::Result<Vec<f32>> {
        let dim = self.dimension as usize;
        let byte_offset = HEADER_SIZE + position * dim as u64 * 4;

        let mut bin = self.bin.lock().expect("vectors.bin mutex poisoned");
        bin.seek(SeekFrom::Start(byte_offset))?;
        let mut bytes = vec![0u8; dim * 4];
        bin.read_exact(&mut bytes)?;
        drop(bin);

        let mut out = Vec::with_capacity(dim);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }

    /// Iterate `(chunk_id, position)` pairs in unspecified order. Used
    /// by the HNSW builder to construct its position→chunk_id map
    /// without an extra disk read.
    pub fn iter_chunk_positions(&self) -> impl Iterator<Item = (ChunkId, u64)> + '_ {
        self.index.iter().map(|(id, &pos)| (*id, pos))
    }
}

fn read_header(bin: &mut File) -> io::Result<u32> {
    let mut header = [0u8; HEADER_SIZE as usize];
    bin.read_exact(&mut header)?;
    if &header[0..4] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "vectors.bin: bad magic",
        ));
    }
    let version = header[4];
    if version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("vectors.bin: unknown format version {version}"),
        ));
    }
    let quant = header[5];
    if quant != VectorQuant::F32 as u8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("vectors.bin: unsupported quantization variant {quant}"),
        ));
    }
    let dimension = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "vectors.bin: dimension is 0",
        ));
    }
    Ok(dimension)
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
        let mut pos_bytes = [0u8; 8];
        idx.read_exact(&mut pos_bytes)?;
        map.insert(ChunkId(id_bytes), u64::from_le_bytes(pos_bytes));
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: u32, slot: u32) -> Vec<f32> {
        // Distinct unit vector per slot index — easy to verify roundtrip.
        let mut v = vec![0.0; dim as usize];
        v[(slot as usize) % dim as usize] = 1.0;
        v
    }

    #[test]
    fn roundtrip_single_vector() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");

        let id = ChunkId::from_source(&[1; 32], 0);
        let v = vec![0.1, 0.2, 0.3, 0.4];

        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4).unwrap();
            assert_eq!(w.append(id, &v).unwrap(), 0);
            assert_eq!(w.finalize().unwrap(), 1);
        }

        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        assert_eq!(r.dimension(), 4);
        assert_eq!(r.len(), 1);
        let fetched = r.fetch(id).unwrap().unwrap();
        assert_eq!(fetched, v);
    }

    #[test]
    fn roundtrip_many_vectors_random_order() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let dim = 8;

        let inputs: Vec<(ChunkId, Vec<f32>)> = (0..50)
            .map(|i| {
                (
                    ChunkId::from_source(&[i as u8; 32], i),
                    unit_vec(dim, i as u32),
                )
            })
            .collect();

        {
            let mut w = VectorStoreWriter::create(&bin, &idx, dim).unwrap();
            for (id, v) in &inputs {
                w.append(*id, v).unwrap();
            }
            assert_eq!(w.finalize().unwrap(), inputs.len());
        }

        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        assert_eq!(r.dimension(), dim);
        assert_eq!(r.len(), inputs.len());

        // Fetch in reverse order to make sure positions are wired correctly.
        for (id, expected) in inputs.iter().rev() {
            let fetched = r.fetch(*id).unwrap().unwrap();
            assert_eq!(&fetched, expected);
        }
    }

    #[test]
    fn fetch_unknown_id_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");

        let id = ChunkId::from_source(&[1; 32], 0);
        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4).unwrap();
            w.append(id, &[1.0, 2.0, 3.0, 4.0]).unwrap();
            w.finalize().unwrap();
        }
        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        let bogus = ChunkId::from_source(&[99; 32], 0);
        assert!(r.fetch(bogus).unwrap().is_none());
    }

    #[test]
    fn append_rejects_dimension_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let mut w = VectorStoreWriter::create(&bin, &idx, 4).unwrap();
        let id = ChunkId::from_source(&[1; 32], 0);
        let err = w.append(id, &[0.0; 5]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn create_rejects_zero_dimension() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let err = VectorStoreWriter::create(&bin, &idx, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn empty_writer_finalizes_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let w = VectorStoreWriter::create(&bin, &idx, 16).unwrap();
        assert!(w.is_empty());
        assert_eq!(w.finalize().unwrap(), 0);

        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
        assert_eq!(r.dimension(), 16);
    }

    #[test]
    fn open_rejects_bad_magic() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let id = ChunkId::from_source(&[1; 32], 0);
        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4).unwrap();
            w.append(id, &[1.0, 2.0, 3.0, 4.0]).unwrap();
            w.finalize().unwrap();
        }
        // Corrupt magic
        let mut buf = std::fs::read(&bin).unwrap();
        buf[0] = b'X';
        std::fs::write(&bin, buf).unwrap();
        let err = VectorStoreReader::open(&bin, &idx).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn open_rejects_unknown_format_version() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let id = ChunkId::from_source(&[1; 32], 0);
        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4).unwrap();
            w.append(id, &[1.0, 2.0, 3.0, 4.0]).unwrap();
            w.finalize().unwrap();
        }
        let mut buf = std::fs::read(&bin).unwrap();
        buf[4] = 99;
        std::fs::write(&bin, buf).unwrap();
        let err = VectorStoreReader::open(&bin, &idx).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn open_resume_truncates_and_continues_appending() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let dim = 4u32;

        // First write 5 vectors and drop the writer pre-finalize
        // (simulates crash). chunks.bin has 5 records on disk; idx
        // file doesn't exist yet.
        let ids: Vec<ChunkId> = (0..5)
            .map(|i| ChunkId::from_source(&[i as u8; 32], i as u64))
            .collect();
        {
            let mut w = VectorStoreWriter::create(&bin, &idx, dim).unwrap();
            for (i, id) in ids.iter().enumerate() {
                let v = vec![i as f32; dim as usize];
                w.append(*id, &v).unwrap();
            }
        }
        // Pre-resume file is HEADER + 5 * dim * 4 bytes.
        let pre = std::fs::metadata(&bin).unwrap().len();
        assert_eq!(pre, HEADER_SIZE + 5 * dim as u64 * 4);

        // Resume keeping only first 3 vectors.
        let keep = 3usize;
        let mut w =
            VectorStoreWriter::open_resume(&bin, &idx, dim, &ids[..keep]).unwrap();
        // File got truncated.
        let mid = std::fs::metadata(&bin).unwrap().len();
        assert_eq!(mid, HEADER_SIZE + keep as u64 * dim as u64 * 4);

        // Append two new vectors at the right positions.
        let new_ids: Vec<ChunkId> = (10..12)
            .map(|i| ChunkId::from_source(&[i as u8; 32], i as u64))
            .collect();
        for (i, id) in new_ids.iter().enumerate() {
            let v = vec![100.0 + i as f32; dim as usize];
            assert_eq!(w.append(*id, &v).unwrap(), keep as u64 + i as u64);
        }
        let count = w.finalize().unwrap();
        assert_eq!(count, keep + new_ids.len());

        // All kept vectors fetch correctly; truncated ones are gone.
        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        for (i, id) in ids.iter().take(keep).enumerate() {
            let v = r.fetch(*id).unwrap().unwrap();
            assert_eq!(v, vec![i as f32; dim as usize]);
        }
        for id in &ids[keep..] {
            assert!(r.fetch(*id).unwrap().is_none());
        }
        for (i, id) in new_ids.iter().enumerate() {
            let v = r.fetch(*id).unwrap().unwrap();
            assert_eq!(v, vec![100.0 + i as f32; dim as usize]);
        }
    }

    #[test]
    fn open_resume_rejects_dimension_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let id = ChunkId::from_source(&[1; 32], 0);
        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4).unwrap();
            w.append(id, &[0.0; 4]).unwrap();
        }
        let err = VectorStoreWriter::open_resume(&bin, &idx, 8, &[id]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn idx_entries_sorted_by_chunk_id() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");

        let ids: Vec<ChunkId> = (0..10)
            .map(|i| ChunkId::from_source(&[i; 32], 0))
            .rev()
            .collect();
        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4).unwrap();
            for id in &ids {
                w.append(*id, &[0.0, 0.0, 0.0, 0.0]).unwrap();
            }
            w.finalize().unwrap();
        }

        let mut idx_file = BufReader::new(File::open(&idx).unwrap());
        let mut count_bytes = [0u8; 8];
        idx_file.read_exact(&mut count_bytes).unwrap();
        let count = u64::from_le_bytes(count_bytes) as usize;
        assert_eq!(count, 10);

        let mut prev: Option<ChunkId> = None;
        for _ in 0..count {
            let mut id_bytes = [0u8; 32];
            idx_file.read_exact(&mut id_bytes).unwrap();
            let mut pos_bytes = [0u8; 8];
            idx_file.read_exact(&mut pos_bytes).unwrap();
            let id = ChunkId(id_bytes);
            if let Some(p) = prev {
                assert!(p < id, "idx entries must be ascending by chunk_id");
            }
            prev = Some(id);
        }
    }
}
