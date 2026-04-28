//! Persistent embedding-vector storage for a slot.
//!
//! Two files per slot, paralleling the chunks layout:
//! - `vectors.bin` — fixed 16-byte header, then `N × stride` bytes of
//!   vectors back-to-back. `stride = record_header_bytes + dim *
//!   bytes_per_dim` is determined by the slot's [`VectorQuant`]
//!   choice: f32 = 4×dim, f16 = 2×dim, int8 = 4 + dim (the leading
//!   4 bytes are a per-vector f32 scale factor). No per-record
//!   framing beyond the optional scale, so `vectors.bin[16 +
//!   pos*stride .. +stride]` is the vector at position `pos`.
//! - `vectors.idx` — sorted `(chunk_id, position)` entries. Position is
//!   the index into the dense vector array, not a byte offset.
//!
//! Two-file split is the same reasoning as chunks: streaming append
//! during build, O(1) random-access fetch by chunk id at query time, and
//! a dense in-memory layout that the HNSW index in slice 5 can wrap
//! directly without a copy.
//!
//! Reader storage modes:
//! - `LoadMode::Mmap` (disk-backed serving) — file is mmap'd, fetches
//!   are slice indexes against the OS page cache. Concurrent queries
//!   don't serialize on a Mutex; the working set lives in cache.
//! - `LoadMode::Eager` (RAM-resident serving) — file is read in full
//!   into a `Box<[u8]>` at open time. No disk I/O on subsequent
//!   fetches; predictable latency, no eviction by other processes.
//!
//! Vector readers are only opened on finalized slots — the build path
//! reads vectors back through the in-progress writer's positions, not
//! through `VectorStoreReader`. Both modes are therefore safe to mmap;
//! the `chunks.bin` reader has the analogous live-build complication
//! and handles it via `ChunkStoreReader::open_with_shared_index`.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use super::types::ChunkId;

/// How the reader should hold the slot's `vectors.bin`. Translated from
/// `manifest.serving.mode` at slot-open time. See module docs for the
/// trade-offs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    /// mmap the file; let the OS page cache handle hot/cold.
    Mmap,
    /// Read the entire file into a `Box<[u8]>` at open time.
    Eager,
}

/// `vectors.bin` magic bytes — `b"VECT"`. Lets us detect a wrong-file or
/// truncated header explicitly rather than getting confusing read errors
/// later.
const MAGIC: &[u8; 4] = b"VECT";

/// Format-version byte. Bump if the file layout changes; reader rejects
/// unknown versions.
const FORMAT_VERSION: u8 = 1;

/// Total header size in bytes. Vectors start at this offset.
const HEADER_SIZE: u64 = 16;

/// Quantization variant tag stored in the `vectors.bin` header.
///
/// `F32` (4 bytes/dim, native) is the original format. `F16` (2 bytes/
/// dim via [`half::f16`]) trades ~1% recall for half the on-disk
/// vector footprint. `Int8` (1 byte/dim + 4 bytes/vector scale)
/// trades ~2-5% recall for ~4× compression, using symmetric per-vector
/// quantization: each vector's scalars are divided by `max_abs(v) / 127`
/// at write time and clamped to `i8`. Decode multiplies by the stored
/// scale on read.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorQuant {
    F32 = 0,
    F16 = 1,
    Int8 = 2,
}

impl VectorQuant {
    /// Bytes consumed per scalar dimension on disk.
    pub fn bytes_per_dim(self) -> usize {
        match self {
            VectorQuant::F32 => 4,
            VectorQuant::F16 => 2,
            VectorQuant::Int8 => 1,
        }
    }

    /// Per-record metadata bytes preceding the `dim × bytes_per_dim`
    /// scalar payload. Used by Int8 to store the per-vector scale
    /// factor as a leading f32; F32/F16 have no per-record header.
    pub fn record_header_bytes(self) -> usize {
        match self {
            VectorQuant::F32 | VectorQuant::F16 => 0,
            VectorQuant::Int8 => 4,
        }
    }

    /// On-disk stride per stored vector — the size of one record in
    /// `vectors.bin`. Used everywhere position-to-byte arithmetic
    /// happens (read_at_position, open_resume truncation,
    /// create_or_open_append).
    pub fn record_stride(self, dim: usize) -> usize {
        self.record_header_bytes() + dim * self.bytes_per_dim()
    }

    /// Round-trip from the on-disk header byte. `Err` for unknown /
    /// reserved values.
    pub fn from_header_byte(b: u8) -> io::Result<Self> {
        match b {
            0 => Ok(VectorQuant::F32),
            1 => Ok(VectorQuant::F16),
            2 => Ok(VectorQuant::Int8),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("vectors.bin: unsupported quantization variant {other}"),
            )),
        }
    }
}

impl From<super::config::Quantization> for VectorQuant {
    fn from(q: super::config::Quantization) -> Self {
        match q {
            super::config::Quantization::F32 => VectorQuant::F32,
            super::config::Quantization::F16 => VectorQuant::F16,
            super::config::Quantization::Int8 => VectorQuant::Int8,
        }
    }
}

// vectors.bin layout:
//   [4 bytes] magic = b"VECT"
//   [1 byte]  format_version (= 1)
//   [1 byte]  quantization (0 = f32, 1 = f16, 2 = int8)
//   [2 bytes] reserved (zero)
//   [4 bytes] dimension (u32 LE)
//   [4 bytes] reserved (zero, future use)
//   ── header ends at offset 16 ──
//   N records back-to-back. Per-record layout:
//     F32 / F16: [dim × bytes_per_dim] scalar payload, native LE.
//     Int8:      [4 bytes scale (f32 LE)][dim × i8] — symmetric
//                per-vector quantization, decode as `q as f32 * scale`.
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
    quant: VectorQuant,
    /// Next position to assign to an appended vector.
    next_position: u64,
    entries: Vec<(ChunkId, u64)>,
}

impl VectorStoreWriter {
    /// Create a new `vectors.bin` with the given quantization. F32 is
    /// the historical default; F16 halves on-disk vector size at the
    /// cost of ~1% dense recall (validated against the project's
    /// `recall_eval` harness on a per-bucket basis).
    pub fn create(
        bin_path: &Path,
        idx_path: &Path,
        dimension: u32,
        quant: VectorQuant,
    ) -> io::Result<Self> {
        if dimension == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "dimension must be > 0",
            ));
        }
        let mut bin = BufWriter::new(File::create(bin_path)?);
        bin.write_all(MAGIC)?;
        bin.write_all(&[FORMAT_VERSION, quant as u8, 0, 0])?;
        bin.write_all(&dimension.to_le_bytes())?;
        bin.write_all(&[0u8; 4])?; // reserved
        Ok(Self {
            bin,
            idx_path: idx_path.to_path_buf(),
            dimension,
            quant,
            next_position: 0,
            entries: Vec::new(),
        })
    }

    /// Reopen an existing `vectors.bin` for resumed appending. Verifies
    /// the on-disk header matches `expected_dimension`, truncates the
    /// file to `HEADER_SIZE + chunk_ids.len() * stride` (dropping any
    /// trailing partial vectors), and seeds the in-memory entries with
    /// `chunk_ids` paired with sequential positions `0..len`. The
    /// quantization variant is read back from the header — a resumed
    /// build inherits whatever choice the original `create` call made.
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
        let (on_disk_dim, on_disk_quant) = read_header(&mut probe)?;
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
        let stride = on_disk_quant.record_stride(expected_dimension as usize) as u64;
        let target_size = HEADER_SIZE + len * stride;
        let truncate = OpenOptions::new().write(true).open(bin_path)?;
        truncate.set_len(target_size)?;
        drop(truncate);

        let file = OpenOptions::new().append(true).open(bin_path)?;
        let entries: Vec<(ChunkId, u64)> = chunk_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, i as u64))
            .collect();
        Ok(Self {
            bin: BufWriter::new(file),
            idx_path: idx_path.to_path_buf(),
            dimension: expected_dimension,
            quant: on_disk_quant,
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
    /// vector was assigned. The input is always `&[f32]` (embedders
    /// emit f32); quantization to the slot's chosen format happens
    /// here.
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
        match self.quant {
            VectorQuant::F32 => {
                for value in vector {
                    self.bin.write_all(&value.to_le_bytes())?;
                }
            }
            VectorQuant::F16 => {
                for value in vector {
                    let h = half::f16::from_f32(*value);
                    self.bin.write_all(&h.to_le_bytes())?;
                }
            }
            VectorQuant::Int8 => {
                // Symmetric per-vector quantization. `scale` is chosen
                // so the vector's max-abs value lands at i8::MAX (127);
                // smaller values get proportionally scaled. The all-
                // zero edge case fixes scale at a sentinel non-zero
                // value so decode still produces zeros — without this,
                // dividing by 0 below would NaN.
                let max_abs = vector.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                self.bin.write_all(&scale.to_le_bytes())?;
                let inv = 1.0 / scale;
                for value in vector {
                    let q = (value * inv).round().clamp(-127.0, 127.0) as i8;
                    self.bin.write_all(&[q as u8])?;
                }
            }
        }
        let pos = self.next_position;
        self.next_position += 1;
        self.entries.push((chunk_id, pos));
        Ok(pos)
    }

    /// Flush buffered writes to disk and `sync_data`. Called at
    /// every build-batch boundary so the BatchEmbedded log entry
    /// only lands after the corresponding vectors are durable.
    pub fn flush(&mut self) -> io::Result<()> {
        self.bin.flush()?;
        self.bin.get_ref().sync_data()?;
        Ok(())
    }

    /// Open an existing `vectors.bin` / `vectors.idx` pair for
    /// appending, or create them fresh if absent. Used by the mutation
    /// path to re-engage the delta writer across `Bucket::insert` calls.
    /// Each call must `finalize()` so the next call sees an up-to-date
    /// `vectors.idx`.
    ///
    /// Validates the on-disk header dimension when an existing file is
    /// found — a mismatch means the slot's embedder has somehow drifted
    /// since the prior call. The quantization variant comes from the
    /// existing header for an open-append case, or from the supplied
    /// `quant_for_create` when creating fresh.
    pub fn create_or_open_append(
        bin_path: &Path,
        idx_path: &Path,
        expected_dimension: u32,
        quant_for_create: VectorQuant,
    ) -> io::Result<Self> {
        if !bin_path.exists() {
            return Self::create(bin_path, idx_path, expected_dimension, quant_for_create);
        }

        let mut probe = File::open(bin_path)?;
        let (on_disk_dim, on_disk_quant) = read_header(&mut probe)?;
        drop(probe);
        if on_disk_dim != expected_dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "vectors.bin dimension {on_disk_dim} does not match expected {expected_dimension}",
                ),
            ));
        }

        let entries: Vec<(ChunkId, u64)> = if idx_path.exists() {
            let map = read_idx(idx_path)?;
            let mut e: Vec<(ChunkId, u64)> = map.into_iter().collect();
            e.sort_by_key(|(_, pos)| *pos);
            e
        } else {
            Vec::new()
        };
        let next_position = entries.len() as u64;

        let file = OpenOptions::new().append(true).open(bin_path)?;
        Ok(Self {
            bin: BufWriter::new(file),
            idx_path: idx_path.to_path_buf(),
            dimension: expected_dimension,
            quant: on_disk_quant,
            next_position,
            entries,
        })
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
/// `Send + Sync`. Backed by either an mmap (`LoadMode::Mmap`) or an
/// eagerly-loaded buffer (`LoadMode::Eager`); both expose the same
/// `&[u8]` view of the file's bytes, so `read_at_position` is a slice
/// index + a small f32-decode loop, no syscall and no Mutex.
pub struct VectorStoreReader {
    storage: VectorStorage,
    dimension: u32,
    quant: VectorQuant,
    index: HashMap<ChunkId, u64>,
}

enum VectorStorage {
    /// `mmap`'d view of the file. The OS page cache handles hot/cold;
    /// concurrent reads aren't serialized.
    Mapped(Mmap),
    /// Whole-file copy in process memory. Predictable latency, no
    /// page-cache eviction by other processes.
    Eager(Box<[u8]>),
}

impl VectorStorage {
    fn bytes(&self) -> &[u8] {
        match self {
            VectorStorage::Mapped(m) => m,
            VectorStorage::Eager(v) => v,
        }
    }
}

impl std::fmt::Debug for VectorStoreReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.storage {
            VectorStorage::Mapped(_) => "Mapped",
            VectorStorage::Eager(_) => "Eager",
        };
        f.debug_struct("VectorStoreReader")
            .field("storage", &kind)
            .field("dimension", &self.dimension)
            .field("count", &self.index.len())
            .finish()
    }
}

impl VectorStoreReader {
    /// Open with the default mmap mode. Equivalent to
    /// [`Self::open_with_mode`] with [`LoadMode::Mmap`]; kept as the
    /// short name for tests and call sites that don't dispatch on
    /// serving mode.
    pub fn open(bin_path: &Path, idx_path: &Path) -> io::Result<Self> {
        Self::open_with_mode(bin_path, idx_path, LoadMode::Mmap)
    }

    /// Open with an explicit load mode. Production slot-open paths
    /// translate `manifest.serving.mode` into the appropriate
    /// [`LoadMode`] and call this.
    pub fn open_with_mode(bin_path: &Path, idx_path: &Path, mode: LoadMode) -> io::Result<Self> {
        let storage = match mode {
            LoadMode::Mmap => {
                let file = File::open(bin_path)?;
                // SAFETY: we hold the file open for the lifetime of
                // `Mmap` (it's owned by `VectorStorage::Mapped`), and
                // slot files are immutable post-build — no concurrent
                // truncation / replacement is expected. A user racing
                // `rm` on the slot dir during a query would be invalid
                // either way.
                let mmap = unsafe { Mmap::map(&file)? };
                VectorStorage::Mapped(mmap)
            }
            LoadMode::Eager => {
                let bytes = std::fs::read(bin_path)?;
                VectorStorage::Eager(bytes.into_boxed_slice())
            }
        };
        let (dimension, quant) = read_header_bytes(storage.bytes())?;
        let index = read_idx(idx_path)?;
        Ok(Self {
            storage,
            dimension,
            quant,
            index,
        })
    }

    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Quantization tier this `vectors.bin` was written with. Returned
    /// for callers (HNSW build, manifest validation) that need to
    /// know precisely what's on disk vs. what was *requested* in
    /// `bucket.toml`.
    pub fn quant(&self) -> VectorQuant {
        self.quant
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
    ///
    /// Always returns `Vec<f32>` regardless of on-disk quantization —
    /// the in-memory representation in HNSW build / search remains f32
    /// for now. Per-call promotion is cheap (a few thousand ops at
    /// dim=1024), and the caller is read-once-per-position.
    pub fn read_at_position(&self, position: u64) -> io::Result<Vec<f32>> {
        let dim = self.dimension as usize;
        let stride = self.quant.record_stride(dim);
        let byte_offset = (HEADER_SIZE as usize) + (position as usize) * stride;
        let byte_end = byte_offset + stride;

        let bytes = self.storage.bytes();
        if byte_end > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "vectors.bin: position {position} reads past end ({byte_end} > {})",
                    bytes.len(),
                ),
            ));
        }
        let slice = &bytes[byte_offset..byte_end];

        let mut out = Vec::with_capacity(dim);
        match self.quant {
            VectorQuant::F32 => {
                for chunk in slice.chunks_exact(4) {
                    out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            VectorQuant::F16 => {
                for chunk in slice.chunks_exact(2) {
                    let h = half::f16::from_le_bytes(chunk.try_into().unwrap());
                    out.push(h.to_f32());
                }
            }
            VectorQuant::Int8 => {
                let scale = f32::from_le_bytes(slice[0..4].try_into().unwrap());
                for &b in &slice[4..] {
                    let q = b as i8;
                    out.push(q as f32 * scale);
                }
            }
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

fn read_header_bytes(bytes: &[u8]) -> io::Result<(u32, VectorQuant)> {
    if bytes.len() < HEADER_SIZE as usize {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "vectors.bin: file shorter than header",
        ));
    }
    let header = &bytes[..HEADER_SIZE as usize];
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
    let quant = VectorQuant::from_header_byte(header[5])?;
    let dimension = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "vectors.bin: dimension is 0",
        ));
    }
    Ok((dimension, quant))
}

fn read_header(bin: &mut File) -> io::Result<(u32, VectorQuant)> {
    let mut header = [0u8; HEADER_SIZE as usize];
    bin.read_exact(&mut header)?;
    read_header_bytes(&header)
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
    fn roundtrip_single_vector_f16() {
        // f16 quantization: write a vector, read it back, expect a
        // close-but-not-exact match (f16 has ~3-4 decimal digits of
        // precision in the [0, 1] range).
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");

        let id = ChunkId::from_source(&[1; 32], 0);
        let v = vec![0.1, 0.2, 0.3, 0.4];

        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F16).unwrap();
            assert_eq!(w.append(id, &v).unwrap(), 0);
            assert_eq!(w.finalize().unwrap(), 1);
        }

        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        assert_eq!(r.dimension(), 4);
        assert_eq!(r.quant(), VectorQuant::F16);
        let fetched = r.fetch(id).unwrap().unwrap();
        for (i, (got, want)) in fetched.iter().zip(v.iter()).enumerate() {
            let diff = (got - want).abs();
            assert!(diff < 0.001, "dim {i}: got {got}, want {want}, diff {diff}",);
        }

        // On-disk size: header + 4 dims × 2 bytes = 24 bytes (vs.
        // 32 bytes for the f32 case). A direct measurement so the
        // halving is asserted, not just believed.
        let on_disk = std::fs::metadata(&bin).unwrap().len();
        assert_eq!(on_disk, 16 + 4 * 2);
    }

    #[test]
    fn roundtrip_single_vector_int8() {
        // Symmetric per-vector int8 quantization: max-abs maps to ±127.
        // Pick a vector whose values span a useful range so the
        // quantization step is exercised end-to-end.
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");

        let id = ChunkId::from_source(&[1; 32], 0);
        let v = vec![0.1, 0.5, -0.3, 0.7];
        let max_abs = 0.7f32;
        let scale = max_abs / 127.0;
        // Quantization error per scalar is at most scale/2.
        let tol = scale * 0.6;

        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::Int8).unwrap();
            assert_eq!(w.append(id, &v).unwrap(), 0);
            assert_eq!(w.finalize().unwrap(), 1);
        }

        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        assert_eq!(r.dimension(), 4);
        assert_eq!(r.quant(), VectorQuant::Int8);
        let fetched = r.fetch(id).unwrap().unwrap();
        for (i, (got, want)) in fetched.iter().zip(v.iter()).enumerate() {
            let diff = (got - want).abs();
            assert!(
                diff < tol,
                "dim {i}: got {got}, want {want}, diff {diff}, tol {tol}"
            );
        }

        // On-disk size: header + 1 record × (4 byte scale + 4 dims × 1 byte).
        let on_disk = std::fs::metadata(&bin).unwrap().len();
        assert_eq!(on_disk, 16 + 4 + 4);
    }

    #[test]
    fn int8_all_zero_vector_decodes_as_zero() {
        // The zero edge case: max_abs = 0 would NaN if we naively
        // divided. Writer fixes scale at 1.0 in that case; decode
        // must still produce all zeros.
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");

        let id = ChunkId::from_source(&[2; 32], 0);
        let v = vec![0.0f32; 8];

        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 8, VectorQuant::Int8).unwrap();
            w.append(id, &v).unwrap();
            w.finalize().unwrap();
        }

        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        let fetched = r.fetch(id).unwrap().unwrap();
        for got in &fetched {
            assert_eq!(*got, 0.0);
        }
    }

    #[test]
    fn f16_header_byte_persists_across_open() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");

        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F16).unwrap();
            w.append(ChunkId::from_source(&[1; 32], 0), &[1.0, 2.0, 3.0, 4.0])
                .unwrap();
            w.finalize().unwrap();
        }

        // Reader should see F16, not silently fall back to F32.
        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        assert_eq!(r.quant(), VectorQuant::F16);

        // create_or_open_append should read the on-disk quant rather
        // than overriding with the caller-supplied default. Pass F32
        // as `quant_for_create` and verify the resulting writer
        // inherits F16 from the existing file.
        let w = VectorStoreWriter::create_or_open_append(&bin, &idx, 4, VectorQuant::F32).unwrap();
        assert_eq!(w.quant, VectorQuant::F16);
    }

    #[test]
    fn roundtrip_single_vector() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");

        let id = ChunkId::from_source(&[1; 32], 0);
        let v = vec![0.1, 0.2, 0.3, 0.4];

        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F32).unwrap();
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
            let mut w = VectorStoreWriter::create(&bin, &idx, dim, VectorQuant::F32).unwrap();
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
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F32).unwrap();
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
        let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F32).unwrap();
        let id = ChunkId::from_source(&[1; 32], 0);
        let err = w.append(id, &[0.0; 5]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn create_rejects_zero_dimension() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let err = VectorStoreWriter::create(&bin, &idx, 0, VectorQuant::F32).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn empty_writer_finalizes_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let w = VectorStoreWriter::create(&bin, &idx, 16, VectorQuant::F32).unwrap();
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
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F32).unwrap();
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
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F32).unwrap();
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
            let mut w = VectorStoreWriter::create(&bin, &idx, dim, VectorQuant::F32).unwrap();
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
        let mut w = VectorStoreWriter::open_resume(&bin, &idx, dim, &ids[..keep]).unwrap();
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
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F32).unwrap();
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
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F32).unwrap();
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

    /// Both `LoadMode::Mmap` and `LoadMode::Eager` must produce
    /// byte-equivalent fetch results — the storage layer is the only
    /// thing that differs between disk-backed and RAM-resident
    /// serving.
    #[test]
    fn mmap_and_eager_modes_return_identical_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let dim = 4;

        let inputs: Vec<(ChunkId, Vec<f32>)> = (0..10)
            .map(|i| {
                (
                    ChunkId::from_source(&[i as u8; 32], i),
                    unit_vec(dim, i as u32),
                )
            })
            .collect();

        {
            let mut w = VectorStoreWriter::create(&bin, &idx, dim, VectorQuant::F32).unwrap();
            for (id, v) in &inputs {
                w.append(*id, v).unwrap();
            }
            w.finalize().unwrap();
        }

        let mmap_reader = VectorStoreReader::open_with_mode(&bin, &idx, LoadMode::Mmap).unwrap();
        let eager_reader = VectorStoreReader::open_with_mode(&bin, &idx, LoadMode::Eager).unwrap();

        for (id, expected) in &inputs {
            assert_eq!(&mmap_reader.fetch(*id).unwrap().unwrap(), expected);
            assert_eq!(&eager_reader.fetch(*id).unwrap().unwrap(), expected);
        }
    }

    /// Reading past EOF on an mmap'd file should produce a clean
    /// `UnexpectedEof` rather than panic on slice bounds.
    #[test]
    fn read_at_position_past_end_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let id = ChunkId::from_source(&[1; 32], 0);
        {
            let mut w = VectorStoreWriter::create(&bin, &idx, 4, VectorQuant::F32).unwrap();
            w.append(id, &[0.1, 0.2, 0.3, 0.4]).unwrap();
            w.finalize().unwrap();
        }
        let r = VectorStoreReader::open(&bin, &idx).unwrap();
        let err = r.read_at_position(99).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}
