//! Dense (HNSW over embeddings) index for a slot.
//!
//! Wraps the `hnsw_rs` crate to provide build + search operations over
//! a slot's persisted vectors. The graph is dumped to disk after build
//! ([`DenseIndex::dump_to`]) and reloaded on slot open
//! ([`DenseIndex::load_from`]); a fresh build via [`DenseIndex::build`]
//! is the fallback when the dump is missing or stale (e.g. a slot
//! built before persistence landed, or a partial-write recovery).
//!
//! Distance: [`DistL2`]. Works for any vectors (mock or real) without
//! requiring unit-norm. We'll switch to `DistDot` (or `DistCosine`)
//! once we've wired TEI's `normalize=true` end-to-end and want the
//! ranking-equivalence-with-cheaper-arithmetic win on normalized
//! vectors.
//!
//! Quantization: the in-memory HNSW data type tracks the bucket's
//! [`crate::knowledge::vectors::VectorQuant`] choice. F32 buckets
//! keep `Hnsw<f32, DistL2>`; F16 and Int8 buckets use
//! `Hnsw<f16, DistF16L2>` (Int8 falls back to f16 because per-vector
//! scale prefixes don't fit hnsw_rs's typed `T` slot — switching to
//! f16 still halves `dense.hnsw.data` vs the old f32-only behavior).
//! The choice is recorded inside the hnsw_rs dump itself (its
//! deserialization is type-parameterized), so [`DenseIndex::load_from`]
//! re-derives the correct backend from the bucket's `VectorQuant`
//! and asks hnsw_rs for that specific instantiation.
//!
//! HNSW data ids are vector positions (the index into `vectors.bin`).
//! The position→chunk_id mapping is rebuilt at load time from
//! `vectors.idx` entries; this avoids a parallel persistent file
//! and keeps `vectors.idx` as the single source of truth for chunk
//! identity.
//!
//! On-disk layout: hnsw_rs dumps two sibling files,
//! `<basename>.hnsw.graph` (adjacency lists, ~16 edges/node × layer
//! count) and `<basename>.hnsw.data` (vectors duplicated from
//! `vectors.bin` — yes, the duplication is the cost of avoiding a
//! ~1-min-per-million-vectors HNSW rebuild). For wikipedia-scale
//! buckets (~30M chunks) the duplicate is the price of being able to
//! cold-start in seconds instead of hours.

use std::io;
use std::path::Path;
use std::sync::RwLock;

use half::f16;
use hnsw_rs::anndists::dist::distances::Distance;
use hnsw_rs::hnswio::HnswIo;
use hnsw_rs::prelude::*;

use super::types::{BucketError, ChunkId};
use super::vectors::{VectorQuant, VectorStoreReader};

/// File basename for the HNSW dump inside a slot directory. The
/// hnsw_rs library appends `.hnsw.graph` and `.hnsw.data` to this.
const DENSE_BASENAME: &str = "dense";

/// Sidecar file name. Contains a single u64 (LE) — the number of
/// chunks reflected in the accompanying `dense.hnsw.{graph,data}`
/// dump. Written *after* the hnsw files; presence of this file is
/// the signal that the dump is complete and trustworthy.
///
/// Resume reads this and compares to the last `BatchEmbedded`
/// checkpoint in `build.state`: if they match, the dump can be
/// loaded directly instead of rebuilding the HNSW from vectors.bin.
const DENSE_SNAPSHOT_NAME: &str = "dense.snapshot";

/// HNSW build parameters. Defaults match the design doc's targets
/// (M=16, ef_construction=200) — appropriate for Qwen3-Embedding-0.6B
/// (1024-dim) at workspace-to-wikipedia scales. `max_layer` follows the
/// hnsw_rs convention of 16 layers (cap on the geometric layer
/// distribution).
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// `M` — number of neighbours per layer. 16 is the standard default.
    /// Higher = better recall, more memory.
    pub max_nb_connection: usize,

    /// Effort during graph construction. Higher = better graph quality,
    /// slower build. 200 is a typical default.
    pub ef_construction: usize,

    /// Cap on the number of layers. Hnsw selects per-point layer via
    /// a geometric distribution; this is the upper bound.
    pub max_layer: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_nb_connection: 16,
            ef_construction: 200,
            max_layer: 16,
        }
    }
}

/// L2 distance over half-precision (`f16`) vectors. Promotes each
/// component pair to f32 for the squared-difference accumulator —
/// f16 has too little dynamic range to hold the running sum at
/// realistic embedding dimensions (1024 squared diffs of 0.05 is
/// already ~2.5, fine for f16; the sum across all 1024 dims is ~2560,
/// at the edge of f16's representable range).
///
/// Returns `sqrt(sum)` to match `anndists::DistL2`'s contract — both
/// the relative ordering and the absolute distance value (downstream
/// rerank / fusion) need to read consistently across f32 and f16
/// buckets.
#[derive(Default, Copy, Clone)]
pub struct DistF16L2;

impl Distance<f16> for DistF16L2 {
    fn eval(&self, va: &[f16], vb: &[f16]) -> f32 {
        debug_assert_eq!(va.len(), vb.len());
        let norm: f32 = va
            .iter()
            .zip(vb.iter())
            .map(|(a, b)| {
                let d = a.to_f32() - b.to_f32();
                d * d
            })
            .sum();
        norm.sqrt()
    }
}

/// HNSW backend type for a slot. Tracks the in-memory vector
/// representation that hnsw_rs holds — `f32` for plain buckets, `f16`
/// for any quantized bucket. Int8 vectors.bin storage uses per-vector
/// scale prefixes that don't fit hnsw_rs's typed `T`, so int8 buckets
/// fall back to f16 in HNSW (still halves `dense.hnsw.data` vs the
/// old f32-only path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseQuant {
    F32,
    F16,
}

impl DenseQuant {
    /// Map the bucket's vectors.bin quantization to the HNSW backend
    /// it should use. F32 → f32 HNSW (no quantization noise on the
    /// search side). F16 → f16 HNSW (the natural pairing). Int8 →
    /// f16 HNSW (per-vector-scale int8 doesn't fit a typed `T`; the
    /// f16 fallback still halves the dump file).
    pub fn from_vector_quant(q: VectorQuant) -> Self {
        match q {
            VectorQuant::F32 => DenseQuant::F32,
            VectorQuant::F16 | VectorQuant::Int8 => DenseQuant::F16,
        }
    }
}

/// In-memory dense index over a slot's vectors.
///
/// Owns the `Hnsw` graph and a position→chunk_id translation table.
/// Mutable interior — `insert` takes a write lock, `search` takes a
/// read lock. This is what makes per-batch incremental insert during a
/// build work while queries against the same slot stay concurrent.
///
/// Persisted to `<slot>/dense.hnsw.{graph,data}` via [`Self::dump_to`].
/// On slot reload, [`Self::load_from`] reads those files; if they're
/// missing or stale, [`Self::build`] reconstructs from
/// `vectors.bin`/`vectors.idx`.
pub struct DenseIndex {
    /// Hnsw graph + by_position table held under one RwLock so
    /// inserts are atomic across both. Cheap reads (RwLock::read +
    /// hnsw.search internally read-only) — query latency is
    /// dominated by the search itself, not the lock.
    inner: RwLock<DenseInner>,
}

/// `'static` because Hnsw owns its inserted data via internal
/// reference-counting; we don't keep external references alive.
///
/// Branch on the on-disk vector type. Both arms keep the same
/// `by_position` translation table — its shape is independent of
/// the dense vector representation, so callers don't have to care
/// which arm is active.
enum DenseInner {
    F32 {
        hnsw: Hnsw<'static, f32, DistL2>,
        /// `by_position[d_id]` is the chunk id corresponding to HNSW
        /// data id `d_id`. Grown to position+1 on each insert; entries
        /// between the old length and the new position are filled with
        /// the zero chunk id (not normally observable since search
        /// results are filtered by position validity).
        by_position: Vec<ChunkId>,
    },
    F16 {
        hnsw: Hnsw<'static, f16, DistF16L2>,
        by_position: Vec<ChunkId>,
    },
}

impl DenseInner {
    fn by_position(&self) -> &[ChunkId] {
        match self {
            DenseInner::F32 { by_position, .. } | DenseInner::F16 { by_position, .. } => {
                by_position
            }
        }
    }

    fn by_position_mut(&mut self) -> &mut Vec<ChunkId> {
        match self {
            DenseInner::F32 { by_position, .. } | DenseInner::F16 { by_position, .. } => {
                by_position
            }
        }
    }
}

impl std::fmt::Debug for DenseIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        let (len, quant) = match inner.as_deref() {
            Ok(DenseInner::F32 { by_position, .. }) => (by_position.len(), DenseQuant::F32),
            Ok(DenseInner::F16 { by_position, .. }) => (by_position.len(), DenseQuant::F16),
            Err(_) => (0, DenseQuant::F32),
        };
        f.debug_struct("DenseIndex")
            .field("len", &len)
            .field("quant", &quant)
            .finish_non_exhaustive()
    }
}

impl DenseIndex {
    /// Construct an empty index for the given dense backend, sized
    /// with `max_elements_hint` as a hint to hnsw_rs's internal
    /// allocator. Inserts beyond the hint still work — hnsw_rs treats
    /// it as a sizing suggestion, not a hard cap — but undersizing
    /// forces reallocation. Tune to the expected chunk count when
    /// known; over-estimate is cheap.
    pub fn empty(quant: DenseQuant, params: HnswParams, max_elements_hint: usize) -> Self {
        let inner = match quant {
            DenseQuant::F32 => {
                let hnsw: Hnsw<'static, f32, DistL2> = Hnsw::new(
                    params.max_nb_connection,
                    max_elements_hint.max(1),
                    params.max_layer,
                    params.ef_construction,
                    DistL2 {},
                );
                DenseInner::F32 {
                    hnsw,
                    by_position: Vec::new(),
                }
            }
            DenseQuant::F16 => {
                let hnsw: Hnsw<'static, f16, DistF16L2> = Hnsw::new(
                    params.max_nb_connection,
                    max_elements_hint.max(1),
                    params.max_layer,
                    params.ef_construction,
                    DistF16L2,
                );
                DenseInner::F16 {
                    hnsw,
                    by_position: Vec::new(),
                }
            }
        };
        Self {
            inner: RwLock::new(inner),
        }
    }

    /// Quantization backend in use.
    pub fn quant(&self) -> DenseQuant {
        match &*self.inner.read().expect("DenseIndex lock poisoned") {
            DenseInner::F32 { .. } => DenseQuant::F32,
            DenseInner::F16 { .. } => DenseQuant::F16,
        }
    }

    /// Insert one vector at `position` (the index into vectors.bin).
    /// Takes a write lock for the brief duration of the HNSW insert
    /// and by_position update. Caller is responsible for monotonically
    /// increasing positions starting from 0; gaps are tolerated but the
    /// rest of the codebase doesn't currently produce them.
    ///
    /// Vectors come in as f32 regardless of the backend; the f16 path
    /// converts before insert. Per-call conversion cost is small at
    /// realistic dims (1024 f32→f16 promotions ≈ a few µs).
    pub fn insert(&self, chunk_id: ChunkId, position: u64, vector: &[f32]) {
        let mut inner = self.inner.write().expect("DenseIndex lock poisoned");
        match &mut *inner {
            DenseInner::F32 { hnsw, .. } => {
                hnsw.insert((vector, position as usize));
            }
            DenseInner::F16 { hnsw, .. } => {
                let v16: Vec<f16> = vector.iter().map(|&x| f16::from_f32(x)).collect();
                hnsw.insert((&v16, position as usize));
            }
        }
        let by_position = inner.by_position_mut();
        if by_position.len() <= position as usize {
            // Pad with the zero chunk id; HNSW's filter_map in search
            // still produces correct results because we only return
            // by_position[d_id] entries that came from real inserts
            // (any d_id we didn't insert wouldn't appear in HNSW
            // search output).
            by_position.resize(position as usize + 1, ChunkId([0; 32]));
        }
        by_position[position as usize] = chunk_id;
    }

    /// Build a fresh index from a vectors store. Reads each vector
    /// once in position order and inserts into the HNSW graph. Used
    /// on slot reload when a dump is missing/stale.
    ///
    /// The HNSW backend is selected from the vectors.bin quantization
    /// — see [`DenseQuant::from_vector_quant`] for the mapping.
    pub fn build(vectors: &VectorStoreReader, params: HnswParams) -> Result<Self, BucketError> {
        let count = vectors.len();
        let quant = DenseQuant::from_vector_quant(vectors.quant());

        let mut pairs: Vec<(u64, ChunkId)> = vectors
            .iter_chunk_positions()
            .map(|(id, pos)| (pos, id))
            .collect();
        pairs.sort_by_key(|(pos, _)| *pos);

        let index = Self::empty(quant, params, count);
        for (position, chunk_id) in pairs.iter().copied() {
            let vector = vectors
                .read_at_position(position)
                .map_err(BucketError::Io)?;
            index.insert(chunk_id, position, &vector);
        }
        debug_assert_eq!(index.len(), count);
        Ok(index)
    }

    pub fn len(&self) -> usize {
        self.inner
            .read()
            .map(|i| i.by_position().len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Search for the `top_k` nearest neighbours of `query`. Returns
    /// `(chunk_id, distance)` pairs ordered by ascending distance
    /// (closest first).
    ///
    /// Empty when the index is empty. `ef_search` is computed as
    /// `max(top_k * 2, 64)` — generous enough to keep recall high at
    /// workspace scales without paying much for the over-fetch.
    ///
    /// Query is always f32; the f16 backend promotes it once before
    /// dispatch. The conversion is hot-path-cheap (one pass over the
    /// query vector, no allocation per HNSW edge probe).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(ChunkId, f32)> {
        let inner = self.inner.read().expect("DenseIndex lock poisoned");
        if inner.by_position().is_empty() || top_k == 0 {
            return Vec::new();
        }
        let ef = (top_k * 2).max(64);
        match &*inner {
            DenseInner::F32 { hnsw, by_position } => {
                let neighbours = hnsw.search(query, top_k, ef);
                neighbours
                    .into_iter()
                    .filter_map(|n| by_position.get(n.d_id).map(|&id| (id, n.distance)))
                    .collect()
            }
            DenseInner::F16 { hnsw, by_position } => {
                let q16: Vec<f16> = query.iter().map(|&x| f16::from_f32(x)).collect();
                let neighbours = hnsw.search(&q16, top_k, ef);
                neighbours
                    .into_iter()
                    .filter_map(|n| by_position.get(n.d_id).map(|&id| (id, n.distance)))
                    .collect()
            }
        }
    }

    /// Dump the in-memory HNSW graph to `slot_dir`. Writes
    /// `dense.hnsw.graph` and `dense.hnsw.data`. Used both as a
    /// per-checkpoint barrier during long builds and as the
    /// final-build artifact a future slot-open uses to skip the
    /// rebuild.
    ///
    /// Stale on-disk dumps (including the sidecar) are removed first so
    /// `file_dump` doesn't silently fall back to a random-suffixed
    /// basename when prior files exist (its anti-mmap-clobber default),
    /// and so a partial dump can never be mistaken for a complete one.
    /// The sidecar is the *last* file written — its presence means the
    /// hnsw files reflect the recorded chunk count.
    pub fn dump_to(&self, slot_dir: &Path) -> Result<(), BucketError> {
        let snapshot_path = slot_dir.join(DENSE_SNAPSHOT_NAME);
        if snapshot_path.exists() {
            std::fs::remove_file(&snapshot_path).map_err(BucketError::Io)?;
        }
        for ext in [".hnsw.graph", ".hnsw.data"] {
            let p = slot_dir.join(format!("{DENSE_BASENAME}{ext}"));
            if p.exists() {
                std::fs::remove_file(&p).map_err(BucketError::Io)?;
            }
        }
        // Hold the read lock while dumping so concurrent inserts don't
        // grow the HNSW between file_dump and reading get_nb_point —
        // sidecar must reflect exactly what's in the dump files.
        let snapshot_count = {
            let inner = self.inner.read().expect("DenseIndex lock poisoned");
            match &*inner {
                DenseInner::F32 { hnsw, .. } => {
                    hnsw.file_dump(slot_dir, DENSE_BASENAME)
                        .map_err(|e| BucketError::Other(format!("hnsw file_dump: {e}")))?;
                    hnsw.get_nb_point() as u64
                }
                DenseInner::F16 { hnsw, .. } => {
                    hnsw.file_dump(slot_dir, DENSE_BASENAME)
                        .map_err(|e| BucketError::Other(format!("hnsw file_dump: {e}")))?;
                    hnsw.get_nb_point() as u64
                }
            }
        };
        std::fs::write(&snapshot_path, snapshot_count.to_le_bytes()).map_err(BucketError::Io)?;
        Ok(())
    }

    /// Load a previously-dumped HNSW from `slot_dir`, pairing it with
    /// the `vectors`'s `chunk_id`-by-position table so search results
    /// can map back to chunk ids the same way `build` does.
    ///
    /// Returns `BucketError::Other` (with a `not found` substring) when
    /// the dump files are missing — callers that want a "load or
    /// rebuild" semantic should match on this and fall back to
    /// [`Self::build`].
    pub fn load_from(slot_dir: &Path, vectors: &VectorStoreReader) -> Result<Self, BucketError> {
        let graph_path = slot_dir.join(format!("{DENSE_BASENAME}.hnsw.graph"));
        let data_path = slot_dir.join(format!("{DENSE_BASENAME}.hnsw.data"));
        if !graph_path.exists() || !data_path.exists() {
            return Err(BucketError::Other(format!(
                "hnsw dump not found in {} (looking for {DENSE_BASENAME}.hnsw.graph + .data)",
                slot_dir.display(),
            )));
        }

        // Rebuild the position→chunk_id table from vectors.idx —
        // same shape as `build`, just without inserting into HNSW.
        let mut pairs: Vec<(u64, ChunkId)> = vectors
            .iter_chunk_positions()
            .map(|(id, pos)| (pos, id))
            .collect();
        pairs.sort_by_key(|(pos, _)| *pos);
        let by_position: Vec<ChunkId> = pairs.into_iter().map(|(_, id)| id).collect();

        let quant = DenseQuant::from_vector_quant(vectors.quant());
        load_with_quant_and_positions(slot_dir, by_position, quant)
    }

    /// Load a dumped HNSW with a caller-supplied `by_position` table.
    ///
    /// Used by the resume path during a build, where `vectors.idx`
    /// hasn't been written yet — the by_position list comes from
    /// re-running the chunker over the durable Planned records.
    /// Otherwise behaves identically to [`Self::load_from`]: the
    /// hnsw_rs files must exist, and the dump's point count must
    /// match `by_position.len()`.
    ///
    /// `quant` selects the HNSW backend; resume callers know it from
    /// the bucket's manifest (or their build config) before any vectors
    /// reader is open.
    pub fn load_from_with_positions(
        slot_dir: &Path,
        by_position: Vec<ChunkId>,
        quant: DenseQuant,
    ) -> Result<Self, BucketError> {
        let graph_path = slot_dir.join(format!("{DENSE_BASENAME}.hnsw.graph"));
        let data_path = slot_dir.join(format!("{DENSE_BASENAME}.hnsw.data"));
        if !graph_path.exists() || !data_path.exists() {
            return Err(BucketError::Other(format!(
                "hnsw dump not found in {} (looking for {DENSE_BASENAME}.hnsw.graph + .data)",
                slot_dir.display(),
            )));
        }
        load_with_quant_and_positions(slot_dir, by_position, quant)
    }
}

/// Shared implementation for the two `load_from*` entry points.
/// Branches on `quant` to ask hnsw_rs for the correct typed backend.
fn load_with_quant_and_positions(
    slot_dir: &Path,
    by_position: Vec<ChunkId>,
    quant: DenseQuant,
) -> Result<DenseIndex, BucketError> {
    // The hnsw_rs `load_hnsw` signature constrains the returned
    // Hnsw to outlive its HnswIo source. We need `Hnsw<'static>`
    // for `DenseIndex` to stay loadable into a long-lived registry
    // cache, so we leak the HnswIo (a ~100-byte struct with a
    // PathBuf, basename, and an Arc counter) to give it 'static.
    // One-time, slot-load-only allocation; doesn't accumulate
    // across query traffic.
    let io: &'static mut HnswIo = Box::leak(Box::new(HnswIo::new(slot_dir, DENSE_BASENAME)));
    let inner = match quant {
        DenseQuant::F32 => {
            let hnsw: Hnsw<'static, f32, DistL2> = io
                .load_hnsw()
                .map_err(|e| BucketError::Other(format!("hnsw load: {e}")))?;
            let hnsw_count = hnsw.get_nb_point();
            if hnsw_count != by_position.len() {
                return Err(BucketError::Other(format!(
                    "hnsw dump count {hnsw_count} does not match expected count {}",
                    by_position.len(),
                )));
            }
            DenseInner::F32 { hnsw, by_position }
        }
        DenseQuant::F16 => {
            let hnsw: Hnsw<'static, f16, DistF16L2> = io
                .load_hnsw()
                .map_err(|e| BucketError::Other(format!("hnsw load: {e}")))?;
            let hnsw_count = hnsw.get_nb_point();
            if hnsw_count != by_position.len() {
                return Err(BucketError::Other(format!(
                    "hnsw dump count {hnsw_count} does not match expected count {}",
                    by_position.len(),
                )));
            }
            DenseInner::F16 { hnsw, by_position }
        }
    };
    Ok(DenseIndex {
        inner: RwLock::new(inner),
    })
}

/// Read the sidecar `dense.snapshot` file written by [`DenseIndex::dump_to`].
/// Returns `Ok(None)` when the sidecar is missing — that means either
/// no dump has been written yet, or a partial dump landed without
/// finishing. Either way the caller should fall back to rebuilding
/// from vectors.bin rather than trusting the on-disk hnsw files.
pub fn read_dump_snapshot(slot_dir: &Path) -> io::Result<Option<u64>> {
    let path = slot_dir.join(DENSE_SNAPSHOT_NAME);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    if bytes.len() != 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("dense.snapshot has {} bytes, expected 8", bytes.len()),
        ));
    }
    Ok(Some(u64::from_le_bytes(bytes.try_into().unwrap())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::VectorStoreWriter;

    fn build_test_vectors(dim: u32, n: usize) -> tempfile::TempDir {
        build_test_vectors_quant(dim, n, crate::knowledge::vectors::VectorQuant::F32)
    }

    fn build_test_vectors_quant(
        dim: u32,
        n: usize,
        quant: crate::knowledge::vectors::VectorQuant,
    ) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let mut w = VectorStoreWriter::create(&bin, &idx, dim, quant).unwrap();
        for i in 0..n {
            let id = ChunkId::from_source(&[i as u8; 32], i as u64);
            // Distinctive vector: one-hot at position i % dim, scaled by i.
            let mut v = vec![0.0f32; dim as usize];
            v[i % dim as usize] = (i + 1) as f32;
            w.append(id, &v).unwrap();
        }
        w.finalize().unwrap();
        tmp
    }

    fn open_reader(tmp: &tempfile::TempDir) -> VectorStoreReader {
        VectorStoreReader::open(
            &tmp.path().join("vectors.bin"),
            &tmp.path().join("vectors.idx"),
        )
        .unwrap()
    }

    #[test]
    fn build_and_search_returns_nearest_neighbour() {
        let tmp = build_test_vectors(8, 50);
        let vectors = open_reader(&tmp);
        let index = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        assert_eq!(index.len(), 50);

        // Query exactly matches the vector at position 5: one-hot at
        // dim 5, scaled by 6. We expect that chunk id back as the top
        // result with distance ~0.
        let mut query = vec![0.0f32; 8];
        query[5] = 6.0;
        let results = index.search(&query, 5);
        assert!(!results.is_empty());

        let expected_id = ChunkId::from_source(&[5u8; 32], 5);
        assert_eq!(results[0].0, expected_id);
        assert!(
            results[0].1 < 0.001,
            "distance to exact match should be ~0; got {}",
            results[0].1
        );
    }

    #[test]
    fn search_returns_sorted_by_distance() {
        let tmp = build_test_vectors(8, 50);
        let vectors = open_reader(&tmp);
        let index = DenseIndex::build(&vectors, HnswParams::default()).unwrap();

        let mut query = vec![0.0f32; 8];
        query[3] = 1.5;
        let results = index.search(&query, 10);
        assert!(!results.is_empty());

        for w in results.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "results must be sorted ascending by distance, got {} then {}",
                w[0].1,
                w[1].1,
            );
        }
    }

    #[test]
    fn empty_index_search_returns_empty() {
        let tmp = build_test_vectors(8, 0);
        let vectors = open_reader(&tmp);
        let index = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        assert!(index.is_empty());
        assert!(index.search(&[0.0; 8], 5).is_empty());
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let tmp = build_test_vectors(8, 10);
        let vectors = open_reader(&tmp);
        let index = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        assert!(index.search(&[0.0; 8], 0).is_empty());
    }

    #[test]
    fn dump_and_load_roundtrip_returns_same_top_neighbour() {
        let tmp = build_test_vectors(8, 50);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        built.dump_to(tmp.path()).unwrap();
        assert!(tmp.path().join("dense.hnsw.graph").exists());
        assert!(tmp.path().join("dense.hnsw.data").exists());

        let loaded = DenseIndex::load_from(tmp.path(), &vectors).unwrap();
        assert_eq!(loaded.len(), built.len());

        // Same exact-match query both indexes should agree on.
        let mut query = vec![0.0f32; 8];
        query[5] = 6.0;
        let from_built = built.search(&query, 1);
        let from_loaded = loaded.search(&query, 1);
        assert_eq!(from_built[0].0, from_loaded[0].0);
        assert!(from_loaded[0].1 < 0.001);
    }

    #[test]
    fn load_from_missing_dump_returns_error() {
        let tmp = build_test_vectors(8, 10);
        let vectors = open_reader(&tmp);
        // No dump_to call; load should fail with a descriptive error.
        let err = DenseIndex::load_from(tmp.path(), &vectors).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found") || msg.contains("dump"),
            "unexpected error message: {msg}",
        );
    }

    #[test]
    fn dump_writes_sidecar_with_point_count() {
        // dump_to() must write dense.snapshot last, with the LE u64
        // matching the HNSW's point count. Resume relies on this for
        // checkpoint matching.
        let tmp = build_test_vectors(8, 17);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        built.dump_to(tmp.path()).unwrap();

        let snap = read_dump_snapshot(tmp.path()).unwrap();
        assert_eq!(snap, Some(17));
    }

    #[test]
    fn read_dump_snapshot_missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(read_dump_snapshot(tmp.path()).unwrap(), None);
    }

    #[test]
    fn read_dump_snapshot_rejects_wrong_size() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("dense.snapshot"), b"oops").unwrap();
        let err = read_dump_snapshot(tmp.path()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn dump_clears_stale_sidecar_before_writing_files() {
        // If a prior run wrote a sidecar but the current dump fails
        // mid-write, no leftover sidecar can claim "the dump is OK".
        // We simulate by manually planting a stale sidecar, then
        // verifying dump_to() replaces it with the correct value.
        let tmp = build_test_vectors(8, 5);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();

        std::fs::write(tmp.path().join("dense.snapshot"), 999u64.to_le_bytes()).unwrap();
        built.dump_to(tmp.path()).unwrap();
        assert_eq!(read_dump_snapshot(tmp.path()).unwrap(), Some(5));
    }

    #[test]
    fn load_from_with_positions_roundtrips() {
        // The resume path uses load_from_with_positions because
        // vectors.idx isn't written until finalize. Same hnsw, but
        // by_position comes from chunk-id rederivation rather than
        // disk.
        let tmp = build_test_vectors(8, 20);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        built.dump_to(tmp.path()).unwrap();

        let mut pairs: Vec<(u64, ChunkId)> = vectors
            .iter_chunk_positions()
            .map(|(id, pos)| (pos, id))
            .collect();
        pairs.sort_by_key(|(pos, _)| *pos);
        let by_position: Vec<ChunkId> = pairs.into_iter().map(|(_, id)| id).collect();

        let loaded =
            DenseIndex::load_from_with_positions(tmp.path(), by_position, DenseQuant::F32).unwrap();
        assert_eq!(loaded.len(), built.len());

        let mut q = vec![0.0f32; 8];
        q[3] = 4.0;
        let from_built = built.search(&q, 1);
        let from_loaded = loaded.search(&q, 1);
        assert_eq!(from_built[0].0, from_loaded[0].0);
    }

    #[test]
    fn load_from_with_positions_rejects_count_mismatch() {
        let tmp = build_test_vectors(8, 10);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        built.dump_to(tmp.path()).unwrap();

        // Hand in a too-short by_position list — load must reject.
        let too_short = vec![ChunkId::from_source(&[1; 32], 0); 5];
        let err = DenseIndex::load_from_with_positions(tmp.path(), too_short, DenseQuant::F32)
            .unwrap_err();
        assert!(
            err.to_string().contains("does not match"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn dump_overwrites_existing_files() {
        let tmp = build_test_vectors(8, 20);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        built.dump_to(tmp.path()).unwrap();
        // Second dump on the same dir must succeed (and not silently
        // create dense-XXX.hnsw.* with a random suffix — that
        // happens when we don't pre-clean).
        built.dump_to(tmp.path()).unwrap();
        // Exactly one pair of files; no random-suffixed siblings.
        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("hnsw"))
            .collect();
        assert_eq!(
            entries.len(),
            2,
            "expected exactly dense.hnsw.graph + .data, got {entries:?}",
        );
    }

    #[test]
    fn build_is_deterministic_in_inputs() {
        // Same vectors → same chunk ids. We don't assert ordering of
        // results across builds (HNSW search has heuristic component),
        // but we do assert that both builds find the exact-match
        // chunk first.
        let tmp = build_test_vectors(16, 30);
        let vectors = open_reader(&tmp);
        let a = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        let b = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        assert_eq!(a.len(), b.len());

        let mut query = vec![0.0f32; 16];
        query[7] = 8.0;
        let ra = a.search(&query, 1);
        let rb = b.search(&query, 1);
        assert_eq!(ra[0].0, rb[0].0);
    }

    #[test]
    fn dense_quant_from_vector_quant_maps_int8_to_f16() {
        use crate::knowledge::vectors::VectorQuant;
        // F32 stays F32 (no quantization noise on the search side);
        // F16 and Int8 both land on f16 HNSW (per-vector-scale Int8
        // doesn't fit hnsw_rs's typed T, so the f16 fallback is the
        // best available).
        assert_eq!(
            DenseQuant::from_vector_quant(VectorQuant::F32),
            DenseQuant::F32
        );
        assert_eq!(
            DenseQuant::from_vector_quant(VectorQuant::F16),
            DenseQuant::F16
        );
        assert_eq!(
            DenseQuant::from_vector_quant(VectorQuant::Int8),
            DenseQuant::F16
        );
    }

    #[test]
    fn build_from_f16_vectors_uses_f16_backend() {
        let tmp = build_test_vectors_quant(8, 50, crate::knowledge::vectors::VectorQuant::F16);
        let vectors = open_reader(&tmp);
        let index = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        assert_eq!(index.len(), 50);
        assert_eq!(index.quant(), DenseQuant::F16);

        // Same exact-match query as the f32 case — f16 has enough
        // precision at small dim that the top result still has
        // distance ~0.
        let mut query = vec![0.0f32; 8];
        query[5] = 6.0;
        let results = index.search(&query, 5);
        assert!(!results.is_empty());

        let expected_id = ChunkId::from_source(&[5u8; 32], 5);
        assert_eq!(results[0].0, expected_id);
        assert!(
            results[0].1 < 0.001,
            "distance to exact match should be ~0; got {}",
            results[0].1,
        );
    }

    #[test]
    fn build_from_int8_vectors_uses_f16_backend() {
        // Int8 vectors.bin → f16 HNSW. The vectors.bin int8 path
        // already loses some precision at the per-vector-scale step,
        // so we only check that the top-1 nearest neighbour is the
        // exact match (its quantized vector is also one-hot).
        let tmp = build_test_vectors_quant(8, 50, crate::knowledge::vectors::VectorQuant::Int8);
        let vectors = open_reader(&tmp);
        let index = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        assert_eq!(index.len(), 50);
        assert_eq!(index.quant(), DenseQuant::F16);

        let mut query = vec![0.0f32; 8];
        query[5] = 6.0;
        let results = index.search(&query, 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, ChunkId::from_source(&[5u8; 32], 5));
    }

    #[test]
    fn dump_and_load_roundtrip_f16() {
        // Same shape as the f32 roundtrip test but with f16 backend
        // — verifies the on-disk format round-trips through
        // hnsw_rs's typed serde.
        let tmp = build_test_vectors_quant(8, 50, crate::knowledge::vectors::VectorQuant::F16);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        assert_eq!(built.quant(), DenseQuant::F16);
        built.dump_to(tmp.path()).unwrap();

        let loaded = DenseIndex::load_from(tmp.path(), &vectors).unwrap();
        assert_eq!(loaded.quant(), DenseQuant::F16);
        assert_eq!(loaded.len(), built.len());

        let mut query = vec![0.0f32; 8];
        query[5] = 6.0;
        let from_built = built.search(&query, 1);
        let from_loaded = loaded.search(&query, 1);
        assert_eq!(from_built[0].0, from_loaded[0].0);
    }

    #[test]
    fn f16_dump_is_smaller_than_f32_dump() {
        // The whole point of this slice — f16 dense.hnsw.data must
        // be roughly half the size of f32 for the same vectors.
        // Exact ratio depends on hnsw_rs framing overhead so we
        // assert a generous bound (data file < 70% of f32 case).
        let dim: u32 = 64;
        let n = 200;

        let f32_tmp = build_test_vectors_quant(dim, n, crate::knowledge::vectors::VectorQuant::F32);
        let f32_vec = open_reader(&f32_tmp);
        DenseIndex::build(&f32_vec, HnswParams::default())
            .unwrap()
            .dump_to(f32_tmp.path())
            .unwrap();
        let f32_data = std::fs::metadata(f32_tmp.path().join("dense.hnsw.data"))
            .unwrap()
            .len();

        let f16_tmp = build_test_vectors_quant(dim, n, crate::knowledge::vectors::VectorQuant::F16);
        let f16_vec = open_reader(&f16_tmp);
        DenseIndex::build(&f16_vec, HnswParams::default())
            .unwrap()
            .dump_to(f16_tmp.path())
            .unwrap();
        let f16_data = std::fs::metadata(f16_tmp.path().join("dense.hnsw.data"))
            .unwrap()
            .len();

        let ratio = f16_data as f64 / f32_data as f64;
        assert!(
            ratio < 0.7,
            "f16 dense.hnsw.data ({f16_data}) should be < 70% of f32 ({f32_data}); got ratio={ratio:.3}",
        );
    }

    #[test]
    fn load_from_with_positions_f16_roundtrips() {
        // Resume path uses load_from_with_positions because vectors.idx
        // isn't written until finalize. Exercise the f16 branch with a
        // caller-supplied by_position list.
        let tmp = build_test_vectors_quant(8, 20, crate::knowledge::vectors::VectorQuant::F16);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        built.dump_to(tmp.path()).unwrap();

        let mut pairs: Vec<(u64, ChunkId)> = vectors
            .iter_chunk_positions()
            .map(|(id, pos)| (pos, id))
            .collect();
        pairs.sort_by_key(|(pos, _)| *pos);
        let by_position: Vec<ChunkId> = pairs.into_iter().map(|(_, id)| id).collect();

        let loaded =
            DenseIndex::load_from_with_positions(tmp.path(), by_position, DenseQuant::F16).unwrap();
        assert_eq!(loaded.quant(), DenseQuant::F16);
        assert_eq!(loaded.len(), built.len());

        let mut q = vec![0.0f32; 8];
        q[3] = 4.0;
        let from_built = built.search(&q, 1);
        let from_loaded = loaded.search(&q, 1);
        assert_eq!(from_built[0].0, from_loaded[0].0);
    }
}
