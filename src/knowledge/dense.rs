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
//! [`crate::knowledge::vectors::VectorQuant`] choice. F32 →
//! `Hnsw<f32, DistL2>`, F16 → `Hnsw<f16, DistF16L2>`, Int8 →
//! `Hnsw<i8, DistInt8L2>`. The int8 backend uses a single
//! dataset-wide scale (calibrated from the first batch of vectors
//! during build, persisted in the `dense.int8_scale` sidecar); for
//! L2 *ordering* the scale's `s²` constant factor cancels, so
//! `DistInt8L2` is just `(a − b)²` over i8 values with i32 arithmetic
//! to avoid underflow on `i8::MIN`. The hnsw_rs dump is type-
//! parameterized, so [`DenseIndex::load_from`] re-derives the correct
//! backend from the bucket's `VectorQuant` and asks hnsw_rs for that
//! specific instantiation.
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

/// L2 distance over signed-byte (`i8`) vectors. The vectors are
/// already-quantized i8 representations of unit-norm-ish embeddings;
/// the f32→i8 scale is implicit and identical for both inputs, so
/// L2 *ordering* over the i8 representation is preserved (constant
/// `scale²` factor falls out of `(s·a − s·b)² = s²·(a−b)²`). The
/// returned value is in i8-units, not real-world units — fine for
/// ranking / chunk-id-set comparison; downstream rerank uses its
/// own scoring.
///
/// i8 difference computed in i32 to avoid the `-128 - 1` underflow
/// edge case. Squared diff fits in 17 bits, sum across 1024 dims
/// fits comfortably in i64.
#[derive(Default, Copy, Clone)]
pub struct DistInt8L2;

impl Distance<i8> for DistInt8L2 {
    fn eval(&self, va: &[i8], vb: &[i8]) -> f32 {
        debug_assert_eq!(va.len(), vb.len());
        let norm: i64 = va
            .iter()
            .zip(vb.iter())
            .map(|(a, b)| {
                let d = *a as i32 - *b as i32;
                (d * d) as i64
            })
            .sum();
        (norm as f32).sqrt()
    }
}

/// HNSW backend type for a slot. Tracks the in-memory vector
/// representation that hnsw_rs holds. Mirrors the bucket's
/// `serving.quantization` choice: F32 → `Hnsw<f32, DistL2>`,
/// F16 → `Hnsw<f16, DistF16L2>`, Int8 → `Hnsw<i8, DistInt8L2>`.
///
/// The int8 backend uses a single dataset-wide scale (calibrated
/// from the first batch of vectors during build, persisted in the
/// `dense.int8_scale` sidecar). It assumes inputs are roughly
/// unit-norm — true for typical text embedders like
/// Qwen3-Embedding-0.6B; non-normalized inputs would saturate at
/// ±127 and lose recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseQuant {
    F32,
    F16,
    Int8,
}

impl DenseQuant {
    /// Map the bucket's vectors.bin quantization to the HNSW backend
    /// it should use. F32 → f32 HNSW (no quantization noise). F16 →
    /// f16 HNSW. Int8 → int8 HNSW (quarters `dense.hnsw.data` vs f32).
    pub fn from_vector_quant(q: VectorQuant) -> Self {
        match q {
            VectorQuant::F32 => DenseQuant::F32,
            VectorQuant::F16 => DenseQuant::F16,
            VectorQuant::Int8 => DenseQuant::Int8,
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
/// Branch on the on-disk vector type. All arms keep the same
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
    Int8 {
        hnsw: Hnsw<'static, i8, DistInt8L2>,
        by_position: Vec<ChunkId>,
        /// f32-to-i8 scale: `i8 = clamp(round(f32 / scale), -127, 127)`.
        /// `None` until calibrated from the first batch of vectors via
        /// [`DenseIndex::set_int8_scale`]; after that the value is
        /// fixed for the lifetime of this index. Persisted in the
        /// `dense.int8_scale` sidecar so a reload reuses the same scale
        /// and quantizes queries identically.
        scale: Option<f32>,
    },
}

impl DenseInner {
    fn by_position(&self) -> &[ChunkId] {
        match self {
            DenseInner::F32 { by_position, .. }
            | DenseInner::F16 { by_position, .. }
            | DenseInner::Int8 { by_position, .. } => by_position,
        }
    }

    fn by_position_mut(&mut self) -> &mut Vec<ChunkId> {
        match self {
            DenseInner::F32 { by_position, .. }
            | DenseInner::F16 { by_position, .. }
            | DenseInner::Int8 { by_position, .. } => by_position,
        }
    }
}

/// File name for the int8 backend's f32→i8 scale sidecar. Single
/// LE f32 (4 bytes). Written by [`DenseIndex::dump_to`] right next to
/// `dense.hnsw.{graph,data}`; read by [`DenseIndex::load_from`] when
/// the bucket's `VectorQuant::Int8` directs us to the int8 backend.
///
/// Absent on F32 / F16 buckets — its presence implicitly marks an
/// int8 dump.
const DENSE_INT8_SCALE_NAME: &str = "dense.int8_scale";

/// Quantize one f32 vector into i8 using `scale`. Standard symmetric
/// rounding-then-clamp; matches `vectors.rs`'s int8 encoding shape but
/// uses the dataset-wide scale (vectors.bin uses per-vector scale).
fn quantize_f32_to_i8(vector: &[f32], scale: f32) -> Vec<i8> {
    debug_assert!(scale > 0.0, "int8 scale must be positive, got {scale}");
    let inv = 1.0 / scale;
    vector
        .iter()
        .map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Compute the int8 calibration scale for a sample of f32 vectors.
/// Returns `max_abs * 1.1 / 127` — 10% headroom keeps later batches
/// (whose max-abs may slightly exceed the calibration sample) from
/// saturating at ±127. The `1.0e-6` floor guards against the all-
/// zero edge case (avoids dividing by zero at quantize time).
pub(super) fn calibrate_int8_scale(samples: &[Vec<f32>]) -> f32 {
    let max_abs: f32 = samples
        .iter()
        .flat_map(|v| v.iter())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    (max_abs * 1.1 / 127.0).max(1.0e-6)
}

impl std::fmt::Debug for DenseIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        let (len, quant) = match inner.as_deref() {
            Ok(DenseInner::F32 { by_position, .. }) => (by_position.len(), DenseQuant::F32),
            Ok(DenseInner::F16 { by_position, .. }) => (by_position.len(), DenseQuant::F16),
            Ok(DenseInner::Int8 { by_position, .. }) => (by_position.len(), DenseQuant::Int8),
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
    ///
    /// For [`DenseQuant::Int8`] backends, the f32→i8 scale starts
    /// uncalibrated; the caller must invoke [`Self::set_int8_scale`]
    /// (typically with [`calibrate_int8_scale`] applied to the first
    /// batch of vectors) before any inserts. Calling `insert` /
    /// `search` against an uncalibrated int8 index panics in debug
    /// and returns wrong results in release — the contract is "set
    /// the scale once, then proceed."
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
            DenseQuant::Int8 => {
                let hnsw: Hnsw<'static, i8, DistInt8L2> = Hnsw::new(
                    params.max_nb_connection,
                    max_elements_hint.max(1),
                    params.max_layer,
                    params.ef_construction,
                    DistInt8L2,
                );
                DenseInner::Int8 {
                    hnsw,
                    by_position: Vec::new(),
                    scale: None,
                }
            }
        };
        Self {
            inner: RwLock::new(inner),
        }
    }

    /// Set the f32→i8 scale on an int8 backend. Must be called exactly
    /// once before the first insert / search. Idempotent at the same
    /// scale value is fine; setting a different scale after the fact
    /// would corrupt the index (already-inserted vectors were quantized
    /// at the old scale), so we reject it.
    ///
    /// Panics if called on a non-int8 backend — the call shouldn't
    /// happen and a panic surfaces the wiring bug rather than silently
    /// proceeding with a wrong scale.
    pub fn set_int8_scale(&self, new_scale: f32) -> Result<(), BucketError> {
        debug_assert!(
            new_scale > 0.0,
            "int8 scale must be positive, got {new_scale}"
        );
        let mut inner = self.inner.write().expect("DenseIndex lock poisoned");
        match &mut *inner {
            DenseInner::Int8 { scale, .. } => match *scale {
                None => {
                    *scale = Some(new_scale);
                    Ok(())
                }
                Some(existing) if (existing - new_scale).abs() < f32::EPSILON => Ok(()),
                Some(existing) => Err(BucketError::Other(format!(
                    "int8 scale already set to {existing}; cannot reset to {new_scale}",
                ))),
            },
            _ => panic!("set_int8_scale called on non-int8 DenseIndex backend"),
        }
    }

    /// Read the int8 scale, if set. Useful for tests and for callers
    /// that need to persist the scale alongside other slot metadata.
    /// Returns `None` for non-int8 backends and for uncalibrated int8
    /// backends.
    pub fn int8_scale(&self) -> Option<f32> {
        match &*self.inner.read().expect("DenseIndex lock poisoned") {
            DenseInner::Int8 { scale, .. } => *scale,
            _ => None,
        }
    }

    /// Quantization backend in use.
    pub fn quant(&self) -> DenseQuant {
        match &*self.inner.read().expect("DenseIndex lock poisoned") {
            DenseInner::F32 { .. } => DenseQuant::F32,
            DenseInner::F16 { .. } => DenseQuant::F16,
            DenseInner::Int8 { .. } => DenseQuant::Int8,
        }
    }

    /// Insert one vector at `position` (the index into vectors.bin).
    /// Takes a write lock for the brief duration of the HNSW insert
    /// and by_position update. Caller is responsible for monotonically
    /// increasing positions starting from 0; gaps are tolerated but the
    /// rest of the codebase doesn't currently produce them.
    ///
    /// Vectors come in as f32 regardless of the backend; f16 / int8
    /// paths convert before insert. Per-call conversion cost is small
    /// at realistic dims (1024 f32→f16/i8 conversions ≈ a few µs).
    /// For int8, [`Self::set_int8_scale`] must have been called first
    /// — debug-panics otherwise.
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
            DenseInner::Int8 { hnsw, scale, .. } => {
                let s = scale.expect("int8 DenseIndex insert called before set_int8_scale");
                let vi8 = quantize_f32_to_i8(vector, s);
                hnsw.insert((&vi8, position as usize));
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
    /// — see [`DenseQuant::from_vector_quant`] for the mapping. For
    /// the int8 backend, this method does a calibration first-pass
    /// over a prefix of vectors (up to 256, or the whole store if
    /// smaller) to find max-abs and set the scale. The first-pass is
    /// independent of the second insert-pass — it doesn't materialize
    /// vectors twice; it just samples enough to fix the scale.
    pub fn build(vectors: &VectorStoreReader, params: HnswParams) -> Result<Self, BucketError> {
        let count = vectors.len();
        let quant = DenseQuant::from_vector_quant(vectors.quant());

        let mut pairs: Vec<(u64, ChunkId)> = vectors
            .iter_chunk_positions()
            .map(|(id, pos)| (pos, id))
            .collect();
        pairs.sort_by_key(|(pos, _)| *pos);

        let index = Self::empty(quant, params, count);

        // Int8 calibration: sample up to the first 256 vectors to
        // compute the dataset-wide scale before inserting any. 256 is
        // generous for unit-norm embeddings; max-abs converges quickly.
        if quant == DenseQuant::Int8 {
            let sample_n = pairs.len().min(256);
            let mut samples: Vec<Vec<f32>> = Vec::with_capacity(sample_n);
            for (position, _) in pairs.iter().take(sample_n).copied() {
                let v = vectors
                    .read_at_position(position)
                    .map_err(BucketError::Io)?;
                samples.push(v);
            }
            let scale = calibrate_int8_scale(&samples);
            index.set_int8_scale(scale)?;
        }

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
    /// Query is always f32; non-f32 backends promote/quantize it once
    /// before dispatch. The conversion is hot-path-cheap (one pass over
    /// the query vector, no allocation per HNSW edge probe).
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
            DenseInner::Int8 {
                hnsw,
                by_position,
                scale,
            } => {
                let s = scale.expect("int8 DenseIndex search called before set_int8_scale");
                let qi8 = quantize_f32_to_i8(query, s);
                let neighbours = hnsw.search(&qi8, top_k, ef);
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
        let int8_scale_path = slot_dir.join(DENSE_INT8_SCALE_NAME);
        if snapshot_path.exists() {
            std::fs::remove_file(&snapshot_path).map_err(BucketError::Io)?;
        }
        if int8_scale_path.exists() {
            std::fs::remove_file(&int8_scale_path).map_err(BucketError::Io)?;
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
        let (snapshot_count, int8_scale) = {
            let inner = self.inner.read().expect("DenseIndex lock poisoned");
            match &*inner {
                DenseInner::F32 { hnsw, .. } => {
                    hnsw.file_dump(slot_dir, DENSE_BASENAME)
                        .map_err(|e| BucketError::Other(format!("hnsw file_dump: {e}")))?;
                    (hnsw.get_nb_point() as u64, None)
                }
                DenseInner::F16 { hnsw, .. } => {
                    hnsw.file_dump(slot_dir, DENSE_BASENAME)
                        .map_err(|e| BucketError::Other(format!("hnsw file_dump: {e}")))?;
                    (hnsw.get_nb_point() as u64, None)
                }
                DenseInner::Int8 { hnsw, scale, .. } => {
                    let s = scale.ok_or_else(|| {
                        BucketError::Other(
                            "dump_to: int8 DenseIndex has no calibrated scale".to_string(),
                        )
                    })?;
                    hnsw.file_dump(slot_dir, DENSE_BASENAME)
                        .map_err(|e| BucketError::Other(format!("hnsw file_dump: {e}")))?;
                    (hnsw.get_nb_point() as u64, Some(s))
                }
            }
        };
        // Sidecar order: hnsw files first (above), then int8_scale (if
        // applicable), then dense.snapshot last. Loaders use the
        // presence of dense.snapshot as the "dump is complete" signal,
        // so the int8_scale must be flushed before that signal is
        // visible — otherwise a crash between snapshot and scale could
        // leave a snapshot pointing at a missing scale.
        if let Some(s) = int8_scale {
            std::fs::write(&int8_scale_path, s.to_le_bytes()).map_err(BucketError::Io)?;
        }
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
        DenseQuant::Int8 => {
            // Read the f32→i8 calibration scale from the sidecar
            // before touching the hnsw files. Absent / wrong-sized
            // sidecar means the dump is from a pre-int8-HNSW run or
            // partially written — caller (registry slot-open) falls
            // back to the rebuild-from-vectors path.
            let scale_path = slot_dir.join(DENSE_INT8_SCALE_NAME);
            let scale = read_int8_scale_sidecar(&scale_path)?;
            let hnsw: Hnsw<'static, i8, DistInt8L2> = io
                .load_hnsw()
                .map_err(|e| BucketError::Other(format!("hnsw load: {e}")))?;
            let hnsw_count = hnsw.get_nb_point();
            if hnsw_count != by_position.len() {
                return Err(BucketError::Other(format!(
                    "hnsw dump count {hnsw_count} does not match expected count {}",
                    by_position.len(),
                )));
            }
            DenseInner::Int8 {
                hnsw,
                by_position,
                scale: Some(scale),
            }
        }
    };
    Ok(DenseIndex {
        inner: RwLock::new(inner),
    })
}

/// Read the `dense.int8_scale` sidecar (4-byte LE f32). Errors when
/// the file is missing, the wrong size, or holds a non-positive value
/// — all of which mean the int8 dump can't be trusted.
fn read_int8_scale_sidecar(path: &Path) -> Result<f32, BucketError> {
    let bytes = std::fs::read(path).map_err(|e| {
        BucketError::Other(format!(
            "int8 scale sidecar missing or unreadable at {}: {e}",
            path.display(),
        ))
    })?;
    if bytes.len() != 4 {
        return Err(BucketError::Other(format!(
            "int8 scale sidecar at {} has {} bytes, expected 4",
            path.display(),
            bytes.len(),
        )));
    }
    let scale = f32::from_le_bytes(bytes.try_into().unwrap());
    if !(scale > 0.0 && scale.is_finite()) {
        return Err(BucketError::Other(format!(
            "int8 scale sidecar holds non-positive / non-finite value: {scale}",
        )));
    }
    Ok(scale)
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
    fn dense_quant_from_vector_quant_one_to_one() {
        use crate::knowledge::vectors::VectorQuant;
        // Each vectors.bin quant maps to its own HNSW backend:
        // F32 → f32, F16 → f16, Int8 → int8 (with calibrated
        // dataset-wide scale).
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
            DenseQuant::Int8
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
    fn build_from_int8_vectors_uses_int8_backend() {
        // Int8 vectors.bin → int8 HNSW with a calibrated dataset-wide
        // scale. Both the per-vector-scale (vectors.bin storage) and
        // the dataset-wide-scale (HNSW) introduce some precision loss,
        // so we only check that the top-1 nearest neighbour is the
        // exact match.
        let tmp = build_test_vectors_quant(8, 50, crate::knowledge::vectors::VectorQuant::Int8);
        let vectors = open_reader(&tmp);
        let index = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        assert_eq!(index.len(), 50);
        assert_eq!(index.quant(), DenseQuant::Int8);
        assert!(
            index.int8_scale().is_some(),
            "build() must calibrate the int8 scale before returning",
        );

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

    #[test]
    fn dist_int8_l2_matches_f32_ordering() {
        // Sanity check on `DistInt8L2`: a query that's a closer match
        // to one of two i8 candidates must rank that candidate first.
        let dist = DistInt8L2;
        let q: Vec<i8> = vec![10, 20, 30, 40];
        let near: Vec<i8> = vec![11, 19, 31, 39]; // small per-component diff
        let far: Vec<i8> = vec![50, 50, 50, 50]; // big per-component diff
        let d_near = dist.eval(&q, &near);
        let d_far = dist.eval(&q, &far);
        assert!(
            d_near < d_far,
            "near ({d_near}) should rank closer than far ({d_far})",
        );
        // sqrt(0) = 0 — identity vectors have zero distance.
        assert_eq!(dist.eval(&q, &q), 0.0);
    }

    #[test]
    fn dist_int8_l2_handles_full_range_without_overflow() {
        // i8 range is [-128, 127]. (-128 - 127) = -255 squared = 65025.
        // Sum across 1024 dims fits in i64. Verify no overflow / panic.
        let dist = DistInt8L2;
        let a: Vec<i8> = vec![-128i8; 1024];
        let b: Vec<i8> = vec![127i8; 1024];
        let d = dist.eval(&a, &b);
        // Expected: sqrt(1024 * 255^2) = 255 * sqrt(1024) = 255 * 32 = 8160
        assert!((d - 8160.0).abs() < 1.0, "got d={d}, expected ~8160");
    }

    #[test]
    fn calibrate_int8_scale_uses_max_abs_with_headroom() {
        // max-abs across the sample is 0.5; expected scale = 0.5 *
        // 1.1 / 127 ≈ 0.004331.
        let samples = vec![vec![0.1f32, -0.5, 0.3], vec![0.4, -0.2, 0.0]];
        let s = calibrate_int8_scale(&samples);
        let expected = 0.5 * 1.1 / 127.0;
        assert!(
            (s - expected).abs() < 1.0e-7,
            "scale {s} should be {expected}",
        );
    }

    #[test]
    fn calibrate_int8_scale_floor_for_all_zero_input() {
        // All-zero samples → max_abs = 0 → would divide by zero at
        // quantize time without the floor. The floor must produce a
        // strictly positive scale.
        let samples = vec![vec![0.0f32, 0.0, 0.0]; 5];
        let s = calibrate_int8_scale(&samples);
        assert!(s > 0.0, "scale {s} must be strictly positive");
        assert!(s.is_finite(), "scale {s} must be finite");
    }

    #[test]
    fn set_int8_scale_rejects_conflicting_reset() {
        let index = DenseIndex::empty(DenseQuant::Int8, HnswParams::default(), 16);
        index.set_int8_scale(0.001).unwrap();
        // Same scale: idempotent.
        assert!(index.set_int8_scale(0.001).is_ok());
        // Different scale: error (would invalidate already-inserted
        // quantized vectors).
        let err = index.set_int8_scale(0.002).unwrap_err();
        assert!(
            err.to_string().contains("already set"),
            "unexpected error: {err}",
        );
    }

    #[test]
    #[should_panic(expected = "non-int8 DenseIndex backend")]
    fn set_int8_scale_panics_on_f32_backend() {
        // Calling set_int8_scale on the wrong backend is a wiring bug;
        // panic is the right surface.
        let index = DenseIndex::empty(DenseQuant::F32, HnswParams::default(), 16);
        let _ = index.set_int8_scale(0.001);
    }

    #[test]
    fn dump_and_load_roundtrip_int8() {
        let tmp = build_test_vectors_quant(8, 50, crate::knowledge::vectors::VectorQuant::Int8);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        assert_eq!(built.quant(), DenseQuant::Int8);
        let scale_before = built.int8_scale().expect("calibrated by build()");
        built.dump_to(tmp.path()).unwrap();

        // The int8_scale sidecar must exist after dump_to.
        let scale_path = tmp.path().join("dense.int8_scale");
        assert!(scale_path.exists(), "dense.int8_scale sidecar missing");
        assert_eq!(
            std::fs::metadata(&scale_path).unwrap().len(),
            4,
            "sidecar must be exactly 4 bytes (one LE f32)",
        );

        let loaded = DenseIndex::load_from(tmp.path(), &vectors).unwrap();
        assert_eq!(loaded.quant(), DenseQuant::Int8);
        assert_eq!(loaded.int8_scale(), Some(scale_before));
        assert_eq!(loaded.len(), built.len());

        let mut query = vec![0.0f32; 8];
        query[5] = 6.0;
        let from_built = built.search(&query, 1);
        let from_loaded = loaded.search(&query, 1);
        assert_eq!(from_built[0].0, from_loaded[0].0);
    }

    #[test]
    fn int8_dump_is_smaller_than_f16_dump() {
        // The whole point of int8-in-HNSW: 1 byte/dim vs f16's 2
        // bytes/dim. dense.hnsw.data should be roughly half of f16
        // for the same vector count. Generous bound (< 70% of f16,
        // matching the f16-vs-f32 test's framing).
        let dim: u32 = 64;
        let n = 200;

        let f16_tmp = build_test_vectors_quant(dim, n, crate::knowledge::vectors::VectorQuant::F16);
        let f16_vec = open_reader(&f16_tmp);
        DenseIndex::build(&f16_vec, HnswParams::default())
            .unwrap()
            .dump_to(f16_tmp.path())
            .unwrap();
        let f16_data = std::fs::metadata(f16_tmp.path().join("dense.hnsw.data"))
            .unwrap()
            .len();

        let int8_tmp =
            build_test_vectors_quant(dim, n, crate::knowledge::vectors::VectorQuant::Int8);
        let int8_vec = open_reader(&int8_tmp);
        DenseIndex::build(&int8_vec, HnswParams::default())
            .unwrap()
            .dump_to(int8_tmp.path())
            .unwrap();
        let int8_data = std::fs::metadata(int8_tmp.path().join("dense.hnsw.data"))
            .unwrap()
            .len();

        let ratio = int8_data as f64 / f16_data as f64;
        assert!(
            ratio < 0.7,
            "int8 dense.hnsw.data ({int8_data}) should be < 70% of f16 ({f16_data}); got ratio={ratio:.3}",
        );
    }

    #[test]
    fn load_from_with_positions_int8_roundtrips() {
        // Resume path for int8 backend.
        let tmp = build_test_vectors_quant(8, 20, crate::knowledge::vectors::VectorQuant::Int8);
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
            DenseIndex::load_from_with_positions(tmp.path(), by_position, DenseQuant::Int8)
                .unwrap();
        assert_eq!(loaded.quant(), DenseQuant::Int8);
        assert_eq!(loaded.int8_scale(), built.int8_scale());

        let mut q = vec![0.0f32; 8];
        q[3] = 4.0;
        let from_built = built.search(&q, 1);
        let from_loaded = loaded.search(&q, 1);
        assert_eq!(from_built[0].0, from_loaded[0].0);
    }

    #[test]
    fn load_from_int8_missing_sidecar_returns_error() {
        // Build + dump int8, then delete the sidecar — load_from must
        // error out so the registry's load-or-rebuild fallback kicks in.
        let tmp = build_test_vectors_quant(8, 20, crate::knowledge::vectors::VectorQuant::Int8);
        let vectors = open_reader(&tmp);
        let built = DenseIndex::build(&vectors, HnswParams::default()).unwrap();
        built.dump_to(tmp.path()).unwrap();

        std::fs::remove_file(tmp.path().join("dense.int8_scale")).unwrap();

        let err = DenseIndex::load_from(tmp.path(), &vectors).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("int8 scale sidecar"),
            "unexpected error: {msg}",
        );
    }
}
