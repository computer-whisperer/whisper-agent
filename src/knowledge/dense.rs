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

use hnsw_rs::hnswio::HnswIo;
use hnsw_rs::prelude::*;

use super::types::{BucketError, ChunkId};
use super::vectors::VectorStoreReader;

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
struct DenseInner {
    hnsw: Hnsw<'static, f32, DistL2>,
    /// `by_position[d_id]` is the chunk id corresponding to HNSW data
    /// id `d_id`. Grown to position+1 on each insert; entries between
    /// the old length and the new position are filled with the zero
    /// chunk id (not normally observable since search results are
    /// filtered by position validity).
    by_position: Vec<ChunkId>,
}

impl std::fmt::Debug for DenseIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.inner.read().map(|i| i.by_position.len()).unwrap_or(0);
        f.debug_struct("DenseIndex")
            .field("len", &len)
            .finish_non_exhaustive()
    }
}

impl DenseIndex {
    /// Construct an empty index, sized with `max_elements_hint` as a
    /// hint to hnsw_rs's internal allocator. Inserts beyond the hint
    /// still work — hnsw_rs treats it as a sizing suggestion, not a
    /// hard cap — but undersizing forces reallocation. Tune to the
    /// expected chunk count when known; over-estimate is cheap.
    pub fn empty(params: HnswParams, max_elements_hint: usize) -> Self {
        let hnsw: Hnsw<'static, f32, DistL2> = Hnsw::new(
            params.max_nb_connection,
            max_elements_hint.max(1),
            params.max_layer,
            params.ef_construction,
            DistL2 {},
        );
        Self {
            inner: RwLock::new(DenseInner {
                hnsw,
                by_position: Vec::new(),
            }),
        }
    }

    /// Insert one vector at `position` (the index into vectors.bin).
    /// Takes a write lock for the brief duration of the HNSW insert
    /// and by_position update. Caller is responsible for monotonically
    /// increasing positions starting from 0; gaps are tolerated but the
    /// rest of the codebase doesn't currently produce them.
    pub fn insert(&self, chunk_id: ChunkId, position: u64, vector: &[f32]) {
        let mut inner = self.inner.write().expect("DenseIndex lock poisoned");
        inner.hnsw.insert((vector, position as usize));
        if inner.by_position.len() <= position as usize {
            // Pad with the zero chunk id; HNSW's filter_map in search
            // still produces correct results because we only return
            // by_position[d_id] entries that came from real inserts
            // (any d_id we didn't insert wouldn't appear in HNSW
            // search output).
            inner
                .by_position
                .resize(position as usize + 1, ChunkId([0; 32]));
        }
        inner.by_position[position as usize] = chunk_id;
    }

    /// Build a fresh index from a vectors store. Reads each vector
    /// once in position order and inserts into the HNSW graph. Used
    /// on slot reload when a dump is missing/stale.
    pub fn build(vectors: &VectorStoreReader, params: HnswParams) -> Result<Self, BucketError> {
        let count = vectors.len();

        let mut pairs: Vec<(u64, ChunkId)> = vectors
            .iter_chunk_positions()
            .map(|(id, pos)| (pos, id))
            .collect();
        pairs.sort_by_key(|(pos, _)| *pos);

        let index = Self::empty(params, count);
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
        self.inner.read().map(|i| i.by_position.len()).unwrap_or(0)
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
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(ChunkId, f32)> {
        let inner = self.inner.read().expect("DenseIndex lock poisoned");
        if inner.by_position.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let ef = (top_k * 2).max(64);
        let neighbours = inner.hnsw.search(query, top_k, ef);
        neighbours
            .into_iter()
            .filter_map(|n| inner.by_position.get(n.d_id).map(|&id| (id, n.distance)))
            .collect()
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
            inner
                .hnsw
                .file_dump(slot_dir, DENSE_BASENAME)
                .map_err(|e| BucketError::Other(format!("hnsw file_dump: {e}")))?;
            inner.hnsw.get_nb_point() as u64
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

        // The hnsw_rs `load_hnsw` signature constrains the returned
        // Hnsw to outlive its HnswIo source. We need `Hnsw<'static>`
        // for `DenseIndex` to stay loadable into a long-lived
        // registry cache, so we leak the HnswIo (a ~100-byte struct
        // with a PathBuf, basename, and an Arc counter) to give it
        // 'static. One-time, slot-load-only allocation; doesn't
        // accumulate across query traffic.
        let io: &'static mut HnswIo = Box::leak(Box::new(HnswIo::new(slot_dir, DENSE_BASENAME)));
        let hnsw: Hnsw<'static, f32, DistL2> = io
            .load_hnsw()
            .map_err(|e| BucketError::Other(format!("hnsw load: {e}")))?;

        // Rebuild the position→chunk_id table from vectors.idx —
        // same shape as `build`, just without inserting into HNSW.
        let mut pairs: Vec<(u64, ChunkId)> = vectors
            .iter_chunk_positions()
            .map(|(id, pos)| (pos, id))
            .collect();
        pairs.sort_by_key(|(pos, _)| *pos);
        let by_position: Vec<ChunkId> = pairs.into_iter().map(|(_, id)| id).collect();

        let hnsw_count = hnsw.get_nb_point();
        if hnsw_count != by_position.len() {
            return Err(BucketError::Other(format!(
                "hnsw dump count {hnsw_count} does not match vectors.idx count {} — dump is stale",
                by_position.len(),
            )));
        }

        Ok(Self {
            inner: RwLock::new(DenseInner { hnsw, by_position }),
        })
    }

    /// Load a dumped HNSW with a caller-supplied `by_position` table.
    ///
    /// Used by the resume path during a build, where `vectors.idx`
    /// hasn't been written yet — the by_position list comes from
    /// re-running the chunker over the durable Planned records.
    /// Otherwise behaves identically to [`Self::load_from`]: the
    /// hnsw_rs files must exist, and the dump's point count must
    /// match `by_position.len()`.
    pub fn load_from_with_positions(
        slot_dir: &Path,
        by_position: Vec<ChunkId>,
    ) -> Result<Self, BucketError> {
        let graph_path = slot_dir.join(format!("{DENSE_BASENAME}.hnsw.graph"));
        let data_path = slot_dir.join(format!("{DENSE_BASENAME}.hnsw.data"));
        if !graph_path.exists() || !data_path.exists() {
            return Err(BucketError::Other(format!(
                "hnsw dump not found in {} (looking for {DENSE_BASENAME}.hnsw.graph + .data)",
                slot_dir.display(),
            )));
        }
        let io: &'static mut HnswIo = Box::leak(Box::new(HnswIo::new(slot_dir, DENSE_BASENAME)));
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
        Ok(Self {
            inner: RwLock::new(DenseInner { hnsw, by_position }),
        })
    }
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
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let mut w =
            VectorStoreWriter::create(&bin, &idx, dim, crate::knowledge::vectors::VectorQuant::F32)
                .unwrap();
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

        let loaded = DenseIndex::load_from_with_positions(tmp.path(), by_position).unwrap();
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
        let err = DenseIndex::load_from_with_positions(tmp.path(), too_short).unwrap_err();
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
}
