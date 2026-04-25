//! Dense (HNSW over embeddings) index for a slot.
//!
//! Wraps the `hnsw_rs` crate to provide build + search operations over
//! a slot's persisted vectors. The HNSW graph is rebuilt on slot load
//! (no on-disk persistence in slice 5 — vectors.bin is the durable
//! input; rebuild cost is negligible at workspace scales and a known
//! TODO for the eventual wikipedia-scale slot-load path).
//!
//! Distance: [`DistL2`]. Works for any vectors (mock or real) without
//! requiring unit-norm. We'll switch to `DistDot` (or `DistCosine`)
//! once we've wired TEI's `normalize=true` end-to-end and want the
//! ranking-equivalence-with-cheaper-arithmetic win on normalized
//! vectors.
//!
//! HNSW data ids are vector positions (the index into `vectors.bin`).
//! The position→chunk_id mapping is rebuilt at build time from
//! `vectors.idx` entries; this avoids a parallel persistent file
//! and keeps `vectors.idx` as the single source of truth for chunk
//! identity.

use hnsw_rs::prelude::*;

use super::types::{BucketError, ChunkId};
use super::vectors::VectorStoreReader;

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
/// Owns the `Hnsw` graph and a position→chunk_id translation table. Not
/// persisted in slice 5 — rebuilt from `vectors.bin`/`vectors.idx` at
/// slot load. The trade-off is: faster slice 5 (no file format to
/// design) at the cost of slot-load time. Acceptable for workspace
/// buckets (<10k chunks → milliseconds); becomes a real cost at
/// wikipedia scale (50M chunks → ~30 minutes), at which point we add
/// HNSW persistence as its own slice.
pub struct DenseIndex {
    /// `'static` because Hnsw owns its inserted data via internal
    /// reference-counting; we don't keep external references alive.
    hnsw: Hnsw<'static, f32, DistL2>,

    /// `by_position[d_id]` is the chunk id corresponding to HNSW data
    /// id `d_id`. Built from `VectorStoreReader::iter_chunk_positions`
    /// at construction time.
    by_position: Vec<ChunkId>,
}

impl std::fmt::Debug for DenseIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseIndex")
            .field("len", &self.by_position.len())
            .finish_non_exhaustive()
    }
}

impl DenseIndex {
    /// Build a fresh index from a vectors store. Reads each vector
    /// once in position order and inserts into the HNSW graph.
    pub fn build(vectors: &VectorStoreReader, params: HnswParams) -> Result<Self, BucketError> {
        let count = vectors.len();

        // Construct the position→chunk_id translation. iter_chunk_positions
        // returns pairs in unspecified order; sort by position so
        // by_position[i] is the chunk at HNSW data id i.
        let mut pairs: Vec<(u64, ChunkId)> = vectors
            .iter_chunk_positions()
            .map(|(id, pos)| (pos, id))
            .collect();
        pairs.sort_by_key(|(pos, _)| *pos);
        let by_position: Vec<ChunkId> = pairs.into_iter().map(|(_, id)| id).collect();
        debug_assert_eq!(by_position.len(), count);

        // Hnsw::new requires max_elements > 0 even for an empty index.
        // Use 1 as the floor — empty indexes are valid but search will
        // short-circuit via is_empty.
        let hnsw: Hnsw<'static, f32, DistL2> = Hnsw::new(
            params.max_nb_connection,
            count.max(1),
            params.max_layer,
            params.ef_construction,
            DistL2 {},
        );

        for position in 0..count as u64 {
            let vector = vectors
                .read_at_position(position)
                .map_err(BucketError::Io)?;
            hnsw.insert((vector.as_slice(), position as usize));
        }

        Ok(Self { hnsw, by_position })
    }

    pub fn len(&self) -> usize {
        self.by_position.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_position.is_empty()
    }

    /// Search for the `top_k` nearest neighbours of `query`. Returns
    /// `(chunk_id, distance)` pairs ordered by ascending distance
    /// (closest first).
    ///
    /// Empty when the index is empty. `ef_search` is computed as
    /// `max(top_k * 2, 64)` — generous enough to keep recall high at
    /// workspace scales without paying much for the over-fetch.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(ChunkId, f32)> {
        if self.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let ef = (top_k * 2).max(64);
        let neighbours = self.hnsw.search(query, top_k, ef);
        neighbours
            .into_iter()
            .filter_map(|n| self.by_position.get(n.d_id).map(|&id| (id, n.distance)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::VectorStoreWriter;

    fn build_test_vectors(dim: u32, n: usize) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("vectors.bin");
        let idx = tmp.path().join("vectors.idx");
        let mut w = VectorStoreWriter::create(&bin, &idx, dim).unwrap();
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
