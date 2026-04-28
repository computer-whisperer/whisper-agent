//! Per-slot mapping from a record's `source_id` (article title, file
//! path, …) to every [`ChunkId`] ever emitted under that source.
//!
//! ## Why this exists
//!
//! Tracked-bucket delta application needs to answer "what chunks does
//! page X currently have?" so the apply step can tombstone the old
//! chunks before inserting fresh chunks for the new revision. Nothing
//! else in the slot's on-disk shape answers that question:
//! `chunks.idx` is keyed by [`ChunkId`] only; tantivy's schema is
//! `(chunk_id, text)` only; `chunks.bin` carries `source_id` per record
//! but isn't indexed by it. The `SourceIndex` is the missing reverse
//! lookup.
//!
//! ## Lifecycle
//!
//! - **Build**: at slot open, walk `chunks.bin` (and the optional delta
//!   chunks layer) sequentially — see
//!   [`ChunkStoreReader::iter_chunk_source_ids`](super::chunks::ChunkStoreReader::iter_chunk_source_ids).
//!   For simplewiki (~870k chunks across ~280k pages) the rebuild is a
//!   few seconds and the resulting map is ~22 MB.
//! - **Insert**: `Bucket::insert` calls [`SourceIndex::add`] for each
//!   freshly persisted chunk so subsequent deltas see it.
//! - **Tombstone**: no-op. The index is an *accumulator* — entries are
//!   never removed. See "Why we never remove" below.
//! - **Compact**: rebuilt from scratch against the new base slot,
//!   collapsing accumulated history.
//!
//! Memory-only for now. A persistent `source_index.bin` sidecar is the
//! natural follow-up for wikipedia-scale slots, where the chunks.bin
//! scan at slot open becomes a felt cost.
//!
//! ## Why we never remove
//!
//! Re-edits of the same page produce fresh `ChunkId`s (the chunk id is
//! `blake3(source_record_hash ‖ chunk_offset)` and `source_record_hash`
//! changes whenever the page text changes). So a delta application
//! that calls `tombstone(old_ids)` followed by `insert(new_chunks)`
//! never collides on chunk ids. Re-tombstoning an already-tombstoned
//! id is a sorted-set no-op. Keeping the old (tombstoned) ids in the
//! index is therefore harmless and lets the data structure stay a
//! pure accumulator.

use std::collections::HashMap;
use std::sync::RwLock;

use super::chunks::ChunkStoreReader;
use super::types::ChunkId;

/// `source_id → Vec<ChunkId>` accumulator. See module docs.
#[derive(Debug, Default)]
pub struct SourceIndex {
    map: RwLock<HashMap<String, Vec<ChunkId>>>,
}

impl SourceIndex {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build by walking the base chunks reader (and the optional delta
    /// chunks reader, if the slot has had inserts since the last
    /// compaction). Records from the delta layer are appended to the
    /// same `Vec<ChunkId>` as base entries — the index doesn't
    /// distinguish base vs delta, since tombstone+insert just needs
    /// the union for "every chunk id we've ever emitted under this
    /// source_id."
    pub fn build(
        base: &ChunkStoreReader,
        delta: Option<&ChunkStoreReader>,
    ) -> std::io::Result<Self> {
        let mut map: HashMap<String, Vec<ChunkId>> = HashMap::new();
        for (chunk_id, source_id) in base.iter_chunk_source_ids()? {
            map.entry(source_id).or_default().push(chunk_id);
        }
        if let Some(d) = delta {
            for (chunk_id, source_id) in d.iter_chunk_source_ids()? {
                map.entry(source_id).or_default().push(chunk_id);
            }
        }
        Ok(Self {
            map: RwLock::new(map),
        })
    }

    /// Every `ChunkId` ever associated with `source_id` in this slot,
    /// including any tombstoned entries. Empty `Vec` for unknown
    /// source_ids — the caller treats that as "first time we've seen
    /// this page" and skips the tombstone step.
    pub fn get(&self, source_id: &str) -> Vec<ChunkId> {
        self.map
            .read()
            .expect("SourceIndex lock poisoned")
            .get(source_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Append a freshly persisted chunk under `source_id`. Called from
    /// `Bucket::insert` after the chunk lands on disk so subsequent
    /// delta applications find it.
    pub fn add(&self, source_id: &str, chunk_id: ChunkId) {
        self.map
            .write()
            .expect("SourceIndex lock poisoned")
            .entry(source_id.to_string())
            .or_default()
            .push(chunk_id);
    }

    /// Distinct `source_id` count. Test / observability only.
    pub fn distinct_sources(&self) -> usize {
        self.map.read().expect("SourceIndex lock poisoned").len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::chunks::{ChunkStoreReader, ChunkStoreWriter};
    use crate::knowledge::types::SourceRef;

    fn cid(seed: u8) -> ChunkId {
        ChunkId([seed; 32])
    }

    fn sref(source_id: &str) -> SourceRef {
        SourceRef {
            source_id: source_id.to_string(),
            locator: None,
        }
    }

    /// Round-trip a base store with mixed source_ids and verify each
    /// source_id maps to the expected chunk_ids in insertion order.
    /// Same `source_id` appearing twice (page with multiple chunks)
    /// must accumulate to two entries.
    #[test]
    fn build_groups_chunks_by_source_id() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");

        let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
        w.append(cid(1), &sref("alpha"), "alpha-a").unwrap();
        w.append(cid(2), &sref("alpha"), "alpha-b").unwrap();
        w.append(cid(3), &sref("beta"), "beta-a").unwrap();
        w.append(cid(4), &sref("alpha"), "alpha-c").unwrap();
        w.finalize().unwrap();

        let r = ChunkStoreReader::open(&bin, &idx).unwrap();
        let si = SourceIndex::build(&r, None).unwrap();

        assert_eq!(si.distinct_sources(), 2);
        assert_eq!(si.get("alpha"), vec![cid(1), cid(2), cid(4)]);
        assert_eq!(si.get("beta"), vec![cid(3)]);
        assert!(si.get("missing").is_empty());
    }

    /// Delta records are appended onto the base entries for the same
    /// source_id (no dedup — the apply path tombstones+inserts, so
    /// duplicate ids across base and delta are expected when a page
    /// has been re-chunked but the old chunks haven't been compacted
    /// out yet).
    #[test]
    fn build_unions_base_and_delta_under_same_source_id() {
        let tmp = tempfile::tempdir().unwrap();

        let base_bin = tmp.path().join("chunks.bin");
        let base_idx = tmp.path().join("chunks.idx");
        let mut bw = ChunkStoreWriter::create(&base_bin, &base_idx).unwrap();
        bw.append(cid(1), &sref("page-x"), "v1-a").unwrap();
        bw.append(cid(2), &sref("page-x"), "v1-b").unwrap();
        bw.finalize().unwrap();

        let delta_bin = tmp.path().join("delta_chunks.bin");
        let delta_idx = tmp.path().join("delta_chunks.idx");
        let mut dw = ChunkStoreWriter::create(&delta_bin, &delta_idx).unwrap();
        dw.append(cid(10), &sref("page-x"), "v2-a").unwrap();
        dw.append(cid(11), &sref("page-y"), "v2-only").unwrap();
        dw.finalize().unwrap();

        let base = ChunkStoreReader::open(&base_bin, &base_idx).unwrap();
        let delta = ChunkStoreReader::open(&delta_bin, &delta_idx).unwrap();
        let si = SourceIndex::build(&base, Some(&delta)).unwrap();

        let mut x = si.get("page-x");
        x.sort();
        assert_eq!(x, vec![cid(1), cid(2), cid(10)]);
        assert_eq!(si.get("page-y"), vec![cid(11)]);
    }

    /// `add` after `build` accumulates without losing pre-built state
    /// — the same shape `Bucket::insert` will exercise live.
    #[test]
    fn add_appends_to_existing_source_id_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("chunks.bin");
        let idx = tmp.path().join("chunks.idx");
        let mut w = ChunkStoreWriter::create(&bin, &idx).unwrap();
        w.append(cid(1), &sref("alpha"), "a").unwrap();
        w.finalize().unwrap();
        let r = ChunkStoreReader::open(&bin, &idx).unwrap();
        let si = SourceIndex::build(&r, None).unwrap();

        si.add("alpha", cid(2));
        si.add("gamma", cid(99));

        assert_eq!(si.get("alpha"), vec![cid(1), cid(2)]);
        assert_eq!(si.get("gamma"), vec![cid(99)]);
        assert_eq!(si.distinct_sources(), 2);
    }
}
