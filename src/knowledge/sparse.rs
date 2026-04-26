//! Sparse (BM25 over chunk text via tantivy) index for a slot.
//!
//! Wraps `tantivy` to provide build + search over the slot's chunk
//! text. The tantivy index lives in its own subdirectory under the
//! slot (`<slot>/index.tantivy/`) and is fully tantivy-managed —
//! segment files, schema, etc.
//!
//! Schema (slice 6):
//! - `chunk_id` — `STORED` raw 32-byte blake3 digest. Retrieved at
//!   query time to map hits back to chunk ids; not searchable.
//! - `text`    — `TEXT` indexed chunk content. The chunk text also
//!   lives in `chunks.bin` (durable storage), so we don't `STORED` it
//!   here — saves disk and ram, at the cost of an extra `chunks.bin`
//!   read when populating Candidate fields.
//!
//! Tokenizer is the tantivy default (`SimpleTokenizer + LowerCaser +
//! StopWordFilter`). Per-bucket tokenizer config (`[search_paths.sparse]
//! tokenizer = "..."` from `bucket.toml`) wires through here in a future
//! slice when we have multiple options to swap between.

use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, STORED, Schema, TEXT, Value};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use super::types::{BucketError, ChunkId};

/// Memory budget for tantivy `IndexWriter`. 50MB matches the standard
/// from tantivy's `basic_search` example — fine for any reasonable
/// build throughput. Tunable when wikipedia-scale build perf becomes
/// the constraint.
const WRITER_MEMORY_BUDGET: usize = 50_000_000;

const FIELD_CHUNK_ID: &str = "chunk_id";
const FIELD_TEXT: &str = "text";

fn build_schema() -> (Schema, Field, Field) {
    let mut builder = Schema::builder();
    let chunk_id_field = builder.add_bytes_field(FIELD_CHUNK_ID, STORED);
    let text_field = builder.add_text_field(FIELD_TEXT, TEXT);
    let schema = builder.build();
    (schema, chunk_id_field, text_field)
}

/// Builder/writer for a slot's tantivy index. One commit per
/// [`finalize`](Self::finalize); intermediate adds are buffered in the
/// writer's memory arena. For wikipedia-scale builds, callers may want
/// to commit periodically to cap peak memory; for slice 6 we commit
/// once at the end.
pub struct SparseIndexBuilder {
    writer: IndexWriter,
    chunk_id_field: Field,
    text_field: Field,
    count: usize,
}

impl std::fmt::Debug for SparseIndexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseIndexBuilder")
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl SparseIndexBuilder {
    /// Create a new tantivy index in `dir`. Directory must not already
    /// contain a tantivy index.
    pub fn create(dir: &Path) -> Result<Self, BucketError> {
        std::fs::create_dir_all(dir).map_err(BucketError::Io)?;
        let (schema, chunk_id_field, text_field) = build_schema();
        let index = Index::create_in_dir(dir, schema)
            .map_err(|e| BucketError::Other(format!("tantivy create_in_dir: {e}")))?;
        let writer: IndexWriter = index
            .writer(WRITER_MEMORY_BUDGET)
            .map_err(|e| BucketError::Other(format!("tantivy writer: {e}")))?;
        Ok(Self {
            writer,
            chunk_id_field,
            text_field,
            count: 0,
        })
    }

    /// Reopen an existing tantivy index in `dir` for resumed appending.
    /// Tantivy's segment-based commit machinery does the durability
    /// work natively — we just open a fresh writer over the existing
    /// index. The pre-existing committed segments stay; new
    /// `add` calls produce new segments that land on the next commit.
    ///
    /// The resume code uses the `delete_then_add` discipline (each
    /// [`Self::add`] of a chunk_id deletes any prior doc with the same
    /// chunk_id first), so re-adding chunks already in the index is
    /// idempotent. That's important when a partial-batch commit
    /// landed before the BatchEmbedded build-state record fsynced.
    pub fn open_resume(dir: &Path) -> Result<Self, BucketError> {
        let index = Index::open_in_dir(dir)
            .map_err(|e| BucketError::Other(format!("tantivy open_in_dir: {e}")))?;
        let writer: IndexWriter = index
            .writer(WRITER_MEMORY_BUDGET)
            .map_err(|e| BucketError::Other(format!("tantivy writer: {e}")))?;
        let schema = index.schema();
        let chunk_id_field = schema
            .get_field(FIELD_CHUNK_ID)
            .map_err(|e| BucketError::Other(format!("missing field chunk_id: {e}")))?;
        let text_field = schema
            .get_field(FIELD_TEXT)
            .map_err(|e| BucketError::Other(format!("missing field text: {e}")))?;
        // `count` is the count of *additional* docs since open. The
        // total committed size is whatever tantivy already knew about
        // plus this. Resume-time chunk-count cross-checks therefore
        // happen at the slot level (chunks.bin len), not via this
        // counter alone.
        Ok(Self {
            writer,
            chunk_id_field,
            text_field,
            count: 0,
        })
    }

    /// Add a chunk to the index. Idempotent in the sense that the same
    /// chunk_id called twice produces a single doc — the prior doc is
    /// deleted first via `delete_term`. This is what lets resume
    /// re-add chunks that may have been partially committed by an
    /// interrupted prior attempt without producing duplicates.
    pub fn add(&mut self, chunk_id: ChunkId, text: &str) -> Result<(), BucketError> {
        let term = Term::from_field_bytes(self.chunk_id_field, &chunk_id.0);
        self.writer.delete_term(term);
        let mut doc = TantivyDocument::default();
        doc.add_bytes(self.chunk_id_field, chunk_id.0.as_slice());
        doc.add_text(self.text_field, text);
        self.writer
            .add_document(doc)
            .map_err(|e| BucketError::Other(format!("tantivy add_document: {e}")))?;
        self.count += 1;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Commit pending writes without consuming the builder. Used at
    /// every chunk-batch boundary so that an interrupted build's
    /// tantivy state is durable up to the same record count as the
    /// chunks/vectors files.
    pub fn commit(&mut self) -> Result<(), BucketError> {
        self.writer
            .commit()
            .map_err(|e| BucketError::Other(format!("tantivy commit: {e}")))?;
        Ok(())
    }

    /// Final commit. Returns the docs added during this builder's
    /// lifetime — for resume, that's only the new docs (not the total
    /// size).
    pub fn finalize(mut self) -> Result<usize, BucketError> {
        self.writer
            .commit()
            .map_err(|e| BucketError::Other(format!("tantivy commit: {e}")))?;
        Ok(self.count)
    }
}

/// Read-only view over a slot's tantivy index.
pub struct SparseIndex {
    index: Index,
    reader: IndexReader,
    chunk_id_field: Field,
    text_field: Field,
}

impl std::fmt::Debug for SparseIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseIndex")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl SparseIndex {
    pub fn open(dir: &Path) -> Result<Self, BucketError> {
        let index = Index::open_in_dir(dir)
            .map_err(|e| BucketError::Other(format!("tantivy open_in_dir: {e}")))?;
        let reader: IndexReader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| BucketError::Other(format!("tantivy reader: {e}")))?;
        let schema = index.schema();
        let chunk_id_field = schema
            .get_field(FIELD_CHUNK_ID)
            .map_err(|e| BucketError::Other(format!("missing field chunk_id: {e}")))?;
        let text_field = schema
            .get_field(FIELD_TEXT)
            .map_err(|e| BucketError::Other(format!("missing field text: {e}")))?;
        Ok(Self {
            index,
            reader,
            chunk_id_field,
            text_field,
        })
    }

    pub fn len(&self) -> usize {
        self.reader.searcher().num_docs() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Run a BM25 query and return `(chunk_id, score)` pairs ordered
    /// by descending score (highest match first). Empty if `top_k`
    /// is 0, the index is empty, or no documents match.
    ///
    /// A malformed query (e.g. unclosed quote) surfaces as
    /// [`BucketError::Other`] — tantivy's query parser is strict.
    pub fn search(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(ChunkId, f32)>, BucketError> {
        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        let query = parser
            .parse_query(query_text)
            .map_err(|e| BucketError::Other(format!("tantivy parse_query: {e}")))?;
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(top_k))
            .map_err(|e| BucketError::Other(format!("tantivy search: {e}")))?;

        let mut out = Vec::with_capacity(top_docs.len());
        for (score, addr) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(addr)
                .map_err(|e| BucketError::Other(format!("tantivy doc fetch: {e}")))?;
            let val = doc
                .get_first(self.chunk_id_field)
                .ok_or_else(|| BucketError::Other("hit doc missing chunk_id field".into()))?;
            let bytes = val
                .as_bytes()
                .ok_or_else(|| BucketError::Other("chunk_id field is not bytes".into()))?;
            if bytes.len() != 32 {
                return Err(BucketError::Other(format!(
                    "chunk_id field is {} bytes, expected 32",
                    bytes.len(),
                )));
            }
            let mut id_bytes = [0u8; 32];
            id_bytes.copy_from_slice(bytes);
            out.push((ChunkId(id_bytes), score));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index_with(chunks: &[(ChunkId, &str)]) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("index.tantivy");
        let mut builder = SparseIndexBuilder::create(&dir).unwrap();
        for (id, text) in chunks {
            builder.add(*id, text).unwrap();
        }
        builder.finalize().unwrap();
        tmp
    }

    fn open(tmp: &tempfile::TempDir) -> SparseIndex {
        SparseIndex::open(&tmp.path().join("index.tantivy")).unwrap()
    }

    #[test]
    fn build_and_search_returns_matching_chunks() {
        let id_a = ChunkId::from_source(&[1; 32], 0);
        let id_b = ChunkId::from_source(&[2; 32], 0);
        let id_c = ChunkId::from_source(&[3; 32], 0);
        let tmp = build_index_with(&[
            (id_a, "the quick brown fox jumps over the lazy dog"),
            (id_b, "lorem ipsum dolor sit amet consectetur"),
            (id_c, "the rain in spain falls mainly on the plain"),
        ]);
        let index = open(&tmp);
        assert_eq!(index.len(), 3);

        let results = index.search("fox", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id_a);
        assert!(results[0].1 > 0.0);

        // Multi-doc match
        let results = index.search("the", 5).unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<ChunkId> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&id_a));
        assert!(ids.contains(&id_c));
    }

    #[test]
    fn search_orders_by_descending_score() {
        let id_a = ChunkId::from_source(&[1; 32], 0);
        let id_b = ChunkId::from_source(&[2; 32], 0);
        let tmp = build_index_with(&[
            (id_a, "rust rust rust rust programming"),
            (id_b, "rust programming language"),
        ]);
        let index = open(&tmp);

        let results = index.search("rust", 5).unwrap();
        assert_eq!(results.len(), 2);
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "expected descending scores, got {} then {}",
                w[0].1,
                w[1].1,
            );
        }
        // The doc with more occurrences of "rust" wins
        assert_eq!(results[0].0, id_a);
    }

    #[test]
    fn search_no_match_returns_empty() {
        let id_a = ChunkId::from_source(&[1; 32], 0);
        let tmp = build_index_with(&[(id_a, "alpha bravo charlie")]);
        let index = open(&tmp);
        let results = index.search("delta", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("index.tantivy");
        let builder = SparseIndexBuilder::create(&dir).unwrap();
        builder.finalize().unwrap();
        let index = SparseIndex::open(&dir).unwrap();
        assert!(index.is_empty());
        assert!(index.search("anything", 5).unwrap().is_empty());
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let id_a = ChunkId::from_source(&[1; 32], 0);
        let tmp = build_index_with(&[(id_a, "hello")]);
        let index = open(&tmp);
        assert!(index.search("hello", 0).unwrap().is_empty());
    }

    #[test]
    fn case_insensitive_match() {
        let id_a = ChunkId::from_source(&[1; 32], 0);
        let tmp = build_index_with(&[(id_a, "Cargo Build")]);
        let index = open(&tmp);
        let results = index.search("CARGO", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id_a);
    }

    #[test]
    fn malformed_query_surfaces_as_error() {
        let id_a = ChunkId::from_source(&[1; 32], 0);
        let tmp = build_index_with(&[(id_a, "hello world")]);
        let index = open(&tmp);
        // Unclosed quote — tantivy's parser rejects this.
        let res = index.search("\"unclosed", 5);
        assert!(res.is_err(), "expected parser error, got {res:?}");
    }
}
