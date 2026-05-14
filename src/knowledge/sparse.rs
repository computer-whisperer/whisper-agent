//! Sparse (BM25 over chunk text via tantivy) index for a slot.
//!
//! Wraps `tantivy` to provide build + search over the slot's chunk
//! text. The tantivy index lives in its own subdirectory under the
//! slot (`<slot>/index.tantivy/`) and is fully tantivy-managed —
//! segment files, schema, etc.
//!
//! Schema (slice 6):
//! - `chunk_id` — `STORED | INDEXED` raw 32-byte blake3 digest.
//!   Retrieved at query time to map hits back to chunk ids; also
//!   indexed so `delete_term` can target a specific doc on tombstone
//!   (and on resume's idempotent re-add).
//! - `text`    — `TEXT` indexed chunk content. The chunk text also
//!   lives in `chunks.bin` (durable storage), so we don't `STORED` it
//!   here — saves disk and ram, at the cost of an extra `chunks.bin`
//!   read when populating Candidate fields.
//!
//! Tokenizer is the tantivy default (`SimpleTokenizer + RemoveLongFilter
//! + LowerCaser`).
//!
//! Per-bucket tokenizer config (`[search_paths.sparse] tokenizer = "..."`
//! from `bucket.toml`) wires through here in a future slice when we have
//! multiple options to swap between.

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, EmptyQuery, Query, QueryParser, TermQuery};
use tantivy::schema::{Field, INDEXED, IndexRecordOption, STORED, Schema, TEXT, Value};
use tantivy::tokenizer::TokenStream;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use super::types::{BucketError, ChunkId};

/// Memory budget for tantivy `IndexWriter`. 50MB matches the standard
/// from tantivy's `basic_search` example — fine for any reasonable
/// build throughput. Tunable when wikipedia-scale build perf becomes
/// the constraint.
const WRITER_MEMORY_BUDGET: usize = 50_000_000;

/// Natural-language sparse queries over enwiki get expensive when the
/// query parser expands every common word into a scoring clause. Keep a
/// small rare-term disjunction instead; dense retrieval already covers
/// the broad semantic side of the query.
const MAX_PLAIN_QUERY_TERMS: usize = 6;
const HIGH_DF_RATIO: f32 = 0.05;

const FIELD_CHUNK_ID: &str = "chunk_id";
const FIELD_TEXT: &str = "text";

fn build_schema() -> (Schema, Field, Field) {
    let mut builder = Schema::builder();
    let chunk_id_field = builder.add_bytes_field(FIELD_CHUNK_ID, STORED | INDEXED);
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

    /// Tombstone-side counterpart to [`Self::add`]: queue
    /// `delete_term` calls for each id. Caller must `commit` /
    /// `finalize` to make the deletes durable. No-op on empty input.
    pub fn delete_chunks(&mut self, chunk_ids: &[ChunkId]) {
        for id in chunk_ids {
            let term = Term::from_field_bytes(self.chunk_id_field, &id.0);
            self.writer.delete_term(term);
        }
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
    /// Query-parser syntax is honored when it parses cleanly. If prose
    /// happens to contain parser metacharacters or malformed syntax, we
    /// fall back to the same pruned plain-term query used for natural
    /// language so `knowledge_query` remains robust for LLM-generated
    /// text.
    pub fn search(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(ChunkId, f32)>, BucketError> {
        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let total_t = Instant::now();
        let searcher = self.reader.searcher();
        let doc_count = searcher.num_docs();

        let parse_t = Instant::now();
        let (query, query_diag) = if looks_like_tantivy_syntax(query_text) {
            let parser = QueryParser::for_index(&self.index, vec![self.text_field]);
            match parser.parse_query(query_text) {
                Ok(query) => (
                    query,
                    SparseQueryDiag {
                        mode: "parser",
                        raw_terms: 0,
                        used_terms: 0,
                        zero_df_terms: 0,
                        high_df_terms: 0,
                        max_df: 0,
                        df_ms: 0,
                    },
                ),
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "knowledge_sparse: parser rejected query; falling back to plain terms",
                    );
                    self.build_pruned_plain_query(
                        query_text,
                        doc_count,
                        "plain_after_parser_error",
                    )?
                }
            }
        } else {
            self.build_pruned_plain_query(query_text, doc_count, "plain_pruned")?
        };
        let parse_ms = parse_t.elapsed().as_millis();

        let search_t = Instant::now();
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(top_k).order_by_score())
            .map_err(|e| BucketError::Other(format!("tantivy search: {e}")))?;
        let search_ms = search_t.elapsed().as_millis();

        let doc_fetch_t = Instant::now();
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
        let doc_fetch_ms = doc_fetch_t.elapsed().as_millis();
        tracing::info!(
            mode = query_diag.mode,
            docs = doc_count,
            raw_terms = query_diag.raw_terms,
            used_terms = query_diag.used_terms,
            zero_df_terms = query_diag.zero_df_terms,
            high_df_terms = query_diag.high_df_terms,
            max_df = query_diag.max_df,
            hits = out.len(),
            parse_ms = parse_ms as u64,
            df_ms = query_diag.df_ms,
            search_ms = search_ms as u64,
            doc_fetch_ms = doc_fetch_ms as u64,
            total_ms = total_t.elapsed().as_millis() as u64,
            "knowledge_sparse: tantivy timing",
        );
        Ok(out)
    }

    fn build_pruned_plain_query(
        &self,
        query_text: &str,
        doc_count: u64,
        mode: &'static str,
    ) -> Result<(Box<dyn Query>, SparseQueryDiag), BucketError> {
        let mut analyzer = self
            .index
            .tokenizer_for_field(self.text_field)
            .map_err(|e| BucketError::Other(format!("tantivy tokenizer_for_field: {e}")))?;
        let mut stream = analyzer.token_stream(query_text);
        let mut seen: HashSet<String> = HashSet::new();
        let mut raw_terms = 0usize;
        let mut terms = Vec::new();
        while stream.advance() {
            raw_terms += 1;
            let text = stream.token().text.clone();
            if seen.insert(text.clone()) {
                terms.push(text);
            }
        }

        if terms.is_empty() {
            return Ok((
                Box::new(EmptyQuery),
                SparseQueryDiag {
                    mode,
                    raw_terms,
                    used_terms: 0,
                    zero_df_terms: 0,
                    high_df_terms: 0,
                    max_df: 0,
                    df_ms: 0,
                },
            ));
        }

        let searcher = self.reader.searcher();
        let mut term_stats = Vec::with_capacity(terms.len());
        let mut zero_df_terms = 0usize;
        let mut high_df_terms = 0usize;
        let mut max_df = 0u64;
        let df_t = Instant::now();
        for term_text in terms {
            let term = Term::from_field_text(self.text_field, &term_text);
            let df = searcher
                .doc_freq(&term)
                .map_err(|e| BucketError::Other(format!("tantivy doc_freq: {e}")))?;
            max_df = max_df.max(df);
            if df == 0 {
                zero_df_terms += 1;
                continue;
            }
            let high_df = doc_count > 0 && (df as f32 / doc_count as f32) > HIGH_DF_RATIO;
            if high_df {
                high_df_terms += 1;
            }
            term_stats.push((term, df, high_df));
        }
        let df_ms = df_t.elapsed().as_millis() as u64;
        term_stats.sort_by_key(|(_, df, _)| *df);

        let mut selected: Vec<Term> = term_stats
            .iter()
            .filter(|(_, _, high_df)| !*high_df)
            .take(MAX_PLAIN_QUERY_TERMS)
            .map(|(term, _, _)| term.clone())
            .collect();

        // If every matching term is broad, keep the least-broad terms
        // instead of falling back to a parser query over all broad terms.
        if selected.is_empty() {
            selected = term_stats
                .iter()
                .take(MAX_PLAIN_QUERY_TERMS.min(3))
                .map(|(term, _, _)| term.clone())
                .collect();
        }

        let used_terms = selected.len();
        let query: Box<dyn Query> = match selected.len() {
            0 => Box::new(EmptyQuery),
            1 => Box::new(TermQuery::new(
                selected.remove(0),
                IndexRecordOption::WithFreqs,
            )),
            _ => {
                let term_queries: Vec<Box<dyn Query>> = selected
                    .into_iter()
                    .map(|term| {
                        Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs))
                            as Box<dyn Query>
                    })
                    .collect();
                let min_should = if term_queries.len() >= 4 { 2 } else { 1 };
                Box::new(BooleanQuery::union_with_minimum_required_clauses(
                    term_queries,
                    min_should,
                ))
            }
        };

        Ok((
            query,
            SparseQueryDiag {
                mode,
                raw_terms,
                used_terms,
                zero_df_terms,
                high_df_terms,
                max_df,
                df_ms,
            },
        ))
    }
}

#[derive(Debug, Clone)]
struct SparseQueryDiag {
    mode: &'static str,
    raw_terms: usize,
    used_terms: usize,
    zero_df_terms: usize,
    high_df_terms: usize,
    max_df: u64,
    df_ms: u64,
}

fn looks_like_tantivy_syntax(query_text: &str) -> bool {
    query_text.bytes().any(|b| {
        matches!(
            b,
            b'"' | b':'
                | b'+'
                | b'-'
                | b'('
                | b')'
                | b'['
                | b']'
                | b'{'
                | b'}'
                | b'*'
                | b'~'
                | b'^'
        )
    })
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
    fn malformed_query_falls_back_to_plain_terms() {
        let id_a = ChunkId::from_source(&[1; 32], 0);
        let tmp = build_index_with(&[(id_a, "hello unclosed world")]);
        let index = open(&tmp);
        // Unclosed quote: tantivy's parser rejects this, but LLM
        // query text often contains stray punctuation. Sparse search
        // should keep retrieval alive by treating it as plain text.
        let results = index.search("\"unclosed", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id_a);
    }

    #[test]
    fn prose_with_parser_punctuation_searches_as_plain_terms() {
        let id_a = ChunkId::from_source(&[1; 32], 0);
        let tmp = build_index_with(&[(
            id_a,
            "Burger King is a chain of hamburger fast food restaurants founded in Florida",
        )]);
        let index = open(&tmp);

        let results = index
            .search(
                "Let's try `about(topic=\"burger joints in Florida\")`: maybe Burger King?",
                5,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id_a);
    }

    #[test]
    fn plain_query_pruning_keeps_rare_matching_terms() {
        let id_common = ChunkId::from_source(&[1; 32], 0);
        let id_needle = ChunkId::from_source(&[2; 32], 0);
        let mut chunks = vec![(id_needle, "common filler needle")];
        for i in 0..20 {
            let id = ChunkId::from_source(&[3 + i as u8; 32], 0);
            chunks.push((id, "common filler"));
        }
        let tmp = build_index_with(&chunks);
        let index = open(&tmp);

        let results = index.search("missing common needle", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, id_needle);
        assert!(!results.iter().any(|(id, _)| *id == id_common));
    }
}
