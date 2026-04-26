# Knowledge DB — progress tracker

Living status doc for the knowledge-bucket initiative. Pairs with the
authoritative design at [`design_knowledge_db.md`](design_knowledge_db.md).

This doc is intentionally short — slice list + what's next + dangling
cleanup. Architecture decisions live in the design doc; commit messages
have the per-slice detail.

Last updated: **2026-04-26**.

## Slices landed

| #   | Title                                                          | Commit     |
|-----|----------------------------------------------------------------|------------|
| 0   | retrieval-provider plumbing (EmbeddingProvider / RerankProvider / TEI clients) | `a1e5ed2` |
| —   | living design doc                                              | `91c416b` |
| 1   | bucket-model foundation (types, Bucket trait, TOML schemas)    | `cf18513` |
| 2   | SourceAdapter trait + MarkdownDir, Chunker + TokenBasedChunker | `9ff37e7` |
| 3   | chunks.bin storage + DiskBucket — first end-to-end pipeline    | `4a8f3a2` |
| 4   | wire embedder + persist vectors (vectors.bin / vectors.idx)    | `7f66ccd` |
| 5   | HNSW dense index — first working `dense_search`                | `a67e5b2` |
| 6   | tantivy sparse index — first working `sparse_search`           | `898f467` |
| 7   | multi-bucket query layer with reranker fusion                  | `94309de` |
| 8a  | bucket registry + `ListBuckets` wire surface                   | `33c04e9` |
| 8b  | WebUI knowledge-buckets modal — read-only bucket list          | `9e9925d` |
| MW  | MediaWiki XML source adapter — streaming, bz2, raw wikitext    | `03d4a7e` |
| —   | `build_simplewiki` one-off binary (driver for the validation)  | `9c0d864` |

## End-to-end validation (Simple English Wikipedia, mock embedder)

Driven via `build_simplewiki` against
`simplewiki-latest-pages-articles.xml.bz2` (331 MB compressed).
Run on 2026-04-26:

| Metric                  | Value             |
|-------------------------|-------------------|
| Source pages emitted    | 280,502           |
| Source read errors      | 0                 |
| Chunks indexed          | 685,236           |
| Mock vector dim         | 384               |
| Disk size (bucket)      | 2.5 GiB           |
| chunks.bin              | 1.1 GiB           |
| vectors.bin             | 1004 MiB          |
| chunks.idx + vectors.idx| 27 MiB each       |
| tantivy index           | 391 MiB           |
| Build wall-clock        | 47 min            |
| HNSW build (685k vec)   | ~46 min (dominant)|
| Query latency           | 3–8 ms            |
| Per-chunk disk          | ~3.7 KiB          |

**Quality**: Sparse path (BM25 over raw wikitext) returns
semantically dead-on results out of the box for tested queries
("albert einstein theory of relativity", "world war two pacific
theater", "python programming language", "octopus intelligence",
"constitution of the united states") — every query's top-5 was
on-topic. Validates the choice to defer wikitext → plaintext
extraction; embeddings + BM25 tolerate the markup.

**Surprises / things to follow up on:**
- HNSW build is ~98% of total wall-clock at this scale. Wikipedia-
  scale builds (~30M chunks) are flat-out infeasible without HNSW
  persistence — the cleanup item below is now the gating constraint
  for the next scale step, not just a future TODO.
- Mock embedder makes the dense path semantically random; the
  sparse path alone is doing all the work in the validation queries
  above. A real TEI run is the natural next quality test, but is
  blocked on a TEI server existing in the test environment.
- Extrapolated full-enwiki size at this dim: ~130 GiB. With
  Qwen3-Embedding-0.6B (1024-dim, 2.7× vector growth): ~250 GiB,
  matching the design doc's ~300 GiB estimate.

## End-to-end validation (50k simplewiki, real TEI)

Run on 2026-04-26 against `[embedding_providers.qwen3_embed_0_6b]`
(Qwen/Qwen3-Embedding-0.6B at `http://10.1.0.198:8080`) and
`[rerank_providers.bge_reranker_v2_m3]` (BAAI/bge-reranker-v2-m3 at
`http://10.1.0.198:8081`), batch size 128.

| Metric                  | Value             |
|-------------------------|-------------------|
| Pages emitted           | 50,000            |
| Chunks indexed          | 133,356           |
| Embedder dim            | 1024              |
| Build wall-clock        | 31 min (1879s)    |
| Embed throughput        | ~34 pages/s       |
| HNSW build              | ~7 min            |
| Disk size               | 833 MiB           |
| Query latency           | 160–230 ms        |

**Quality**: dense path now contributes 3–5 of the top-5 hits across
all five sample queries (vs ~0 with the mock embedder). Most clear
demo is "octopus intelligence and learning": dense finds octopus-
camouflage and RNA-editing chunks that pure BM25 wouldn't surface
from a keyword match. Top-5 was on-topic for every query.

**Bug found + fixed**: TEI 1.9.3 ignores the `top_n` field on
`/rerank` requests; `QueryEngine` was trusting the reranker to honor
it. Fixed in `23ba44d` by capping locally after the rerank call —
contract holds against any provider now.

**Tuning**: `EMBED_BATCH_SIZE` bumped 32 → 128 (commit `2da2698`).
At 32, per-batch HTTP overhead dominated; 128 gave a 2.6× end-to-end
speedup on the 50k run with no observed downside. Still under TEI's
130-chunk per-batch token budget for ~500-token chunks.

**Full-simplewiki extrapolation** (TEI, batch=128):
~2.3h embed + ~36 min HNSW = ~3 hours. Down from the ~7h projected
at batch=32. Still gated by HNSW persistence for anything bigger
than simplewiki.

## Deferred / not started

Numbered loosely so the conversation can refer to slices, not
chronological — order may shuffle as the dataset reveals what hurts.

| #   | Title                                                          | Why deferred |
|-----|----------------------------------------------------------------|--------------|
| 8c  | WebUI bucket lifecycle — Create / Delete / StartBuild + progress events | Hand-writing `bucket.toml` + driving `build_slot` from tests is enough to stand up a wikipedia bucket today; the form / progress UI can come once we know what real builds need to surface. |
| 9   | WebUI query panel — bucket multi-select + ranked results       | Built on top of `QueryEngine`. Useful as soon as a real bucket exists to query against — likely the next user-facing slice after MediaWiki. |
| 10+ | `knowledge_query` builtin tool                                 | Wire `QueryEngine` into the runtime tool catalog so an agent can call it. Needs scope/permission gating per `[allow.knowledge_buckets]`. |
| 10+ | `managed` source kind + `knowledge_modify` tool                | For pod memory / agent-authored notes. Mutations via `Bucket::insert` / `Bucket::tombstone`. |
| 10+ | Compaction triggers (delta-ratio / tombstone-ratio thresholds, manual) | Deferred until mutation path exists; until then there's no delta layer to compact. |
| 10+ | Live-mode post-turn relevance nudge                            | Per design doc § "Live retrieval mode". Cross-cuts scheduler — not load-bearing for v1. |
| 10+ | Per-pod buckets (`<pods_root>/<pod>/buckets/`)                 | Server-scope is enough until multi-pod isolation is a felt need. Today's `BucketScope::Pod` enum variant is unwired. |

## Dangling cleanup (none blocking)

- **HNSW persistence** — `DenseIndex::build` rebuilds from `vectors.bin` on
  every slot load *and* dominates initial build wall-clock (~46 of 47 min
  on the simplewiki validation run for 685k vectors). Wikipedia-scale
  builds (~30M chunks) are infeasible without persistence. **Now the
  gating constraint for the next scale step.**
- **`tokenizers` crate integration** — `TokenBasedChunker` uses a
  `chars_per_token=4` heuristic. For chunk-id stability across embedder
  swaps the chunker has to use real BPE. Land before any "production"
  bucket gets built.
- **mmap upgrade for `chunks.bin` / `vectors.bin`** — currently
  `Mutex<File>`. Profile before changing; query latency hasn't been
  measured at scale.
- **GC pass for orphaned slot directories** — cancelled / failed builds
  leave `slots/<id>/` on disk. Need a sweep on registry load (or a
  `compact` op) to clean these up.
- **`MockEmbedder` / `MockReranker` duplicated inline** in
  `disk_bucket.rs` and `query.rs` test modules. Move to a shared
  `test_support` mod when a third call site appears.
- **Per-bucket tantivy tokenizer config** — `bucket.toml`'s
  `search_paths.sparse.tokenizer` is parsed but ignored at slot-build
  time. Wire through when we have multiple options worth swapping.
- **Multi-bucket parallel fan-out in `QueryEngine`** — currently
  sequential across buckets, parallel within (dense + sparse). Bound by
  bucket count, not chunk count, so cheap to leave until a multi-bucket
  query is observably slow.

## Pointers

- Design: [`design_knowledge_db.md`](design_knowledge_db.md)
- Bucket model: `src/knowledge/`
- Wire types: `crates/whisper-agent-protocol/src/lib.rs` (search
  `BucketSummary`)
- WebUI modal: `crates/whisper-agent-webui/src/app/modals/buckets.rs`
- Targeted embedder for wikipedia plan: Qwen3-Embedding-0.6B (1024-dim).
