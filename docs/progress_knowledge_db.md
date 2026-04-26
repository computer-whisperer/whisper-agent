# Knowledge DB — progress tracker

Living status doc for the knowledge-bucket initiative. Pairs with the
authoritative design at [`design_knowledge_db.md`](design_knowledge_db.md).

This doc is intentionally short — slice list + what's next + dangling
cleanup. Architecture decisions live in the design doc; commit messages
have the per-slice detail.

Last updated: **2026-04-25**.

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

## Currently working

**MediaWiki XML source adapter.** Goal: get a representative dataset
(Simple English Wikipedia first, full enwiki as stretch) so the pipeline
can be validated end-to-end at non-toy scale and the query path can be
tired-kicked against real content.

Agreed defaults:
- Streaming `quick-xml` parser
- bz2 streaming via the `bzip2` crate (Wikipedia ships `.xml.bz2`)
- Pass raw wikitext through to the chunker (no plaintext extraction
  yet — defer until RAG quality matters)
- Default to ns=0 (main articles) only
- Drop redirects and empty pages

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
  every slot load. Fine at workspace scale, ~30 min for 50M vectors at
  enwiki scale. The first slot-load that hurts is when we cross it.
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
