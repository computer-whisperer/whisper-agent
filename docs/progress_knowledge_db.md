# Knowledge DB — progress tracker

Living status doc for the knowledge-bucket initiative. Pairs with the
authoritative design at [`design_knowledge_db.md`](design_knowledge_db.md).

This doc is intentionally short — slice list + what's next + dangling
cleanup. Architecture decisions live in the design doc; commit messages
have the per-slice detail.

Last updated: **2026-04-29**.

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
| 9   | `QueryBuckets` wire path + WebUI search form                   | `9c82346` |
| HP  | HNSW persistence — dump on build, reload on open               | `1658a8f` |
| 8c  | WebUI bucket lifecycle — Create / Delete / StartBuild / Cancel + progress events | `da52b6a` |
| 8d  | `source_ref` plumbed through `QueryHit` + display in UI        | `fec561a` |
| 8e  | WebUI search results — expandable + color-coded retrieval path | `436e8be` |
| KQ  | `knowledge_query` builtin tool + per-pod `[allow.knowledge_buckets]` scope | `797331e` |
| R   | Resumable builds — `build.state` log, `resume_slot`, queryable-during-build, periodic mid-build HNSW dumps, planning-phase factoring | `da0c5c5..3f6ec38` |
| BPE | Real BPE chunker with hf-hub auto-fetch + heuristic fallback   | `70edc51` |
| MMAP| `mmap` chunks/vectors readers + `ServingMode` dispatch         | `8429d07` |
| M   | Mutation triad — `insert` / `tombstone` / `compact` + delta layer + reload-aware slot + pre-release hardening | `9ff15cb..c666f7f` |
| T   | Tracked source kind — config variant, `FeedDriver` + `WikipediaDriver`, feed-state file, build-pipeline wiring, `FeedWorker` (cadence + delta polling) | `28f18d4..8add262` |
| W   | Wiki-scale fixes — multistream `bz2` decoder, embed-call cap, content-hash dedup, pipelined chunker→embedder→writer with concurrent embed requests, resume preamble observability | `fa64527..9564927` |
| RC  | Replay in-flight build progress to freshly-connected clients   | `fd587bb` |
| DA  | Delta application — `Bucket::apply_delta`, `SourceIndex` reverse-lookup, FeedWorker drives apply per tick | `7702d34..82b1bcd` |
| PN  | "Poll now" wire path — bounded trigger channel, scheduler dispatch, WebUI button | `bb29d77` |
| BS  | `Bucket::insert` per-API-call cap fix — apply_delta no longer blows past TEI's `max_client_batch_size` | `ec6a9f7` |
| RS  | Background resync — manual + scheduled. `resolve_resync_source`, `reset_delta_cursor_to`, `BuildIntent::Resync`, `last_resync_at` persistence, FeedWorker resync arm, "Resync now" button | `07bdc26..9c4d7c3` |
| Q1  | `recall_eval` binary — dense-recall@K harness for quantization regression testing | `025be2a` |
| Q2a | f16 quantization in vectors.bin — half on-disk vector size, ~1% recall noise floor | `26541e3` |
| Q3  | int8 quantization in vectors.bin — symmetric per-vector scale, ~4× compression | `5e91517` |
| GC  | Orphan-slot GC at registry load — Failed / no-manifest / superseded Planning/Building dirs swept; active + resumable preserved | `91a08bc` |
| ETA | Elapsed-time stopwatch on build progress rows — wire surface, replay-aware, scales s→m→h→d | `abda41b` |
| AB  | apply_delta batch — hoisted per-page tombstone+insert into one tantivy commit pair, saturated embedder | `6c25f4e` |
| BT  | Boot-time fs-walk fix — replaced recursive `dir_size`/`total_dir_bytes` walks with manifest stat reads, eliminating ~50s startup hang | `0dfa1ba` |
| KM  | `knowledge_modify` — LLM-callable insert/tombstone for managed buckets, EmptySource bootstrap, scope-gated | `efeebb7..591ad65` |
| MQ  | Multi-bucket query fan-out — `join_all` across per-bucket dense+sparse futures; max-of-per-bucket latency | `a5ff7d2` |
| HQ  | HNSW-side f16 quantization — `DistF16L2`, enum-dispatched `DenseInner::{F32, F16}`, plumbed through `DenseIndex::{empty, build, load_from*}`. Halves `dense.hnsw.data` for any quantized bucket. Drive-by: resume rebuild path stride bug (was hardcoded f32) | `0a29cb8` |
| HQ8 | HNSW-side int8 quantization — `DistInt8L2` (i32 arithmetic, scale-cancels-out), `DenseInner::Int8` with calibrated dataset-wide scale, `dense.int8_scale` sidecar. Quarters `dense.hnsw.data` for int8 buckets vs f32. First-batch calibration in the streaming build path; first-256-sample calibration in `DenseIndex::build` and the resume rebuild path. | `5ac7b6a` |
| PB1 | Pod-scope bucket discovery — `BucketEntry.{scope,pod_id}`, `BucketRegistry.pod_buckets` per-pod sub-registry, `register_pod_buckets` walks `<pod_dir>/buckets/`, scheduler boot folds each loaded pod in, `BucketSummary.pod_id` on the wire. ListBuckets surfaces pod-scope entries; query / lifecycle wire ops land in follow-ups. | `91ded94` |
| PB2 | Pod-scope query path — `loaded_bucket_pod` + scope-keyed cache, `find_entry(scope, pod_id, name)`, `resolve_query_targets` resolver with `pod:`/`server:` prefix grammar (bare names auto-disambiguate against pod's own pod-scope buckets ∪ server-scope allow list; ambiguity errors with the disambiguating prefix). Wired through `knowledge_query` and `knowledge_modify`. 10 resolver unit tests. | `8e6ff40` |
| PB3a| Lifecycle wire schema — `pod_id: Option<String>` on `CreateBucket`/`DeleteBucket`/`StartBucketBuild`/`CancelBucketBuild`/`ResyncBucket`/`PollFeedNow`/`QueryBuckets` plus matching events `BucketDeleted`/`BucketBuildStarted`/`BucketBuildProgress`/`BucketBuildEnded`/`FeedPollAccepted`. Server handlers reject `pod_id=Some` with a clear "not yet implemented" Error; behavior to land in `PB3b`. | `3e97534` |
| PB3b| Pod-scope CRUD/build behavior — `BucketKey {pod_id, name}` rekeys `active_bucket_builds`/`active_bucket_progress` so server-scope and pod-scope share neither namespace nor in-flight build state. `BucketTaskUpdate::{Progress,BuildEnded}` carry `pod_id`; `run_build` threaded with `pod_id`. Registry gains `insert_pod_entry`/`remove_pod_entry`/`refresh_pod_entry`/`evict_loaded_pod`. `CreateBucket`/`DeleteBucket`/`StartBucketBuild`/`CancelBucketBuild` handlers branch on `pod_id` — pod-scope buckets live at `<pod_dir>/buckets/<id>/` and route through the per-pod sub-registry. `ResyncBucket`/`PollFeedNow`/`QueryBuckets` pod-scope still rejected (deferred to PB3c). 829 lib tests passing. | `f27cdd2` |
| PB3c| Pod-scope `QueryBuckets` — handler routes via `BucketRegistry::find_entry(scope, pod_id, name)` for entry lookup and `loaded_bucket_pod(pod_id, name)` for the cached HNSW load. Same-named buckets in server vs. a pod resolve to distinct entries because the lookup branches on scope; error messages disambiguate the two on miss. `ResyncBucket`/`PollFeedNow` pod-scope still rejected (deferred to PB3d — needs feed-worker re-keying for tracked pod-scope sources). 829 lib tests passing. | `b6a01c2` |
| PB3d| Pod-scope `ResyncBucket` + `PollFeedNow` — `active_feed_workers` re-keyed by `BucketKey`, `spawn_for_registry` walks `pod_buckets` in addition to `buckets`, resync-request channel re-typed to `(Option<String>, String)` so a pod-scope worker's scheduled fire dispatches back to the correct sub-registry. `FeedWorker` carries `pod_id` and `apply_one_delta` branches between `loaded_bucket` (server) and `loaded_bucket_pod` (pod). Handlers drop the `pod_id.is_some()` rejection — entry lookup goes through `find_entry(scope, pod_id, name)`, in-flight builds key by `BucketKey`, `run_build` is now passed actual `pod_id` (was hardcoded `None` for resync), and `delete_bucket` removes feed-worker entries by scope-aware key so a same-named server-scope worker isn't mistakenly stopped when a pod's bucket is deleted. 829 lib tests passing. | `9fb81c9` |

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

## End-to-end validation (full simplewiki, real TEI, tracked-source)

Run on 2026-04-27 against the same Qwen3-Embedding-0.6B + bge-reranker-v2-m3
endpoints, this time as a tracked-source bucket (`wikipedia_simple` →
`WikipediaDriver` → cached `simplewiki-20260401-pages-articles-multistream.xml.bz2`).
First run on the post-pipeline-rework code (pipelined embedder +
content-hash dedup + BPE chunker + multistream bz2 + mmap readers).

| Metric                  | Value             |
|-------------------------|-------------------|
| Source pages emitted    | 280,502           |
| Chunks indexed          | 870,996           |
| Embedder dim            | 1024              |
| Build wall-clock        | 2h 28m            |
| Bucket disk size        | 9.4 GiB           |
| chunks.bin              | 1.14 GiB          |
| vectors.bin             | 3.57 GiB          |
| dense.hnsw (data+graph) | 4.16 GiB          |
| index.tantivy           | 435 MiB           |
| build.state             | 38 MiB            |
| Per-chunk disk          | ~11.4 KiB         |
| chunk_count == vector_count | ✓ (dedup verified after fix in `9564927`) |

**Quality**: WebUI search is dead-on across hand-typed queries — the
content-hash dedup that fixed the 870996/873035 mismatch in the
previous attempt also cleared up the resulting query-time confusion.

**Extrapolation to full enwiki** (Qwen3-0.6B, all-namespaces ≈ ~30M
chunks, ~35× this run): ~3.6 days at this throughput. Pre-quantization,
disk would be ~340 GiB — close to the design doc's 300 GiB estimate.

## Quantization spot-test (full simplewiki, real TEI)

Run on 2026-04-28. Three buckets built from the same pinned base
snapshot (`simplewiki-20260401`), same 870,996 chunks each, queried
with the built-in 15-query simplewiki list via `recall_eval` against
the f32 baseline (clean, zero deltas).

| variant | disk    | build wall-clock | mean recall@10 | median | min   | perfect |
|---------|---------|-------------------|----------------|--------|-------|---------|
| f32     | 8.8 GiB | 2h 28m            | 1.000 (ref)    | —      | —     | 15/15   |
| f16     | 7.1 GiB | 2h 24m            | **0.960**      | 1.000  | 0.800 | 10/15   |
| int8    | 6.3 GiB | 2h 22m            | **0.920**      | 1.000  | 0.000 | 12/15   |

f16 quality matches expectations (~1–4% top-10 noise floor). int8 has
one outlier query ("human immune system white blood cells") with
recall@10 = 0.00 that recovers to recall@50 = 0.88 — top-K cutoff
shuffle from quant noise, not semantic divergence; for typical RAG
flows that pull more candidates and rerank, this is fine.

**Methodology gotcha**: a first pass against the live `wikipedia_simple`
(which had delta `20260427` applied) gave bogus recall numbers (mean
0.63) because that bucket's content set differs from the pure-base
f16/int8 (118k delta chunks + 107k tombstones). Switched to the
backup `wikipedia_simple.bak` (clean snapshot, 0 deltas) for the
real numbers above. Recall comparisons need identical content sets.

**Build wall-clock gotcha**: the int8 manifest reports
`build_started_at = 05:03 UTC` and `build_completed_at = 15:52 UTC`,
which would suggest a ~10h 49m build. But the bucket dir was
pre-staged at 05:03; the *actual* TEI traffic didn't begin until the
operator kicked off the build at ~13:30 UTC the next morning. True
wall-clock is ~2h 22m, in line with f16/f32. The
`build_started_at` field tracks the first slot-creation moment, not
the first embedder request — worth a follow-up to make this less
misleading on next inspection.

**Disk-size note**: f16 saves only ~19% and int8 ~28% of total bucket
disk because `dense.hnsw.{graph,data}` is still f32 — the HNSW-side
quantization deferred item below would compound these savings.

## Deferred / not started

Numbered loosely so the conversation can refer to slices, not
chronological — order may shuffle as the dataset reveals what hurts.

| #   | Title                                                          | Why deferred |
|-----|----------------------------------------------------------------|--------------|
| 10+ | Auto-compaction triggers (delta-ratio / tombstone-ratio thresholds, scheduled) | `Bucket::compact` is callable manually; auto-trigger heuristics still TBD. Real thresholds need observation on actual mutating buckets; should also have time-based triggers (e.g. compact pod memory daily). |
| 10+ | Live-mode post-turn relevance nudge                            | Per design doc § "Live retrieval mode". Cross-cuts scheduler — not load-bearing for v1. |
| 10+ | Per-pod buckets — lifecycle ops (`CreateBucket`/etc. with `pod_id`), WebUI scope toggle, scoped wire IDs on `QueryBuckets` | Foundation + LLM-tool query path landed in `PB1`/`PB2`. Outstanding: scoped strings on `QueryBuckets`/`CreateBucket`/`DeleteBucket`/`StartBucketBuild`/`CancelBucketBuild`/`ResyncBucket`/`PollFeedNow`, plus a WebUI scope selector in the create form. |

## Dangling cleanup (none blocking)

- ~~**HNSW persistence**~~ — landed in `1658a8f`. Slot open now reads
  `dense.hnsw.{graph,data}` instead of rebuilding (~390× faster cold
  load on simplewiki/tei_50k). Initial *build* wall-clock is unchanged
  — full enwiki rebuild is still gated by build-side throughput, but
  reuse across server restarts is solved.
- ~~**`tokenizers` crate integration**~~ — landed in `70edc51`. Real
  BPE via `hf-hub` auto-fetch with the char-heuristic kept as offline
  fallback. Chunk ids now stable across embedder rotations as the
  design intended.
- ~~**mmap upgrade for `chunks.bin` / `vectors.bin`**~~ — landed in
  `8429d07`. Reader path uses `mmap` with a `ServingMode` dispatch so
  RAM-resident vs disk-backed buckets share the file format.
- ~~**Resumable / incremental builds**~~ — landed in the `R` slice
  group (`da0c5c5..3f6ec38`). `build.state` log + `resume_slot` +
  periodic mid-build HNSW dumps + queryable-during-build all in.
  Cancel / crash now picks up at the last `BatchEmbedded` checkpoint
  instead of restarting from zero.
- ~~**GC pass for orphaned slot directories**~~ — landed in `91a08bc`.
  `gc_orphan_slots` runs at `BucketRegistry::load`; removes Failed,
  no-manifest, unparseable-manifest, and superseded
  Planning/Building slots. Active and most-recent resumable are
  always preserved; Ready/Archived are owned by a separate retention
  sweep.
- **`build_simplewiki` binary** — now obsolete for non-stress-testing
  flows; the WebUI lifecycle covers everything it did. Keep around
  for batch / scripted scenarios but consider deleting once the wiki
  bench harness has a more permanent home.
- **`MockEmbedder` / `MockReranker` duplicated inline** in
  `disk_bucket.rs` and `query.rs` test modules. Move to a shared
  `test_support` mod when a third call site appears.
- **Per-bucket tantivy tokenizer config** — `bucket.toml`'s
  `search_paths.sparse.tokenizer` is parsed but ignored at slot-build
  time. Wire through when we have multiple options worth swapping.
- ~~**Multi-bucket parallel fan-out in `QueryEngine`**~~ — landed in
  `a5ff7d2`. `futures::future::join_all` across per-bucket search
  futures; total query latency is now max-of-per-bucket instead of
  sum-of-per-bucket. Relevant once a pod has multiple buckets in
  scope (e.g. memory + wikipedia).
- ~~**Scheduler-thread block on `DiskBucket::open`**~~ — landed in
  `12bda29`. `handle_start_bucket_build` and `handle_resync_bucket`
  no longer call `DiskBucket::open` on the scheduler task; both
  the open and the resumable-slot scan run inside `run_build`'s
  `spawn_blocking`, with errors flowing through the existing
  `BuildEnded { Error }` path.
- **`build_started_at` is creation time, not first-embedder time**
  — the int8 manifest's apparent ~10h build (05:03 → 15:52 UTC) was
  actually 2h 22m of TEI traffic; the field stamps when the slot dir
  was created, which can predate the actual build dispatch by hours.
  Cosmetic but misleading in postmortems; consider stamping
  `build_started_at` from `run_build`'s entry point instead.

## Pointers

- Design: [`design_knowledge_db.md`](design_knowledge_db.md)
- Bucket model: `src/knowledge/`
- Wire types: `crates/whisper-agent-protocol/src/lib.rs` (search
  `BucketSummary`)
- WebUI modal: `crates/whisper-agent-webui/src/app/modals/buckets.rs`
- Targeted embedder for wikipedia plan: Qwen3-Embedding-0.6B (1024-dim).
