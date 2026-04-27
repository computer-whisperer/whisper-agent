# Knowledge Database

A retrieval system for whisper-agent: standing datasets (wikipedia, arxiv, paper sets) and pod-local datasets (a project workspace, a notes directory) made queryable via embedding search + reranking. Everything below is the plumbing that makes retrieval cheap, fresh, and auditable.

**Two delivery modes**, both load-bearing:

- A `knowledge_query` builtin tool the LLM can invoke deliberately. Used both as the initial integration shape *and* as the "expand" affordance for the live mode below.
- A **post-turn relevance-gated nudge**: after each assistant turn, the system runs retrieval against the LLM's recent output (likely the thinking block), and *if* anything scores above a relevance threshold, injects a brief system-reminder-style turn pointing at the hit — "related to what you were just thinking: this chunk of the Apollo program article, score 0.84. Use your tool to read it." The chunk content is *not* in the injection; the LLM uses `knowledge_query` to actually fetch it if it cares. Latency-relaxed (few hundred ms is fine, runs between turns), naturally below the rolling cache breakpoint, no breakage of conversation flow. See [Live retrieval mode](#live-retrieval-mode-post-turn-relevance-nudge).

Companion docs: [`design_pod_thread_scheduler.md`](design_pod_thread_scheduler.md) (the resource-registry model this slots into); [`design_permissions.md`](design_permissions.md) (the `[allow]` pattern that gates which buckets a pod can reach).

**Status:** v1 retrieval providers landed 2026-04-25 (`EmbeddingProvider` + `RerankProvider` traits, TEI implementations, hot-swap parity with `[backends.*]`). Bucket / vector-store model is the next agreed work; this doc captures that discussion as it develops.

## Decisions log

| Date       | Decision                                                                                          | Notes |
|------------|---------------------------------------------------------------------------------------------------|-------|
| 2026-04-25 | Two provider traits (`EmbeddingProvider`, `RerankProvider`) — not one combined trait.            | TEI hosts `/embed` and `/rerank` as separate single-model servers; the split mirrors that. |
| 2026-04-25 | Hot-swap from day one via the same `diff_catalog<T>` machine that handles `[backends.*]`.        | Runtime add/remove of providers was an explicit user requirement. |
| 2026-04-25 | Embedding model for the wikipedia-scale plan: **Qwen3-Embedding-0.6B** (1024-dim).               | User dropped from 4B variant after a 3-hour simplewiki playground build, in preparation for full-wiki scale. |
| 2026-04-25 | Buckets store **cached chunk text alongside vectors**, not just vectors-plus-source-pointer.     | Required for linked datasets to remain queryable when source moves; also wanted by reranking. See [Stored vs linked is one decision](#stored-vs-linked-is-one-decision-cache-the-chunks). |
| 2026-04-25 | Live mode is a **post-turn relevance-gated nudge** as a system-reminder-style turn — not pre-turn context augmentation. | The LLM gets a teaser pointing at a relevant chunk; uses the `knowledge_query` tool to actually retrieve it. Sits below the rolling cache breakpoint as a normal conversation turn; cache-control is a non-issue. Latency-relaxed. See [Live retrieval mode](#live-retrieval-mode-post-turn-relevance-nudge). |
| 2026-04-25 | **Both RAM-resident and disk-backed serving are supported modes** — chosen per-slot via deployment config, not by code path. | RAM is a natural fit for c-srv3's 3 TB DDR3 (sub-ms search); disk-backed is the right answer for smaller boxes with NVMe SSDs. Same file format serves both; `Bucket::search` reads `serving_mode` from the slot manifest. See [Serving modes](#serving-modes-ram-resident-vs-disk-backed). |
| 2026-04-25 | **Hybrid retrieval is required**: every bucket runs dense (embedding) + sparse (BM25) in parallel, with the reranker fusing candidates from all paths and all buckets. | Dense and sparse have different failure modes — neither alone is enough for an agent that swaps between code, prose, and structured content. The reranker is the unification layer that makes cross-bucket and cross-path scoring meaningful. See [Hybrid retrieval](#hybrid-retrieval-dense--sparse--rerank). |
| 2026-04-25 | **Tantivy** for the BM25 path; `hnsw_rs` (or equivalent) for the dense path; bincode + manifest gluing them together. | No all-in-one solution gives us best-in-class for both index types. The bucket directory becomes the integration point: HNSW index + tantivy index + chunks + manifest, all sharing chunk ids. |
| 2026-04-25 | **Chunk id is `blake3(source_record_hash ‖ chunk_offset_in_record)`** — stable across slots within a bucket as long as chunker config is unchanged. Embedder rotation reuses chunk ids; chunker changes invalidate them. | Live-mode nudges and tool-result citations name chunk ids; rotation must not break references in conversation history. |
| 2026-04-25 | **Bucket id = directory name, immutable.** LLM-facing tool surface uses scope prefixes (`server:wikipedia_en`, `pod:my_workspace`). | Same audit-trail pattern as pods. Scope prefix avoids shadowing surprises and makes `[allow.knowledge_buckets]` semantics simple — bare names refer to server scope, pod-local always prefixed. |
| 2026-04-25 | Server-level buckets are scheduler-resource-registry entries (alongside Backend / McpHost / HostEnv); pod-local buckets are not registry resources. | Server buckets are shared, ref-counted, GC'd on idle. Pod-local buckets are owned by the pod, lifecycle tied to it. Both expose the same `Bucket` trait; the registry just doesn't track pod-local instances. |
| 2026-04-25 | **Buckets support live mutation** via three primitives — `insert`, `tombstone`, `compact` — implemented as a delta layer over the immutable slot base. Tantivy handles in/out natively; HNSW gets a separate delta graph and tombstone filter. | Pod memory, wikipedia delta dumps, and arxiv ingest all share this mechanism. Updates are tombstone+insert (new chunk id); compaction is a new-slot build from current logical state. See [Incremental updates and compaction](#incremental-updates-and-compaction). |
| 2026-04-25 | **Queries get direct shared read access to the active slot** (Arc-shared, library-native concurrency); queries do not message-pass through the indexing thread. | Queueing queries behind HNSW inserts on a single thread would waste cores and break the live-mode latency budget. Tantivy's reader/writer model and a RwLock around HNSW give safe concurrent reads. See [Concurrency model](#concurrency-model-queries-vs-builds). |
| 2026-04-25 | **Source kinds are `stored`, `linked`, and `managed`.** "Managed" is for buckets that are authored entirely through the API (pod memory) — no external archive or path, mutations come exclusively through `insert` / `tombstone`. | Pod memory is durable but has no source layer in the stored/linked sense. Treating it as a third kind keeps the source enum explicit rather than overloading "linked-with-no-path." |
| 2026-04-25 | **Live (auto-injection) mode is deferred — initial implementation is the explicit `knowledge_query` tool plus a basic webui search panel.** | The webui search lets us kick the tires on retrieval quality and tune thresholds before binding any of it to agentic threads. Live mode is additive once the foundation is solid; nothing in the architecture is blocked by deferring it. |
| 2026-04-25 | Indexing CPU runs on a **dedicated OS thread per active build**, not on the tokio runtime.        | HNSW build + chunking would otherwise starve the scheduler. Single-owner write discipline; main runtime communicates via mpsc. See [Runtime model](#runtime-model-where-cpu-runs). |
| 2026-04-25 | Retrieval `Bucket::search` runs inline on a tokio worker for v1; revisit only if a search exceeds ~5 ms. | HNSW search is single-digit µs RAM-resident, ~1–3 ms even with bad memory locality. API stays `async fn` so a future move to `spawn_blocking` is a contained change. |
| 2026-04-26 | **A fourth source kind, `tracked`** — self-maintained feed where the system owns base + delta state against an external archive, polled on a configured cadence. Joins `stored` / `linked` / `managed`. | `stored` is "user hands us a frozen archive URL." `tracked` is "user names a feed family; system handles acquisition + currency." Different lifecycles and different state shape — squashing them obscures both. See [Tracked sources](#tracked-sources-self-maintained-feeds). |
| 2026-04-26 | **Closed `FeedDriver` enum, one variant per source family.** v1 ships `wikipedia`. | Closed enum keeps trait dispatch trivial and avoids any config-driven plugin loading. New drivers (Wikidata, arxiv-feed, ...) are added one at a time as needs surface. |
| 2026-04-26 | **Background monthly resync default-on for tracked buckets** — with a per-bucket toggle and an explicit "Resync now" UI button. | Daily incrementals can drift over time (silent log/oversight events, schema bumps); periodic resync against a fresh base snapshot is the correctness reset. Default-on so users don't have to know about this; toggleable + manual-trigger so an operator can take over when the build cost (e.g. ~4 days for enwiki on 0.6B, ~13 days for all-wiki) is inconvenient. |
| 2026-04-26 | **Turnkey UI for tracked-bucket creation.** A "Create tracked bucket" affordance produces the right `bucket.toml` from a small form per driver; the user is not expected to know URL conventions or schedule formats. | The whole point of `tracked` + closed-enum drivers is shipping a 1-click experience for buckets the system knows how to maintain. If users still have to hand-author the toml, the abstraction has failed. Hand-authored toml stays supported for scripted setups. |

Open forks tracked in [Open questions](#open-questions).

## What we're building

A "knowledge bucket" is a configurable, queryable index built from a source. Four flavors of source:

- **Stored** — we own a reproducible archive that the user pointed us at (a wikipedia XML dump file, an arxiv PDF cache). The archive lives on disk under our control; we can re-ingest from it any time. We don't manage its currency — what the user gave us is what the bucket reflects.
- **Linked** — we have a pointer (a workspace path, a URL) that we recorded with a hash at scan time. The target may move, change, or disappear. We tolerate all three by snapshotting chunk text into the bucket when we ingest.
- **Tracked** — a self-maintained feed: the user names a feed family (Wikipedia, future Wikidata / arxiv-feed) and the system handles acquisition + currency. Maintains a base snapshot on disk plus an applied-delta cursor, polling the feed on a configured cadence. Built on top of the `stored` machinery (shares parsers and slot lifecycle); distinguished by who owns the snapshot's currency. See [Tracked sources](#tracked-sources-self-maintained-feeds).
- **Managed** — content authored exclusively through the API (pod memory, agent-authored notes). No external source layer; mutations come via [`Bucket::insert`](#the-bucket-trait) / [`Bucket::tombstone`](#the-bucket-trait).

All four flavors expose the same shape downstream: a query returns ranked `(score, chunk_text, source_ref)` triples. The `source_ref` is best-effort — for stored / tracked buckets it deterministically resolves; for linked buckets it may dangle, in which case the cached chunk text is what the LLM gets; managed buckets carry only the API-supplied source ref.

Buckets live at two scopes, by analogy with backends and behaviors:

- `<server_root>/knowledge/<bucket>/` — server-level standing buckets, shared across pods. Wikipedia, arxiv-cs, paper sets.
- `<pod>/knowledge/<bucket>/` — pod-local buckets, not visible to other pods. A workspace index, a notes directory.

A pod's `[allow.knowledge_buckets]` list (analogous to `[allow.backends]`) gates which server-level buckets the pod's threads can query. Pod-local buckets are implicitly accessible to that pod only.

## The five layers

```
Source        → original archive or linked pointer + scan-time hash
   ↓ ingest
Chunks        → text units with provenance, cached in the bucket
   ↓ embed
Vectors       → one f32 vector per chunk, tagged with embedding model id
   ↓ index
ANN structure → HNSW graph (or replacement) over vectors
   ↑
Query path    → embed query → ANN search → fetch chunk text → rerank
```

These are five **independent lifecycle layers**, and the build pipeline is just the chain `ingest → chunk → embed → index`. We should be able to re-run any suffix of that chain without redoing earlier stages:

- Source change (linked workspace edited) → re-ingest + downstream
- Chunker change → re-chunk + re-embed + re-index (chunk boundaries dictate vector identity)
- Embedding-model swap → re-embed + re-index (chunks unchanged)
- Index-engine swap (e.g. HNSW → something else) → re-index only (vectors unchanged)

The "re-embed" case is the load-bearing one — when we upgrade the embedding model we want to keep the existing index queryable while a new one builds in the background, then atomically promote. See [Build slots](#build-slots).

## Hybrid retrieval: dense + sparse + rerank

Every bucket runs **two retrieval paths in parallel** — dense (embedding similarity over an HNSW index) and sparse (BM25 over a tantivy index) — with a **reranker** fusing candidates from all paths and all queried buckets into a final ranking. This is the standard shape for serious RAG deployments, and it's load-bearing for whisper-agent specifically because a single thread frequently queries across heterogeneous buckets (workspace code, wikipedia prose, paper PDFs) where dense-only or sparse-only would systematically fail.

### Why both paths are necessary

| Failure mode of dense-only | Failure mode of sparse-only |
|----------------------------|-----------------------------|
| Misses exact-term anchors: CVE numbers, version strings, function names, identifiers | Misses semantic equivalents: "ways to keep data after a crash" vs "durability mechanisms" |
| Embedding model bias toward common patterns; rare terminology gets averaged out | No notion of similarity beyond word overlap |
| Hard to score across buckets — embedding space differs by training data | Score depends on corpus size and term frequency distribution |

The hybrid output is consistently better than either path alone, and the gap widens for the kinds of queries an agent actually makes (mixed code-and-prose, often containing both natural-language intent and exact identifiers).

### Why the reranker is the unification layer

Score distributions are heterogeneous in two dimensions:

1. **Across buckets** — embedding cosine against a wikipedia bucket isn't comparable to embedding cosine against a code-comments bucket; BM25 scores depend on within-bucket statistics.
2. **Across paths** — dense cosine and BM25 raw score are on entirely different scales even within one bucket.

Naively merging "top-K from each source by score" is meaningless. The reranker scores `(query, chunk_text)` pairs fresh, independent of where the candidate came from — its output is the only credibly-comparable signal across the union. This is **why** reranking is in the path on every query, not just an optional polish step.

It's also why the live-mode relevance threshold is on **reranker score**, not on dense or sparse score.

### The hybrid query path

```
query text
   │
   ├── for each allowed bucket, in parallel:
   │     ├── dense path:  embed(query) → HNSW search → top-K_dense
   │     └── sparse path: tokenize     → tantivy BM25 → top-K_sparse
   │
   ├── union all candidates across buckets and paths (with bucket+path provenance)
   ├── deduplicate by (bucket, chunk_id) — a chunk can hit both paths in one bucket
   ├── rerank((query, chunk_text)*) — one batched call to the reranker provider
   └── return top-K_final by reranker score
```

`top-K_dense` and `top-K_sparse` are over-fetch parameters (probably ~16–32 each per bucket). The reranker only needs enough candidate breadth to find the genuinely-best matches; over-fetching past ~64 total per bucket diminishing-returns hard.

### Bucket-level capability flags

A bucket configures which paths it serves:

```toml
[search_paths.dense]
enabled = true

[search_paths.sparse]
enabled   = true
tokenizer = "default"
```

Either path can be disabled per bucket. Some content is well-suited to one but not the other:
- Highly structured tables with rare identifiers → sparse-only is often better (and embeddings of structured rows are notoriously poor).
- Pure narrative prose with consistent vocabulary → dense-only is sometimes acceptable, though we'd typically still want sparse as a safety net.
- Default for general-purpose buckets: both.

If a bucket has only one path enabled, the query path simply skips the missing one for that bucket.

### Storage and CPU implications

- **Storage delta**: tantivy indexes run ~30% of source text size. For all-wiki chunked text (~110 GB), that's ~35 GB additional per slot. Pushes the all-wiki bucket from ~400 GB to ~435 GB; no meaningful budget impact.
- **Build time**: tantivy indexing is ~100s of MB/s — far faster than the embedding pass that dominates wall-clock. It runs as a parallel phase alongside embedding and finishes well before the embedding queue drains.
- **Per-query CPU**: tantivy BM25 search is single-digit ms (similar order to disk-backed HNSW). Multi-bucket fan-out parallelizes across both paths. Reranker is the latency dominator (100–300 ms network round-trip to TEI), not the indexes.

## Stored vs linked is one decision: cache the chunks

The unifying choice is whether chunk text lives **inside** the bucket or only at the source. We cache chunks in the bucket, for both flavors:

- **Linked datasets stay queryable** when the workspace moves, files get edited, or the path goes away. We still serve the snapshot we indexed. A subsequent rescan can re-snapshot the changed parts.
- **Reranking** wants chunk text right next to the vector — if rerank had to hit the source archive every query, we'd couple latency to archive availability.
- **Storage cost is modest.** For a 1024-dim f32 vector at 4 KB and a 500-token chunk at ~2 KB of UTF-8 text, the chunks roughly *double* the storage relative to vectors-only. Vectors dominate either way (see [Volume estimates](#volume-estimates)).

This collapses the stored/linked distinction into a property of the **source layer only**: stored = reproducible archive, linked = recorded pointer. Layers 2–5 (chunks, vectors, index) look identical.

## Volume estimates

Sizing the design for the actual scale we're planning, with the embedding model the user moved to (Qwen3-Embedding-0.6B, 1024-dim).

**Per-chunk storage cost** (chunked at 500 tokens, 50-token overlap → ~450 effective new tokens/chunk):

| Component | Bytes | Notes |
|-----------|-------|-------|
| Vector (1024-dim f32) | 4 096 | 2 048 if we go f16; ~1 KB if we add int8 quantization |
| HNSW graph node | ~150 | M=16, layered; rough avg edge cost |
| Chunk text (UTF-8) | ~2 000 | 500 tokens × ~4 bytes/token English |
| Metadata (ids, hash, byte range) | ~100 | source_id, chunk_id, content hash, offsets |
| **Per chunk total** | **~6.4 KB** | |

**Wikipedia scale estimates** (rough — these are within ±30%, good enough to size budgets):

| Dataset | Articles | Tokens | Chunks | Index | + Archive | Bucket total |
|---------|----------|--------|--------|-------|-----------|--------------|
| Simple English wiki | ~285 K | ~50 M | ~110 K | ~0.7 GB | ~0.5 GB | **~1.2 GB** |
| English wiki (full) | ~6.8 M | ~6 B | ~13 M | ~80 GB | ~25 GB | **~105 GB** |
| All wiki (all languages) | ~58 M | ~22 B | ~50 M | ~300 GB | ~80 GB | **~400 GB** |
| arxiv (papers, longer docs) | ~2.5 M | ~25 B | ~55 M | ~330 GB | varies | **~400+ GB** |

**Sparse-index delta (tantivy):** tantivy indexes run ~30% of source text size. For all-wiki at ~110 GB cached chunk text, that's ~35 GB additional per slot. Pushes the all-wiki bucket from ~400 GB to ~435 GB; meaningful additions roll into the budget figures below without changing the order of magnitude.

**With double slots during model migration** (current index live, new index building alongside): roughly 2× the index column. All-wiki + arxiv + double-slot headroom = **~1.5 TB**.

**Future-proofing for larger embedding models** (vector storage scales linearly with dim):

| Model | Dim | Multiplier on vectors | All-wiki index |
|-------|-----|-----------------------|----------------|
| Qwen3-Embedding-0.6B | 1024 | 1× | 300 GB |
| Qwen3-Embedding-4B | 2560 | 2.5× | 540 GB |
| Qwen3-Embedding-8B | 4096 | 4× | 720 GB |

f16 quantization halves the vector column with negligible quality loss for nearest-neighbor search; we should keep that as a v1.5 optimization, not a v1 decision.

**Budget recommendation:**
- **2–3 TB** comfortably covers all-wiki + arxiv + workspace pod buckets + double-slot headroom on the 0.6B model. This is the natural target.
- **5 TB** allows experimenting with 4B model variants on subsets without juggling.
- **10 TB** is only required if we want all-wiki + arxiv + multiple paper sets + 8B model + multiple slots — i.e. a research scenario, not a production target.

The user's "few TB" allocation is the right place to land.

**Build time** is the real bottleneck, not storage. Extrapolating from the simplewiki run (3 hours on Qwen3-Embedding-4B):

- Throughput ≈ 5 K tokens/sec at the 4B variant.
- 0.6B variant should run roughly 3–5× faster at the same batch size (memory-bandwidth-limited rather than param-count-limited; larger batches help more on the smaller model).
- **English wiki on 0.6B**: ~4 days continuous.
- **All-wiki on 0.6B**: ~13 days continuous.
- **All-wiki + arxiv on 0.6B**: ~4 weeks continuous.

This is tractable only if the build is resumable across restarts and if partial progress is queryable. See [Build pipeline](#build-pipeline).

If the wall-clock matters more than dollar/watt cost, parallelizing across multiple TEI replicas is a clean horizontal scaling option — the `EmbeddingProvider` trait doesn't preclude wiring multiple endpoints behind a round-robin or work-stealing dispatcher. Out of scope for v1 but cheap to add later.

## Runtime model: where CPU runs

Whisper-agent's main scheduler runs on a tokio runtime under the working assumption that *nothing it does is CPU-intensive* — everything is async I/O, HTTP connections, JSON ser/de of model events, file reads. The knowledge system breaks that assumption in places, and we have to be deliberate about where the CPU work lands so it doesn't starve the scheduler.

### Indexing CPU work (heavy, bursty, off the runtime)

| Stage | CPU profile | At all-wiki scale |
|---|---|---|
| Source parse (MediaWiki XML, PDF, markdown walk) | hundreds of MB/s for markdown; PDFs are 5–50 ms each | hours total, parallelizable |
| Chunker / tokenizer (`tokenizers` crate) | ~10–50 MB/s per core | hours over 22B tokens, parallelizable |
| Content hashing (BLAKE3) | multi-GB/s | minutes — negligible |
| JSON ser/de of vector batches in/out of TEI | ~100 MB/s per core for f32-array JSON | meaningful at 200 GB of vector traffic; v1.5 optimization candidate |
| **HNSW graph build** | M × log(N) distance evals per insert (~few ms per chunk on 1024-dim) | ~30 min single-threaded, ~5–10 min parallel — but spread across days if built incrementally as vectors arrive |

Build cost is dominated by chunking + JSON; HNSW disappears under TEI's network latency *if we insert vectors as they arrive* rather than building the graph in a final pass. We should design for incremental insert.

**Placement: a dedicated OS thread per active build.** Owns the slot's file handles + HNSW index, pulls work from a `tokio::sync::mpsc` channel fed by the embedding pipeline, runs synchronous Rust code throughout. Compared to `tokio::task::spawn_blocking` for each batch:

- Single-owner write discipline (no `Arc<Mutex<Index>>`).
- Natural pause/resume/cancel via channel close.
- Easier to bound total resource use across multiple concurrent builds.
- Maps cleanly to the manifest-and-checkpoint mental model.

The main runtime sees only a small async API: `start_build`, `stop_build`, `build_status` — message-passing to the worker thread.

### Retrieval CPU work (small, per-call, on the runtime)

| Stage | CPU profile |
|---|---|
| Query embedding (HTTP to TEI) | negligible CPU; 50–200 ms network |
| **HNSW search** | ~0.8 ms at ef=64 / 50M vectors / 1024-dim; ~2.6 ms at ef=200 |
| Chunk text fetch (mmap from `chunks.bin`) | negligible |
| Rerank (HTTP to TEI) | negligible CPU; 100–300 ms network |

The HNSW search is the only meaningful CPU work in the query path, and it's small enough to run on a tokio worker. The `Bucket::search` API stays `async fn`; we run the call inline for v1, measure, and switch to `spawn_blocking` per-bucket if any single search exceeds ~5 ms in practice.

### Implications of the live retrieval mode

Live mode injects results as a **new conversation turn** rather than as augmented context for the upcoming turn — see [Live retrieval mode](#live-retrieval-mode-post-turn-relevance-nudge). That framing keeps the runtime story simple:

- **No latency-hiding rendezvous.** Retrieval fires *after* the assistant turn completes, in the gap between the LLM finishing and the next assistant turn beginning. A few hundred ms is acceptable because the user is reading / typing; nothing is blocked on it. If retrieval finishes after the user submits their next message, the result still lands as a turn before the next dispatch.
- **No cache-control gymnastics.** The injected turn is a normal conversation turn, sitting below the rolling cache breakpoint by definition. Old cache stays valid; the new turn participates in caching the same way any other turn does.
- **Per-call HNSW cost varies by serving mode.** RAM-resident search is sub-millisecond (see [Serving modes](#serving-modes-ram-resident-vs-disk-backed)); disk-backed is 5–50 ms on NVMe depending on cache state. Either is fine inline on tokio workers — the disk-backed case is at the edge but still well under the few-hundred-ms total budget for the post-turn nudge. Builds and queries don't share memory regardless of mode (per-slot ownership), so build CPU never directly contends with query CPU.

### Process boundary (deferred)

A future option is to split indexing into a separate `whisper-agent-indexer` binary communicating with the main server over IPC. Hard isolation, separate scheduling, but adds deployment complexity and complicates hot-swap. Not worth it for v1; revisit if we find we want to scale indexing horizontally across machines or run builds on a different host than the server.

## Live retrieval mode: post-turn relevance nudge

> **Status: deferred to v2.** The initial implementation lands the explicit `knowledge_query` tool plus a webui search panel; live mode is additive and not blocking on any of the foundation. This section captures the agreed long-term shape so the foundation doesn't accidentally close doors.

The endpoint shape — and the part the rest of the design has to cleanly support — is **automatic retrieval after each assistant turn, gated by a relevance threshold, surfaced as a system-reminder-style turn**. Not an inline context augmentation.

### Trigger and pipeline

```
assistant turn N completes (LLM stream finishes)
   ↓
post-turn hook fires with thinking + assistant content as input
   ↓
build query text (probably from thinking block — that's where the LLM
revealed what it was actually working on, before formatting for output)
   ↓
embed query (TEI /embed; 50–200 ms)
   ↓
search active buckets the pod has access to (sub-ms RAM-resident)
   ↓
score top-K candidates against threshold
   ↓
if any pass: format a brief "related results" turn and enqueue it
            into the conversation log, ordered before the next assistant turn
if nothing passes: do nothing
```

The injected turn looks roughly like:

```
<knowledge_nudge>
Related to your last thinking: chunk 47 of "Apollo program"
(bucket: wikipedia_en, score: 0.84). Use knowledge_query with
bucket="wikipedia_en", chunk_id=47 if you want the full text.
</knowledge_nudge>
```

The chunk content is deliberately omitted — the system has already paid for retrieval but the LLM may not care, and stuffing every nudge with full chunk text bloats context. The `knowledge_query` tool is the affordance for "this looks worth reading; pull it in."

### Properties this gives us

- **No cache invalidation.** The nudge is a regular conversation turn, below the rolling cache breakpoint by construction.
- **No latency cost on the user-perceived path.** The nudge fires post-turn; the user's next message dispatch only waits for it if it's still in flight when they submit. With sub-ms search and ~150 ms embed, it's almost always done first.
- **Naturally self-throttling.** No threshold-passing results = no nudge. A boring conversation produces no noise.
- **The tool is the expand mechanism.** The LLM has a single, consistent way to drill in (the `knowledge_query` tool), regardless of whether the prompt was a nudge or the LLM thought to query unprompted.

### Concurrency with user input

If the user submits the next message *before* retrieval completes, two ordering choices:

1. **Block the next dispatch on retrieval** (small wait, conversation always coherent).
2. **Drop the nudge if it didn't land in time** (no wait, sometimes no injection).

I'd default to (1) with a hard timeout of ~1 second — beyond that, drop. The wait is invisible compared to the LLM's own time-to-first-token on the next turn.

### What we still need to decide

The actual user-facing tuning of this lives in [Open questions](#open-questions) — what query text we synthesize, what threshold value, whether multiple bucket hits get one nudge each or merged into one, whether the nudge contains a snippet preview vs purely a pointer.

## Serving modes: RAM-resident vs disk-backed

Two supported serving modes, chosen per-slot in the manifest. Same on-disk file format serves both; `Bucket::search` reads the mode flag and either dereferences in-memory arrays or follows mmap'd pages.

The mode is a *deployment* decision, not a code-path divergence — the build pipeline writes the same files regardless, and a slot can be reloaded in a different mode after a config change without a rebuild.

### RAM-resident

For each active slot:

- **Vectors**: `vectors.bin` is loaded into a `Vec<f32>` (or `Box<[f32]>`) on slot activation. ~300 GB for all-wiki on 0.6B.
- **HNSW graph**: neighbor lists loaded into in-memory adjacency arrays. Around 5–10% of vector size.
- **Chunks**: `chunks.bin` is mmap'd regardless of mode (retrieval only fetches the small subset that scored well, page cache handles it efficiently).
- **Manifest + metadata index**: small, loaded eagerly into a `HashMap<ChunkId, ChunkRef>`.

Search latency: **single-digit µs to ~1 ms** depending on `ef`, dominated by dot-product cost on cached vectors.

This is the natural mode for c-srv3 (3 TB DDR3) and the configuration we're sizing the wikipedia-scale plan around. Memory budget at our targets:

| Configuration | Resident size |
|---|---|
| All-wiki + arxiv on 0.6B (vectors + graph, both slots during migration) | ~700 GB |
| Add 4B model experimental subset (wiki only) | + ~270 GB |
| Add a few pod-local workspace buckets | negligible |
| **Comfortable headroom for 3 TB box** | **~2 TB free** |

Slot rotation peaks at ~2× steady-state for the bucket being rotated (load-new before drop-old, refcount-driven). All-wiki: ~600 GB peak vs ~300 GB steady.

### Disk-backed

Same files, accessed via mmap rather than eagerly loaded:

- **Vectors**: `vectors.bin` is mmap'd. HNSW search dereferences vectors at the offsets the graph traversal visits; the OS page cache fills with whatever's hot.
- **HNSW graph**: also mmap'd. Upper layers (small, touched by every query) stay resident in cache effectively for free; the leaf-layer long tail pages in and out.
- **Chunks, manifest**: same as RAM mode (mmap'd / eager respectively).

Search latency: **~5–50 ms typical on NVMe** for a cold-on-leaf-pages query; faster as the working set warms. ~100+ ms on rotational disk; we don't recommend that configuration but the code doesn't prevent it.

The right deployment for: developer workstations, small servers, anyone with a substantial NVMe SSD but RAM budget under the index size. A box with 32 GB RAM and a 1 TB NVMe can serve a 200 GB wikipedia bucket — slowly compared to RAM-resident, but well within the latency budget for the post-turn live-retrieval nudge (a few hundred ms is fine).

### Choosing the mode

Per-slot config in the manifest:

```toml
[serving]
mode = "ram"      # or "disk"
```

Heuristic for the default if unset: load resident if `vectors_size + graph_size < 0.5 * available_RAM`, else mmap. Explicit override always wins. We probably want to surface this as a knob in the webui slot detail panel.

The choice can be flipped without a rebuild — re-activating a slot with a changed `serving_mode` just reloads it. Useful if a deployment's RAM situation changes (added DRAM, started running a second large bucket, etc.).

### Implications for quantization

Quantization (f16 / int8) is more impactful for disk-backed serving than RAM-resident: cutting vector size in half doubles the working set that fits in page cache, and halves disk bandwidth on cold queries. f16 is essentially free quality-wise for nearest-neighbor search; int8 needs a calibration pass but cuts to 1/4. RAM-resident benefits less because vectors are already resident — the win is just allocator footprint.

This is a v1.5 candidate either way; design the `vectors.bin` file format with quantization as a manifest field rather than baking f32 in.

### Build vs serve memory contention

In RAM mode, the indexing thread allocates the *new* slot's HNSW in its own memory, separate from the active slot's serving memory. No shared state, no contention. Slot promotion swaps which allocation queries hit.

In disk-backed mode, the picture is the same — the indexing thread writes to the new slot's files; queries hit the active slot's files via mmap. No shared mutable state. Promotion is an `active` symlink swap; in-flight queries against the old slot finish against the old mmap (reference-held), then the old mapping is released.

## Bucket and chunk identity

**Bucket id = directory name, immutable.** Same precedent as pods: directories are stable on disk, names are stable in references, you can't accidentally rename one without breaking historical citations. `bucket.toml` carries a freely-editable `name` for display purposes.

**LLM-facing scope prefixes** disambiguate between server-level and pod-local buckets in the namespace:

- `server:wikipedia_en` — a server-level bucket the pod has been granted access to via `[allow.knowledge_buckets]`.
- `pod:my_workspace` — a pod-local bucket, always implicitly available to that pod.

The `[allow.knowledge_buckets]` list contains bare server-level names (no prefix) — the prefix exists only at the LLM tool boundary, where ambiguity matters. Internally we carry a typed `BucketId { scope: Scope, name: String }`.

**Chunk id = `blake3(source_record_hash ‖ chunk_offset_in_record)`.** Stable across slots within a bucket as long as the chunker config is unchanged. Concrete properties:

- New embedder model → new slot, **same chunk ids**. Live-mode nudges from previous conversation history still resolve.
- New chunker config → chunks are by definition different units, so chunk ids change. Citations from before become stale, but that's correct: the chunks they referred to don't exist anymore as the same units.
- Compaction (rebuild of base from current state) → preserves chunk ids by construction; chunks that survived go through unchanged.

The build pipeline can dedupe work using chunk id: a restarted build that already wrote vectors for chunks 1..N can skip them on resume. Same property covers the case where compaction rebuilds the slot — chunks already in the source-state-at-compaction-time keep their ids.

## The Bucket trait

The abstraction boundary that the rest of whisper-agent talks to. Hides HNSW, tantivy, slot files, and concurrency model behind a stable async API.

```rust
trait Bucket: Send + Sync {
    fn id(&self) -> &BucketId;
    fn status(&self) -> BucketStatus;

    // --- query path ---

    fn dense_search<'a>(
        &'a self,
        query_vec: &'a [f32],
        top_k: usize,
    ) -> BoxFuture<'a, Result<Vec<Candidate>, BucketError>>;

    fn sparse_search<'a>(
        &'a self,
        query_text: &'a str,
        top_k: usize,
    ) -> BoxFuture<'a, Result<Vec<Candidate>, BucketError>>;

    fn fetch_chunk<'a>(
        &'a self,
        chunk_id: ChunkId,
    ) -> BoxFuture<'a, Result<Chunk, BucketError>>;

    // --- mutation path ---

    fn insert<'a>(
        &'a self,
        new_chunks: Vec<NewChunk>,
    ) -> BoxFuture<'a, Result<Vec<ChunkId>, BucketError>>;

    fn tombstone<'a>(
        &'a self,
        chunk_ids: Vec<ChunkId>,
    ) -> BoxFuture<'a, Result<(), BucketError>>;

    fn compact<'a>(&'a self) -> BoxFuture<'a, Result<(), BucketError>>;
}
```

Buckets with a path disabled in `bucket.toml` simply return empty candidate lists from the corresponding `*_search` method — the query path doesn't need per-bucket capability awareness.

`BucketStatus` covers the active-slot state: `Ready`, `Building { slot_id, progress }`, `Compacting { slot_id, progress }`, `NoActiveSlot`, `Failed { reason }`. Live mode and the UI both consume it.

`Candidate` carries the fields the reranker fusion needs:

```rust
struct Candidate {
    bucket_id: BucketId,
    chunk_id: ChunkId,
    chunk_text: String,        // mmap'd from chunks.bin; cheap to take
    source_score: f32,         // dense cosine or BM25 raw — provenance only
    path: SearchPath,          // Dense | Sparse — for dedup and observability
}
```

## Slot state machine

Buckets themselves don't carry state — they exist (with a directory) or they don't. **Slots** carry state:

```
                  rebuild trigger / config change
                              │
                              ▼
   ─────►  planning  ──►  building  ──►  ready  ──►  archived
                            │  ▲              │
                            │  │              │  active symlink
                            │  └────  failed ─┘  swap demotes
                            │                    previous active
                            ▼
                          failed
```

Multiple slots can be `ready` simultaneously — typical state during a model migration or after compaction is "new slot ready + old slot still active until promotion." Promotion is the `active` symlink swap; the demoted slot moves to `archived` after a retention period (default 7 days for stored buckets, immediate for pod-local linked buckets where rebuild is cheap).

Within `building`, the resumable sub-states live in `build.state`: `planning_done`, `chunking_done`, `embedding_progress(N/M)`, `sparse_indexing_done`, `final_dense_pass_done`. A restarted build reads `build.state` and picks up at the right sub-state.

`Compacting` is just a `Building` with the input source set to "current bucket state" — same state machine, different ingest.

## Configuration schemas

Two TOML files describe a bucket's persistent state: `bucket.toml` at the bucket directory root (mutable, user-editable) and `slots/<slot_id>/manifest.toml` per slot (frozen at slot creation, system-managed). Together with `build.state` (the resumability checkpoint, append-only structured log) they form the bucket's source of truth on disk.

### `bucket.toml` — bucket-level config

```toml
# <root>/knowledge/<bucket>/bucket.toml
#
# Bucket id is the directory name — never set here.
# Everything in this file is user-editable; changes take effect on the
# next slot build (existing slots reflect the config that was active when
# they were built, captured in their manifest).

# --- Identity (display only) ---

name        = "Wikipedia (English)"
description = "Full English wikipedia, main namespace, 2026-04 snapshot"
created_at  = "2026-04-25T10:23:11Z"          # informational

# --- Source: tagged enum on `kind` ---

[source]
kind         = "stored"                        # stored | linked | managed
adapter      = "mediawiki_xml"                 # source-adapter id; see [Source adapters]
archive_path = "source/enwiki-2026-04-15.xml.bz2"

# Adapter-specific knobs may follow as additional fields on [source]
# or in a [source.<adapter>] sub-table. Example for mediawiki_xml:
# include_redirects = false
# include_namespaces = [0]                     # main namespace only

# Linked variant (workspaces, notes directories):
# [source]
# kind                = "linked"
# adapter             = "markdown_dir"
# path                = "/home/me/projects/notes"
# include_globs       = ["**/*.md"]
# exclude_globs       = ["node_modules/**", ".git/**"]
# rescan_on_pod_start = true
# rescan_strategy     = "diff_by_hash"         # diff_by_hash | rebuild

# Managed variant (pod memory, agent-authored content — no external source):
# [source]
# kind = "managed"

# Tracked variant (self-maintained feed — system owns base + delta currency):
# [source]
# kind           = "tracked"
# driver         = "wikipedia"   # closed enum; one variant per supported feed family
# language       = "en"          # driver-specific knob
# delta_cadence  = "daily"       # daily | weekly | manual
# resync_cadence = "monthly"     # monthly | quarterly | manual
# # mirror = "https://dumps.wikimedia.org"   # optional override

# --- Chunker: strategy + params ---

[chunker]
strategy       = "token_based"                 # token_based | markdown_aware (future)
chunk_tokens   = 500
overlap_tokens = 50

# For markdown_aware (future):
# strategy              = "markdown_aware"
# split_at              = ["##", "###"]
# merge_short_threshold = 200

# --- Search paths: which retrieval modes the bucket exposes ---
# Sub-tables, not flat bools — TOML can't have `sparse = true` and
# `[search_paths.sparse] tokenizer = ...` simultaneously.

[search_paths.dense]
enabled = true

[search_paths.sparse]
enabled   = true
tokenizer = "default"                          # default | code | en_stem | raw

# --- Defaults applied when a new slot is built ---
# Slot-creation operations may override these; the chosen values get
# frozen into the new slot's manifest at build time.

[defaults]
embedder      = "tei_qwen3_embed_0_6b"         # provider name from server config
serving_mode  = "ram"                          # ram | disk
quantization  = "f32"                          # f32 | f16 | int8 (v1.5)

# --- Compaction policy ---

[compaction]
delta_ratio_threshold     = 0.20               # compact when delta > 20% of base
tombstone_ratio_threshold = 0.10               # or when tombstones > 10% of total
min_interval              = "6h"               # never compact more often than this
max_interval              = "30d"              # always compact at least this often
```

### `slots/<slot_id>/manifest.toml` — slot manifest

```toml
# <root>/knowledge/<bucket>/slots/<slot_id>/manifest.toml
#
# Written by the build pipeline, never edited by the user. Most fields
# are frozen at slot creation; [stats] and `state` update during the
# lifetime of the slot.

slot_id              = "01HVAB1234567890ABCDEFGHJK"   # ULID, immutable
state                = "ready"                         # planning | building | ready | failed | archived
created_at           = "2026-04-25T10:00:00Z"
build_started_at     = "2026-04-25T10:01:32Z"
build_completed_at   = "2026-04-29T03:14:00Z"          # null while still building

# --- Frozen chunker snapshot ---
# The exact chunker config used to produce this slot's chunks. Immutable
# after the chunking stage finishes. Mutations must use the same config —
# this is what guarantees chunk-id stability within the bucket.

[chunker_snapshot]
strategy       = "token_based"
chunk_tokens   = 500
overlap_tokens = 50

# --- Embedder used to build vectors ---
# Provider name binds to the current server config; model_id is what the
# embedder actually reported via `/info` at build time. If the provider
# is rebound to a different model later, queries fail an embedder-id check
# at search time rather than silently mixing dimensions.

[embedder]
provider  = "tei_qwen3_embed_0_6b"
model_id  = "Qwen/Qwen3-Embedding-0.6B"
dimension = 1024

# --- Sparse path snapshot (omitted if sparse disabled for this slot) ---

[sparse]
tokenizer = "default"

# --- Serving mode for this slot ---
# Can be flipped without rebuild — re-activating the slot in the new mode
# triggers a reload (RAM allocation vs mmap setup). Value here is the
# current setting; persistence on flip writes back here.

[serving]
mode         = "ram"                           # ram | disk
quantization = "f32"                           # f32 | f16 | int8

# --- Build statistics ---
# Updated as the build progresses; final values once state = ready.
# delta_chunk_count and tombstone_count grow during the slot's lifetime
# from insert / tombstone operations, until reset to 0 by compaction.

[stats]
source_records    = 6_843_217
chunk_count       = 13_487_213
vector_count      = 13_487_213
delta_chunk_count = 0
tombstone_count   = 0
disk_size_bytes   = 84_213_456_789
ram_size_bytes    = 56_789_123_456             # only meaningful in ram mode

# --- Compaction lineage (omitted for slots built from source) ---
# Set on slots produced by compaction so we can trace lineage and GC
# pre-compaction slots after retention.

[lineage]
prior_slot                  = "01HV9876ABCDEFGHJKLMNPQRST"
compacted_at                = "2026-05-15T02:30:00Z"
compaction_dropped_chunks   = 1_234_567
```

### Mutation rules

Which fields are user-editable, which are system-managed:

| File | User edits freely | System-managed |
|---|---|---|
| `bucket.toml` | All fields. Changes apply to *next* slot build (or future scan / compaction); existing slots are unaffected. | Nothing — this file is the user's. |
| Slot `manifest.toml` | Nothing. | All fields. `state`, `[stats]`, build timestamps update during the slot's life; everything else is frozen at creation. |
| `build.state` | Nothing. | Append-only structured log; the build worker reads it on resume. Not TOML — uses a binary or jsonl format optimized for sequential append. |
| `build.log` | Nothing — readable for debugging. | Append-only text log of build events: errors, retries, stage transitions, etc. |

Editing a slot manifest by hand isn't supported and may produce undefined behavior. Editing `bucket.toml` is supported and re-read on the next operation that consults it (slot build, scan, compaction).

## Build/buy: the vector store

Three honest options for the ANN + chunk + metadata storage:

Hybrid retrieval rules out any all-in-one solution: we need best-in-class for both dense ANN and BM25 sparse search, and no single library does both well. The realistic options:

**(A) hnsw_rs + tantivy + bincode + manifest.** Two best-in-class Rust-native libraries, glued by our own slot directory schema. Chunks + vectors + HNSW + tantivy index + manifest live alongside each other in the slot directory, sharing chunk ids. We own bucket layout, build orchestration, persistence — everything the [Build pipeline](#build-pipeline) already requires anyway. Both libraries support the access patterns we need (incremental append, mmap-friendly storage, fast search).

**(B) LanceDB + tantivy.** LanceDB for the dense path (its columnar storage + ANN), tantivy for sparse. LanceDB solves vectors+metadata storage but doesn't dodge any of the hard parts (build orchestration, model-change lifecycle, linked-source rescan are ours either way). Brings in a substantial foreign-type surface (Arrow). Doesn't naturally compose with tantivy at the bucket-directory level — they're two separate storage engines.

**(C) sqlite + sqlite-vec + FTS5.** Both indexes in one sqlite database. Transactionally consistent, single file per slot, concurrent reads. But: sqlite-vec is brute-force-with-tricks rather than HNSW, struggling past ~1M vectors per index; FTS5 is decent but slower than tantivy at scale. Awkward fit for wikipedia-scale.

**Recommendation: (A).** Reasoning:

1. **Hybrid pushes us toward composing best-in-class libraries** rather than adopting one all-in-one engine that does each thing okay.
2. The hard parts (build orchestration, model-change lifecycle, linked-source rescan, source adapters, slot rotation) are orthogonal to either index choice — we don't dodge them with (B) or (C).
3. For workspace-local pod buckets at thousand-chunk scale, (A) is trivially fast and (B)/(C) are overkill.
4. For wikipedia at ~50M chunks, hnsw_rs handles it directly; tantivy handles it directly. If we ever cross into territory where a single-process index starts hurting, the right move is **shard the bucket by topic** — same answer for any of the three options.
5. The bucket directory layout becomes the integration point. The rest of the system talks to a `Bucket` trait, not to hnsw_rs or tantivy directly, leaving the door open for a future engine swap if we need it.

We pay zero foreign-dep tax for unused features, and we get the index types we actually need.

## Bucket directory layout

Sketch (subject to revision as we work through the build pipeline):

```
<root>/knowledge/<bucket>/                # <root> is server_root or pod dir
  bucket.toml                             # bucket config: source, chunker, embedder, search_paths, ...
  source/                                 # for stored buckets only
    enwiki-2026-04-15.xml.bz2             # the original archive; never modified
  slots/
    01HVAB.../                            # one slot per (chunker, embedder) tuple
      manifest.toml                       # which chunker, which embedding model, dim, ...
      chunks.bin                          # base chunk text + metadata; immutable after build
      vectors.bin                         # base f32 vectors aligned to chunks.bin offsets
      index.hnsw                          # base HNSW graph (dense path)
      index.tantivy/                      # tantivy index (sparse path) — handles in/out natively
      delta_chunks.bin                    # appended chunks since last compaction
      delta_vectors.bin                   # appended vectors since last compaction
      delta_index.hnsw                    # delta HNSW for new chunks; merged with base at search
      tombstones.bin                      # sorted ChunkId list, mmap'd, binary-search filter
      build.state                         # work-queue + checkpoint for resumable builds
      build.log                           # append-only build event log for debugging
    active                                # symlink to the slot currently serving queries
```

Slot directories are immutable after build completion. A new chunker or embedding model produces a new slot; promotion is an `active` symlink swap. This makes both [model-change lifecycle](#build-slots) and rollback trivially cheap.

## Build pipeline

A wikipedia-scale embed run is a multi-day IO-bound stream against a TEI server. It must survive process restarts, hardware reboots, and TEI flapping. The pipeline:

1. **Plan** — walk the source, produce `(source_id, byte_range, content_hash)` records, write to `build.state`. Cheap relative to embedding (minutes for full wiki).
2. **Chunk** — for each source record, emit chunks with provenance. Chunks have stable ids derived from `(source_hash, chunk_offset)` so re-runs are deterministic and resumable.
3. **Embed (dense path)** — pull chunks off the queue, batch them, send to TEI, write `(chunk_id, vector)` to `vectors.bin`. Checkpoint after each batch.
4. **Index dense** — append each new vector to the HNSW graph as it arrives, or run a final-pass build at the end. Incremental append is preferred — keeps build memory bounded and makes the slot queryable on the dense path during the embed pass.
5. **Index sparse (BM25)** — feed each chunk's text into tantivy alongside (3). Tantivy indexing runs at hundreds of MB/s — far faster than embedding — so the sparse path finishes well before embedding does. The slot becomes queryable on the sparse path within hours, even when dense is still days away.

Resumability state lives in `build.state`. On startup the build task reads it and skips chunks that already have vectors written. Per-batch failure is captured in `build.log` and the chunk is re-queued.

**Throttling.** TEI server is shared with online query traffic. The build task pulls a shared `EmbeddingProvider` from the scheduler and respects a configurable rate limit (chunks/sec or tokens/sec). The provider's `RateLimited` error variant is honored — back off and retry.

**Partial-progress queries.** While a build is running, the slot is queryable as soon as the index has any nodes. The user's experience: a fresh wikipedia bucket is useful within an hour and just gets better over the following days. We mark the slot as `building` in its manifest until the embed queue is empty and the final HNSW pass is done; the UI surfaces this state.

## Build slots

When we change the embedding model (or chunker), we don't mutate the existing index. We:

1. Create a new slot directory with a fresh slot id.
2. Build it from the same `source/` archive (or, for linked, from a fresh rescan).
3. While building, queries continue to hit the old `active` slot.
4. When the new slot finishes, atomically swap the `active` symlink.
5. Optionally retain the old slot for N days as a rollback safety net, then GC.

This is the load-bearing reason the directory layout has versioned slots rather than one flat index. It also turns "try out a new embedding model" from a destructive action into a parallel one, which matters when builds take days.

## Incremental updates and compaction

Buckets aren't just built once and frozen. Pod memory is edited continuously by the LLM; wikipedia ships monthly delta dumps; arxiv trickles in new papers daily; linked workspace folders get edited. All of these converge to the same three operation primitives on the active slot, and a compaction mechanism that periodically folds accumulated changes back into a clean base.

### Three primitives

```rust
fn insert(new_chunks: Vec<NewChunk>) -> Result<Vec<ChunkId>>;
fn tombstone(chunk_ids: Vec<ChunkId>) -> Result<()>;
fn compact() -> Result<()>;
```

- **Insert**: embeds chunks, appends to `delta_chunks.bin` / `delta_vectors.bin`, inserts into `delta_index.hnsw` and into tantivy. Returns the assigned chunk ids when durable.
- **Tombstone**: appends ids to `tombstones.bin` (sorted, mmap'd) and submits delete-by-term to tantivy. Subsequent queries filter them out.
- **Update is `tombstone(old_id) + insert(new_chunk)`** — content change means new chunk id by construction (chunk id derives from content hash). No special-case "update" path.
- **Compact**: rebuilds the slot from current logical state (base + delta − tombstones), drops tombstoned chunks, produces a fresh slot directory; promotion is the standard `active` symlink swap.

### Query semantics with deltas

- **Dense path** searches both base HNSW and delta HNSW, merges candidates by score, filters tombstoned ids before returning. Slight overhead vs single-index search, but delta is small relative to base in steady state.
- **Sparse path** uses tantivy's native segment-based machinery. The reader handle reflects all committed segments (base + appended); deletes are honored automatically.

### Compaction mechanics

Compaction is **a new-slot build with the same chunker + embedder config**, with input source = "current bucket state" instead of "raw source archive." Same code path as initial build:

1. Enumerate current chunks (base ∪ delta − tombstones), with their existing chunk ids preserved.
2. Vectors for non-tombstoned chunks are copied from base/delta — no re-embedding needed.
3. New HNSW + tantivy indexes built from the surviving set.
4. New slot manifest written; `state = ready`.
5. Promotion swaps `active` symlink. Old slot retained for rollback (default 7 days for server buckets, immediate GC for pod-local).

Triggers:
- **Heuristic**: delta size > 20% of base, or tombstones > 10% of total chunks. Background scheduler watches for the threshold.
- **Manual**: admin via webui; LLM-callable for pod-local mutable buckets via `knowledge_modify`.
- **After bulk-rescan** of a linked bucket that produced significant deltas.

### Trigger sources for mutation

Three ways changes arrive at a bucket:

1. **Source rescan** for linked buckets. Walks the source, computes per-record hashes, diffs against the slot's known state, produces inserts (new/changed records) and tombstones (removed records). Triggered manually, on pod start (config flag), or by an external event. Pure rebuild from scratch is fine for small workspace buckets where it's faster than diff-walk.
2. **External event notification** for stored buckets that update. A wikipedia delta-dump arrives → an admin task ingests it as inserts + tombstones. Arxiv RSS fires → new-paper inserts. The notification mechanism is out of scope for v1; for now these run as scheduled jobs or admin actions.
3. **LLM-driven via `knowledge_modify` tool**. For pod memory specifically: a builtin tool exposing `insert(text, source_ref)` / `tombstone(chunk_id)` against the pod's memory bucket. The LLM uses this to record durable notes, update existing entries, and clean out stale ones. Gated by the pod's `[allow.tools]` table the same way other tools are.
4. **Tracked-bucket scheduler** for `tracked` buckets. A per-bucket worker wakes on the configured `delta_cadence`, asks the driver for new deltas since `last_applied_delta_id`, downloads each, parses through the same source adapter the bucket uses for its base, and applies inserts / tombstones via the same mutation API. Background monthly resync (configurable via `resync_cadence`) builds a fresh slot off the latest base snapshot and rotates when ready. See [Tracked sources](#tracked-sources-self-maintained-feeds).

All four converge to the same `Bucket::insert` / `Bucket::tombstone` API. They differ only in initiator and frequency.

### Why no file-watching for linked buckets

Active filesystem watching (inotify) was considered and rejected as too much machinery for unclear benefit. The realistic shape is:
- Pod memory is changed by the LLM via the `knowledge_modify` tool, which directly invokes the API — no need to detect file changes after the fact.
- Linked workspace buckets are rescanned explicitly when meaningful (pod start, admin trigger, after the agent has done significant edits).
- Wikipedia / arxiv / paper buckets are ingested via event notification when delta sources publish.

If file-watching turns out to be necessary later, it slots in as another notification source feeding the same `insert` / `tombstone` API; nothing in the bucket model has to change.

## Concurrency model: queries vs builds

Indexing CPU lives off the main runtime ([Runtime model](#runtime-model-where-cpu-runs)) — but the live-mode latency budget assumes queries get **direct shared read access** to the active slot's indexes. Queries are not message-passed through the indexing thread. Reasoning:

- Message-passing through one OS thread serializes retrieval, queueing queries behind HNSW inserts and tantivy commits. With c-srv3's many cores, this would be a deliberate waste.
- Sub-ms RAM-resident search is fast enough that the synchronization overhead of shared access (RwLock acquire, refcount increment) is itself the dominant cost, not the contention.
- Library-level concurrency contracts already give us safe shared access for tantivy, and a thin RwLock wrapper for HNSW.

### Ownership table

| Operation | Owner | Mechanism |
|---|---|---|
| Build a new slot | dedicated indexing thread for that build | new `SlotState` lives off-registry until promotion |
| Mutate active slot (insert / tombstone) | per-bucket mutation worker thread | tantivy `IndexWriter` (single-owner); HNSW `RwLock` write side |
| Query active slot | any tokio worker | shared `Arc<SlotState>`; tantivy `IndexReader`; HNSW `RwLock` read side |
| Slot promotion | scheduler | atomic swap of registry's inner `Arc<SlotState>` |
| TEI `embed`/`rerank` | any tokio worker | `Arc<dyn EmbeddingProvider>`; reqwest connection pool |

### Library-level guarantees we rely on

- **Tantivy** has a native multi-reader + single-writer model. Segments are immutable; the reader holds a snapshot of committed segments at the moment it opened. Writers append new segments and atomically update the segment list. Concurrent reads see a consistent snapshot; commits are visible after a `reader.reload()`.
- **`hnsw_rs`** doesn't promise concurrent insert/search safety. We wrap the index handle in `parking_lot::RwLock`. Searches take read-locks (concurrent across tokio workers); inserts take write-locks in batches. With sub-ms search, brief write-lock holds during insert batches don't meaningfully impact read traffic. If measurements ever show contention hurting, the alternatives are a lock-free HNSW library or a write-batched flush (queue inserts, apply during quiescent windows).

### Slot rotation under concurrent queries

The registry holds `RwLock<Arc<SlotState>>` per bucket (or an atomic `Arc` swap primitive — either works). Promotion replaces the inner Arc:

```rust
// scheduler-side, on slot promotion:
let mut active = bucket.active.write();
let old = std::mem::replace(&mut *active, Arc::new(new_slot));
drop(active);   // release the registry write lock immediately
drop(old);      // refcount may not be zero — drops when last in-flight query completes
```

Queries grab their own `Arc<SlotState>` clone before searching, so an in-flight query that captured the old state finishes against it without holding any locks. The old slot's RAM-resident data frees once the last query Arc is dropped. New queries see the new slot.

This works identically in both serving modes — RAM-resident `Vec<f32>` and mmap'd files both ride along on the `Arc<SlotState>` lifetime.

### Multiple TEI clients

The provider traits are already `Arc<dyn ...>` and Send + Sync. Multiple async tasks call `embed()` and `rerank()` concurrently with no special handling — `reqwest::Client` underneath handles HTTP keep-alive and connection pooling. The build pipeline and the live-query path each grab their own Arc to the provider; they don't share connection state, just the upstream TEI server.

Throttling is per-provider rather than global: an embedding provider can be configured with a max concurrent in-flight requests; both build and query paths get fair scheduling against that pool. Build traffic respects `RateLimited` errors (back off + retry); query traffic surfaces them as a transient bucket error.

## Source adapters

A small `SourceAdapter` trait that produces the layer-1 → layer-2 transformation:

```rust
trait SourceAdapter {
    fn enumerate(&self) -> BoxStream<SourceRecord>;
    fn read(&self, record: &SourceRecord) -> BoxFuture<Result<String>>;
}
```

Initial adapters:
- `MarkdownDir` — directory of `.md` files, source records are file paths. The first thing we'll need for workspace buckets.
- `MediaWikiXml` — wikipedia XML dumps, source records are articles. Comes second when we start the wiki bucket.
- `Arxiv` — arxiv metadata + PDF extraction. Adds a PDF-extract dependency (probably `pdf-extract` or `lopdf`); needs investigation.

Adapters are pluggable per-bucket via `bucket.toml`; new source types don't change the rest of the pipeline.

## Tracked sources: self-maintained feeds

The fourth source kind. Where `stored` is "user hands us a frozen archive URL" and `linked` is "user hands us a path that may move," `tracked` is "user names a *feed family* (Wikipedia today; future Wikidata, arxiv-feed) and the system handles acquisition + currency." The system maintains a base snapshot + applied-delta cursor on disk, polling the feed on a configured cadence.

The data layer is identical to `stored`: same `MediaWikiXml` parser, same chunker, same slot lifecycle. The only addition is the **acquisition layer** — the `FeedDriver` trait — and the per-bucket scheduler that exercises it.

### The FeedDriver trait

A closed enum of supported feeds. Each variant knows the URL conventions, version layout, and delta semantics of one source family:

```rust
trait FeedDriver: Send + Sync {
    /// Latest available base snapshot id (e.g. "20260401" for Wikipedia).
    async fn latest_base(&self) -> Result<SnapshotId, FeedError>;

    /// Download the base snapshot to `dest`. Idempotent: if `dest` already
    /// exists with the right size + checksum, no-op.
    async fn fetch_base(&self, id: &SnapshotId, dest: &Path) -> Result<(), FeedError>;

    /// List delta ids posted after `since`, sorted ascending by timestamp.
    /// `None` means "list everything since the base snapshot."
    async fn list_deltas_since(&self, since: Option<&DeltaId>) -> Result<Vec<DeltaId>, FeedError>;

    /// Download one delta to `dest`. Idempotent.
    async fn fetch_delta(&self, id: &DeltaId, dest: &Path) -> Result<(), FeedError>;

    /// Source-adapter id used to parse this driver's downloads (e.g.
    /// `"mediawiki_xml"`). Acquisition and parsing are orthogonal: the same
    /// MediaWiki XML parser handles both stored-bucket archives and
    /// tracked-bucket downloads.
    fn parse_adapter(&self) -> &'static str;
}
```

Closed enum dispatch:

```rust
enum FeedDriverImpl {
    Wikipedia(WikipediaDriver),
}
```

Adding `Wikidata` later means a new variant + a new file; no config-driven plugin loading.

### WikipediaDriver

Wraps `https://dumps.wikimedia.org` URL conventions:

- **Base** — `/<lang>wiki/<YYYYMMDD>/<lang>wiki-<YYYYMMDD>-pages-articles-multistream.xml.bz2` — monthly cadence, ~24 GB compressed for enwiki.
- **Deltas** — `/other/incr/<lang>wiki/<YYYYMMDD>/<lang>wiki-<YYYYMMDD>-pages-meta-hist-incr.xml.bz2` — daily cadence with ~12h posting latency, ~810 MB/day for enwiki.

The incremental dump service self-describes as *"experimental — at any time it may not be working for a day, a week, or a month."* The driver retries with exponential backoff and surfaces `last_error` to the bucket UI; if the delta service is down for an extended period, the next monthly resync provides correctness reset.

Both base and delta files are MediaWiki XML — same format the existing `MediaWikiXml` source adapter already streams. The driver's only contribution beyond URL building is HTTP fetch with checksum verification (md5sums.txt is published alongside each dump) and retry/backoff policy.

### On-disk layout for tracked buckets

```
<root>/knowledge/<bucket>/
  bucket.toml
  source-cache/                       # tracked-bucket only
    base/
      20260401/
        enwiki-20260401-pages-articles-multistream.xml.bz2
    deltas/
      20260402/enwiki-20260402-pages-meta-hist-incr.xml.bz2
      20260403/...
  feed-state.toml                     # tracked-bucket only
  slots/
    01HVAB.../
    active                            # symlink to current slot
```

`source-cache/` is **bucket-scoped**, not slot-scoped — a slot rebuild reuses the cached base download. This matters both for monthly resync (no re-download of the same base) and for reproducible re-builds during development.

`feed-state.toml` carries the per-bucket cursor — small, frequently rewritten:

```toml
current_base_snapshot_id = "20260401"
last_applied_delta_id    = "20260424"
last_check_at            = "2026-04-26T11:30:00Z"
last_check_outcome       = "ok"             # ok | retry | error
last_error               = ""
```

Retention on `source-cache/deltas/`: keep N days post-application (default 30; ~25 GB for enwiki) for debug + reproducibility. Configurable per bucket; never auto-deleted while still pending application.

### Update flow: applying daily deltas

A scheduler-owned per-bucket `feed_worker` runs on the configured `delta_cadence`:

1. Wake (cadence trigger or manual button).
2. `driver.list_deltas_since(last_applied_delta_id)` → ordered list.
3. For each delta in order:
   - `driver.fetch_delta(...)` → `source-cache/deltas/<id>/`
   - Parse via the bucket's source adapter.
   - For each `<page>` in the delta XML: emit `tombstone(old_chunk_ids_for_page)` + `insert(new_chunks)` through the existing mutation API.
   - On success, update `feed-state.toml.last_applied_delta_id` (append-then-rename for crash safety).
4. Compaction triggers fire normally off the existing 20%-delta / 10%-tombstone heuristics in `bucket.toml`.

Crash recovery: on worker startup, re-read `last_applied_delta_id` and resume from the next id. Because delta application produces fresh `chunk_id`s by content hash, re-running a partially-applied delta is idempotent (insert dedupes by chunk_id; tombstones are sorted-set inserts).

### Background monthly resync

A separate schedule (configurable via `resync_cadence`, default `monthly`) builds a fresh slot off the latest base snapshot:

1. `driver.latest_base()` → if unchanged from `current_base_snapshot_id`, no-op.
2. `driver.fetch_base(...)` → `source-cache/base/<id>/`.
3. Plan a new slot in `building` state — same code path as initial build, input is the new base file.
4. While building, the active slot continues serving + applying daily deltas.
5. New slot reaches `ready` → catch up on deltas accumulated during the build (driver lists deltas since the new base's snapshot date, applies them via the same mutation flow).
6. Slot rotation promotes the new slot; old slot archived per the existing 7-day retention.

Default-on; per-bucket toggle in `bucket.toml` (`resync_cadence = "manual"` disables) and an explicit "Resync now" button in the UI for ad-hoc triggering.

Resync is the correctness reset: silent log/oversight events that don't surface cleanly in daily incrementals get caught by re-ingesting from the authoritative monthly base.

### Turnkey UI

A "Create tracked bucket" affordance in the WebUI takes a small form per driver. For Wikipedia:

- Language code (default `en`).
- Optional mirror override.
- Optional delta / resync cadence overrides (defaults are sane).
- Estimated size + build time displayed prominently — an enwiki initial build is ~24 GB download + ~4 days continuous embed on Qwen3-Embedding-0.6B; all-wiki is ~13 days.

The system writes `bucket.toml`, places the bucket in the registry, and queues the initial base download + build automatically. Hand-authored toml stays supported for power users / scripted setups.

### Implementation prerequisites

Two existing dangling-cleanup items become load-bearing once the WebUI exposes a one-click Wikipedia bucket:

- **Resumable builds** (chunking + embedding checkpoint in `build.state`). An enwiki initial build is multi-day; a crash mid-build that forces starting over is unacceptable for a turnkey button. Already in scope per `progress_knowledge_db.md`.
- **Real BPE tokenizer integration** (`tokenizers` crate) for `TokenBasedChunker`. The current `chars_per_token=4` heuristic is fine for stress-testing but would burn chunks on a "production" Wikipedia bucket. Already flagged in the progress doc as "Land before any production bucket gets built."

Both can be sequenced before the user-clickable UI lands without slowing the data-model side of tracked sources (the toml schema, `FeedDriver` trait, `WikipediaDriver` URL/listing logic, on-disk layout) — those land independently and are testable without a real ingest.

## Query path

```rust
async fn query(
    bucket_ids: &[BucketId],
    text: &str,
    top_k: usize,
) -> Vec<(score, chunk_text, source_ref)> {
    // Embed once, share across all buckets (dense path).
    let q_vec = embedding_provider.embed(&[text]).await?;

    // Fan out across buckets and paths in parallel.
    let candidates: Vec<Candidate> = bucket_ids.par_iter()
        .flat_map(|id| {
            let bucket = bucket(id);
            let dense  = bucket.dense_search(&q_vec, top_k_dense);
            let sparse = bucket.sparse_search(text, top_k_sparse);
            dense.into_iter().chain(sparse).map(|c| c.tag(id))
        })
        .collect();

    // Deduplicate (a chunk may hit on both paths in one bucket).
    let unique = dedupe_by_chunk_id(candidates);

    // Single batched rerank across the entire union.
    let reranked = rerank_provider
        .rerank(text, &unique, top_k)
        .await?;

    reranked
}
```

Per-bucket over-fetch factors `top_k_dense` and `top_k_sparse` are tuned for reranker breadth — typically 16–32 each. Over `top_k * 4` total per bucket diminishing-returns hard.

The embedding and reranker providers are bucket-independent — they're scheduler-level providers configured via `[embedding_providers.*]` / `[rerank_providers.*]` (already landed). Each bucket knows its own embedding model id (recorded in slot manifest) and refuses queries whose embedding vector dimension doesn't match. Sparse path takes raw query text and uses the bucket's tantivy tokenizer (configured per-bucket; defaults work for most natural-language content).

Buckets with only one search path enabled (per `bucket.toml` `[search_paths]`) simply skip the missing path — query path still works.

## Open questions

Tracked here as we work through the design. Resolved questions move to the [Decisions log](#decisions-log).

- **Chunking strategy.** Token-based with overlap is the obvious starting point, but markdown-structure-aware chunking (split on `##` headers, merge short sections) tends to give better retrieval for documentation-style sources. Per-bucket configurable in `bucket.toml`. Will need playground experimentation — user noted explicitly we'll iterate on this.
- **Tantivy tokenizer / language config per bucket.** Tantivy's defaults work for English natural-language content. Code-heavy buckets may want different tokenization (preserve identifiers, no stemming); multilingual content may want per-language analyzers. Probably surface as `[search_paths.sparse]` config in `bucket.toml`.
- **Per-path over-fetch tuning (`top_k_dense`, `top_k_sparse`).** How many candidates each retrieval path produces before reranking. Starting estimate ~16–32 each per bucket. Worth empirical tuning once we have the reranker in the loop on real queries.
- **Reranker batch sizing.** A multi-bucket query produces a candidate union potentially in the hundreds. Reranker latency scales with batch size. Cap at ~64 total? Configurable per query? Worth a measurement pass before nailing down.
- **Multi-bucket query API.** Does `knowledge_query` take a list of bucket ids, or a tag-set? Current sketch is bucket ids; tags would let pods declare "search anything tagged `wikipedia`" without enumerating.
- **Build trigger for stored buckets.** Admin-only via webui? An admin-only `knowledge_admin` tool callable from a privileged pod? Leaning admin-only for stored bucket *initial builds* and *compactions*; mutations on pod-local memory buckets are LLM-callable per the [Incremental updates](#incremental-updates-and-compaction) section.
- **Query synthesis for the post-turn nudge.** What text do we embed? The LLM's thinking block (probably best — that's where the actual work happened, before output formatting), the assistant message, the user message, or a concat? Worth playground experimentation.
- **Relevance threshold tuning.** What score gates the nudge? Likely a per-bucket calibrated threshold on reranker score (the only credibly-comparable signal across buckets / paths). Probably starts as a config knob, defaults dialed in by observation.
- **Nudge contents.** Pure pointer ("chunk 47 in Apollo program, score 0.84"), pointer + short snippet, or full chunk text? The first is cheapest and matches the user's stated intent; the third saves a tool round-trip but bloats context every time. Worth deciding before wiring.
- **Multi-bucket nudge format.** If two buckets each produce a hit above threshold, do we get two nudges back-to-back, one merged nudge, or top-1 across buckets? Leaning top-1 across buckets for simplicity.
- **Where the post-turn hook lives in the thread state machine.** Probably a new transition: when an assistant turn enters its terminal state, fire the retrieval task. Implementation depth tracked when we get to it.
- **Compaction trigger thresholds.** Heuristic placeholders are 20% delta-vs-base size and 10% tombstones. Real values need observation on actual mutating buckets; should probably also have time-based triggers (compact pod memory at least daily).
- **`knowledge_modify` tool schema.** What's the LLM's surface for inserting/tombstoning chunks in a pod-local mutable bucket? Per-tool schema design, gated by `[allow.tools]`. Specifying chunk content vs. specifying a source reference for the system to chunk are two reasonable shapes.
- **Pod memory as a default bucket.** Do new pods automatically get a `pod:memory` bucket, or is it opt-in per pod? Default-create with the option to disable seems most useful, but worth a deliberate call.
- **Mutation propagation latency.** How fast do `insert`/`tombstone` show up in queries? Tantivy commits are batchable for throughput; HNSW append is immediate-ish. Need a story for "I just told the agent to remember X — when can I query for it?" Likely sub-second for both, but worth measuring.
- **Concurrent index writes.** Can a slot accept new chunks (from a partial rescan) while serving queries? HNSW supports append; we just need the lock discipline figured out. Defer until rescan path is being implemented.
- **Quantization.** f32 → f16 is a low-risk halving of vector storage; int8 calibrated quantization is a 4× win but requires a calibration pass. Bigger impact on disk-backed serving (doubles the working set that fits in page cache) than RAM-resident. Defer to v1.5; design `vectors.bin` so the precision is part of the manifest, not baked into the file format.
- **GPU vs CPU TEI.** Out of scope — that's a deployment decision, not an architecture one. The provider trait doesn't care.
- **Resource registry integration.** Buckets are scheduler-level resources analogous to Backend / McpHost. They probably want a `Bucket` entry in the resource registry once there's a consumer (the `knowledge_query` tool). Defer until that consumer exists.
