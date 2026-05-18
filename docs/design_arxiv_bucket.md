# Arxiv Tracked Bucket

A `tracked`-kind knowledge bucket fed by arxiv's bulk source archive. Builds on the `FeedDriver` machinery introduced for wikipedia; this doc covers the arxiv-specific decisions: format strategy, acquisition cost trade-offs, source-bundle parsing, and storage layout.

Companion doc: [`design_knowledge_db.md`](design_knowledge_db.md) — the bucket / slot / FeedDriver architecture this slots into.

**Status:** Design in progress. Manifest pulled, one tarball sampled, and the filter pipeline prototyped end-to-end (2026-05-15). Stage-1 results inline below; see [`tools/arxiv_filter/`](../tools/arxiv_filter/) for the filter tool. EC2 stage-2 test and full base pull not yet run.

## Decisions log

| Date       | Decision                                                                                          | Notes |
|------------|---------------------------------------------------------------------------------------------------|-------|
| 2026-05-15 | **LaTeX-source-only ingest. No PDF extraction path in v1.** Papers with no source bundle (~8%) are skipped; tracked in a manifest so we can reconsider later. | Pandoc on source produces dramatically cleaner text than any PDF extractor we'd plausibly use. 8% coverage gap is acceptable, weighted toward older / non-mathy fields. |
| 2026-05-15 | **Pandoc subprocess as the tex→text parser.** No in-process Rust LaTeX parser. | Pandoc handles `\input`-resolution, macro expansion, math fallback to verbatim TeX, and is robust across LaTeX dialects. Subprocess overhead is irrelevant at our throughput — embedding is the bottleneck. |
| 2026-05-15 | **`00README.json`-driven entry point selection.** Every source bundle carries a JSON manifest naming the toplevel tex; the adapter respects it. | Eliminates the "guess the main tex" problem. Universally present in modern bundles (every bundle sampled had one). |
| 2026-05-15 | **EC2 filter-in-region is the default base-pull strategy.** Spin up an instance in `us-east-1`, stream tarballs from S3 (free intra-region), extract tex-only on-instance, egress ~130 GB of compressed text. | Cuts egress cost from ~$593 (full bucket) to ~$35 (compressed tex only). Operational complexity is bounded — one-shot script, not a long-lived service. See [Acquisition cost options](#acquisition-cost-options). |
| 2026-05-15 | **Tarball-atomic egress.** We download whole monthly tarballs even though we only want a subset of file types inside. | S3 has no sub-object selection. Once we accept the EC2-filter approach this is invisible — figures and PDFs never leave us-east-1. |
| 2026-05-15 | **Monthly deltas pulled direct (no EC2).** New tarballs each month run ~30 GB, ~$3 egress at full retention. | EC2 spin-up overhead doesn't pay back for that volume. Reserve the EC2 path for the one-time base pull. |
| 2026-05-15 | **Metadata acquisition via OAI-PMH, not bundle README.** Bundle README.json carries tex-compile metadata, not category / abstract / authors — those need the (free) metadata API. | Two acquisition paths but cleanly separated. OAI-PMH gives full archive bibliographic metadata at no cost; sets us up for category-filtered retrieval downstream. |
| 2026-05-15 | **Source kind = `tracked` with new `FeedDriver::Arxiv` variant.** Same shape as wikipedia: closed enum, `fetch_base` / `list_deltas_since` / `fetch_delta` interface. | The bulk source bucket is naturally tarball-incremental — papers post-base appear as new tarballs, never modifications to old ones. Maps cleanly to the existing tracked-bucket cursor model. |

## Goals

- A standing arxiv bucket queryable alongside wikipedia, suitable for code, math, and ML retrieval.
- Source-first content: LaTeX rather than PDF text extraction.
- Coverage scope decision (full archive vs. subset) **driven by measured cost data**, not a priori guesses.
- Maintained currency: monthly delta cadence via the same tracked-bucket scheduler as wikipedia.
- Reproducible build: cached source archive lives under our control on ceph; slot rebuilds reuse it without re-paying S3 egress.

Non-goals for v1:
- PDF extraction fallback (the ~8% PDF-only papers are skipped).
- Figure / image embeddings (figures discarded at extract time in the cost-optimized path).
- Version history (we ingest only the latest version of each paper).
- Cross-paper citation graph traversal (interesting but separable; not in v1).

## Measured baseline

Numbers below come from the actual manifest (2026-05-07 snapshot) and one sampled tarball (`src/arXiv_src_2604_001.tar`, 152 papers).

### Manifest totals

| Metric | Value |
|---|---|
| Total tarballs | 12,374 |
| Total bucket size | **6.59 TB** compressed |
| Total papers | 3,028,007 |
| Avg paper bundle (compressed, with figures) | 2.1 MB |
| Tarball avg | 533 MB |

### Per-year size trajectory

Modern papers are dramatically larger than older ones — driven by higher-res figures and supplementary material, not by tex content:

| Year | Papers | Tarball GB | KB/paper avg |
|---|---|---|---|
| 2010 | 70,125 | 41 | 570 |
| 2015 | 105,280 | 114 | 1,057 |
| 2020 | 178,329 | 446 | 2,443 |
| 2024 | 244,031 | 1,087 | 4,349 |
| 2025 | 284,486 | 1,165 | 4,000 |

### Sample tarball composition (152 papers, 582 MB)

Coverage:
- 140 (92.1%) have a source bundle (`.gz`).
- 12 (7.9%) are PDF-only — no LaTeX available.
- 100% of source bundles contain at least one `.tex` file.

Source bundle composition (140 papers, 581 MB uncompressed):

| Category | Size | Fraction |
|---|---|---|
| tex / bib / bbl / cls / sty / bst | 31.9 MB | **5.5%** |
| Figures (pdf / png / jpg / eps / ...) | 545.3 MB | **93.9%** |
| Other (README / aux) | 3.7 MB | 0.6% |

Per-paper tex content (raw): median 135 KB, mean 222 KB, p90 420 KB, max 4.2 MB.

Compression: bundles barely shrink under gzip (504 MB → 581 MB, 14% expansion) because contents are already-compressed PDFs/PNGs. Pure tex content compresses well on its own (~3–4× with zstd).

### Extrapolated full-archive sizes

| Slice | Raw tex extracted | zstd-compressed tex |
|---|---|---|
| Full bucket (3M papers) | ~360 GB | ~100–130 GB |
| 2020+ (1.2M papers) | ~270 GB | ~75 GB |
| 2024+ (640k papers) | ~140 GB | ~40 GB |

Even the full-archive tex content is ~130 GB compressed — well under the ceph 5 TB budget.

## Acquisition cost options

Egress is the dominant cost. arxiv's source bucket is requester-pays at $0.09/GB to the public internet. Storage budget on ceph is ~5 TB. There are three coherent strategies:

### Option A — Bulk download raw

Download all 6.59 TB of source tarballs over the internet, archive everything to ceph, filter to tex at ingest time.

| Component | Cost |
|---|---|
| S3 egress (6.59 TB × $0.09) | **~$593** |
| S3 GET requests (12,374 × $0.0004/1000) | <$1 |
| **Total** | **~$594** |

| What we get | What we lose |
|---|---|
| Full raw archive on ceph (6.6 TB), including figures | Doesn't fit comfortably in 5 TB budget |
| Future-proof: can re-derive anything later | Slow download (days at residential bandwidth) |
| No additional infrastructure | |

### Option B — EC2 filter-in-region, tex-only egress

Spin up an EC2 instance in `us-east-1`, stream tarballs from S3 (intra-region traffic is **free**), extract tex content on the instance, compress, and egress only the slim output. Same bytes are read from S3 — but most of them never leave AWS, so we don't pay for them.

| Component | Cost |
|---|---|
| S3-to-EC2 intra-region transfer (6.59 TB) | **$0** |
| S3 GET requests (12,374 × $0.0004/1000) | <$1 |
| EC2 compute (c6i.4xlarge spot, ~24 h) | ~$8 |
| EC2 GP3 storage (100 GB × few days) | <$1 |
| Egress of compressed tex output (~130 GB × $0.09) | ~$12 |
| **Total** | **~$22** |

| What we get | What we lose |
|---|---|
| ~130 GB compressed tex on ceph (or ~360 GB raw) | Figures gone forever (would have to re-pay egress to recover) |
| Fits trivially in 5 TB | Mildly more operational complexity (an EC2 run) |
| ~25× cheaper than Option A | Need the filter script to be correct end-to-end before destroying the EC2 |

### Option C — EC2 filter + selective raw retention

Same as Option B, but also egress raw tarballs for recent years (e.g. 2024+, ~1.5 TB) so we keep figures for modern papers — the slice most likely to matter for future work like vision embeddings or citation extraction.

| Component | Cost |
|---|---|
| Intra-region transfer | $0 |
| EC2 compute | ~$8 |
| Tex egress (~130 GB) | ~$12 |
| Raw tarball egress (1.5 TB 2024+) | ~$135 |
| **Total** | **~$155** |

| What we get | What we lose |
|---|---|
| Tex archive (full coverage) + raw recent-year archive | Pre-2024 raw is gone unless re-paid |
| Comfortable 5 TB fit (~1.6 TB used) | Picks a year cutoff somewhat arbitrarily |
| Hedges future "wish we had the figures" risk for the slice most likely to need them | |

### Recommendation

**Option B**, with Option C available as a graceful upgrade if we decide we want figures later.

Rationale:
- The 25× cost saving is real and meaningful.
- LaTeX → text quality is already the upper bound of what we'd get from figures anyway, for the embedding use case.
- We can always pull a specific year's raw tarballs later for ~$60–135 if we change our mind. The choice isn't permanent.
- Operational complexity is one EC2 run, not an ongoing service.

The first concrete commitment is small: do a **scaled test** first (one year's worth of tarballs, ~$1–3 of EC2 compute, $0.10 of egress) to validate the filter script and pandoc throughput end-to-end before committing to the full archive run.

## EC2 filter pipeline (Option B operational shape)

The base pull runs as a one-shot script, not a service. Resumable so we can stop and restart if anything fails partway. Approximate shape:

```
for each tarball in manifest:
  if already-processed-marker present: skip
  download tarball from S3 (intra-region)
  extract to tmpfs
  for each paper bundle (.gz) inside:
    gunzip → tar entries
    read 00README.json to find toplevel tex
    filter to text files only (.tex, .bib, .bbl, .cls, .sty, .bst, 00README.json)
    pandoc toplevel.tex → plain text
    archive: (paper_id, source_files_tar.zst, plaintext.zst)
  write processed-marker, sync output batch to remote storage
  delete local tarball + bundle scratch
```

Output bundled into per-yymm `.tar.zst` files for easy transfer and on-ceph archival.

**Throughput (measured stage 1, 24-core local box, 152-paper tarball):**

- 17.1 s wall-clock for the full tarball.
- **8.9 papers/s** at 24-way parallelism.
- Pandoc timeout 15 s per paper (longer is wasted — hangs never recover).

Extrapolation to EC2:

- c6i.8xlarge (32 vCPU): ~12 papers/s, full archive in ~70 hours.
- c6i.16xlarge (64 vCPU): ~24 papers/s, full archive in ~35 hours.
- c6i.32xlarge (128 vCPU): ~47 papers/s, full archive in ~18 hours.

Spot pricing keeps total compute cost in the **$10–20 range** regardless of
which instance — bigger = shorter wall-clock = less spot-interruption window.
c6i.16xlarge is the natural pick.

**Failure modes:**

- Spot interruption mid-tarball: resumable from last processed-marker.
- Pandoc hangs on a malformed bundle: per-paper timeout (~30 s), log and skip.
- S3 transient error: retry with backoff.
- Disk pressure: tmpfs spool sized for one tarball at a time (~1 GB).

**Validation before committing to full run:**

- ✅ Stage 1 (local prototype): script run on one real tarball (`2604_001`); 90.7% pandoc success on source-available papers; 88× compression ratio. Done 2026-05-15.
- ✅ Stage 2 (scaled test, run locally to save EC2 setup): driver + filter pipeline run end-to-end against 12 tarballs (1,706 papers, 6.02 GB) pulled live from requester-pays S3, filtered locally, uploaded to staging bucket `christian-arxiv-staging`. Resume logic verified. 88.6% pandoc success at scale, 75× compression. Done 2026-05-15.
- ✅ Stage 3 (full archive on c6i.16xlarge spot): 12,374 tarballs processed in **18h 53min** wall clock. Output **108.4 GB** across 12,374 `.tar.zst` files = **60.7× compression** vs. 6.59 TB source. Total cost **~$10 EC2 + ~$10 egress ≈ $20**. Done 2026-05-17 09:03 UTC; output on ceph at `/ceph/christian/LTS/arxiv/base/20260517/`. Two issues surfaced and fixed mid-run: (a) pre-2000 papers ship as raw gzipped LaTeX (no tar wrapper); (b) sequential tarball processing left 56% CPU idle. Patched with single-file-bundle handling and 16-way shard parallelism (`--shard N/M` flag).

### Measured status distribution

Stage 2 (1,706 papers across 12 tarballs, April 2026) is the more reliable sample:

| Status | Count | Fraction | Notes |
|---|---|---|---|
| ok | 1,420 | 83.2% | Plaintext extracted cleanly |
| pandoc_failed | 124 | 7.3% | Custom class commands, pandoc parser quirks |
| pdf_only | 104 | 6.1% | No source; recorded for tracking, no plaintext |
| timeout | 55 | 3.2% | Macro recursion; killed at 15 s |
| other | 3 | 0.2% | no_toplevel / untar_failed / empty_output |

**Pandoc success on source-available papers: 1,420 / 1,602 = 88.6%.**
Total plaintext yield: 83.2% of all papers.

Compression: 6.02 GB input → 80.6 MB filtered output (75× reduction).
Extrapolated full-archive output: ~88 GB compressed.

Plaintext per successful paper: median 47k chars, mean 58k, p90 102k —
typical mid-sized paper's worth of body text.

### Throughput observed

Stage 2 ran on a 24-core local box pulling from S3 over the internet:

| Phase | Average per tarball |
|---|---|
| S3 download | 10.4 s |
| Filter (24 cores) | 28.4 s |
| S3 upload | 1.8 s |
| Total wall | ~40 s |

Filter-alone throughput: 7.15 papers/s on 24 cores. The download dominates
per-tarball time at home-bandwidth; on EC2 intra-region the download time
collapses (multi-Gbps S3 throughput within `us-east-1`).

## The Arxiv source adapter

Same `SourceAdapter` trait as `MarkdownDir` and `MediaWikiXml` — translates archived files into `SourceRecord` instances for the chunker.

Per-paper input (after the EC2 filter):
```
<archive>/<yymm>/<paper_id>/
  plaintext.zst           # pandoc output, primary input for embedding
  source.tar.zst          # filtered tex/bib/etc, in case we ever want to reparse
```

Per-paper `SourceRecord`:
- `id`: arxiv paper id (e.g. `2604.00001`)
- `title`: from OAI-PMH metadata (separate acquisition)
- `categories`: list, from OAI-PMH metadata
- `abstract`: from OAI-PMH metadata
- `text`: plaintext from pandoc
- `source_ref`: `arxiv://<paper_id>` (resolves to `https://arxiv.org/abs/<paper_id>`)

Citations and math handling:
- Math: pandoc leaves unconvertible math envs as verbatim LaTeX strings inside the prose. Acceptable — embeddings can handle this, even if imperfectly.
- Citations: `\cite{}` tokens drop out as gaps. For v1 we accept this; a future pass could resolve `.bbl` files to inline reference text.

## Metadata acquisition (OAI-PMH)

Free, separate from the bulk source path. Provides per-paper:

- Title, authors, abstract
- Primary + secondary categories (e.g. `cs.LG`, `stat.ML`)
- Submission and last-update timestamps
- DOI / journal-ref if published

Two modes:
- **Bootstrap**: harvest all metadata once (~2–3 GB JSON, several hours over polite rate-limits).
- **Daily delta**: OAI-PMH supports incremental harvesting by date; ~thousands of papers/day.

Stored alongside the source archive; joined into `SourceRecord` at chunk time.

Note: the OAI-PMH bootstrap can run **before** the EC2 source pull. Once we have categories for every paper id, we can decide whether to subset the source pull by category — though as covered above, the bulk path doesn't actually save egress on category filtering, only on year ranges.

## Chunking strategy

Pandoc output preserves paragraph structure. Initial strategy:

- **Section-aware**: split on top-level pandoc-emitted headers (Introduction, Methods, Results, etc.) when present, fall back to token-based within long sections.
- **Abstract as first chunk**: prepend the OAI-PMH abstract as chunk 0 of every paper, since it's high-signal and often the only thing a query is matching against.
- **Token target**: 500 tokens with 50-token overlap (matches the wikipedia bucket default).

Subject to refinement after sampling output quality — could go larger (1024 tokens) if Qwen3-Embedding-0.6B handles it well; the model technically supports 32k context but quality typically peaks lower.

## Storage layout

Follows the existing tracked-bucket convention:

```
<server_root>/knowledge/arxiv_cs/
  bucket.toml
  source-cache/
    base/
      20260501/                            # snapshot date (= manifest pull date)
        yymm-2604.tar.zst                  # filtered output, per-month bundles
        yymm-2603.tar.zst
        ...
    deltas/
      20260601/yymm-2605.tar.zst           # next month's tarballs filtered + bundled
      ...
    metadata/
      oai_pmh_full.json.zst                # bootstrap metadata
      oai_pmh_delta_<date>.json.zst        # daily incrementals
  feed-state.toml
  slots/
    01HVAB.../
    active
```

`bucket.toml` example for an arxiv-cs bucket:

```toml
name = "arxiv_cs"
description = "arxiv CS-leaning papers, source-derived plaintext"

[source]
kind = "tracked"
driver = "arxiv"
category_filter = ["cs.*", "stat.ML", "math.OC"]  # applied post-ingest from metadata
delta_cadence = "monthly"
resync_cadence = "quarterly"

[chunker]
strategy = "section_aware"
chunk_tokens = 500
overlap_tokens = 50

[search_paths.dense]
enabled = true
[search_paths.sparse]
enabled = true
tokenizer = "default"

[defaults]
embedder = "qwen3_embed_0_6b"
serving_mode = "ram"
quantization = "int8"
current_base_snapshot_id = "20260501"
```

`category_filter` applied at the `SourceRecord → chunks` boundary: papers outside the filter still occupy bytes in the cache but produce no chunks. Keeps the source cache reusable for retargeted buckets without re-pulling.

## Build pipeline

Plugs into the existing `FeedDriver` machinery. `ArxivDriver` implements:

```rust
async fn latest_base(&self) -> Result<SnapshotId, FeedError>;
async fn fetch_base(&self, id: &SnapshotId, dest: &Path) -> Result<(), FeedError>;
async fn list_deltas_since(&self, since: Option<&DeltaId>) -> Result<Vec<DeltaId>, FeedError>;
async fn fetch_delta(&self, id: &DeltaId, dest: &Path) -> Result<(), FeedError>;
fn parse_adapter(&self) -> &'static str { "arxiv_source" }
```

Adapter `arxiv_source` reads filtered `.tar.zst` bundles + joined OAI-PMH metadata, emits `SourceRecord`.

`SnapshotId` = manifest snapshot date (e.g. `20260501`). `DeltaId` = a single tarball filename (e.g. `arXiv_src_2605_001.tar.filtered`).

`fetch_base` is the heavyweight operation: it triggers the EC2 pipeline. For v1, this can be **manual** — operator runs the EC2 script, drops the output bundles into `source-cache/base/<snapshot>/`, then activates the bucket. The driver's "fetch" is then just a checksum-verify-and-link operation. A fully automated `fetch_base` is a future improvement; manual is fine for v1 since the operation is rare (quarterly resync at most) and operator-supervised by nature.

`fetch_delta` runs locally (monthly tarballs are small enough to pull direct over public internet at $3/month).

## Schedule and cadence

| Operation | Frequency | Approx cost | Approx duration |
|---|---|---|---|
| Base pull (full archive, Option B) | One-time | ~$22 | ~24 h on EC2 |
| Monthly delta | Monthly (~30 GB at current rates) | ~$3 | hours, direct |
| OAI-PMH bootstrap | One-time | $0 | several hours (polite rate-limit) |
| OAI-PMH daily delta | Daily | $0 | minutes |
| Embedding build (0.6B model, full archive) | Per snapshot | electricity only | ~3 weeks continuous |
| Quarterly resync | Quarterly | ~$22 (re-base on EC2) | ~24 h pull + 3 weeks embedding |

The embedding build is the long pole — the source pull is dwarfed by it. This matches the wikipedia experience: bulk acquisition is incidental, embedding is the schedule driver.

## Open questions

- **Subset vs. all?** Once metadata is harvested, do we ingest all 3M papers or filter to CS/math/stat (~800k–1M)? Affects only embedding time, not source acquisition cost. Probably worth starting with full archive ingest and using `category_filter` per-bucket — but final call depends on retrieval quality testing on a small sample.
- **Math handling.** Verbatim LaTeX in chunks is the v1 plan. If retrieval on math-heavy queries underperforms, alternatives: strip math entirely, render math to image + caption, normalize LaTeX with a unicode mapping pass. Defer until we have query traffic.
- **Citation resolution.** `\cite{key}` gaps in pandoc output are a known minor degradation. If it matters, resolve via `.bbl` files (we have them) — adds a step but is mechanical.
- **PDF-only fallback.** ~8% of papers are PDF-only. Skip in v1; track count in the slot manifest. Reconsider only if the gap noticeably hurts retrieval.
- **Version policy.** v1 ingests latest version only. If we ever want version history, the existing `tombstone + insert` primitives handle it (each version is a new chunk id); cost is more chunks per paper.
- **Update mechanism for existing papers.** arxiv replaces papers in place when authors upload new versions, but the *source bundles* in the bulk archive don't get updated until the next monthly tarball. There may be a multi-week lag between an arxiv website update and our reflecting it — acceptable for v1.
