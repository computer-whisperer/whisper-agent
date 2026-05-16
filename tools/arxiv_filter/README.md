# arxiv_filter

One-shot tool for filtering arxiv source tarballs from the bulk S3 archive down
to tex-only content + pandoc-rendered plaintext. Output is a self-contained
`.tar.zst` per input tarball, suitable for archival on ceph and downstream
ingest by the arxiv knowledge bucket.

See [`docs/design_arxiv_bucket.md`](../../docs/design_arxiv_bucket.md) for the
acquisition strategy this implements (Option B — EC2 filter-in-region).

## Pipeline

For each paper bundle inside an arxiv source tarball:

1. If the bundle is PDF-only (no LaTeX source): record `pdf_only` and move on.
2. gunzip + untar the bundle.
3. Read `00README.json` to find the toplevel tex file (~always present in
   modern bundles).
4. Run `pandoc -f latex -t plain --wrap=none <toplevel>` with CWD set to the
   paper directory (so `\input` resolves).
5. Filter the bundle's files to text-only: `.tex`, `.bib`, `.bbl`, `.cls`,
   `.sty`, `.bst`, plus `00README.json`. Discard figures and aux files.
6. Emit `<paper_id>/{metadata.json, plaintext.txt, source/...}` into the output
   `.tar.zst`.

Pandoc subprocess timeout is 15 s per paper — successful conversions finish in
<2 s; longer runs are stuck on macro recursion and won't finish anyway.

## Output schema

```
<basename>.tar.zst
├── manifest.json                    # per-tarball summary, status counts, timings
└── <paper_id>/                      # one dir per paper
    ├── metadata.json                # status, toplevel, plaintext_len, stderr_tail, ...
    ├── plaintext.txt                # pandoc output (omitted on failure)
    └── source/                      # filtered text source (omitted on pdf_only)
        ├── 00README.json
        ├── <toplevel>.tex
        └── *.tex, *.bib, *.bbl, *.cls, *.sty, *.bst
```

Per-paper `metadata.json` `status` values:
- `ok` — pandoc succeeded, plaintext.txt populated
- `pdf_only` — no source bundle was published; skipped
- `no_toplevel` — source bundle had no identifiable toplevel tex
- `gunzip_failed` / `untar_failed` — bundle was corrupt
- `pandoc_failed` — pandoc parser error (custom class commands, malformed tex, ...)
- `timeout` — pandoc didn't finish within 15 s
- `empty_output` — pandoc returned exit 0 but no output

## Usage

```bash
# Local (stage 1 — verify on one tarball)
python3 filter.py /path/to/arXiv_src_YYMM_NNN.tar -o /path/to/output_dir

# Smoke test with a paper limit
python3 filter.py /path/to/arXiv_src_YYMM_NNN.tar -o /tmp/out --limit 20

# Inspect output
zstd -dc /path/to/output/arXiv_src_YYMM_NNN.tar.zst | tar -t | head
zstd -dc /path/to/output/arXiv_src_YYMM_NNN.tar.zst | tar -xO manifest.json | jq '.counts'
```

Defaults to using all CPU cores. Override with `-j N`.

## Measured baseline (stage 1)

Input: `src/arXiv_src_2604_001.tar`, 582 MB, 152 papers from April 2026.

On this 24-core machine:

| Metric | Value |
|---|---|
| Wall time (full tarball) | 17.1 s |
| Throughput | 8.9 papers/s |
| Output size | 6.6 MB (88× compression vs. input) |

Status distribution:

| Status | Count | Fraction |
|---|---|---|
| ok | 127 | 83.6% |
| pdf_only | 12 | 7.9% |
| pandoc_failed | 9 | 5.9% |
| timeout | 4 | 2.6% |

Pandoc success rate on source-available papers: **127 / 140 = 90.7%**.

Extrapolating to the full archive: ~83% plaintext yield, ~130 GB compressed
output for 3M papers.

## Known failure modes (stage 1 observations)

- **Custom class commands** (~3% of source-available papers): pandoc doesn't
  know `\maketitleabstract`, `\abstract{\input{...}}`, etc. defined in
  conference class files. Fixable in principle by mapping common conference
  classes; out of scope for v1.
- **Pandoc parser quirks** (~2%): edge-case TeX like `\vskip +0.1in`,
  `\cite{{key}}`, leading-digit tex filenames trip pandoc's latex reader.
- **Actual LaTeX errors in source** (~1%): some papers were uploaded with bugs
  pdflatex tolerates but pandoc doesn't.
- **Macro recursion hangs** (~2.6%): killed by 15 s timeout.

All failures are recorded in `metadata.json` for that paper, so we can
post-hoc classify and revisit selectively.

## Files

- `filter.py` — process one local tarball, write one `.tar.zst`.
- `drive.py` — walk the manifest, drive S3 download → filter → S3 upload, with resume markers.
- `bootstrap.sh` — install pandoc 3.x, python deps, pull scripts from staging on a fresh EC2 instance.
- `watchdog.sh` — runs on the instance alongside drive.py; polls staging marker count and self-terminates the instance when the archive is complete.
- `STAGE3_RUNBOOK.md` — end-to-end procedure for running the full archive on EC2.

## Stage status (as of 2026-05-15)

- ✅ **Stage 1** — local prototype on one tarball (152 papers).
  Done; 90.7% pandoc success on source-available papers.
- ✅ **Stage 2** — driver pulled 12 tarballs from S3, filtered locally,
  uploaded to staging. 1,706 papers; 88.6% pandoc success at scale; 75×
  compression; 8.2 min wall clock.
- ⏳ **Stage 3** — full archive (12,374 tarballs) on EC2 c6i.16xlarge spot
  in `us-east-1`, ~35 h, ~$30. Setup complete; see [STAGE3_RUNBOOK.md](STAGE3_RUNBOOK.md).

## AWS assets already created

- IAM role + instance profile `arxiv-filter-ec2` with three inline policies:
  - `s3-arxiv-and-staging` — read arxiv source bucket (requester-pays), read/write staging
  - `terminate-self` — `ec2:TerminateInstances` scoped to the calling instance only
  - Plus managed policy `AmazonSSMManagedInstanceCore` for SSM Session Manager
- IAM user `arxiv-bulk` with read on `arxiv` + read/write on staging (for local-machine ops)
- Staging bucket `christian-arxiv-staging` with 30-day lifecycle, public access blocked
- Scripts uploaded to `s3://christian-arxiv-staging/scripts/`
