#!/usr/bin/env python3
"""
Filter one arxiv source tarball down to tex-only content + pandoc plaintext.

Input:  one src/arXiv_src_YYMM_NNN.tar from arxiv's bulk source bucket.
Output: one <basename>.tar.zst containing per-paper directories with:
          <paper_id>/
            plaintext.txt        # pandoc -f latex -t plain (omitted on failure)
            metadata.json        # status, warnings, file counts, timings
            source/              # filtered text source files (omitted on pdf-only)
              <toplevel>.tex, *.bib, *.bbl, *.cls, *.sty, *.bst, 00README.json

Plus a top-level manifest.json with per-tarball summary.
"""

from __future__ import annotations  # Python 3.9 compat for `str | None` etc.

import argparse
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import zstandard as zstd

TEXT_EXTS = {".tex", ".bib", ".bbl", ".cls", ".sty", ".bst"}
# Successful papers finish in <2s. Hangs are usually macro-recursion that
# never terminate, so a short timeout is strictly better than a long one.
PANDOC_TIMEOUT_S = 15
PANDOC_ARGS = ["pandoc", "-f", "latex", "-t", "plain", "--wrap=none"]


def is_text_file(name: str) -> bool:
    base = os.path.basename(name).lower()
    if base == "00readme.json":
        return True
    ext = os.path.splitext(name)[1].lower()
    return ext in TEXT_EXTS


def find_toplevel_tex(paper_dir: Path) -> str | None:
    """Read 00README.json to find the toplevel tex. Fallback: any single .tex
    file, else None."""
    readme = paper_dir / "00README.json"
    if readme.exists():
        try:
            data = json.loads(readme.read_text())
            for src in data.get("sources", []):
                if src.get("usage") == "toplevel":
                    return src.get("filename")
        except (json.JSONDecodeError, OSError):
            pass
    tex_files = list(paper_dir.glob("*.tex"))
    if len(tex_files) == 1:
        return tex_files[0].name
    for candidate in ("main.tex", "paper.tex", "manuscript.tex", "ms.tex"):
        if (paper_dir / candidate).exists():
            return candidate
    return None


def run_pandoc(paper_dir: Path, toplevel: str) -> tuple[str | None, str, str]:
    """Run pandoc on the toplevel tex with CWD set to the paper dir.
    Returns (plaintext or None, status, stderr_excerpt)."""
    try:
        result = subprocess.run(
            PANDOC_ARGS + [toplevel],
            cwd=paper_dir,
            capture_output=True,
            text=True,
            timeout=PANDOC_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout", ""
    except FileNotFoundError:
        return None, "pandoc_missing", ""
    stderr_tail = result.stderr[-400:] if result.stderr else ""
    if result.returncode != 0:
        return None, "pandoc_failed", stderr_tail
    if not result.stdout.strip():
        return None, "empty_output", stderr_tail
    return result.stdout, "ok", stderr_tail


def process_paper(args: tuple[str, bytes, bool]) -> dict:
    """Worker: process one paper bundle.

    args = (paper_id, raw_bytes, is_pdf_only)
    raw_bytes for .gz is the gzipped tar; for .pdf is the PDF bytes.
    Returns a dict carrying metadata + the per-paper output tar bytes.
    """
    paper_id, raw, is_pdf_only = args
    start = time.monotonic()

    if is_pdf_only:
        meta = {
            "paper_id": paper_id,
            "status": "pdf_only",
            "duration_s": round(time.monotonic() - start, 3),
        }
        return _pack_paper_output(paper_id, meta, plaintext=None, source_files=None)

    with tempfile.TemporaryDirectory(prefix=f"arxiv_{paper_id}_") as tmpdir:
        paper_dir = Path(tmpdir) / paper_id
        paper_dir.mkdir()
        try:
            inner_tar = gzip.decompress(raw)
        except (OSError, EOFError) as e:
            meta = {"paper_id": paper_id, "status": "gunzip_failed",
                    "error": str(e), "duration_s": round(time.monotonic() - start, 3)}
            return _pack_paper_output(paper_id, meta, plaintext=None, source_files=None)
        # Modern bundles (post-~2000): gunzipped content is a tar archive.
        # Older single-file papers: gunzipped content is raw LaTeX directly.
        try:
            with tarfile.open(fileobj=io.BytesIO(inner_tar)) as tf:
                tf.extractall(paper_dir, filter="data")
        except tarfile.TarError:
            (paper_dir / "main.tex").write_bytes(inner_tar)
        except OSError as e:
            meta = {"paper_id": paper_id, "status": "untar_failed",
                    "error": str(e), "duration_s": round(time.monotonic() - start, 3)}
            return _pack_paper_output(paper_id, meta, plaintext=None, source_files=None)

        toplevel = find_toplevel_tex(paper_dir)
        if toplevel is None:
            meta = {"paper_id": paper_id, "status": "no_toplevel",
                    "duration_s": round(time.monotonic() - start, 3)}
            return _pack_paper_output(paper_id, meta, plaintext=None, source_files=None)

        plaintext, status, stderr_tail = run_pandoc(paper_dir, toplevel)

        source_files = {}
        for f in paper_dir.rglob("*"):
            if f.is_file() and is_text_file(f.name):
                rel = f.relative_to(paper_dir).as_posix()
                try:
                    source_files[rel] = f.read_bytes()
                except OSError:
                    continue

        meta = {
            "paper_id": paper_id,
            "status": status,
            "toplevel": toplevel,
            "source_file_count": len(source_files),
            "plaintext_len": len(plaintext) if plaintext else 0,
            "stderr_tail": stderr_tail,
            "duration_s": round(time.monotonic() - start, 3),
        }
        return _pack_paper_output(paper_id, meta, plaintext, source_files)


def _pack_paper_output(
    paper_id: str,
    meta: dict,
    plaintext: str | None,
    source_files: dict[str, bytes] | None,
) -> dict:
    """Build the in-memory tar entries representing one paper's output dir."""
    entries = []
    meta_bytes = json.dumps(meta, indent=2, sort_keys=True).encode()
    entries.append((f"{paper_id}/metadata.json", meta_bytes))
    if plaintext:
        entries.append((f"{paper_id}/plaintext.txt", plaintext.encode("utf-8")))
    if source_files:
        for rel, data in source_files.items():
            entries.append((f"{paper_id}/source/{rel}", data))
    return {"paper_id": paper_id, "status": meta["status"], "entries": entries,
            "meta": meta}


def stream_papers_from_tarball(tarball: Path):
    """Yield (paper_id, raw_bytes, is_pdf_only) for each entry in the input."""
    with tarfile.open(tarball, "r|") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            base = os.path.basename(name)
            if base.endswith(".gz"):
                paper_id = base[:-3]
                is_pdf = False
            elif base.endswith(".pdf"):
                paper_id = base[:-4]
                is_pdf = True
            else:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            yield paper_id, f.read(), is_pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tarball", type=Path, help="path to arXiv_src_YYMM_NNN.tar")
    ap.add_argument("-o", "--out-dir", type=Path, required=True,
                    help="output directory; one .tar.zst written here")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 8,
                    help="parallel pandoc workers (default: all cores)")
    ap.add_argument("--zstd-level", type=int, default=19,
                    help="zstd compression level (default: 19)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N papers (smoke testing)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.tarball.stem + ".tar.zst"
    out_path = args.out_dir / out_name

    print(f"[filter] input:  {args.tarball}  ({args.tarball.stat().st_size/1e6:.1f} MB)")
    print(f"[filter] output: {out_path}")
    print(f"[filter] jobs:   {args.jobs}")

    t_start = time.monotonic()
    papers = stream_papers_from_tarball(args.tarball)
    if args.limit:
        papers = (p for i, p in enumerate(papers) if i < args.limit)

    # Open output tar.zst as a streaming writer
    cctx = zstd.ZstdCompressor(level=args.zstd_level, threads=-1)
    raw_fp = open(out_path, "wb")
    zst_fp = cctx.stream_writer(raw_fp)
    tar_out = tarfile.open(fileobj=zst_fp, mode="w|")

    summary = {
        "input_tarball": args.tarball.name,
        "input_size_bytes": args.tarball.stat().st_size,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "papers": [],
        "counts": {},
    }

    processed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(process_paper, papers, chunksize=1):
            processed += 1
            for name, data in result["entries"]:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = int(time.time())
                tar_out.addfile(info, io.BytesIO(data))
            summary["papers"].append(result["meta"])
            summary["counts"][result["status"]] = \
                summary["counts"].get(result["status"], 0) + 1
            if processed % 25 == 0:
                elapsed = time.monotonic() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[filter]   {processed:>4} papers, {rate:.1f}/s")

    summary["duration_s"] = round(time.monotonic() - t_start, 2)
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["total_papers"] = processed

    # Manifest at the top of the output
    manifest_bytes = json.dumps(summary, indent=2, sort_keys=True).encode()
    info = tarfile.TarInfo(name="manifest.json")
    info.size = len(manifest_bytes)
    info.mtime = int(time.time())
    tar_out.addfile(info, io.BytesIO(manifest_bytes))

    tar_out.close()
    zst_fp.close()
    raw_fp.close()

    elapsed = summary["duration_s"]
    rate = processed / elapsed if elapsed > 0 else 0
    print()
    print(f"[filter] done. {processed} papers in {elapsed:.1f}s ({rate:.1f}/s)")
    print(f"[filter] output: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"[filter] status counts: {summary['counts']}")


if __name__ == "__main__":
    main()
