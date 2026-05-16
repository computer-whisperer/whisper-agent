#!/usr/bin/env python3
"""
Drive the arxiv filter against tarballs listed in arxiv's bulk source manifest.

For each selected tarball:
  1. Download from s3://arxiv/<filename> (requester-pays, intra-region free on EC2)
  2. Run tools/arxiv_filter/filter.py on it locally
  3. Upload the filtered .tar.zst to s3://<staging>/filtered/<filename>.zst
  4. Write a marker s3://<staging>/markers/<filename>.done

Resumable: tarballs whose .done marker exists are skipped.

Uses aws-cli via subprocess for S3 transfers — no pip dependencies required, so
the driver runs on any base EC2 AMI that has aws-cli + python3 + pandoc + zstd.
"""

from __future__ import annotations  # Python 3.9 compat for `str | None` etc.

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_BUCKET = "arxiv"
MANIFEST_KEY = "src/arXiv_src_manifest.xml"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, raising on nonzero. Returns the completed process."""
    return subprocess.run(cmd, check=True, **kwargs)


def aws(args: list[str], profile: str | None = None,
        capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["aws"]
    if profile:
        cmd += ["--profile", profile]
    cmd += args
    if capture:
        return run(cmd, capture_output=True, text=True)
    return run(cmd)


def fetch_manifest(profile: str, dest: Path) -> None:
    aws(
        ["s3api", "get-object",
         "--bucket", SOURCE_BUCKET, "--key", MANIFEST_KEY,
         "--request-payer", "requester",
         str(dest)],
        profile=profile, capture=True,
    )


def _yymm_to_yyyymm(yymm: str) -> int:
    """arxiv's yymm uses 2-digit year; arxiv started 1991. yy>=91 → 19xx else 20xx."""
    yy = int(yymm[:2])
    full_year = 1900 + yy if yy >= 91 else 2000 + yy
    return full_year * 100 + int(yymm[2:])


def parse_manifest(path: Path) -> list[dict]:
    """Return list of file entries sorted chronologically by (yyyymm, seq_num)."""
    root = ET.parse(path).getroot()
    files = []
    for f in root.findall("file"):
        yymm = f.findtext("yymm")
        files.append({
            "filename": f.findtext("filename"),
            "yymm": yymm,
            "yyyymm": _yymm_to_yyyymm(yymm),
            "seq_num": int(f.findtext("seq_num")),
            "size": int(f.findtext("size")),
            "num_items": int(f.findtext("num_items")),
            "md5": f.findtext("md5sum"),
        })
    files.sort(key=lambda x: (x["yyyymm"], x["seq_num"]))
    return files


def select_files(files: list[dict], select: str) -> list[dict]:
    """Apply a selection spec.

    Specs:
      "last:N"      — last N tarballs in the manifest (most recent)
      "yymm:NNNN"   — all tarballs for a single yymm
      "from:YYMM"   — all tarballs at or after YYMM
      "all"         — every tarball in the manifest
    """
    if select == "all":
        return files
    if select.startswith("last:"):
        n = int(select.split(":")[1])
        return files[-n:]
    if select.startswith("yymm:"):
        ym = select.split(":")[1]
        return [f for f in files if f["yymm"] == ym]
    if select.startswith("from:"):
        ym = select.split(":")[1]
        cutoff = _yymm_to_yyyymm(ym)
        return [f for f in files if f["yyyymm"] >= cutoff]
    raise SystemExit(f"unknown selector: {select}")


def staging_marker_exists(profile: str, bucket: str, filename: str) -> bool:
    cmd = ["aws"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["s3api", "head-object",
            "--bucket", bucket, "--key", f"markers/{filename}.done"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0


def process_one(entry: dict, profile: str, staging_bucket: str,
                tmpdir: Path, jobs: int) -> dict:
    """Download → filter → upload → mark one tarball. Returns timing dict."""
    fname = entry["filename"]
    base = os.path.basename(fname)
    local_in = tmpdir / base
    out_dir = tmpdir / "filtered"
    out_dir.mkdir(exist_ok=True)
    local_out = out_dir / (Path(base).stem + ".tar.zst")

    timings = {"filename": fname, "size_bytes": entry["size"]}

    t0 = time.monotonic()
    aws(
        ["s3api", "get-object",
         "--bucket", SOURCE_BUCKET, "--key", fname,
         "--request-payer", "requester",
         str(local_in)],
        profile=profile, capture=True,
    )
    timings["download_s"] = round(time.monotonic() - t0, 2)

    t0 = time.monotonic()
    filter_script = Path(__file__).parent / "filter.py"
    run([sys.executable, str(filter_script),
         str(local_in), "-o", str(out_dir),
         "-j", str(jobs)])
    timings["filter_s"] = round(time.monotonic() - t0, 2)
    timings["out_size_bytes"] = local_out.stat().st_size

    t0 = time.monotonic()
    aws(
        ["s3", "cp", str(local_out),
         f"s3://{staging_bucket}/filtered/{local_out.name}"],
        profile=profile, capture=True,
    )
    marker_body = tmpdir / "_empty_marker"
    if not marker_body.exists():
        marker_body.write_bytes(b"")
    aws(
        ["s3api", "put-object",
         "--bucket", staging_bucket,
         "--key", f"markers/{base}.done",
         "--body", str(marker_body)],
        profile=profile, capture=True,
    )
    timings["upload_s"] = round(time.monotonic() - t0, 2)

    local_in.unlink()
    local_out.unlink()

    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="arxiv",
                    help="aws-cli profile to use (default: arxiv)")
    ap.add_argument("--staging-bucket", default="christian-arxiv-staging",
                    help="staging bucket for filtered output + markers")
    ap.add_argument("--select", default="last:1",
                    help="selector: 'last:N' | 'yymm:NNNN' | 'from:YYMM' | 'all'")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/arxiv_drive"),
                    help="local scratch dir for tarballs in flight")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 8,
                    help="parallel pandoc workers per tarball (default: all cores)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show selected tarballs and exit without downloading")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.workdir / "arXiv_src_manifest.xml"

    print(f"[drive] fetching manifest")
    fetch_manifest(args.profile, manifest_path)
    files = parse_manifest(manifest_path)
    print(f"[drive] manifest has {len(files):,} tarballs")

    selected = select_files(files, args.select)
    total_size = sum(f["size"] for f in selected)
    print(f"[drive] selected {len(selected)} tarballs ({total_size/1e9:.1f} GB total)")
    if args.dry_run:
        for f in selected[:20]:
            print(f"  {f['filename']:<40} {f['size']/1e6:>7.1f} MB  {f['num_items']:>5} papers")
        if len(selected) > 20:
            print(f"  ... and {len(selected)-20} more")
        return

    summary_rows = []
    t_overall = time.monotonic()
    for i, entry in enumerate(selected, 1):
        base = os.path.basename(entry["filename"])
        if staging_marker_exists(args.profile, args.staging_bucket, base):
            print(f"[drive] [{i}/{len(selected)}] {base} already processed, skipping")
            continue
        print(f"[drive] [{i}/{len(selected)}] {base} "
              f"({entry['size']/1e6:.1f} MB, {entry['num_items']} papers)")
        try:
            timings = process_one(entry, args.profile, args.staging_bucket,
                                  args.workdir, args.jobs)
            summary_rows.append(timings)
            print(f"[drive]    download={timings['download_s']}s "
                  f"filter={timings['filter_s']}s "
                  f"upload={timings['upload_s']}s "
                  f"out={timings['out_size_bytes']/1e6:.1f} MB")
        except subprocess.CalledProcessError as e:
            print(f"[drive]    FAILED: {e}")
            summary_rows.append({"filename": entry["filename"], "error": str(e)})

    elapsed = time.monotonic() - t_overall
    print(f"\n[drive] done. processed {len(summary_rows)} tarballs in {elapsed/60:.1f} min")
    if summary_rows:
        ok_rows = [r for r in summary_rows if "error" not in r]
        if ok_rows:
            in_bytes = sum(r["size_bytes"] for r in ok_rows)
            out_bytes = sum(r["out_size_bytes"] for r in ok_rows)
            print(f"[drive]   input total:  {in_bytes/1e9:.2f} GB")
            print(f"[drive]   output total: {out_bytes/1e6:.1f} MB (compression {in_bytes/out_bytes:.0f}x)")

    summary_path = args.workdir / "drive_summary.json"
    summary_path.write_text(json.dumps({
        "selected": args.select,
        "total_tarballs": len(selected),
        "rows": summary_rows,
        "elapsed_s": round(elapsed, 1),
    }, indent=2))
    print(f"[drive] summary written to {summary_path}")


if __name__ == "__main__":
    main()
