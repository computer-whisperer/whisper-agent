#!/usr/bin/env python3
"""
Generate webui favicons + Android launcher icons from icon_v1.png.

Re-run this from the repo root after updating icon_v1.png:

    python3 assets/icon/generate.py

What it produces:

    crates/whisper-agent-webui/assets/favicon.png            32x32
    crates/whisper-agent-webui/assets/favicon-192.png        192x192 (PWA)
    android/app/src/main/res/mipmap-{m,h,xh,xxh,xxxh}dpi/
        ic_launcher.png                                       legacy square
        ic_launcher_round.png                                 legacy round
    android/app/src/main/res/mipmap-xxxhdpi/
        ic_launcher_foreground.png                            adaptive fg @ 432

The chroma-key step samples the four corners of icon_v1.png to detect the
dark-navy background, then maps every pixel within a soft distance band to
fully-transparent. Stopgap quality — the edges have some halo on close
inspection but the launcher mask hides it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "assets" / "icon" / "icon_v1.png"

# Density buckets for legacy mipmap fallbacks. Modern launchers (API 26+)
# pull the adaptive icon XML from mipmap-anydpi-v26/ instead, but we still
# need legacy PNGs for older devices and as a fallback.
LEGACY_SIZES = {
    "mdpi": 48,
    "hdpi": 72,
    "xhdpi": 96,
    "xxhdpi": 144,
    "xxxhdpi": 192,
}

# Adaptive-icon foreground canvas at xxxhdpi: 108dp × 4 dp/px = 432.
# Android masks the outer 18dp on each side, so the visible safe area is
# the central 66dp × 66dp = 264 × 264 px circle/squircle. We center the
# mascot in a 280×280 box (slightly larger than the safe zone, so a bit
# of the hexagon may bleed into the mask region — acceptable for v1).
ADAPTIVE_FG_SIZE = 432
ADAPTIVE_FG_INNER = 280


def detect_background(img: Image.Image, sample: int = 24) -> tuple[int, int, int]:
    """Average RGB across the four corners — the icon's solid-fill region."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    corners = np.concatenate(
        [
            arr[:sample, :sample].reshape(-1, 3),
            arr[:sample, w - sample :].reshape(-1, 3),
            arr[h - sample :, :sample].reshape(-1, 3),
            arr[h - sample :, w - sample :].reshape(-1, 3),
        ]
    )
    return tuple(int(c) for c in corners.mean(axis=0))


def chroma_key(
    img: Image.Image, bg: tuple[int, int, int], inner: float = 35.0, outer: float = 90.0
) -> Image.Image:
    """Replace `bg`-similar pixels with transparency, soft falloff between
    `inner` (full transparent) and `outer` (full opaque) RGB distance."""
    arr = np.array(img.convert("RGBA")).astype(np.int32)
    dr = arr[:, :, 0] - bg[0]
    dg = arr[:, :, 1] - bg[1]
    db = arr[:, :, 2] - bg[2]
    distance = np.sqrt(dr * dr + dg * dg + db * db)
    # Soft falloff: distance<inner → 0, distance>outer → 255, between → linear ramp.
    alpha = np.clip((distance - inner) / (outer - inner) * 255.0, 0, 255).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def fit_into(canvas_size: int, inner_size: int, art: Image.Image) -> Image.Image:
    """Place `art` (preserving aspect) into a transparent square `canvas_size`,
    scaled so the longer edge fits in `inner_size`."""
    art = art.copy()
    art.thumbnail((inner_size, inner_size), Image.LANCZOS)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    x = (canvas_size - art.width) // 2
    y = (canvas_size - art.height) // 2
    canvas.paste(art, (x, y), art)
    return canvas


def round_mask(size: int) -> Image.Image:
    """Circular alpha mask the launcher fallback PNGs use."""
    mask = Image.new("L", (size, size), 0)
    from PIL import ImageDraw

    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    return mask


def main() -> int:
    if not SRC.exists():
        print(f"missing source: {SRC}", file=sys.stderr)
        return 1

    src = Image.open(SRC).convert("RGBA")
    bg = detect_background(src)
    print(f"detected background: rgb{bg}")

    # Foreground = chroma-keyed art (mascot only, transparent elsewhere).
    fg = chroma_key(src, bg)

    # --- webui favicons --------------------------------------------------
    # Use the original (with background) for the favicon so the icon reads
    # at a glance against any browser-tab backdrop. Two sizes cover most
    # browser/PWA needs.
    webui_assets = REPO_ROOT / "crates" / "whisper-agent-webui" / "assets"
    webui_assets.mkdir(parents=True, exist_ok=True)
    src_square = src.resize((512, 512), Image.LANCZOS)
    src_square.resize((32, 32), Image.LANCZOS).save(webui_assets / "favicon.png")
    src_square.resize((192, 192), Image.LANCZOS).save(webui_assets / "favicon-192.png")
    print(f"wrote {webui_assets/'favicon.png'} (32x32)")
    print(f"wrote {webui_assets/'favicon-192.png'} (192x192)")

    # --- Android legacy mipmap PNGs --------------------------------------
    # Pre-API-26 devices use these directly. API 26+ launchers prefer the
    # adaptive XML but fall back to these if the mask system is disabled.
    res_root = REPO_ROOT / "android" / "app" / "src" / "main" / "res"
    for density, px in LEGACY_SIZES.items():
        d = res_root / f"mipmap-{density}"
        d.mkdir(parents=True, exist_ok=True)
        # Square: source-with-bg, scaled.
        sq = src.resize((px, px), Image.LANCZOS)
        sq.save(d / "ic_launcher.png")
        # Round: same content, alpha-masked to a circle.
        rd = sq.copy()
        rd.putalpha(round_mask(px))
        rd.save(d / "ic_launcher_round.png")
    print(f"wrote ic_launcher.png + ic_launcher_round.png at {len(LEGACY_SIZES)} densities")

    # --- Android adaptive icon foreground --------------------------------
    # Foreground only — the chroma-keyed mascot, centered in a transparent
    # 432×432 canvas with safe-zone padding. Background layer is a color
    # resource referenced from the adaptive XML (no PNG needed).
    fg_canvas = fit_into(ADAPTIVE_FG_SIZE, ADAPTIVE_FG_INNER, fg)
    xxxh = res_root / "mipmap-xxxhdpi"
    xxxh.mkdir(parents=True, exist_ok=True)
    fg_canvas.save(xxxh / "ic_launcher_foreground.png")
    print(f"wrote {xxxh/'ic_launcher_foreground.png'} ({ADAPTIVE_FG_SIZE}x{ADAPTIVE_FG_SIZE})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
