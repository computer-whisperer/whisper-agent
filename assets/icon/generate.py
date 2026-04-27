#!/usr/bin/env python3
"""
Generate webui favicons + Android launcher icons from icon_v1.svg.

Re-run from the repo root after editing icon_v1.svg:

    python3 assets/icon/generate.py

Requires `resvg` on PATH — the same engine the agent uses for the
view_image tool's SVG rasterization, so what gets shipped here matches
what the model sees when it loads the source. Install via
`cargo install resvg` if it's missing.

What it produces:

    crates/whisper-agent-webui/assets/favicon.png            32x32
    crates/whisper-agent-webui/assets/favicon-192.png        192x192 (PWA)
    android/app/src/main/res/mipmap-{m,h,xh,xxh,xxxh}dpi/
        ic_launcher.png                                       legacy square
        ic_launcher_round.png                                 legacy round
    android/app/src/main/res/mipmap-xxxhdpi/
        ic_launcher_foreground.png                            adaptive fg @ 432

The full SVG (with the radial-gradient backdrop) drives the favicons and
the legacy / round mipmaps. The adaptive-icon foreground is rendered from
a transient copy with the background `<rect>` stripped — the halo glow
above stays in so the launcher mask reads as a soft falloff rather than a
hard mascot cutout. The Android background layer is a solid color
declared in `res/values/colors.xml` (`#010A1E`), matching the SVG's
outer-edge tone.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "assets" / "icon" / "icon_v1.svg"

# Resolution we render the bg-stripped SVG at before cropping for the
# favicons. Larger than any output size so the auto-crop bbox is tight
# (sub-pixel halo gets averaged down on the resize) and the final
# downsample carries enough antialiasing detail.
FAVICON_HIRES = 1024

# Pixels with alpha at or below this count as background when computing
# the favicon crop. The halo gradient fades to alpha 0 at its outer
# edge but rounding leaves a few pixels at single-digit alpha — we
# treat those as crop-out-able so the favicon hugs the mascot rather
# than the technically-non-empty halo bounds.
FAVICON_ALPHA_THRESHOLD = 8

# Padding ratio around the cropped mascot bbox before resize. Small —
# the favicon should fill the tab/PWA tile without bleeding. 0.06 = 6%
# of the bbox's longer axis on each side.
FAVICON_CROP_PADDING = 0.06

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

# Adaptive-icon foreground canvas at xxxhdpi: 108dp × 4 dp/px = 432px.
# Android masks ~18dp on each edge, leaving a 72dp visible disk and a
# 66dp safe-zone guarantee for content. The SVG's hand-authored padding
# already keeps the mascot well within the 264px safe zone at this size,
# so we render the bg-stripped SVG directly at 432×432 without an extra
# inset pass.
ADAPTIVE_FG_SIZE = 432


def render_svg(svg_bytes: bytes, out: Path, size: int) -> None:
    """Rasterize an SVG (passed as raw bytes) to a square PNG at `size`px
    via resvg's stdin → file mode. Raises if resvg exits non-zero."""
    proc = subprocess.run(
        ["resvg", "--width", str(size), "--height", str(size), "-", str(out)],
        input=svg_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"resvg failed (exit {proc.returncode}): {proc.stderr.decode().strip()}"
        )


def render_svg_bytes(svg_bytes: bytes, size: int) -> bytes:
    """Rasterize SVG to PNG bytes at `size`×`size` via resvg, stdin → stdout.
    Used for the favicon path where we crop in-memory before writing."""
    proc = subprocess.run(
        ["resvg", "--width", str(size), "--height", str(size), "-", "-c"],
        input=svg_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"resvg failed (exit {proc.returncode}): {proc.stderr.decode().strip()}"
        )
    return proc.stdout


def alpha_bbox(img: Image.Image, threshold: int) -> tuple[int, int, int, int]:
    """Tight bbox of pixels whose alpha exceeds `threshold`. Returns
    (left, top, right, bottom) in PIL crop convention. Skips faintly
    semi-transparent pixels (e.g. halo gradient tails) that
    `Image.getbbox` would otherwise pull into the bounds."""
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    rows = np.any(alpha > threshold, axis=1)
    cols = np.any(alpha > threshold, axis=0)
    if not rows.any():
        raise RuntimeError("alpha_bbox: image is fully transparent")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1)


def square_padded_crop(
    img: Image.Image, bbox: tuple[int, int, int, int], padding_ratio: float
) -> Image.Image:
    """Center `bbox` inside a square crop of `img` with `padding_ratio`
    breathing room added (relative to the longer axis). The result is
    clamped to image bounds — if the bbox is near a corner the square
    may not be perfectly centered, but the favicon is square already so
    that case never triggers in practice."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    side = max(w, h)
    side = int(round(side * (1 + 2 * padding_ratio)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = side / 2
    iw, ih = img.size
    sx0 = max(0, int(round(cx - half)))
    sy0 = max(0, int(round(cy - half)))
    sx1 = min(iw, sx0 + side)
    sy1 = min(ih, sy0 + side)
    return img.crop((sx0, sy0, sx1, sy1))


# Match the canvas-filling background rect by its `fill="url(#bg)"`
# reference — the SVG's hand-author owns layer naming, so a literal
# substitution is dead simple and stays self-checking via the assertion
# below if the SVG ever loses that anchor.
_BG_RECT_RE = re.compile(r'\s*<rect[^>]*fill="url\(#bg\)"[^>]*/>\s*\n?')


def strip_background(svg_text: str) -> str:
    """Return a copy of `svg_text` with the full-canvas bg rect removed.
    The halo `<circle fill="url(#halo)">` and the mascot layers are
    preserved — we want the soft glow in the adaptive foreground, just
    not the opaque navy sheet behind it."""
    stripped = _BG_RECT_RE.sub("\n", svg_text, count=1)
    if stripped == svg_text:
        raise RuntimeError(
            'could not find bg <rect fill="url(#bg)"/> in icon_v1.svg — '
            "did the SVG structure change? update strip_background to match."
        )
    return stripped


def round_mask(size: int) -> Image.Image:
    """Circular alpha mask the launcher fallback PNGs use."""
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    return mask


def main() -> int:
    if not SRC.exists():
        print(f"missing source: {SRC}", file=sys.stderr)
        return 1
    if shutil.which("resvg") is None:
        print(
            "resvg not on PATH. Install via `cargo install resvg` or your "
            "distro's package manager (Arch: librsvg + rust-resvg, Debian: "
            "the cargo route is the most reliable).",
            file=sys.stderr,
        )
        return 1

    svg_full = SRC.read_text(encoding="utf-8")
    svg_fg = strip_background(svg_full).encode("utf-8")
    svg_full_bytes = svg_full.encode("utf-8")

    # --- webui favicons --------------------------------------------------
    # Browser tabs and PWA tiles read best when the icon is transparent
    # behind the mascot (rather than carrying the SVG's full
    # radial-gradient backdrop, which ends up looking like a hard
    # square frame against the browser chrome) and tightly cropped
    # (the SVG's hand-authored padding is dialed for adaptive-icon
    # safe-zone fit, not for 32-px favicons).
    #
    # Render the bg-stripped SVG at high res, auto-crop to the mascot's
    # alpha bbox + a small breathing-room ratio, then resize down. The
    # 192 doubles as the apple-touch-icon (linked from index.html /
    # login.html alongside the 32).
    webui_assets = REPO_ROOT / "crates" / "whisper-agent-webui" / "assets"
    webui_assets.mkdir(parents=True, exist_ok=True)
    fav_hires = Image.open(io.BytesIO(render_svg_bytes(svg_fg, FAVICON_HIRES))).convert(
        "RGBA"
    )
    bbox = alpha_bbox(fav_hires, FAVICON_ALPHA_THRESHOLD)
    fav_cropped = square_padded_crop(fav_hires, bbox, FAVICON_CROP_PADDING)
    fav_cropped.resize((32, 32), Image.LANCZOS).save(webui_assets / "favicon.png")
    fav_cropped.resize((192, 192), Image.LANCZOS).save(webui_assets / "favicon-192.png")
    print(
        f"wrote {webui_assets/'favicon.png'} (32x32) — cropped from "
        f"{FAVICON_HIRES}px hires, bbox {bbox}, side {fav_cropped.size[0]}px"
    )
    print(f"wrote {webui_assets/'favicon-192.png'} (192x192)")

    # --- Android legacy mipmap PNGs --------------------------------------
    # Pre-API-26 devices use these directly. API 26+ launchers prefer the
    # adaptive XML but fall back to these if the mask system is disabled.
    res_root = REPO_ROOT / "android" / "app" / "src" / "main" / "res"
    for density, px in LEGACY_SIZES.items():
        d = res_root / f"mipmap-{density}"
        d.mkdir(parents=True, exist_ok=True)
        # Square: full SVG (with bg gradient), rendered at native size.
        render_svg(svg_full_bytes, d / "ic_launcher.png", px)
        # Round: re-open the square PNG we just wrote and slap on a
        # circular alpha mask. The radial gradient's darker corners
        # disappear cleanly under the mask.
        rd = Image.open(d / "ic_launcher.png").convert("RGBA")
        rd.putalpha(round_mask(px))
        rd.save(d / "ic_launcher_round.png")
    print(f"wrote ic_launcher.png + ic_launcher_round.png at {len(LEGACY_SIZES)} densities")

    # --- Android adaptive icon foreground --------------------------------
    # Foreground only — the bg-stripped SVG, rendered at 432×432. The
    # launcher composites this over the navy bg color in
    # res/values/colors.xml (#010A1E).
    xxxh = res_root / "mipmap-xxxhdpi"
    xxxh.mkdir(parents=True, exist_ok=True)
    render_svg(svg_fg, xxxh / "ic_launcher_foreground.png", ADAPTIVE_FG_SIZE)
    print(f"wrote {xxxh/'ic_launcher_foreground.png'} ({ADAPTIVE_FG_SIZE}x{ADAPTIVE_FG_SIZE})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
