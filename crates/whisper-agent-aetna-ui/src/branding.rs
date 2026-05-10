//! Bundled brand assets.
//!
//! The full-color logo SVG is parsed once into an [`SvgIcon`] in
//! `Authored` paint mode — aetna preserves the authored fills,
//! strokes, and gradients (radial + linear) at draw time, so the
//! mark renders with its real colors rather than tinted-monochrome
//! like the lucide icons. Filter primitives (Gaussian blur, drop
//! shadow) aren't part of aetna's vector pipeline; the chip
//! consequently reads flatter than the original SVG mockup, but the
//! geometry and gradients carry the identity.

use std::sync::LazyLock;

use aetna_core::SvgIcon;

static LOGO_SVG: &str = include_str!("../assets/logo.svg");

pub static LOGO: LazyLock<SvgIcon> =
    LazyLock::new(|| SvgIcon::parse(LOGO_SVG).expect("logo.svg parses as an SvgIcon"));
