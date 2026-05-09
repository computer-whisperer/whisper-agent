//! Lucide-shaped SVG icons bundled in-crate.
//!
//! Aetna's built-in [`aetna_core::IconName`] registry is small (17
//! entries) — `play`, `pause`, `square-pen`, `trash`, `zap` aren't
//! covered, and `icon_button("trash")` would silently render the
//! `AlertCircle` fallback. So whisper-agent's chrome carries its own
//! lucide-shaped SVGs and constructs [`SvgIcon`]s via
//! [`SvgIcon::parse_current_color`], which makes them stylable by the
//! element's `text_color` / `stroke_width` exactly like the built-ins.
//!
//! Path data is the canonical lucide.dev set (ISC license, free to
//! redistribute). When a future aetna upstream registers any of these
//! names, the corresponding `LazyLock` here can be deleted and call
//! sites switch from `ICON_PLAY.clone()` to `"play"` with no other
//! change.

use std::sync::LazyLock;

use aetna_core::SvgIcon;

/// Lightning bolt — used for a one-shot manual fire ("Run now").
/// Distinct from [`ICON_PLAY`] (which means "resume scheduled
/// execution") so two side-by-side buttons don't both read as `play`.
pub static ICON_ZAP: LazyLock<SvgIcon> =
    LazyLock::new(|| SvgIcon::parse_current_color(LUCIDE_ZAP).expect("lucide zap svg parses"));

/// Standard play triangle — used to resume a paused behavior.
pub static ICON_PLAY: LazyLock<SvgIcon> =
    LazyLock::new(|| SvgIcon::parse_current_color(LUCIDE_PLAY).expect("lucide play svg parses"));

/// Two parallel bars — used to pause a running behavior.
pub static ICON_PAUSE: LazyLock<SvgIcon> =
    LazyLock::new(|| SvgIcon::parse_current_color(LUCIDE_PAUSE).expect("lucide pause svg parses"));

/// Pencil-on-square — Lucide's standard "edit" affordance.
pub static ICON_SQUARE_PEN: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(LUCIDE_SQUARE_PEN).expect("lucide square-pen svg parses")
});

/// Trash can — destructive delete affordance.
pub static ICON_TRASH: LazyLock<SvgIcon> =
    LazyLock::new(|| SvgIcon::parse_current_color(LUCIDE_TRASH).expect("lucide trash svg parses"));

const LUCIDE_ZAP: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/></svg>"##;

const LUCIDE_PLAY: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 3 20 12 6 21 6 3"/></svg>"##;

const LUCIDE_PAUSE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/></svg>"##;

const LUCIDE_SQUARE_PEN: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z"/></svg>"##;

const LUCIDE_TRASH: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>"##;
