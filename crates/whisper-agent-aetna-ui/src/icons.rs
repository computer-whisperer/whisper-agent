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

/// Paperclip — compose attach affordance.
pub static ICON_PAPERCLIP: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(LUCIDE_PAPERCLIP).expect("lucide paperclip svg parses")
});

/// X mark — small "remove" affordance on staged thumbnails.
pub static ICON_X: LazyLock<SvgIcon> =
    LazyLock::new(|| SvgIcon::parse_current_color(LUCIDE_X).expect("lucide x svg parses"));

/// Speech-bubble — empty-state mark for "no threads / no messages".
pub static ICON_MESSAGE_SQUARE: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(LUCIDE_MESSAGE_SQUARE).expect("lucide message-square svg parses")
});

/// Inbox — empty-state mark for "no behaviors yet".
pub static ICON_INBOX: LazyLock<SvgIcon> =
    LazyLock::new(|| SvgIcon::parse_current_color(LUCIDE_INBOX).expect("lucide inbox svg parses"));

/// Network plug — server status icon shown in the sidebar footer.
pub static ICON_SERVER: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(LUCIDE_SERVER).expect("lucide server svg parses")
});

/// Corner-down-left arrow — used in the compose-row keyboard hint to
/// indicate the Enter / Return key.
pub static ICON_RETURN: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(LUCIDE_CORNER_DOWN_LEFT)
        .expect("lucide corner-down-left svg parses")
});

/// Stacked-cylinder database — sidebar-footer affordance opening the
/// knowledge-buckets modal. Lucide doesn't ship `database` in aetna's
/// built-in registry yet; bundled inline like the rest.
pub static ICON_DATABASE: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(LUCIDE_DATABASE).expect("lucide database svg parses")
});

/// Speedometer gauge — header trigger for the per-backend usage /
/// quota dropdown. Distinct visual cue from the model chip's text,
/// so the user reads it as "load meter" rather than just another
/// piece of identity chrome.
pub static ICON_GAUGE: LazyLock<SvgIcon> =
    LazyLock::new(|| SvgIcon::parse_current_color(LUCIDE_GAUGE).expect("lucide gauge svg parses"));

/// Circular refresh arrows — manual "refresh now" button inside the
/// usage dropdown. Lucide's `refresh-cw` (clockwise) shape.
pub static ICON_REFRESH_CW: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(LUCIDE_REFRESH_CW).expect("lucide refresh-cw svg parses")
});

/// Outward diagonal arrows — opens a section's body in the focused
/// text-lightbox modal. Lucide's `maximize-2` shape.
pub static ICON_MAXIMIZE_2: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(LUCIDE_MAXIMIZE_2).expect("lucide maximize-2 svg parses")
});

const LUCIDE_ZAP: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/></svg>"##;

const LUCIDE_PLAY: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 3 20 12 6 21 6 3"/></svg>"##;

const LUCIDE_PAUSE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/></svg>"##;

const LUCIDE_SQUARE_PEN: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z"/></svg>"##;

const LUCIDE_TRASH: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>"##;

const LUCIDE_PAPERCLIP: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13.234 20.252 21 12.3"/><path d="m16 6-8.414 8.586a2 2 0 0 0 0 2.828 2 2 0 0 0 2.828 0l8.414-8.586a4 4 0 0 0 0-5.656 4 4 0 0 0-5.656 0l-8.415 8.585a6 6 0 1 0 8.486 8.486"/></svg>"##;

const LUCIDE_X: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>"##;

const LUCIDE_MESSAGE_SQUARE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>"##;

const LUCIDE_INBOX: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg>"##;

const LUCIDE_SERVER: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/></svg>"##;

const LUCIDE_CORNER_DOWN_LEFT: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 10 4 15 9 20"/><path d="M20 4v7a4 4 0 0 1-4 4H4"/></svg>"##;

const LUCIDE_DATABASE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>"##;

const LUCIDE_GAUGE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>"##;

const LUCIDE_REFRESH_CW: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/></svg>"##;

const LUCIDE_MAXIMIZE_2: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" x2="14" y1="3" y2="10"/><line x1="3" x2="10" y1="21" y2="14"/></svg>"##;
