//! Extended font fallback stack.
//!
//! egui's `default_fonts` feature only ships Ubuntu-Light + Hack +
//! NotoEmoji + emoji-icon-font. Anything outside those ranges renders
//! as `◻` (U+25FB, epaint's hardcoded `PRIMARY_REPLACEMENT_CHAR`) —
//! so LLM output that strays beyond Latin/Greek/Cyrillic/common
//! punctuation (arrows, math operators, box drawing, CJK) becomes
//! tofu boxes.
//!
//! We extend the fallback chain with three Noto fonts — Symbols 2,
//! Math, and Sans CJK SC — installed after the defaults so the
//! existing UI typography stays unchanged and the extras only
//! participate when the primaries don't have the glyph.

use std::sync::Arc;

use egui::{FontData, FontDefinitions, FontFamily};

// `.ttf` / `.otf` bytes are embedded into the wasm binary. The CJK
// OTF is ~16 MB raw (~5 MB gzipped) and dominates the bundle — a
// deliberate trade-off for out-of-the-box CJK rendering.
static NOTO_SYMBOLS2: &[u8] = include_bytes!("../assets/fonts/NotoSansSymbols2-Regular.ttf");
static NOTO_MATH: &[u8] = include_bytes!("../assets/fonts/NotoSansMath-Regular.ttf");
static NOTO_CJK_SC: &[u8] = include_bytes!("../assets/fonts/NotoSansCJKsc-Regular.otf");

pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();

    for (name, bytes) in [
        ("NotoSansSymbols2", NOTO_SYMBOLS2),
        ("NotoSansMath", NOTO_MATH),
        ("NotoSansCJKsc", NOTO_CJK_SC),
    ] {
        fonts
            .font_data
            .insert(name.to_owned(), Arc::new(FontData::from_static(bytes)));
        // Append as lowest-priority fallback for both families so the
        // primary Ubuntu-Light / Hack glyphs still win when they exist.
        for family in [FontFamily::Proportional, FontFamily::Monospace] {
            if let Some(chain) = fonts.families.get_mut(&family) {
                chain.push(name.to_owned());
            }
        }
    }

    ctx.set_fonts(fonts);
}
