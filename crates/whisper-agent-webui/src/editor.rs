//! Shared scaffolding for the tabbed "structured form + raw TOML" editor
//! pattern used by pod and behavior edit modals.
//!
//! Each editor owns its own state struct (so per-editor fields like the
//! pod editor's `sandbox_entry_editor` sub-modal don't leak into every
//! editor) and its own per-tab renderers. The helpers here cover the
//! bits that genuinely repeat:
//!
//! - [`render_footer`] — the Save/Revert/Close bar with dirty indicator.
//! - [`render_raw_toml_tab`] — hint text + code-style textarea.
//! - [`sync_on_tab_switch`] — raw↔struct reparse/reformat on tab change.

use egui::{Color32, RichText, TextEdit, Ui};

/// Which footer button (if any) the user clicked this frame.
#[derive(Default, Debug, Clone, Copy)]
pub struct FooterActions {
    pub save: bool,
    pub revert: bool,
    pub close: bool,
}

/// Render the Save / Revert / Close footer with a dirty indicator and
/// optional inline error banner. Stateless — caller tracks `dirty` /
/// `saving` / `has_data` and reacts to the returned `FooterActions`.
///
/// `has_data` gates Save and Revert; both are disabled until the
/// initial snapshot lands. `dirty` gates Save specifically (nothing to
/// save when working matches baseline). `saving` indicates an
/// in-flight save — buttons disabled, "saving…" label shown.
pub fn render_footer(
    ui: &mut Ui,
    error: Option<&str>,
    has_data: bool,
    dirty: bool,
    saving: bool,
) -> FooterActions {
    let mut actions = FooterActions::default();
    ui.add_space(6.0);
    if let Some(err) = error {
        ui.colored_label(Color32::from_rgb(220, 80, 80), err);
        ui.add_space(4.0);
    }
    ui.separator();
    ui.horizontal(|ui| {
        let save_enabled = has_data && dirty && !saving;
        if ui
            .add_enabled(save_enabled, egui::Button::new("Save"))
            .clicked()
        {
            actions.save = true;
        }
        if ui
            .add_enabled(has_data && dirty && !saving, egui::Button::new("Revert"))
            .clicked()
        {
            actions.revert = true;
        }
        if ui.button("Close").clicked() {
            actions.close = true;
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if saving {
                ui.label(
                    RichText::new("saving…")
                        .italics()
                        .color(Color32::from_gray(160)),
                );
            } else if dirty {
                ui.label(
                    RichText::new("● unsaved changes")
                        .small()
                        .color(Color32::from_rgb(220, 170, 90)),
                );
            } else if has_data {
                ui.label(
                    RichText::new("✓ saved")
                        .small()
                        .color(Color32::from_gray(140)),
                );
            }
        });
    });
    actions
}

/// Render the Raw TOML tab: a hint paragraph above a code-style
/// multiline textarea that fills the available space. Any typing flips
/// `dirty` to true; the caller consumes that to know whether a tab
/// switch needs to reparse this buffer.
pub fn render_raw_toml_tab(ui: &mut Ui, hint_text: &str, raw: &mut String, dirty: &mut bool) {
    ui.add_space(4.0);
    ui.label(
        RichText::new(hint_text)
            .small()
            .color(Color32::from_gray(160)),
    );
    ui.add_space(4.0);
    let resp = ui.add_sized(
        [ui.available_width(), ui.available_height().max(180.0)],
        TextEdit::multiline(raw).code_editor(),
    );
    if resp.changed() {
        *dirty = true;
    }
}

/// Handle the raw-↔-structured sync at a tab boundary.
///
/// - `leaving_raw`: user was on the Raw tab this frame and is
///   switching away. If `raw_dirty`, reparse the text back into
///   `working`; a parse failure returns `Err(message)` and the
///   caller should refuse the switch + surface the error.
/// - `entering_raw`: user is switching *to* the Raw tab. Reformat
///   `working` into `raw_buffer` so the user sees current state,
///   *unless* `raw_dirty` is already true (preserving an in-flight
///   raw edit the user forgot to commit).
///
/// The two flags aren't mutually exclusive — switching from one
/// structured tab to another has both false and this is a cheap
/// no-op.
pub fn sync_on_tab_switch<T>(
    leaving_raw: bool,
    entering_raw: bool,
    working: &mut Option<T>,
    raw_buffer: &mut String,
    raw_dirty: &mut bool,
) -> Result<(), String>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    if leaving_raw && *raw_dirty {
        match toml::from_str::<T>(raw_buffer) {
            Ok(parsed) => {
                *working = Some(parsed);
                *raw_dirty = false;
            }
            Err(e) => {
                return Err(format!(
                    "raw TOML doesn't parse — fix it or click Revert: {e}"
                ));
            }
        }
    }
    if entering_raw
        && !*raw_dirty
        && let Some(w) = working.as_ref()
    {
        *raw_buffer = toml::to_string_pretty(w).unwrap_or_default();
    }
    Ok(())
}
