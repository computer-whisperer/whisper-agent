//! Read-only and edit-with-save viewer modals.
//!
//! - `render_image_lightbox_modal` — reads its target URI from egui
//!   memory keyed by `chat_render::ENLARGED_IMAGE_KEY`, so it's
//!   openable from anywhere via `ctx.memory_mut(...)` without a
//!   dedicated slot in `ChatApp`.
//! - `render_json_viewer_modal` — read-only JSON tree viewer; the
//!   parent owns an `Option<JsonViewerModalState>` which the renderer
//!   clears on close.
//! - `render_file_viewer_modal` — edit-with-save viewer for editable
//!   pod files. Returns a `FileViewerEvent` for the caller to mint a
//!   correlation id and dispatch the `WritePodFile` wire message; the
//!   renderer never touches `ChatApp`.

use egui::{Color32, RichText, TextEdit};

use super::super::chat_render;
use super::super::{FileViewerModalState, JsonViewerModalState, render_json_node};

/// Centered, click-to-dismiss preview of a tool-result attachment.
pub(crate) fn render_image_lightbox_modal(ctx: &egui::Context) {
    let key = egui::Id::new(chat_render::ENLARGED_IMAGE_KEY);
    let uri: Option<String> = ctx.memory(|mem| mem.data.get_temp::<String>(key));
    let Some(uri) = uri else { return };

    let screen = ctx.content_rect();
    let max_h = (screen.height() - 80.0).max(240.0);
    let max_w = (screen.width() - 80.0).max(320.0);
    let mut open = true;
    let mut dismiss = ctx.input(|i| i.key_pressed(egui::Key::Escape));

    egui::Window::new("attachment")
        .collapsible(false)
        .resizable(false)
        .title_bar(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .max_width(max_w)
        .max_height(max_h)
        .open(&mut open)
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button(RichText::new("✕").strong())
                            .on_hover_text("Close (Esc)")
                            .clicked()
                        {
                            dismiss = true;
                        }
                    });
                });
                ui.add(
                    egui::Image::new(&uri)
                        .max_size(egui::vec2(max_w - 24.0, max_h - 60.0))
                        .fit_to_exact_size(egui::vec2(max_w - 24.0, max_h - 60.0)),
                );
            });
        });

    if dismiss || !open {
        ctx.memory_mut(|mem| mem.data.remove::<String>(key));
    }
}

/// Render the read-only JSON tree viewer. Scalars render as
/// `key: value` one-liners; objects and arrays render as
/// collapsible headers with their sizes in the label, default-open
/// at the root and default-closed deeper. Strings are shown
/// in-line with a preview and a hover tooltip carrying the full
/// text, so long message content doesn't blow out the row height.
pub(crate) fn render_json_viewer_modal(
    ctx: &egui::Context,
    slot: &mut Option<JsonViewerModalState>,
) {
    let Some(modal) = slot.take() else {
        return;
    };

    let title = format!("{} — {}", modal.path, modal.pod_id);
    let screen = ctx.content_rect();
    let max_h = (screen.height() - 60.0).max(280.0);
    let max_w = (screen.width() - 60.0).max(420.0);
    let mut open = true;
    let mut close_clicked = false;

    egui::Window::new(title)
        .collapsible(false)
        .resizable(true)
        .default_width(720.0_f32.min(max_w))
        .default_height(560.0_f32.min(max_h))
        .max_width(max_w)
        .max_height(max_h)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            egui::Panel::bottom("json_viewer_footer").show_inside(ui, |ui| {
                ui.add_space(6.0);
                if let Some(err) = modal.error.as_deref() {
                    ui.colored_label(Color32::from_rgb(220, 80, 80), err);
                    ui.add_space(4.0);
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("read-only JSON viewer")
                            .small()
                            .color(Color32::from_gray(160)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Close").clicked() {
                            close_clicked = true;
                        }
                    });
                });
            });
            egui::CentralPanel::default().show_inside(ui, |ui| {
                if let Some(value) = modal.parsed.as_ref() {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            render_json_node(ui, "$", "(root)", value, 0);
                        });
                } else if modal.error.is_none() {
                    ui.add_space(24.0);
                    ui.label(
                        RichText::new("loading…")
                            .italics()
                            .color(Color32::from_gray(160)),
                    );
                }
            });
        });

    if close_clicked || !open {
        return;
    }
    *slot = Some(modal);
}

/// Side-channel actions a `render_file_viewer_modal` call can emit.
pub(crate) enum FileViewerEvent {
    /// Save clicked while the buffer was dirty. Caller mints a
    /// correlation id, sends the `WritePodFile` wire message, and
    /// stamps `pending_correlation` on the (still-open) modal so the
    /// next frame renders the "Saving…" footer.
    SaveRequested {
        pod_id: String,
        path: String,
        content: String,
    },
}

/// Edit-with-save viewer for editable pod files. The buffer rides on
/// `working` (cloned from `baseline` on load); Save fires only when
/// dirty, Revert restores the baseline, Close drops the modal.
pub(crate) fn render_file_viewer_modal(
    ctx: &egui::Context,
    slot: &mut Option<FileViewerModalState>,
) -> Option<FileViewerEvent> {
    let mut modal = slot.take()?;

    let mut open = true;
    let mut save_clicked = false;
    let mut revert_clicked = false;
    let mut close_clicked = false;

    let title = format!("{} — {}", modal.path, modal.pod_id);
    let screen = ctx.content_rect();
    let max_h = (screen.height() - 60.0).max(280.0);
    let max_w = (screen.width() - 60.0).max(420.0);
    let dirty = modal.is_dirty();
    let has_data = modal.working.is_some();
    // "saving" iff we have content and a correlation is in flight
    // (a correlation-in-flight with no content yet = pending read).
    let saving = modal.pending_correlation.is_some() && has_data;

    egui::Window::new(title)
        .collapsible(false)
        .resizable(true)
        .default_width(720.0_f32.min(max_w))
        .default_height(560.0_f32.min(max_h))
        .max_width(max_w)
        .max_height(max_h)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            egui::Panel::bottom("file_viewer_footer").show_inside(ui, |ui| {
                if modal.readonly {
                    ui.add_space(6.0);
                    if let Some(err) = modal.error.as_deref() {
                        ui.colored_label(Color32::from_rgb(220, 80, 80), err);
                        ui.add_space(4.0);
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("read-only — runtime state owned by the scheduler")
                                .small()
                                .color(Color32::from_gray(160)),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Close").clicked() {
                                close_clicked = true;
                            }
                        });
                    });
                } else {
                    let actions = crate::editor::render_footer(
                        ui,
                        modal.error.as_deref(),
                        has_data,
                        dirty,
                        saving,
                    );
                    save_clicked = actions.save;
                    revert_clicked = actions.revert;
                    close_clicked = actions.close;
                }
            });
            egui::CentralPanel::default().show_inside(ui, |ui| {
                let Some(working) = modal.working.as_mut() else {
                    ui.add_space(24.0);
                    if modal.error.is_none() {
                        ui.label(
                            RichText::new("loading…")
                                .italics()
                                .color(Color32::from_gray(160)),
                        );
                    }
                    return;
                };
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let mut edit = TextEdit::multiline(working)
                            .code_editor()
                            .desired_width(ui.available_width());
                        if modal.readonly {
                            edit = edit.interactive(false);
                        }
                        ui.add_sized(
                            [ui.available_width(), ui.available_height().max(200.0)],
                            edit,
                        );
                    });
            });
        });

    if revert_clicked && let Some(b) = modal.baseline.clone() {
        modal.working = Some(b);
        modal.error = None;
    }

    let mut event = None;
    if save_clicked
        && modal.is_dirty()
        && let Some(working) = modal.working.clone()
    {
        // Caller mints the correlation id and stamps
        // `pending_correlation` on the modal in the event reducer; we
        // can't do it here without a borrow on `ChatApp`. The error
        // slot clears now so any prior failure banner doesn't linger
        // for one frame after the save.
        modal.error = None;
        event = Some(FileViewerEvent::SaveRequested {
            pod_id: modal.pod_id.clone(),
            path: modal.path.clone(),
            content: working,
        });
    }

    if close_clicked || !open {
        // Slot was already taken; dropping = closed.
        return event;
    }
    *slot = Some(modal);
    event
}
