//! Add / edit dialog for a single host-env provider catalog entry.
//! Mode-driven: Add allows editing the name, Edit makes it
//! immutable (pod bindings reference providers by name). On Save
//! the form emits a typed event the caller dispatches as either
//! `AddHostEnvProvider` or `UpdateHostEnvProvider`.

use egui::{Color32, RichText, TextEdit};

use super::super::{ProviderEditorModalState, ProviderEditorMode};

/// Save-side action for the provider-editor dialog. Two variants
/// rather than a single one with a `mode` field so the caller
/// reduces straight into the matching wire message without
/// re-matching on the mode.
pub(crate) enum ProviderEditorEvent {
    /// User clicked Save in Add mode. Caller mints a correlation,
    /// stamps `pending_correlation` on the (still-open) modal, and
    /// dispatches `ClientToServer::AddHostEnvProvider`.
    AddRequested {
        name: String,
        url: String,
        token: Option<String>,
    },
    /// User clicked Save in Edit mode. Same handling, but the wire
    /// message is `ClientToServer::UpdateHostEnvProvider`.
    UpdateRequested {
        name: String,
        url: String,
        token: Option<String>,
    },
}

pub(crate) fn render_provider_editor_modal(
    ctx: &egui::Context,
    slot: &mut Option<ProviderEditorModalState>,
) -> Option<ProviderEditorEvent> {
    let mut modal = slot.take()?;
    let mut open = true;
    let mut save_clicked = false;
    let mut cancel_clicked = false;
    let saving = modal.pending_correlation.is_some();
    let title = match modal.mode {
        ProviderEditorMode::Add => "Add host-env provider",
        ProviderEditorMode::Edit => "Edit host-env provider",
    };

    egui::Window::new(title)
        .collapsible(false)
        .resizable(false)
        .default_width(460.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("name");
                let name_edit = TextEdit::singleline(&mut modal.name)
                    .hint_text("catalog name (e.g. 'landlock-laptop')")
                    .desired_width(f32::INFINITY);
                // Edit mode: name is immutable — pod bindings
                // reference providers by name, so renaming would
                // dangle them. Rebuild by removing and re-adding
                // if truly needed.
                let editable = modal.mode == ProviderEditorMode::Add;
                ui.add_enabled(editable, name_edit);
            });
            if modal.mode == ProviderEditorMode::Edit {
                ui.label(
                    RichText::new(
                        "Name is fixed once set — pod bindings reference it. \
                         Remove + re-add to rename.",
                    )
                    .small()
                    .color(Color32::from_gray(150)),
                );
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label("url");
                ui.add(
                    TextEdit::singleline(&mut modal.url)
                        .hint_text("http://host:port")
                        .desired_width(f32::INFINITY),
                );
            });
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label("token");
                ui.add(
                    TextEdit::singleline(&mut modal.token)
                        .password(true)
                        .hint_text(if modal.had_token {
                            "leave blank to clear existing token"
                        } else {
                            "control-plane bearer (optional)"
                        })
                        .desired_width(f32::INFINITY),
                );
            });
            ui.label(
                RichText::new(
                    "The token must match the daemon's --control-token-file \
                     (or leave blank for a --no-auth dev daemon).",
                )
                .small()
                .color(Color32::from_gray(150)),
            );
            if let Some(err) = &modal.error {
                ui.add_space(6.0);
                ui.colored_label(Color32::from_rgb(220, 80, 80), err);
            }
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                let save_enabled =
                    !saving && !modal.name.trim().is_empty() && !modal.url.trim().is_empty();
                let label = if saving { "Saving…" } else { "Save" };
                if ui
                    .add_enabled(save_enabled, egui::Button::new(label))
                    .clicked()
                {
                    save_clicked = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel_clicked = true;
                }
            });
        });

    if save_clicked {
        modal.error = None;
        let name = modal.name.trim().to_string();
        let url = modal.url.trim().to_string();
        let token_raw = modal.token.trim().to_string();
        let token = (!token_raw.is_empty()).then_some(token_raw);
        let event = match modal.mode {
            ProviderEditorMode::Add => ProviderEditorEvent::AddRequested { name, url, token },
            ProviderEditorMode::Edit => ProviderEditorEvent::UpdateRequested { name, url, token },
        };
        // Keep the modal open; it closes on the matching response
        // (HostEnvProviderAdded / HostEnvProviderUpdated) or surfaces
        // the server's error via the Error handler.
        *slot = Some(modal);
        return Some(event);
    }
    if cancel_clicked || !open {
        // Modal closes.
        return None;
    }
    *slot = Some(modal);
    None
}
