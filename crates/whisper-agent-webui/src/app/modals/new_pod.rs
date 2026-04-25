//! "+ New pod" modal — pick an id + display name; the new pod
//! inherits the server-default pod's template (backends, shared
//! MCPs, host envs).

use std::collections::HashMap;

use egui::{Color32, RichText, TextEdit};
use whisper_agent_protocol::PodSummary;

use super::super::NewPodModalState;
use super::super::validate_pod_id_client;

/// Confirm-side actions a `render_new_pod_modal` call can emit.
pub(crate) enum NewPodEvent {
    /// User clicked Create on a non-empty, non-duplicate id. Caller
    /// derives the `PodConfig` from its server-default template
    /// (which the renderer doesn't have access to) and dispatches
    /// `ClientToServer::CreatePod`. The modal has already been
    /// dropped; `PodCreated` will populate `self.pods` on the
    /// round-trip.
    Created { pod_id: String, name: String },
}

pub(crate) fn render_new_pod_modal(
    ctx: &egui::Context,
    slot: &mut Option<NewPodModalState>,
    pods: &HashMap<String, PodSummary>,
    backends_empty: bool,
) -> Option<NewPodEvent> {
    let mut modal = slot.take()?;
    let mut open = true;
    let mut create_clicked = false;
    let mut cancel_clicked = false;

    egui::Window::new("New pod")
        .collapsible(false)
        .resizable(false)
        .default_width(420.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("pod_id");
                ui.add(
                    TextEdit::singleline(&mut modal.pod_id)
                        .hint_text("directory name (e.g. 'whisper-dev')")
                        .desired_width(f32::INFINITY),
                );
            });
            ui.label(
                RichText::new(
                    "Becomes the pod's directory name on disk; immutable after \
                     creation. Letters, numbers, dashes, underscores.",
                )
                .small()
                .color(Color32::from_gray(160)),
            );
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label("name");
                ui.add(
                    TextEdit::singleline(&mut modal.name)
                        .hint_text("display name (free text)")
                        .desired_width(f32::INFINITY),
                );
            });
            if let Some(err) = &modal.error {
                ui.add_space(6.0);
                ui.colored_label(Color32::from_rgb(220, 80, 80), err);
            }
            ui.add_space(8.0);
            ui.label(
                RichText::new(
                    "The new pod inherits the server default pod's template \
                     (backends, shared MCPs, host envs). Use the per-pod Edit \
                     button to change any of these afterwards.",
                )
                .small()
                .color(Color32::from_gray(160)),
            );
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                let create_enabled = !modal.pod_id.trim().is_empty()
                    && !modal.name.trim().is_empty()
                    && !backends_empty;
                if ui
                    .add_enabled(create_enabled, egui::Button::new("Create"))
                    .clicked()
                {
                    create_clicked = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel_clicked = true;
                }
            });
        });

    if create_clicked {
        let pod_id = modal.pod_id.trim().to_string();
        if let Err(msg) = validate_pod_id_client(&pod_id) {
            modal.error = Some(msg.to_string());
            *slot = Some(modal);
            return None;
        }
        if pods.contains_key(&pod_id) {
            modal.error = Some(format!("pod `{pod_id}` already exists"));
            *slot = Some(modal);
            return None;
        }
        let name = modal.name.trim().to_string();
        // Modal closes (slot already empty).
        return Some(NewPodEvent::Created { pod_id, name });
    }
    if cancel_clicked || !open {
        // Modal closes.
        return None;
    }
    *slot = Some(modal);
    None
}
