//! "+ New behavior" dialog. Two fields: directory-friendly id +
//! display name. The new behavior starts as a Manual-trigger stub
//! with an empty prompt; the editor opens automatically on the
//! `BehaviorCreated` round-trip so the user can fill in the rest.

use std::collections::HashMap;

use egui::{Color32, RichText, TextEdit};
use whisper_agent_protocol::{
    BehaviorConfig, BehaviorSummary, BehaviorThreadOverride, RetentionPolicy, TriggerSpec,
};

use super::super::NewBehaviorModalState;
use super::super::editor_render::hint;
use super::super::validate_behavior_id_client;

/// Confirm-side action for the new-behavior dialog.
pub(crate) enum NewBehaviorEvent {
    /// User clicked Create on a non-empty, non-duplicate id. Caller
    /// mints a correlation id, stamps `pending_correlation` on the
    /// (still-open) modal so the next frame renders the "creating…"
    /// label, and dispatches `ClientToServer::CreateBehavior` with
    /// the prepared `config`.
    Created {
        pod_id: String,
        behavior_id: String,
        config: BehaviorConfig,
    },
}

pub(crate) fn render_new_behavior_modal(
    ctx: &egui::Context,
    slot: &mut Option<NewBehaviorModalState>,
    behaviors_by_pod: &HashMap<String, Vec<BehaviorSummary>>,
) -> Option<NewBehaviorEvent> {
    let mut modal = slot.take()?;
    let mut open = true;
    let mut create_clicked = false;
    let mut cancel_clicked = false;
    let saving = modal.pending_correlation.is_some();

    egui::Window::new(format!("New behavior — {}", modal.pod_id))
        .collapsible(false)
        .resizable(false)
        .default_width(420.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("behavior_id");
                ui.add(
                    TextEdit::singleline(&mut modal.behavior_id)
                        .hint_text("directory name (e.g. 'daily-ci-check')")
                        .desired_width(f32::INFINITY),
                );
            });
            hint(
                ui,
                "Becomes the behavior's directory name under the pod; \
                 immutable after creation.",
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
            hint(
                ui,
                "Starts as a manually-triggered behavior with an empty \
                 prompt. Edit in the full editor to add a trigger, \
                 override thread settings, or write the prompt.",
            );
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                let enabled = !modal.behavior_id.trim().is_empty()
                    && !modal.name.trim().is_empty()
                    && !saving;
                if ui
                    .add_enabled(enabled, egui::Button::new("Create"))
                    .clicked()
                {
                    create_clicked = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel_clicked = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if saving {
                        ui.label(
                            RichText::new("creating…")
                                .italics()
                                .color(Color32::from_gray(160)),
                        );
                    }
                });
            });
        });

    if create_clicked {
        let behavior_id = modal.behavior_id.trim().to_string();
        if let Err(msg) = validate_behavior_id_client(&behavior_id) {
            modal.error = Some(msg.to_string());
            *slot = Some(modal);
            return None;
        }
        let exists = behaviors_by_pod
            .get(&modal.pod_id)
            .map(|list| list.iter().any(|b| b.behavior_id == behavior_id))
            .unwrap_or(false);
        if exists {
            modal.error = Some(format!("behavior `{behavior_id}` already exists"));
            *slot = Some(modal);
            return None;
        }
        let config = BehaviorConfig {
            name: modal.name.trim().to_string(),
            description: None,
            trigger: TriggerSpec::Manual,
            thread: BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
            scope: Default::default(),
        };
        let pod_id = modal.pod_id.clone();
        modal.error = None;
        // Restore the modal so the parent can stamp `pending_correlation`
        // on it after minting the correlation id; the next frame draws
        // the "creating…" label and the response handler closes the
        // modal on `BehaviorCreated`.
        *slot = Some(modal);
        return Some(NewBehaviorEvent::Created {
            pod_id,
            behavior_id,
            config,
        });
    }
    if cancel_clicked || !open {
        // Modal closes.
        return None;
    }
    *slot = Some(modal);
    None
}
