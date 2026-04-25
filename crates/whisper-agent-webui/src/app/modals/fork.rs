//! "Fork from this message" confirm dialog.

use egui::{Color32, RichText};

use super::super::ForkModalState;

/// Confirm-side actions a `render_fork_modal` call can emit.
pub(crate) enum ForkEvent {
    /// User clicked Fork. Caller mints a correlation id, stamps
    /// `pending_fork_seed` so the resulting `ThreadCreated` round-trip
    /// can hand the seed text to the new thread, and dispatches
    /// `ClientToServer::ForkThread`.
    Confirmed {
        thread_id: String,
        from_message_index: usize,
        archive_original: bool,
        reset_capabilities: bool,
        seed_text: String,
    },
}

pub(crate) fn render_fork_modal(
    ctx: &egui::Context,
    slot: &mut Option<ForkModalState>,
) -> Option<ForkEvent> {
    let mut modal = slot.take()?;
    let mut open = true;
    let mut confirm_clicked = false;
    let mut cancel_clicked = false;

    egui::Window::new("Fork from this message")
        .collapsible(false)
        .resizable(false)
        .default_width(380.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.label(
                RichText::new(
                    "Forks this thread at the selected user message. The new \
                     thread shares the pod, bindings, config, and tool allowlist, \
                     and starts with the conversation up to (but not including) \
                     that message — ready for you to retype the prompt.",
                )
                .color(Color32::from_gray(190))
                .small(),
            );
            ui.add_space(8.0);
            ui.checkbox(&mut modal.archive_original, "Archive the original thread");
            ui.add_space(4.0);
            ui.label(
                RichText::new(
                    "Archived threads drop off the sidebar list but stay on disk; \
                     they're still readable from the server's pod directory.",
                )
                .color(Color32::from_gray(140))
                .small(),
            );
            ui.add_space(8.0);
            ui.checkbox(
                &mut modal.reset_capabilities,
                "Reset capabilities to pod defaults",
            );
            ui.add_space(4.0);
            ui.label(
                RichText::new(
                    "Unchecked: new thread inherits the source's live bindings, \
                     scope, and config. Checked: re-derive from the pod's current \
                     defaults — use this to pick up newly-added MCP hosts, sandbox \
                     bindings, or cap changes made since the source thread was \
                     created.",
                )
                .color(Color32::from_gray(140))
                .small(),
            );
            ui.add_space(10.0);
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Fork").clicked() {
                    confirm_clicked = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel_clicked = true;
                }
            });
        });

    if confirm_clicked {
        // Modal closes; caller dispatches the wire message.
        return Some(ForkEvent::Confirmed {
            thread_id: modal.thread_id,
            from_message_index: modal.from_message_index,
            archive_original: modal.archive_original,
            reset_capabilities: modal.reset_capabilities,
            seed_text: modal.seed_text,
        });
    }
    if cancel_clicked || !open {
        // Dropped.
        return None;
    }
    *slot = Some(modal);
    None
}
