//! Per-pod editor — structured tabs (Allow / Defaults / Limits) plus a
//! Raw TOML escape hatch. Hosts a per-sandbox-entry sub-modal opened
//! from the Allow tab. Save serializes the working `PodConfig` (or
//! the raw buffer if the Raw tab is dirty) and ships it through
//! `UpdatePodConfig`.

use std::collections::HashMap;

use egui::{Color32, RichText};
use whisper_agent_protocol::{
    BackendSummary, HostEnvProviderInfo, ModelSummary, PodConfig, ResourceSnapshot,
};

use super::super::editor_render::{
    render_pod_editor_allow_tab, render_pod_editor_defaults_tab, render_pod_editor_limits_tab,
    render_pod_editor_raw_tab, render_sandbox_entry_modal,
};
use super::super::{PodEditorModalState, PodEditorTab, SandboxEntryEditorState};

/// Effects a `render_pod_editor_modal` call can emit. Like the
/// behavior editor, the renderer collects each frame's effects into
/// a Vec; the caller iterates and dispatches.
#[allow(clippy::large_enum_variant)]
pub(crate) enum PodEditorEvent {
    /// Currently-effective backend for the Defaults tab's model
    /// picker. Caller passes to `request_models_for` (dedup-guarded).
    RequestModels(String),
    /// Save clicked. Caller mints a correlation, stamps it on the
    /// (still-open) modal, and dispatches `UpdatePodConfig`. The
    /// renderer has already chosen between serializing `working` and
    /// using the raw buffer based on tab + raw_dirty.
    SaveRequested { pod_id: String, toml_text: String },
    /// Revert clicked but no `server_baseline` was cached. Caller
    /// dispatches `GetPod` to fetch a fresh snapshot; the renderer
    /// has already cleared `working` so the next frame shows the
    /// "loading…" placeholder.
    RefreshRequested { pod_id: String },
}

pub(crate) fn render_pod_editor_modal(
    ctx: &egui::Context,
    slot: &mut Option<PodEditorModalState>,
    backends: &[BackendSummary],
    resources: &HashMap<String, ResourceSnapshot>,
    host_env_providers: &[HostEnvProviderInfo],
    models_by_backend: &HashMap<String, Vec<ModelSummary>>,
    buckets: &[whisper_agent_protocol::BucketSummary],
) -> Vec<PodEditorEvent> {
    let mut events = Vec::new();
    let Some(mut modal) = slot.take() else {
        return events;
    };

    // Snapshot the catalogs the form needs into owned data so the
    // inner closures don't have to borrow caller state. These are
    // small (single-digit lists in practice) so the clone is cheap.
    let backend_catalog: Vec<String> = backends.iter().map(|b| b.name.clone()).collect();
    let shared_mcp_catalog: Vec<String> = resources
        .values()
        .filter_map(|r| match r {
            ResourceSnapshot::McpHost {
                label,
                per_task: false,
                ..
            } => Some(label.clone()),
            _ => None,
        })
        .collect();
    let host_env_providers_owned: Vec<HostEnvProviderInfo> = host_env_providers.to_vec();
    let bucket_catalog: Vec<String> = buckets.iter().map(|b| b.id.clone()).collect();
    // Fire ListModels for whichever backend the model combo is about
    // to show against. Dedup-guarded by the caller.
    if let Some(w) = modal.working.as_ref()
        && !w.thread_defaults.backend.is_empty()
    {
        events.push(PodEditorEvent::RequestModels(
            w.thread_defaults.backend.clone(),
        ));
    }

    let mut open = true;
    let mut save_clicked = false;
    let mut cancel_clicked = false;
    let mut revert_clicked = false;
    let mut switch_to: Option<PodEditorTab> = None;
    let mut sandbox_entry_open: Option<SandboxEntryEditorState> = None;
    let mut sandbox_entry_delete: Option<usize> = None;

    let title = format!("Edit pod — {}", modal.pod_id);
    let screen = ctx.content_rect();
    let max_h = (screen.height() - 60.0).max(280.0);
    let max_w = (screen.width() - 60.0).max(420.0);
    let dirty = modal.is_dirty();
    let saving = modal.pending_correlation.is_some();
    let sub_modal_open = modal.sandbox_entry_editor.is_some();

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
            // Disable the parent while the per-sandbox sub-modal is
            // up so clicks on the parent don't interleave with the
            // sub-modal's edits.
            ui.add_enabled_ui(!sub_modal_open, |ui| {
                egui::Panel::bottom("pod_editor_footer").show_inside(ui, |ui| {
                    let actions = crate::editor::render_footer(
                        ui,
                        modal.error.as_deref(),
                        modal.working.is_some(),
                        dirty,
                        saving,
                    );
                    save_clicked = actions.save;
                    revert_clicked = actions.revert;
                    cancel_clicked = actions.close;
                });
                egui::Panel::top("pod_editor_tabs").show_inside(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        for tab in [
                            PodEditorTab::Allow,
                            PodEditorTab::Defaults,
                            PodEditorTab::Limits,
                            PodEditorTab::RawToml,
                        ] {
                            let active = modal.tab == tab;
                            let label = if active {
                                RichText::new(tab.label()).strong()
                            } else {
                                RichText::new(tab.label()).color(Color32::from_gray(170))
                            };
                            if ui.selectable_label(active, label).clicked() && !active {
                                switch_to = Some(tab);
                            }
                        }
                    });
                    ui.add_space(2.0);
                    ui.separator();
                });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    let Some(working) = modal.working.as_mut() else {
                        ui.add_space(24.0);
                        ui.label(
                            RichText::new("loading pod config…")
                                .italics()
                                .color(Color32::from_gray(160)),
                        );
                        return;
                    };
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| match modal.tab {
                            PodEditorTab::Allow => {
                                render_pod_editor_allow_tab(
                                    ui,
                                    working,
                                    &backend_catalog,
                                    &shared_mcp_catalog,
                                    &host_env_providers_owned,
                                    &bucket_catalog,
                                    &mut sandbox_entry_open,
                                    &mut sandbox_entry_delete,
                                );
                            }
                            PodEditorTab::Defaults => {
                                render_pod_editor_defaults_tab(
                                    ui,
                                    working,
                                    &backend_catalog,
                                    models_by_backend,
                                );
                            }
                            PodEditorTab::Limits => {
                                render_pod_editor_limits_tab(ui, working);
                            }
                            PodEditorTab::RawToml => {
                                render_pod_editor_raw_tab(
                                    ui,
                                    &mut modal.raw_buffer,
                                    &mut modal.raw_dirty,
                                );
                            }
                        });
                });
            });
        });

    // Sub-modal: per-sandbox-entry editor. Rendered after the parent
    // so it's drawn above. The parent above is wrapped in an
    // `add_enabled_ui(!sub_modal_open, ...)` so clicks pass through
    // visually but don't interact while this is up.
    if let Some(mut sub) = modal.sandbox_entry_editor.take() {
        let mut sub_open = true;
        let mut sub_save = false;
        let mut sub_cancel = false;
        render_sandbox_entry_modal(
            ctx,
            &mut sub,
            &mut sub_open,
            &mut sub_save,
            &mut sub_cancel,
            host_env_providers,
        );
        if sub_save {
            if sub.entry.name.trim().is_empty() {
                sub.error = Some("name is required".into());
                modal.sandbox_entry_editor = Some(sub);
            } else if sub.entry.provider.trim().is_empty() {
                sub.error = Some("provider is required".into());
                modal.sandbox_entry_editor = Some(sub);
            } else if let Some(working) = modal.working.as_mut() {
                let name = sub.entry.name.trim().to_string();
                // Reject duplicate names within the same allow.host_env table.
                let dup = working
                    .allow
                    .host_env
                    .iter()
                    .enumerate()
                    .any(|(i, e)| e.name == name && Some(i) != sub.index);
                if dup {
                    sub.error = Some(format!("a host env named `{name}` already exists"));
                    modal.sandbox_entry_editor = Some(sub);
                } else {
                    sub.entry.name = name.clone();
                    match sub.index {
                        Some(i) if i < working.allow.host_env.len() => {
                            working.allow.host_env[i] = sub.entry;
                        }
                        _ => working.allow.host_env.push(sub.entry),
                    }
                    // Auto-pick the default for thread_defaults if
                    // it's still empty — otherwise the server's
                    // tightened validation rejects the save and the
                    // user gets a confusing inline error before
                    // they've even visited the Defaults tab.
                    if working.thread_defaults.host_env.is_empty() {
                        working.thread_defaults.host_env = vec![name];
                    }
                    modal.raw_dirty = false;
                }
            }
        } else if sub_cancel || !sub_open {
            // Sub-modal closes.
        } else {
            modal.sandbox_entry_editor = Some(sub);
        }
    }
    if let Some(idx) = sandbox_entry_delete
        && let Some(working) = modal.working.as_mut()
        && idx < working.allow.host_env.len()
    {
        // Also fix up thread_defaults.host_env: if it pointed at
        // the deleted entry, pick the first remaining one (or
        // empty when the allow list is now empty — the defaults
        // picker renders a read-only "(shared MCPs only)" in
        // that case). Leaves the form valid by construction
        // instead of relying on a server-side error to surface
        // a dangling reference.
        let removed = working.allow.host_env.remove(idx);
        // Drop the deleted entry from thread_defaults.host_env
        // (list-shape, may contain zero or more names). If that
        // empties the list and other allow entries remain, reseed
        // with the first surviving one so the form stays valid by
        // construction — mirrors the old single-value behavior.
        working
            .thread_defaults
            .host_env
            .retain(|n| n != &removed.name);
        if working.thread_defaults.host_env.is_empty()
            && let Some(fallback) = working.allow.host_env.first()
        {
            working.thread_defaults.host_env = vec![fallback.name.clone()];
        }
    }
    if let Some(sub) = sandbox_entry_open {
        modal.sandbox_entry_editor = Some(sub);
    }

    // Tab switch happens after the inner closure so we can do the
    // raw->structured reparse without holding any UI borrows.
    if let Some(target) = switch_to {
        let leaving_raw = modal.tab == PodEditorTab::RawToml && target != PodEditorTab::RawToml;
        let entering_raw = target == PodEditorTab::RawToml;
        match crate::editor::sync_on_tab_switch::<PodConfig>(
            leaving_raw,
            entering_raw,
            &mut modal.working,
            &mut modal.raw_buffer,
            &mut modal.raw_dirty,
        ) {
            Ok(()) => {
                modal.tab = target;
                modal.error = None;
            }
            Err(msg) => modal.error = Some(msg),
        }
    }

    if save_clicked && let Some(working) = &modal.working {
        let toml_text = if modal.tab == PodEditorTab::RawToml && modal.raw_dirty {
            modal.raw_buffer.clone()
        } else {
            match toml::to_string_pretty(working) {
                Ok(s) => s,
                Err(e) => {
                    modal.error = Some(format!("encode pod.toml: {e}"));
                    *slot = Some(modal);
                    return events;
                }
            }
        };
        modal.error = None;
        let pod_id = modal.pod_id.clone();
        // Keep the modal open; caller stamps `pending_correlation`.
        *slot = Some(modal);
        events.push(PodEditorEvent::SaveRequested { pod_id, toml_text });
        return events;
    }
    if revert_clicked {
        if let Some(baseline) = &modal.server_baseline {
            modal.working = Some(baseline.clone());
            modal.raw_buffer = toml::to_string_pretty(baseline).unwrap_or_default();
            modal.raw_dirty = false;
            modal.error = None;
            modal.pending_correlation = None;
            *slot = Some(modal);
        } else {
            let pod_id = modal.pod_id.clone();
            modal.working = None;
            modal.error = None;
            modal.pending_correlation = None;
            *slot = Some(modal);
            events.push(PodEditorEvent::RefreshRequested { pod_id });
        }
        return events;
    }
    if cancel_clicked || !open {
        // Modal closes (drop modal).
        return events;
    }
    *slot = Some(modal);
    events
}
