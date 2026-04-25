//! Per-behavior editor — structured tabs (Trigger / Thread / Scope /
//! Retention / Prompt) plus a Raw TOML escape hatch for the config.
//! Save ships the working `BehaviorConfig` + prompt through
//! `UpdateBehavior`.

use std::collections::HashMap;

use egui::{Color32, RichText};
use whisper_agent_protocol::{BackendSummary, BehaviorConfig, ModelSummary, PodConfig};

use super::super::editor_render::{
    render_behavior_editor_prompt_tab, render_behavior_editor_raw_tab,
    render_behavior_editor_retention_tab, render_behavior_editor_scope_tab,
    render_behavior_editor_thread_tab, render_behavior_editor_trigger_tab,
};
use super::super::{BehaviorEditorModalState, BehaviorEditorTab};

/// Effects a `render_behavior_editor_modal` call can emit. The
/// renderer collects each frame's effects into a single Vec; the
/// caller iterates and dispatches. The size disparity between
/// `RequestModels` (a String) and `SaveRequested` (a full
/// `BehaviorConfig`) is fine because each frame produces at most a
/// handful of events that are immediately destructured.
#[allow(clippy::large_enum_variant)]
pub(crate) enum BehaviorEditorEvent {
    /// The currently-effective backend for the model picker. Caller
    /// passes this to `request_models_for` (dedup-guarded internally
    /// so calling each frame is cheap).
    RequestModels(String),
    /// Save clicked. Caller mints a correlation, stamps it on the
    /// (still-open) modal, and dispatches `UpdateBehavior`.
    SaveRequested {
        pod_id: String,
        behavior_id: String,
        config: BehaviorConfig,
        prompt: String,
    },
}

pub(crate) fn render_behavior_editor_modal(
    ctx: &egui::Context,
    slot: &mut Option<BehaviorEditorModalState>,
    backends: &[BackendSummary],
    pod_configs: &HashMap<String, PodConfig>,
    models_by_backend: &HashMap<String, Vec<ModelSummary>>,
) -> Vec<BehaviorEditorEvent> {
    let mut events = Vec::new();
    let Some(mut modal) = slot.take() else {
        return events;
    };
    let backend_catalog: Vec<String> = backends.iter().map(|b| b.name.clone()).collect();
    let pod_backend_names: Vec<String> = pod_configs
        .get(&modal.pod_id)
        .map(|cfg| cfg.allow.backends.clone())
        .unwrap_or_default();
    let pod_host_env_names: Vec<String> = pod_configs
        .get(&modal.pod_id)
        .map(|cfg| cfg.allow.host_env.iter().map(|h| h.name.clone()).collect())
        .unwrap_or_default();
    let pod_mcp_host_names: Vec<String> = pod_configs
        .get(&modal.pod_id)
        .map(|cfg| cfg.allow.mcp_hosts.clone())
        .unwrap_or_default();
    // Pod's default backend — used as the "effective" backend when
    // the behavior doesn't override `bindings.backend`. The model
    // combo reads it to decide which catalog entry's model list to
    // show. Empty when the pod config hasn't landed yet; in that
    // case the combo renders a "(pick a backend first)" hint.
    let pod_default_backend: String = pod_configs
        .get(&modal.pod_id)
        .map(|cfg| cfg.thread_defaults.backend.clone())
        .unwrap_or_default();
    // Fire ListModels for whichever backend the model combo is
    // about to show against. Dedup-guarded by the caller.
    let effective_backend_for_models = modal
        .working_config
        .as_ref()
        .and_then(|c| c.thread.bindings.backend.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| pod_default_backend.clone());
    if !effective_backend_for_models.is_empty() {
        events.push(BehaviorEditorEvent::RequestModels(
            effective_backend_for_models,
        ));
    }

    let mut save_clicked = false;
    let mut revert_clicked = false;
    let mut close_clicked = false;
    let mut switch_to: Option<BehaviorEditorTab> = None;
    let mut open = true;

    let title = format!("Edit behavior — {}/{}", modal.pod_id, modal.behavior_id);
    let screen = ctx.content_rect();
    let max_h = (screen.height() - 60.0).max(280.0);
    let max_w = (screen.width() - 60.0).max(420.0);
    let dirty = modal.is_dirty();
    let saving = modal.pending_correlation.is_some();
    let has_data = modal.working_config.is_some();

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
            egui::Panel::bottom("behavior_editor_footer").show_inside(ui, |ui| {
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
            });
            egui::Panel::top("behavior_editor_tabs").show_inside(ui, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    for tab in [
                        BehaviorEditorTab::Trigger,
                        BehaviorEditorTab::Thread,
                        BehaviorEditorTab::Scope,
                        BehaviorEditorTab::Retention,
                        BehaviorEditorTab::Prompt,
                        BehaviorEditorTab::RawToml,
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
                if modal.working_config.is_none() && modal.error.is_none() {
                    ui.add_space(24.0);
                    ui.label(
                        RichText::new("loading behavior…")
                            .italics()
                            .color(Color32::from_gray(160)),
                    );
                    return;
                }
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| match modal.tab {
                        BehaviorEditorTab::Trigger => {
                            if let Some(cfg) = modal.working_config.as_mut() {
                                render_behavior_editor_trigger_tab(ui, cfg);
                            }
                        }
                        BehaviorEditorTab::Thread => {
                            if let Some(cfg) = modal.working_config.as_mut() {
                                render_behavior_editor_thread_tab(
                                    ui,
                                    cfg,
                                    &backend_catalog,
                                    &pod_host_env_names,
                                    &pod_mcp_host_names,
                                    models_by_backend,
                                    &pod_default_backend,
                                );
                            }
                        }
                        BehaviorEditorTab::Scope => {
                            if let Some(cfg) = modal.working_config.as_mut() {
                                render_behavior_editor_scope_tab(
                                    ui,
                                    cfg,
                                    &pod_backend_names,
                                    &pod_host_env_names,
                                    &pod_mcp_host_names,
                                );
                            }
                        }
                        BehaviorEditorTab::Retention => {
                            if let Some(cfg) = modal.working_config.as_mut() {
                                render_behavior_editor_retention_tab(ui, cfg);
                            }
                        }
                        BehaviorEditorTab::Prompt => {
                            render_behavior_editor_prompt_tab(ui, &mut modal.working_prompt);
                        }
                        BehaviorEditorTab::RawToml => {
                            render_behavior_editor_raw_tab(
                                ui,
                                &mut modal.raw_buffer,
                                &mut modal.raw_dirty,
                            );
                        }
                    });
            });
        });

    // Tab switch (post-show so no UI borrow).
    if let Some(target) = switch_to {
        let leaving_raw =
            modal.tab == BehaviorEditorTab::RawToml && target != BehaviorEditorTab::RawToml;
        let entering_raw = target == BehaviorEditorTab::RawToml;
        match crate::editor::sync_on_tab_switch::<BehaviorConfig>(
            leaving_raw,
            entering_raw,
            &mut modal.working_config,
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

    if save_clicked && let Some(working) = &modal.working_config {
        // If the raw tab has pending edits, reparse it and use that;
        // otherwise serialize from working. Matches the pod editor's
        // precedence.
        let config = if modal.tab == BehaviorEditorTab::RawToml && modal.raw_dirty {
            match toml::from_str::<BehaviorConfig>(&modal.raw_buffer) {
                Ok(c) => c,
                Err(e) => {
                    modal.error = Some(format!("raw TOML doesn't parse: {e}"));
                    *slot = Some(modal);
                    return events;
                }
            }
        } else {
            working.clone()
        };
        modal.error = None;
        let pod_id = modal.pod_id.clone();
        let behavior_id = modal.behavior_id.clone();
        let prompt = modal.working_prompt.clone();
        // Keep the modal open; caller stamps `pending_correlation` on it.
        *slot = Some(modal);
        events.push(BehaviorEditorEvent::SaveRequested {
            pod_id,
            behavior_id,
            config,
            prompt,
        });
        return events;
    }
    if revert_clicked {
        if let Some(baseline) = &modal.baseline_config {
            modal.working_config = Some(baseline.clone());
            modal.raw_buffer = toml::to_string_pretty(baseline).unwrap_or_default();
            modal.raw_dirty = false;
        }
        modal.working_prompt = modal.baseline_prompt.clone();
        modal.error = None;
        modal.pending_correlation = None;
        *slot = Some(modal);
        return events;
    }
    if close_clicked || !open {
        // Modal closes.
        return events;
    }
    *slot = Some(modal);
    events
}
