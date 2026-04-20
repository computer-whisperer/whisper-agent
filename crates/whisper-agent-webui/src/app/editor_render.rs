//! Pod-editor + behavior-editor + sandbox-entry modal rendering.
//!
//! Every function here is a free function that takes an `egui::Ui` plus
//! mutable references to the piece of working state it edits. Kept free
//! of `ChatApp` so the renderers are testable in isolation and reusable
//! in a future inline-preview surface. The parent (`app.rs`) owns the
//! modal state lifecycle; these functions only paint and mutate.
//!
//! Also hosts the small label/summary helpers the renderers share
//! (`approval_policy_label`, `trigger_kind_label`, `spec_type_label`,
//! `section_heading`, `hint`) plus the `BehaviorSnapshot → BehaviorSummary`
//! adapter used by both the editor and the list cache.

use std::collections::{BTreeMap, HashMap};

use egui::{Color32, ComboBox, DragValue, Grid, RichText, ScrollArea, TextEdit};
use whisper_agent_protocol::sandbox::{
    AccessMode, Mount, NetworkPolicy, PathAccess, ResourceLimits,
};
use whisper_agent_protocol::{
    BehaviorConfig, BehaviorSnapshot as BehaviorSnapshotProto, BehaviorSummary, CatchUp,
    Disposition, HostEnvProviderInfo, HostEnvSpec, ModelSummary, Overlap, PodConfig,
    RetentionPolicy, TriggerSpec,
};

fn disposition_label(d: Disposition) -> &'static str {
    match d {
        Disposition::Allow => "allow",
        Disposition::AllowWithPrompt => "allow with prompt",
        Disposition::Deny => "deny",
    }
}

use super::{SandboxEntryEditorState, spec_label};

// ====================================================================
// Pod editor — structured form
//
// The renderer in `render_pod_editor_modal` dispatches to one of these
// per-tab helpers, plus the per-sandbox-entry sub-modal. They take
// `working: &mut PodConfig` rather than the modal struct so they can be
// reused trivially in tests / future inline previews; mutations land
// directly on `working` and the parent treats any change as "dirty"
// via its baseline diff.
// ====================================================================

pub(super) fn render_pod_editor_allow_tab(
    ui: &mut egui::Ui,
    working: &mut PodConfig,
    backend_catalog: &[String],
    shared_mcp_catalog: &[String],
    host_env_providers: &[HostEnvProviderInfo],
    sandbox_open: &mut Option<SandboxEntryEditorState>,
    sandbox_delete: &mut Option<usize>,
) {
    ui.add_space(4.0);
    section_heading(ui, "Identity");
    Grid::new("pod_editor_identity")
        .num_columns(2)
        .min_col_width(110.0)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            ui.label("name");
            ui.add(
                TextEdit::singleline(&mut working.name)
                    .hint_text("display name")
                    .desired_width(f32::INFINITY),
            );
            ui.end_row();
            ui.label("description");
            let mut desc = working.description.clone().unwrap_or_default();
            let resp = ui.add(
                TextEdit::multiline(&mut desc)
                    .hint_text("optional — surfaced in the pod list")
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            if resp.changed() {
                working.description = if desc.trim().is_empty() {
                    None
                } else {
                    Some(desc)
                };
            }
            ui.end_row();
        });

    ui.add_space(10.0);
    section_heading(ui, "Allowed backends");
    hint(
        ui,
        "Threads in this pod may bind to any backend listed here. \
         `thread_defaults.backend` must be one of these.",
    );
    if backend_catalog.is_empty() {
        ui.label(
            RichText::new("(no backends — server hasn't reported any yet)")
                .italics()
                .color(Color32::from_gray(160)),
        );
    } else {
        ui.horizontal_wrapped(|ui| {
            for name in backend_catalog {
                let mut on = working.allow.backends.iter().any(|b| b == name);
                if ui.checkbox(&mut on, name).changed() {
                    if on {
                        if !working.allow.backends.iter().any(|b| b == name) {
                            working.allow.backends.push(name.clone());
                        }
                    } else {
                        working.allow.backends.retain(|b| b != name);
                    }
                }
            }
        });
    }

    ui.add_space(10.0);
    section_heading(ui, "Allowed shared MCP hosts");
    hint(
        ui,
        "Singleton MCP hosts the pod can use. `thread_defaults.mcp_hosts` \
         must reference these by name.",
    );
    if shared_mcp_catalog.is_empty() {
        ui.label(
            RichText::new("(no shared MCP hosts configured server-side)")
                .italics()
                .color(Color32::from_gray(160)),
        );
    } else {
        ui.horizontal_wrapped(|ui| {
            for name in shared_mcp_catalog {
                let mut on = working.allow.mcp_hosts.iter().any(|m| m == name);
                if ui.checkbox(&mut on, name).changed() {
                    if on {
                        if !working.allow.mcp_hosts.iter().any(|m| m == name) {
                            working.allow.mcp_hosts.push(name.clone());
                        }
                    } else {
                        working.allow.mcp_hosts.retain(|m| m != name);
                    }
                }
            }
        });
    }

    ui.add_space(10.0);
    section_heading(ui, "Host environments");
    hint(
        ui,
        "Each entry is a named (provider, spec) pair threads in this pod can bind \
         to. Every entry dispatches to one of the server's configured \
         `[[host_env_providers]]` daemons. A pod with zero entries is valid — its \
         threads run with shared MCPs only (no bash / file / edit tools).",
    );
    if working.allow.host_env.is_empty() {
        ui.label(
            RichText::new(
                "(no host envs — threads in this pod will have shared MCP tools \
                 only)",
            )
            .italics()
            .color(Color32::from_gray(160)),
        );
    } else {
        Grid::new("pod_editor_host_envs")
            .num_columns(5)
            .spacing([12.0, 4.0])
            .min_col_width(100.0)
            .striped(true)
            .show(ui, |ui| {
                ui.label(RichText::new("name").strong());
                ui.label(RichText::new("provider").strong());
                ui.label(RichText::new("type").strong());
                ui.label(RichText::new("summary").strong());
                ui.label("");
                ui.end_row();
                for (i, entry) in working.allow.host_env.iter().enumerate() {
                    ui.label(&entry.name);
                    ui.label(&entry.provider);
                    ui.label(spec_type_label(&entry.spec));
                    ui.label(RichText::new(spec_label(&entry.spec)).color(Color32::from_gray(170)));
                    ui.horizontal(|ui| {
                        if ui.small_button("Edit").clicked() {
                            *sandbox_open =
                                Some(SandboxEntryEditorState::new_for_index(i, entry.clone()));
                        }
                        if ui
                            .small_button(
                                RichText::new("Delete").color(Color32::from_rgb(220, 100, 100)),
                            )
                            .clicked()
                        {
                            *sandbox_delete = Some(i);
                        }
                    });
                    ui.end_row();
                }
            });
    }
    ui.add_space(4.0);
    if ui.button("+ Add host env").clicked() {
        let default_provider = host_env_providers.first().map(|p| p.name.as_str());
        *sandbox_open = Some(SandboxEntryEditorState::new_for_add(default_provider));
    }
    ui.add_space(8.0);
}

pub(super) fn render_pod_editor_defaults_tab(
    ui: &mut egui::Ui,
    working: &mut PodConfig,
    backend_catalog: &[String],
    models_by_backend: &HashMap<String, Vec<ModelSummary>>,
) {
    ui.add_space(4.0);
    hint(
        ui,
        "These values seed every thread created in this pod. They can still be \
         overridden per-thread at create-time by the webui's compose form.",
    );
    ui.add_space(4.0);
    Grid::new("pod_editor_defaults")
        .num_columns(2)
        .min_col_width(140.0)
        .spacing([12.0, 8.0])
        .show(ui, |ui| {
            // Backend
            ui.label("backend");
            let backend_in_allow = working
                .allow
                .backends
                .iter()
                .any(|b| b == &working.thread_defaults.backend);
            ComboBox::from_id_salt("pod_editor_defaults_backend")
                .selected_text(if working.thread_defaults.backend.is_empty() {
                    "(none)".to_string()
                } else {
                    working.thread_defaults.backend.clone()
                })
                .show_ui(ui, |ui| {
                    if working.allow.backends.is_empty() {
                        ui.label(
                            RichText::new("(no backends in [allow])")
                                .italics()
                                .color(Color32::from_gray(160)),
                        );
                    }
                    for name in &working.allow.backends {
                        ui.selectable_value(
                            &mut working.thread_defaults.backend,
                            name.clone(),
                            name,
                        );
                    }
                    let extras: Vec<String> = backend_catalog
                        .iter()
                        .filter(|b| !working.allow.backends.iter().any(|x| x == *b))
                        .cloned()
                        .collect();
                    if !extras.is_empty() {
                        ui.separator();
                        ui.label(
                            RichText::new("not in [allow] — picking would error on save")
                                .small()
                                .color(Color32::from_gray(160)),
                        );
                        for name in extras {
                            ui.selectable_value(
                                &mut working.thread_defaults.backend,
                                name.clone(),
                                RichText::new(name).color(Color32::from_rgb(220, 170, 90)),
                            );
                        }
                    }
                });
            ui.end_row();
            if !backend_in_allow && !working.thread_defaults.backend.is_empty() {
                ui.label("");
                ui.label(
                    RichText::new(format!(
                        "`{}` is not in allow.backends — server will reject this on save",
                        working.thread_defaults.backend
                    ))
                    .small()
                    .color(Color32::from_rgb(220, 90, 90)),
                );
                ui.end_row();
            }

            ui.label("model");
            render_model_combo(
                ui,
                "pod_editor_defaults_model",
                &working.thread_defaults.backend,
                &mut working.thread_defaults.model,
                models_by_backend,
            );
            ui.end_row();

            ui.label("system prompt file");
            ui.add(
                TextEdit::singleline(&mut working.thread_defaults.system_prompt_file)
                    .hint_text("path relative to the pod directory (empty = none)")
                    .desired_width(f32::INFINITY),
            );
            ui.end_row();

            ui.label("max tokens");
            ui.add(
                DragValue::new(&mut working.thread_defaults.max_tokens)
                    .range(1..=200_000)
                    .speed(50.0),
            );
            ui.end_row();

            ui.label("max turns");
            ui.add(
                DragValue::new(&mut working.thread_defaults.max_turns)
                    .range(1..=10_000)
                    .speed(1.0),
            );
            ui.end_row();

            ui.label("tool gate default");
            ComboBox::from_id_salt("pod_editor_tools_default")
                .selected_text(disposition_label(working.allow.tools.default))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut working.allow.tools.default,
                        Disposition::Allow,
                        disposition_label(Disposition::Allow),
                    );
                    ui.selectable_value(
                        &mut working.allow.tools.default,
                        Disposition::AllowWithPrompt,
                        disposition_label(Disposition::AllowWithPrompt),
                    );
                    ui.selectable_value(
                        &mut working.allow.tools.default,
                        Disposition::Deny,
                        disposition_label(Disposition::Deny),
                    );
                });
            ui.end_row();

            // Per-tool override count — full per-tool editor is deferred.
            // Edit overrides via the raw-toml tab for now.
            let override_count = working.allow.tools.overrides.len();
            ui.label("tool overrides");
            if override_count == 0 {
                ui.label("(none)");
            } else {
                ui.label(format!("{override_count} tool(s) — edit via raw TOML"));
            }
            ui.end_row();

            ui.label("host env");
            // Multi-select over the pod's `[allow.host_env]` entries.
            // Threads with zero bindings run bare (no host-env tools,
            // only shared MCPs); threads with multiple bindings
            // provision each env in parallel and the model sees
            // every env's tools prefixed with the env's allow-list
            // name (e.g. `c-dtop_write_file`).
            //
            // Pods with an empty allow.host_env can't offer bindings
            // at all; keep the value forced to empty in that case to
            // stop stale names from lingering after allow edits.
            let sb_in_allow = working
                .thread_defaults
                .host_env
                .iter()
                .all(|name| working.allow.host_env.iter().any(|s| &s.name == name));
            if working.allow.host_env.is_empty() {
                working.thread_defaults.host_env.clear();
                ui.label(
                    RichText::new(
                        "— (this pod has no host envs in [allow] — threads here get \
                         shared MCPs only)",
                    )
                    .italics()
                    .color(Color32::from_gray(160)),
                );
            } else {
                ui.horizontal_wrapped(|ui| {
                    for entry in &working.allow.host_env {
                        let mut on = working
                            .thread_defaults
                            .host_env
                            .iter()
                            .any(|n| n == &entry.name);
                        if ui.checkbox(&mut on, &entry.name).changed() {
                            if on {
                                if !working
                                    .thread_defaults
                                    .host_env
                                    .iter()
                                    .any(|n| n == &entry.name)
                                {
                                    working.thread_defaults.host_env.push(entry.name.clone());
                                }
                            } else {
                                working
                                    .thread_defaults
                                    .host_env
                                    .retain(|n| n != &entry.name);
                            }
                        }
                    }
                });
            }
            ui.end_row();
            if !sb_in_allow {
                // Surfaces only the names that fell out of allow — the
                // subset-diff keeps the hint actionable even when most
                // of the list is still valid.
                let invalid: Vec<&String> = working
                    .thread_defaults
                    .host_env
                    .iter()
                    .filter(|n| !working.allow.host_env.iter().any(|s| s.name == **n))
                    .collect();
                ui.label("");
                ui.label(
                    RichText::new(format!(
                        "{} not in allow.host_env — server will reject on save",
                        invalid
                            .iter()
                            .map(|s| format!("`{s}`"))
                            .collect::<Vec<_>>()
                            .join(", "),
                    ))
                    .small()
                    .color(Color32::from_rgb(220, 90, 90)),
                );
                ui.end_row();
            }
        });

    ui.add_space(10.0);
    section_heading(ui, "Default MCP hosts");
    hint(
        ui,
        "Threads created in this pod start subscribed to these shared MCP hosts \
         (in addition to the per-thread primary).",
    );
    if working.allow.mcp_hosts.is_empty() {
        ui.label(
            RichText::new("(no shared MCP hosts in [allow])")
                .italics()
                .color(Color32::from_gray(160)),
        );
    } else {
        ui.horizontal_wrapped(|ui| {
            for name in working.allow.mcp_hosts.clone() {
                let mut on = working.thread_defaults.mcp_hosts.iter().any(|m| m == &name);
                if ui.checkbox(&mut on, &name).changed() {
                    if on {
                        if !working.thread_defaults.mcp_hosts.iter().any(|m| m == &name) {
                            working.thread_defaults.mcp_hosts.push(name);
                        }
                    } else {
                        working.thread_defaults.mcp_hosts.retain(|m| m != &name);
                    }
                }
            }
        });
    }
    ui.add_space(8.0);
}

pub(super) fn render_pod_editor_limits_tab(ui: &mut egui::Ui, working: &mut PodConfig) {
    ui.add_space(4.0);
    hint(
        ui,
        "Pod-level caps. Most installations keep the defaults; tighten when a \
         pod's threads contend for a constrained resource (model quota, sandbox \
         capacity).",
    );
    ui.add_space(6.0);
    Grid::new("pod_editor_limits")
        .num_columns(2)
        .min_col_width(180.0)
        .spacing([12.0, 8.0])
        .show(ui, |ui| {
            ui.label("max concurrent threads");
            ui.add(
                DragValue::new(&mut working.limits.max_concurrent_threads)
                    .range(1..=1000)
                    .speed(1.0),
            );
            ui.end_row();
        });
    ui.add_space(8.0);
}

pub(super) fn render_pod_editor_raw_tab(ui: &mut egui::Ui, raw: &mut String, dirty: &mut bool) {
    crate::editor::render_raw_toml_tab(
        ui,
        "Raw pod.toml. Edits here override the structured tabs on save. \
         Switching back to a structured tab tries to parse this text first; \
         a parse error keeps you here so the edit isn't lost.",
        raw,
        dirty,
    );
}

pub(super) fn render_behavior_editor_trigger_tab(ui: &mut egui::Ui, cfg: &mut BehaviorConfig) {
    ui.add_space(4.0);
    section_heading(ui, "Identity");
    Grid::new("behavior_editor_identity")
        .num_columns(2)
        .min_col_width(110.0)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            ui.label("name");
            ui.add(
                TextEdit::singleline(&mut cfg.name)
                    .hint_text("display name")
                    .desired_width(f32::INFINITY),
            );
            ui.end_row();
            ui.label("description");
            let mut desc = cfg.description.clone().unwrap_or_default();
            let resp = ui.add(
                TextEdit::multiline(&mut desc)
                    .hint_text("optional — shown in the behaviors list")
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            if resp.changed() {
                cfg.description = if desc.trim().is_empty() {
                    None
                } else {
                    Some(desc)
                };
            }
            ui.end_row();
        });

    ui.add_space(10.0);
    section_heading(ui, "Trigger");
    hint(
        ui,
        "Manual behaviors fire only when you click Run. Cron behaviors \
         fire on a schedule in their configured timezone. Webhook \
         behaviors fire on HTTP POST to the endpoint shown below.",
    );

    let current_kind = match &cfg.trigger {
        TriggerSpec::Manual => "manual",
        TriggerSpec::Cron { .. } => "cron",
        TriggerSpec::Webhook { .. } => "webhook",
    };
    ui.horizontal(|ui| {
        for kind in ["manual", "cron", "webhook"] {
            let active = kind == current_kind;
            if ui.selectable_label(active, kind).clicked() && !active {
                cfg.trigger = match kind {
                    "manual" => TriggerSpec::Manual,
                    "cron" => TriggerSpec::Cron {
                        schedule: "0 9 * * *".into(),
                        timezone: "UTC".into(),
                        overlap: Overlap::Skip,
                        catch_up: CatchUp::One,
                    },
                    "webhook" => TriggerSpec::Webhook {
                        overlap: Overlap::Skip,
                    },
                    _ => unreachable!(),
                };
            }
        }
    });

    ui.add_space(8.0);
    match &mut cfg.trigger {
        TriggerSpec::Manual => {
            hint(
                ui,
                "No further configuration. Use RunBehavior (or the Run \
                 button in the list) to fire.",
            );
        }
        TriggerSpec::Cron {
            schedule,
            timezone,
            overlap,
            catch_up,
        } => {
            let schedule_valid = crate::cron_preview::parse_schedule(schedule).is_ok();
            let tz_valid = crate::cron_preview::parse_tz(timezone).is_ok();

            ui.label("schedule");
            let mut sched_edit = TextEdit::singleline(schedule)
                .hint_text("5-field crontab (e.g. '0 9 * * *')")
                .desired_width(f32::INFINITY);
            if !schedule_valid {
                sched_edit = sched_edit.text_color(egui::Color32::from_rgb(220, 80, 80));
            }
            ui.add(sched_edit);
            crate::cron_preview::render_schedule_presets(ui, schedule);

            ui.add_space(6.0);
            ui.label("timezone");
            let mut tz_edit = TextEdit::singleline(timezone)
                .hint_text("IANA zone (e.g. 'America/Los_Angeles')")
                .desired_width(f32::INFINITY);
            if !tz_valid {
                tz_edit = tz_edit.text_color(egui::Color32::from_rgb(220, 80, 80));
            }
            ui.add(tz_edit);
            crate::cron_preview::render_tz_presets(ui, timezone);

            ui.add_space(8.0);
            Grid::new("behavior_editor_cron_opts")
                .num_columns(2)
                .min_col_width(110.0)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    ui.label("overlap");
                    render_overlap_picker(ui, overlap);
                    ui.end_row();
                    ui.label("catch_up");
                    render_catch_up_picker(ui, catch_up);
                    ui.end_row();
                });

            ui.add_space(10.0);
            crate::cron_preview::render_preview(ui, schedule, timezone);
        }
        TriggerSpec::Webhook { overlap } => {
            Grid::new("behavior_editor_webhook")
                .num_columns(2)
                .min_col_width(110.0)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    ui.label("overlap");
                    render_overlap_picker(ui, overlap);
                    ui.end_row();
                });
            ui.add_space(6.0);
            hint(
                ui,
                "POST endpoint (relative to this server): \
                 /triggers/<pod_id>/<behavior_id>. Empty body → Null \
                 payload. Non-empty bodies must be valid JSON.",
            );
        }
    }
}

fn render_overlap_picker(ui: &mut egui::Ui, overlap: &mut Overlap) {
    ui.horizontal(|ui| {
        for (label, value, tip) in [
            (
                "skip",
                Overlap::Skip,
                "drop the fire if the previous run is still in flight",
            ),
            (
                "queue_one",
                Overlap::QueueOne,
                "park one pending payload; fire when the previous run finishes",
            ),
            (
                "allow",
                Overlap::Allow,
                "always fire, concurrent runs are OK",
            ),
        ] {
            let active = *overlap == value;
            if ui
                .selectable_label(active, label)
                .on_hover_text(tip)
                .clicked()
                && !active
            {
                *overlap = value;
            }
        }
    });
}

fn render_catch_up_picker(ui: &mut egui::Ui, catch_up: &mut CatchUp) {
    ui.horizontal(|ui| {
        for (label, value, tip) in [
            (
                "none",
                CatchUp::None,
                "on server restart, skip missed fires silently",
            ),
            (
                "one",
                CatchUp::One,
                "on server restart, fire once for missed windows",
            ),
            (
                "all",
                CatchUp::All,
                "reserved — currently capped to one fire, logs a warning",
            ),
        ] {
            let active = *catch_up == value;
            if ui
                .selectable_label(active, label)
                .on_hover_text(tip)
                .clicked()
                && !active
            {
                *catch_up = value;
            }
        }
    });
}

pub(super) fn render_behavior_editor_thread_tab(
    ui: &mut egui::Ui,
    cfg: &mut BehaviorConfig,
    backend_catalog: &[String],
    pod_host_env_names: &[String],
    pod_mcp_host_names: &[String],
    models_by_backend: &HashMap<String, Vec<ModelSummary>>,
    pod_default_backend: &str,
) {
    ui.add_space(4.0);
    section_heading(ui, "Thread overrides");
    hint(
        ui,
        "Every field is optional. Unset fields inherit from the pod's \
         thread_defaults at fire time. Bindings must resolve against \
         the pod's [allow] table — set too aggressively, the server \
         rejects the save.",
    );
    ui.add_space(4.0);

    Grid::new("behavior_editor_thread_plain")
        .num_columns(2)
        .min_col_width(140.0)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            ui.label("model");
            let mut model_set = cfg.thread.model.is_some();
            let mut model_val = cfg.thread.model.clone().unwrap_or_default();
            // Effective backend for the model combo: the binding
            // override if set, else the pod default. An empty result
            // shows up as "(pick a backend first)" inside the combo.
            let effective_backend = cfg
                .thread
                .bindings
                .backend
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or(pod_default_backend);
            ui.horizontal(|ui| {
                if ui.checkbox(&mut model_set, "override").changed() {
                    cfg.thread.model = if model_set {
                        Some(model_val.clone())
                    } else {
                        None
                    };
                }
                ui.add_enabled_ui(model_set, |ui| {
                    render_model_combo(
                        ui,
                        "behavior_thread_model",
                        effective_backend,
                        &mut model_val,
                        models_by_backend,
                    );
                });
                if model_set {
                    cfg.thread.model = Some(model_val);
                }
            });
            ui.end_row();

            render_optional_u32_row(
                ui,
                "max_tokens",
                &mut cfg.thread.max_tokens,
                1,
                u32::MAX,
                16384,
            );
            render_optional_u32_row(ui, "max_turns", &mut cfg.thread.max_turns, 1, 500, 30);

            // approval_policy row removed — tool gate now lives at pod
            // level as `allow.tools` (edited on the Allow tab). Behavior
            // threads inherit the pod's tool gate.
        });

    ui.add_space(10.0);
    section_heading(ui, "Binding overrides");
    hint(
        ui,
        "Each `Some` replaces the corresponding pod default at fire \
         time. The pod's [allow] table is authoritative — the server \
         rejects saves whose bindings fall outside it.",
    );
    ui.add_space(4.0);
    render_optional_string_picker(
        ui,
        "backend",
        &mut cfg.thread.bindings.backend,
        backend_catalog,
        "(inherit pod default)",
    );
    ui.add_space(6.0);
    ui.label("host_env");
    let mut host_env_set = cfg.thread.bindings.host_env.is_some();
    if ui
        .checkbox(&mut host_env_set, "override pod default host_env")
        .changed()
    {
        cfg.thread.bindings.host_env = if host_env_set { Some(Vec::new()) } else { None };
    }
    if let Some(selected) = cfg.thread.bindings.host_env.as_mut() {
        if pod_host_env_names.is_empty() {
            ui.label(
                RichText::new("(pod declares no host envs in [allow])")
                    .italics()
                    .color(Color32::from_gray(160)),
            );
        } else {
            ui.horizontal_wrapped(|ui| {
                for name in pod_host_env_names {
                    let mut on = selected.iter().any(|n| n == name);
                    if ui.checkbox(&mut on, name).changed() {
                        if on {
                            if !selected.iter().any(|n| n == name) {
                                selected.push(name.clone());
                            }
                        } else {
                            selected.retain(|n| n != name);
                        }
                    }
                }
            });
        }
    } else {
        ui.label(
            RichText::new("(inherit pod default)")
                .italics()
                .color(Color32::from_gray(160)),
        );
    }

    ui.add_space(6.0);
    ui.label("mcp_hosts");
    let mut mcp_set = cfg.thread.bindings.mcp_hosts.is_some();
    if ui
        .checkbox(&mut mcp_set, "override pod default mcp_hosts")
        .changed()
    {
        cfg.thread.bindings.mcp_hosts = if mcp_set { Some(Vec::new()) } else { None };
    }
    if let Some(selected) = cfg.thread.bindings.mcp_hosts.as_mut() {
        if pod_mcp_host_names.is_empty() {
            ui.label(
                RichText::new("(pod declares no shared MCP hosts)")
                    .italics()
                    .color(Color32::from_gray(160)),
            );
        } else {
            ui.horizontal_wrapped(|ui| {
                for name in pod_mcp_host_names {
                    let mut on = selected.iter().any(|n| n == name);
                    if ui.checkbox(&mut on, name).changed() {
                        if on {
                            selected.push(name.clone());
                        } else {
                            selected.retain(|n| n != name);
                        }
                    }
                }
            });
        }
    }
}

/// Shared helper: optional-u32 field with an "override" checkbox +
/// DragValue. Used for max_tokens / max_turns.
fn render_optional_u32_row(
    ui: &mut egui::Ui,
    label: &str,
    field: &mut Option<u32>,
    min: u32,
    max: u32,
    default_if_enabled: u32,
) {
    ui.label(label);
    let mut on = field.is_some();
    let mut value = field.unwrap_or(default_if_enabled);
    ui.horizontal(|ui| {
        if ui.checkbox(&mut on, "override").changed() {
            *field = if on { Some(value) } else { None };
        }
        ui.add_enabled_ui(on, |ui| {
            if ui
                .add(egui::DragValue::new(&mut value).range(min..=max).speed(1.0))
                .changed()
                && on
            {
                *field = Some(value);
            }
        });
    });
    ui.end_row();
}

/// ComboBox over the models cached for `backend`. Writes the chosen
/// id into `current` on click. `selected_text` echoes `current` as-is
/// so a value hand-edited in Raw TOML (or inherited from an old
/// config) shows up in the header even if it isn't in the catalog.
///
/// `backend` empty → hint; list absent → "(loading…)"; list empty →
/// note that the backend returned nothing. Callers must independently
/// fire `ChatApp::request_models_for(backend)` to populate the cache;
/// this widget only reads from it.
fn render_model_combo(
    ui: &mut egui::Ui,
    id_salt: &str,
    backend: &str,
    current: &mut String,
    models_by_backend: &HashMap<String, Vec<ModelSummary>>,
) {
    ComboBox::from_id_salt(id_salt)
        .width(280.0)
        .selected_text(if current.is_empty() {
            "(pick a model)".to_string()
        } else {
            current.clone()
        })
        .show_ui(ui, |ui| {
            if backend.is_empty() {
                ui.label(
                    RichText::new("(pick a backend first)")
                        .italics()
                        .color(Color32::from_gray(160)),
                );
                return;
            }
            match models_by_backend.get(backend) {
                None => {
                    ui.label(
                        RichText::new("(loading models…)")
                            .italics()
                            .color(Color32::from_gray(160)),
                    );
                }
                Some(list) if list.is_empty() => {
                    ui.label(
                        RichText::new("(backend returned no models)")
                            .italics()
                            .color(Color32::from_gray(160)),
                    );
                }
                Some(list) => {
                    for m in list {
                        let label = match &m.display_name {
                            Some(d) => format!("{}  ({})", m.id, d),
                            None => m.id.clone(),
                        };
                        if ui.selectable_label(*current == m.id, label).clicked() {
                            *current = m.id.clone();
                        }
                    }
                }
            }
        });
}

/// Optional-string picker with override toggle + ComboBox of valid
/// choices. Used for backend and host_env binding overrides.
fn render_optional_string_picker(
    ui: &mut egui::Ui,
    label: &str,
    field: &mut Option<String>,
    choices: &[String],
    inherit_placeholder: &str,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut on = field.is_some();
        let mut value = field.clone().unwrap_or_default();
        if ui.checkbox(&mut on, "override").changed() {
            *field = if on {
                Some(choices.first().cloned().unwrap_or_default())
            } else {
                None
            };
            if let Some(v) = field {
                value = v.clone();
            }
        }
        if on {
            egui::ComboBox::from_id_salt(format!("behavior_picker_{label}"))
                .selected_text(if value.is_empty() {
                    "(pick one)"
                } else {
                    value.as_str()
                })
                .show_ui(ui, |ui| {
                    for choice in choices {
                        if ui.selectable_label(&value == choice, choice).clicked() {
                            value = choice.clone();
                        }
                    }
                });
            *field = Some(value);
        } else {
            ui.label(
                RichText::new(inherit_placeholder)
                    .italics()
                    .color(Color32::from_gray(160)),
            );
        }
    });
}

pub(super) fn render_behavior_editor_retention_tab(ui: &mut egui::Ui, cfg: &mut BehaviorConfig) {
    ui.add_space(4.0);
    section_heading(ui, "On completion");
    hint(
        ui,
        "What to do with a thread this behavior spawned once it \
         reaches a terminal state. Applies only to behavior-spawned \
         threads — interactive threads always Keep. Sweep runs hourly.",
    );
    ui.add_space(6.0);

    let current_kind = match &cfg.on_completion {
        RetentionPolicy::Keep => "keep",
        RetentionPolicy::ArchiveAfterDays { .. } => "archive_after_days",
        RetentionPolicy::DeleteAfterDays { .. } => "delete_after_days",
    };
    ui.horizontal(|ui| {
        for (label, tip) in [
            ("keep", "never sweep — retains the thread JSON forever"),
            (
                "archive_after_days",
                "move the JSON to <pod>/.archived/threads/ after N days",
            ),
            (
                "delete_after_days",
                "rm the JSON after N days; no forensic copy",
            ),
        ] {
            let active = label == current_kind;
            if ui
                .selectable_label(active, label)
                .on_hover_text(tip)
                .clicked()
                && !active
            {
                cfg.on_completion = match label {
                    "keep" => RetentionPolicy::Keep,
                    "archive_after_days" => RetentionPolicy::ArchiveAfterDays { days: 30 },
                    "delete_after_days" => RetentionPolicy::DeleteAfterDays { days: 30 },
                    _ => unreachable!(),
                };
            }
        }
    });
    ui.add_space(8.0);
    match &mut cfg.on_completion {
        RetentionPolicy::Keep => {}
        RetentionPolicy::ArchiveAfterDays { days } | RetentionPolicy::DeleteAfterDays { days } => {
            Grid::new("behavior_editor_retention")
                .num_columns(2)
                .min_col_width(110.0)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    ui.label("days");
                    ui.add(egui::DragValue::new(days).range(1..=3650).speed(1.0));
                    ui.end_row();
                });
        }
    }
}

pub(super) fn render_behavior_editor_prompt_tab(ui: &mut egui::Ui, prompt: &mut String) {
    ui.add_space(4.0);
    hint(
        ui,
        "prompt.md — delivered as the initial user message when the \
         behavior fires. `{{payload}}` is substituted with the \
         trigger payload (pretty JSON; empty for Null payloads). The \
         pod's system_prompt_file remains the thread's system prompt.",
    );
    ui.add_space(4.0);
    ui.add_sized(
        [ui.available_width(), ui.available_height().max(180.0)],
        TextEdit::multiline(prompt).desired_rows(12),
    );
}

pub(super) fn render_behavior_editor_raw_tab(
    ui: &mut egui::Ui,
    raw: &mut String,
    dirty: &mut bool,
) {
    crate::editor::render_raw_toml_tab(
        ui,
        "Raw behavior.toml. Edits here override the structured tabs on save. \
         Prompt text is edited in the Prompt tab, not here. Switching back \
         to a structured tab tries to parse this text first; a parse error \
         keeps you here so the edit isn't lost.",
        raw,
        dirty,
    );
}

pub(super) fn render_sandbox_entry_modal(
    ctx: &egui::Context,
    sub: &mut SandboxEntryEditorState,
    open: &mut bool,
    save: &mut bool,
    cancel: &mut bool,
    providers: &[HostEnvProviderInfo],
) {
    let title = match sub.index {
        Some(_) => "Edit host env".to_string(),
        None => "Add host env".to_string(),
    };
    let screen = ctx.content_rect();
    let max_h = (screen.height() - 80.0).max(280.0);
    let max_w = (screen.width() - 80.0).max(420.0);

    egui::Window::new(title)
        .collapsible(false)
        .resizable(true)
        .default_width(560.0_f32.min(max_w))
        .default_height(480.0_f32.min(max_h))
        .max_width(max_w)
        .max_height(max_h)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(open)
        .show(ctx, |ui| {
            egui::Panel::bottom("sandbox_entry_footer").show_inside(ui, |ui| {
                ui.add_space(6.0);
                if let Some(err) = &sub.error {
                    ui.colored_label(Color32::from_rgb(220, 80, 80), err);
                    ui.add_space(4.0);
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        *save = true;
                    }
                    if ui.button("Cancel").clicked() {
                        *cancel = true;
                    }
                });
            });
            egui::CentralPanel::default().show_inside(ui, |ui| {
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.add_space(4.0);
                        Grid::new("sandbox_entry_top")
                            .num_columns(2)
                            .min_col_width(100.0)
                            .spacing([12.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("name");
                                ui.add(
                                    TextEdit::singleline(&mut sub.entry.name)
                                        .hint_text("e.g. landlock-rw, container-cargo")
                                        .desired_width(f32::INFINITY),
                                );
                                ui.end_row();

                                ui.label("provider");
                                ComboBox::from_id_salt("sandbox_entry_provider")
                                    .selected_text(if sub.entry.provider.is_empty() {
                                        "(pick one)".into()
                                    } else {
                                        sub.entry.provider.clone()
                                    })
                                    .show_ui(ui, |ui| {
                                        if providers.is_empty() {
                                            ui.label(
                                                RichText::new(
                                                    "no providers configured — add \
                                                     `[[host_env_providers]]` entries to \
                                                     whisper-agent.toml",
                                                )
                                                .italics()
                                                .color(Color32::from_gray(160)),
                                            );
                                        }
                                        for p in providers {
                                            ui.selectable_value(
                                                &mut sub.entry.provider,
                                                p.name.clone(),
                                                &p.name,
                                            );
                                        }
                                    });
                                ui.end_row();
                                ui.label("type");
                                let current = spec_type_label(&sub.entry.spec);
                                ComboBox::from_id_salt("sandbox_entry_type")
                                    .selected_text(current)
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_label(
                                                matches!(
                                                    sub.entry.spec,
                                                    HostEnvSpec::Landlock { .. }
                                                ),
                                                "landlock",
                                            )
                                            .clicked()
                                            && !matches!(
                                                sub.entry.spec,
                                                HostEnvSpec::Landlock { .. }
                                            )
                                        {
                                            sub.entry.spec = HostEnvSpec::Landlock {
                                                allowed_paths: Vec::new(),
                                                network: NetworkPolicy::default(),
                                            };
                                        }
                                        if ui
                                            .selectable_label(
                                                matches!(
                                                    sub.entry.spec,
                                                    HostEnvSpec::Container { .. }
                                                ),
                                                "container",
                                            )
                                            .clicked()
                                            && !matches!(
                                                sub.entry.spec,
                                                HostEnvSpec::Container { .. }
                                            )
                                        {
                                            sub.entry.spec = HostEnvSpec::Container {
                                                image: String::new(),
                                                mounts: Vec::new(),
                                                network: NetworkPolicy::default(),
                                                limits: None,
                                                env: BTreeMap::new(),
                                            };
                                        }
                                    });
                                ui.end_row();
                            });
                        ui.add_space(8.0);
                        ui.separator();
                        ui.add_space(8.0);
                        match &mut sub.entry.spec {
                            HostEnvSpec::Landlock {
                                allowed_paths,
                                network,
                            } => {
                                render_landlock_body(ui, allowed_paths, network);
                            }
                            HostEnvSpec::Container {
                                image,
                                mounts,
                                network,
                                limits,
                                env,
                            } => {
                                render_container_body(ui, image, mounts, network, limits, env);
                            }
                        }
                    });
            });
        });
}

fn render_landlock_body(
    ui: &mut egui::Ui,
    allowed_paths: &mut Vec<PathAccess>,
    network: &mut NetworkPolicy,
) {
    section_heading(ui, "Allowed paths");
    hint(
        ui,
        "Each row grants the listed access to one host path. Paths not listed \
         are denied entirely.",
    );
    let mut delete: Option<usize> = None;
    Grid::new("landlock_paths")
        .num_columns(3)
        .spacing([8.0, 4.0])
        .min_col_width(80.0)
        .show(ui, |ui| {
            ui.label(RichText::new("path").strong());
            ui.label(RichText::new("mode").strong());
            ui.label("");
            ui.end_row();
            for (i, p) in allowed_paths.iter_mut().enumerate() {
                ui.add(
                    TextEdit::singleline(&mut p.path)
                        .hint_text("/absolute/path")
                        .desired_width(280.0),
                );
                access_mode_combo(ui, &mut p.mode, &format!("landlock_mode_{i}"));
                if ui.small_button("✕").on_hover_text("remove path").clicked() {
                    delete = Some(i);
                }
                ui.end_row();
            }
        });
    if let Some(i) = delete {
        allowed_paths.remove(i);
    }
    ui.add_space(2.0);
    if ui.button("+ Add path").clicked() {
        allowed_paths.push(PathAccess {
            path: String::new(),
            mode: AccessMode::ReadOnly,
        });
    }

    ui.add_space(10.0);
    section_heading(ui, "Network");
    network_policy_editor(ui, network, "landlock_network");
}

fn render_container_body(
    ui: &mut egui::Ui,
    image: &mut String,
    mounts: &mut Vec<Mount>,
    network: &mut NetworkPolicy,
    limits: &mut Option<ResourceLimits>,
    env: &mut BTreeMap<String, String>,
) {
    Grid::new("container_top")
        .num_columns(2)
        .min_col_width(100.0)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            ui.label("image");
            ui.add(
                TextEdit::singleline(image)
                    .hint_text("e.g. docker.io/library/rust:1.83")
                    .desired_width(f32::INFINITY),
            );
            ui.end_row();
        });

    ui.add_space(10.0);
    section_heading(ui, "Mounts");
    hint(
        ui,
        "Bind-mounts from the host into the container. Mode controls write access.",
    );
    let mut del_mount: Option<usize> = None;
    Grid::new("container_mounts")
        .num_columns(4)
        .spacing([8.0, 4.0])
        .min_col_width(80.0)
        .show(ui, |ui| {
            ui.label(RichText::new("host").strong());
            ui.label(RichText::new("guest").strong());
            ui.label(RichText::new("mode").strong());
            ui.label("");
            ui.end_row();
            for (i, m) in mounts.iter_mut().enumerate() {
                ui.add(
                    TextEdit::singleline(&mut m.host)
                        .hint_text("/host/path")
                        .desired_width(180.0),
                );
                ui.add(
                    TextEdit::singleline(&mut m.guest)
                        .hint_text("/guest/path")
                        .desired_width(180.0),
                );
                access_mode_combo(ui, &mut m.mode, &format!("container_mount_mode_{i}"));
                if ui.small_button("✕").on_hover_text("remove mount").clicked() {
                    del_mount = Some(i);
                }
                ui.end_row();
            }
        });
    if let Some(i) = del_mount {
        mounts.remove(i);
    }
    ui.add_space(2.0);
    if ui.button("+ Add mount").clicked() {
        mounts.push(Mount {
            host: String::new(),
            guest: String::new(),
            mode: AccessMode::ReadOnly,
        });
    }

    ui.add_space(10.0);
    section_heading(ui, "Network");
    network_policy_editor(ui, network, "container_network");

    ui.add_space(10.0);
    section_heading(ui, "Resource limits");
    let mut enabled = limits.is_some();
    if ui.checkbox(&mut enabled, "set explicit limits").changed() {
        if enabled {
            *limits = Some(ResourceLimits {
                cpus: None,
                memory_mb: None,
                timeout_s: None,
            });
        } else {
            *limits = None;
        }
    }
    if let Some(lim) = limits.as_mut() {
        Grid::new("container_limits")
            .num_columns(2)
            .min_col_width(120.0)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("cpus");
                optional_uint_field(ui, &mut lim.cpus, "container_limits_cpu", 1, 256, 1.0);
                ui.end_row();
                ui.label("memory (MiB)");
                optional_uint_field(
                    ui,
                    &mut lim.memory_mb,
                    "container_limits_mem",
                    16,
                    1_048_576,
                    64.0,
                );
                ui.end_row();
                ui.label("timeout (sec)");
                optional_uint_field(
                    ui,
                    &mut lim.timeout_s,
                    "container_limits_timeout",
                    1,
                    86_400,
                    10.0,
                );
                ui.end_row();
            });
    }

    ui.add_space(10.0);
    section_heading(ui, "Environment");
    hint(ui, "Extra env vars set inside the container.");
    let mut to_remove: Option<String> = None;
    let mut to_rename: Option<(String, String)> = None;
    Grid::new("container_env")
        .num_columns(3)
        .spacing([8.0, 4.0])
        .min_col_width(100.0)
        .show(ui, |ui| {
            ui.label(RichText::new("key").strong());
            ui.label(RichText::new("value").strong());
            ui.label("");
            ui.end_row();
            for (k, v) in env.iter_mut() {
                let mut k_buf = k.clone();
                let resp = ui.add(
                    TextEdit::singleline(&mut k_buf)
                        .hint_text("KEY")
                        .desired_width(160.0),
                );
                if resp.lost_focus() && k_buf != *k {
                    to_rename = Some((k.clone(), k_buf));
                }
                ui.add(
                    TextEdit::singleline(v)
                        .hint_text("value")
                        .desired_width(220.0),
                );
                if ui.small_button("✕").on_hover_text("remove key").clicked() {
                    to_remove = Some(k.clone());
                }
                ui.end_row();
            }
        });
    if let Some(k) = to_remove {
        env.remove(&k);
    }
    if let Some((old, new)) = to_rename
        && !new.is_empty()
        && !env.contains_key(&new)
        && let Some(v) = env.remove(&old)
    {
        env.insert(new, v);
    }
    ui.add_space(2.0);
    if ui.button("+ Add env var").clicked() {
        // Find a free placeholder key — `KEY`, `KEY_1`, ...
        let mut idx = 0u32;
        let mut key = String::from("KEY");
        while env.contains_key(&key) {
            idx += 1;
            key = format!("KEY_{idx}");
        }
        env.insert(key, String::new());
    }
}

fn network_policy_editor(ui: &mut egui::Ui, policy: &mut NetworkPolicy, salt: &str) {
    let mut variant: u8 = match policy {
        NetworkPolicy::Unrestricted => 0,
        NetworkPolicy::Isolated => 1,
        NetworkPolicy::AllowList { .. } => 2,
    };
    ui.horizontal(|ui| {
        if ui.radio_value(&mut variant, 0, "unrestricted").clicked() {
            *policy = NetworkPolicy::Unrestricted;
        }
        if ui
            .radio_value(&mut variant, 1, "isolated (no network)")
            .clicked()
        {
            *policy = NetworkPolicy::Isolated;
        }
        if ui.radio_value(&mut variant, 2, "allow-list").clicked()
            && !matches!(policy, NetworkPolicy::AllowList { .. })
        {
            *policy = NetworkPolicy::AllowList { hosts: Vec::new() };
        }
    });
    if let NetworkPolicy::AllowList { hosts } = policy {
        ui.add_space(4.0);
        let mut delete: Option<usize> = None;
        Grid::new(format!("{salt}_hosts"))
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label(RichText::new("host").strong());
                ui.label("");
                ui.end_row();
                for (i, h) in hosts.iter_mut().enumerate() {
                    ui.add(
                        TextEdit::singleline(h)
                            .hint_text("e.g. crates.io")
                            .desired_width(280.0),
                    );
                    if ui.small_button("✕").on_hover_text("remove host").clicked() {
                        delete = Some(i);
                    }
                    ui.end_row();
                }
            });
        if let Some(i) = delete {
            hosts.remove(i);
        }
        if ui.button("+ Add host").clicked() {
            hosts.push(String::new());
        }
    }
}

fn access_mode_combo(ui: &mut egui::Ui, mode: &mut AccessMode, salt: &str) {
    ComboBox::from_id_salt(salt)
        .selected_text(match mode {
            AccessMode::ReadOnly => "read-only",
            AccessMode::ReadWrite => "read-write",
        })
        .width(110.0)
        .show_ui(ui, |ui| {
            ui.selectable_value(mode, AccessMode::ReadOnly, "read-only");
            ui.selectable_value(mode, AccessMode::ReadWrite, "read-write");
        });
}

/// DragValue for an `Option<u32>` with an enable checkbox. When unchecked,
/// the field is None (server uses its own default).
fn optional_uint_field(
    ui: &mut egui::Ui,
    value: &mut Option<u32>,
    salt: &str,
    min: u32,
    max: u32,
    speed: f32,
) {
    let _ = salt;
    let mut enabled = value.is_some();
    ui.horizontal(|ui| {
        if ui.checkbox(&mut enabled, "").changed() {
            if enabled {
                *value = Some(min);
            } else {
                *value = None;
            }
        }
        ui.add_enabled_ui(enabled, |ui| {
            let mut v = value.unwrap_or(min);
            let resp = ui.add(DragValue::new(&mut v).range(min..=max).speed(speed));
            if resp.changed() && enabled {
                *value = Some(v);
            }
        });
    });
}

pub(super) fn section_heading(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).strong().size(14.0));
    ui.add_space(2.0);
}

pub(super) fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).small().color(Color32::from_gray(160)));
}

/// Convert a `BehaviorSnapshot` into the equivalent `BehaviorSummary`
/// shape the list cache holds. Mirrors the server-side
/// `Behavior::summary` but stays client-local because the list cache
/// is a client concern.
pub(super) fn behavior_summary_from_snapshot(snapshot: &BehaviorSnapshotProto) -> BehaviorSummary {
    let (name, description, trigger_kind) = match &snapshot.config {
        Some(cfg) => (
            cfg.name.clone(),
            cfg.description.clone(),
            Some(trigger_kind_label(&cfg.trigger).to_string()),
        ),
        None => (snapshot.behavior_id.clone(), None, None),
    };
    BehaviorSummary {
        behavior_id: snapshot.behavior_id.clone(),
        pod_id: snapshot.pod_id.clone(),
        name,
        description,
        trigger_kind,
        enabled: snapshot.state.enabled,
        run_count: snapshot.state.run_count,
        last_fired_at: snapshot.state.last_fired_at.clone(),
        load_error: snapshot.load_error.clone(),
    }
}

pub(super) fn trigger_kind_label(trigger: &TriggerSpec) -> &'static str {
    match trigger {
        TriggerSpec::Manual => "manual",
        TriggerSpec::Cron { .. } => "cron",
        TriggerSpec::Webhook { .. } => "webhook",
    }
}

pub(super) fn spec_type_label(spec: &HostEnvSpec) -> &'static str {
    match spec {
        HostEnvSpec::Landlock { .. } => "landlock",
        HostEnvSpec::Container { .. } => "container",
    }
}
