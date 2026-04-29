//! Stateless render helpers extracted from `app.rs`.
//!
//! Free functions, no `ChatApp` access. Small UI building blocks
//! (status chips, list rows, progress bars), the per-thread inspector
//! / failure / sudo banners that read from `TaskView` + `PendingSudo`,
//! plus the wasm/native cfg shims for `webui_origin`,
//! `open_in_new_tab`, and the `OAUTH_AVAILABLE` flag. Kept here so the
//! main app file stays focused on stateful flow and modal lifecycle.

use std::collections::HashMap;
use std::collections::HashSet;

use egui::{Color32, Grid, RichText, ScrollArea, TextEdit};
use whisper_agent_protocol::{
    BackendSummary, HostEnvBinding, HostEnvProviderInfo, HostEnvProviderOrigin,
    HostEnvReachability, HostEnvSpec, ResourceSnapshot, ResourceStateLabel, SharedMcpAuthPublic,
    SharedMcpHostInfo, ThreadStateLabel,
};

use super::editor_render::section_heading;
use super::{PendingSudo, TaskView};

/// One row in the Settings → LLM backends list. Shows name, kind,
/// default model, and auth-mode badge; for `chatgpt_subscription`
/// backends, a right-aligned "Rotate credentials" button that bubbles
/// the backend name up through `rotate_request` so the caller can open
/// the paste-`auth.json` sub-form. Never renders credential material.
pub(super) fn render_backend_settings_row(
    ui: &mut egui::Ui,
    backend: &BackendSummary,
    rotate_request: &mut Option<String>,
) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(&backend.name).strong());
        if backend.auth_mode.as_deref() == Some("chatgpt_subscription") {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("Rotate credentials")
                    .on_hover_text("Paste a fresh ~/.codex/auth.json to rotate the ChatGPT subscription tokens")
                    .clicked()
                {
                    *rotate_request = Some(backend.name.clone());
                }
            });
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label(
            RichText::new(format!("kind: {}", backend.kind))
                .small()
                .color(Color32::from_gray(170)),
        );
        ui.label(RichText::new("·").small().color(Color32::from_gray(120)));
        let auth = backend.auth_mode.as_deref().unwrap_or("(none)");
        ui.label(
            RichText::new(format!("auth: {auth}"))
                .small()
                .color(Color32::from_gray(170)),
        );
        if let Some(model) = backend.default_model.as_deref() {
            ui.label(RichText::new("·").small().color(Color32::from_gray(120)));
            ui.label(
                RichText::new(format!("default model: {model}"))
                    .small()
                    .color(Color32::from_gray(170)),
            );
        }
    });
}

/// One shared-MCP-host entry in the settings list. Name + live status
/// on the first line; URL, origin, auth-kind on the second; edit /
/// remove buttons on the third. Remove uses a two-click guard.
pub(super) fn render_shared_mcp_host_row(
    ui: &mut egui::Ui,
    host: &SharedMcpHostInfo,
    remove_armed: &mut HashSet<String>,
    edit_request: &mut Option<SharedMcpHostInfo>,
    remove_request: &mut Option<String>,
) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(&host.name).strong());
        let (label, color) = if host.connected {
            ("connected", Color32::from_rgb(0x88, 0xbb, 0x88))
        } else if !host.last_error.is_empty() {
            ("connect failed", Color32::from_rgb(0xd0, 0x70, 0x70))
        } else {
            ("not connected", Color32::from_gray(170))
        };
        ui.label(RichText::new(label).small().color(color));
    });
    ui.horizontal_wrapped(|ui| {
        ui.label(
            RichText::new(format!("url: {}", host.url))
                .small()
                .color(Color32::from_gray(170)),
        );
        ui.label(RichText::new("·").small().color(Color32::from_gray(120)));
        let origin = match host.origin {
            HostEnvProviderOrigin::Seeded => "seeded",
            HostEnvProviderOrigin::Manual => "manual",
            HostEnvProviderOrigin::RuntimeOverlay => "cli-overlay",
        };
        ui.label(
            RichText::new(format!("origin: {origin}"))
                .small()
                .color(Color32::from_gray(170)),
        );
        ui.label(RichText::new("·").small().color(Color32::from_gray(120)));
        let auth = match &host.auth {
            SharedMcpAuthPublic::None => "anonymous".to_string(),
            SharedMcpAuthPublic::Bearer => "bearer".to_string(),
            SharedMcpAuthPublic::Oauth2 { issuer, .. } => format!("oauth2 ({issuer})"),
        };
        ui.label(
            RichText::new(format!("auth: {auth}"))
                .small()
                .color(Color32::from_gray(170)),
        );
        ui.label(RichText::new("·").small().color(Color32::from_gray(120)));
        // Prefix is a three-state in storage but the operator only
        // needs the rendered shape: <prefix>_<tool>, or "(none)" when
        // the catalog has explicitly disabled prefixing.
        let prefix_label = match host.prefix.as_deref() {
            None => format!("prefix: {}_", host.name),
            Some("") => "prefix: (none)".to_string(),
            Some(p) => format!("prefix: {p}_"),
        };
        ui.label(
            RichText::new(prefix_label)
                .small()
                .color(Color32::from_gray(170)),
        );
    });
    if !host.last_error.is_empty() {
        ui.label(
            RichText::new(&host.last_error)
                .small()
                .color(Color32::from_rgb(0xd0, 0x70, 0x70)),
        );
    }
    let is_overlay = matches!(host.origin, HostEnvProviderOrigin::RuntimeOverlay);
    ui.horizontal(|ui| {
        // Edit is disabled on both CLI overlays and OAuth entries:
        // overlays can't mutate at runtime (shadowed by catalog);
        // OAuth entries would have their tokens silently overwritten
        // by the Bearer/Anonymous fields in the Edit form, wiping the
        // whole authorization handshake. For OAuth, the only
        // supported "edit" is remove + re-add (re-running the flow).
        let is_oauth = matches!(host.auth, SharedMcpAuthPublic::Oauth2 { .. });
        let edit_enabled = !is_overlay && !is_oauth;
        let edit_hover = if is_overlay {
            "CLI --shared-mcp-host overlays can't be edited at runtime"
        } else if is_oauth {
            "OAuth hosts can't be edited — remove and re-add to change URL or re-authorize"
        } else {
            "Edit url or auth"
        };
        if ui
            .add_enabled(edit_enabled, egui::Button::new("Edit").small())
            .on_hover_text(edit_hover)
            .clicked()
        {
            *edit_request = Some(host.clone());
        }
        let armed = remove_armed.contains(&host.name);
        let remove_label = if armed { "Confirm remove" } else { "Remove" };
        let remove_hover = if is_overlay {
            "CLI overlay — restart without the flag to unregister"
        } else {
            "Remove from catalog. Fails if any thread is currently using this host."
        };
        if ui
            .add_enabled(!is_overlay, egui::Button::new(remove_label).small())
            .on_hover_text(remove_hover)
            .clicked()
        {
            if armed {
                remove_armed.remove(&host.name);
                *remove_request = Some(host.name.clone());
            } else {
                remove_armed.insert(host.name.clone());
            }
        }
    });
}

/// The origin the webui is served from (e.g. `http://127.0.0.1:8080`).
/// Used as the base for the OAuth redirect URI — we hand it to the
/// server so it builds a redirect URL the browser will actually be
/// able to reach. Only meaningful on the wasm32 (browser) target;
/// desktop builds return an empty string and the UI gates the OAuth
/// option off.
#[cfg(target_arch = "wasm32")]
pub(super) fn webui_origin() -> String {
    web_sys::window()
        .and_then(|w| w.location().origin().ok())
        .unwrap_or_default()
}
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn webui_origin() -> String {
    String::new()
}

/// Open `url` in a new browser tab. Used by the OAuth flow to hand
/// the user off to the authorization server. No-op on desktop —
/// native OAuth would route through the system browser via
/// a crate like `webbrowser`, which we'll add when desktop needs it.
#[cfg(target_arch = "wasm32")]
pub(super) fn open_in_new_tab(url: &str) {
    if let Some(window) = web_sys::window() {
        let _ = window.open_with_url_and_target(url, "_blank");
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn open_in_new_tab(_url: &str) {}

/// True when OAuth flows are actually usable — the webui is in a
/// browser that can open a new tab + receive a redirect on the
/// server's `/oauth/callback` route. False on desktop (no browser
/// to drive the flow); the UI disables the OAuth radio option.
#[cfg(target_arch = "wasm32")]
pub(super) const OAUTH_AVAILABLE: bool = true;
#[cfg(not(target_arch = "wasm32"))]
pub(super) const OAUTH_AVAILABLE: bool = false;

/// Three-section list of the server's currently-tracked resources
/// (host envs, MCP hosts, backends). Each section is a collapsible
/// header that defaults open; entries inside are sorted by id so the
/// order is stable across snapshots.
pub(super) fn render_resource_list(
    ui: &mut egui::Ui,
    resources: &HashMap<String, ResourceSnapshot>,
) {
    let mut host_envs: Vec<&ResourceSnapshot> = Vec::new();
    let mut mcp_hosts: Vec<&ResourceSnapshot> = Vec::new();
    let mut backends: Vec<&ResourceSnapshot> = Vec::new();
    for r in resources.values() {
        match r {
            ResourceSnapshot::HostEnv { .. } => host_envs.push(r),
            ResourceSnapshot::McpHost { .. } => mcp_hosts.push(r),
            ResourceSnapshot::Backend { .. } => backends.push(r),
        }
    }
    host_envs.sort_by_key(|r| r.id().to_string());
    mcp_hosts.sort_by_key(|r| r.id().to_string());
    backends.sort_by_key(|r| r.id().to_string());

    ScrollArea::vertical().show(ui, |ui| {
        egui::CollapsingHeader::new(format!("Host envs ({})", host_envs.len()))
            .default_open(true)
            .show(ui, |ui| {
                for r in &host_envs {
                    render_resource_row(ui, r);
                }
                if host_envs.is_empty() {
                    ui.label(
                        RichText::new("(none)")
                            .color(Color32::from_gray(140))
                            .small(),
                    );
                }
            });
        egui::CollapsingHeader::new(format!("MCP Hosts ({})", mcp_hosts.len()))
            .default_open(true)
            .show(ui, |ui| {
                for r in &mcp_hosts {
                    render_resource_row(ui, r);
                }
                if mcp_hosts.is_empty() {
                    ui.label(
                        RichText::new("(none)")
                            .color(Color32::from_gray(140))
                            .small(),
                    );
                }
            });
        egui::CollapsingHeader::new(format!("Backends ({})", backends.len()))
            .default_open(true)
            .show(ui, |ui| {
                for r in &backends {
                    render_resource_row(ui, r);
                }
                if backends.is_empty() {
                    ui.label(
                        RichText::new("(none)")
                            .color(Color32::from_gray(140))
                            .small(),
                    );
                }
            });
    });
}

fn render_resource_row(ui: &mut egui::Ui, resource: &ResourceSnapshot) {
    let (label, sub, state, users) = match resource {
        ResourceSnapshot::HostEnv {
            id,
            provider,
            spec,
            state,
            users,
            ..
        } => {
            let sub = format!("{provider} · {}", spec_label(spec));
            (id.clone(), sub, *state, users.len())
        }
        ResourceSnapshot::McpHost {
            id,
            label,
            url,
            tools,
            state,
            users,
            ..
        } => (
            label.clone(),
            format!("{} · {} tools · {}", id, tools.len(), url),
            *state,
            users.len(),
        ),
        ResourceSnapshot::Backend {
            name,
            backend_kind,
            default_model,
            state,
            users,
            ..
        } => {
            let model = default_model.as_deref().unwrap_or("(no default)");
            (
                name.clone(),
                format!("{backend_kind} · {model}"),
                *state,
                users.len(),
            )
        }
    };
    let (chip, chip_color) = resource_state_chip(state);
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).strong());
        ui.label(RichText::new(format!("[{chip}]")).color(chip_color).small());
        ui.label(
            RichText::new(format!("{users} users"))
                .color(Color32::from_gray(160))
                .small(),
        );
    });
    if !sub.is_empty() {
        ui.label(RichText::new(sub).color(Color32::from_gray(150)).small());
    }
    ui.add_space(4.0);
}

pub(super) fn spec_label(spec: &HostEnvSpec) -> String {
    match spec {
        HostEnvSpec::Container { image, .. } => format!("container: {image}"),
        HostEnvSpec::Landlock { allowed_paths, .. } => {
            format!("landlock · {} paths", allowed_paths.len())
        }
    }
}

pub(super) fn provider_origin_chip(origin: HostEnvProviderOrigin) -> (&'static str, Color32) {
    match origin {
        HostEnvProviderOrigin::Seeded => ("seeded", Color32::from_rgb(140, 160, 200)),
        HostEnvProviderOrigin::Manual => ("manual", Color32::from_rgb(140, 200, 160)),
        HostEnvProviderOrigin::RuntimeOverlay => ("cli-overlay", Color32::from_rgb(200, 180, 120)),
    }
}

pub(super) fn provider_reachability_chip(
    r: &HostEnvReachability,
) -> (&'static str, Color32, Option<String>) {
    match r {
        HostEnvReachability::Unknown => ("probing", Color32::from_gray(150), None),
        HostEnvReachability::Reachable { at } => (
            "reachable",
            Color32::from_rgb(120, 180, 120),
            Some(format!("last probe OK · {at}")),
        ),
        HostEnvReachability::Unreachable { since, last_error } => (
            "unreachable",
            Color32::from_rgb(220, 110, 110),
            Some(format!("since {since} · {last_error}")),
        ),
    }
}

pub(super) fn resource_state_chip(state: ResourceStateLabel) -> (&'static str, Color32) {
    match state {
        ResourceStateLabel::Provisioning => ("provisioning", Color32::from_rgb(180, 160, 90)),
        ResourceStateLabel::Ready => ("ready", Color32::from_rgb(120, 180, 120)),
        ResourceStateLabel::Errored => ("errored", Color32::from_rgb(200, 110, 110)),
        ResourceStateLabel::Lost => ("lost", Color32::from_rgb(210, 140, 90)),
        ResourceStateLabel::TornDown => ("torn down", Color32::from_gray(140)),
    }
}

/// Draw the transient prefill-progress indicator. Shows only while a
/// llamacpp-backed turn is ingesting its prompt, cleared on first delta
/// by the [`ChatApp::handle_wire`] reducers. Formatted like
/// `prefilling 3,200 / 15,000 tokens · 21%` so the user can see both
/// the absolute numbers (helps for "how big is my context?") and the
/// fraction (helps for "how much longer?"). The bar itself carries no
/// text because egui's built-in text rendering inside the bar clashes
/// with the explicit label we already draw above it.
pub(super) fn render_prefill_progress(ui: &mut egui::Ui, processed: u32, total: u32) {
    ui.add_space(4.0);
    egui::Frame::group(ui.style())
        .fill(Color32::from_rgb(32, 40, 52))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                let fraction = if total == 0 {
                    0.0
                } else {
                    (processed as f32 / total as f32).clamp(0.0, 1.0)
                };
                let pct = (fraction * 100.0).round() as u32;
                ui.label(
                    RichText::new(format!(
                        "prefilling {} / {} tokens · {}%",
                        format_thousands(processed),
                        format_thousands(total),
                        pct,
                    ))
                    .color(Color32::from_rgb(180, 200, 230))
                    .small(),
                );
                ui.add_space(2.0);
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_height(6.0)
                        .fill(Color32::from_rgb(90, 140, 220)),
                );
            });
        });
    ui.add_space(4.0);
}

/// Format a non-negative integer with comma thousand-separators.
/// Standalone rather than pulled from a crate so the webui doesn't
/// grow a dep for two callers.
fn format_thousands(n: u32) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

pub(super) fn state_chip(state: ThreadStateLabel) -> (&'static str, Color32) {
    match state {
        ThreadStateLabel::Idle => ("idle", Color32::from_gray(160)),
        ThreadStateLabel::Working => ("working", Color32::from_rgb(120, 180, 240)),
        ThreadStateLabel::Completed => ("completed", Color32::from_rgb(120, 200, 140)),
        ThreadStateLabel::Failed => ("failed", Color32::from_rgb(220, 110, 110)),
        ThreadStateLabel::Cancelled => ("cancelled", Color32::from_rgb(180, 140, 140)),
    }
}

/// Collapsible inspector at the top of the thread view. Surfaces
/// everything a snapshot carries that isn't already visible as a
/// conversation item: pod/thread identity, timestamps, bindings,
/// sampling caps, approval policy, trigger origin, and the full
/// system prompt. Collapsed by default — most sessions never open
/// it. egui's `CollapsingHeader` persists open/closed state by
/// `id_salt` so flipping the arrow survives repaints.
pub(super) fn render_thread_context_inspector(ui: &mut egui::Ui, thread_id: &str, view: &TaskView) {
    let salt = format!("thread-context-{thread_id}");
    egui::CollapsingHeader::new(
        RichText::new("Thread context")
            .small()
            .color(Color32::from_gray(180)),
    )
    .id_salt(salt)
    .default_open(false)
    .show(ui, |ui| {
        let inspector = &view.inspector;
        Grid::new(format!("thread-context-grid-{thread_id}"))
            .num_columns(2)
            .min_col_width(120.0)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                kv_row(ui, "thread_id", thread_id);
                kv_row(ui, "pod_id", &view.summary.pod_id);
                kv_row(ui, "state", state_chip(view.summary.state).0);
                if let Some(title) = view.summary.title.as_deref() {
                    kv_row(ui, "title", title);
                }
                if !inspector.created_at.is_empty() {
                    kv_row(ui, "created_at", &inspector.created_at);
                }
                if !view.summary.last_active.is_empty() {
                    kv_row(ui, "last_active", &view.summary.last_active);
                }
                let backend_val = if view.backend.is_empty() {
                    "(server default)".to_string()
                } else {
                    view.backend.clone()
                };
                let model_val = if view.model.is_empty() {
                    "(backend default)".to_string()
                } else {
                    view.model.clone()
                };
                kv_row(ui, "backend", &backend_val);
                kv_row(ui, "model", &model_val);
                if inspector.max_tokens > 0 {
                    kv_row(ui, "max_tokens", &inspector.max_tokens.to_string());
                }
                if inspector.max_turns > 0 {
                    kv_row(ui, "max_turns", &inspector.max_turns.to_string());
                }
                let host_env_label = if inspector.bindings.host_env.is_empty() {
                    "(none — shared MCPs only)".to_string()
                } else {
                    inspector
                        .bindings
                        .host_env
                        .iter()
                        .map(|b| match b {
                            HostEnvBinding::Named { name } => name.clone(),
                            HostEnvBinding::Inline { provider, .. } => {
                                format!("(inline, provider = {provider})")
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                kv_row(ui, "host_env", &host_env_label);
                let mcp_label = if inspector.bindings.mcp_hosts.is_empty() {
                    "(none)".to_string()
                } else {
                    inspector.bindings.mcp_hosts.join(", ")
                };
                kv_row(ui, "mcp_hosts", &mcp_label);
                kv_row(
                    ui,
                    "total_usage_in",
                    &view.total_usage.input_tokens.to_string(),
                );
                kv_row(
                    ui,
                    "total_usage_out",
                    &view.total_usage.output_tokens.to_string(),
                );
                if view.total_usage.cache_read_input_tokens > 0
                    || view.total_usage.cache_creation_input_tokens > 0
                {
                    kv_row(
                        ui,
                        "cache (read/write)",
                        &format!(
                            "{}/{}",
                            view.total_usage.cache_read_input_tokens,
                            view.total_usage.cache_creation_input_tokens
                        ),
                    );
                }
            });

        ui.add_space(6.0);
        section_heading(ui, "Scope");
        render_scope_summary(ui, thread_id, &inspector.scope);

        if let Some(origin) = inspector.origin.as_ref() {
            ui.add_space(6.0);
            section_heading(ui, "Trigger origin");
            Grid::new(format!("thread-origin-grid-{thread_id}"))
                .num_columns(2)
                .min_col_width(120.0)
                .spacing([12.0, 4.0])
                .show(ui, |ui| {
                    kv_row(ui, "behavior_id", &origin.behavior_id);
                    kv_row(ui, "fired_at", &origin.fired_at);
                    if !origin.trigger_payload.is_null() {
                        let payload_text = serde_json::to_string_pretty(&origin.trigger_payload)
                            .unwrap_or_default();
                        ui.label("trigger_payload");
                        ui.add(
                            TextEdit::multiline(&mut payload_text.as_str())
                                .code_editor()
                                .desired_rows(payload_text.lines().count().clamp(1, 8) as usize)
                                .desired_width(f32::INFINITY),
                        );
                        ui.end_row();
                    }
                });
        }

        // System prompt + tool manifest used to render here; they now
        // live as `Role::System` / `Role::Tools` messages at the head
        // of the conversation and render inline in the chat log
        // (default-collapsed).
    });
    ui.add_space(6.0);
}

/// Render one row of the inspector's key-value grid. Keys right-aligned
/// in a muted tone, values left-aligned as plain text.
fn kv_row(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.label(RichText::new(key).small().color(Color32::from_gray(160)));
    ui.label(RichText::new(value).small());
    ui.end_row();
}

/// Compact projection of a thread's `Scope` into the inspector. Shows
/// the bindings-side admission sets, the typed caps, and whether the
/// thread has an interactive escalation channel. Tool dispositions ride
/// alongside the catalog in the chat log so they aren't duplicated
/// here — the common "what can this thread do?" question is typically
/// about bindings + caps, which the rest of the UI doesn't surface.
fn render_scope_summary(
    ui: &mut egui::Ui,
    thread_id: &str,
    scope: &whisper_agent_protocol::permission::Scope,
) {
    use whisper_agent_protocol::permission::{Escalation, SetOrAll};

    fn fmt_set(set: &SetOrAll<String>) -> String {
        match set {
            SetOrAll::All => "(all)".to_string(),
            SetOrAll::Only { items } if items.is_empty() => "(none)".to_string(),
            SetOrAll::Only { items } => items.iter().cloned().collect::<Vec<_>>().join(", "),
        }
    }

    Grid::new(format!("thread-scope-grid-{thread_id}"))
        .num_columns(2)
        .min_col_width(120.0)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            kv_row(ui, "backends", &fmt_set(&scope.backends));
            kv_row(ui, "host_envs", &fmt_set(&scope.host_envs));
            kv_row(ui, "mcp_hosts", &fmt_set(&scope.mcp_hosts));
            kv_row(ui, "tools default", &format!("{:?}", scope.tools.default));
            if !scope.tools.overrides.is_empty() {
                let overrides = scope
                    .tools
                    .overrides
                    .iter()
                    .map(|(k, v)| format!("{k}={v:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                kv_row(ui, "tools overrides", &overrides);
            }
            kv_row(ui, "pod_modify", &format!("{:?}", scope.pod_modify));
            kv_row(ui, "dispatch", &format!("{:?}", scope.dispatch));
            kv_row(ui, "behaviors", &format!("{:?}", scope.behaviors));
            let esc = match scope.escalation {
                Escalation::Interactive { .. } => "interactive",
                Escalation::None => "autonomous",
            };
            kv_row(ui, "escalation", esc);
        });
}

/// Render one three-way-approval banner per pending sudo for
/// `thread_id`. Three buttons: Approve (once), Remember (approve +
/// admit the tool name for the rest of the thread), Reject. Args are
/// pretty-printed below the tool name so the user can see exactly what
/// the model wants to run.
pub(super) fn render_sudo_banners(
    ui: &mut egui::Ui,
    thread_id: &str,
    pending: &HashMap<u64, PendingSudo>,
    reject_drafts: &mut HashMap<u64, String>,
    decisions_out: &mut Vec<(
        u64,
        whisper_agent_protocol::permission::SudoDecision,
        Option<String>,
    )>,
) {
    use whisper_agent_protocol::permission::SudoDecision;

    let mut ids: Vec<u64> = pending
        .iter()
        .filter_map(|(id, s)| (s.thread_id == thread_id).then_some(*id))
        .collect();
    ids.sort_unstable();

    for function_id in ids {
        let Some(s) = pending.get(&function_id) else {
            continue;
        };
        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(40, 40, 58))
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("sudo requested")
                                .color(Color32::from_rgb(180, 200, 250))
                                .strong(),
                        );
                        ui.label(
                            RichText::new(format!("fn #{function_id}"))
                                .small()
                                .color(Color32::from_gray(180)),
                        );
                    });
                    ui.add_space(2.0);
                    ui.label(
                        RichText::new(format!("tool: `{}`", s.tool_name))
                            .color(Color32::from_rgb(220, 230, 250))
                            .monospace(),
                    );
                    let args_text = serde_json::to_string_pretty(&s.args)
                        .unwrap_or_else(|_| s.args.to_string());
                    ui.add_space(2.0);
                    egui::ScrollArea::vertical()
                        .id_salt(("sudo_args_scroll", function_id))
                        .max_height(200.0)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(args_text)
                                    .small()
                                    .monospace()
                                    .color(Color32::from_gray(200)),
                            );
                        });
                    if !s.reason.trim().is_empty() {
                        ui.add_space(2.0);
                        ui.label(
                            RichText::new(format!("reason: {}", s.reason))
                                .small()
                                .color(Color32::from_gray(210)),
                        );
                    }
                    ui.add_space(4.0);
                    let draft = reject_drafts.entry(function_id).or_default();
                    ui.horizontal(|ui| {
                        if ui.button("Approve").clicked() {
                            decisions_out.push((function_id, SudoDecision::ApproveOnce, None));
                        }
                        if ui.button("Remember").clicked() {
                            decisions_out.push((function_id, SudoDecision::ApproveRemember, None));
                        }
                        if ui.button("Reject").clicked() {
                            let reason = draft.trim();
                            let reason = (!reason.is_empty()).then(|| reason.to_string());
                            decisions_out.push((function_id, SudoDecision::Reject, reason));
                        }
                        ui.add(
                            TextEdit::singleline(draft)
                                .hint_text("reject reason (optional)")
                                .desired_width(ui.available_width()),
                        );
                    });
                });
            });
        ui.add_space(4.0);
    }
}

/// Persistent banner for a task that's entered the `Failed` state.
/// Survives resnapshot because `failure` is captured from the
/// snapshot itself rather than derived from the per-event items list.
pub(super) fn render_failure_banner(ui: &mut egui::Ui, view: &TaskView) {
    let Some(detail) = view.failure.as_deref() else {
        return;
    };
    if view.summary.state != ThreadStateLabel::Failed {
        return;
    }
    egui::Frame::group(ui.style())
        .fill(Color32::from_rgb(64, 28, 28))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new("thread failed")
                        .color(Color32::from_rgb(240, 140, 140))
                        .strong(),
                );
                ui.add_space(2.0);
                ui.label(
                    RichText::new(detail)
                        .color(Color32::from_rgb(240, 210, 210))
                        .monospace(),
                );
            });
        });
    ui.add_space(4.0);
}

/// Side-channel actions a `render_provider_row` call can emit. The
/// caller pairs each event with the row's `info` (which the renderer
/// doesn't re-emit) and turns it into modal-state changes / wire
/// messages. Mirrors the `ChatItemEvent` pattern in `chat_render` so
/// the row renderer doesn't need `&mut ChatApp`.
pub(super) enum ProviderRowEvent {
    /// Edit clicked — caller opens the provider editor preloaded from
    /// this row's info.
    EditRequested,
    /// Remove clicked while the row was not yet armed — caller arms it
    /// (next click confirms).
    RemoveArmed,
    /// Remove clicked while the row was armed — caller dispatches the
    /// wire RPC and records a `ProviderRemovePending` entry.
    RemoveConfirmed,
}

/// One row in the Providers tab. The display state (`armed`,
/// `removing`, `pending_error`) is computed by the caller from
/// `ChatApp.provider_remove_armed` / `provider_remove_pending`; the
/// renderer reads them and emits at most one `ProviderRowEvent` per
/// frame, which the caller reduces back into mutations.
pub(super) fn render_provider_row(
    ui: &mut egui::Ui,
    info: &HostEnvProviderInfo,
    armed: bool,
    removing: bool,
    pending_error: Option<&str>,
) -> Option<ProviderRowEvent> {
    let (origin_chip, origin_color) = provider_origin_chip(info.origin);
    let (reach_chip, reach_color, reach_detail) = provider_reachability_chip(&info.reachability);
    ui.horizontal(|ui| {
        ui.label(RichText::new(&info.name).strong());
        ui.label(
            RichText::new(format!("[{origin_chip}]"))
                .color(origin_color)
                .small(),
        );
        ui.label(
            RichText::new(format!("[{reach_chip}]"))
                .color(reach_color)
                .small(),
        );
        if info.has_token {
            ui.label(RichText::new("auth").color(Color32::from_gray(160)).small());
        } else {
            ui.label(
                RichText::new("no auth")
                    .color(Color32::from_rgb(180, 140, 90))
                    .small(),
            );
        }
    });
    ui.label(
        RichText::new(&info.url)
            .color(Color32::from_gray(160))
            .small()
            .monospace(),
    );
    if let Some(detail) = &reach_detail {
        ui.label(RichText::new(detail).color(reach_color).small());
    }

    // Actions. Runtime-overlay entries (CLI flags) can't be edited or
    // removed via the UI — the corresponding server-side command
    // refuses them. Show the buttons disabled with a hint so the user
    // isn't confused.
    let is_overlay = info.origin == HostEnvProviderOrigin::RuntimeOverlay;
    let mut event: Option<ProviderRowEvent> = None;

    ui.horizontal(|ui| {
        let edit_btn = ui.add_enabled(!is_overlay && !removing, egui::Button::new("Edit"));
        if edit_btn.clicked() {
            event = Some(ProviderRowEvent::EditRequested);
        }

        let remove_label = if removing {
            "Removing…"
        } else if armed {
            "Confirm remove"
        } else {
            "Remove"
        };
        let remove_btn = ui.add_enabled(!is_overlay && !removing, egui::Button::new(remove_label));
        if remove_btn.clicked() {
            event = Some(if armed {
                ProviderRowEvent::RemoveConfirmed
            } else {
                ProviderRowEvent::RemoveArmed
            });
        }
        if is_overlay {
            ui.label(
                RichText::new("CLI-overlay (drop --host-env-provider flag to manage here)")
                    .small()
                    .color(Color32::from_gray(150)),
            );
        }
    });
    if let Some(err) = pending_error {
        ui.label(
            RichText::new(format!("remove failed: {err}"))
                .small()
                .color(Color32::from_rgb(220, 80, 80)),
        );
    }
    event
}
