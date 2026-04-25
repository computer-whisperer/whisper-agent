//! Server-settings modal opened from the cog in the top bar. Hosts
//! four tabs (LLM backends / Host-env providers / Shared MCP hosts /
//! Server config) plus two stacked sub-forms (codex auth.json paste /
//! shared-MCP add-or-edit). Mutation paths require an admin token;
//! plain client tokens get a polite error from the server which
//! surfaces in the per-tab banner.
//!
//! Event flow: each tab/sub-form mutates `SettingsModalState`
//! (codex_rotate, shared_mcp_editor, server_config working buffer,
//! etc.) directly through the borrowed slot, and emits
//! `SettingsEvent`s for anything that needs `ChatApp` (correlation
//! ids, wire dispatch, sibling-modal slots).

use std::collections::{HashMap, HashSet};

use egui::{Color32, RichText, ScrollArea, TextEdit};
use whisper_agent_protocol::{
    BackendSummary, HostEnvProviderInfo, SharedMcpAuthInput, SharedMcpAuthPublic, SharedMcpHostInfo,
};

use super::super::widgets::{
    OAUTH_AVAILABLE, ProviderRowEvent, render_backend_settings_row, render_provider_row,
    render_shared_mcp_host_row, webui_origin,
};
use super::super::{
    CodexRotateState, ProviderRemovePending, ServerConfigEditorState, SettingsModalState,
    SettingsTab, SharedMcpAuthChoice, SharedMcpEditorMode, SharedMcpEditorState,
};

/// Side-channel actions a `render_settings_modal` call can emit.
/// Caller reduces each into a wire dispatch + companion-modal slot
/// mutation.
#[allow(clippy::large_enum_variant)]
pub(crate) enum SettingsEvent {
    /// Codex rotate sub-form: Save clicked. Caller mints a
    /// correlation, stamps `pending_correlation` on the (still-open)
    /// sub-form, and dispatches `UpdateCodexAuth`.
    CodexRotateSave { backend: String, contents: String },

    /// Server-config tab: first open triggered a fetch. Caller sends
    /// `FetchServerConfig` with the embedded correlation; the renderer
    /// has already stashed it in `state.fetch_correlation`.
    FetchServerConfig { correlation_id: String },
    /// Server-config tab: Save clicked. Caller dispatches
    /// `UpdateServerConfig`; correlation is in `state.save_correlation`.
    UpdateServerConfig {
        correlation_id: String,
        toml_text: String,
    },

    /// Shared-MCP tab: Remove confirmed on a row.
    RemoveSharedMcpHost { name: String },
    /// Shared-MCP sub-form: Save clicked in Add mode. Caller mints
    /// the correlation, stamps `pending_correlation` on the sub-form,
    /// and dispatches `AddSharedMcpHost`.
    AddSharedMcpHost {
        name: String,
        url: String,
        auth: SharedMcpAuthInput,
    },
    /// Shared-MCP sub-form: Save clicked in Edit mode. Caller mints
    /// the correlation, stamps it on the sub-form, and dispatches
    /// `UpdateSharedMcpHost`.
    UpdateSharedMcpHost {
        name: String,
        url: String,
        auth: Option<SharedMcpAuthInput>,
    },

    /// Host-env-providers tab: + Add provider clicked.
    OpenAddProvider,
    /// Host-env-providers tab: row Edit clicked.
    OpenEditProvider(HostEnvProviderInfo),
    /// Host-env-providers tab: row Remove confirmed.
    RemoveHostEnvProvider { name: String },
}

pub(crate) fn render_settings_modal(
    ctx: &egui::Context,
    slot: &mut Option<SettingsModalState>,
    backends: &[BackendSummary],
    shared_mcp_hosts: &[SharedMcpHostInfo],
    host_env_providers: &[HostEnvProviderInfo],
    provider_remove_armed: &mut HashSet<String>,
    provider_remove_pending: &HashMap<String, ProviderRemovePending>,
) -> Vec<SettingsEvent> {
    let mut events = Vec::new();
    let Some(mut modal) = slot.take() else {
        return events;
    };

    let mut open = true;
    let mut rotate_request: Option<String> = None;
    let mut shared_mcp_add_request = false;
    let mut shared_mcp_edit_request: Option<SharedMcpHostInfo> = None;
    let mut shared_mcp_remove_request: Option<String> = None;

    egui::Window::new("Server settings")
        .collapsible(false)
        .resizable(true)
        .default_width(520.0)
        .default_height(400.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(modal.active_tab == SettingsTab::Backends, "LLM backends")
                    .clicked()
                {
                    modal.active_tab = SettingsTab::Backends;
                }
                if ui
                    .selectable_label(
                        modal.active_tab == SettingsTab::HostEnvProviders,
                        "Host-env providers",
                    )
                    .clicked()
                {
                    modal.active_tab = SettingsTab::HostEnvProviders;
                }
                if ui
                    .selectable_label(
                        modal.active_tab == SettingsTab::SharedMcp,
                        "Shared MCP hosts",
                    )
                    .clicked()
                {
                    modal.active_tab = SettingsTab::SharedMcp;
                }
                if ui
                    .selectable_label(
                        modal.active_tab == SettingsTab::ServerConfig,
                        "Server config",
                    )
                    .clicked()
                {
                    modal.active_tab = SettingsTab::ServerConfig;
                }
            });
            ui.separator();
            ui.add_space(4.0);
            match modal.active_tab {
                SettingsTab::Backends => render_backends_tab(
                    ui,
                    backends,
                    &modal.codex_rotate_banner,
                    &mut rotate_request,
                ),
                SettingsTab::HostEnvProviders => render_host_env_providers_tab(
                    ui,
                    host_env_providers,
                    provider_remove_armed,
                    provider_remove_pending,
                    &mut events,
                ),
                SettingsTab::SharedMcp => render_shared_mcp_tab(
                    ui,
                    shared_mcp_hosts,
                    &modal.shared_mcp_banner,
                    &mut modal.shared_mcp_remove_armed,
                    &mut shared_mcp_add_request,
                    &mut shared_mcp_edit_request,
                    &mut shared_mcp_remove_request,
                ),
                SettingsTab::ServerConfig => {
                    render_server_config_tab(ui, &mut modal.server_config, &mut events);
                }
            }
        });

    if let Some(backend) = rotate_request {
        modal.codex_rotate_banner = None;
        modal.codex_rotate = Some(CodexRotateState {
            backend,
            contents: String::new(),
            error: None,
            pending_correlation: None,
        });
    }

    if shared_mcp_add_request {
        modal.shared_mcp_banner = None;
        modal.shared_mcp_editor = Some(SharedMcpEditorState {
            mode: SharedMcpEditorMode::Add,
            name: String::new(),
            url: String::new(),
            auth_choice: SharedMcpAuthChoice::Anonymous,
            bearer: String::new(),
            oauth_scope: String::new(),
            auth_kind_on_load: SharedMcpAuthPublic::None,
            error: None,
            pending_correlation: None,
            oauth_in_flight: false,
        });
    }
    if let Some(host) = shared_mcp_edit_request {
        modal.shared_mcp_banner = None;
        // On Edit the initial choice mirrors what the host has:
        // Bearer → staged Bearer with `keep existing` semantics;
        // None → Anonymous; Oauth2 → also shown as Bearer so the
        // user isn't offered "reopen the OAuth flow" here (Update
        // doesn't support Oauth2Start anyway).
        let auth_choice = match &host.auth {
            SharedMcpAuthPublic::None => SharedMcpAuthChoice::Anonymous,
            SharedMcpAuthPublic::Bearer => SharedMcpAuthChoice::Bearer,
            SharedMcpAuthPublic::Oauth2 { .. } => SharedMcpAuthChoice::Bearer,
        };
        modal.shared_mcp_editor = Some(SharedMcpEditorState {
            mode: SharedMcpEditorMode::Edit,
            name: host.name,
            url: host.url,
            auth_choice,
            bearer: String::new(),
            oauth_scope: String::new(),
            auth_kind_on_load: host.auth,
            error: None,
            pending_correlation: None,
            oauth_in_flight: false,
        });
    }
    if let Some(name) = shared_mcp_remove_request {
        events.push(SettingsEvent::RemoveSharedMcpHost { name });
    }

    // Paste-auth.json sub-form. Rendered outside the main window so
    // egui stacks it on top; closing it returns to the list.
    let mut keep_main_open = true;
    if modal.codex_rotate.is_some() {
        render_codex_rotate_subform(ctx, &mut modal, &mut keep_main_open, &mut events);
    }
    if modal.shared_mcp_editor.is_some() {
        render_shared_mcp_editor_subform(ctx, &mut modal, &mut events);
    }

    if open && keep_main_open {
        *slot = Some(modal);
    }
    events
}

/// Backends list inside the server-settings modal. Renders one row
/// per configured backend (name, kind, default-model, auth-mode)
/// plus a Rotate button for `chatgpt_subscription` backends that
/// opens the paste-auth.json sub-form. Credential material is
/// never displayed.
fn render_backends_tab(
    ui: &mut egui::Ui,
    backends: &[BackendSummary],
    banner: &Option<Result<String, (String, String)>>,
    rotate_request: &mut Option<String>,
) {
    ui.label(
        RichText::new(
            "Configured LLM backends the scheduler can dispatch to. \
             Auth mode names where the credential lives; the credential \
             itself is never sent to the client.",
        )
        .small()
        .color(Color32::from_gray(150)),
    );
    ui.add_space(6.0);

    if let Some(banner) = banner {
        match banner {
            Ok(backend) => {
                ui.colored_label(
                    Color32::from_rgb(0x88, 0xbb, 0x88),
                    format!("Rotated Codex credentials for `{backend}`."),
                );
            }
            Err((backend, detail)) => {
                ui.colored_label(
                    Color32::from_rgb(0xd0, 0x70, 0x70),
                    format!("Rotation failed for `{backend}`: {detail}"),
                );
            }
        }
        ui.add_space(6.0);
    }

    if backends.is_empty() {
        ui.label(
            RichText::new(
                "No backends configured. Seed them via [backends.*] in \
                 whisper-agent.toml.",
            )
            .small()
            .color(Color32::from_gray(150)),
        );
        return;
    }

    ScrollArea::vertical().show(ui, |ui| {
        for b in backends {
            render_backend_settings_row(ui, b, rotate_request);
            ui.add_space(2.0);
            ui.separator();
        }
    });
}

/// Paste-auth.json sub-form. Stays open while a request is in
/// flight (disables the Save button). Success / error banners live
/// on the parent modal, not here — we close on either outcome.
fn render_codex_rotate_subform(
    ctx: &egui::Context,
    modal: &mut SettingsModalState,
    keep_main_open: &mut bool,
    events: &mut Vec<SettingsEvent>,
) {
    let Some(mut sub) = modal.codex_rotate.take() else {
        return;
    };
    let mut open = true;
    let mut save_clicked = false;
    let mut cancel_clicked = false;
    let saving = sub.pending_correlation.is_some();

    egui::Window::new(format!("Rotate Codex credentials — {}", sub.backend))
        .collapsible(false)
        .resizable(true)
        .default_width(520.0)
        .default_height(360.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            ui.label(
                RichText::new(
                    "Paste the full contents of a working ~/.codex/auth.json \
                     (the file produced by `codex login` on a machine with a \
                     ChatGPT subscription). Server validates the JSON, writes \
                     it to the backend's configured path, and swaps the \
                     in-memory tokens — no restart needed.",
                )
                .small()
                .color(Color32::from_gray(160)),
            );
            ui.add_space(6.0);
            ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                ui.add(
                    TextEdit::multiline(&mut sub.contents)
                        .code_editor()
                        .hint_text("{\"tokens\": { ... }}")
                        .desired_width(f32::INFINITY)
                        .desired_rows(10),
                );
            });
            if let Some(err) = &sub.error {
                ui.add_space(6.0);
                ui.colored_label(Color32::from_rgb(0xd0, 0x70, 0x70), err);
            }
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                let enabled = !saving && !sub.contents.trim().is_empty();
                let label = if saving { "Saving…" } else { "Save" };
                if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
                    save_clicked = true;
                }
                if ui
                    .add_enabled(!saving, egui::Button::new("Cancel"))
                    .clicked()
                {
                    cancel_clicked = true;
                }
            });
        });

    if cancel_clicked {
        open = false;
    }

    if save_clicked {
        sub.error = None;
        events.push(SettingsEvent::CodexRotateSave {
            backend: sub.backend.clone(),
            contents: sub.contents.clone(),
        });
    }

    if !open {
        // Cancelled / closed — discard the in-flight form. If a
        // response later lands with the stashed correlation, the
        // handler harmlessly falls through (no match).
        let _ = sub;
    } else {
        modal.codex_rotate = Some(sub);
    }
    // The sub-form is always rendered on top of the main modal;
    // we never request the main modal to close from here.
    let _ = keep_main_open;
}

/// Server-config tab: raw TOML editor for `whisper-agent.toml`.
/// Admin-only; the server rejects the fetch/update calls from
/// non-admin connections and the error banner surfaces the
/// refusal. Fetches on first open (lazy) and keeps its working
/// draft across tab switches.
fn render_server_config_tab(
    ui: &mut egui::Ui,
    state_slot: &mut Option<ServerConfigEditorState>,
    events: &mut Vec<SettingsEvent>,
) {
    // Lazy init: first open triggers a FetchServerConfig.
    if state_slot.is_none() {
        let corr = local_correlation("server-config-fetch");
        events.push(SettingsEvent::FetchServerConfig {
            correlation_id: corr.clone(),
        });
        *state_slot = Some(ServerConfigEditorState {
            original: None,
            working: String::new(),
            fetch_correlation: Some(corr),
            save_correlation: None,
            banner: None,
        });
    }
    let state = state_slot.as_mut().expect("initialized above");

    ui.label(
        RichText::new(
            "Edits the server-level whisper-agent.toml. Backend-catalog \
             changes hot-swap immediately and cancel any thread using a \
             removed or modified backend. Other sections (shared_mcp_hosts, \
             host_env_providers, secrets, auth) persist to disk but require \
             a server restart.",
        )
        .small()
        .color(Color32::from_gray(170)),
    );
    ui.add_space(6.0);

    // Outcome banner from the last save, if any.
    match &state.banner {
        Some(Ok(summary)) => {
            ui.colored_label(Color32::from_rgb(0x88, 0xbb, 0x88), "Saved.");
            if !summary.cancelled_threads.is_empty() {
                ui.label(format!(
                    "Cancelled {} thread(s): {}",
                    summary.cancelled_threads.len(),
                    summary.cancelled_threads.join(", "),
                ));
            }
            if !summary.restart_required_sections.is_empty() {
                ui.colored_label(
                    Color32::from_rgb(0xd0, 0xb0, 0x70),
                    format!(
                        "Restart required for: {}",
                        summary.restart_required_sections.join(", "),
                    ),
                );
            }
            if !summary.pods_with_missing_backends.is_empty() {
                ui.colored_label(
                    Color32::from_rgb(0xd0, 0xb0, 0x70),
                    format!(
                        "Pods referencing removed backends: {}",
                        summary.pods_with_missing_backends.join(", "),
                    ),
                );
            }
            ui.add_space(6.0);
        }
        Some(Err(msg)) => {
            ui.colored_label(
                Color32::from_rgb(0xd0, 0x70, 0x70),
                format!("Save failed: {msg}"),
            );
            ui.add_space(6.0);
        }
        None => {}
    }

    let fetch_in_flight = state.fetch_correlation.is_some();
    let save_in_flight = state.save_correlation.is_some();

    if fetch_in_flight && state.original.is_none() {
        ui.label(
            RichText::new("Loading whisper-agent.toml…")
                .small()
                .color(Color32::from_gray(150)),
        );
        return;
    }

    ScrollArea::vertical().max_height(280.0).show(ui, |ui| {
        ui.add_enabled(
            !save_in_flight,
            egui::TextEdit::multiline(&mut state.working)
                .font(egui::TextStyle::Monospace)
                .code_editor()
                .desired_width(f32::INFINITY)
                .desired_rows(20),
        );
    });

    ui.add_space(6.0);
    let modified = state
        .original
        .as_deref()
        .map(|o| o != state.working)
        .unwrap_or(false);
    ui.horizontal(|ui| {
        let save_clicked = ui
            .add_enabled(modified && !save_in_flight, egui::Button::new("Save"))
            .clicked();
        let revert_clicked = ui
            .add_enabled(modified && !save_in_flight, egui::Button::new("Revert"))
            .clicked();
        if save_in_flight {
            ui.label(
                RichText::new("Applying…")
                    .small()
                    .color(Color32::from_gray(160)),
            );
        }
        if save_clicked {
            let corr = local_correlation("server-config-save");
            state.banner = None;
            state.save_correlation = Some(corr.clone());
            events.push(SettingsEvent::UpdateServerConfig {
                correlation_id: corr,
                toml_text: state.working.clone(),
            });
        }
        if revert_clicked && let Some(original) = state.original.as_deref() {
            state.working = original.to_string();
            state.banner = None;
        }
    });
}

/// Shared MCP hosts tab. Admin-only operations (add/edit/remove)
/// are rendered as buttons; a non-admin connection receives an
/// `Error` reply which the banner surfaces. Bearer tokens never
/// come back from the server — the UI only shows `auth_kind`.
#[allow(clippy::too_many_arguments)]
fn render_shared_mcp_tab(
    ui: &mut egui::Ui,
    shared_mcp_hosts: &[SharedMcpHostInfo],
    banner: &Option<Result<String, String>>,
    remove_armed: &mut HashSet<String>,
    add_request: &mut bool,
    edit_request: &mut Option<SharedMcpHostInfo>,
    remove_request: &mut Option<String>,
) {
    ui.label(
        RichText::new(
            "Shared MCP hosts the scheduler connects to at startup \
             (one singleton session per name, shared across all \
             threads that opt in). Third-party endpoints often \
             require a bearer token; Step 2 adds OAuth for servers \
             that need a browser-driven authorization flow.",
        )
        .small()
        .color(Color32::from_gray(150)),
    );
    ui.add_space(6.0);

    if let Some(banner) = banner {
        match banner {
            Ok(name) => {
                ui.colored_label(
                    Color32::from_rgb(0x88, 0xbb, 0x88),
                    format!("Saved `{name}`."),
                );
            }
            Err(detail) => {
                ui.colored_label(Color32::from_rgb(0xd0, 0x70, 0x70), detail.to_string());
            }
        }
        ui.add_space(6.0);
    }

    ui.horizontal(|ui| {
        if ui.button("+ Add host").clicked() {
            *add_request = true;
        }
    });
    ui.add_space(4.0);

    if shared_mcp_hosts.is_empty() {
        ui.label(
            RichText::new(
                "No shared MCP hosts configured. Add one above or \
                 seed via [shared_mcp_hosts] in whisper-agent.toml.",
            )
            .small()
            .color(Color32::from_gray(150)),
        );
        return;
    }

    ScrollArea::vertical().show(ui, |ui| {
        for host in shared_mcp_hosts {
            render_shared_mcp_host_row(ui, host, remove_armed, edit_request, remove_request);
            ui.add_space(2.0);
            ui.separator();
        }
    });
}

/// Paste-bearer / edit-url sub-form for Shared MCP hosts.
/// Dispatches `AddSharedMcpHost` or `UpdateSharedMcpHost` on Save
/// (via emitted events) and tracks the correlation so the matching
/// response routes the form closed (success) or an inline error
/// back into `sub.error` (failure).
fn render_shared_mcp_editor_subform(
    ctx: &egui::Context,
    modal: &mut SettingsModalState,
    events: &mut Vec<SettingsEvent>,
) {
    let Some(mut sub) = modal.shared_mcp_editor.take() else {
        return;
    };
    let mut open = true;
    let mut save_clicked = false;
    let mut cancel_clicked = false;
    let saving = sub.pending_correlation.is_some();
    let title = match sub.mode {
        SharedMcpEditorMode::Add => "Add shared MCP host".to_string(),
        SharedMcpEditorMode::Edit => format!("Edit shared MCP host — {}", sub.name),
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
                let editable = sub.mode == SharedMcpEditorMode::Add;
                ui.add_enabled(
                    editable,
                    TextEdit::singleline(&mut sub.name)
                        .hint_text("catalog name (e.g. 'slack', 'fetch')")
                        .desired_width(f32::INFINITY),
                );
            });
            if sub.mode == SharedMcpEditorMode::Edit {
                ui.label(
                    RichText::new(
                        "Name is fixed — pods and threads reference it. \
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
                    TextEdit::singleline(&mut sub.url)
                        .hint_text("https://mcp.example.com/...")
                        .desired_width(f32::INFINITY),
                );
            });
            ui.add_space(6.0);

            // Auth picker. Radio row — Anonymous / Bearer / OAuth.
            // OAuth is Add-only; on Edit it's silently disabled.
            ui.label(
                RichText::new("Authentication")
                    .small()
                    .color(Color32::from_gray(170)),
            );
            ui.horizontal(|ui| {
                ui.radio_value(&mut sub.auth_choice, SharedMcpAuthChoice::Anonymous, "None");
                ui.radio_value(&mut sub.auth_choice, SharedMcpAuthChoice::Bearer, "Bearer");
                let oauth_enabled = sub.mode == SharedMcpEditorMode::Add && OAUTH_AVAILABLE;
                let resp = ui
                    .add_enabled(
                        oauth_enabled,
                        egui::RadioButton::new(
                            sub.auth_choice == SharedMcpAuthChoice::Oauth2,
                            "OAuth",
                        ),
                    )
                    .on_disabled_hover_text(if !OAUTH_AVAILABLE {
                        "OAuth requires the browser webui"
                    } else {
                        "OAuth is only available when adding a host"
                    });
                if oauth_enabled && resp.clicked() {
                    sub.auth_choice = SharedMcpAuthChoice::Oauth2;
                }
            });
            ui.add_space(4.0);

            match sub.auth_choice {
                SharedMcpAuthChoice::Anonymous => {
                    if sub.mode == SharedMcpEditorMode::Edit
                        && matches!(sub.auth_kind_on_load, SharedMcpAuthPublic::Bearer)
                    {
                        ui.label(
                            RichText::new("Saving will clear the existing bearer token.")
                                .small()
                                .color(Color32::from_rgb(0xd0, 0xa0, 0x70)),
                        );
                    }
                }
                SharedMcpAuthChoice::Bearer => {
                    let had_bearer = matches!(sub.auth_kind_on_load, SharedMcpAuthPublic::Bearer);
                    ui.horizontal(|ui| {
                        ui.label("bearer");
                        ui.add(
                            TextEdit::singleline(&mut sub.bearer)
                                .password(true)
                                .hint_text(if had_bearer {
                                    "leave blank to keep existing"
                                } else {
                                    "paste bearer token"
                                })
                                .desired_width(f32::INFINITY),
                        );
                    });
                }
                SharedMcpAuthChoice::Oauth2 => {
                    ui.horizontal(|ui| {
                        ui.label("scope");
                        ui.add(
                            TextEdit::singleline(&mut sub.oauth_scope)
                                .hint_text("optional; space-separated (defaults to AS metadata)")
                                .desired_width(f32::INFINITY),
                        );
                    });
                    ui.label(
                        RichText::new(
                            "Saving opens the authorization server in a new \
                             tab. Grant consent there; this window stays open \
                             until the flow completes.",
                        )
                        .small()
                        .color(Color32::from_gray(150)),
                    );
                }
            }

            if sub.oauth_in_flight {
                ui.add_space(6.0);
                ui.colored_label(
                    Color32::from_rgb(0x88, 0xbb, 0xd8),
                    "Waiting for authorization… complete the flow in the \
                     browser tab that opened.",
                );
            }

            if let Some(err) = &sub.error {
                ui.add_space(6.0);
                ui.colored_label(Color32::from_rgb(0xd0, 0x70, 0x70), err);
            }
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                let save_enabled = !saving
                    && !sub.oauth_in_flight
                    && !sub.name.trim().is_empty()
                    && !sub.url.trim().is_empty();
                let label = if saving || sub.oauth_in_flight {
                    "Waiting…"
                } else if sub.auth_choice == SharedMcpAuthChoice::Oauth2 {
                    "Authorize"
                } else {
                    "Save"
                };
                if ui
                    .add_enabled(save_enabled, egui::Button::new(label))
                    .clicked()
                {
                    save_clicked = true;
                }
                if ui
                    .add_enabled(!saving, egui::Button::new("Cancel"))
                    .clicked()
                {
                    cancel_clicked = true;
                }
            });
        });

    if cancel_clicked {
        open = false;
    }

    if save_clicked {
        sub.error = None;
        let bearer = sub.bearer.trim().to_string();
        let scope = sub.oauth_scope.trim().to_string();
        let event = match (sub.mode, sub.auth_choice) {
            (SharedMcpEditorMode::Add, SharedMcpAuthChoice::Oauth2) => {
                // Grab the origin the webui is served from so the
                // redirect_uri on the authorization URL matches what
                // the browser will actually reach our /oauth/callback
                // on.
                let redirect_base = webui_origin();
                SettingsEvent::AddSharedMcpHost {
                    name: sub.name.trim().to_string(),
                    url: sub.url.trim().to_string(),
                    auth: SharedMcpAuthInput::Oauth2Start {
                        scope: (!scope.is_empty()).then_some(scope),
                        redirect_base,
                    },
                }
            }
            (SharedMcpEditorMode::Add, SharedMcpAuthChoice::Bearer) => {
                SettingsEvent::AddSharedMcpHost {
                    name: sub.name.trim().to_string(),
                    url: sub.url.trim().to_string(),
                    auth: SharedMcpAuthInput::Bearer { token: bearer },
                }
            }
            (SharedMcpEditorMode::Add, SharedMcpAuthChoice::Anonymous) => {
                SettingsEvent::AddSharedMcpHost {
                    name: sub.name.trim().to_string(),
                    url: sub.url.trim().to_string(),
                    auth: SharedMcpAuthInput::None,
                }
            }
            (SharedMcpEditorMode::Edit, choice) => {
                let auth = match choice {
                    SharedMcpAuthChoice::Anonymous => Some(SharedMcpAuthInput::None),
                    SharedMcpAuthChoice::Bearer => {
                        // Blank bearer with existing bearer on file
                        // = "keep existing" (send None to leave auth
                        // untouched); blank + no existing = explicit
                        // clear; non-blank = set bearer.
                        let had_bearer =
                            matches!(sub.auth_kind_on_load, SharedMcpAuthPublic::Bearer);
                        if bearer.is_empty() && had_bearer {
                            None
                        } else if bearer.is_empty() {
                            Some(SharedMcpAuthInput::None)
                        } else {
                            Some(SharedMcpAuthInput::Bearer { token: bearer })
                        }
                    }
                    // Oauth2 is disabled on Edit; this arm is
                    // unreachable in practice but the match wants
                    // exhaustiveness.
                    SharedMcpAuthChoice::Oauth2 => None,
                };
                SettingsEvent::UpdateSharedMcpHost {
                    name: sub.name.clone(),
                    url: sub.url.trim().to_string(),
                    auth,
                }
            }
        };
        events.push(event);
    }

    if !open {
        let _ = sub;
    } else {
        modal.shared_mcp_editor = Some(sub);
    }
}

/// Host-env-providers tab. Lists registered providers with origin +
/// reachability badges. "+ Add provider" opens the provider editor;
/// per-row Edit / Remove dispatch through the row event reducer.
fn render_host_env_providers_tab(
    ui: &mut egui::Ui,
    host_env_providers: &[HostEnvProviderInfo],
    provider_remove_armed: &mut HashSet<String>,
    provider_remove_pending: &HashMap<String, ProviderRemovePending>,
    events: &mut Vec<SettingsEvent>,
) {
    ui.label(
        RichText::new(
            "Sandbox daemons threads can provision isolated host envs \
             against. Each thread that binds a host env gets its own \
             landlock-isolated MCP host from the daemon named here.",
        )
        .small()
        .color(Color32::from_gray(150)),
    );
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui.button("+ Add provider").clicked() {
            events.push(SettingsEvent::OpenAddProvider);
        }
    });
    ui.add_space(4.0);

    if host_env_providers.is_empty() {
        ui.label(
            RichText::new(
                "No providers registered. Add one above or seed via \
                 [[host_env_providers]] in whisper-agent.toml.",
            )
            .small()
            .color(Color32::from_gray(150)),
        );
        return;
    }

    ScrollArea::vertical().show(ui, |ui| {
        for provider in host_env_providers {
            let pending = provider_remove_pending.get(&provider.name);
            let removing = pending.is_some_and(|p| p.error.is_none());
            let pending_error = pending.and_then(|p| p.error.as_deref());
            let armed = provider_remove_armed.contains(&provider.name);
            if let Some(event) = render_provider_row(ui, provider, armed, removing, pending_error) {
                match event {
                    ProviderRowEvent::EditRequested => {
                        events.push(SettingsEvent::OpenEditProvider(provider.clone()));
                    }
                    ProviderRowEvent::RemoveArmed => {
                        provider_remove_armed.insert(provider.name.clone());
                    }
                    ProviderRowEvent::RemoveConfirmed => {
                        provider_remove_armed.remove(&provider.name);
                        events.push(SettingsEvent::RemoveHostEnvProvider {
                            name: provider.name.clone(),
                        });
                    }
                }
            }
            ui.add_space(2.0);
            ui.separator();
        }
    });
}

/// Renderer-local correlation id. The wire response handler matches
/// by exact id only; our `settings-…` prefix can't collide with the
/// `c<n>` ids minted by `ChatApp::next_correlation_id`. We mint our
/// own here so the lazy-fetch path inside `render_server_config_tab`
/// (and the parallel save path) can stamp `state.fetch_correlation`
/// / `state.save_correlation` without needing a `&mut ChatApp`.
fn local_correlation(kind: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("settings-{kind}-{n:08x}")
}
