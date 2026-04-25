//! Left-panel pod-grouped thread tree + per-pod sub-sections
//! (Interactive threads, Behaviors with nested-thread expansion),
//! the file-browser modal, and the small open-X helpers that route
//! sidebar clicks into the appropriate modal slot or wire dispatch.
//!
//! Unlike `widgets` and `modals`, the sidebar code is implemented as
//! `impl ChatApp` extension methods rather than free functions: it
//! reads dozens of `ChatApp` fields (pods / tasks / task_order /
//! behaviors_by_pod / pod_files / dispatch_subscribed_threads /
//! collapsed_pods / archive_armed_pod / delete_armed_behavior /
//! every modal slot / expanded_* sets) and is single-use, so the
//! free-function-with-event-vec shape would balloon the API surface
//! without buying isolated testability. Multiple `impl ChatApp`
//! blocks across submodules is a normal Rust pattern.

use std::collections::{HashMap, HashSet};

use egui::{Color32, RichText, ScrollArea};
use whisper_agent_protocol::{
    BehaviorSnapshot as BehaviorSnapshotProto, BehaviorSummary, ClientToServer, FsEntry,
};

use super::editor_render::behavior_summary_from_snapshot;
use super::widgets::state_chip;
use super::{
    BehaviorEditorModalState, BehaviorEditorTab, BehaviorRowAction, ChatApp, FileViewerModalState,
    JsonViewerModalState, NewBehaviorModalState, NewPodModalState, PodEditorModalState,
    PodFileDispatch, SIDEBAR_BODY_COLOR, SIDEBAR_DANGER_COLOR, SIDEBAR_DIM_COLOR,
    SIDEBAR_ERROR_TEXT_COLOR, SIDEBAR_MUTED_COLOR, SIDEBAR_WARNING_COLOR, THREAD_ROW_PREVIEW_COUNT,
    add_sidebar_thread_row, format_relative_time, sidebar_button, sidebar_icon_button,
    sidebar_subsection_header,
};

impl ChatApp {
    /// Renders the left panel as a pod-grouped tree: each pod gets a header
    /// (with display name + thread count) and its threads nest underneath
    /// as selectable rows. Threads whose `pod_id` doesn't match any known
    /// pod get bucketed under a synthetic "(unknown pod)" group — happens
    /// in practice when a thread arrives via `ThreadCreated` before the
    /// `ListPods` round-trip completes.
    pub(super) fn render_thread_tree(&mut self, ui: &mut egui::Ui) {
        // Scale the sidebar's Small text style up so subsection headers,
        // thread rows, and sub-buttons read at a comfortable size while
        // pod-name headings (TextStyle::Body) stay at their default. The
        // mutation is scoped to this ui via Arc COW.
        if let Some(small) = ui.style_mut().text_styles.get_mut(&egui::TextStyle::Small) {
            small.size *= 1.2;
        }

        if ui.button("+ New pod").clicked() {
            self.new_pod_modal = Some(NewPodModalState::new());
        }
        ui.separator();

        // Group threads by pod_id, preserving the existing newest-first
        // sort within each pod (task_order is already created_at desc).
        let order = self.task_order.clone();
        let mut by_pod: HashMap<String, Vec<String>> = HashMap::new();
        for thread_id in &order {
            let Some(view) = self.tasks.get(thread_id) else {
                continue;
            };
            by_pod
                .entry(view.summary.pod_id.clone())
                .or_default()
                .push(thread_id.clone());
        }

        // Pod header order: known pods alphabetically by display name, then
        // the synthetic "(unknown pod)" bucket if non-empty. Stable across
        // renders so headers don't jitter as state churns.
        let mut pod_ids: Vec<String> = self.pods.keys().cloned().collect();
        pod_ids.sort_by(|a, b| {
            let na = self.pods.get(a).map(|p| p.name.as_str()).unwrap_or(a);
            let nb = self.pods.get(b).map(|p| p.name.as_str()).unwrap_or(b);
            na.cmp(nb)
        });
        // Surface any pod_ids that have threads but no PodSummary — typically
        // new threads created before ListPods returned.
        for pid in by_pod.keys() {
            if !self.pods.contains_key(pid) && !pod_ids.contains(pid) {
                pod_ids.push(pid.clone());
            }
        }

        ScrollArea::vertical().show(ui, |ui| {
            for pid in &pod_ids {
                self.render_pod_section(ui, pid, by_pod.get(pid).map(|v| v.as_slice()));
            }
        });
    }

    fn render_pod_section(
        &mut self,
        ui: &mut egui::Ui,
        pod_id: &str,
        thread_ids: Option<&[String]>,
    ) {
        let (label, pod_behaviors_enabled) = match self.pods.get(pod_id) {
            Some(summary) => (
                format!(
                    "{} ({})",
                    summary.name,
                    thread_ids.map(|t| t.len()).unwrap_or(0)
                ),
                summary.behaviors_enabled,
            ),
            None => (
                format!("{pod_id} ({})", thread_ids.map(|t| t.len()).unwrap_or(0)),
                true,
            ),
        };
        let state_id = ui.make_persistent_id(format!("pod-section-{pod_id}"));
        let default_open = !self.collapsed_pods.contains(pod_id);
        let is_default_pod = pod_id == self.server_default_pod_id;
        let mut archive_clicked = false;
        let mut archive_confirmed = false;
        let mut archive_disarm = false;
        let mut edit_config_clicked = false;
        let mut open_files_clicked = false;
        let mut toggle_pod_behaviors_to: Option<bool> = None;
        let mut behavior_actions: Vec<BehaviorRowAction> = Vec::new();
        let header = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            state_id,
            default_open,
        )
        .show_header(ui, |ui| {
            // Same `Sides::shrink_left().truncate()` pattern the
            // behavior header uses: toolbar on the right keeps its
            // natural width, pod name takes the rest and truncates
            // with an ellipsis when the sidebar is narrow. Pod-level
            // toolbar actions (edit config, pause all behaviors,
            // archive) modify the pod as a whole.
            egui::Sides::new().shrink_left().truncate().show(
                ui,
                |ui| {
                    ui.add(egui::Label::new(RichText::new(label).strong()).truncate());
                },
                |ui| {
                    if !is_default_pod {
                        let armed = self.archive_armed_pod.as_deref() == Some(pod_id);
                        if armed {
                            if sidebar_button(
                                ui,
                                RichText::new("Confirm archive").color(SIDEBAR_DANGER_COLOR),
                                true,
                            )
                            .clicked()
                            {
                                archive_confirmed = true;
                            }
                            if sidebar_button(ui, RichText::new("Cancel"), true).clicked() {
                                archive_disarm = true;
                            }
                        } else if sidebar_icon_button(ui, "🗄", "Archive pod", true).clicked() {
                            archive_clicked = true;
                        }
                    }
                    let (pod_pause_icon, pod_pause_tip) = if pod_behaviors_enabled {
                        ("⏸", "Pause all behaviors in this pod")
                    } else {
                        ("▶", "Resume behaviors in this pod")
                    };
                    if sidebar_icon_button(ui, pod_pause_icon, pod_pause_tip, true).clicked() {
                        toggle_pod_behaviors_to = Some(!pod_behaviors_enabled);
                    }
                    if sidebar_icon_button(ui, "⚙", "Edit pod config", true).clicked() {
                        edit_config_clicked = true;
                    }
                    if sidebar_icon_button(ui, "📁", "Browse pod files", true).clicked() {
                        open_files_clicked = true;
                    }
                    if !pod_behaviors_enabled {
                        ui.label(RichText::new("paused").small().color(SIDEBAR_WARNING_COLOR));
                    }
                },
            );
        });
        let is_open = header.is_open();
        header.body(|ui| {
            // Partition the pod's threads into interactive (origin=None)
            // vs. per-behavior buckets. Each `thread_ids` slice is already
            // newest-first (inherits task_order), so the per-bucket Vecs
            // land newest-first too.
            let mut interactive: Vec<String> = Vec::new();
            let mut by_behavior: HashMap<String, Vec<String>> = HashMap::new();
            if let Some(thread_ids) = thread_ids {
                for tid in thread_ids {
                    let Some(view) = self.tasks.get(tid) else {
                        continue;
                    };
                    match &view.summary.origin {
                        None => interactive.push(tid.clone()),
                        Some(origin) => by_behavior
                            .entry(origin.behavior_id.clone())
                            .or_default()
                            .push(tid.clone()),
                    }
                }
            }
            self.render_interactive_threads(ui, pod_id, &interactive);
            self.render_behaviors_panel(ui, pod_id, &by_behavior, &mut behavior_actions);
        });
        // Track collapse state so it persists across renders.
        if is_open {
            self.collapsed_pods.remove(pod_id);
        } else {
            self.collapsed_pods.insert(pod_id.to_string());
        }
        if archive_clicked {
            self.archive_armed_pod = Some(pod_id.to_string());
        } else if archive_disarm {
            self.archive_armed_pod = None;
        } else if archive_confirmed {
            self.archive_armed_pod = None;
            self.send(ClientToServer::ArchivePod {
                pod_id: pod_id.to_string(),
            });
        }
        if edit_config_clicked {
            self.open_pod_editor(pod_id.to_string());
        }
        if open_files_clicked {
            self.file_tree_modal_pod = Some(pod_id.to_string());
            self.ensure_pod_dir_fetched(pod_id, "");
        }
        if let Some(enabled) = toggle_pod_behaviors_to {
            self.send(ClientToServer::SetPodBehaviorsEnabled {
                correlation_id: None,
                pod_id: pod_id.to_string(),
                enabled,
            });
        }
        self.apply_behavior_row_actions(pod_id, behavior_actions);
    }

    fn open_pod_editor(&mut self, pod_id: String) {
        self.send(ClientToServer::GetPod {
            correlation_id: None,
            pod_id: pod_id.clone(),
        });
        self.pod_editor_modal = Some(PodEditorModalState::new(pod_id));
    }

    /// Classify a pod-relative file path for the file-tree click
    /// dispatcher. Strings mirror constants owned by the server
    /// (`pod::POD_TOML`, `pod::behaviors::{BEHAVIOR_TOML, BEHAVIOR_PROMPT,
    /// BEHAVIORS_DIR}`) — kept in sync by hand because the webui crate
    /// deliberately doesn't depend on the server.
    ///
    /// Files with a `.json` extension explicitly fall through to
    /// `Unknown`: a proper JSON viewer is planned separately and we
    /// don't want users editing thread JSONs or state.json as plain
    /// text by mistake. Every other file — known specializations
    /// aside — routes to the generic text editor, whose server-side
    /// read will reject binaries via a null-byte sniff.
    fn classify_pod_file_path(path: &str) -> PodFileDispatch {
        if path == "pod.toml" {
            return PodFileDispatch::PodConfig;
        }
        if let Some(rest) = path.strip_prefix("behaviors/")
            && let Some((id, suffix)) = rest.split_once('/')
            && !id.is_empty()
            && !suffix.is_empty()
            && !suffix.contains('/')
        {
            match suffix {
                "behavior.toml" => return PodFileDispatch::BehaviorConfig(id.to_string()),
                "prompt.md" => return PodFileDispatch::BehaviorPrompt(id.to_string()),
                _ => {}
            }
        }
        if path.ends_with(".json") {
            return PodFileDispatch::JsonViewer(path.to_string());
        }
        PodFileDispatch::TextEditor(path.to_string())
    }

    /// Fire a `ListPodDir` for `(pod_id, path)` iff we don't already
    /// have a cached listing or a request in flight. Path is the
    /// pod-relative directory ("" = pod root). Shallow — children of
    /// expanded subdirectories are fetched one round-trip at a time.
    fn ensure_pod_dir_fetched(&mut self, pod_id: &str, path: &str) {
        let key = (pod_id.to_string(), path.to_string());
        if self.pod_files.contains_key(&key) || self.pod_files_requested.contains(&key) {
            return;
        }
        self.pod_files_requested.insert(key);
        self.send(ClientToServer::ListPodDir {
            correlation_id: None,
            pod_id: pod_id.to_string(),
            path: if path.is_empty() {
                None
            } else {
                Some(path.to_string())
            },
        });
    }

    /// Render the file-tree modal — a centered Window rooted at a
    /// specific pod's directory. Trigger: the folder icon in the pod
    /// header. The tree body is the same shallow-lazy renderer used
    /// for individual clicks to the pod / behavior / generic editors,
    /// so opening a file from here behaves identically to any other
    /// launch path.
    pub(super) fn render_file_tree_modal(&mut self, ctx: &egui::Context) {
        let Some(pod_id) = self.file_tree_modal_pod.clone() else {
            return;
        };

        let title = format!("Files — {pod_id}");
        let screen = ctx.content_rect();
        let max_h = (screen.height() - 60.0).max(280.0);
        let max_w = (screen.width() - 60.0).max(360.0);
        let mut open = true;

        egui::Window::new(title)
            .collapsible(false)
            .resizable(true)
            .default_width(480.0_f32.min(max_w))
            .default_height(520.0_f32.min(max_h))
            .max_width(max_w)
            .max_height(max_h)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .open(&mut open)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.render_pod_dir(ui, &pod_id, "");
                    });
            });

        if !open {
            self.file_tree_modal_pod = None;
        }
    }

    /// Render one directory's entries. Directories become further
    /// collapsing headers; files are plain labels (read-only entries
    /// are dimmed). Click-to-open dispatch lands in a later phase.
    fn render_pod_dir(&mut self, ui: &mut egui::Ui, pod_id: &str, path: &str) {
        let key = (pod_id.to_string(), path.to_string());
        let Some(entries_ref) = self.pod_files.get(&key) else {
            ui.label(
                RichText::new("  loading…")
                    .small()
                    .italics()
                    .color(SIDEBAR_MUTED_COLOR),
            );
            return;
        };
        if entries_ref.is_empty() {
            ui.label(
                RichText::new("  (empty)")
                    .small()
                    .italics()
                    .color(SIDEBAR_MUTED_COLOR),
            );
            return;
        }
        // Release the borrow on `pod_files` before recursing: child
        // renders may mutate the same map (via ensure_pod_dir_fetched →
        // the inbound event loop) on later frames.
        let entries: Vec<FsEntry> = entries_ref.clone();
        let mut to_fetch: Vec<String> = Vec::new();
        let mut to_dispatch: Option<PodFileDispatch> = None;
        for entry in &entries {
            let child_path = if path.is_empty() {
                entry.name.clone()
            } else {
                format!("{path}/{}", entry.name)
            };
            if entry.is_dir {
                let dir_state_id = ui.make_persistent_id(format!("pod-dir-{pod_id}::{child_path}"));
                let dir_header = egui::collapsing_header::CollapsingState::load_with_default_open(
                    ui.ctx(),
                    dir_state_id,
                    false,
                )
                .show_header(ui, |ui| {
                    ui.add(
                        egui::Label::new(
                            RichText::new(format!("{}/", entry.name)).small().strong(),
                        )
                        .truncate(),
                    );
                });
                let is_open = dir_header.is_open();
                dir_header.body(|ui| {
                    self.render_pod_dir(ui, pod_id, &child_path);
                });
                if is_open {
                    to_fetch.push(child_path);
                }
            } else {
                let dispatch = Self::classify_pod_file_path(&child_path);
                let mut text = RichText::new(&entry.name).small();
                if entry.readonly {
                    text = text.color(SIDEBAR_MUTED_COLOR);
                }
                if ui.selectable_label(false, text).clicked() {
                    to_dispatch = Some(dispatch);
                }
            }
        }
        for child in to_fetch {
            self.ensure_pod_dir_fetched(pod_id, &child);
        }
        if let Some(dispatch) = to_dispatch {
            match dispatch {
                PodFileDispatch::PodConfig => {
                    self.open_pod_editor(pod_id.to_string());
                }
                PodFileDispatch::BehaviorConfig(behavior_id) => {
                    self.open_behavior_editor(pod_id.to_string(), behavior_id);
                }
                PodFileDispatch::BehaviorPrompt(behavior_id) => {
                    self.open_behavior_editor_on_tab(
                        pod_id.to_string(),
                        behavior_id,
                        BehaviorEditorTab::Prompt,
                    );
                }
                PodFileDispatch::TextEditor(path) => {
                    self.open_file_viewer(pod_id.to_string(), path);
                }
                PodFileDispatch::JsonViewer(path) => {
                    self.open_json_viewer(pod_id.to_string(), path);
                }
            }
        }
    }

    /// Fire a `ListBehaviors` for `pod_id` iff we haven't already.
    /// Called on pod discovery (PodList / PodCreated) so the pod
    /// section shows pre-existing behaviors without waiting for the
    /// user to open the pod editor. `PodSnapshot` (from GetPod) also
    /// populates the cache when the editor is opened; the dedup
    /// guard means both paths stay consistent.
    pub(super) fn ensure_behaviors_fetched(&mut self, pod_id: &str) {
        if self.behaviors_requested.contains(pod_id) {
            return;
        }
        self.behaviors_requested.insert(pod_id.to_string());
        self.send(ClientToServer::ListBehaviors {
            correlation_id: None,
            pod_id: pod_id.to_string(),
        });
    }

    pub(super) fn open_behavior_editor(&mut self, pod_id: String, behavior_id: String) {
        self.open_behavior_editor_on_tab(pod_id, behavior_id, BehaviorEditorTab::Trigger);
    }

    /// Same as [`open_behavior_editor`] but lets the caller choose which
    /// tab the modal opens on. Used by the file-tree dispatch so a click
    /// on `behaviors/<id>/prompt.md` lands directly on the Prompt tab
    /// instead of making the user navigate there.
    fn open_behavior_editor_on_tab(
        &mut self,
        pod_id: String,
        behavior_id: String,
        tab: BehaviorEditorTab,
    ) {
        self.send(ClientToServer::GetBehavior {
            correlation_id: None,
            pod_id: pod_id.clone(),
            behavior_id: behavior_id.clone(),
        });
        let mut state = BehaviorEditorModalState::new(pod_id, behavior_id);
        state.tab = tab;
        self.behavior_editor_modal = Some(state);
    }

    /// Open the generic text-editor modal on `<pod_id>/<path>`. Sends
    /// `ReadPodFile` immediately; the returned `PodFileContent`
    /// populates `working` + `baseline` + `readonly`. While the read
    /// is in flight the modal renders a "loading…" placeholder, and
    /// any server error on the read surfaces inline via the matching
    /// correlation id.
    fn open_file_viewer(&mut self, pod_id: String, path: String) {
        let correlation = self.next_correlation_id();
        let mut state = FileViewerModalState::new(pod_id.clone(), path.clone());
        state.pending_correlation = Some(correlation.clone());
        self.file_viewer_modal = Some(state);
        self.send(ClientToServer::ReadPodFile {
            correlation_id: Some(correlation),
            pod_id,
            path,
        });
    }

    /// Open the JSON tree viewer on `<pod_id>/<path>`. Same read path
    /// as [`open_file_viewer`]; divergence lives in the
    /// `PodFileContent` handler, which parses the content into a
    /// `serde_json::Value` when the correlation matches this modal.
    fn open_json_viewer(&mut self, pod_id: String, path: String) {
        let correlation = self.next_correlation_id();
        let mut state = JsonViewerModalState::new(pod_id.clone(), path.clone());
        state.pending_correlation = Some(correlation.clone());
        self.json_viewer_modal = Some(state);
        self.send(ClientToServer::ReadPodFile {
            correlation_id: Some(correlation),
            pod_id,
            path,
        });
    }

    /// Render the behaviors sub-section of a pod header, with each
    /// behavior's recent threads nested underneath its row. Produces
    /// `BehaviorRowAction` tokens in `actions` for the enclosing
    /// `render_pod_section` to act on after the closure returns — keeps
    /// mutating state (sending wire messages, opening modals) out of
    /// the rendering closure where the egui borrow graph is ugly.
    ///
    /// `threads_by_behavior` is keyed by `behavior_id`; any entry whose
    /// key is not in `behaviors_by_pod[pod_id]` is rendered as an
    /// orphan bucket under "Deleted behaviors" — threads spawned by a
    /// behavior that was later removed still deserve to be visible and
    /// selectable.
    fn render_behaviors_panel(
        &self,
        ui: &mut egui::Ui,
        pod_id: &str,
        threads_by_behavior: &HashMap<String, Vec<String>>,
        actions: &mut Vec<BehaviorRowAction>,
    ) {
        ui.add_space(4.0);
        ui.separator();
        ui.horizontal(|ui| {
            sidebar_subsection_header(ui, "Behaviors");
            if sidebar_icon_button(ui, "➕", "New behavior", true).clicked() {
                actions.push(BehaviorRowAction::New);
            }
        });
        let empty: Vec<BehaviorSummary> = Vec::new();
        let behaviors = self.behaviors_by_pod.get(pod_id).unwrap_or(&empty);
        if behaviors.is_empty() && threads_by_behavior.is_empty() {
            ui.label(
                RichText::new("  (none)")
                    .small()
                    .italics()
                    .color(SIDEBAR_MUTED_COLOR),
            );
            return;
        }
        for row in behaviors {
            let threads = threads_by_behavior
                .get(&row.behavior_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            self.render_behavior_row(ui, row, threads, actions);
        }
        // Orphan threads: behavior_id present in threads_by_behavior but not
        // in the known behaviors list. Typically means the behavior was
        // deleted while its spawned threads are still around.
        let known: HashSet<&str> = behaviors.iter().map(|b| b.behavior_id.as_str()).collect();
        let mut orphan_ids: Vec<&String> = threads_by_behavior
            .keys()
            .filter(|k| !known.contains(k.as_str()))
            .collect();
        orphan_ids.sort();
        if !orphan_ids.is_empty() {
            ui.add_space(2.0);
            sidebar_subsection_header(ui, "Deleted behaviors");
            for behavior_id in orphan_ids {
                let threads = threads_by_behavior
                    .get(behavior_id)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                self.render_orphan_behavior_threads(ui, pod_id, behavior_id, threads, actions);
            }
        }
    }

    fn render_behavior_row(
        &self,
        ui: &mut egui::Ui,
        row: &BehaviorSummary,
        threads: &[String],
        actions: &mut Vec<BehaviorRowAction>,
    ) {
        let label_text = match &row.trigger_kind {
            Some(kind) => format!("{} [{}]", row.name, kind),
            None => format!("{} [errored]", row.name),
        };
        let label_color = if row.load_error.is_some() {
            SIDEBAR_ERROR_TEXT_COLOR
        } else if !row.enabled {
            SIDEBAR_DIM_COLOR
        } else {
            SIDEBAR_BODY_COLOR
        };
        // Each behavior is its own collapsible container: header shows
        // name + toolbar, body shows last-fired timestamp and the
        // behavior's recent threads. Default-open when the behavior has
        // threads so history stays visible; default-closed when it has
        // never fired so the sidebar doesn't grow empty rows.
        let state_id = ui.make_persistent_id((
            "behavior-row",
            row.pod_id.as_str(),
            row.behavior_id.as_str(),
        ));
        let default_open = !threads.is_empty();
        let armed = self
            .delete_armed_behavior
            .as_ref()
            .map(|(p, b)| p == &row.pod_id && b == &row.behavior_id)
            .unwrap_or(false);
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            state_id,
            default_open,
        )
        .show_header(ui, |ui| {
            // `Sides::shrink_left().truncate()` paints the right-side
            // toolbar at its natural width first, then gives the label
            // side whatever's left with `Truncate` wrap mode. This is
            // what keeps the behavior name from walking under the
            // action icons when the sidebar is narrow — the name just
            // ends in an ellipsis instead.
            egui::Sides::new().shrink_left().truncate().show(
                ui,
                |ui| {
                    ui.add(
                        egui::Label::new(
                            RichText::new(label_text)
                                .small()
                                .strong()
                                .color(label_color),
                        )
                        .truncate(),
                    );
                    if !row.enabled {
                        ui.label(RichText::new("paused").small().color(SIDEBAR_WARNING_COLOR));
                    }
                },
                |ui| {
                    if armed {
                        if sidebar_button(
                            ui,
                            RichText::new("Confirm").color(SIDEBAR_DANGER_COLOR),
                            true,
                        )
                        .clicked()
                        {
                            actions.push(BehaviorRowAction::ConfirmDelete {
                                pod_id: row.pod_id.clone(),
                                behavior_id: row.behavior_id.clone(),
                            });
                        }
                        if sidebar_button(ui, RichText::new("Cancel"), true).clicked() {
                            actions.push(BehaviorRowAction::DisarmDelete);
                        }
                    } else {
                        // Sides' right sub-ui uses a right-to-left
                        // layout: widgets stack from the far right
                        // inward, so visual order is the reverse of
                        // the call order. We want: ⏻ | ▶ | ✎ | 🗑
                        // reading left-to-right — call 🗑 first.
                        if sidebar_icon_button(ui, "🗑", "Delete behavior", true).clicked() {
                            actions.push(BehaviorRowAction::ArmDelete {
                                pod_id: row.pod_id.clone(),
                                behavior_id: row.behavior_id.clone(),
                            });
                        }
                        if sidebar_icon_button(ui, "✎", "Edit behavior", true).clicked() {
                            actions.push(BehaviorRowAction::Edit {
                                pod_id: row.pod_id.clone(),
                                behavior_id: row.behavior_id.clone(),
                            });
                        }
                        // Errored behaviors can't be run — disable Run
                        // until the user fixes the config.
                        let run_enabled = row.load_error.is_none();
                        if sidebar_icon_button(ui, "▶", "Run now", run_enabled).clicked() {
                            actions.push(BehaviorRowAction::Run {
                                pod_id: row.pod_id.clone(),
                                behavior_id: row.behavior_id.clone(),
                            });
                        }
                        // Enable/disable toggle: ⏻ (power symbol) is
                        // a universal on/off metaphor that doesn't
                        // collide with the ▶ Run glyph next to it.
                        // Bright color when enabled, dim when paused.
                        let (toggle_color, toggle_tip) = if row.enabled {
                            (SIDEBAR_BODY_COLOR, "Pause behavior")
                        } else {
                            (SIDEBAR_DIM_COLOR, "Resume behavior")
                        };
                        if sidebar_icon_button(
                            ui,
                            RichText::new("⏻").color(toggle_color),
                            toggle_tip,
                            true,
                        )
                        .clicked()
                        {
                            actions.push(BehaviorRowAction::SetEnabled {
                                pod_id: row.pod_id.clone(),
                                behavior_id: row.behavior_id.clone(),
                                enabled: !row.enabled,
                            });
                        }
                    }
                },
            );
        })
        .body(|ui| {
            if let Some(err) = &row.load_error {
                ui.label(
                    RichText::new(format!("⚠ {err}"))
                        .small()
                        .color(SIDEBAR_ERROR_TEXT_COLOR),
                );
            } else if let Some(last) = &row.last_fired_at {
                ui.label(
                    RichText::new(format!("last fired: {}", format_relative_time(last)))
                        .small()
                        .color(SIDEBAR_MUTED_COLOR),
                );
            } else {
                ui.label(
                    RichText::new("no runs yet")
                        .small()
                        .italics()
                        .color(SIDEBAR_MUTED_COLOR),
                );
            }
            if !threads.is_empty() {
                self.render_nested_thread_list(ui, &row.pod_id, &row.behavior_id, threads, actions);
            }
        });
    }

    /// Render threads spawned by a behavior whose config is no longer in
    /// `behaviors_by_pod` — usually because the behavior was deleted.
    /// Still selectable so the user can archive/review the surviving
    /// thread rows.
    fn render_orphan_behavior_threads(
        &self,
        ui: &mut egui::Ui,
        pod_id: &str,
        behavior_id: &str,
        threads: &[String],
        actions: &mut Vec<BehaviorRowAction>,
    ) {
        ui.label(
            RichText::new(format!("  {behavior_id}  ({})", threads.len()))
                .small()
                .italics()
                .color(SIDEBAR_MUTED_COLOR),
        );
        self.render_nested_thread_list(ui, pod_id, behavior_id, threads, actions);
    }

    /// Shared renderer for a behavior's (or orphan bucket's) recent
    /// threads. Shows the first `THREAD_ROW_PREVIEW_COUNT` rows by default
    /// with a "Show N more" toggle; when expanded, reveals the full list
    /// with a "Show less" toggle.
    fn render_nested_thread_list(
        &self,
        ui: &mut egui::Ui,
        pod_id: &str,
        behavior_id: &str,
        threads: &[String],
        actions: &mut Vec<BehaviorRowAction>,
    ) {
        let key = (pod_id.to_string(), behavior_id.to_string());
        let expanded = self.expanded_behavior_threads.contains(&key);
        let shown = if expanded {
            threads.len()
        } else {
            threads.len().min(THREAD_ROW_PREVIEW_COUNT)
        };
        for tid in &threads[..shown] {
            self.render_nested_thread_button(ui, tid, actions);
        }
        let hidden = threads.len().saturating_sub(shown);
        let toggle_clicked = if hidden > 0 {
            sidebar_button(ui, RichText::new(format!("Show {hidden} more")), true).clicked()
        } else if expanded && threads.len() > THREAD_ROW_PREVIEW_COUNT {
            sidebar_button(ui, RichText::new("Show less"), true).clicked()
        } else {
            false
        };
        if toggle_clicked {
            actions.push(BehaviorRowAction::ToggleExpandThreads {
                pod_id: pod_id.to_string(),
                behavior_id: behavior_id.to_string(),
            });
        }
    }

    /// Render one nested-thread button, emitting a `SelectThread` action
    /// on click. Used under both real and orphan behavior buckets.
    fn render_nested_thread_button(
        &self,
        ui: &mut egui::Ui,
        thread_id: &str,
        actions: &mut Vec<BehaviorRowAction>,
    ) {
        let Some(view) = self.tasks.get(thread_id) else {
            return;
        };
        let is_selected = self.selected.as_deref() == Some(thread_id);
        let title = view
            .summary
            .title
            .clone()
            .unwrap_or_else(|| thread_id[..thread_id.len().min(14)].to_string());
        let (chip, chip_color) = state_chip(view.summary.state);
        let text = RichText::new(format!("{title}  [{chip}]")).color(if is_selected {
            Color32::WHITE
        } else {
            chip_color
        });
        let row = add_sidebar_thread_row(ui, is_selected, text);
        if row.clicked() {
            actions.push(BehaviorRowAction::SelectThread {
                thread_id: thread_id.to_string(),
            });
        }
    }

    /// Render the interactive-threads subsection under a pod. "Interactive"
    /// here means threads the user created directly (no behavior origin).
    /// Always renders the section header (with its `➕ new thread`
    /// affordance) — the `+` is the primary entry point for creating a
    /// thread in this pod, so hiding it when the list is empty would
    /// leave an empty pod unusable.
    ///
    /// Dispatched-thread children (`dispatched_by.is_some()`) are
    /// grouped under their parent in DFS order so the nesting is
    /// visible in the sidebar day one. Orphaned children whose parent
    /// isn't in the current interactive set fall back to top-level
    /// with a `dispatched_by` prefix marker.
    fn render_interactive_threads(
        &mut self,
        ui: &mut egui::Ui,
        pod_id: &str,
        interactive: &[String],
    ) {
        ui.add_space(4.0);
        let mut new_thread_clicked = false;
        ui.horizontal(|ui| {
            sidebar_subsection_header(ui, format!("Interactive ({})", interactive.len()));
            if sidebar_icon_button(ui, "➕", "New thread in this pod", true).clicked() {
                new_thread_clicked = true;
            }
        });
        if new_thread_clicked {
            self.selected = None;
            self.composing_new = true;
            self.compose_pod_id = Some(pod_id.to_string());
            self.input.clear();
        }
        if interactive.is_empty() {
            ui.label(
                RichText::new("  (no threads yet)")
                    .small()
                    .italics()
                    .color(SIDEBAR_MUTED_COLOR),
            );
            return;
        }
        // Reorder the flat list into DFS-by-dispatch: each root is
        // followed by its dispatched children (transitively). Returned
        // as Vec<(thread_id, depth)>; depth 0 = root, 1 = first-level
        // child, etc. Threads outside the interactive set (e.g. lost
        // parent) are treated as roots so nothing gets dropped.
        let ordered = self.order_interactive_with_dispatch_nesting(interactive);
        let expanded = self.expanded_interactive_pods.contains(pod_id);
        let shown = if expanded {
            ordered.len()
        } else {
            ordered.len().min(THREAD_ROW_PREVIEW_COUNT)
        };
        let mut clicked: Option<String> = None;
        for (tid, depth) in &ordered[..shown] {
            let Some(view) = self.tasks.get(tid) else {
                continue;
            };
            let is_selected = self.selected.as_deref() == Some(tid.as_str());
            let title = view
                .summary
                .title
                .clone()
                .unwrap_or_else(|| tid[..tid.len().min(14)].to_string());
            let (chip, chip_color) = state_chip(view.summary.state);
            // Prefix: continuation (`↩`) and/or dispatched (`↳`)
            // markers; the two are orthogonal — a continuation of a
            // dispatched thread carries both flags. Depth indent
            // visualizes the dispatch chain for nested dispatches.
            let indent: String = "  ".repeat(*depth);
            let dispatch_marker = if view.summary.dispatched_by.is_some() {
                "↳ "
            } else {
                ""
            };
            let continuation_marker = if view.summary.continued_from.is_some() {
                "↩ "
            } else {
                ""
            };
            let text = RichText::new(format!(
                "{indent}{dispatch_marker}{continuation_marker}{title}  [{chip}]"
            ))
            .color(if is_selected {
                Color32::WHITE
            } else {
                chip_color
            });
            let row = add_sidebar_thread_row(ui, is_selected, text);
            if row.clicked() {
                clicked = Some(tid.clone());
            }
        }
        let hidden = ordered.len().saturating_sub(shown);
        let toggle = if hidden > 0 {
            sidebar_button(ui, RichText::new(format!("Show {hidden} more")), true).clicked()
        } else if expanded && ordered.len() > THREAD_ROW_PREVIEW_COUNT {
            sidebar_button(ui, RichText::new("Show less"), true).clicked()
        } else {
            false
        };
        if toggle {
            if expanded {
                self.expanded_interactive_pods.remove(pod_id);
            } else {
                self.expanded_interactive_pods.insert(pod_id.to_string());
            }
        }
        if let Some(tid) = clicked {
            self.select_task(tid);
        }
    }

    /// Reorder a flat list of interactive thread ids into DFS-nested
    /// order by `dispatched_by`: each root is followed by its
    /// dispatched children (recursively). Children whose parent isn't
    /// in `flat` are promoted to roots so nothing is lost; cycles
    /// (shouldn't happen — the scheduler enforces a depth cap) are
    /// broken by a visited set. Returns `(thread_id, depth)` pairs.
    fn order_interactive_with_dispatch_nesting(&self, flat: &[String]) -> Vec<(String, usize)> {
        use std::collections::HashMap;
        let in_set: std::collections::HashSet<&str> = flat.iter().map(|s| s.as_str()).collect();
        // parent_id → ordered list of direct children. The newest-first
        // order of `flat` is preserved within each sibling bucket
        // because we walk `flat` in order and push_back.
        let mut children_of: HashMap<String, Vec<String>> = HashMap::new();
        let mut roots: Vec<String> = Vec::new();
        for tid in flat {
            let view = match self.tasks.get(tid) {
                Some(v) => v,
                None => continue,
            };
            match &view.summary.dispatched_by {
                Some(parent) if in_set.contains(parent.as_str()) => {
                    children_of
                        .entry(parent.clone())
                        .or_default()
                        .push(tid.clone());
                }
                _ => roots.push(tid.clone()),
            }
        }
        let mut out: Vec<(String, usize)> = Vec::with_capacity(flat.len());
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        fn dfs(
            id: &str,
            depth: usize,
            children_of: &HashMap<String, Vec<String>>,
            visited: &mut std::collections::HashSet<String>,
            out: &mut Vec<(String, usize)>,
        ) {
            if !visited.insert(id.to_string()) {
                return;
            }
            out.push((id.to_string(), depth));
            if let Some(kids) = children_of.get(id) {
                for child in kids {
                    dfs(child, depth + 1, children_of, visited, out);
                }
            }
        }
        for root in &roots {
            dfs(root, 0, &children_of, &mut visited, &mut out);
        }
        // Safety net: any thread we didn't visit (because its parent
        // was in the set but the chain was broken somewhere) gets
        // appended at depth 0 so it remains visible.
        for tid in flat {
            if !visited.contains(tid) {
                out.push((tid.clone(), 0));
                visited.insert(tid.clone());
            }
        }
        out
    }

    fn apply_behavior_row_actions(&mut self, pod_id: &str, actions: Vec<BehaviorRowAction>) {
        for action in actions {
            match action {
                BehaviorRowAction::New => {
                    self.new_behavior_modal = Some(NewBehaviorModalState::new(pod_id.to_string()));
                }
                BehaviorRowAction::Edit {
                    pod_id,
                    behavior_id,
                } => {
                    self.open_behavior_editor(pod_id, behavior_id);
                }
                BehaviorRowAction::Run {
                    pod_id,
                    behavior_id,
                } => {
                    self.send(ClientToServer::RunBehavior {
                        correlation_id: None,
                        pod_id,
                        behavior_id,
                        payload: None,
                    });
                }
                BehaviorRowAction::ArmDelete {
                    pod_id,
                    behavior_id,
                } => {
                    self.delete_armed_behavior = Some((pod_id, behavior_id));
                }
                BehaviorRowAction::DisarmDelete => {
                    self.delete_armed_behavior = None;
                }
                BehaviorRowAction::ConfirmDelete {
                    pod_id,
                    behavior_id,
                } => {
                    self.delete_armed_behavior = None;
                    self.send(ClientToServer::DeleteBehavior {
                        correlation_id: None,
                        pod_id,
                        behavior_id,
                    });
                }
                BehaviorRowAction::SetEnabled {
                    pod_id,
                    behavior_id,
                    enabled,
                } => {
                    self.send(ClientToServer::SetBehaviorEnabled {
                        correlation_id: None,
                        pod_id,
                        behavior_id,
                        enabled,
                    });
                }
                BehaviorRowAction::SelectThread { thread_id } => {
                    self.select_task(thread_id);
                }
                BehaviorRowAction::ToggleExpandThreads {
                    pod_id,
                    behavior_id,
                } => {
                    let key = (pod_id, behavior_id);
                    if !self.expanded_behavior_threads.remove(&key) {
                        self.expanded_behavior_threads.insert(key);
                    }
                }
            }
        }
    }

    /// Populate the behavior editor modal from a `BehaviorSnapshot`
    /// event. Also updates the per-pod summary cache with the latest
    /// summary-shaped view of the same data. Called on initial load
    /// (correlation_id None) and after a successful Update.
    pub(super) fn apply_behavior_snapshot(
        &mut self,
        _correlation_id: Option<String>,
        snapshot: BehaviorSnapshotProto,
    ) {
        // Refresh the list-cached summary so the pod detail view
        // stays in sync with the latest config.
        if let Some(list) = self.behaviors_by_pod.get_mut(&snapshot.pod_id) {
            let summary = behavior_summary_from_snapshot(&snapshot);
            if let Some(existing) = list
                .iter_mut()
                .find(|b| b.behavior_id == snapshot.behavior_id)
            {
                *existing = summary;
            } else {
                list.push(summary);
                list.sort_by(|a, b| a.behavior_id.cmp(&b.behavior_id));
            }
        }
        // If the editor is open for this behavior and hasn't loaded
        // yet, populate it. `working.is_none()` is the load gate —
        // subsequent updates (from a successful Save round-trip) are
        // applied via the `BehaviorUpdated` handler instead.
        if let Some(modal) = self.behavior_editor_modal.as_mut()
            && modal.pod_id == snapshot.pod_id
            && modal.behavior_id == snapshot.behavior_id
            && modal.working_config.is_none()
        {
            modal.working_config = snapshot.config.clone();
            modal.baseline_config = snapshot.config.clone();
            modal.working_prompt = snapshot.prompt.clone();
            modal.baseline_prompt = snapshot.prompt.clone();
            modal.raw_buffer = snapshot.toml_text.clone();
            modal.raw_dirty = false;
            modal.error = snapshot.load_error.clone();
        }
    }
}
