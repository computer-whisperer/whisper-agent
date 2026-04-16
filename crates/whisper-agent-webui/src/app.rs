//! Multi-task chat UI.
//!
//! Pure-egui rendering — compiles on both native and wasm. Networking lives in the
//! wasm-only `web_entry` module in `lib.rs`. The two communicate through:
//!   - `inbound`: a shared queue of decoded [`ServerToClient`] events plus
//!     [`ConnectionEvent`]s the WebSocket glue injects directly.
//!   - `send_fn`: a closure provided at construction time. On wasm it serializes to
//!     CBOR and calls `WebSocket::send_with_u8_array`; on native it's a no-op stub.
//!
//! The UI maintains a view model per task (title, state chip, message list). On task
//! selection we send `SubscribeToTask`; the server's `TaskSnapshot` response rebuilds
//! the task's display items. Subsequent turn events append to them.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use egui::{Color32, ComboBox, RichText, ScrollArea, TextEdit};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::sandbox::{AccessMode, NetworkPolicy, PathAccess};
use whisper_agent_protocol::{
    ApprovalChoice, BackendSummary, ClientToServer, ContentBlock, Conversation, Message,
    ModelSummary, ResourceSnapshot, ResourceStateLabel, Role, SandboxSpec, ServerToClient,
    TaskConfigOverride, TaskStateLabel, TaskSummary, ToolResultContent, Usage,
};

/// A user-saved sandbox configuration. Currently only encodes Landlock; future
/// variants (container image, resource limits) plug in alongside.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LandlockPreset {
    name: String,
    /// Read-write workspace path.
    workspace: String,
    /// If true, the whole filesystem is granted read-only access on top of the
    /// workspace's read-write scope. Useful for tasks that need to inspect
    /// system files (/etc, /proc, the user's other projects) without being
    /// able to modify them.
    read_everywhere: bool,
}

impl LandlockPreset {
    fn to_spec(&self) -> SandboxSpec {
        let mut allowed_paths = vec![PathAccess {
            path: self.workspace.clone(),
            mode: AccessMode::ReadWrite,
        }];
        if self.read_everywhere {
            allowed_paths.push(PathAccess {
                path: "/".into(),
                mode: AccessMode::ReadOnly,
            });
        }
        SandboxSpec::Landlock {
            allowed_paths,
            network: NetworkPolicy::Unrestricted,
        }
    }
}

/// Selection in the new-task sandbox dropdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SandboxChoice {
    /// Use whatever the server configured as default.
    #[default]
    ServerDefault,
    /// Explicitly no sandbox.
    None,
    /// Index into ChatApp::presets.
    Preset(usize),
}

#[cfg(target_arch = "wasm32")]
const PRESETS_STORAGE_KEY: &str = "whisper-agent.sandbox-presets";

#[cfg(target_arch = "wasm32")]
fn load_presets() -> Vec<LandlockPreset> {
    let Some(window) = web_sys::window() else {
        return Vec::new();
    };
    let Ok(Some(storage)) = window.local_storage() else {
        return Vec::new();
    };
    let Ok(Some(json)) = storage.get_item(PRESETS_STORAGE_KEY) else {
        return Vec::new();
    };
    serde_json::from_str(&json).unwrap_or_default()
}

#[cfg(not(target_arch = "wasm32"))]
fn load_presets() -> Vec<LandlockPreset> {
    Vec::new()
}

#[cfg(target_arch = "wasm32")]
fn save_presets(presets: &[LandlockPreset]) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Ok(Some(storage)) = window.local_storage() else {
        return;
    };
    let Ok(json) = serde_json::to_string(presets) else {
        return;
    };
    let _ = storage.set_item(PRESETS_STORAGE_KEY, &json);
}

#[cfg(not(target_arch = "wasm32"))]
fn save_presets(_presets: &[LandlockPreset]) {}

/// Events pushed into [`Inbound`]. In addition to decoded wire messages we pipe in
/// connection-level signals (open/close/error) so the UI can show a connection status
/// distinct from per-task state.
// `Wire(TaskSnapshot)` dwarfs the connection variants. Boxing would change every
// inbound enqueue site; the queue is shallow and short-lived, so the bytes saved
// per item don't justify the churn. Same trade-off as `whisper-agent-protocol::ServerToClient`.
#[allow(clippy::large_enum_variant)]
pub enum InboundEvent {
    Wire(ServerToClient),
    ConnectionOpened,
    ConnectionClosed { detail: String },
    ConnectionError { detail: String },
}

pub type Inbound = Rc<RefCell<VecDeque<InboundEvent>>>;
pub type SendFn = Box<dyn Fn(ClientToServer)>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConnectionStatus {
    Connecting,
    Connected,
    Closed,
    Error,
}

impl ConnectionStatus {
    fn label(self) -> (&'static str, Color32) {
        match self {
            Self::Connecting => ("connecting…", Color32::from_rgb(200, 180, 60)),
            Self::Connected => ("connected", Color32::from_rgb(80, 180, 100)),
            Self::Closed => ("closed", Color32::from_rgb(180, 120, 80)),
            Self::Error => ("error", Color32::from_rgb(220, 90, 90)),
        }
    }
}

fn state_chip(state: TaskStateLabel) -> (&'static str, Color32) {
    match state {
        TaskStateLabel::Idle => ("idle", Color32::from_gray(160)),
        TaskStateLabel::Working => ("working", Color32::from_rgb(120, 180, 240)),
        TaskStateLabel::AwaitingApproval => ("approval", Color32::from_rgb(240, 180, 90)),
        TaskStateLabel::Completed => ("completed", Color32::from_rgb(120, 200, 140)),
        TaskStateLabel::Failed => ("failed", Color32::from_rgb(220, 110, 110)),
        TaskStateLabel::Cancelled => ("cancelled", Color32::from_rgb(180, 140, 140)),
    }
}

enum DisplayItem {
    User {
        text: String,
    },
    AssistantText {
        text: String,
    },
    /// Model's chain-of-thought (Anthropic extended-thinking, OpenAI-compat
    /// `reasoning_content`, or inline `<think>...</think>`). Rendered as a
    /// collapsing header so it's preserved without dominating the conversation.
    Reasoning {
        text: String,
    },
    ToolCall {
        tool_use_id: String,
        name: String,
        args_preview: String,
        result: Option<String>,
        is_error: bool,
    },
    SystemNote {
        text: String,
        is_error: bool,
    },
}

struct TaskView {
    summary: TaskSummary,
    items: Vec<DisplayItem>,
    total_usage: Usage,
    subscribed: bool,
    /// Currently-open approval requests, in arrival order. Cleared on snapshot so
    /// re-subscribe can re-seed them without duplicates.
    pending_approvals: Vec<PendingApproval>,
    /// Backend alias the server resolved for this task. Populated from TaskSnapshot.
    /// Empty string means "server default" — the status bar resolves that to the
    /// known default_backend name at render time.
    backend: String,
    /// Model the task was created with. Populated from TaskSnapshot.
    model: String,
    /// Failure detail carried by the snapshot, if the task ended up in `Failed`.
    /// Rendered as a persistent banner so it survives re-subscribe (unlike items,
    /// which get rebuilt from the conversation on every snapshot).
    failure: Option<String>,
    /// Tool names this task has approved-and-remembered. Mirrors
    /// `TaskSnapshot.tool_allowlist`; refreshed by `TaskAllowlistUpdated`.
    tool_allowlist: Vec<String>,
}

struct PendingApproval {
    approval_id: String,
    name: String,
    args_preview: String,
    destructive: bool,
    read_only: bool,
    /// True after the local user clicked Approve/Reject — buttons disable until the
    /// server's TaskApprovalResolved arrives (and removes the entry entirely).
    submitted: bool,
}

impl TaskView {
    fn new(summary: TaskSummary) -> Self {
        Self {
            summary,
            items: Vec::new(),
            total_usage: Usage::default(),
            subscribed: false,
            pending_approvals: Vec::new(),
            backend: String::new(),
            model: String::new(),
            failure: None,
            tool_allowlist: Vec::new(),
        }
    }
}

pub struct ChatApp {
    conn_status: ConnectionStatus,
    conn_detail: Option<String>,

    tasks: HashMap<String, TaskView>,
    /// Display order for the sidebar — sorted by creation time, latest first.
    task_order: Vec<String>,
    selected: Option<String>,
    /// True when the input box composes a new task; false when it messages the selected task.
    composing_new: bool,

    input: String,
    inbound: Inbound,
    send_fn: SendFn,
    list_requested: bool,

    // --- Model-backend catalog ---
    backends: Vec<BackendSummary>,
    default_backend: String,
    backends_requested: bool,
    /// Cached model lists keyed by backend name.
    models_by_backend: HashMap<String, Vec<ModelSummary>>,
    /// Backends we've already sent a ListModels request for — dedup so UI changes
    /// don't re-request repeatedly.
    models_requested: HashSet<String>,
    /// Backend chosen in the new-task picker. None = follow server default.
    picker_backend: Option<String>,
    /// Model chosen in the new-task picker. None = follow backend default.
    picker_model: Option<String>,
    picker_sandbox: SandboxChoice,
    /// User-saved sandbox presets, persisted in localStorage (WASM) or kept
    /// in memory (native test builds).
    presets: Vec<LandlockPreset>,
    /// Modal state for creating a new preset. `None` = closed.
    preset_modal: Option<PresetModalState>,

    // --- Resource registry (Phase 1c read-only inspector) ---
    /// Snapshot of every resource the server has reported. Keyed by resource id.
    resources: HashMap<String, ResourceSnapshot>,
    resources_requested: bool,
    /// Which view the left side panel is showing.
    left_mode: LeftPanelMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LeftPanelMode {
    #[default]
    Tasks,
    Resources,
}

struct PresetModalState {
    name: String,
    workspace: String,
    read_everywhere: bool,
}

impl PresetModalState {
    fn new() -> Self {
        Self {
            name: String::new(),
            workspace: String::new(),
            read_everywhere: false,
        }
    }
}

impl ChatApp {
    pub fn new(inbound: Inbound, send_fn: SendFn) -> Self {
        Self {
            conn_status: ConnectionStatus::Connecting,
            conn_detail: None,
            tasks: HashMap::new(),
            task_order: Vec::new(),
            selected: None,
            composing_new: true,
            input: String::new(),
            inbound,
            send_fn,
            list_requested: false,
            backends: Vec::new(),
            default_backend: String::new(),
            backends_requested: false,
            models_by_backend: HashMap::new(),
            models_requested: HashSet::new(),
            picker_backend: None,
            picker_model: None,
            picker_sandbox: SandboxChoice::default(),
            presets: load_presets(),
            preset_modal: None,
            resources: HashMap::new(),
            resources_requested: false,
            left_mode: LeftPanelMode::default(),
        }
    }

    /// Resolve the picker-selected backend name, falling back to the server default.
    fn effective_picker_backend(&self) -> &str {
        self.picker_backend
            .as_deref()
            .unwrap_or(&self.default_backend)
    }

    /// Resolve the picker-selected model. Fallback chain:
    ///   1. user's explicit picker_model
    ///   2. backend's default_model from the catalog
    ///   3. first model from the fetched ModelsList for that backend
    ///   4. empty (display as "(loading…)")
    fn effective_picker_model(&self) -> String {
        if let Some(m) = &self.picker_model {
            return m.clone();
        }
        let backend = self.effective_picker_backend();
        if let Some(m) = self
            .backends
            .iter()
            .find(|b| b.name == backend)
            .and_then(|b| b.default_model.clone())
        {
            return m;
        }
        self.models_by_backend
            .get(backend)
            .and_then(|list| list.first())
            .map(|m| m.id.clone())
            .unwrap_or_default()
    }

    fn request_models_for(&mut self, backend: &str) {
        if backend.is_empty() || self.models_requested.contains(backend) {
            return;
        }
        self.models_requested.insert(backend.to_string());
        self.send(ClientToServer::ListModels {
            correlation_id: None,
            backend: backend.to_string(),
        });
    }

    fn send(&self, msg: ClientToServer) {
        (self.send_fn)(msg);
    }

    fn drain_inbound(&mut self) {
        let events: Vec<InboundEvent> = self.inbound.borrow_mut().drain(..).collect();
        for event in events {
            self.handle_event(event);
        }
    }

    fn handle_event(&mut self, event: InboundEvent) {
        match event {
            InboundEvent::ConnectionOpened => {
                self.conn_status = ConnectionStatus::Connected;
                self.conn_detail = None;
                if !self.list_requested {
                    self.send(ClientToServer::ListTasks {
                        correlation_id: None,
                    });
                    self.list_requested = true;
                }
                if !self.backends_requested {
                    self.send(ClientToServer::ListBackends {
                        correlation_id: None,
                    });
                    self.backends_requested = true;
                }
                if !self.resources_requested {
                    self.send(ClientToServer::ListResources {
                        correlation_id: None,
                    });
                    self.resources_requested = true;
                }
            }
            InboundEvent::ConnectionClosed { detail } => {
                self.conn_status = ConnectionStatus::Closed;
                self.conn_detail = Some(detail);
            }
            InboundEvent::ConnectionError { detail } => {
                self.conn_status = ConnectionStatus::Error;
                self.conn_detail = Some(detail);
            }
            InboundEvent::Wire(msg) => self.handle_wire(msg),
        }
    }

    fn handle_wire(&mut self, msg: ServerToClient) {
        match msg {
            ServerToClient::TaskCreated {
                task_id,
                summary,
                correlation_id: _,
            } => {
                self.upsert_task(summary);
                self.recompute_order();
                // Auto-select newly created tasks.
                if self.selected.as_deref() != Some(&task_id) {
                    self.select_task(task_id);
                }
            }
            ServerToClient::TaskStateChanged { task_id, state } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.summary.state = state;
                }
            }
            ServerToClient::TaskTitleUpdated { task_id, title } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.summary.title = Some(title);
                }
            }
            ServerToClient::TaskArchived { task_id } => {
                self.tasks.remove(&task_id);
                self.task_order.retain(|id| id != &task_id);
                if self.selected.as_deref() == Some(&task_id) {
                    self.selected = None;
                    self.composing_new = true;
                }
            }
            ServerToClient::TaskList { tasks, .. } => {
                self.tasks
                    .retain(|id, _| tasks.iter().any(|t| &t.task_id == id));
                for summary in tasks {
                    self.upsert_task(summary);
                }
                self.recompute_order();
            }
            ServerToClient::TaskSnapshot { task_id, snapshot } => {
                let items = conversation_to_items(&snapshot.conversation);
                let backend = snapshot.config.backend.clone();
                let model = snapshot.config.model.clone();
                let failure = snapshot.failure.clone();
                let allowlist = snapshot.tool_allowlist.clone();
                let view = self
                    .tasks
                    .entry(task_id.clone())
                    .or_insert_with(|| TaskView::new(snapshot_summary(&snapshot)));
                view.summary.state = snapshot.state;
                view.summary.title = snapshot.title;
                view.total_usage = snapshot.total_usage;
                view.items = items;
                view.subscribed = true;
                view.backend = backend;
                view.model = model;
                view.failure = failure;
                view.tool_allowlist = allowlist;
                // Pending-approval events that follow the snapshot will re-seed this.
                view.pending_approvals.clear();
            }
            ServerToClient::TaskAllowlistUpdated {
                task_id,
                tool_allowlist,
            } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.tool_allowlist = tool_allowlist;
                }
            }
            ServerToClient::TaskAssistantBegin { .. } => {}
            ServerToClient::TaskAssistantText { task_id, text } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.items.push(DisplayItem::AssistantText { text });
                }
            }
            ServerToClient::TaskAssistantReasoning { task_id, text } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.items.push(DisplayItem::Reasoning { text });
                }
            }
            ServerToClient::TaskAssistantTextDelta { task_id, delta } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    if let Some(DisplayItem::AssistantText { text }) = view.items.last_mut() {
                        text.push_str(&delta);
                    } else {
                        view.items.push(DisplayItem::AssistantText { text: delta });
                    }
                }
            }
            ServerToClient::TaskToolCallBegin {
                task_id,
                tool_use_id,
                name,
                args_preview,
            } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.items.push(DisplayItem::ToolCall {
                        tool_use_id,
                        name,
                        args_preview,
                        result: None,
                        is_error: false,
                    });
                }
            }
            ServerToClient::TaskToolCallEnd {
                task_id,
                tool_use_id,
                result_preview,
                is_error,
            } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    for item in view.items.iter_mut().rev() {
                        if let DisplayItem::ToolCall {
                            tool_use_id: id,
                            result,
                            is_error: err,
                            ..
                        } = item
                            && id == &tool_use_id
                        {
                            *result = Some(result_preview);
                            *err = is_error;
                            break;
                        }
                    }
                }
            }
            ServerToClient::TaskAssistantEnd { task_id, usage, .. } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.total_usage.add(&usage);
                }
            }
            ServerToClient::TaskLoopComplete { .. } => {}
            ServerToClient::TaskPendingApproval {
                task_id,
                approval_id,
                name,
                args_preview,
                destructive,
                read_only,
                ..
            } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    // Deduplicate — the same approval can arrive again if the client
                    // re-subscribes while a decision is still outstanding.
                    if !view
                        .pending_approvals
                        .iter()
                        .any(|p| p.approval_id == approval_id)
                    {
                        view.pending_approvals.push(PendingApproval {
                            approval_id,
                            name,
                            args_preview,
                            destructive,
                            read_only,
                            submitted: false,
                        });
                    }
                }
            }
            ServerToClient::TaskApprovalResolved {
                task_id,
                approval_id,
                ..
            } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.pending_approvals
                        .retain(|p| p.approval_id != approval_id);
                }
            }
            ServerToClient::Error {
                task_id, message, ..
            } => {
                if let Some(tid) = task_id.as_ref()
                    && let Some(view) = self.tasks.get_mut(tid)
                {
                    // Persist on the view so it's visible as a banner even after a
                    // resnapshot wipes `items`. The scheduler also records the same
                    // detail on the task's Failed state; this mirrors it locally so
                    // the UI doesn't have to wait on a re-subscribe round-trip.
                    view.failure = Some(message.clone());
                    view.items.push(DisplayItem::SystemNote {
                        text: message,
                        is_error: true,
                    });
                } else {
                    // No task scope — surface via conn detail so the banner reflects it.
                    self.conn_detail = Some(message);
                }
            }
            ServerToClient::BackendsList {
                default_backend,
                backends,
                ..
            } => {
                self.default_backend = default_backend;
                self.backends = backends;
                // Pre-fetch the default backend's models so the picker is ready on
                // first open without a visible delay.
                let default = self.default_backend.clone();
                self.request_models_for(&default);
            }
            ServerToClient::ModelsList {
                backend, models, ..
            } => {
                self.models_by_backend.insert(backend, models);
            }
            ServerToClient::ResourceList { resources, .. } => {
                self.resources.clear();
                for r in resources {
                    self.resources.insert(r.id().to_string(), r);
                }
            }
            ServerToClient::ResourceCreated { resource }
            | ServerToClient::ResourceUpdated { resource } => {
                self.resources.insert(resource.id().to_string(), resource);
            }
            ServerToClient::ResourceDestroyed { id, .. } => {
                self.resources.remove(&id);
            }
            // Phase 2d.i: pod CRUD wire surface lands but the webui's pod
            // list/editor are Phase 2e. Drop these so the build stays
            // exhaustive.
            ServerToClient::PodList { .. }
            | ServerToClient::PodSnapshot { .. }
            | ServerToClient::PodCreated { .. }
            | ServerToClient::PodConfigUpdated { .. }
            | ServerToClient::PodArchived { .. } => {}
        }
    }

    fn upsert_task(&mut self, summary: TaskSummary) {
        let id = summary.task_id.clone();
        self.tasks
            .entry(id)
            .and_modify(|v| v.summary = summary.clone())
            .or_insert_with(|| TaskView::new(summary));
    }

    fn recompute_order(&mut self) {
        let mut ids: Vec<String> = self.tasks.keys().cloned().collect();
        ids.sort_by(|a, b| {
            let ta = self
                .tasks
                .get(a)
                .map(|v| v.summary.created_at.clone())
                .unwrap_or_default();
            let tb = self
                .tasks
                .get(b)
                .map(|v| v.summary.created_at.clone())
                .unwrap_or_default();
            tb.cmp(&ta)
        });
        self.task_order = ids;
    }

    fn select_task(&mut self, task_id: String) {
        if self.selected.as_deref() == Some(&task_id) {
            return;
        }
        self.selected = Some(task_id.clone());
        self.composing_new = false;
        let need_subscribe = self
            .tasks
            .get(&task_id)
            .map(|v| !v.subscribed)
            .unwrap_or(true);
        if need_subscribe {
            self.send(ClientToServer::SubscribeToTask { task_id });
        }
    }

    fn submit(&mut self) {
        let text = std::mem::take(&mut self.input);
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.composing_new || self.selected.is_none() {
            let override_ = self.build_creation_override();
            self.send(ClientToServer::CreateTask {
                correlation_id: None,
                initial_message: trimmed.to_string(),
                config_override: override_,
            });
        } else if let Some(task_id) = self.selected.clone() {
            if let Some(view) = self.tasks.get_mut(&task_id) {
                view.items.push(DisplayItem::User {
                    text: trimmed.to_string(),
                });
            }
            self.send(ClientToServer::SendUserMessage {
                task_id,
                text: trimmed.to_string(),
            });
        }
    }

    /// Build a TaskConfigOverride from the picker's current state. Only includes the
    /// fields the user explicitly set — unset fields fall through to the server's
    /// default_task_config on the other end.
    fn build_creation_override(&self) -> Option<TaskConfigOverride> {
        let backend = self.picker_backend.clone();
        // If the user picked a backend but didn't touch the model dropdown, pin down
        // the model explicitly so the server doesn't fall back to the DEFAULT
        // backend's default_model (which would be wrong for the picked backend).
        // Prefer the picked backend's default_model, else the first fetched model,
        // else None (server will then pass empty to the backend, which single-model
        // local endpoints typically ignore).
        let model = self.picker_model.clone().or_else(|| {
            let b = backend.as_ref()?;
            self.backends
                .iter()
                .find(|bs| &bs.name == b)
                .and_then(|bs| bs.default_model.clone())
                .or_else(|| {
                    self.models_by_backend
                        .get(b)
                        .and_then(|list| list.first())
                        .map(|m| m.id.clone())
                })
        });
        let sandbox = match self.picker_sandbox {
            SandboxChoice::ServerDefault => None,
            SandboxChoice::None => Some(SandboxSpec::None),
            SandboxChoice::Preset(idx) => self.presets.get(idx).map(|p| p.to_spec()),
        };
        if backend.is_none() && model.is_none() && sandbox.is_none() {
            return None;
        }
        Some(TaskConfigOverride {
            backend,
            model,
            sandbox,
            ..Default::default()
        })
    }
}

fn snapshot_summary(s: &whisper_agent_protocol::TaskSnapshot) -> TaskSummary {
    TaskSummary {
        task_id: s.task_id.clone(),
        title: s.title.clone(),
        state: s.state,
        created_at: s.created_at.clone(),
        last_active: s.last_active.clone(),
    }
}

fn conversation_to_items(conv: &Conversation) -> Vec<DisplayItem> {
    let mut items = Vec::new();
    for msg in conv.messages() {
        add_message_items(msg, &mut items);
    }
    items
}

fn add_message_items(msg: &Message, out: &mut Vec<DisplayItem>) {
    match msg.role {
        Role::User => {
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        out.push(DisplayItem::User { text: text.clone() })
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let text = tool_result_text(content);
                        // Backfill the result onto a matching ToolCall item, or push a system note.
                        let mut matched = false;
                        for existing in out.iter_mut().rev() {
                            if let DisplayItem::ToolCall {
                                tool_use_id: id,
                                result,
                                is_error: err,
                                ..
                            } = existing
                                && id == tool_use_id
                                && result.is_none()
                            {
                                *result = Some(text.clone());
                                *err = *is_error;
                                matched = true;
                                break;
                            }
                        }
                        if !matched {
                            out.push(DisplayItem::SystemNote {
                                text: format!("[orphaned tool_result {tool_use_id}]: {text}"),
                                is_error: *is_error,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        Role::Assistant => {
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        out.push(DisplayItem::AssistantText { text: text.clone() });
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        out.push(DisplayItem::ToolCall {
                            tool_use_id: id.clone(),
                            name: name.clone(),
                            args_preview: truncate(
                                serde_json::to_string(input).unwrap_or_default(),
                                200,
                            ),
                            result: None,
                            is_error: false,
                        });
                    }
                    ContentBlock::Thinking { thinking, .. } => {
                        out.push(DisplayItem::Reasoning {
                            text: thinking.clone(),
                        });
                    }
                    _ => {}
                }
            }
        }
    }
}

fn tool_result_text(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(t) => t.clone(),
        ToolResultContent::Blocks(blocks) => {
            let mut out = String::new();
            for b in blocks {
                if let ContentBlock::Text { text } = b {
                    out.push_str(text);
                }
            }
            out
        }
    }
}

fn truncate(mut s: String, max: usize) -> String {
    if s.len() > max {
        s.truncate(max);
        s.push('…');
    }
    s
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_inbound();

        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("whisper-agent");
                ui.separator();
                let (label, color) = self.conn_status.label();
                ui.label(RichText::new(label).color(color));
                if let Some(d) = &self.conn_detail {
                    ui.label(RichText::new(d).color(Color32::from_gray(160)).small());
                }
                if let Some(view) = self.selected.as_ref().and_then(|id| self.tasks.get(id)) {
                    ui.separator();
                    let (text, c) = state_chip(view.summary.state);
                    ui.label(RichText::new(text).color(c));
                    let backend_label = if view.backend.is_empty() {
                        &self.default_backend
                    } else {
                        &view.backend
                    };
                    if !backend_label.is_empty() {
                        ui.separator();
                        ui.label(
                            RichText::new(format!("{}/{}", backend_label, view.model))
                                .color(Color32::from_gray(180))
                                .small(),
                        );
                    }
                    let u = view.total_usage;
                    ui.separator();
                    ui.label(
                        RichText::new(format!(
                            "tokens: {}↑ {}↓  cache: {}r/{}c",
                            u.input_tokens,
                            u.output_tokens,
                            u.cache_read_input_tokens,
                            u.cache_creation_input_tokens
                        ))
                        .color(Color32::from_gray(160))
                        .small(),
                    );
                }
            });
        });

        egui::SidePanel::left("task_list")
            .resizable(true)
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.left_mode == LeftPanelMode::Tasks, "Tasks")
                        .clicked()
                    {
                        self.left_mode = LeftPanelMode::Tasks;
                    }
                    if ui
                        .selectable_label(
                            self.left_mode == LeftPanelMode::Resources,
                            format!("Resources ({})", self.resources.len()),
                        )
                        .clicked()
                    {
                        self.left_mode = LeftPanelMode::Resources;
                    }
                });
                ui.separator();
                match self.left_mode {
                    LeftPanelMode::Tasks => self.render_task_list(ui),
                    LeftPanelMode::Resources => render_resource_list(ui, &self.resources),
                }
            });

        let input_enabled = matches!(self.conn_status, ConnectionStatus::Connected);
        let hint = if self.composing_new || self.selected.is_none() {
            if input_enabled {
                "Describe a new task"
            } else {
                "(connecting)"
            }
        } else {
            if input_enabled {
                "Message this task"
            } else {
                "(connecting)"
            }
        };

        let show_picker =
            (self.composing_new || self.selected.is_none()) && !self.backends.is_empty();
        let mut request_models: Option<String> = None;
        egui::TopBottomPanel::bottom("input_bar")
            .resizable(false)
            .show(ctx, |ui| {
                if show_picker {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("backend")
                                .small()
                                .color(Color32::from_gray(180)),
                        );
                        let current_backend = self.effective_picker_backend().to_string();
                        let before = current_backend.clone();
                        ComboBox::from_id_salt("picker_backend")
                            .selected_text(&current_backend)
                            .show_ui(ui, |ui| {
                                for b in &self.backends {
                                    ui.selectable_value(
                                        &mut self.picker_backend,
                                        Some(b.name.clone()),
                                        format!("{}  ({})", b.name, b.kind),
                                    );
                                }
                            });
                        let after = self.effective_picker_backend().to_string();
                        if before != after {
                            // Reset the model pick so the backend's default wins.
                            self.picker_model = None;
                            request_models = Some(after.clone());
                        }

                        ui.separator();
                        ui.label(
                            RichText::new("model")
                                .small()
                                .color(Color32::from_gray(180)),
                        );
                        let current_model = self.effective_picker_model();
                        let models_for_backend = self
                            .models_by_backend
                            .get(&after)
                            .cloned()
                            .unwrap_or_default();
                        ComboBox::from_id_salt("picker_model")
                            .selected_text(if current_model.is_empty() {
                                "(loading…)"
                            } else {
                                &current_model
                            })
                            .show_ui(ui, |ui| {
                                if models_for_backend.is_empty() {
                                    ui.label(
                                        RichText::new("(no models listed — defaults apply)")
                                            .small()
                                            .color(Color32::from_gray(160)),
                                    );
                                }
                                for m in &models_for_backend {
                                    let label = match &m.display_name {
                                        Some(d) => format!("{}  ({})", m.id, d),
                                        None => m.id.clone(),
                                    };
                                    ui.selectable_value(
                                        &mut self.picker_model,
                                        Some(m.id.clone()),
                                        label,
                                    );
                                }
                            });

                        ui.separator();
                        ui.label(
                            RichText::new("sandbox")
                                .small()
                                .color(Color32::from_gray(180)),
                        );
                        let sandbox_label: String = match self.picker_sandbox {
                            SandboxChoice::ServerDefault => "default".into(),
                            SandboxChoice::None => "none".into(),
                            SandboxChoice::Preset(idx) => self
                                .presets
                                .get(idx)
                                .map(|p| p.name.clone())
                                .unwrap_or_else(|| "(missing preset)".into()),
                        };
                        let mut open_modal = false;
                        ComboBox::from_id_salt("picker_sandbox")
                            .selected_text(sandbox_label)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.picker_sandbox,
                                    SandboxChoice::ServerDefault,
                                    "default  (server decides)",
                                );
                                ui.selectable_value(
                                    &mut self.picker_sandbox,
                                    SandboxChoice::None,
                                    "none  (bare metal)",
                                );
                                if !self.presets.is_empty() {
                                    ui.separator();
                                    for (i, p) in self.presets.iter().enumerate() {
                                        ui.selectable_value(
                                            &mut self.picker_sandbox,
                                            SandboxChoice::Preset(i),
                                            &p.name,
                                        );
                                    }
                                }
                                ui.separator();
                                if ui.button("+ New preset…").clicked() {
                                    open_modal = true;
                                }
                            });
                        if open_modal && self.preset_modal.is_none() {
                            self.preset_modal = Some(PresetModalState::new());
                        }
                    });
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if let Some(task_id) = self.selected.clone() {
                        if ui.button("Cancel").clicked() {
                            self.send(ClientToServer::CancelTask {
                                task_id: task_id.clone(),
                            });
                        }
                        if ui.button("Archive").clicked() {
                            self.send(ClientToServer::ArchiveTask { task_id });
                        }
                        ui.separator();
                    }
                    ui.add_enabled_ui(input_enabled, |ui| {
                        let send_pressed = ui.button("Send").clicked();
                        let response = ui.add_sized(
                            [ui.available_width(), 28.0],
                            TextEdit::singleline(&mut self.input).hint_text(hint),
                        );
                        let enter_pressed =
                            response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        if (send_pressed || enter_pressed) && input_enabled {
                            self.submit();
                            response.request_focus();
                        }
                    });
                });
                ui.add_space(4.0);
            });
        if let Some(backend) = request_models {
            self.request_models_for(&backend);
        }

        let mut pending_decisions: Vec<(String, String, ApprovalChoice, bool)> = Vec::new();
        let mut allowlist_revocations: Vec<(String, String)> = Vec::new();
        egui::CentralPanel::default().show(ctx, |ui| match self.selected.clone() {
            None => {
                ui.vertical_centered(|ui| {
                    ui.add_space(60.0);
                    ui.label(
                        RichText::new(
                            "no task selected — type a prompt below to create a new task",
                        )
                        .color(Color32::from_gray(140)),
                    );
                });
            }
            Some(task_id) => match self.tasks.get_mut(&task_id) {
                None => {
                    ui.label(
                        RichText::new(format!("task {task_id} not found"))
                            .color(Color32::from_rgb(220, 120, 120)),
                    );
                }
                Some(view) => {
                    render_failure_banner(ui, view);
                    render_allowlist_chips(ui, &task_id, view, &mut allowlist_revocations);
                    render_approval_banner(ui, &task_id, view, &mut pending_decisions);
                    ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                        if view.items.is_empty() {
                            ui.vertical_centered(|ui| {
                                ui.add_space(40.0);
                                ui.label(
                                    RichText::new("(no messages yet)")
                                        .color(Color32::from_gray(140)),
                                );
                            });
                        } else {
                            for item in &view.items {
                                render_item(ui, item);
                                ui.add_space(6.0);
                            }
                        }
                    });
                }
            },
        });

        // Submit any approval decisions that were clicked this frame.
        for (task_id, approval_id, decision, remember) in pending_decisions {
            self.send(ClientToServer::ApprovalDecision {
                task_id,
                approval_id,
                decision,
                remember,
            });
        }
        for (task_id, tool_name) in allowlist_revocations {
            self.send(ClientToServer::RemoveToolAllowlistEntry { task_id, tool_name });
        }

        self.render_preset_modal(ctx);
    }
}

impl ChatApp {
    /// Renders the "new sandbox preset" modal if open. On Save, appends a new
    /// preset to `self.presets`, persists, and selects it in the picker.
    fn render_task_list(&mut self, ui: &mut egui::Ui) {
        if ui.button("+ New task").clicked() {
            self.selected = None;
            self.composing_new = true;
            self.input.clear();
        }
        ui.separator();
        ScrollArea::vertical().show(ui, |ui| {
            let order = self.task_order.clone();
            for task_id in order {
                let Some(view) = self.tasks.get(&task_id) else {
                    continue;
                };
                let is_selected = self.selected.as_deref() == Some(&task_id);
                let title = view
                    .summary
                    .title
                    .clone()
                    .unwrap_or_else(|| task_id[..task_id.len().min(14)].to_string());
                let (chip, chip_color) = state_chip(view.summary.state);
                let row = ui.add_sized(
                    [ui.available_width(), 0.0],
                    egui::Button::selectable(
                        is_selected,
                        RichText::new(format!("{title}  [{chip}]")).color(if is_selected {
                            Color32::WHITE
                        } else {
                            chip_color
                        }),
                    ),
                );
                if row.clicked() {
                    self.select_task(task_id.clone());
                }
            }
        });
    }

    fn render_preset_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.preset_modal.take() else {
            return;
        };
        let mut open = true;
        let mut save_clicked = false;
        let mut cancel_clicked = false;

        egui::Window::new("New sandbox preset")
            .collapsible(false)
            .resizable(false)
            .default_width(420.0)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .open(&mut open)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("name");
                    ui.add(
                        TextEdit::singleline(&mut modal.name)
                            .hint_text("e.g. 'rust-dev'")
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("workspace");
                    ui.add(
                        TextEdit::singleline(&mut modal.workspace)
                            .hint_text("/absolute/path (read-write)")
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.add_space(4.0);
                ui.checkbox(
                    &mut modal.read_everywhere,
                    "Read everywhere (grant read-only access to the entire filesystem)",
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Sandbox uses Linux landlock. Files outside the workspace are \
                         read-only (if 'Read everywhere' is on) or denied entirely.",
                    )
                    .small()
                    .color(Color32::from_gray(160)),
                );
                ui.add_space(8.0);
                ui.separator();
                ui.horizontal(|ui| {
                    let save_enabled =
                        !modal.name.trim().is_empty() && !modal.workspace.trim().is_empty();
                    if ui
                        .add_enabled(save_enabled, egui::Button::new("Save"))
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
            let preset = LandlockPreset {
                name: modal.name.trim().to_string(),
                workspace: modal.workspace.trim().to_string(),
                read_everywhere: modal.read_everywhere,
            };
            self.presets.push(preset);
            save_presets(&self.presets);
            self.picker_sandbox = SandboxChoice::Preset(self.presets.len() - 1);
            // Modal closes (we don't put it back).
        } else if cancel_clicked || !open {
            // Modal closes.
        } else {
            // Stay open — restore modal state.
            self.preset_modal = Some(modal);
        }
    }
}

/// Persistent banner for a task that's entered the Failed state. Survives resnapshot
/// because `failure` is captured from the snapshot itself rather than derived from
/// the per-event items list.
fn render_resource_list(ui: &mut egui::Ui, resources: &HashMap<String, ResourceSnapshot>) {
    let mut sandboxes: Vec<&ResourceSnapshot> = Vec::new();
    let mut mcp_hosts: Vec<&ResourceSnapshot> = Vec::new();
    let mut backends: Vec<&ResourceSnapshot> = Vec::new();
    for r in resources.values() {
        match r {
            ResourceSnapshot::Sandbox { .. } => sandboxes.push(r),
            ResourceSnapshot::McpHost { .. } => mcp_hosts.push(r),
            ResourceSnapshot::Backend { .. } => backends.push(r),
        }
    }
    sandboxes.sort_by_key(|r| r.id().to_string());
    mcp_hosts.sort_by_key(|r| r.id().to_string());
    backends.sort_by_key(|r| r.id().to_string());

    ScrollArea::vertical().show(ui, |ui| {
        egui::CollapsingHeader::new(format!("Sandboxes ({})", sandboxes.len()))
            .default_open(true)
            .show(ui, |ui| {
                for r in &sandboxes {
                    render_resource_row(ui, r);
                }
                if sandboxes.is_empty() {
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
        ResourceSnapshot::Sandbox {
            id,
            spec,
            state,
            users,
            ..
        } => (id.clone(), spec_label(spec), *state, users.len()),
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
        ui.label(
            RichText::new(sub)
                .color(Color32::from_gray(150))
                .small(),
        );
    }
    ui.add_space(4.0);
}

fn spec_label(spec: &SandboxSpec) -> String {
    match spec {
        SandboxSpec::None => "no isolation".into(),
        SandboxSpec::Container { image, .. } => format!("container: {image}"),
        SandboxSpec::Landlock { allowed_paths, .. } => {
            format!("landlock · {} paths", allowed_paths.len())
        }
    }
}

fn resource_state_chip(state: ResourceStateLabel) -> (&'static str, Color32) {
    match state {
        ResourceStateLabel::Provisioning => ("provisioning", Color32::from_rgb(180, 160, 90)),
        ResourceStateLabel::Ready => ("ready", Color32::from_rgb(120, 180, 120)),
        ResourceStateLabel::Errored => ("errored", Color32::from_rgb(200, 110, 110)),
        ResourceStateLabel::TornDown => ("torn down", Color32::from_gray(140)),
    }
}

fn render_failure_banner(ui: &mut egui::Ui, view: &TaskView) {
    let Some(detail) = view.failure.as_deref() else {
        return;
    };
    if view.summary.state != TaskStateLabel::Failed {
        return;
    }
    egui::Frame::group(ui.style())
        .fill(Color32::from_rgb(64, 28, 28))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new("task failed")
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

fn render_approval_banner(
    ui: &mut egui::Ui,
    task_id: &str,
    view: &mut TaskView,
    pending_decisions: &mut Vec<(String, String, ApprovalChoice, bool)>,
) {
    if view.pending_approvals.is_empty() {
        return;
    }
    egui::Frame::group(ui.style())
        .fill(Color32::from_rgb(56, 44, 32))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new("approval required")
                        .color(Color32::from_rgb(240, 200, 120))
                        .strong(),
                );
                ui.add_space(2.0);
                for approval in view.pending_approvals.iter_mut() {
                    ui.horizontal_top(|ui| {
                        let hint = if approval.destructive {
                            RichText::new("destructive")
                                .color(Color32::from_rgb(220, 110, 110))
                                .small()
                        } else if approval.read_only {
                            RichText::new("read-only")
                                .color(Color32::from_gray(180))
                                .small()
                        } else {
                            RichText::new("unannotated")
                                .color(Color32::from_gray(180))
                                .small()
                        };
                        ui.label(hint);
                        ui.label(
                            RichText::new(format!("{}({})", approval.name, approval.args_preview))
                                .color(Color32::from_gray(220))
                                .monospace(),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.add_enabled_ui(!approval.submitted, |ui| {
                            if ui.button("Approve").clicked() {
                                approval.submitted = true;
                                pending_decisions.push((
                                    task_id.to_string(),
                                    approval.approval_id.clone(),
                                    ApprovalChoice::Approve,
                                    false,
                                ));
                            }
                            if ui
                                .button(format!("Always allow {}", approval.name))
                                .on_hover_text(
                                    "Approve this call and skip future approvals for \
                                     this tool name (this task only).",
                                )
                                .clicked()
                            {
                                approval.submitted = true;
                                pending_decisions.push((
                                    task_id.to_string(),
                                    approval.approval_id.clone(),
                                    ApprovalChoice::Approve,
                                    true,
                                ));
                            }
                            if ui.button("Reject").clicked() {
                                approval.submitted = true;
                                pending_decisions.push((
                                    task_id.to_string(),
                                    approval.approval_id.clone(),
                                    ApprovalChoice::Reject,
                                    false,
                                ));
                            }
                        });
                        if approval.submitted {
                            ui.label(
                                RichText::new("submitted…")
                                    .color(Color32::from_gray(160))
                                    .italics(),
                            );
                        }
                    });
                    ui.add_space(4.0);
                }
            });
        });
    ui.add_space(6.0);
}

/// Compact strip of "always-allowed" tool names with revoke buttons. Hidden when
/// the allowlist is empty so it doesn't take vertical space in the common case.
fn render_allowlist_chips(
    ui: &mut egui::Ui,
    task_id: &str,
    view: &TaskView,
    revocations: &mut Vec<(String, String)>,
) {
    if view.tool_allowlist.is_empty() {
        return;
    }
    ui.horizontal_wrapped(|ui| {
        ui.label(
            RichText::new("always allow:")
                .small()
                .color(Color32::from_gray(170)),
        );
        for tool_name in &view.tool_allowlist {
            ui.label(
                RichText::new(tool_name)
                    .small()
                    .color(Color32::from_rgb(170, 200, 230))
                    .monospace(),
            );
            if ui
                .small_button("×")
                .on_hover_text(format!("Stop always-allowing `{tool_name}`"))
                .clicked()
            {
                revocations.push((task_id.to_string(), tool_name.clone()));
            }
            ui.add_space(2.0);
        }
    });
    ui.add_space(4.0);
}

fn render_item(ui: &mut egui::Ui, item: &DisplayItem) {
    match item {
        DisplayItem::User { text } => {
            ui.horizontal_top(|ui| {
                ui.label(
                    RichText::new("you")
                        .color(Color32::from_rgb(120, 180, 240))
                        .strong(),
                );
                ui.label(text);
            });
        }
        DisplayItem::AssistantText { text } => {
            ui.horizontal_top(|ui| {
                ui.label(
                    RichText::new("agent")
                        .color(Color32::from_rgb(160, 220, 160))
                        .strong(),
                );
                ui.label(text);
            });
        }
        DisplayItem::Reasoning { text } => {
            ui.horizontal_top(|ui| {
                ui.label(
                    RichText::new("think")
                        .color(Color32::from_rgb(180, 160, 220))
                        .strong(),
                );
                let preview = text.lines().next().unwrap_or("").trim();
                let header = if preview.len() > 80 {
                    format!("{}…", &preview[..80])
                } else if preview.is_empty() {
                    "(reasoning)".to_string()
                } else {
                    preview.to_string()
                };
                // Default-collapsed: reasoning is preserved but doesn't dominate the view.
                egui::CollapsingHeader::new(
                    RichText::new(header)
                        .color(Color32::from_gray(140))
                        .italics(),
                )
                .id_salt(("reasoning", text.as_ptr() as usize))
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(RichText::new(text).color(Color32::from_gray(170)).italics());
                });
            });
        }
        DisplayItem::ToolCall {
            name,
            args_preview,
            result,
            is_error,
            ..
        } => {
            ui.horizontal_top(|ui| {
                ui.label(
                    RichText::new("tool")
                        .color(Color32::from_rgb(220, 180, 100))
                        .strong(),
                );
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(format!("{name}({args_preview})"))
                            .color(Color32::from_gray(200))
                            .monospace(),
                    );
                    match result {
                        None => {
                            ui.label(
                                RichText::new("running…")
                                    .color(Color32::from_gray(140))
                                    .italics(),
                            );
                        }
                        Some(text) => {
                            let color = if *is_error {
                                Color32::from_rgb(220, 120, 120)
                            } else {
                                Color32::from_gray(180)
                            };
                            ui.label(RichText::new(text).color(color).monospace().small());
                        }
                    }
                });
            });
        }
        DisplayItem::SystemNote { text, is_error } => {
            let color = if *is_error {
                Color32::from_rgb(220, 120, 120)
            } else {
                Color32::from_gray(160)
            };
            ui.label(RichText::new(text).color(color).italics());
        }
    }
}
