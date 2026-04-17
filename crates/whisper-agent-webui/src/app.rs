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
//! selection we send `SubscribeToThread`; the server's `ThreadSnapshot` response rebuilds
//! the task's display items. Subsequent turn events append to them.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::rc::Rc;

use egui::{Color32, ComboBox, DragValue, Grid, RichText, ScrollArea, TextEdit};
use whisper_agent_protocol::sandbox::{
    AccessMode, Mount, NetworkPolicy, PathAccess, ResourceLimits,
};
use whisper_agent_protocol::{
    ApprovalChoice, ApprovalPolicy, BackendSummary, ClientToServer, ContentBlock, Conversation,
    HostEnvProviderInfo, Message, ModelSummary, NamedHostEnv, PodAllow, PodConfig, PodLimits, PodSummary,
    ResourceSnapshot, ResourceStateLabel, Role, HostEnvSpec, ServerToClient, ThreadBindingsRequest,
    ThreadConfigOverride, ThreadDefaults, ThreadStateLabel, ThreadSummary, ToolResultContent, Usage,
};

/// Events pushed into [`Inbound`]. In addition to decoded wire messages we pipe in
/// connection-level signals (open/close/error) so the UI can show a connection status
/// distinct from per-task state.
// `Wire(ThreadSnapshot)` dwarfs the connection variants. Boxing would change every
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

fn state_chip(state: ThreadStateLabel) -> (&'static str, Color32) {
    match state {
        ThreadStateLabel::Idle => ("idle", Color32::from_gray(160)),
        ThreadStateLabel::Working => ("working", Color32::from_rgb(120, 180, 240)),
        ThreadStateLabel::AwaitingApproval => ("approval", Color32::from_rgb(240, 180, 90)),
        ThreadStateLabel::Completed => ("completed", Color32::from_rgb(120, 200, 140)),
        ThreadStateLabel::Failed => ("failed", Color32::from_rgb(220, 110, 110)),
        ThreadStateLabel::Cancelled => ("cancelled", Color32::from_rgb(180, 140, 140)),
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
        /// Short header summary, e.g. for `edit_file` we render a one-line
        /// `path` instead of dumping the JSON. Falls back to truncated
        /// JSON when no specialized summary applies.
        summary: String,
        /// Pretty-printed JSON args for the expanded raw view. `None`
        /// when the server didn't carry full args (legacy snapshot path
        /// before the protocol change, or future size-cap rejection).
        args_pretty: Option<String>,
        /// Pre-computed diff payload — populated only for `edit_file` /
        /// `write_file` tool calls when full args are available. Lets
        /// the renderer show a unified diff inline.
        diff: Option<DiffPayload>,
        result: Option<String>,
        is_error: bool,
    },
    SystemNote {
        text: String,
        is_error: bool,
    },
}

/// Pre-computed inputs for the unified-diff renderer. Built once at
/// item-build time so the renderer doesn't have to re-parse JSON every
/// frame. `is_creation = true` for `write_file` (no prior content;
/// renderer shows it as all-`+` lines).
#[derive(Clone)]
struct DiffPayload {
    path: String,
    old_text: String,
    new_text: String,
    is_creation: bool,
}

struct TaskView {
    summary: ThreadSummary,
    items: Vec<DisplayItem>,
    total_usage: Usage,
    subscribed: bool,
    /// Currently-open approval requests, in arrival order. Cleared on snapshot so
    /// re-subscribe can re-seed them without duplicates.
    pending_approvals: Vec<PendingApproval>,
    /// Backend alias the server resolved for this task. Populated from ThreadSnapshot.
    /// Empty string means "server default" — the status bar resolves that to the
    /// known default_backend name at render time.
    backend: String,
    /// Model the task was created with. Populated from ThreadSnapshot.
    model: String,
    /// Failure detail carried by the snapshot, if the task ended up in `Failed`.
    /// Rendered as a persistent banner so it survives re-subscribe (unlike items,
    /// which get rebuilt from the conversation on every snapshot).
    failure: Option<String>,
    /// Tool names this task has approved-and-remembered. Mirrors
    /// `ThreadSnapshot.tool_allowlist`; refreshed by `ThreadAllowlistUpdated`.
    tool_allowlist: Vec<String>,
}

struct PendingApproval {
    approval_id: String,
    name: String,
    args_preview: String,
    destructive: bool,
    read_only: bool,
    /// True after the local user clicked Approve/Reject — buttons disable until the
    /// server's ThreadApprovalResolved arrives (and removes the entry entirely).
    submitted: bool,
}

impl TaskView {
    fn new(summary: ThreadSummary) -> Self {
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
    /// Name of the `[[allow.host_env]]` entry the compose form is
    /// targeting for the new thread. `None` = inherit the target pod's
    /// `thread_defaults.host_env`. Always resolves to an entry in the
    /// pod's allow list — the webui never invents inline specs, and
    /// the server rejects unknown names. Reset back to `None` after
    /// every submit so the next compose starts from the pod default,
    /// not whatever the previous thread used.
    compose_host_env: Option<String>,

    // --- Resource registry (Phase 1c read-only inspector) ---
    /// Snapshot of every resource the server has reported. Keyed by resource id.
    resources: HashMap<String, ResourceSnapshot>,
    resources_requested: bool,

    // --- Pods (Phase 2e: pod-grouped left panel) ---
    /// Pod summaries keyed by `pod_id`. Used to render the pod headers in the
    /// left panel and resolve display names for thread rows. Threads carry
    /// `pod_id` directly (since 2d.iii) so the source of truth for "which
    /// threads are in this pod" lives in `tasks`, not here.
    pods: HashMap<String, PodSummary>,
    pods_requested: bool,
    /// Server's host-env-provider catalog. Populated lazily on first
    /// ListHostEnvProviders round-trip; used by the pod editor's
    /// per-entry provider dropdown. Can be empty — a server started
    /// without any `[[host_env_providers]]` entries is a valid
    /// configuration; pods in it just can't declare host envs.
    host_env_providers: Vec<HostEnvProviderInfo>,
    host_env_providers_requested: bool,
    /// Set of pod ids the user has manually collapsed in the left panel.
    /// Default is "all expanded"; toggling a header inverts membership.
    /// Persisted only in memory — re-expands across reloads.
    collapsed_pods: HashSet<String>,
    /// Modal state for the "+ New pod" form. `None` = closed.
    new_pod_modal: Option<NewPodModalState>,
    /// Pod id whose "Archive" button has been clicked once and is waiting
    /// for confirmation. Cleared by clicking elsewhere or confirming.
    /// At most one pod can be armed at a time — clicking a different pod's
    /// Archive button replaces it.
    archive_armed_pod: Option<String>,
    /// Modal state for the per-pod raw-TOML config editor. `None` = closed.
    pod_editor_modal: Option<PodEditorModalState>,
    /// Monotonic counter used to mint correlation_ids for in-flight
    /// requests we want to match round-trip events to. Intentionally
    /// not persisted — collisions across reloads aren't a problem
    /// because every WebSocket reconnect drops in-flight state.
    next_correlation_seq: u64,
    /// Pod the in-progress new-thread compose targets. `None` = the
    /// server's default pod. Set by the per-pod "+ Thread" button in
    /// each pod section header; cleared/reset by the global "+ New
    /// thread" button. Only meaningful when
    /// `composing_new || selected.is_none()`.
    compose_pod_id: Option<String>,
    /// Server-reported id of the pod that receives `CreateThread {
    /// pod_id: None }`. Lifted out of `PodList` so the webui knows which
    /// pod to clone when bootstrapping fresh pods.
    server_default_pod_id: String,
    /// Cached config of the server's default pod, fetched lazily after
    /// `PodList`. Used as the template for "+ New pod" so a fresh pod
    /// inherits the working sandbox / shared-mcp setup instead of
    /// starting from a stub. `None` until the `GetPod` round-trip lands.
    default_pod_template: Option<PodConfig>,
    /// Cached full `PodConfig` keyed by pod id. Populated lazily via
    /// `GetPod` when the compose form opens — the webui needs the
    /// pod's `allow.host_env` table to render the host-env picker
    /// without inventing state of its own. `pod_configs_requested`
    /// tracks in-flight fetches so we don't spam round-trips on every
    /// repaint.
    pod_configs: HashMap<String, PodConfig>,
    pod_configs_requested: HashSet<String>,

    /// Which view the left side panel is showing.
    left_mode: LeftPanelMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LeftPanelMode {
    /// Pod-grouped tree of every thread the user can see.
    #[default]
    Threads,
    Resources,
}

/// State for the per-pod config editor. The modal is tabbed: three
/// structured tabs (Allow / Defaults / Limits) edit a working
/// `PodConfig` directly, and a fourth (Raw TOML) is the escape hatch
/// for paste-and-go or fields the structured form doesn't cover (e.g.
/// every `[[allow.host_env]]` body has its own dedicated sub-modal, but
/// power users can still hand-edit them in raw text).
///
/// Lifecycle: open then issue `GetPod`; on snapshot, populate `working`
/// and `server_baseline`; user edits; Save serializes `working` (or
/// `raw_buffer` if the Raw tab is canonical) and sends `UpdatePodConfig`
/// with a fresh correlation_id. Server validation errors land via the
/// matching `Error` event and surface inline, leaving the user's edits
/// intact for fixing.
struct PodEditorModalState {
    pod_id: String,
    /// Working in-memory edit. `None` until the snapshot lands.
    /// All structured tabs read/write this directly.
    working: Option<PodConfig>,
    /// The server's last known config — used as the Revert baseline
    /// and to drive the dirty indicator (Save is enabled only when
    /// `working` differs from this).
    server_baseline: Option<PodConfig>,
    /// Backing buffer for the Raw TOML tab. Whenever the user enters
    /// the Raw tab, we regenerate this from `working` *unless*
    /// `raw_dirty` says they have unsaved raw edits to preserve.
    raw_buffer: String,
    /// True when the Raw tab's text has diverged from the structured
    /// `working`. Cleared whenever `working` is updated from raw or
    /// vice-versa (e.g. on tab switch with a clean Raw buffer).
    raw_dirty: bool,
    /// Active tab.
    tab: PodEditorTab,
    /// Last server validation error, surfaced inline in the footer.
    /// Cleared when the user starts a new save or switches tabs.
    error: Option<String>,
    /// Set to the correlation_id of the in-flight `UpdatePodConfig`.
    /// `Some` means a save is currently in flight (Save button reads
    /// "saving…" and is disabled).
    pending_correlation: Option<String>,
    /// Sub-modal state for editing one `[[allow.host_env]]` entry.
    /// `Some` means the sub-modal is open and consuming input
    /// (the parent tabs render but are non-interactive while it's up).
    sandbox_entry_editor: Option<SandboxEntryEditorState>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PodEditorTab {
    Allow,
    Defaults,
    Limits,
    RawToml,
}

impl PodEditorTab {
    fn label(self) -> &'static str {
        match self {
            PodEditorTab::Allow => "Allow",
            PodEditorTab::Defaults => "Thread defaults",
            PodEditorTab::Limits => "Limits",
            PodEditorTab::RawToml => "Raw TOML",
        }
    }
}

impl PodEditorModalState {
    fn new(pod_id: String) -> Self {
        Self {
            pod_id,
            working: None,
            server_baseline: None,
            raw_buffer: String::new(),
            raw_dirty: false,
            tab: PodEditorTab::Allow,
            error: None,
            pending_correlation: None,
            sandbox_entry_editor: None,
        }
    }

    /// Has the user changed anything since the snapshot landed?
    fn is_dirty(&self) -> bool {
        match (&self.working, &self.server_baseline) {
            (Some(w), Some(s)) => w != s || self.raw_dirty,
            _ => false,
        }
    }
}

/// Sub-modal for editing one `[[allow.host_env]]` entry. Stays in front
/// of the parent pod editor; the parent's tabs render but are
/// non-interactive while this is open. Save writes the entry back to
/// `working.allow.host_env` at `index` (or appends when `index` is
/// `None`).
struct SandboxEntryEditorState {
    /// Position in `working.allow.host_env`. `None` for "add new".
    index: Option<usize>,
    /// Working copy of the entry — applied back to `working` on save.
    entry: NamedHostEnv,
    /// Local validation hint (empty name, etc.). Server-side checks
    /// land later via the parent modal's pod-level Save round-trip.
    error: Option<String>,
}

impl SandboxEntryEditorState {
    fn new_for_index(index: usize, entry: NamedHostEnv) -> Self {
        Self {
            index: Some(index),
            entry,
            error: None,
        }
    }

    fn new_for_add(default_provider: Option<&str>) -> Self {
        // Seed with the first configured provider so the dropdown lands
        // on a valid choice. Falls back to an empty string when no
        // providers are configured — the "Save" button then blocks via
        // validation until the user fixes it.
        let provider = default_provider.unwrap_or_default().to_string();
        Self {
            index: None,
            entry: NamedHostEnv {
                name: String::new(),
                provider,
                spec: HostEnvSpec::Landlock {
                    allowed_paths: Vec::new(),
                    network: NetworkPolicy::Unrestricted,
                },
            },
            error: None,
        }
    }
}

/// State for the "+ New pod" modal. The user picks a directory-friendly
/// `pod_id` (immutable on disk) and a display `name` (free text). The
/// resulting pod inherits the server default pod's shape — same
/// backends, shared MCP hosts, and host-env allow list — so freshly
/// created pods are immediately usable. The pod editor is the only
/// place to tighten or extend these.
struct NewPodModalState {
    pod_id: String,
    name: String,
    error: Option<String>,
}

impl NewPodModalState {
    fn new() -> Self {
        Self {
            pod_id: String::new(),
            name: String::new(),
            error: None,
        }
    }
}

/// Validate a pod_id against the rules `Persister::create_pod` will
/// enforce server-side. Done client-side so the user sees the error
/// inline instead of as a returned `ServerToClient::Error`.
fn validate_pod_id_client(id: &str) -> Result<(), &'static str> {
    if id.is_empty() {
        return Err("pod_id is empty");
    }
    if id.starts_with('.') {
        return Err("pod_id may not start with '.'");
    }
    if id.contains('/') || id.contains('\\') || id.contains('\0') || id == ".." {
        return Err("pod_id contains illegal characters");
    }
    Ok(())
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
            compose_host_env: None,
            resources: HashMap::new(),
            resources_requested: false,
            pods: HashMap::new(),
            pods_requested: false,
            host_env_providers: Vec::new(),
            host_env_providers_requested: false,
            collapsed_pods: HashSet::new(),
            new_pod_modal: None,
            archive_armed_pod: None,
            pod_editor_modal: None,
            next_correlation_seq: 0,
            compose_pod_id: None,
            server_default_pod_id: String::new(),
            default_pod_template: None,
            pod_configs: HashMap::new(),
            pod_configs_requested: HashSet::new(),
            left_mode: LeftPanelMode::default(),
        }
    }

    /// Pod the compose form currently targets. `compose_pod_id` is
    /// `None` when the user clicked the global "+ New thread" button —
    /// we fall back to the server's default pod. Returns `None` when
    /// neither is known yet (brand-new connection before PodList).
    fn compose_target_pod_id(&self) -> Option<&str> {
        self.compose_pod_id
            .as_deref()
            .or_else(|| {
                if self.server_default_pod_id.is_empty() {
                    None
                } else {
                    Some(&self.server_default_pod_id)
                }
            })
    }

    /// Ensure the target pod's config is cached. Dispatches a GetPod
    /// round-trip the first time a given pod is needed; later composes
    /// reuse the cached snapshot. Pod-config updates arrive as
    /// `PodConfigUpdated` events, which overwrite the cached copy.
    fn ensure_pod_config(&mut self, pod_id: &str) {
        if self.pod_configs.contains_key(pod_id)
            || self.pod_configs_requested.contains(pod_id)
        {
            return;
        }
        self.pod_configs_requested.insert(pod_id.to_string());
        self.send(ClientToServer::GetPod {
            pod_id: pod_id.to_string(),
            correlation_id: None,
        });
    }

    fn next_correlation_id(&mut self) -> String {
        self.next_correlation_seq = self.next_correlation_seq.wrapping_add(1);
        format!("c{}", self.next_correlation_seq)
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
                    self.send(ClientToServer::ListThreads {
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
                if !self.pods_requested {
                    self.send(ClientToServer::ListPods {
                        correlation_id: None,
                    });
                    self.pods_requested = true;
                }
                if !self.host_env_providers_requested {
                    self.send(ClientToServer::ListHostEnvProviders {
                        correlation_id: None,
                    });
                    self.host_env_providers_requested = true;
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
            ServerToClient::ThreadCreated {
                thread_id,
                summary,
                correlation_id: _,
            } => {
                self.upsert_task(summary);
                self.recompute_order();
                // Auto-select newly created tasks.
                if self.selected.as_deref() != Some(&thread_id) {
                    self.select_task(thread_id);
                }
            }
            ServerToClient::ThreadStateChanged { thread_id, state } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.summary.state = state;
                }
            }
            ServerToClient::ThreadTitleUpdated { thread_id, title } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.summary.title = Some(title);
                }
            }
            ServerToClient::ThreadArchived { thread_id } => {
                self.tasks.remove(&thread_id);
                self.task_order.retain(|id| id != &thread_id);
                if self.selected.as_deref() == Some(&thread_id) {
                    self.selected = None;
                    self.composing_new = true;
                }
            }
            ServerToClient::ThreadList { tasks, .. } => {
                self.tasks
                    .retain(|id, _| tasks.iter().any(|t| &t.thread_id == id));
                for summary in tasks {
                    self.upsert_task(summary);
                }
                self.recompute_order();
            }
            ServerToClient::ThreadSnapshot { thread_id, snapshot } => {
                let items = conversation_to_items(&snapshot.conversation);
                let backend = snapshot.bindings.backend.clone();
                let model = snapshot.config.model.clone();
                let failure = snapshot.failure.clone();
                let allowlist = snapshot.tool_allowlist.clone();
                let view = self
                    .tasks
                    .entry(thread_id.clone())
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
            ServerToClient::ThreadAllowlistUpdated {
                thread_id,
                tool_allowlist,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.tool_allowlist = tool_allowlist;
                }
            }
            ServerToClient::ThreadBindingsChanged {
                thread_id, bindings, ..
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.backend = bindings.backend;
                }
            }
            ServerToClient::ThreadAssistantBegin { .. } => {}
            ServerToClient::ThreadAssistantText { thread_id, text } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.items.push(DisplayItem::AssistantText { text });
                }
            }
            ServerToClient::ThreadAssistantReasoning { thread_id, text } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.items.push(DisplayItem::Reasoning { text });
                }
            }
            ServerToClient::ThreadAssistantTextDelta { thread_id, delta } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    if let Some(DisplayItem::AssistantText { text }) = view.items.last_mut() {
                        text.push_str(&delta);
                    } else {
                        view.items.push(DisplayItem::AssistantText { text: delta });
                    }
                }
            }
            ServerToClient::ThreadToolCallBegin {
                thread_id,
                tool_use_id,
                name,
                args_preview,
                args,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.items.push(build_tool_call_item(
                        tool_use_id,
                        name,
                        args.as_ref(),
                        args_preview,
                    ));
                }
            }
            ServerToClient::ThreadToolCallEnd {
                thread_id,
                tool_use_id,
                result_preview,
                is_error,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
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
            ServerToClient::ThreadAssistantEnd { thread_id, usage, .. } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.total_usage.add(&usage);
                }
            }
            ServerToClient::ThreadLoopComplete { .. } => {}
            ServerToClient::ThreadPendingApproval {
                thread_id,
                approval_id,
                name,
                args_preview,
                destructive,
                read_only,
                ..
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
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
            ServerToClient::ThreadApprovalResolved {
                thread_id,
                approval_id,
                ..
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.pending_approvals
                        .retain(|p| p.approval_id != approval_id);
                }
            }
            ServerToClient::Error {
                thread_id,
                message,
                correlation_id,
            } => {
                // If the editor minted this correlation, surface the error
                // inline in the modal instead of as a global banner —
                // failed validation should leave the user's edits intact.
                if let Some(modal) = self.pod_editor_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                if let Some(tid) = thread_id.as_ref()
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
            ServerToClient::HostEnvProvidersList { providers, .. } => {
                self.host_env_providers = providers;
            }
            ServerToClient::PodList {
                pods,
                default_pod_id,
                ..
            } => {
                self.pods = pods.into_iter().map(|p| (p.pod_id.clone(), p)).collect();
                // Fetch the default pod's config so "+ New pod" can clone
                // its sandbox / shared-mcp setup. Cheap round-trip; only
                // sent when the id changes (guarded by string equality).
                if !default_pod_id.is_empty() && default_pod_id != self.server_default_pod_id {
                    self.server_default_pod_id = default_pod_id.clone();
                    self.send(ClientToServer::GetPod {
                        correlation_id: None,
                        pod_id: default_pod_id,
                    });
                }
            }
            ServerToClient::PodCreated { pod, .. } => {
                self.pods.insert(pod.pod_id.clone(), pod);
            }
            ServerToClient::PodConfigUpdated {
                pod_id,
                toml_text,
                parsed,
                correlation_id,
            } => {
                // Mirror the new top-level fields (name/description) onto the
                // summary so the left panel reflects edits without waiting
                // for a full ListPods refresh. thread_count is unchanged by
                // a config edit.
                if let Some(summary) = self.pods.get_mut(&pod_id) {
                    summary.name = parsed.name.clone();
                    summary.description = parsed.description.clone();
                    summary.created_at = parsed.created_at.clone();
                }
                if pod_id == self.server_default_pod_id {
                    self.default_pod_template = Some(parsed.clone());
                }
                // Refresh the compose-form cache so an open compose
                // picker sees the edit immediately.
                self.pod_configs.insert(pod_id.clone(), parsed.clone());
                // The editor stays open across saves — refresh its
                // baseline so subsequent edits are diffed against the
                // newly-persisted state, not the stale one. We keep
                // the user's `working` value if they're matching the
                // correlation we just sent (their own save
                // round-tripped — `working` already matches `parsed`),
                // and we replace `working` if this update came from
                // another client (otherwise their off-screen edits
                // would silently clobber the local view).
                if let Some(modal) = self.pod_editor_modal.as_mut()
                    && modal.pod_id == pod_id
                {
                    let our_save = modal.pending_correlation.is_some()
                        && modal.pending_correlation == correlation_id;
                    modal.server_baseline = Some(parsed.clone());
                    if our_save {
                        // Refresh `working` from the server's parse too —
                        // necessary when we saved from the Raw tab (where
                        // `working` wasn't the source of truth) and a no-op
                        // when we saved from a structured tab.
                        modal.working = Some(parsed);
                        modal.pending_correlation = None;
                        modal.error = None;
                        modal.raw_buffer = toml_text;
                        modal.raw_dirty = false;
                    } else if !modal.is_dirty() {
                        // Foreign update + we have no local edits =>
                        // adopt it cleanly. If we *do* have edits,
                        // leave them alone; the next Save will collide
                        // server-side and we'll show that error.
                        modal.working = Some(parsed);
                        modal.raw_buffer = toml_text;
                        modal.raw_dirty = false;
                        modal.error = None;
                    }
                }
            }
            ServerToClient::PodArchived { pod_id } => {
                self.pods.remove(&pod_id);
                // Drop any threads we were tracking under the archived pod —
                // the server won't send further events for them and they're
                // unreachable from the UI now.
                self.tasks.retain(|_, v| v.summary.pod_id != pod_id);
                self.recompute_order();
                if let Some(sel) = &self.selected
                    && !self.tasks.contains_key(sel)
                {
                    self.selected = None;
                    self.composing_new = true;
                }
                if self.compose_pod_id.as_deref() == Some(pod_id.as_str()) {
                    self.compose_pod_id = None;
                }
                if self.archive_armed_pod.as_deref() == Some(pod_id.as_str()) {
                    self.archive_armed_pod = None;
                }
            }
            ServerToClient::PodSnapshot { snapshot, .. } => {
                // Cache the default pod's config as a template for fresh
                // "+ New pod" creation.
                if snapshot.pod_id == self.server_default_pod_id {
                    self.default_pod_template = Some(snapshot.config.clone());
                }
                // Populate the editor modal if it's open and waiting on
                // this pod's text.
                if let Some(modal) = self.pod_editor_modal.as_mut()
                    && modal.pod_id == snapshot.pod_id
                    && modal.working.is_none()
                {
                    modal.server_baseline = Some(snapshot.config.clone());
                    modal.working = Some(snapshot.config.clone());
                    modal.raw_buffer = snapshot.toml_text.clone();
                    modal.raw_dirty = false;
                }
                // Update the compose-form cache so the host-env picker
                // reflects the current pod config even when the user
                // re-edits the pod without closing the compose form.
                self.pod_configs_requested.remove(&snapshot.pod_id);
                self.pod_configs.insert(snapshot.pod_id.clone(), snapshot.config);
            }
        }
    }

    fn upsert_task(&mut self, summary: ThreadSummary) {
        let id = summary.thread_id.clone();
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

    fn select_task(&mut self, thread_id: String) {
        if self.selected.as_deref() == Some(&thread_id) {
            return;
        }
        self.selected = Some(thread_id.clone());
        self.composing_new = false;
        self.compose_pod_id = None;
        let need_subscribe = self
            .tasks
            .get(&thread_id)
            .map(|v| !v.subscribed)
            .unwrap_or(true);
        if need_subscribe {
            self.send(ClientToServer::SubscribeToThread { thread_id });
        }
    }

    fn submit(&mut self) {
        let text = std::mem::take(&mut self.input);
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.composing_new || self.selected.is_none() {
            let (config_override, bindings_request) = self.build_creation_override();
            self.send(ClientToServer::CreateThread {
                correlation_id: None,
                pod_id: self.compose_pod_id.clone(),
                initial_message: trimmed.to_string(),
                config_override,
                bindings_request,
            });
            // Reset to "inherit" so the next compose doesn't silently
            // reuse the previous thread's override.
            self.compose_host_env = None;
        } else if let Some(thread_id) = self.selected.clone() {
            if let Some(view) = self.tasks.get_mut(&thread_id) {
                view.items.push(DisplayItem::User {
                    text: trimmed.to_string(),
                });
            }
            self.send(ClientToServer::SendUserMessage {
                thread_id,
                text: trimmed.to_string(),
            });
        }
    }

    /// Build the override pair for a CreateThread from the picker's current
    /// state. Only includes the fields the user explicitly set — anything
    /// unset falls through to the pod's `thread_defaults` on the server.
    /// Returns `(config_override, bindings_request)`; either may be `None`
    /// when the user didn't touch the corresponding picker.
    fn build_creation_override(
        &self,
    ) -> (Option<ThreadConfigOverride>, Option<ThreadBindingsRequest>) {
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
        // Host env is always a reference by name into the target
        // pod's `allow.host_env` table. `None` → server applies
        // `thread_defaults.host_env`. The webui never constructs
        // inline specs; the server rejects Inline at the wire boundary.
        let host_env = self.compose_host_env.clone();
        let config_override = if model.is_some() {
            Some(ThreadConfigOverride {
                model,
                ..Default::default()
            })
        } else {
            None
        };
        let bindings_request = if backend.is_some() || host_env.is_some() {
            Some(ThreadBindingsRequest {
                backend,
                host_env,
                mcp_hosts: None,
            })
        } else {
            None
        };
        (config_override, bindings_request)
    }
}

fn snapshot_summary(s: &whisper_agent_protocol::ThreadSnapshot) -> ThreadSummary {
    ThreadSummary {
        thread_id: s.thread_id.clone(),
        pod_id: s.pod_id.clone(),
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
                        let preview =
                            truncate(serde_json::to_string(input).unwrap_or_default(), 200);
                        out.push(build_tool_call_item(
                            id.clone(),
                            name.clone(),
                            Some(input),
                            preview,
                        ));
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

/// Build a `DisplayItem::ToolCall` from the wire shape (full input
/// `Value` available). `args_preview` is the server-truncated string;
/// we only fall back to it for the summary when the structured args
/// don't have a specialized renderer.
///
/// Special cases:
///   * `edit_file` / `write_file` — pull `path` for the header, and
///     compute a `DiffPayload` so the renderer can show old vs new.
///   * `bash` — pull `command` for the header.
///   * everything else — use the truncated JSON preview as the summary.
fn build_tool_call_item(
    tool_use_id: String,
    name: String,
    args: Option<&serde_json::Value>,
    args_preview: String,
) -> DisplayItem {
    let summary = tool_summary(&name, args, &args_preview);
    let args_pretty = args.and_then(|v| serde_json::to_string_pretty(v).ok());
    let diff = args.and_then(|v| extract_diff(&name, v));
    DisplayItem::ToolCall {
        tool_use_id,
        name,
        summary,
        args_pretty,
        diff,
        result: None,
        is_error: false,
    }
}

fn tool_summary(name: &str, args: Option<&serde_json::Value>, fallback: &str) -> String {
    let Some(v) = args else {
        return fallback.to_string();
    };
    let pick = |key: &str| v.get(key).and_then(|s| s.as_str()).map(str::to_owned);
    match name {
        "edit_file" | "write_file" | "read_file" => pick("path").unwrap_or_else(|| fallback.into()),
        "bash" => pick("command")
            .map(|c| truncate(c, 120))
            .unwrap_or_else(|| fallback.into()),
        "grep" => pick("pattern").unwrap_or_else(|| fallback.into()),
        "glob" => pick("pattern").unwrap_or_else(|| fallback.into()),
        "list_dir" => pick("path").unwrap_or_else(|| ".".into()),
        _ => fallback.to_string(),
    }
}

fn extract_diff(name: &str, args: &serde_json::Value) -> Option<DiffPayload> {
    let s = |key: &str| args.get(key).and_then(|v| v.as_str()).map(str::to_owned);
    match name {
        "edit_file" => Some(DiffPayload {
            path: s("path")?,
            old_text: s("old_string")?,
            new_text: s("new_string")?,
            is_creation: false,
        }),
        "write_file" => Some(DiffPayload {
            path: s("path")?,
            old_text: String::new(),
            new_text: s("content")?,
            is_creation: true,
        }),
        _ => None,
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
                        .selectable_label(self.left_mode == LeftPanelMode::Threads, "Threads")
                        .clicked()
                    {
                        self.left_mode = LeftPanelMode::Threads;
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
                    LeftPanelMode::Threads => self.render_thread_tree(ui),
                    LeftPanelMode::Resources => render_resource_list(ui, &self.resources),
                }
            });

        let input_enabled = matches!(self.conn_status, ConnectionStatus::Connected);
        let composing = self.composing_new || self.selected.is_none();
        // Reserve a String only when we need to interpolate a pod name
        // into the hint; the static-string fallback is the common case.
        let pod_hint: Option<String> = if composing && input_enabled {
            self.compose_pod_id.as_ref().map(|pid| {
                let display = self.pods.get(pid).map(|p| p.name.as_str()).unwrap_or(pid);
                format!("Describe a new thread in `{display}`")
            })
        } else {
            None
        };
        let hint: &str = match (composing, input_enabled, pod_hint.as_deref()) {
            (_, false, _) => "(connecting)",
            (true, true, Some(s)) => s,
            (true, true, None) => "Describe a new task",
            (false, true, _) => "Message this task",
        };

        let show_picker =
            (self.composing_new || self.selected.is_none()) && !self.backends.is_empty();
        if show_picker && let Some(pod_id) = self.compose_target_pod_id().map(str::to_owned) {
            self.ensure_pod_config(&pod_id);
        }
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

                        // Host env picker: dropdown over the target
                        // pod's `allow.host_env` entries. First row =
                        // "inherit pod default" (the server resolves
                        // `compose_host_env = None` to
                        // pod.thread_defaults.host_env, which the pod
                        // validator guarantees names one of these
                        // entries). Hidden entirely when the pod has
                        // no host envs (threads there run with shared
                        // MCPs only) or when the pod config hasn't
                        // landed yet — inheriting is a safe default.
                        let pod_config = self
                            .compose_target_pod_id()
                            .and_then(|id| self.pod_configs.get(id));
                        if let Some(pod_config) = pod_config
                            && !pod_config.allow.host_env.is_empty()
                        {
                            ui.separator();
                            ui.label(
                                RichText::new("host env")
                                    .small()
                                    .color(Color32::from_gray(180)),
                            );
                            let default_name = &pod_config.thread_defaults.host_env;
                            let inherit_label = format!("pod default ({default_name})");
                            let selected_label = match &self.compose_host_env {
                                None => inherit_label.clone(),
                                Some(name) => name.clone(),
                            };
                            ComboBox::from_id_salt("picker_host_env")
                                .selected_text(selected_label)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.compose_host_env,
                                        None,
                                        inherit_label,
                                    );
                                    ui.separator();
                                    for nh in &pod_config.allow.host_env {
                                        ui.selectable_value(
                                            &mut self.compose_host_env,
                                            Some(nh.name.clone()),
                                            &nh.name,
                                        );
                                    }
                                });
                        }
                    });
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if let Some(thread_id) = self.selected.clone() {
                        if ui.button("Cancel").clicked() {
                            self.send(ClientToServer::CancelThread {
                                thread_id: thread_id.clone(),
                            });
                        }
                        if ui.button("Archive").clicked() {
                            self.send(ClientToServer::ArchiveThread { thread_id });
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
            Some(thread_id) => match self.tasks.get_mut(&thread_id) {
                None => {
                    ui.label(
                        RichText::new(format!("task {thread_id} not found"))
                            .color(Color32::from_rgb(220, 120, 120)),
                    );
                }
                Some(view) => {
                    render_failure_banner(ui, view);
                    render_allowlist_chips(ui, &thread_id, view, &mut allowlist_revocations);
                    render_approval_banner(ui, &thread_id, view, &mut pending_decisions);
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
        for (thread_id, approval_id, decision, remember) in pending_decisions {
            self.send(ClientToServer::ApprovalDecision {
                thread_id,
                approval_id,
                decision,
                remember,
            });
        }
        for (thread_id, tool_name) in allowlist_revocations {
            self.send(ClientToServer::RemoveToolAllowlistEntry { thread_id, tool_name });
        }

        self.render_new_pod_modal(ctx);
        self.render_pod_editor_modal(ctx);
    }
}

impl ChatApp {
    /// Renders the left panel as a pod-grouped tree: each pod gets a header
    /// (with display name + thread count) and its threads nest underneath
    /// as selectable rows. Threads whose `pod_id` doesn't match any known
    /// pod get bucketed under a synthetic "(unknown pod)" group — happens
    /// in practice when a thread arrives via `ThreadCreated` before the
    /// `ListPods` round-trip completes.
    fn render_thread_tree(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("+ New thread").clicked() {
                self.selected = None;
                self.composing_new = true;
                self.compose_pod_id = None;
                self.input.clear();
            }
            if ui.button("+ New pod").clicked() {
                self.new_pod_modal = Some(NewPodModalState::new());
            }
        });
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
        let label = match self.pods.get(pod_id) {
            Some(summary) => format!("{}  ({})", summary.name, thread_ids.map(|t| t.len()).unwrap_or(0)),
            None => format!("{pod_id}  ({})", thread_ids.map(|t| t.len()).unwrap_or(0)),
        };
        let collapsed_id = format!("pod-section-{pod_id}");
        let default_open = !self.collapsed_pods.contains(pod_id);
        let is_default_pod = pod_id == self.server_default_pod_id;
        let mut archive_clicked = false;
        let mut archive_confirmed = false;
        let mut archive_disarm = false;
        let mut edit_config_clicked = false;
        let header = egui::CollapsingHeader::new(RichText::new(label).strong())
            .id_salt(&collapsed_id)
            .default_open(default_open)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .small_button(RichText::new("+ Thread in this pod").small())
                        .clicked()
                    {
                        self.selected = None;
                        self.composing_new = true;
                        self.compose_pod_id = Some(pod_id.to_string());
                        self.input.clear();
                    }
                    if ui
                        .small_button(RichText::new("Edit config").small())
                        .clicked()
                    {
                        edit_config_clicked = true;
                    }
                    if !is_default_pod {
                        let armed = self.archive_armed_pod.as_deref() == Some(pod_id);
                        if armed {
                            if ui
                                .small_button(
                                    RichText::new("Confirm archive")
                                        .small()
                                        .color(Color32::from_rgb(220, 90, 90)),
                                )
                                .clicked()
                            {
                                archive_confirmed = true;
                            }
                            if ui.small_button(RichText::new("Cancel").small()).clicked() {
                                archive_disarm = true;
                            }
                        } else if ui
                            .small_button(
                                RichText::new("Archive")
                                    .small()
                                    .color(Color32::from_gray(160)),
                            )
                            .clicked()
                        {
                            archive_clicked = true;
                        }
                    }
                });
                let Some(thread_ids) = thread_ids else {
                    ui.label(RichText::new("(no threads)").italics().color(Color32::from_gray(140)));
                    return;
                };
                for thread_id in thread_ids {
                    let Some(view) = self.tasks.get(thread_id) else {
                        continue;
                    };
                    let is_selected = self.selected.as_deref() == Some(thread_id.as_str());
                    let title = view
                        .summary
                        .title
                        .clone()
                        .unwrap_or_else(|| thread_id[..thread_id.len().min(14)].to_string());
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
                        self.select_task(thread_id.clone());
                    }
                }
            });
        // Track collapse state so it persists across renders.
        if header.fully_closed() {
            self.collapsed_pods.insert(pod_id.to_string());
        } else {
            self.collapsed_pods.remove(pod_id);
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
    }

    fn open_pod_editor(&mut self, pod_id: String) {
        self.send(ClientToServer::GetPod {
            correlation_id: None,
            pod_id: pod_id.clone(),
        });
        self.pod_editor_modal = Some(PodEditorModalState::new(pod_id));
    }

    /// Render the "+ New pod" modal. The user picks an id + display
    /// name; the new pod inherits the server default pod's config as
    /// a template. The pod editor (opened from the per-pod "Edit"
    /// button) is the place to change backends, shared MCP hosts, or
    /// host envs afterwards.
    fn render_new_pod_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.new_pod_modal.take() else {
            return;
        };
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
                        && !self.backends.is_empty();
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
                self.new_pod_modal = Some(modal);
                return;
            }
            if self.pods.contains_key(&pod_id) {
                modal.error = Some(format!("pod `{pod_id}` already exists"));
                self.new_pod_modal = Some(modal);
                return;
            }
            let config = self.fresh_pod_config(modal.name.trim().to_string());
            self.send(ClientToServer::CreatePod {
                correlation_id: None,
                pod_id,
                config,
            });
            // Modal closes; PodCreated event will populate self.pods on
            // the round-trip.
        } else if cancel_clicked || !open {
            // Modal closes.
        } else {
            self.new_pod_modal = Some(modal);
        }
    }

    fn render_pod_editor_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.pod_editor_modal.take() else {
            return;
        };

        // Snapshot the catalogs the form needs into owned data so the
        // inner closures don't have to borrow `self`. These are small
        // (single-digit lists in practice) so the clone is cheap.
        let backend_catalog: Vec<String> =
            self.backends.iter().map(|b| b.name.clone()).collect();
        let shared_mcp_catalog: Vec<String> = self
            .resources
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
        let host_env_providers: Vec<HostEnvProviderInfo> = self.host_env_providers.clone();

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
                    egui::TopBottomPanel::bottom("pod_editor_footer")
                        .show_inside(ui, |ui| {
                            ui.add_space(6.0);
                            if let Some(err) = &modal.error {
                                ui.colored_label(Color32::from_rgb(220, 80, 80), err);
                                ui.add_space(4.0);
                            }
                            ui.separator();
                            ui.horizontal(|ui| {
                                let save_enabled =
                                    modal.working.is_some() && dirty && !saving;
                                if ui
                                    .add_enabled(save_enabled, egui::Button::new("Save"))
                                    .clicked()
                                {
                                    save_clicked = true;
                                }
                                if ui
                                    .add_enabled(
                                        modal.working.is_some() && dirty && !saving,
                                        egui::Button::new("Revert"),
                                    )
                                    .clicked()
                                {
                                    revert_clicked = true;
                                }
                                if ui.button("Close").clicked() {
                                    cancel_clicked = true;
                                }
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if saving {
                                            ui.label(
                                                RichText::new("saving…")
                                                    .italics()
                                                    .color(Color32::from_gray(160)),
                                            );
                                        } else if dirty {
                                            ui.label(
                                                RichText::new("● unsaved changes")
                                                    .small()
                                                    .color(Color32::from_rgb(220, 170, 90)),
                                            );
                                        } else if modal.working.is_some() {
                                            ui.label(
                                                RichText::new("✓ saved")
                                                    .small()
                                                    .color(Color32::from_gray(140)),
                                            );
                                        }
                                    },
                                );
                            });
                        });
                    egui::TopBottomPanel::top("pod_editor_tabs")
                        .show_inside(ui, |ui| {
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
                                        RichText::new(tab.label())
                                            .color(Color32::from_gray(170))
                                    };
                                    if ui
                                        .selectable_label(active, label)
                                        .clicked()
                                        && !active
                                    {
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
                                        &host_env_providers,
                                        &mut sandbox_entry_open,
                                        &mut sandbox_entry_delete,
                                    );
                                }
                                PodEditorTab::Defaults => {
                                    render_pod_editor_defaults_tab(
                                        ui,
                                        working,
                                        &backend_catalog,
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
                &self.host_env_providers,
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
                    let dup = working.allow.host_env.iter().enumerate().any(|(i, e)| {
                        e.name == name && Some(i) != sub.index
                    });
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
                            working.thread_defaults.host_env = name;
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
            if working.thread_defaults.host_env == removed.name {
                working.thread_defaults.host_env = working
                    .allow
                    .host_env
                    .first()
                    .map(|nh| nh.name.clone())
                    .unwrap_or_default();
            }
        }
        if let Some(sub) = sandbox_entry_open {
            modal.sandbox_entry_editor = Some(sub);
        }

        // Tab switch happens after the inner closure so we can do the
        // raw->structured reparse without holding any UI borrows.
        if let Some(target) = switch_to {
            if modal.tab == PodEditorTab::RawToml && target != PodEditorTab::RawToml {
                if modal.raw_dirty {
                    match toml::from_str::<PodConfig>(&modal.raw_buffer) {
                        Ok(parsed) => {
                            modal.working = Some(parsed);
                            modal.raw_dirty = false;
                            modal.error = None;
                            modal.tab = target;
                        }
                        Err(e) => {
                            modal.error = Some(format!(
                                "raw TOML doesn't parse — fix it or click Revert: {e}"
                            ));
                        }
                    }
                } else {
                    modal.tab = target;
                }
            } else {
                if target == PodEditorTab::RawToml
                    && let Some(working) = &modal.working
                    && !modal.raw_dirty
                {
                    modal.raw_buffer = toml::to_string_pretty(working).unwrap_or_default();
                }
                modal.tab = target;
                modal.error = None;
            }
        }

        if save_clicked
            && let Some(working) = &modal.working
        {
            let toml_text = if modal.tab == PodEditorTab::RawToml && modal.raw_dirty {
                modal.raw_buffer.clone()
            } else {
                match toml::to_string_pretty(working) {
                    Ok(s) => s,
                    Err(e) => {
                        modal.error = Some(format!("encode pod.toml: {e}"));
                        self.pod_editor_modal = Some(modal);
                        return;
                    }
                }
            };
            let correlation = self.next_correlation_id();
            modal.pending_correlation = Some(correlation.clone());
            modal.error = None;
            self.send(ClientToServer::UpdatePodConfig {
                correlation_id: Some(correlation),
                pod_id: modal.pod_id.clone(),
                toml_text,
            });
            self.pod_editor_modal = Some(modal);
        } else if revert_clicked {
            if let Some(baseline) = &modal.server_baseline {
                modal.working = Some(baseline.clone());
                modal.raw_buffer = toml::to_string_pretty(baseline).unwrap_or_default();
                modal.raw_dirty = false;
                modal.error = None;
                modal.pending_correlation = None;
            } else {
                self.send(ClientToServer::GetPod {
                    correlation_id: None,
                    pod_id: modal.pod_id.clone(),
                });
                modal.working = None;
                modal.error = None;
                modal.pending_correlation = None;
            }
            self.pod_editor_modal = Some(modal);
        } else if cancel_clicked || !open {
            // Modal closes (drop modal).
        } else {
            self.pod_editor_modal = Some(modal);
        }
    }

    /// Build a sensible default `PodConfig` for a fresh pod created from
    /// the webui. Clones `default_pod_template` (the server's known-good
    /// default pod config) when available so the new pod inherits a
    /// working sandbox + shared-MCP setup. Falls back to a minimal stub
    /// keyed off the catalog summary when no template has arrived yet
    /// (rare — only on first connect before the GetPod round-trip).
    /// `created_at` is left empty; the server stamps it on CreatePod.
    fn fresh_pod_config(&self, name: String) -> PodConfig {
        if let Some(template) = &self.default_pod_template {
            let mut cfg = template.clone();
            cfg.name = name;
            cfg.description = None;
            cfg.created_at = String::new();
            return cfg;
        }
        let backend_names: Vec<String> = self.backends.iter().map(|b| b.name.clone()).collect();
        let default_backend = if !self.default_backend.is_empty() {
            self.default_backend.clone()
        } else {
            backend_names.first().cloned().unwrap_or_default()
        };
        PodConfig {
            name,
            description: None,
            created_at: String::new(),
            allow: PodAllow {
                backends: backend_names,
                mcp_hosts: Vec::new(),
                host_env: Vec::<NamedHostEnv>::new(),
            },
            thread_defaults: ThreadDefaults {
                backend: default_backend,
                model: String::new(),
                system_prompt_file: "system_prompt.md".into(),
                max_tokens: 16384,
                max_turns: 30,
                approval_policy: ApprovalPolicy::AutoApproveAll,
                host_env: String::new(),
                mcp_hosts: Vec::new(),
            },
            limits: PodLimits::default(),
        }
    }
}

/// Persistent banner for a task that's entered the Failed state. Survives resnapshot
/// because `failure` is captured from the snapshot itself rather than derived from
/// the per-event items list.
fn render_resource_list(ui: &mut egui::Ui, resources: &HashMap<String, ResourceSnapshot>) {
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
        ui.label(
            RichText::new(sub)
                .color(Color32::from_gray(150))
                .small(),
        );
    }
    ui.add_space(4.0);
}

fn spec_label(spec: &HostEnvSpec) -> String {
    match spec {
        HostEnvSpec::Container { image, .. } => format!("container: {image}"),
        HostEnvSpec::Landlock { allowed_paths, .. } => {
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
    if view.summary.state != ThreadStateLabel::Failed {
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
    thread_id: &str,
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
                                    thread_id.to_string(),
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
                                    thread_id.to_string(),
                                    approval.approval_id.clone(),
                                    ApprovalChoice::Approve,
                                    true,
                                ));
                            }
                            if ui.button("Reject").clicked() {
                                approval.submitted = true;
                                pending_decisions.push((
                                    thread_id.to_string(),
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
    thread_id: &str,
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
                revocations.push((thread_id.to_string(), tool_name.clone()));
            }
            ui.add_space(2.0);
        }
    });
    ui.add_space(4.0);
}

// ====================================================================
// Conversation renderer — event-log layout
//
// Each item draws as a full-width row with a 3px role-colored gutter on
// the left edge. User messages get a faint background tint to mark
// them as section breaks (they're rare relative to model output).
// Agent text and reasoning have transparent backgrounds — the gutter
// alone carries the role signal so the bulk of the conversation reads
// as a quiet stream of model output, not a chat log.
//
// Tool calls render as collapsible rows whose header is `name summary
// [state]`. `edit_file` / `write_file` calls expand by default and
// show a unified diff (built into the item's `diff` payload at
// build_tool_call_item time); other tools collapse by default with
// the full pretty-printed JSON args + truncated result available
// inside.
// ====================================================================

const GUTTER_WIDTH: f32 = 3.0;

const COLOR_USER: Color32 = Color32::from_rgb(120, 180, 240);
const COLOR_AGENT: Color32 = Color32::from_rgb(120, 200, 140);
const COLOR_REASONING: Color32 = Color32::from_rgb(170, 150, 200);
const COLOR_TOOL: Color32 = Color32::from_rgb(220, 180, 100);
const COLOR_ERROR: Color32 = Color32::from_rgb(220, 120, 120);
const COLOR_NEUTRAL: Color32 = Color32::from_gray(140);

fn item_palette(item: &DisplayItem) -> (Color32, Color32) {
    // (gutter color, frame fill)
    match item {
        DisplayItem::User { .. } => (
            COLOR_USER,
            Color32::from_rgba_unmultiplied(120, 180, 240, 18),
        ),
        DisplayItem::AssistantText { .. } => (COLOR_AGENT, Color32::TRANSPARENT),
        DisplayItem::Reasoning { .. } => (COLOR_REASONING, Color32::TRANSPARENT),
        DisplayItem::ToolCall { is_error: true, .. } => (COLOR_ERROR, Color32::TRANSPARENT),
        DisplayItem::ToolCall { .. } => (COLOR_TOOL, Color32::TRANSPARENT),
        DisplayItem::SystemNote { is_error: true, .. } => (
            COLOR_ERROR,
            Color32::from_rgba_unmultiplied(220, 120, 120, 18),
        ),
        DisplayItem::SystemNote { .. } => (COLOR_NEUTRAL, Color32::TRANSPARENT),
    }
}

fn render_item(ui: &mut egui::Ui, item: &DisplayItem) {
    let (gutter_color, fill) = item_palette(item);
    let frame = egui::Frame::default()
        .fill(fill)
        .inner_margin(egui::Margin {
            left: 12,
            right: 8,
            top: 4,
            bottom: 4,
        });
    let resp = frame.show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        match item {
            DisplayItem::User { text } => render_user(ui, text),
            DisplayItem::AssistantText { text } => render_assistant_text(ui, text),
            DisplayItem::Reasoning { text } => render_reasoning(ui, text),
            DisplayItem::ToolCall {
                tool_use_id,
                name,
                summary,
                args_pretty,
                diff,
                result,
                is_error,
            } => render_tool_call(
                ui,
                tool_use_id,
                name,
                summary,
                args_pretty.as_deref(),
                diff.as_ref(),
                result.as_deref(),
                *is_error,
            ),
            DisplayItem::SystemNote { text, is_error } => render_system_note(ui, text, *is_error),
        }
    });
    // Paint the gutter into the reserved 3px on the left side of the
    // frame's inner margin. Done after .show() returns so the painted
    // rect lands on top of the (subtle) frame fill — fine because the
    // gutter sits inside the inner margin where there's no content.
    let r = resp.response.rect;
    let gutter_rect = egui::Rect::from_min_max(
        r.min,
        egui::pos2(r.min.x + GUTTER_WIDTH, r.max.y),
    );
    ui.painter().rect_filled(gutter_rect, 0.0, gutter_color);
}

fn render_user(ui: &mut egui::Ui, text: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.label(
            RichText::new("USER")
                .color(COLOR_USER)
                .strong()
                .small(),
        );
        ui.add_space(8.0);
        ui.label(text);
    });
}

fn render_assistant_text(ui: &mut egui::Ui, text: &str) {
    // No role label — the gutter color carries it. This is the bulk
    // of the conversation; we want it visually quiet.
    ui.label(text);
}

fn render_reasoning(ui: &mut egui::Ui, text: &str) {
    let id = ui.make_persistent_id(("reasoning", text.as_ptr() as usize));
    let preview = text.lines().next().unwrap_or("").trim();
    let header = if preview.is_empty() {
        "(reasoning)".to_string()
    } else if preview.chars().count() > 80 {
        let mut h: String = preview.chars().take(80).collect();
        h.push('…');
        h
    } else {
        preview.to_string()
    };
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
        .show_header(ui, |ui| {
            ui.label(
                RichText::new("REASONING")
                    .color(COLOR_REASONING)
                    .strong()
                    .small(),
            );
            ui.add_space(8.0);
            ui.label(
                RichText::new(header)
                    .color(Color32::from_gray(140))
                    .italics(),
            );
        })
        .body(|ui| {
            ui.label(RichText::new(text).color(Color32::from_gray(170)).italics());
        });
}

fn render_system_note(ui: &mut egui::Ui, text: &str, is_error: bool) {
    let color = if is_error { COLOR_ERROR } else { COLOR_NEUTRAL };
    ui.label(RichText::new(text).color(color).italics());
}

fn render_tool_call(
    ui: &mut egui::Ui,
    tool_use_id: &str,
    name: &str,
    summary: &str,
    args_pretty: Option<&str>,
    diff: Option<&DiffPayload>,
    result: Option<&str>,
    is_error: bool,
) {
    let id = ui.make_persistent_id(("tool", tool_use_id));
    // Default-open the diff variants — the diff itself is the
    // interesting content, and forcing the user to click into every
    // edit ruins the scanability of an ongoing edit session.
    let default_open = diff.is_some();
    let (chip_text, chip_color) = match (result, is_error) {
        (None, _) => ("running", Color32::from_rgb(200, 180, 60)),
        (Some(_), true) => ("error", COLOR_ERROR),
        (Some(_), false) => ("ok", Color32::from_rgb(140, 200, 140)),
    };
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, default_open)
        .show_header(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(name)
                        .color(COLOR_TOOL)
                        .strong()
                        .monospace(),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new(summary)
                        .color(Color32::from_gray(180))
                        .monospace(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(chip_text)
                            .color(chip_color)
                            .small()
                            .strong(),
                    );
                });
            });
        })
        .body(|ui| {
            if let Some(diff) = diff {
                render_diff(ui, diff);
            } else if let Some(args) = args_pretty {
                ui.label(
                    RichText::new(args)
                        .color(Color32::from_gray(170))
                        .monospace()
                        .small(),
                );
            }
            if let Some(result) = result {
                ui.add_space(6.0);
                ui.label(
                    RichText::new("result")
                        .color(Color32::from_gray(140))
                        .small()
                        .strong(),
                );
                let color = if is_error {
                    Color32::from_rgb(220, 140, 140)
                } else {
                    Color32::from_gray(180)
                };
                ui.label(RichText::new(result).color(color).monospace().small());
            }
        });
}

fn render_diff(ui: &mut egui::Ui, diff: &DiffPayload) {
    use similar::{ChangeTag, TextDiff};
    let header = if diff.is_creation {
        format!("(new) {}", diff.path)
    } else {
        diff.path.clone()
    };
    ui.label(
        RichText::new(header)
            .color(Color32::from_gray(180))
            .monospace()
            .small()
            .strong(),
    );
    let text_diff = TextDiff::from_lines(&diff.old_text, &diff.new_text);
    let fg_eq = Color32::from_gray(150);
    let fg_del = Color32::from_rgb(230, 130, 130);
    let fg_add = Color32::from_rgb(150, 220, 170);
    egui::Frame::default()
        .fill(Color32::from_gray(22))
        .inner_margin(egui::Margin {
            left: 6,
            right: 6,
            top: 4,
            bottom: 4,
        })
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.style_mut().spacing.item_spacing.y = 0.0;
            for change in text_diff.iter_all_changes() {
                let (prefix, color) = match change.tag() {
                    ChangeTag::Equal => (' ', fg_eq),
                    ChangeTag::Delete => ('-', fg_del),
                    ChangeTag::Insert => ('+', fg_add),
                };
                let raw = change.value();
                let trimmed = raw.strip_suffix('\n').unwrap_or(raw);
                ui.label(
                    RichText::new(format!("{prefix}{trimmed}"))
                        .color(color)
                        .monospace()
                        .small(),
                );
            }
        });
}

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

fn render_pod_editor_allow_tab(
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
                    ui.label(
                        RichText::new(spec_label(&entry.spec))
                            .color(Color32::from_gray(170)),
                    );
                    ui.horizontal(|ui| {
                        if ui.small_button("Edit").clicked() {
                            *sandbox_open = Some(
                                SandboxEntryEditorState::new_for_index(i, entry.clone()),
                            );
                        }
                        if ui.small_button(RichText::new("Delete").color(
                            Color32::from_rgb(220, 100, 100),
                        )).clicked() {
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

fn render_pod_editor_defaults_tab(
    ui: &mut egui::Ui,
    working: &mut PodConfig,
    backend_catalog: &[String],
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
                                RichText::new(name)
                                    .color(Color32::from_rgb(220, 170, 90)),
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
            ui.add(
                TextEdit::singleline(&mut working.thread_defaults.model)
                    .hint_text("e.g. claude-sonnet-4-6")
                    .desired_width(f32::INFINITY),
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

            ui.label("approval policy");
            ComboBox::from_id_salt("pod_editor_defaults_approval")
                .selected_text(approval_policy_label(working.thread_defaults.approval_policy))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut working.thread_defaults.approval_policy,
                        ApprovalPolicy::AutoApproveAll,
                        approval_policy_label(ApprovalPolicy::AutoApproveAll),
                    );
                    ui.selectable_value(
                        &mut working.thread_defaults.approval_policy,
                        ApprovalPolicy::PromptDestructive,
                        approval_policy_label(ApprovalPolicy::PromptDestructive),
                    );
                });
            ui.end_row();

            ui.label("host env");
            // Pods with zero allow.host_env entries don't offer a host
            // env to their threads at all — the default is fixed at
            // empty and there's nothing to choose. Pods with entries
            // must pick a default from them (the server's validator
            // rejects mismatches). There is no "(none)" escape hatch:
            // a thread with no host env binding is a shape for pods
            // that declare zero entries, not a fallback we let other
            // pods silently land on.
            let sb_in_allow = working.thread_defaults.host_env.is_empty()
                || working
                    .allow
                    .host_env
                    .iter()
                    .any(|s| s.name == working.thread_defaults.host_env);
            if working.allow.host_env.is_empty() {
                // Keep the value forced to empty — switching to a named
                // entry from here is meaningless until the allow list
                // gains one on the Allow tab.
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
                ComboBox::from_id_salt("pod_editor_defaults_sandbox")
                    .selected_text(if working.thread_defaults.host_env.is_empty() {
                        "(pick one)".to_string()
                    } else {
                        working.thread_defaults.host_env.clone()
                    })
                    .show_ui(ui, |ui| {
                        for entry in &working.allow.host_env {
                            ui.selectable_value(
                                &mut working.thread_defaults.host_env,
                                entry.name.clone(),
                                &entry.name,
                            );
                        }
                    });
            }
            ui.end_row();
            if !sb_in_allow {
                ui.label("");
                ui.label(
                    RichText::new(format!(
                        "`{}` is not in allow.host_env — server will reject on save",
                        working.thread_defaults.host_env
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

fn render_pod_editor_limits_tab(ui: &mut egui::Ui, working: &mut PodConfig) {
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

fn render_pod_editor_raw_tab(ui: &mut egui::Ui, raw: &mut String, dirty: &mut bool) {
    ui.add_space(4.0);
    hint(
        ui,
        "Raw pod.toml. Edits here override the structured tabs on save. \
         Switching back to a structured tab tries to parse this text first; \
         a parse error keeps you here so the edit isn't lost.",
    );
    ui.add_space(4.0);
    let resp = ui.add_sized(
        [ui.available_width(), ui.available_height().max(180.0)],
        TextEdit::multiline(raw).code_editor(),
    );
    if resp.changed() {
        *dirty = true;
    }
}

fn render_sandbox_entry_modal(
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
            egui::TopBottomPanel::bottom("sandbox_entry_footer").show_inside(ui, |ui| {
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
                                                matches!(sub.entry.spec, HostEnvSpec::Landlock { .. }),
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
                                                matches!(sub.entry.spec, HostEnvSpec::Container { .. }),
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
        if ui.radio_value(&mut variant, 1, "isolated (no network)").clicked() {
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

fn section_heading(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).strong().size(14.0));
    ui.add_space(2.0);
}

fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(
        RichText::new(text)
            .small()
            .color(Color32::from_gray(160)),
    );
}

fn approval_policy_label(p: ApprovalPolicy) -> &'static str {
    match p {
        ApprovalPolicy::AutoApproveAll => "auto — approve all tool calls",
        ApprovalPolicy::PromptDestructive => "prompt — ask before non-readonly tool calls",
    }
}

fn spec_type_label(spec: &HostEnvSpec) -> &'static str {
    match spec {
        HostEnvSpec::Landlock { .. } => "landlock",
        HostEnvSpec::Container { .. } => "container",
    }
}
