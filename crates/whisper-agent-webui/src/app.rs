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

mod chat_render;
mod editor_render;

use self::chat_render::render_item;
use self::editor_render::{
    approval_policy_label, behavior_summary_from_snapshot, hint, render_behavior_editor_prompt_tab,
    render_behavior_editor_raw_tab, render_behavior_editor_retention_tab,
    render_behavior_editor_thread_tab, render_behavior_editor_trigger_tab,
    render_pod_editor_allow_tab, render_pod_editor_defaults_tab, render_pod_editor_limits_tab,
    render_pod_editor_raw_tab, render_sandbox_entry_modal, section_heading,
};

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use egui::{Color32, ComboBox, Grid, RichText, ScrollArea, TextEdit};
use egui_commonmark::CommonMarkCache;
use whisper_agent_protocol::sandbox::NetworkPolicy;
use whisper_agent_protocol::{
    ApprovalChoice, ApprovalPolicy, BackendSummary, BehaviorConfig, BehaviorOrigin,
    BehaviorSnapshot as BehaviorSnapshotProto, BehaviorSummary, BehaviorThreadOverride,
    ClientToServer, ContentBlock, Conversation, HostEnvBinding, HostEnvProviderInfo, HostEnvSpec,
    Message, ModelSummary, NamedHostEnv, PodAllow, PodConfig, PodLimits, PodSummary,
    ResourceSnapshot, ResourceStateLabel, RetentionPolicy, Role, ServerToClient, ThreadBindings,
    ThreadBindingsRequest, ThreadConfigOverride, ThreadDefaults, ThreadStateLabel, ThreadSummary,
    ToolResultContent, TriggerSpec, Usage,
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

/// How many thread rows to show under each sidebar subsection (interactive
/// threads and each behavior bucket) before the "Show N more" affordance
/// collapses the tail. Keeps long-running behavior pods from drowning the
/// sidebar in repeated rows.
const THREAD_ROW_PREVIEW_COUNT: usize = 3;

// Shared palette for the sidebar. Named so the hierarchy is explicit and
// so the panel's tone doesn't drift as new rows/captions are added.
const SIDEBAR_SUBSECTION_COLOR: Color32 = Color32::from_gray(200);
const SIDEBAR_BODY_COLOR: Color32 = Color32::from_gray(210);
const SIDEBAR_MUTED_COLOR: Color32 = Color32::from_gray(150);
const SIDEBAR_DIM_COLOR: Color32 = Color32::from_gray(130);
const SIDEBAR_DANGER_COLOR: Color32 = Color32::from_rgb(220, 90, 90);
const SIDEBAR_WARNING_COLOR: Color32 = Color32::from_rgb(220, 170, 90);
const SIDEBAR_ERROR_TEXT_COLOR: Color32 = Color32::from_rgb(220, 120, 120);

/// Small-bold subsection header under a pod section ("Interactive",
/// "Behaviors", "Deleted behaviors").
fn sidebar_subsection_header(ui: &mut egui::Ui, text: impl Into<String>) {
    ui.label(
        RichText::new(text.into())
            .small()
            .strong()
            .color(SIDEBAR_SUBSECTION_COLOR),
    );
}

/// Compact sidebar button with uniform padding, optionally disabled.
/// Wraps `add_enabled` so the enabled-vs-disabled variants share the
/// same `Button::small()` frame.
fn sidebar_button(ui: &mut egui::Ui, text: RichText, enabled: bool) -> egui::Response {
    ui.add_enabled(enabled, egui::Button::new(text.small()).small())
}

/// Full-width, left-aligned selectable row for the sidebar thread list.
/// `ui.add_sized(...)` would wrap the widget in
/// `Layout::centered_and_justified`, which centers the button's text;
/// cross-justifying a top-down-left layout keeps the rounded highlight
/// full-width while the button reads `ui.layout()` to left-align its
/// label.
fn add_sidebar_thread_row(ui: &mut egui::Ui, selected: bool, text: RichText) -> egui::Response {
    let layout = egui::Layout::top_down(egui::Align::LEFT).with_cross_justify(true);
    ui.allocate_ui_with_layout(egui::Vec2::new(ui.available_width(), 0.0), layout, |ui| {
        ui.add(egui::Button::selectable(selected, text.small()))
    })
    .inner
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
    },
    /// Tool response — rendered as its own chat row at the position
    /// where it landed in the conversation. Always default-
    /// collapsed; click to expand. Matches the "quiet log" treatment
    /// of tool calls and reasoning — the chat stream stays scannable,
    /// and the operator opens individual rows to see full content.
    ToolResult {
        tool_use_id: String,
        /// Best-effort tool name, looked up from the matching
        /// `DisplayItem::ToolCall` when available. Empty string if
        /// the call isn't in the current view (orphan result).
        name: String,
        text: String,
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
    /// Extra snapshot-carried fields the context inspector renders.
    /// Populated on `ThreadSnapshot`; kept in sync via the per-field
    /// update events (`ThreadBindingsChanged`, etc). Small enough to
    /// hold per-view rather than shipping an extra request when the
    /// inspector opens.
    inspector: ThreadInspector,
}

/// Everything the thread-context inspector surfaces that isn't
/// already on `TaskView` (backend, model, failure, allowlist, usage,
/// summary). Split into its own struct so `TaskView` stays readable
/// at a glance — the inspector fields are a handful of seldom-changing
/// reference values, not per-turn state.
#[derive(Clone, Default)]
struct ThreadInspector {
    system_prompt: String,
    max_tokens: u32,
    max_turns: u32,
    approval_policy: Option<ApprovalPolicy>,
    bindings: ThreadBindings,
    origin: Option<BehaviorOrigin>,
    created_at: String,
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
        let created_at = summary.created_at.clone();
        let origin = summary.origin.clone();
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
            inspector: ThreadInspector {
                system_prompt: String::new(),
                max_tokens: 0,
                max_turns: 0,
                approval_policy: None,
                bindings: ThreadBindings::default(),
                origin,
                created_at,
            },
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
    /// Pod ids whose behavior catalog has already been requested via
    /// `ListBehaviors`. `PodSnapshot.behaviors` also populates the
    /// cache (when the pod editor runs `GetPod`), but bare
    /// `PodList` doesn't carry behaviors — so on startup the pod
    /// section shows empty until we fire one `ListBehaviors` per
    /// known pod. Guarded by this set so we don't re-request on
    /// every `PodList` refresh.
    behaviors_requested: HashSet<String>,
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
    /// Pod ids whose interactive-threads subsection is expanded past the
    /// default preview count. Absence = show the first
    /// `THREAD_ROW_PREVIEW_COUNT` rows with a "Show N more" affordance.
    expanded_interactive_pods: HashSet<String>,
    /// `(pod_id, behavior_id)` pairs whose thread list is expanded past
    /// the default preview count. Absence = preview mode.
    expanded_behavior_threads: HashSet<(String, String)>,
    /// Modal state for the "+ New pod" form. `None` = closed.
    new_pod_modal: Option<NewPodModalState>,
    /// Pod id whose "Archive" button has been clicked once and is waiting
    /// for confirmation. Cleared by clicking elsewhere or confirming.
    /// At most one pod can be armed at a time — clicking a different pod's
    /// Archive button replaces it.
    archive_armed_pod: Option<String>,
    /// Modal state for the per-pod raw-TOML config editor. `None` = closed.
    pod_editor_modal: Option<PodEditorModalState>,
    /// Per-pod cache of behavior summaries. Populated from PodSnapshot
    /// (which inlines a behavior list) and refreshed by
    /// BehaviorCreated / Updated / Deleted / StateChanged events.
    /// Lives here rather than under `pods` because the PodSummary wire
    /// type deliberately stays lightweight.
    behaviors_by_pod: HashMap<String, Vec<BehaviorSummary>>,
    /// Per-(pod,behavior) id whose delete button has been clicked once
    /// and is waiting for confirmation. Two-click UX mirrors the pod
    /// archive button.
    delete_armed_behavior: Option<(String, String)>,
    /// Modal state for the per-behavior editor. `None` = closed.
    behavior_editor_modal: Option<BehaviorEditorModalState>,
    /// Modal state for the "+ New behavior" form. `None` = closed.
    new_behavior_modal: Option<NewBehaviorModalState>,
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

    /// Shared parse-result cache for the chat-log markdown renderer.
    /// CommonMarkViewer hashes the input text per call and reuses the
    /// parsed AST, so keeping one cache on the app avoids re-parsing
    /// the entire scrollback every frame.
    md_cache: CommonMarkCache,
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

/// State for the per-behavior editor modal. Edits two things in
/// parallel: the `behavior.toml` (via structured tabs or a raw tab)
/// and the sibling `prompt.md` (via its own Prompt tab). A save ships
/// both together through `UpdateBehavior`; state.json is
/// scheduler-maintained and not touched here.
///
/// Lifecycle: open → issue `GetBehavior` → on snapshot, populate
/// `working_config` + `working_prompt` + baselines → user edits →
/// Save serializes working (or raw_buffer if raw is dirty) + prompt,
/// ships `UpdateBehavior` with a correlation_id → `BehaviorUpdated`
/// or `Error` resolves the correlation.
struct BehaviorEditorModalState {
    pod_id: String,
    behavior_id: String,
    /// Parsed working copy. `None` until the snapshot lands.
    working_config: Option<BehaviorConfig>,
    /// Working prompt text. Empty until snapshot lands or when the
    /// behavior has no prompt.
    working_prompt: String,
    /// Server-known baselines for Revert + dirty detection.
    baseline_config: Option<BehaviorConfig>,
    baseline_prompt: String,
    /// Raw TOML tab backing buffer. Regenerated from working_config on
    /// tab entry unless raw_dirty says there's an unsaved raw edit.
    raw_buffer: String,
    raw_dirty: bool,
    tab: BehaviorEditorTab,
    error: Option<String>,
    pending_correlation: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BehaviorEditorTab {
    Trigger,
    Thread,
    Retention,
    Prompt,
    RawToml,
}

impl BehaviorEditorTab {
    fn label(self) -> &'static str {
        match self {
            BehaviorEditorTab::Trigger => "Trigger",
            BehaviorEditorTab::Thread => "Thread",
            BehaviorEditorTab::Retention => "Retention",
            BehaviorEditorTab::Prompt => "Prompt",
            BehaviorEditorTab::RawToml => "Raw TOML",
        }
    }
}

impl BehaviorEditorModalState {
    fn new(pod_id: String, behavior_id: String) -> Self {
        Self {
            pod_id,
            behavior_id,
            working_config: None,
            working_prompt: String::new(),
            baseline_config: None,
            baseline_prompt: String::new(),
            raw_buffer: String::new(),
            raw_dirty: false,
            tab: BehaviorEditorTab::Trigger,
            error: None,
            pending_correlation: None,
        }
    }

    /// True when anything — structured config, prompt, or raw buffer —
    /// has diverged from the server-known baseline.
    fn is_dirty(&self) -> bool {
        let config_dirty = match (&self.working_config, &self.baseline_config) {
            (Some(w), Some(b)) => w != b || self.raw_dirty,
            _ => false,
        };
        let prompt_dirty =
            self.baseline_config.is_some() && self.working_prompt != self.baseline_prompt;
        config_dirty || prompt_dirty
    }
}

/// State for the "+ New behavior" dialog. Two fields: the
/// directory-friendly `behavior_id` (immutable on disk) and the
/// display `name` (free text). On Save we send a `CreateBehavior`
/// with a minimal Manual-triggered stub; the editor modal opens
/// automatically on the `BehaviorCreated` round-trip so the user can
/// fill in trigger / thread / prompt details immediately.
struct NewBehaviorModalState {
    pod_id: String,
    behavior_id: String,
    name: String,
    error: Option<String>,
    /// In-flight create correlation_id. The editor opens when we
    /// receive the matching `BehaviorCreated`.
    pending_correlation: Option<String>,
}

impl NewBehaviorModalState {
    fn new(pod_id: String) -> Self {
        Self {
            pod_id,
            behavior_id: String::new(),
            name: String::new(),
            error: None,
            pending_correlation: None,
        }
    }
}

/// Deferred action produced by the behavior-row render closure.
/// Collected during rendering and replayed by the enclosing pod
/// section once the render closure returns — keeps mutating state
/// (opening modals, sending wire messages) out of the nested borrow.
enum BehaviorRowAction {
    New,
    Edit {
        pod_id: String,
        behavior_id: String,
    },
    Run {
        pod_id: String,
        behavior_id: String,
    },
    ArmDelete {
        pod_id: String,
        behavior_id: String,
    },
    DisarmDelete,
    ConfirmDelete {
        pod_id: String,
        behavior_id: String,
    },
    SetEnabled {
        pod_id: String,
        behavior_id: String,
        enabled: bool,
    },
    /// User clicked a thread row nested under a behavior bucket.
    SelectThread {
        thread_id: String,
    },
    /// User clicked the "Show N more" / "Show less" affordance under a
    /// behavior bucket. Toggles membership in
    /// `AppState::expanded_behavior_threads`.
    ToggleExpandThreads {
        pod_id: String,
        behavior_id: String,
    },
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

/// Same rules as `validate_pod_id_client` — behavior ids become
/// directory names under `<pod>/behaviors/`, so the constraint set is
/// identical.
fn validate_behavior_id_client(id: &str) -> Result<(), &'static str> {
    if id.is_empty() {
        return Err("behavior_id is empty");
    }
    if id.starts_with('.') {
        return Err("behavior_id may not start with '.'");
    }
    if id.contains('/') || id.contains('\\') || id.contains('\0') || id == ".." {
        return Err("behavior_id contains illegal characters");
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
            behaviors_requested: HashSet::new(),
            host_env_providers: Vec::new(),
            host_env_providers_requested: false,
            collapsed_pods: HashSet::new(),
            expanded_interactive_pods: HashSet::new(),
            expanded_behavior_threads: HashSet::new(),
            new_pod_modal: None,
            archive_armed_pod: None,
            pod_editor_modal: None,
            behaviors_by_pod: HashMap::new(),
            delete_armed_behavior: None,
            behavior_editor_modal: None,
            new_behavior_modal: None,
            next_correlation_seq: 0,
            compose_pod_id: None,
            server_default_pod_id: String::new(),
            default_pod_template: None,
            pod_configs: HashMap::new(),
            pod_configs_requested: HashSet::new(),
            left_mode: LeftPanelMode::default(),
            md_cache: CommonMarkCache::default(),
        }
    }

    /// Pod the compose form currently targets. `compose_pod_id` is
    /// `None` when the user clicked the global "+ New thread" button —
    /// we fall back to the server's default pod. Returns `None` when
    /// neither is known yet (brand-new connection before PodList).
    fn compose_target_pod_id(&self) -> Option<&str> {
        self.compose_pod_id.as_deref().or_else(|| {
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
        if self.pod_configs.contains_key(pod_id) || self.pod_configs_requested.contains(pod_id) {
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
            ServerToClient::ThreadSnapshot {
                thread_id,
                snapshot,
            } => {
                let items = conversation_to_items(&snapshot.conversation);
                let backend = snapshot.bindings.backend.clone();
                let model = snapshot.config.model.clone();
                let failure = snapshot.failure.clone();
                let allowlist = snapshot.tool_allowlist.clone();
                let inspector = ThreadInspector {
                    system_prompt: snapshot.config.system_prompt.clone(),
                    max_tokens: snapshot.config.max_tokens,
                    max_turns: snapshot.config.max_turns,
                    approval_policy: Some(snapshot.config.approval_policy),
                    bindings: snapshot.bindings.clone(),
                    origin: snapshot.origin.clone(),
                    created_at: snapshot.created_at.clone(),
                };
                let view = self
                    .tasks
                    .entry(thread_id.clone())
                    .or_insert_with(|| TaskView::new(snapshot_summary(&snapshot)));
                view.summary.state = snapshot.state;
                view.summary.title = snapshot.title;
                view.summary.origin = snapshot.origin.clone();
                view.total_usage = snapshot.total_usage;
                view.items = items;
                view.subscribed = true;
                view.backend = backend;
                view.model = model;
                view.failure = failure;
                view.tool_allowlist = allowlist;
                view.inspector = inspector;
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
                thread_id,
                bindings,
                ..
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.backend = bindings.backend.clone();
                    view.inspector.bindings = bindings;
                }
            }
            ServerToClient::ThreadCompacted {
                thread_id,
                new_thread_id,
                ..
            } => {
                // The continuation thread already arrived via its own
                // `ThreadCreated` event with `continued_from = None`
                // in the summary — the linkage stamp happens on the
                // server after `create_task` returns. Patch it in
                // now so the list tier reflects the ancestor.
                if let Some(view) = self.tasks.get_mut(&new_thread_id) {
                    view.summary.continued_from = Some(thread_id);
                }
            }
            ServerToClient::ThreadUserMessage { thread_id, text } => {
                // User-role message appended to the conversation.
                // Fires for both user-typed follow-ups (the webui used
                // to add these optimistically; that's now removed so
                // the server echo is the single source of truth) and
                // server-injected text (compaction continuation seeds,
                // behavior-trigger prompts). Async dispatch callbacks
                // travel a distinct event (`ThreadToolResultMessage`).
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.items.push(DisplayItem::User { text });
                }
            }
            ServerToClient::ThreadToolResultMessage { thread_id, text } => {
                // Tool-output text appended to the conversation —
                // typically an async `dispatch_thread` callback
                // (XML envelope carrying the child's final result).
                // Pushed as its own `DisplayItem::ToolResult` row at
                // the chronological position where it landed; the
                // default-open check is based on proximity to the
                // matching tool call in the items list so async
                // callbacks (separated from their call by an
                // assistant turn) arrive expanded while immediately-
                // -following results stay collapsed.
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    push_tool_result_from_text(&mut view.items, &text);
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
                    push_tool_result(&mut view.items, tool_use_id, result_preview, is_error);
                }
            }
            ServerToClient::ThreadAssistantEnd {
                thread_id, usage, ..
            } => {
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
                // If the pod editor minted this correlation, surface the
                // error inline in the modal instead of as a global banner
                // — failed validation should leave the user's edits
                // intact.
                if let Some(modal) = self.pod_editor_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // Behavior editor pending save.
                if let Some(modal) = self.behavior_editor_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // "+ New behavior" pending create.
                if let Some(modal) = self.new_behavior_modal.as_mut()
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
                // PodList summaries don't carry behavior catalogs —
                // fire one ListBehaviors per pod we haven't seen yet
                // so the pod sections render pre-existing behaviors on
                // first connect. `behaviors_requested` dedups so a
                // PodList refresh doesn't re-request.
                let pod_ids: Vec<String> = self.pods.keys().cloned().collect();
                for pid in pod_ids {
                    self.ensure_behaviors_fetched(&pid);
                }
            }
            ServerToClient::PodCreated { pod, .. } => {
                let pod_id = pod.pod_id.clone();
                self.pods.insert(pod.pod_id.clone(), pod);
                self.ensure_behaviors_fetched(&pod_id);
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
            ServerToClient::PodSystemPromptUpdated { .. } => {
                // No rendered view of the prompt text today, so nothing
                // for the UI to refresh. The event is still delivered so
                // a future "inspect current system prompt" panel can
                // stay in sync without polling.
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
                // Pod snapshots inline the behavior catalog so the
                // behaviors panel renders without an extra round trip
                // after opening a pod's detail view.
                self.behaviors_by_pod
                    .insert(snapshot.pod_id.clone(), snapshot.behaviors);
                self.pod_configs
                    .insert(snapshot.pod_id.clone(), snapshot.config);
            }
            ServerToClient::BehaviorList {
                pod_id, behaviors, ..
            } => {
                self.behaviors_by_pod.insert(pod_id, behaviors);
            }
            ServerToClient::BehaviorSnapshot {
                correlation_id,
                snapshot,
            } => {
                self.apply_behavior_snapshot(correlation_id, snapshot);
            }
            ServerToClient::BehaviorStateChanged {
                pod_id,
                behavior_id,
                state,
            } => {
                if let Some(list) = self.behaviors_by_pod.get_mut(&pod_id)
                    && let Some(row) = list.iter_mut().find(|b| b.behavior_id == behavior_id)
                {
                    row.run_count = state.run_count;
                    row.last_fired_at = state.last_fired_at.clone();
                    row.enabled = state.enabled;
                }
            }
            ServerToClient::PodBehaviorsEnabledChanged {
                correlation_id: _,
                pod_id,
                enabled,
            } => {
                if let Some(pod) = self.pods.get_mut(&pod_id) {
                    pod.behaviors_enabled = enabled;
                }
            }
            ServerToClient::BehaviorCreated {
                correlation_id,
                summary,
            } => {
                let list = self
                    .behaviors_by_pod
                    .entry(summary.pod_id.clone())
                    .or_default();
                if let Some(existing) = list
                    .iter_mut()
                    .find(|b| b.behavior_id == summary.behavior_id)
                {
                    *existing = summary.clone();
                } else {
                    list.push(summary.clone());
                    list.sort_by(|a, b| a.behavior_id.cmp(&b.behavior_id));
                }
                // If the creation was initiated from the "+ New behavior"
                // modal, close that and open the editor for the new one.
                if let Some(new_modal) = &self.new_behavior_modal
                    && new_modal.pending_correlation == correlation_id
                    && correlation_id.is_some()
                {
                    let pod_id = summary.pod_id.clone();
                    let behavior_id = summary.behavior_id.clone();
                    self.new_behavior_modal = None;
                    self.open_behavior_editor(pod_id, behavior_id);
                }
            }
            ServerToClient::BehaviorUpdated {
                correlation_id,
                snapshot,
            } => {
                if let Some(list) = self.behaviors_by_pod.get_mut(&snapshot.pod_id) {
                    let summary = behavior_summary_from_snapshot(&snapshot);
                    if let Some(existing) = list
                        .iter_mut()
                        .find(|b| b.behavior_id == snapshot.behavior_id)
                    {
                        *existing = summary;
                    } else {
                        list.push(summary);
                    }
                }
                // If this update matches our in-flight save, reset the
                // editor's baseline so dirty flips back to clean.
                if let Some(modal) = self.behavior_editor_modal.as_mut()
                    && modal.pod_id == snapshot.pod_id
                    && modal.behavior_id == snapshot.behavior_id
                    && (modal.pending_correlation.is_some()
                        && modal.pending_correlation == correlation_id)
                {
                    if let Some(config) = &snapshot.config {
                        modal.baseline_config = Some(config.clone());
                        // Keep the user's working state — they may have
                        // edited further during the round-trip. But if
                        // they haven't, align working with baseline so
                        // raw_buffer regenerates cleanly on next Raw
                        // tab entry.
                        if modal.working_config.as_ref() == modal.baseline_config.as_ref() {
                            modal.raw_buffer = snapshot.toml_text.clone();
                            modal.raw_dirty = false;
                        }
                    }
                    modal.baseline_prompt = snapshot.prompt.clone();
                    modal.pending_correlation = None;
                    modal.error = None;
                }
            }
            ServerToClient::BehaviorDeleted {
                pod_id,
                behavior_id,
                ..
            } => {
                if let Some(list) = self.behaviors_by_pod.get_mut(&pod_id) {
                    list.retain(|b| b.behavior_id != behavior_id);
                }
                if self.delete_armed_behavior.as_ref()
                    == Some(&(pod_id.clone(), behavior_id.clone()))
                {
                    self.delete_armed_behavior = None;
                }
                if let Some(modal) = &self.behavior_editor_modal
                    && modal.pod_id == pod_id
                    && modal.behavior_id == behavior_id
                {
                    self.behavior_editor_modal = None;
                }
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
            // Don't add optimistically — the server echoes every
            // user-role append via `ThreadUserMessage`, so adding
            // here would double the message. Server-local echo
            // latency is negligible; the extra round-trip is a
            // millisecond at most.
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
        origin: s.origin.clone(),
        continued_from: s.continued_from.clone(),
        dispatched_by: s.dispatched_by.clone(),
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
                if let ContentBlock::Text { text } = block {
                    out.push(DisplayItem::User { text: text.clone() });
                }
            }
        }
        Role::ToolResult => {
            // Role::ToolResult carries either structured ToolResult
            // content blocks (synchronous tool-call results) or plain
            // text (async `dispatch_thread` callbacks). Both get
            // rendered as their own chat row at their chronological
            // position — no fusion into earlier tool-call items —
            // so the operator can see each tool response where it
            // actually landed in the conversation. Proximity to the
            // matching tool-call item drives the default-collapsed
            // behavior.
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        push_tool_result_from_text(out, text);
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let text = tool_result_text(content);
                        push_tool_result(out, tool_use_id.clone(), text, *is_error);
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

/// Push a `DisplayItem::ToolResult` row built from a text payload.
/// For `dispatch_thread` async callbacks the payload is a
/// `<dispatched-thread-notification>` XML envelope — we extract the
/// originating `tool-use-id`, the inner `<result>`, and the `<status>`
/// to drive `is_error`. Any other shape falls back to a `SystemNote`
/// row so unknown text stays visible.
fn push_tool_result_from_text(items: &mut Vec<DisplayItem>, text: &str) {
    if let Some((tool_use_id, result_body, is_error)) = parse_dispatch_notification(text) {
        push_tool_result(items, tool_use_id, result_body, is_error);
        return;
    }
    items.push(DisplayItem::SystemNote {
        text: text.to_string(),
        is_error: false,
    });
}

/// Push a `DisplayItem::ToolResult` at the tail of `items`, deriving
/// the tool name from the most-recent matching `DisplayItem::ToolCall`
/// so the row's header can label itself with the tool that produced
/// the result. The row always default-collapses at render time.
fn push_tool_result(
    items: &mut Vec<DisplayItem>,
    tool_use_id: String,
    text: String,
    is_error: bool,
) {
    let name = items
        .iter()
        .rev()
        .find_map(|item| match item {
            DisplayItem::ToolCall {
                tool_use_id: id,
                name,
                ..
            } if id == &tool_use_id => Some(name.clone()),
            _ => None,
        })
        .unwrap_or_default();
    items.push(DisplayItem::ToolResult {
        tool_use_id,
        name,
        text,
        is_error,
    });
}

/// Parse a `<dispatched-thread-notification>` XML envelope emitted by
/// the server's async dispatch flush. Returns `(tool_use_id, result,
/// is_error)` on a match. Deliberately string-based rather than a
/// full XML parser — our envelope is a fixed shape, tiny, and the
/// server XML-escapes its fields, so a tag-bounded scan is enough.
/// Returns `None` on any structural mismatch so the caller falls
/// back to a generic display.
fn parse_dispatch_notification(text: &str) -> Option<(String, String, bool)> {
    if !text
        .trim_start()
        .starts_with("<dispatched-thread-notification>")
    {
        return None;
    }
    let tool_use_id = extract_tagged_value(text, "tool-use-id")?;
    let result = extract_tagged_value(text, "result").unwrap_or_default();
    let status = extract_tagged_value(text, "status").unwrap_or_default();
    let is_error = matches!(status.as_str(), "failed" | "cancelled");
    // Unescape the XML-escaped <result> body so diffs / code show
    // through correctly rather than as `&lt;T&gt;`.
    Some((tool_use_id, unescape_xml(&result), is_error))
}

fn extract_tagged_value(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end_rel = text[start..].find(&close)?;
    Some(text[start..start + end_rel].to_string())
}

fn unescape_xml(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
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
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_inbound();

        egui::Panel::top("status_bar").show_inside(ui, |ui| {
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

        egui::Panel::left("task_list")
            .resizable(true)
            .default_size(220.0)
            .show_inside(ui, |ui| {
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
        egui::Panel::bottom("input_bar")
            .resizable(false)
            .show_inside(ui, |ui| {
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
                            self.send(ClientToServer::ArchiveThread {
                                thread_id: thread_id.clone(),
                            });
                        }
                        // Compact is only meaningful on idle/completed
                        // threads (the server rejects mid-turn
                        // compaction); gate with `input_enabled` so
                        // the user sees a disabled affordance rather
                        // than a rejection error. Per-thread
                        // `compaction.enabled = false` still round-
                        // trips to a server error if clicked — acceptable
                        // for v1 since the default ships enabled.
                        ui.add_enabled_ui(input_enabled, |ui| {
                            if ui
                                .button("Compact")
                                .on_hover_text(
                                    "Summarize the conversation into a new thread. The \
                                     current thread stays as history; a fresh thread \
                                     seeded with the summary becomes the active one.",
                                )
                                .clicked()
                            {
                                self.send(ClientToServer::CompactThread {
                                    thread_id,
                                    correlation_id: None,
                                });
                            }
                        });
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
        // Pre-split borrows: the closure needs `&mut self.tasks` (via
        // get_mut) and `&mut self.md_cache` simultaneously. Destructuring
        // up front lets the closure capture each field independently
        // instead of re-borrowing through `&mut self`.
        let selected = self.selected.clone();
        let tasks = &mut self.tasks;
        let md_cache = &mut self.md_cache;
        egui::CentralPanel::default().show_inside(ui, |ui| match selected {
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
            Some(thread_id) => match tasks.get_mut(&thread_id) {
                None => {
                    ui.label(
                        RichText::new(format!("task {thread_id} not found"))
                            .color(Color32::from_rgb(220, 120, 120)),
                    );
                }
                Some(view) => {
                    render_thread_context_inspector(ui, &thread_id, view);
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
                                render_item(ui, md_cache, item);
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
            self.send(ClientToServer::RemoveToolAllowlistEntry {
                thread_id,
                tool_name,
            });
        }

        let ctx = ui.ctx().clone();
        self.render_new_pod_modal(&ctx);
        self.render_pod_editor_modal(&ctx);
        self.render_new_behavior_modal(&ctx);
        self.render_behavior_editor_modal(&ctx);
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
        // Scale the sidebar's Small text style up so subsection headers,
        // thread rows, and sub-buttons read at a comfortable size while
        // pod-name headings (TextStyle::Body) stay at their default. The
        // mutation is scoped to this ui via Arc COW.
        if let Some(small) = ui.style_mut().text_styles.get_mut(&egui::TextStyle::Small) {
            small.size *= 1.2;
        }

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
        let (label, pod_behaviors_enabled) = match self.pods.get(pod_id) {
            Some(summary) => (
                format!(
                    "{}  ({})",
                    summary.name,
                    thread_ids.map(|t| t.len()).unwrap_or(0)
                ),
                summary.behaviors_enabled,
            ),
            None => (
                format!("{pod_id}  ({})", thread_ids.map(|t| t.len()).unwrap_or(0)),
                true,
            ),
        };
        let collapsed_id = format!("pod-section-{pod_id}");
        let default_open = !self.collapsed_pods.contains(pod_id);
        let is_default_pod = pod_id == self.server_default_pod_id;
        let mut archive_clicked = false;
        let mut archive_confirmed = false;
        let mut archive_disarm = false;
        let mut edit_config_clicked = false;
        let mut toggle_pod_behaviors_to: Option<bool> = None;
        let mut behavior_actions: Vec<BehaviorRowAction> = Vec::new();
        let header = egui::CollapsingHeader::new(RichText::new(label).strong())
            .id_salt(&collapsed_id)
            .default_open(default_open)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if sidebar_button(ui, RichText::new("+ Thread in this pod"), true).clicked() {
                        self.selected = None;
                        self.composing_new = true;
                        self.compose_pod_id = Some(pod_id.to_string());
                        self.input.clear();
                    }
                    if sidebar_button(ui, RichText::new("Edit config"), true).clicked() {
                        edit_config_clicked = true;
                    }
                    let pod_pause_label = if pod_behaviors_enabled {
                        "Pause behaviors"
                    } else {
                        "Resume behaviors"
                    };
                    if sidebar_button(ui, RichText::new(pod_pause_label), true).clicked() {
                        toggle_pod_behaviors_to = Some(!pod_behaviors_enabled);
                    }
                    if !pod_behaviors_enabled {
                        ui.label(RichText::new("paused").small().color(SIDEBAR_WARNING_COLOR));
                    }
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
                        } else if sidebar_button(
                            ui,
                            RichText::new("Archive").color(SIDEBAR_MUTED_COLOR),
                            true,
                        )
                        .clicked()
                        {
                            archive_clicked = true;
                        }
                    }
                });
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
                let has_any_threads = !interactive.is_empty() || !by_behavior.is_empty();
                self.render_interactive_threads(ui, pod_id, &interactive);
                self.render_behaviors_panel(ui, pod_id, &by_behavior, &mut behavior_actions);
                if !has_any_threads {
                    ui.label(
                        RichText::new("(no threads)")
                            .small()
                            .italics()
                            .color(SIDEBAR_MUTED_COLOR),
                    );
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

    /// Fire a `ListBehaviors` for `pod_id` iff we haven't already.
    /// Called on pod discovery (PodList / PodCreated) so the pod
    /// section shows pre-existing behaviors without waiting for the
    /// user to open the pod editor. `PodSnapshot` (from GetPod) also
    /// populates the cache when the editor is opened; the dedup
    /// guard means both paths stay consistent.
    fn ensure_behaviors_fetched(&mut self, pod_id: &str) {
        if self.behaviors_requested.contains(pod_id) {
            return;
        }
        self.behaviors_requested.insert(pod_id.to_string());
        self.send(ClientToServer::ListBehaviors {
            correlation_id: None,
            pod_id: pod_id.to_string(),
        });
    }

    fn open_behavior_editor(&mut self, pod_id: String, behavior_id: String) {
        self.send(ClientToServer::GetBehavior {
            correlation_id: None,
            pod_id: pod_id.clone(),
            behavior_id: behavior_id.clone(),
        });
        self.behavior_editor_modal = Some(BehaviorEditorModalState::new(pod_id, behavior_id));
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
            if sidebar_button(ui, RichText::new("+ New"), true).clicked() {
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
        ui.horizontal(|ui| {
            let label_text = match &row.trigger_kind {
                Some(kind) => format!("{}  [{}]", row.name, kind),
                None => format!("{}  [errored]", row.name),
            };
            let color = if row.load_error.is_some() {
                SIDEBAR_ERROR_TEXT_COLOR
            } else if !row.enabled {
                SIDEBAR_DIM_COLOR
            } else {
                SIDEBAR_BODY_COLOR
            };
            ui.label(RichText::new(label_text).small().strong().color(color));
            if !row.enabled {
                ui.label(RichText::new("paused").small().color(SIDEBAR_WARNING_COLOR));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let armed = self
                    .delete_armed_behavior
                    .as_ref()
                    .map(|(p, b)| p == &row.pod_id && b == &row.behavior_id)
                    .unwrap_or(false);
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
                    if sidebar_button(ui, RichText::new("Delete").color(SIDEBAR_MUTED_COLOR), true)
                        .clicked()
                    {
                        actions.push(BehaviorRowAction::ArmDelete {
                            pod_id: row.pod_id.clone(),
                            behavior_id: row.behavior_id.clone(),
                        });
                    }
                    if sidebar_button(ui, RichText::new("Edit"), true).clicked() {
                        actions.push(BehaviorRowAction::Edit {
                            pod_id: row.pod_id.clone(),
                            behavior_id: row.behavior_id.clone(),
                        });
                    }
                    // Errored behaviors can't be run — disable Run until
                    // the user fixes the config.
                    let run_enabled = row.load_error.is_none();
                    if sidebar_button(ui, RichText::new("Run"), run_enabled).clicked() {
                        actions.push(BehaviorRowAction::Run {
                            pod_id: row.pod_id.clone(),
                            behavior_id: row.behavior_id.clone(),
                        });
                    }
                    let pause_label = if row.enabled { "Pause" } else { "Resume" };
                    if sidebar_button(ui, RichText::new(pause_label), true).clicked() {
                        actions.push(BehaviorRowAction::SetEnabled {
                            pod_id: row.pod_id.clone(),
                            behavior_id: row.behavior_id.clone(),
                            enabled: !row.enabled,
                        });
                    }
                }
            });
        });
        if let Some(err) = &row.load_error {
            ui.label(
                RichText::new(format!("  ⚠ {err}"))
                    .small()
                    .color(SIDEBAR_ERROR_TEXT_COLOR),
            );
        } else if let Some(last) = &row.last_fired_at {
            ui.label(
                RichText::new(format!("  last fired: {last}"))
                    .small()
                    .color(SIDEBAR_MUTED_COLOR),
            );
        }
        if !threads.is_empty() {
            self.render_nested_thread_list(ui, &row.pod_id, &row.behavior_id, threads, actions);
        }
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
    /// Skipped entirely when the pod has no such threads.
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
        if interactive.is_empty() {
            return;
        }
        ui.add_space(4.0);
        sidebar_subsection_header(ui, format!("Interactive  ({})", interactive.len()));
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
    fn apply_behavior_snapshot(
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

    fn render_new_behavior_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.new_behavior_modal.take() else {
            return;
        };
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
                self.new_behavior_modal = Some(modal);
                return;
            }
            let existing = self
                .behaviors_by_pod
                .get(&modal.pod_id)
                .map(|list| list.iter().any(|b| b.behavior_id == behavior_id))
                .unwrap_or(false);
            if existing {
                modal.error = Some(format!("behavior `{behavior_id}` already exists"));
                self.new_behavior_modal = Some(modal);
                return;
            }
            let correlation = self.next_correlation_id();
            let config = BehaviorConfig {
                name: modal.name.trim().to_string(),
                description: None,
                trigger: TriggerSpec::Manual,
                thread: BehaviorThreadOverride::default(),
                on_completion: RetentionPolicy::default(),
            };
            modal.pending_correlation = Some(correlation.clone());
            modal.error = None;
            self.send(ClientToServer::CreateBehavior {
                correlation_id: Some(correlation),
                pod_id: modal.pod_id.clone(),
                behavior_id,
                config,
                prompt: String::new(),
            });
            self.new_behavior_modal = Some(modal);
        } else if cancel_clicked || !open {
            // Modal closes.
        } else {
            self.new_behavior_modal = Some(modal);
        }
    }

    fn render_pod_editor_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.pod_editor_modal.take() else {
            return;
        };

        // Snapshot the catalogs the form needs into owned data so the
        // inner closures don't have to borrow `self`. These are small
        // (single-digit lists in practice) so the clone is cheap.
        let backend_catalog: Vec<String> = self.backends.iter().map(|b| b.name.clone()).collect();
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
        // Snapshot model lists so the Defaults tab's model combo can
        // render without borrowing `self`. Dedup-guarded fetches for
        // the currently-selected backend fire every frame — egui
        // repaints rapidly, and `request_models_for` short-circuits
        // on the second visit.
        if let Some(w) = modal.working.as_ref()
            && !w.thread_defaults.backend.is_empty()
        {
            let b = w.thread_defaults.backend.clone();
            self.request_models_for(&b);
        }
        let models_by_backend = self.models_by_backend.clone();

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
                                        &models_by_backend,
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

    /// Per-behavior editor modal. Structured tabs edit a working
    /// `BehaviorConfig` + prompt; Raw TOML is the escape hatch for the
    /// config (prompt has its own tab, not raw-TOML-editable). Save
    /// ships both through `UpdateBehavior`.
    fn render_behavior_editor_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.behavior_editor_modal.take() else {
            return;
        };
        let backend_catalog: Vec<String> = self.backends.iter().map(|b| b.name.clone()).collect();
        let pod_host_env_names: Vec<String> = self
            .pod_configs
            .get(&modal.pod_id)
            .map(|cfg| cfg.allow.host_env.iter().map(|h| h.name.clone()).collect())
            .unwrap_or_default();
        let pod_mcp_host_names: Vec<String> = self
            .pod_configs
            .get(&modal.pod_id)
            .map(|cfg| cfg.allow.mcp_hosts.clone())
            .unwrap_or_default();
        // Pod's default backend — used as the "effective" backend when
        // the behavior doesn't override `bindings.backend`. The model
        // combo reads it to decide which catalog entry's model list to
        // show. Empty when the pod config hasn't landed yet; in that
        // case the combo renders a "(pick a backend first)" hint.
        let pod_default_backend: String = self
            .pod_configs
            .get(&modal.pod_id)
            .map(|cfg| cfg.thread_defaults.backend.clone())
            .unwrap_or_default();
        // Fire ListModels for whichever backend the model combo is
        // about to show against. Dedup-guarded, so running every frame
        // costs a HashSet lookup after the first fetch.
        let effective_backend_for_models = modal
            .working_config
            .as_ref()
            .and_then(|c| c.thread.bindings.backend.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| pod_default_backend.clone());
        if !effective_backend_for_models.is_empty() {
            self.request_models_for(&effective_backend_for_models);
        }
        let models_by_backend = self.models_by_backend.clone();

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
                                        &models_by_backend,
                                        &pod_default_backend,
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
            // If the raw tab has pending edits, reparse it and use
            // that; otherwise serialize from working. Matches the pod
            // editor's precedence.
            let config = if modal.tab == BehaviorEditorTab::RawToml && modal.raw_dirty {
                match toml::from_str::<BehaviorConfig>(&modal.raw_buffer) {
                    Ok(c) => c,
                    Err(e) => {
                        modal.error = Some(format!("raw TOML doesn't parse: {e}"));
                        self.behavior_editor_modal = Some(modal);
                        return;
                    }
                }
            } else {
                working.clone()
            };
            let correlation = self.next_correlation_id();
            modal.pending_correlation = Some(correlation.clone());
            modal.error = None;
            self.send(ClientToServer::UpdateBehavior {
                correlation_id: Some(correlation),
                pod_id: modal.pod_id.clone(),
                behavior_id: modal.behavior_id.clone(),
                config,
                prompt: modal.working_prompt.clone(),
            });
            self.behavior_editor_modal = Some(modal);
        } else if revert_clicked {
            if let Some(baseline) = &modal.baseline_config {
                modal.working_config = Some(baseline.clone());
                modal.raw_buffer = toml::to_string_pretty(baseline).unwrap_or_default();
                modal.raw_dirty = false;
            }
            modal.working_prompt = modal.baseline_prompt.clone();
            modal.error = None;
            modal.pending_correlation = None;
            self.behavior_editor_modal = Some(modal);
        } else if close_clicked || !open {
            // Modal closes.
        } else {
            self.behavior_editor_modal = Some(modal);
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
                approval_policy: ApprovalPolicy::PromptPodModify,
                host_env: String::new(),
                mcp_hosts: Vec::new(),
                compaction: Default::default(),
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

fn resource_state_chip(state: ResourceStateLabel) -> (&'static str, Color32) {
    match state {
        ResourceStateLabel::Provisioning => ("provisioning", Color32::from_rgb(180, 160, 90)),
        ResourceStateLabel::Ready => ("ready", Color32::from_rgb(120, 180, 120)),
        ResourceStateLabel::Errored => ("errored", Color32::from_rgb(200, 110, 110)),
        ResourceStateLabel::TornDown => ("torn down", Color32::from_gray(140)),
    }
}

/// Collapsible inspector at the top of the thread view. Surfaces
/// everything a snapshot carries that isn't already visible as a
/// conversation item: pod/thread identity, timestamps, bindings,
/// sampling caps, approval policy, trigger origin, and the full
/// system prompt. Collapsed by default — most sessions never open
/// it. egui's `CollapsingHeader` persists open/closed state by
/// `id_salt` so flipping the arrow survives repaints.
fn render_thread_context_inspector(ui: &mut egui::Ui, thread_id: &str, view: &TaskView) {
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
                if let Some(policy) = inspector.approval_policy {
                    kv_row(ui, "approval_policy", approval_policy_label(policy));
                }
                let host_env_label = match &inspector.bindings.host_env {
                    Some(HostEnvBinding::Named { name }) => name.clone(),
                    Some(HostEnvBinding::Inline { provider, .. }) => {
                        format!("(inline, provider = {provider})")
                    }
                    None => "(none — shared MCPs only)".to_string(),
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
                if !view.tool_allowlist.is_empty() {
                    kv_row(ui, "tool_allowlist", &view.tool_allowlist.join(", "));
                }
            });

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

        ui.add_space(6.0);
        section_heading(ui, "System prompt");
        if inspector.system_prompt.is_empty() {
            ui.label(
                RichText::new("(empty — model runs with no system prompt)")
                    .italics()
                    .color(Color32::from_gray(160)),
            );
        } else {
            // Read-only: users edit the prompt via the pod editor's
            // system_prompt_file, not here. Sized to fit a few lines of
            // content up to a reasonable cap so very long prompts
            // don't eat the whole viewport.
            let mut prompt = inspector.system_prompt.as_str();
            let line_count = inspector.system_prompt.lines().count().clamp(3, 20);
            ui.add_sized(
                [ui.available_width(), 0.0],
                TextEdit::multiline(&mut prompt)
                    .code_editor()
                    .desired_rows(line_count)
                    .interactive(false),
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::{ContentBlock, Conversation, Message, ToolResultContent};

    fn conv_with_tool_call() -> Conversation {
        let mut conv = Conversation::new();
        conv.push(Message::user_text("hi"));
        conv.push(Message::assistant_blocks(vec![
            ContentBlock::Text {
                text: "running a tool".into(),
            },
            ContentBlock::ToolUse {
                id: "tu-1".into(),
                name: "bash".into(),
                input: serde_json::json!({ "command": "ls" }),
            },
        ]));
        conv.push(Message::tool_result_blocks(vec![
            ContentBlock::ToolResult {
                tool_use_id: "tu-1".into(),
                content: ToolResultContent::Text("file1\nfile2\n".into()),
                is_error: false,
            },
        ]));
        conv
    }

    #[test]
    fn snapshot_rebuild_emits_sync_tool_result_row() {
        let items = conversation_to_items(&conv_with_tool_call());
        // Tool call item still appears at its original position.
        let has_tool_call = items.iter().any(|i| {
            matches!(
                i,
                DisplayItem::ToolCall { tool_use_id, .. } if tool_use_id == "tu-1"
            )
        });
        assert!(has_tool_call, "tool call row should be present");
        // Tool result lands as its own row, immediately after the
        // call, carrying the tool name and default-collapsed.
        let result = items
            .iter()
            .find_map(|i| match i {
                DisplayItem::ToolResult {
                    tool_use_id,
                    name,
                    text,
                    is_error,
                } if tool_use_id == "tu-1" => Some((name.clone(), text.clone(), *is_error)),
                _ => None,
            })
            .expect("tool result row should be present");
        assert_eq!(result.0, "bash", "tool_result inherits the call's name");
        assert_eq!(result.1, "file1\nfile2\n");
        assert!(!result.2);
    }

    #[test]
    fn snapshot_rebuild_of_real_file_shows_tool_calls() {
        // Regression-style fixture: parse the real persisted thread
        // the user reported as missing tool calls in the webui and
        // assert that add_message_items produces ToolCall items for
        // every assistant tool_use in the file.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../sandbox/pods/workspace/threads/task-18a79d4b2206aa08.json");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipping: fixture {:?} not available in this env", path);
            return;
        };
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let conv: Conversation =
            serde_json::from_value(val.get("conversation").unwrap().clone()).unwrap();
        let items = conversation_to_items(&conv);
        let tool_call_count = items
            .iter()
            .filter(|i| matches!(i, DisplayItem::ToolCall { .. }))
            .count();
        let roles_debug: Vec<&'static str> = items
            .iter()
            .map(|i| match i {
                DisplayItem::User { .. } => "user",
                DisplayItem::AssistantText { .. } => "assistant_text",
                DisplayItem::Reasoning { .. } => "reasoning",
                DisplayItem::ToolCall { .. } => "tool_call",
                DisplayItem::ToolResult { .. } => "tool_result",
                DisplayItem::SystemNote { .. } => "system_note",
            })
            .collect();
        assert!(
            tool_call_count > 0,
            "expected at least one ToolCall item, got items={roles_debug:?}"
        );
        // Every tool_use in the real file should land with a
        // matching `DisplayItem::ToolResult` row produced by the
        // walk — either from a sync tool_result block or from an
        // async XML callback.
        use std::collections::HashSet;
        let mut call_ids: HashSet<String> = HashSet::new();
        let mut result_ids: HashSet<String> = HashSet::new();
        for item in &items {
            match item {
                DisplayItem::ToolCall { tool_use_id, .. } => {
                    call_ids.insert(tool_use_id.clone());
                }
                DisplayItem::ToolResult { tool_use_id, .. } => {
                    result_ids.insert(tool_use_id.clone());
                }
                _ => {}
            }
        }
        assert_eq!(
            call_ids, result_ids,
            "every tool call in the real file should have a matching ToolResult row"
        );
        // The real file has 4 sync tool_results and 2 async XML
        // callbacks (bound to two of the same tool_use_ids — async
        // dispatch emits an initial ack plus a later callback). Six
        // rows total.
        let tool_results = items
            .iter()
            .filter(|i| matches!(i, DisplayItem::ToolResult { .. }))
            .count();
        assert_eq!(tool_results, 6, "expected 6 tool_result rows");
    }

    #[test]
    fn snapshot_rebuild_emits_two_rows_for_async_dispatch() {
        // Async `dispatch_thread` produces an initial sync ack plus
        // a later XML-envelope callback — two separate tool_result
        // rows bound to the same tool_use_id. Both should appear
        // distinctly in the snapshot replay.
        let mut conv = Conversation::new();
        conv.push(Message::user_text("dispatch async"));
        conv.push(Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: "tu-async".into(),
            name: "dispatch_thread".into(),
            input: serde_json::json!({ "prompt": "go", "sync": false }),
        }]));
        conv.push(Message::tool_result_blocks(vec![
            ContentBlock::ToolResult {
                tool_use_id: "tu-async".into(),
                content: ToolResultContent::Text("Dispatched task-X".into()),
                is_error: false,
            },
        ]));
        conv.push(Message::assistant_blocks(vec![ContentBlock::Text {
            text: "waiting for the dispatched thread".into(),
        }]));
        conv.push(Message::tool_result_text(
            "<dispatched-thread-notification>\n  <thread-id>task-X</thread-id>\n  \
             <tool-use-id>tu-async</tool-use-id>\n  <status>completed</status>\n  \
             <summary>done</summary>\n  <result>the real final answer with &lt;code&gt; in it</result>\n  \
             <usage><total_tokens>10</total_tokens><tool_uses>0</tool_uses><duration_ms>5</duration_ms></usage>\n\
             </dispatched-thread-notification>",
        ));
        let items = conversation_to_items(&conv);
        let results: Vec<String> = items
            .iter()
            .filter_map(|i| match i {
                DisplayItem::ToolResult {
                    tool_use_id, text, ..
                } if tool_use_id == "tu-async" => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(results.len(), 2, "expected ack row + async callback row");
        assert_eq!(results[0], "Dispatched task-X");
        assert_eq!(results[1], "the real final answer with <code> in it");
    }
}
