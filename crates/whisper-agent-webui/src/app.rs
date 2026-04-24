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

use self::chat_render::{ChatItemEvent, render_item};
use self::editor_render::{
    behavior_summary_from_snapshot, hint, render_behavior_editor_prompt_tab,
    render_behavior_editor_raw_tab, render_behavior_editor_retention_tab,
    render_behavior_editor_scope_tab, render_behavior_editor_thread_tab,
    render_behavior_editor_trigger_tab, render_pod_editor_allow_tab,
    render_pod_editor_defaults_tab, render_pod_editor_limits_tab, render_pod_editor_raw_tab,
    render_sandbox_entry_modal, section_heading,
};

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::time::Duration;

use egui::{Color32, ComboBox, Grid, RichText, ScrollArea, TextEdit};
use egui_commonmark::CommonMarkCache;
use whisper_agent_protocol::sandbox::NetworkPolicy;
use whisper_agent_protocol::{
    AllowMap, BackendSummary, BehaviorConfig, BehaviorOrigin,
    BehaviorSnapshot as BehaviorSnapshotProto, BehaviorSummary, BehaviorThreadOverride,
    ClientToServer, ContentBlock, Conversation, FsEntry, FunctionKind, FunctionSummary,
    HostEnvBinding, HostEnvProviderInfo, HostEnvProviderOrigin, HostEnvReachability, HostEnvSpec,
    Message, ModelSummary, NamedHostEnv, PodAllow, PodConfig, PodLimits, PodSummary,
    ResourceSnapshot, ResourceStateLabel, RetentionPolicy, Role, ServerToClient,
    SharedMcpAuthInput, SharedMcpAuthPublic, SharedMcpHostInfo, ThreadBindings,
    ThreadBindingsRequest, ThreadConfigOverride, ThreadDefaults, ThreadStateLabel, ThreadSummary,
    ToolResultContent, TriggerSpec, TurnLog, Usage,
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

/// Which UI action a click on a pod-file row triggers. Known
/// specializations (pod.toml → pod editor, behaviors/* → behavior
/// editor) dispatch to their modal. JSON files route to a read-only
/// tree viewer; everything else goes through the generic text
/// editor, whose server-side read rejects binaries via a null-byte
/// sniff.
enum PodFileDispatch {
    PodConfig,
    BehaviorConfig(String),
    BehaviorPrompt(String),
    /// Pod-relative path for the generic text editor.
    TextEditor(String),
    /// Pod-relative path for the JSON tree viewer.
    JsonViewer(String),
}

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
/// Idle time after the last keystroke before a `SetThreadDraft`
/// flushes. Thread-switch and submit flush immediately regardless.
const DRAFT_DEBOUNCE: Duration = Duration::from_millis(500);

// Shared palette for the sidebar. Named so the hierarchy is explicit and
// so the panel's tone doesn't drift as new rows/captions are added.
const SIDEBAR_SUBSECTION_COLOR: Color32 = Color32::from_gray(200);
const SIDEBAR_BODY_COLOR: Color32 = Color32::from_gray(210);
const SIDEBAR_MUTED_COLOR: Color32 = Color32::from_gray(150);
const SIDEBAR_DIM_COLOR: Color32 = Color32::from_gray(130);
const SIDEBAR_DANGER_COLOR: Color32 = Color32::from_rgb(220, 90, 90);
const SIDEBAR_WARNING_COLOR: Color32 = Color32::from_rgb(220, 170, 90);
const SIDEBAR_ERROR_TEXT_COLOR: Color32 = Color32::from_rgb(220, 120, 120);

/// Recursive renderer for the JSON tree viewer. `path` uniquely
/// identifies this node (so egui's persistent collapse state doesn't
/// collide across siblings); `label` is the display key
/// (`foo` / `[3]` / `(root)`). Scalars become one-line labels;
/// objects and arrays become collapsible headers with their child
/// counts baked into the label. Only the root is default-open so a
/// thread JSON doesn't dump the full conversation tree on first
/// paint.
fn render_json_node(
    ui: &mut egui::Ui,
    path: &str,
    label: &str,
    value: &serde_json::Value,
    depth: usize,
) {
    const STRING_PREVIEW_BYTES: usize = 80;
    match value {
        serde_json::Value::Null => {
            ui.label(
                RichText::new(format!("{label}: null"))
                    .small()
                    .monospace()
                    .color(Color32::from_gray(150)),
            );
        }
        serde_json::Value::Bool(b) => {
            ui.label(RichText::new(format!("{label}: {b}")).small().monospace());
        }
        serde_json::Value::Number(n) => {
            ui.label(RichText::new(format!("{label}: {n}")).small().monospace());
        }
        serde_json::Value::String(s) => {
            // Escape via Debug so quotes/newlines render visibly.
            let full = format!("{s:?}");
            let preview = if full.len() > STRING_PREVIEW_BYTES {
                let mut cut = STRING_PREVIEW_BYTES;
                while !full.is_char_boundary(cut) && cut > 0 {
                    cut -= 1;
                }
                format!("{}…", &full[..cut])
            } else {
                full.clone()
            };
            let row = ui.label(
                RichText::new(format!("{label}: {preview}"))
                    .small()
                    .monospace(),
            );
            if full.len() > STRING_PREVIEW_BYTES {
                row.on_hover_text(s);
            }
        }
        serde_json::Value::Array(arr) => {
            let header = format!(
                "{label}: [{} item{}]",
                arr.len(),
                if arr.len() == 1 { "" } else { "s" }
            );
            let state_id = ui.make_persistent_id(("json-array", path));
            egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                state_id,
                depth == 0,
            )
            .show_header(ui, |ui| {
                ui.label(RichText::new(header).small().monospace().strong());
            })
            .body(|ui| {
                for (i, item) in arr.iter().enumerate() {
                    let child_path = format!("{path}/{i}");
                    let child_label = format!("[{i}]");
                    render_json_node(ui, &child_path, &child_label, item, depth + 1);
                }
            });
        }
        serde_json::Value::Object(obj) => {
            let header = format!(
                "{label}: {{ {} key{} }}",
                obj.len(),
                if obj.len() == 1 { "" } else { "s" }
            );
            let state_id = ui.make_persistent_id(("json-object", path));
            egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                state_id,
                depth == 0,
            )
            .show_header(ui, |ui| {
                ui.label(RichText::new(header).small().monospace().strong());
            })
            .body(|ui| {
                for (k, v) in obj.iter() {
                    let child_path = format!("{path}/{k}");
                    render_json_node(ui, &child_path, k, v, depth + 1);
                }
            });
        }
    }
}

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

/// Compact single-glyph action button with a hover tooltip. Shares the
/// `Button::small()` frame with `sidebar_button` so icon and text
/// variants mix cleanly in the same row, but the glyph itself renders
/// at default text size — `.small()` on Unicode icons like ⏸/🗑
/// makes them hard to hit and hard to read. Accepts `impl Into<WidgetText>`
/// so callers can pass a plain `&str` for the common case or a styled
/// `RichText` when the glyph needs to shift color (e.g. the ⏻ toggle
/// that tints muted when its underlying state is paused).
///
/// `min_size` pins every icon button to the same dimensions regardless
/// of which font the glyph came from. The sidebar mixes glyphs from
/// NotoEmoji (🗑, 🗄), NotoSansSymbols2 (⏻, ✎), emoji-icon-font (⚙,
/// ⏸, ➕), and the default text font (▶) — each with different
/// intrinsic metrics, which otherwise produces buttons of visibly
/// different heights sitting on wobbling baselines when they line up
/// in a row.
fn sidebar_icon_button(
    ui: &mut egui::Ui,
    icon: impl Into<egui::WidgetText>,
    tooltip: &str,
    enabled: bool,
) -> egui::Response {
    let btn = egui::Button::new(icon)
        .small()
        .min_size(egui::vec2(22.0, 18.0));
    ui.add_enabled(enabled, btn)
        .on_hover_text(tooltip)
        .on_disabled_hover_text(tooltip)
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

/// Format an RFC3339 timestamp as a compact relative string for the
/// sidebar: "just now", "5 min ago", "3 h ago", "yesterday", "4 d ago".
/// Falls back to the original string if parsing fails — keeps the UI
/// resilient to server-side format drift.
fn format_relative_time(rfc3339: &str) -> String {
    use chrono::{DateTime, Utc};
    let Ok(parsed) = DateTime::parse_from_rfc3339(rfc3339) else {
        return rfc3339.to_string();
    };
    let now = Utc::now();
    let secs = (now - parsed.with_timezone(&Utc)).num_seconds();
    // Negative = clock skew; treat as "just now" rather than "in 3s".
    if secs < 30 {
        return "just now".to_string();
    }
    let mins = secs / 60;
    if mins < 60 {
        return format!("{mins} min ago");
    }
    let hours = mins / 60;
    if hours < 24 {
        return format!("{hours} h ago");
    }
    let days = hours / 24;
    if days == 1 {
        return "yesterday".to_string();
    }
    format!("{days} d ago")
}

fn state_chip(state: ThreadStateLabel) -> (&'static str, Color32) {
    match state {
        ThreadStateLabel::Idle => ("idle", Color32::from_gray(160)),
        ThreadStateLabel::Working => ("working", Color32::from_rgb(120, 180, 240)),
        ThreadStateLabel::Completed => ("completed", Color32::from_rgb(120, 200, 140)),
        ThreadStateLabel::Failed => ("failed", Color32::from_rgb(220, 110, 110)),
        ThreadStateLabel::Cancelled => ("cancelled", Color32::from_rgb(180, 140, 140)),
    }
}

/// Short human-readable label for a `FunctionKind`. Used in the
/// functions-popover rows; shorter than the wire variant name so the
/// popover stays compact.
fn function_kind_label(kind: FunctionKind) -> &'static str {
    match kind {
        FunctionKind::CreateThread => "create",
        FunctionKind::CompactThread => "compact",
        FunctionKind::CancelThread => "cancel",
        FunctionKind::RunBehavior => "behavior",
        FunctionKind::BuiltinToolCall => "tool",
        FunctionKind::McpToolUse => "mcp tool",
        FunctionKind::Sudo => "sudo",
    }
}

/// Elapsed-time string for the functions popover, stopwatch-style.
/// Unlike `format_relative_time`, this targets short in-flight
/// durations — most Functions live for seconds, not hours.
fn format_elapsed(started_at_rfc3339: &str) -> String {
    use chrono::{DateTime, Utc};
    let Ok(parsed) = DateTime::parse_from_rfc3339(started_at_rfc3339) else {
        return "—".to_string();
    };
    let now = Utc::now();
    let secs = (now - parsed.with_timezone(&Utc)).num_seconds().max(0);
    if secs < 60 {
        format!("{secs}s")
    } else {
        let m = secs / 60;
        let s = secs % 60;
        format!("{m}:{s:02}")
    }
}

/// Priority-ordered target label for a `FunctionSummary`: the most
/// specific identifier we have (tool > behavior > thread > pod > dash).
fn function_target_label(summary: &FunctionSummary) -> String {
    if let Some(t) = &summary.tool_name {
        return t.clone();
    }
    if let Some(b) = &summary.behavior_id {
        return format!("bh: {b}");
    }
    if let Some(t) = &summary.thread_id {
        // Truncate thread ids — they're long hashes and eat popover width.
        let short = if t.len() > 16 { &t[..16] } else { t.as_str() };
        return format!("thread: {short}…");
    }
    if let Some(p) = &summary.pod_id {
        return format!("pod: {p}");
    }
    "—".to_string()
}

enum DisplayItem {
    User {
        text: String,
        /// Absolute index into the server's `Conversation.messages()`
        /// — `DisplayItem` positions don't map 1:1 (TurnStats,
        /// multi-row assistant turns) so the fork action needs the
        /// real index to send to the server.
        msg_index: usize,
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
    /// Placeholder row shown while the model is still streaming a tool
    /// call's args JSON. Swapped out for a real [`DisplayItem::ToolCall`]
    /// the moment `ThreadToolCallBegin` lands for the same
    /// `tool_use_id`. Purely ephemeral — never survives a snapshot
    /// rebuild.
    ToolCallStreaming {
        tool_use_id: String,
        name: String,
        args_chars: u32,
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
        /// Streaming output accumulated from `ThreadToolCallContent`
        /// events while the call is in flight. Empty until the first
        /// content chunk arrives. Discarded once the final `result`
        /// lands — the persisted `ToolResult` content block is the
        /// source of truth for replay.
        streaming_output: String,
        /// Fused tool response. Populated by `push_tool_result` when
        /// the matching tool_result arrives without an intervening
        /// `User`/`AssistantText` boundary — the common case for sync
        /// calls and for the initial ack of an async `dispatch_thread`.
        /// `None` while the call is still in flight. When populated,
        /// the render shows a status chip + one-line preview in the
        /// header and the full text in the expanded body.
        result: Option<FusedToolResult>,
    },
    /// Standalone tool response — rendered as its own chat row when
    /// the result is "distant" from its call (separated by at least
    /// one assistant or user turn). Typical source: async
    /// `dispatch_thread` callback landing after the conversation has
    /// moved on. Always default-collapsed.
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
    /// Thread-prefix setup message captured at creation: the system
    /// prompt (`Role::System` — `text` holds the prompt body) or the
    /// tool-manifest snapshot (`Role::Tools` — `text` holds a
    /// human-readable rendering of the advertised tools). Rendered
    /// inline at the top of the chat log as a default-collapsed row
    /// so what the model saw is visible in-place rather than hidden
    /// behind a side-panel inspector.
    SetupPrompt {
        text: String,
    },
    SetupTools {
        /// Count of tool schemas in the manifest (for the collapsed header).
        count: usize,
        /// Human-readable rendering of the manifest — one entry per
        /// tool, name + description + input-schema — shown when the
        /// row is expanded. Precomputed at item-build time so the
        /// renderer doesn't re-serialize every frame.
        text: String,
    },
    /// Per-LLM-call diagnostic footer, emitted once at the end of
    /// every assistant turn. Sourced live from `ThreadAssistantEnd`
    /// and from `ThreadSnapshot.turn_log` on replay. Rendered as a
    /// compact gray one-liner showing input/output tokens and cache
    /// hit/miss counts — the primary use case is diagnosing why a
    /// cache breakpoint missed when we expected a hit.
    TurnStats {
        usage: Usage,
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

/// Tool result fused onto its originating `DisplayItem::ToolCall`.
/// Rendered as the expanded body's result section and as a one-line
/// preview + status chip in the collapsed header.
#[derive(Clone)]
pub(crate) struct FusedToolResult {
    pub text: String,
    pub is_error: bool,
}

struct TaskView {
    summary: ThreadSummary,
    items: Vec<DisplayItem>,
    total_usage: Usage,
    subscribed: bool,
    /// Backend alias the server resolved for this task. Populated from ThreadSnapshot.
    /// Empty string means "no backend bound" — the status bar renders that
    /// as `—` and model calls would fail until the thread is rebound.
    backend: String,
    /// Model the task was created with. Populated from ThreadSnapshot.
    model: String,
    /// Failure detail carried by the snapshot, if the task ended up in `Failed`.
    /// Rendered as a persistent banner so it survives re-subscribe (unlike items,
    /// which get rebuilt from the conversation on every snapshot).
    failure: Option<String>,
    /// Extra snapshot-carried fields the context inspector renders.
    /// Populated on `ThreadSnapshot`. Small enough to hold per-view
    /// rather than shipping an extra request when the inspector opens.
    inspector: ThreadInspector,
    /// Running count of `Message`s in the server's `Conversation` for
    /// this thread. Tracks the absolute message index at which the
    /// *next* append will land, so `DisplayItem::User.msg_index`
    /// stamps can stay in step with the server's conversation view.
    /// Seeded from `ThreadSnapshot.conversation.len()`; bumped by each
    /// event that appends a message (`ThreadUserMessage`,
    /// `ThreadToolResultMessage`, `ThreadAssistantEnd`, and — via
    /// `pending_tool_batch` below — the sync tool_result batch append
    /// the server performs after all `ToolCallEnd`s of a turn).
    conv_message_count: usize,
    /// Set to `true` on `ThreadToolCallBegin`; flushed to a
    /// `conv_message_count += 1` on the next non-tool event. The server
    /// pushes exactly one `Role::ToolResult` message to the conversation
    /// once all tool calls of an assistant turn resolve (thread.rs step()
    /// transitions out of `AwaitingTools`), but it doesn't emit a
    /// dedicated event for that append — the sync path only surfaces per-
    /// tool `ToolCallEnd`s. Without this flag, a turn with N tool calls
    /// leaves `conv_message_count` under by 1 and every subsequent
    /// `DisplayItem::User.msg_index` stamped by streaming is wrong,
    /// which is how fork_thread's server-side user-role check ends up
    /// rejecting a perfectly valid fork target.
    pending_tool_batch: bool,
    /// Server-confirmed draft. Authoritative for re-populating the
    /// compose box when the user switches back to this thread; when
    /// this thread *is* selected, `ChatApp.input` leads and this
    /// field tracks whatever the server last acknowledged.
    draft: String,
    /// Latest `(tokens_processed, tokens_total)` reported by the
    /// backend while prefilling the next assistant turn. Set by
    /// `ThreadPrefillProgress` events (only emitted by the llamacpp
    /// driver today), cleared the moment the first delta of that turn
    /// arrives or the turn ends. Held here so a future UI pass can
    /// render a transient progress bar without round-tripping to
    /// state that's cleared on every snapshot rebuild.
    prefill_progress: Option<(u32, u32)>,
}

/// Everything the thread-context inspector surfaces that isn't
/// already on `TaskView` (backend, model, failure, allowlist, usage,
/// summary). Split into its own struct so `TaskView` stays readable
/// at a glance — the inspector fields are a handful of seldom-changing
/// reference values, not per-turn state.
///
/// The system prompt used to live here; it now rides at the head of
/// the thread's `Conversation` and is rendered inline in the chat log
/// (default-collapsed), so the inspector no longer displays it
/// separately. Keeps the conversation log as the single source of
/// truth for what the model actually saw.
/// In-flight `sudo` approval request the server is awaiting a decision
/// on — the wrapped tool name, inner args, and the model's
/// justification. Stored on `ChatApp` keyed by `function_id`; rendered
/// as a banner above the selected thread's chat log.
#[derive(Clone)]
struct PendingSudo {
    thread_id: String,
    tool_name: String,
    args: serde_json::Value,
    reason: String,
}

#[derive(Clone, Default)]
struct ThreadInspector {
    max_tokens: u32,
    max_turns: u32,
    bindings: ThreadBindings,
    origin: Option<BehaviorOrigin>,
    created_at: String,
    /// Thread's current effective permission scope. Updated on every
    /// `ThreadSnapshot` — so escalation grants that widen the scope
    /// land here the moment the server re-broadcasts the snapshot.
    scope: whisper_agent_protocol::permission::Scope,
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
            backend: String::new(),
            model: String::new(),
            failure: None,
            inspector: ThreadInspector {
                max_tokens: 0,
                max_turns: 0,
                bindings: ThreadBindings::default(),
                origin,
                created_at,
                scope: whisper_agent_protocol::permission::Scope::default(),
            },
            conv_message_count: 0,
            pending_tool_batch: false,
            draft: String::new(),
            prefill_progress: None,
        }
    }

    /// Cash in the `pending_tool_batch` flag into a `conv_message_count`
    /// bump. Called from the `handle_wire` prologue whenever an event
    /// arrives that isn't part of the tool-streaming trio — i.e. the
    /// server is moving on from the tool phase, which on the server side
    /// means `thread.rs::step()` has already pushed the batched
    /// `Role::ToolResult` message. Idempotent when the flag is clear.
    fn flush_pending_tool_batch(&mut self) {
        if self.pending_tool_batch {
            self.conv_message_count += 1;
            self.pending_tool_batch = false;
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
    backends_requested: bool,
    /// Cached model lists keyed by backend name.
    models_by_backend: HashMap<String, Vec<ModelSummary>>,
    /// Backends we've already sent a ListModels request for — dedup so UI changes
    /// don't re-request repeatedly.
    models_requested: HashSet<String>,
    /// Backend chosen in the new-thread picker. None = follow the
    /// compose target pod's `thread_defaults.backend` (or, if no pod
    /// is resolved yet, the server default).
    picker_backend: Option<String>,
    /// Model chosen in the new-thread picker. None = follow the pod's
    /// `thread_defaults.model` when the backend is also unresolved;
    /// when the user overrides the backend we fall through to the
    /// backend catalog's default model instead.
    picker_model: Option<String>,
    /// `[[allow.host_env]]` entries the compose form is targeting for
    /// the new thread. `None` = inherit the target pod's
    /// `thread_defaults.host_env`; `Some(vec)` = replace exactly (empty
    /// vec means "no host env — bare thread"). Every name must resolve
    /// in the pod's allow list — the webui never invents inline specs,
    /// and the server rejects unknown names. Reset back to `None` after
    /// every submit so the next compose starts from the pod default,
    /// not whatever the previous thread used.
    compose_host_env: Option<Vec<String>>,
    /// Compose target pod id we last resolved the picker overrides
    /// against. When this changes (e.g. the user clicks "+ Thread" in
    /// a different pod), we clear `picker_backend`/`picker_model`/
    /// `compose_host_env` so the pickers re-anchor on the new pod's
    /// `thread_defaults`. Without this, a pick in pod A would stick
    /// when the user switches to pod B and no longer reflect that
    /// pod's defaults.
    last_composed_pod: Option<String>,

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
    /// Server's shared-MCP-host catalog. Populated lazily on the first
    /// ListSharedMcpHosts round-trip (triggered when the settings
    /// modal opens the Shared MCP tab). Read-only for non-admin
    /// clients; admins can add/edit/remove from the settings panel.
    shared_mcp_hosts: Vec<SharedMcpHostInfo>,
    shared_mcp_hosts_requested: bool,
    /// Modal state for the per-provider add/edit form. `None` = closed.
    /// Opened from the Providers tab (+Add or Edit row button).
    provider_editor_modal: Option<ProviderEditorModalState>,
    /// Open state for the cog-button "Server settings" modal. `None`
    /// when closed. The inner state holds which settings tab is
    /// currently selected.
    settings_modal: Option<SettingsModalState>,
    /// Provider names whose Remove button has been clicked once and is
    /// waiting for confirmation. Two-click UX mirrors the pod archive
    /// and behavior delete flows. Cleared on confirm / outside click /
    /// successful remove response.
    provider_remove_armed: HashSet<String>,
    /// Per-provider in-flight remove correlation + any returned error.
    /// The row renders "removing..." while a correlation is present;
    /// on `HostEnvProviderRemoved` the entry is cleared; on `Error` the
    /// message is stored for display and the correlation is cleared.
    provider_remove_pending: HashMap<String, ProviderRemovePending>,
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
    /// Modal state for the "fork thread from here" confirm dialog.
    /// Opened by the hover-reveal button on a user-message row;
    /// closed on confirm / cancel / ESC.
    fork_modal: Option<ForkModalState>,
    /// Egui-clock timestamp (seconds since app start) of the last
    /// `input` change not yet debounce-flushed as `SetThreadDraft`.
    /// `f64` instead of [`std::time::Instant`] because `Instant::now()`
    /// panics on wasm32 — the webui's primary target.
    last_input_change_at: Option<f64>,
    /// `(correlation_id, seed_text)` for an outstanding fork: the
    /// matching `ThreadCreated` triggers a `SetThreadDraft` carrying
    /// `seed_text` against the new thread id.
    pending_fork_seed: Option<(String, String)>,
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

    /// Modal state for the generic text editor — the fallback for pod
    /// files that don't route to the pod or behavior editors.
    /// `None` = closed.
    file_viewer_modal: Option<FileViewerModalState>,
    /// Modal state for the JSON tree viewer. Opened by clicking any
    /// `.json` file in the tree. `None` = closed.
    json_viewer_modal: Option<JsonViewerModalState>,
    /// Pod id whose file tree is currently open in its modal viewer.
    /// `None` = closed. The tree's per-directory expansion state and
    /// cached listings live in egui memory and `pod_files` respectively
    /// — reopening the modal resumes where the user left off.
    file_tree_modal_pod: Option<String>,

    /// Shallow directory listings keyed by `(pod_id, pod_relative_path)`.
    /// Empty path is the pod root. Populated by `PodDirListing`
    /// responses and consumed by the sidebar's per-pod file-tree
    /// subsection. Never invalidated today — the tree refreshes only
    /// on reload (future work: wire a refresh button / push events).
    pod_files: HashMap<(String, String), Vec<FsEntry>>,
    /// `(pod_id, path)` pairs we've already asked the server about.
    /// Guards against repeatedly firing `ListPodDir` during the time
    /// between the expansion that triggered the fetch and the
    /// response landing.
    pod_files_requested: HashSet<(String, String)>,

    /// Snapshot of every Function currently registered on the server,
    /// keyed by `function_id`. `BTreeMap` for deterministic rendering
    /// order (oldest first) in the status-bar popover — functions are
    /// generally short-lived, and users scanning the list want a
    /// stable layout that doesn't shuffle as new entries arrive.
    /// Populated from `FunctionList` on connect and kept in sync via
    /// `FunctionStarted` / `FunctionEnded` broadcasts.
    active_functions: std::collections::BTreeMap<u64, FunctionSummary>,
    /// Guards against resending `ListFunctions` on every idle
    /// ConnectionOpened (the current reconnect handler reissues the
    /// list-bootstrap suite).
    functions_requested: bool,

    /// Pending `sudo` approval prompts awaiting user decision. Keyed
    /// by `function_id` — the id the server expects back in
    /// `ResolveSudo`. Populated by `SudoRequested`, drained by
    /// `SudoResolved` (or by an explicit click that optimistically
    /// removes the entry).
    pending_sudos: HashMap<u64, PendingSudo>,
    /// Draft text the user typed into the reject-reason field of a
    /// sudo banner, keyed by `function_id`.
    sudo_reject_drafts: HashMap<u64, String>,

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

/// Confirm dialog for "fork from this message". Default to
/// archive-on: a fork usually means "try a different branch," and
/// the original is noise in the sidebar from then on.
struct ForkModalState {
    thread_id: String,
    from_message_index: usize,
    archive_original: bool,
    /// When `true`, the server re-derives scope, bindings, config,
    /// and tool_surface from the pod's current `thread_defaults`
    /// instead of inheriting them from the source thread. Default
    /// `false` — explicit opt-in, since most forks are "continue
    /// where I left off" with the source's live settings.
    reset_capabilities: bool,
    /// Captured at click time so `confirm` can seed the new thread's
    /// draft with the original prompt for in-place editing.
    seed_text: String,
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

/// State for the per-provider add/edit modal. `mode` decides whether
/// the name field is editable (Add) or read-only (Edit); everything
/// else is shared. On Save the form dispatches `AddHostEnvProvider` or
/// `UpdateHostEnvProvider` and stores the correlation_id so the
/// corresponding response can find and close the modal.
///
/// Token is a free-text field; for Edit, the initial value is blank —
/// the server never sends tokens back — and saving with an empty
/// token means "clear the token on this entry" (matches the
/// protocol's UpdateHostEnvProvider `token: None` semantics).
struct ProviderEditorModalState {
    mode: ProviderEditorMode,
    name: String,
    url: String,
    token: String,
    /// Tracks whether the existing entry had a token at modal open —
    /// used only to render a "currently authenticated" hint next to
    /// the blank token field in Edit mode.
    had_token: bool,
    error: Option<String>,
    pending_correlation: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProviderEditorMode {
    Add,
    Edit,
}

/// State for the "Server settings" modal opened from the cog in the
/// top bar. Host to one or more inner tabs; today only "LLM backends"
/// exists. Kept as a modal rather than a left-panel tab because
/// settings are a low-frequency escape hatch, not part of day-to-day
/// navigation.
#[derive(Default)]
struct SettingsModalState {
    active_tab: SettingsTab,
    /// Open state for the "rotate Codex credentials" sub-form. `None`
    /// when no rotation is in progress; `Some` when the user clicked
    /// Rotate on a `chatgpt_subscription` row.
    codex_rotate: Option<CodexRotateState>,
    /// Last rotation outcome (banner). `Ok(backend)` shows a brief
    /// success line above the list; `Err((backend, detail))` an error.
    /// Cleared the next time the user opens a rotation form.
    codex_rotate_banner: Option<Result<String, (String, String)>>,
    /// Add/edit form for the Shared MCP tab. `None` when the list is
    /// shown; `Some` when the user clicked +Add or Edit.
    shared_mcp_editor: Option<SharedMcpEditorState>,
    /// Most-recent Shared MCP add/update outcome (banner).
    shared_mcp_banner: Option<Result<String, String>>,
    /// Shared MCP host names whose Remove button has been clicked
    /// once and is waiting for confirmation. Cleared on confirm / on
    /// remove response / when the tab closes.
    shared_mcp_remove_armed: HashSet<String>,
    /// Editor state for the Server-config tab. `None` until the tab
    /// has been opened at least once; once initialized, persists
    /// across tab switches so in-progress edits aren't lost.
    server_config: Option<ServerConfigEditorState>,
}

/// Summary of a successful `UpdateServerConfig` — shown as a banner
/// on the server-config editor.
#[derive(Debug, Clone)]
struct ServerConfigSaveSummary {
    cancelled_threads: Vec<String>,
    restart_required_sections: Vec<String>,
    pods_with_missing_backends: Vec<String>,
}

/// Editor state for the server-config tab. Fetched lazily on first
/// open — `original` is populated only after the server replies with
/// `ServerConfigFetched`.
struct ServerConfigEditorState {
    /// Raw TOML as last fetched from the server. `None` while the
    /// fetch round-trip is in flight (we render a spinner).
    original: Option<String>,
    /// Text the user is currently editing. Seeded from `original`
    /// when fetch completes.
    working: String,
    /// Correlation id of the in-flight `FetchServerConfig`, if any.
    fetch_correlation: Option<String>,
    /// Correlation id of the in-flight `UpdateServerConfig`, if any.
    save_correlation: Option<String>,
    /// Last save outcome. `Ok` displays the success summary;
    /// `Err(msg)` renders inline above the editor.
    banner: Option<Result<ServerConfigSaveSummary, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SettingsTab {
    #[default]
    Backends,
    /// Host-env provider catalog (sandbox daemons threads provision
    /// jails against). Used to live in the left sidebar next to
    /// Threads/Resources; moved here so all server-wide settings are
    /// in one place and the sidebar can focus on per-thread state.
    HostEnvProviders,
    SharedMcp,
    /// Raw editor for `whisper-agent.toml`. Admin-only — the server
    /// rejects `FetchServerConfig` / `UpdateServerConfig` from
    /// non-admin connections. On save, the server hot-swaps the
    /// backend catalog and cancels any thread using a
    /// removed/modified backend.
    ServerConfig,
}

/// Inline add/edit form state for one Shared MCP host entry.
/// `mode = Add` collects name + url + optional bearer; `mode = Edit`
/// locks the name (pod bindings reference it) and allows url + auth
/// changes. `auth_keep` is Edit-only — true means "leave the existing
/// auth alone" (the default); setting it to false unlocks the bearer
/// field where a blank value means "clear auth" and a non-blank value
/// means "set bearer".
struct SharedMcpEditorState {
    mode: SharedMcpEditorMode,
    name: String,
    url: String,
    /// Which auth variant the operator is currently composing.
    /// Defaults to `Anonymous` on Add, `keep existing` equivalent on
    /// Edit. Selecting `Oauth2Start` disables the bearer field and
    /// (on Save) dispatches the webui-orchestrated OAuth flow; the
    /// server replies with `SharedMcpOauthFlowStarted` carrying the
    /// authorization URL to open.
    auth_choice: SharedMcpAuthChoice,
    /// Staged bearer. Only consulted when `auth_choice` is `Bearer`.
    bearer: String,
    /// Optional scope string (space-separated) for OAuth flows.
    /// Omitted when empty so the server falls back to the AS's
    /// `scopes_supported`.
    oauth_scope: String,
    /// The auth kind on the entry when the editor opened, so the UI
    /// can describe the "keep existing auth" option meaningfully.
    auth_kind_on_load: SharedMcpAuthPublic,
    error: Option<String>,
    pending_correlation: Option<String>,
    /// True while an OAuth flow has been started and the authz URL
    /// has been opened in a new tab, but the final
    /// `SharedMcpHostAdded` hasn't arrived yet. The form stays open
    /// in a "waiting for consent" banner until the callback fires.
    oauth_in_flight: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SharedMcpEditorMode {
    Add,
    Edit,
}

/// Which auth variant the user is composing in the editor. The Edit
/// mode adds a fourth `KeepExisting` semantic via the
/// `SharedMcpAuthInput::None` wire value being elided (see the save
/// path); we model that inline rather than as a variant to keep the
/// radio-row UI simple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SharedMcpAuthChoice {
    /// Anonymous — no Authorization header sent.
    Anonymous,
    /// Static bearer pasted by the operator.
    Bearer,
    /// OAuth 2.1 via the authorization_code + PKCE flow. Only
    /// selectable on Add (Edit of an existing OAuth entry routes
    /// token refresh through a separate path, not this form).
    Oauth2,
}

/// Sub-form shown over the Settings modal when the user clicks
/// "Rotate credentials" on a `chatgpt_subscription` backend. Collects
/// the pasted `auth.json` blob and dispatches `UpdateCodexAuth`. The
/// form stays open while `pending_correlation` is set and closes when
/// the matching response / error lands.
struct CodexRotateState {
    backend: String,
    contents: String,
    error: Option<String>,
    pending_correlation: Option<String>,
}

impl ProviderEditorModalState {
    fn new_add() -> Self {
        Self {
            mode: ProviderEditorMode::Add,
            name: String::new(),
            url: String::new(),
            token: String::new(),
            had_token: false,
            error: None,
            pending_correlation: None,
        }
    }

    fn new_edit(info: &HostEnvProviderInfo) -> Self {
        Self {
            mode: ProviderEditorMode::Edit,
            name: info.name.clone(),
            url: info.url.clone(),
            token: String::new(),
            had_token: info.has_token,
            error: None,
            pending_correlation: None,
        }
    }
}

/// Tracks one in-flight `RemoveHostEnvProvider` request. `correlation`
/// is the id we'll match against `HostEnvProviderRemoved` or `Error`;
/// `error` carries any server-side refusal message (e.g. "referenced
/// by pods [...]") so the row can render it inline.
struct ProviderRemovePending {
    correlation: String,
    error: Option<String>,
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
    Scope,
    Retention,
    Prompt,
    RawToml,
}

impl BehaviorEditorTab {
    fn label(self) -> &'static str {
        match self {
            BehaviorEditorTab::Trigger => "Trigger",
            BehaviorEditorTab::Thread => "Thread",
            BehaviorEditorTab::Scope => "Scope",
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

/// State for the generic pod-file text editor. Opened by clicking a
/// file in the file tree that doesn't route to a specialized editor
/// (pod.toml / behavior config / behavior prompt have their own
/// modals). The editor reads the file once via `ReadPodFile`,
/// populates `working` + `baseline`, and lets the user edit the
/// buffer and save with `WritePodFile`. `readonly` mirrors the
/// server's `is_readonly_path` decision — read-only files hide the
/// Save / Revert buttons entirely so the modal is clearly a viewer.
struct FileViewerModalState {
    pod_id: String,
    path: String,
    /// In-memory edit buffer. `None` until the `PodFileContent`
    /// round-trip lands.
    working: Option<String>,
    /// Last-known server content. Used for the Revert baseline and
    /// the dirty indicator (Save enabled only when they diverge).
    baseline: Option<String>,
    /// Reflects `is_readonly_path` on the file. Runtime state
    /// (thread JSONs, pod_state.json, behaviors/*/state.json) loads
    /// with `true` here and can't be saved.
    readonly: bool,
    /// Last read/write error surfaced inline in the footer.
    error: Option<String>,
    /// Correlation id of the in-flight `WritePodFile`. `Some` during
    /// a save; cleared by the matching `PodFileWritten` (or error).
    pending_correlation: Option<String>,
}

impl FileViewerModalState {
    fn new(pod_id: String, path: String) -> Self {
        Self {
            pod_id,
            path,
            working: None,
            baseline: None,
            readonly: false,
            error: None,
            pending_correlation: None,
        }
    }

    fn is_dirty(&self) -> bool {
        match (&self.working, &self.baseline) {
            (Some(w), Some(b)) => w != b,
            _ => false,
        }
    }
}

/// State for the JSON tree viewer — the read-only counterpart to
/// `FileViewerModalState` used for every `.json` path a user clicks
/// in the file tree. Parses the content on arrival; a parse failure
/// surfaces as `error` with `parsed = None` (raw bytes aren't shown
/// — JSON in a pod dir is always machine-written and a broken file
/// is more useful to surface than silently render as text).
struct JsonViewerModalState {
    pod_id: String,
    path: String,
    /// Parsed value. `None` while the read is in flight OR when the
    /// file failed to parse as JSON — `error` disambiguates.
    parsed: Option<serde_json::Value>,
    error: Option<String>,
    pending_correlation: Option<String>,
}

impl JsonViewerModalState {
    fn new(pod_id: String, path: String) -> Self {
        Self {
            pod_id,
            path,
            parsed: None,
            error: None,
            pending_correlation: None,
        }
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
            backends_requested: false,
            models_by_backend: HashMap::new(),
            models_requested: HashSet::new(),
            picker_backend: None,
            picker_model: None,
            compose_host_env: None,
            last_composed_pod: None,
            resources: HashMap::new(),
            resources_requested: false,
            pods: HashMap::new(),
            pods_requested: false,
            behaviors_requested: HashSet::new(),
            host_env_providers: Vec::new(),
            host_env_providers_requested: false,
            shared_mcp_hosts: Vec::new(),
            shared_mcp_hosts_requested: false,
            provider_editor_modal: None,
            settings_modal: None,
            provider_remove_armed: HashSet::new(),
            provider_remove_pending: HashMap::new(),
            collapsed_pods: HashSet::new(),
            expanded_interactive_pods: HashSet::new(),
            expanded_behavior_threads: HashSet::new(),
            fork_modal: None,
            last_input_change_at: None,
            pending_fork_seed: None,
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
            pod_files: HashMap::new(),
            pod_files_requested: HashSet::new(),
            file_viewer_modal: None,
            json_viewer_modal: None,
            file_tree_modal_pod: None,
            active_functions: std::collections::BTreeMap::new(),
            functions_requested: false,
            pending_sudos: HashMap::new(),
            sudo_reject_drafts: HashMap::new(),
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

    /// Resolve the picker-selected backend name. Fallback chain:
    ///   1. user's explicit `picker_backend`
    ///   2. compose target pod's `thread_defaults.backend` (when the
    ///      pod config has landed)
    ///   3. first backend from the catalog (alphabetical), so the
    ///      picker always has *something* preselected when the pod
    ///      default is empty
    fn effective_picker_backend(&self) -> &str {
        if let Some(b) = self.picker_backend.as_deref() {
            return b;
        }
        if let Some(pid) = self.compose_target_pod_id()
            && let Some(cfg) = self.pod_configs.get(pid)
            && !cfg.thread_defaults.backend.is_empty()
        {
            return &cfg.thread_defaults.backend;
        }
        self.backends
            .iter()
            .map(|b| b.name.as_str())
            .min()
            .unwrap_or("")
    }

    /// Resolve the picker-selected model. Fallback chain:
    ///   1. user's explicit `picker_model`
    ///   2. compose target pod's `thread_defaults.model` — only when
    ///      the backend is also coming from the pod default (picker
    ///      backend override → pair the model override with it too,
    ///      since the pod's model won't be valid on a different backend)
    ///   3. effective backend's `default_model` from the catalog
    ///   4. first model from the fetched ModelsList for that backend
    ///   5. empty (display as "(loading…)")
    fn effective_picker_model(&self) -> String {
        if let Some(m) = &self.picker_model {
            return m.clone();
        }
        if self.picker_backend.is_none()
            && let Some(pid) = self.compose_target_pod_id()
            && let Some(cfg) = self.pod_configs.get(pid)
            && !cfg.thread_defaults.model.is_empty()
        {
            return cfg.thread_defaults.model.clone();
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
                if !self.shared_mcp_hosts_requested {
                    self.send(ClientToServer::ListSharedMcpHosts {
                        correlation_id: None,
                    });
                    self.shared_mcp_hosts_requested = true;
                }
                if !self.functions_requested {
                    self.send(ClientToServer::ListFunctions {
                        correlation_id: None,
                    });
                    self.functions_requested = true;
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
        // Flush a pending sync-tool-batch append on the first event that
        // isn't part of the tool-streaming trio. `thread.rs::step()`
        // pushes exactly one `Role::ToolResult` message to the
        // conversation when all tool calls of a turn resolve, without a
        // dedicated event — so arming on `ToolCallBegin` and flushing
        // here keeps `conv_message_count` in step with server state.
        if let Some(tid) = pending_tool_batch_flush_thread_id(&msg)
            && let Some(view) = self.tasks.get_mut(tid)
        {
            view.flush_pending_tool_batch();
        }
        match msg {
            ServerToClient::ThreadCreated {
                thread_id,
                summary,
                correlation_id,
            } => {
                self.upsert_task(summary);
                self.recompute_order();
                // Fork-seed handoff: if this `ThreadCreated` carries
                // the correlation_id we stamped on the outbound
                // `ForkThread`, the server's `fork_task` just minted
                // this id. Issue a `SetThreadDraft` so the new
                // thread's persisted draft holds the forked-from
                // user-message text. Do this *before* `select_task`,
                // which triggers a `SubscribeToThread` whose snapshot
                // will then include the just-written draft.
                let seed_match = self
                    .pending_fork_seed
                    .as_ref()
                    .is_some_and(|(cid, _)| correlation_id.as_deref() == Some(cid.as_str()));
                if seed_match && let Some((_, text)) = self.pending_fork_seed.take() {
                    self.send(ClientToServer::SetThreadDraft {
                        thread_id: thread_id.clone(),
                        text: text.clone(),
                    });
                    // Mirror into the local cache: `select_task` below
                    // loads `self.input` from `view.draft`, and the
                    // snapshot with the server-side draft may not have
                    // landed yet.
                    if let Some(view) = self.tasks.get_mut(&thread_id) {
                        view.draft = text;
                    }
                }
                if self.selected.as_deref() != Some(&thread_id) {
                    // Don't yank focus for background spawns.
                    // Behavior-triggered threads carry an `origin`;
                    // dispatch_thread-spawned children carry
                    // `dispatched_by`. Bare creates, forks, and
                    // compaction continuations all leave both fields
                    // None and keep the current auto-focus UX the
                    // user expects after taking a direct action.
                    let background = self.tasks.get(&thread_id).is_some_and(|v| {
                        v.summary.origin.is_some() || v.summary.dispatched_by.is_some()
                    });
                    if !background {
                        self.select_task(thread_id);
                    }
                }
            }
            ServerToClient::ThreadStateChanged { thread_id, state } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.summary.state = state;
                    // Any transition away from Working invalidates
                    // transient stream decorations — a mid-args
                    // tool-call placeholder that never got its
                    // matching ToolCallBegin, or a prefill bar whose
                    // turn just got cancelled. Leaving these on
                    // screen is the "stuck spinner" bug.
                    if state != ThreadStateLabel::Working {
                        view.prefill_progress = None;
                        view.items
                            .retain(|it| !matches!(it, DisplayItem::ToolCallStreaming { .. }));
                    }
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
                let items = conversation_to_items(&snapshot.conversation, &snapshot.turn_log);
                let backend = snapshot.bindings.backend.clone();
                let model = snapshot.config.model.clone();
                let failure = snapshot.failure.clone();
                let inspector = ThreadInspector {
                    max_tokens: snapshot.config.max_tokens,
                    max_turns: snapshot.config.max_turns,
                    bindings: snapshot.bindings.clone(),
                    origin: snapshot.origin.clone(),
                    created_at: snapshot.created_at.clone(),
                    scope: snapshot.scope.clone(),
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
                view.inspector = inspector;
                view.conv_message_count = snapshot.conversation.len();
                // Any in-flight tool batch carried over by the client is
                // moot — snapshot length is authoritative, so reset the
                // flush flag to avoid double-counting the same append.
                view.pending_tool_batch = false;
                view.draft = snapshot.draft.clone();
                // Sync compose box from the just-arrived snapshot
                // when we're looking at this thread and haven't
                // started typing yet. Existing `input` content wins
                // — a user who switched back before the snapshot
                // landed shouldn't have their typing clobbered.
                if self.selected.as_deref() == Some(&thread_id) && self.input.is_empty() {
                    self.input = snapshot.draft;
                    self.last_input_change_at = None;
                }
            }
            ServerToClient::ThreadDraftUpdated { thread_id, text } => {
                // Skip redundant echoes (reconnect + resubscribe can
                // replay a draft we already have) so a same-text
                // update doesn't stomp the cursor on a selected
                // thread.
                let same = self.tasks.get(&thread_id).is_some_and(|v| v.draft == text);
                if same {
                    return;
                }
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.draft = text.clone();
                }
                if self.selected.as_deref() == Some(&thread_id) {
                    self.input = text;
                    self.last_input_change_at = None;
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
                    let msg_index = view.conv_message_count;
                    view.conv_message_count += 1;
                    view.items.push(DisplayItem::User { text, msg_index });
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
                    view.conv_message_count += 1;
                    push_tool_result_from_text(&mut view.items, &text);
                }
            }
            ServerToClient::ThreadAssistantBegin { thread_id, .. } => {
                // A fresh assistant turn starts — drop any progress
                // bar we were showing from an earlier turn so it
                // can't bleed into this one.
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.prefill_progress = None;
                }
            }
            ServerToClient::ThreadPrefillProgress {
                thread_id,
                tokens_processed,
                tokens_total,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    // Drop late stream events buffered in the
                    // scheduler's channel when the thread has
                    // already transitioned out of Working — without
                    // this, a just-cancelled turn can have a
                    // progress bar re-appear after the state change
                    // arrived.
                    if view.summary.state == ThreadStateLabel::Working {
                        view.prefill_progress = Some((tokens_processed, tokens_total));
                    }
                }
            }
            ServerToClient::ThreadAssistantTextDelta { thread_id, delta } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.prefill_progress = None;
                    if let Some(DisplayItem::AssistantText { text }) = view.items.last_mut() {
                        text.push_str(&delta);
                    } else {
                        view.items.push(DisplayItem::AssistantText { text: delta });
                    }
                }
            }
            ServerToClient::ThreadAssistantReasoningDelta { thread_id, delta } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.prefill_progress = None;
                    if let Some(DisplayItem::Reasoning { text }) = view.items.last_mut() {
                        text.push_str(&delta);
                    } else {
                        view.items.push(DisplayItem::Reasoning { text: delta });
                    }
                }
            }
            ServerToClient::ThreadToolCallStreaming {
                thread_id,
                tool_use_id,
                name,
                args_chars,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    // Guard against late stream events buffered in
                    // the scheduler's channel arriving after the
                    // thread left Working — otherwise a cancelled
                    // turn can get a stale placeholder added back.
                    if view.summary.state != ThreadStateLabel::Working {
                        return;
                    }
                    // Upsert: if we already have a streaming placeholder
                    // for this call, update its char count in place so
                    // the row doesn't re-order. Otherwise append a new
                    // one at the tail.
                    let existing = view.items.iter_mut().rev().find_map(|it| match it {
                        DisplayItem::ToolCallStreaming {
                            tool_use_id: id, ..
                        } if *id == tool_use_id => Some(it),
                        _ => None,
                    });
                    if let Some(DisplayItem::ToolCallStreaming {
                        name: existing_name,
                        args_chars: existing_chars,
                        ..
                    }) = existing
                    {
                        *existing_name = name;
                        *existing_chars = args_chars;
                    } else {
                        view.items.push(DisplayItem::ToolCallStreaming {
                            tool_use_id,
                            name,
                            args_chars,
                        });
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
                    // Arm the batch-append flag; a later non-tool event
                    // flushes it into `conv_message_count += 1` to match
                    // the `Role::ToolResult` message the server pushes
                    // once all tool calls of this turn resolve.
                    view.pending_tool_batch = true;
                    // Remove any in-flight streaming placeholder for
                    // this call; the full tool-call row below replaces
                    // it with name + args + diff etc.
                    view.items.retain(|it| {
                        !matches!(
                            it,
                            DisplayItem::ToolCallStreaming {
                                tool_use_id: id, ..
                            } if *id == tool_use_id
                        )
                    });
                    view.items.push(build_tool_call_item(
                        tool_use_id,
                        name,
                        args.as_ref(),
                        args_preview,
                    ));
                }
            }
            ServerToClient::ThreadToolCallContent {
                thread_id,
                tool_use_id,
                block,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    append_streaming_output(&mut view.items, &tool_use_id, &block);
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
                    view.conv_message_count += 1;
                    view.items.push(DisplayItem::TurnStats { usage });
                }
            }
            ServerToClient::ThreadLoopComplete { .. } => {}
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
                // File viewer in-flight read or save.
                if let Some(modal) = self.file_viewer_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // JSON viewer in-flight read.
                if let Some(modal) = self.json_viewer_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // Provider editor modal pending add / update.
                if let Some(modal) = self.provider_editor_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // Codex-auth rotation sub-form pending save. Route the
                // detail into the sub-form's error field so the user
                // can fix the paste without retyping; the sub-form
                // stays open until they explicitly cancel.
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(sub) = settings.codex_rotate.as_mut()
                    && correlation_id.is_some()
                    && sub.pending_correlation == correlation_id
                {
                    sub.error = Some(message);
                    sub.pending_correlation = None;
                    return;
                }
                // Shared-MCP editor sub-form pending add/update. Keep
                // the form open and surface the connect/validation
                // failure inline so the operator can fix and retry.
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(sub) = settings.shared_mcp_editor.as_mut()
                    && correlation_id.is_some()
                    && sub.pending_correlation == correlation_id
                {
                    sub.error = Some(message);
                    sub.pending_correlation = None;
                    return;
                }
                // Shared-MCP remove response — no sub-form, just a
                // banner on the parent tab.
                if let Some(settings) = self.settings_modal.as_mut()
                    && correlation_id.is_some()
                    && message.starts_with("remove_shared_mcp_host:")
                {
                    settings.shared_mcp_banner = Some(Err(message));
                    return;
                }
                // Server-config editor — match either the in-flight
                // fetch or the in-flight save correlation. Either
                // failure stays on the editor as an inline banner.
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(editor) = settings.server_config.as_mut()
                    && correlation_id.is_some()
                {
                    if editor.fetch_correlation == correlation_id {
                        editor.banner = Some(Err(format!("fetch: {message}")));
                        editor.fetch_correlation = None;
                        return;
                    }
                    if editor.save_correlation == correlation_id {
                        editor.banner = Some(Err(message));
                        editor.save_correlation = None;
                        return;
                    }
                }
                // Provider remove pending on a specific row. Match by
                // correlation rather than iterating names so a stale
                // remove that targets a now-gone name still resolves.
                if let Some(cid) = correlation_id.as_deref()
                    && let Some((name, _)) = self
                        .provider_remove_pending
                        .iter()
                        .find(|(_, p)| p.correlation == cid)
                        .map(|(n, p)| (n.clone(), p))
                {
                    if let Some(p) = self.provider_remove_pending.get_mut(&name) {
                        p.error = Some(message);
                    }
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
            ServerToClient::BackendsList { backends, .. } => {
                self.backends = backends;
                // Pre-fetch the alphabetically-first backend's models so the
                // picker is ready on first open without a visible delay
                // — that's the entry `effective_picker_backend` falls back
                // to when no pod default is set.
                if let Some(first) = self.backends.iter().map(|b| b.name.clone()).min() {
                    self.request_models_for(&first);
                }
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
            ServerToClient::PodDirListing {
                pod_id,
                path,
                entries,
                ..
            } => {
                let key = (pod_id, path);
                self.pod_files_requested.remove(&key);
                self.pod_files.insert(key, entries);
            }
            ServerToClient::PodFileContent {
                pod_id,
                path,
                content,
                readonly,
                correlation_id,
            } => {
                if let Some(modal) = self.file_viewer_modal.as_mut()
                    && modal.pod_id == pod_id
                    && modal.path == path
                    && modal.pending_correlation == correlation_id
                {
                    modal.baseline = Some(content.clone());
                    modal.working = Some(content);
                    modal.readonly = readonly;
                    modal.pending_correlation = None;
                    modal.error = None;
                } else if let Some(modal) = self.json_viewer_modal.as_mut()
                    && modal.pod_id == pod_id
                    && modal.path == path
                    && modal.pending_correlation == correlation_id
                {
                    modal.pending_correlation = None;
                    match serde_json::from_str::<serde_json::Value>(&content) {
                        Ok(value) => {
                            modal.parsed = Some(value);
                            modal.error = None;
                        }
                        Err(e) => {
                            modal.parsed = None;
                            modal.error = Some(format!("parse JSON: {e}"));
                        }
                    }
                }
            }
            ServerToClient::PodFileWritten {
                pod_id,
                path,
                correlation_id,
            } => {
                if let Some(modal) = self.file_viewer_modal.as_mut()
                    && modal.pod_id == pod_id
                    && modal.path == path
                    && modal.pending_correlation == correlation_id
                {
                    // Save succeeded — adopt the working buffer as the
                    // new baseline so the dirty indicator flips to
                    // "saved" and the tree viewer's cached entry (if
                    // any) remains consistent with disk.
                    if let Some(w) = modal.working.clone() {
                        modal.baseline = Some(w);
                    }
                    modal.pending_correlation = None;
                    modal.error = None;
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
            ServerToClient::FunctionList { functions, .. } => {
                // Snapshot replaces the local map wholesale — the
                // server's registry is the source of truth, and
                // any stale ids we tracked before reconnect should
                // be evicted.
                self.active_functions.clear();
                for summary in functions {
                    self.active_functions.insert(summary.function_id, summary);
                }
            }
            ServerToClient::FunctionStarted { summary } => {
                self.active_functions.insert(summary.function_id, summary);
            }
            ServerToClient::FunctionEnded { function_id, .. } => {
                self.active_functions.remove(&function_id);
            }
            ServerToClient::HostEnvProviderAdded {
                provider,
                correlation_id,
            }
            | ServerToClient::HostEnvProviderUpdated {
                provider,
                correlation_id,
            } => {
                if let Some(existing) = self
                    .host_env_providers
                    .iter_mut()
                    .find(|p| p.name == provider.name)
                {
                    *existing = provider;
                } else {
                    self.host_env_providers.push(provider);
                    self.host_env_providers.sort_by(|a, b| a.name.cmp(&b.name));
                }
                // Close the editor modal if it was waiting on this
                // correlation. Edits that bypassed the modal (server-
                // side seed, another client's CRUD) land here with
                // `None` and leave the modal alone.
                if correlation_id.is_some()
                    && let Some(modal) = self.provider_editor_modal.as_ref()
                    && modal.pending_correlation == correlation_id
                {
                    self.provider_editor_modal = None;
                }
            }
            ServerToClient::HostEnvProviderRemoved {
                name,
                correlation_id,
            } => {
                self.host_env_providers.retain(|p| p.name != name);
                // Clear any pending-remove state if this was our
                // request. Another client's remove lands here too
                // (correlation_id: None); nothing to clean up in that
                // case beyond the list itself.
                if let Some(pending) = self.provider_remove_pending.get(&name)
                    && correlation_id.as_deref() == Some(pending.correlation.as_str())
                {
                    self.provider_remove_pending.remove(&name);
                }
                self.provider_remove_armed.remove(&name);
            }
            ServerToClient::CodexAuthUpdated {
                backend,
                correlation_id,
            } => {
                // Only close the sub-form if this ack matches our own
                // pending rotation. (The server currently only unicasts
                // this event to the requester, but treating a stray
                // broadcast as benign costs nothing.)
                if let Some(settings) = self.settings_modal.as_mut() {
                    let is_ours = settings.codex_rotate.as_ref().is_some_and(|s| {
                        s.pending_correlation.is_some()
                            && s.pending_correlation == correlation_id
                            && s.backend == backend
                    });
                    if is_ours {
                        settings.codex_rotate = None;
                        settings.codex_rotate_banner = Some(Ok(backend));
                    }
                }
            }
            ServerToClient::SharedMcpHostsList { hosts, .. } => {
                self.shared_mcp_hosts = hosts;
            }
            ServerToClient::SharedMcpOauthFlowStarted {
                authorization_url,
                correlation_id,
                name: _,
            } => {
                // Open the authorization URL in a new browser tab
                // and flip the editor form into "waiting for
                // authorization" mode so the operator can't send
                // another request. The final SharedMcpHostAdded
                // (or Error) resolves the flow.
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(sub) = settings.shared_mcp_editor.as_mut()
                    && sub.pending_correlation.is_some()
                    && sub.pending_correlation == correlation_id
                {
                    sub.oauth_in_flight = true;
                }
                open_in_new_tab(&authorization_url);
            }
            ServerToClient::SharedMcpHostAdded {
                host,
                correlation_id,
            }
            | ServerToClient::SharedMcpHostUpdated {
                host,
                correlation_id,
            } => {
                // Replace-or-insert by name so adds and updates both
                // converge the local list without a round-trip.
                if let Some(existing) = self
                    .shared_mcp_hosts
                    .iter_mut()
                    .find(|h| h.name == host.name)
                {
                    *existing = host.clone();
                } else {
                    self.shared_mcp_hosts.push(host.clone());
                    self.shared_mcp_hosts.sort_by(|a, b| a.name.cmp(&b.name));
                }
                // When the open editor's pending correlation matches,
                // the server accepted the save — close the form and
                // show a success banner.
                if let Some(settings) = self.settings_modal.as_mut() {
                    let close_editor = settings.shared_mcp_editor.as_ref().is_some_and(|s| {
                        s.pending_correlation.is_some() && s.pending_correlation == correlation_id
                    });
                    if close_editor {
                        settings.shared_mcp_editor = None;
                        settings.shared_mcp_banner = Some(Ok(host.name.clone()));
                    }
                }
            }
            ServerToClient::SharedMcpHostRemoved {
                name,
                correlation_id: _,
            } => {
                self.shared_mcp_hosts.retain(|h| h.name != name);
                if let Some(settings) = self.settings_modal.as_mut() {
                    settings.shared_mcp_remove_armed.remove(&name);
                    settings.shared_mcp_banner = Some(Ok(format!("Removed `{name}`.")));
                }
            }
            ServerToClient::SudoRequested {
                function_id,
                thread_id,
                tool_name,
                args,
                reason,
            } => {
                self.pending_sudos.insert(
                    function_id,
                    PendingSudo {
                        thread_id,
                        tool_name,
                        args,
                        reason,
                    },
                );
            }
            ServerToClient::SudoResolved { function_id, .. } => {
                self.pending_sudos.remove(&function_id);
                self.sudo_reject_drafts.remove(&function_id);
            }
            ServerToClient::ServerConfigFetched {
                toml_text,
                correlation_id,
            } => {
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(editor) = settings.server_config.as_mut()
                    && correlation_id.is_some()
                    && editor.fetch_correlation == correlation_id
                {
                    editor.original = Some(toml_text.clone());
                    editor.working = toml_text;
                    editor.fetch_correlation = None;
                }
            }
            ServerToClient::ServerConfigUpdateResult {
                cancelled_threads,
                restart_required_sections,
                pods_with_missing_backends,
                correlation_id,
            } => {
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(editor) = settings.server_config.as_mut()
                    && correlation_id.is_some()
                    && editor.save_correlation == correlation_id
                {
                    // Successful save: the working text is now the
                    // authoritative on-disk content — seed
                    // `original` so the "modified" indicator
                    // correctly shows no diff and the Revert button
                    // greys out.
                    editor.original = Some(editor.working.clone());
                    editor.save_correlation = None;
                    editor.banner = Some(Ok(ServerConfigSaveSummary {
                        cancelled_threads,
                        restart_required_sections,
                        pods_with_missing_backends,
                    }));
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
        // Flush before switching so a mid-debounce switch can't lose
        // the tail of what the user just typed.
        self.flush_pending_draft();
        self.selected = Some(thread_id.clone());
        self.composing_new = false;
        self.compose_pod_id = None;
        self.input = self
            .tasks
            .get(&thread_id)
            .map(|v| v.draft.clone())
            .unwrap_or_default();
        self.last_input_change_at = None;
        let need_subscribe = self
            .tasks
            .get(&thread_id)
            .map(|v| !v.subscribed)
            .unwrap_or(true);
        if need_subscribe {
            self.send(ClientToServer::SubscribeToThread { thread_id });
        }
    }

    /// Send `SetThreadDraft` if the compose box has diverged from
    /// the selected thread's cached draft. Called on thread switch
    /// and when the debounce window expires.
    fn flush_pending_draft(&mut self) {
        self.last_input_change_at = None;
        let Some(thread_id) = self.selected.clone() else {
            return;
        };
        let Some(view) = self.tasks.get_mut(&thread_id) else {
            return;
        };
        if view.draft == self.input {
            return;
        }
        view.draft = self.input.clone();
        self.send(ClientToServer::SetThreadDraft {
            thread_id,
            text: self.input.clone(),
        });
    }

    fn submit(&mut self) {
        let text = std::mem::take(&mut self.input);
        // Clear the persisted draft server-side. `CreateThread` has no
        // existing thread to target, so this only runs on follow-ups.
        if let Some(thread_id) = self.selected.clone()
            && !self.composing_new
            && let Some(view) = self.tasks.get_mut(&thread_id)
            && !view.draft.is_empty()
        {
            view.draft.clear();
            self.send(ClientToServer::SetThreadDraft {
                thread_id,
                text: String::new(),
            });
        }
        self.last_input_change_at = None;
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
        // Host env is always a list of references by name into the
        // target pod's `allow.host_env` table. `None` → server applies
        // `thread_defaults.host_env`; `Some(vec)` → replace exactly
        // (empty vec means bare thread). The webui never constructs
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

/// Thread id of a `ServerToClient` event for the purpose of flushing a
/// pending sync-tool-batch append. Returns `None` for the three tool-
/// streaming events (Begin / Content / End) — those are the *tool*
/// phase that the flag was armed for, so they must not flush — and for
/// events that aren't associated with a single thread (pod/resource
/// catalog updates, acks, etc.), which are also outside the per-thread
/// append stream.
fn pending_tool_batch_flush_thread_id(msg: &ServerToClient) -> Option<&str> {
    match msg {
        // Tool streaming — do not flush.
        ServerToClient::ThreadToolCallBegin { .. }
        | ServerToClient::ThreadToolCallContent { .. }
        | ServerToClient::ThreadToolCallEnd { .. } => None,
        // Snapshot resets the counter itself; let its handler take
        // over rather than flush here.
        ServerToClient::ThreadSnapshot { .. } => None,
        // Per-thread events — flush before processing.
        ServerToClient::ThreadUserMessage { thread_id, .. }
        | ServerToClient::ThreadToolResultMessage { thread_id, .. }
        | ServerToClient::ThreadAssistantBegin { thread_id, .. }
        | ServerToClient::ThreadPrefillProgress { thread_id, .. }
        | ServerToClient::ThreadToolCallStreaming { thread_id, .. }
        | ServerToClient::ThreadAssistantTextDelta { thread_id, .. }
        | ServerToClient::ThreadAssistantReasoningDelta { thread_id, .. }
        | ServerToClient::ThreadAssistantEnd { thread_id, .. }
        | ServerToClient::ThreadLoopComplete { thread_id, .. }
        | ServerToClient::ThreadStateChanged { thread_id, .. }
        | ServerToClient::ThreadTitleUpdated { thread_id, .. }
        | ServerToClient::ThreadDraftUpdated { thread_id, .. }
        | ServerToClient::ThreadArchived { thread_id } => Some(thread_id.as_str()),
        _ => None,
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

fn conversation_to_items(conv: &Conversation, turn_log: &TurnLog) -> Vec<DisplayItem> {
    // Interleave a `DisplayItem::TurnStats` after each Assistant-role
    // message, pulled in order from `turn_log.entries`. The runtime
    // pushes exactly one entry per `integrate_model_response`, so entry
    // N corresponds to the Nth Assistant message. Extra entries (never
    // expected) are dropped; a short log (older threads, or
    // mid-migration) leaves trailing turns without a stats row rather
    // than crashing.
    let mut items = Vec::new();
    let mut entry_iter = turn_log.entries.iter();
    for (msg_index, msg) in conv.messages().iter().enumerate() {
        add_message_items(msg, msg_index, &mut items);
        if msg.role == Role::Assistant
            && let Some(entry) = entry_iter.next()
        {
            items.push(DisplayItem::TurnStats { usage: entry.usage });
        }
    }
    items
}

fn add_message_items(msg: &Message, msg_index: usize, out: &mut Vec<DisplayItem>) {
    match msg.role {
        Role::System => {
            // System prompt lives at the head of the conversation as a
            // single `ContentBlock::Text`. Empty prompts produce no row
            // so the chat log doesn't start with a meaningless
            // "(empty)" entry.
            let text = msg
                .content
                .iter()
                .find_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .unwrap_or("");
            if !text.is_empty() {
                out.push(DisplayItem::SetupPrompt {
                    text: text.to_string(),
                });
            }
        }
        Role::Tools => {
            // Tool manifest: one `ContentBlock::ToolSchema` per tool.
            // The collapsed row shows the count; expanded shows
            // name + description + input-schema for each entry.
            let mut rendered = String::new();
            let mut count = 0usize;
            for block in &msg.content {
                if let ContentBlock::ToolSchema {
                    name,
                    description,
                    input_schema,
                } = block
                {
                    if count > 0 {
                        rendered.push_str("\n\n");
                    }
                    rendered.push_str(name);
                    if !description.is_empty() {
                        rendered.push_str(" — ");
                        rendered.push_str(description);
                    }
                    rendered.push('\n');
                    rendered
                        .push_str(&serde_json::to_string_pretty(input_schema).unwrap_or_default());
                    count += 1;
                }
            }
            if count > 0 {
                out.push(DisplayItem::SetupTools {
                    count,
                    text: rendered,
                });
            }
        }
        Role::User => {
            for block in &msg.content {
                if let ContentBlock::Text { text } = block {
                    out.push(DisplayItem::User {
                        text: text.clone(),
                        msg_index,
                    });
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
                    ContentBlock::ToolUse {
                        id, name, input, ..
                    } => {
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

/// Route a tool_result payload onto the items stream.
///
/// If the matching `DisplayItem::ToolCall` can be found walking
/// backward without crossing a `User` / `AssistantText` boundary, the
/// result is fused onto that call (set the call's `result` slot). The
/// common case — sync tool calls and the initial ack of an async
/// `dispatch_thread` — renders as a single combined row instead of
/// two stacked rows, cutting chat-log noise.
///
/// Otherwise (boundary crossed, or no matching call in view) push a
/// standalone `DisplayItem::ToolResult` at the tail. This covers the
/// distant async-callback case where the originating call lives
/// several turns earlier.
/// Append a streaming `ContentBlock` to the matching in-flight tool
/// call's `streaming_output` buffer. Only text blocks have a natural
/// inline rendering today; future non-text block kinds will show as
/// placeholders.
fn append_streaming_output(items: &mut [DisplayItem], tool_use_id: &str, block: &ContentBlock) {
    for item in items.iter_mut().rev() {
        match item {
            DisplayItem::ToolCall {
                tool_use_id: id,
                streaming_output,
                result: None,
                ..
            } if id == tool_use_id => {
                match block {
                    ContentBlock::Text { text } => streaming_output.push_str(text),
                    _ => streaming_output.push_str("[non-text content]"),
                }
                return;
            }
            DisplayItem::User { .. } | DisplayItem::AssistantText { .. } => return,
            _ => continue,
        }
    }
}

fn push_tool_result(
    items: &mut Vec<DisplayItem>,
    tool_use_id: String,
    text: String,
    is_error: bool,
) {
    for item in items.iter_mut().rev() {
        match item {
            DisplayItem::ToolCall {
                tool_use_id: id,
                result,
                ..
            } if id == &tool_use_id => {
                // Fuse only if the call doesn't already have a
                // result. A second arriving result for the same
                // tool_use_id is unusual but can happen (error
                // retries, protocol quirks); fall through to the
                // standalone-row path so the newer data isn't lost
                // silently.
                if result.is_none() {
                    *result = Some(FusedToolResult { text, is_error });
                    return;
                }
                break;
            }
            // Crossing an assistant/user turn means the result is
            // chronologically separated from its call — push as a
            // standalone row at its arrival position.
            DisplayItem::User { .. } | DisplayItem::AssistantText { .. } => break,
            _ => continue,
        }
    }
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
        streaming_output: String::new(),
        result: None,
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
        let mut cut = max;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
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
                ui.label(
                    RichText::new(concat!("v", env!("CARGO_PKG_VERSION")))
                        .color(Color32::from_gray(140))
                        .small(),
                )
                .on_hover_text("whisper-agent-webui crate version (independent of the server)");
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
                    let backend_label = view.backend.as_str();
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
                // Right-aligned: cog (server settings) first, then the
                // in-flight Functions chip to its left when present.
                // Cog is always visible; Functions chip is hidden at
                // zero so the bar doesn't grow a permanent "nothing to
                // see" widget.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let cog = ui
                        .button(RichText::new("⚙").small())
                        .on_hover_text("Server settings");
                    if cog.clicked() {
                        // Toggle: clicking the cog while open closes
                        // the modal, matching a typical settings-button
                        // affordance.
                        self.settings_modal = match self.settings_modal.take() {
                            Some(_) => None,
                            None => Some(SettingsModalState::default()),
                        };
                    }

                    let count = self.active_functions.len();
                    if count == 0 {
                        return;
                    }
                    // Tick the frame while anything is in flight so
                    // the elapsed column advances without needing a
                    // pointer move or wire event to drive a repaint.
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(500));
                    let chip = ui
                        .button(
                            RichText::new(format!("⚡ {count} active"))
                                .small()
                                .color(Color32::from_rgb(220, 180, 100)),
                        )
                        .on_hover_text("In-flight Functions — click to list");
                    egui::Popup::from_toggle_button_response(&chip).show(|ui| {
                        ui.set_min_width(280.0);
                        ui.label(RichText::new("Active Functions").strong().small());
                        ui.add_space(2.0);
                        ui.separator();
                        for summary in self.active_functions.values() {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(function_kind_label(summary.kind))
                                        .small()
                                        .strong()
                                        .color(Color32::from_gray(210)),
                                );
                                ui.label(
                                    RichText::new(function_target_label(summary))
                                        .small()
                                        .color(Color32::from_gray(170)),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            RichText::new(format_elapsed(&summary.started_at))
                                                .small()
                                                .color(Color32::from_gray(150))
                                                .monospace(),
                                        );
                                    },
                                );
                            });
                        }
                    });
                });
            });
        });

        egui::Panel::left("task_list")
            .resizable(true)
            .default_size(300.0)
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
        // Resolve the compose target via `compose_target_pod_id`, not
        // `compose_pod_id` directly — the former falls back to the
        // server default pod, so fresh load still names a pod instead
        // of showing the bare "Describe a new thread" hint with no
        // scope.
        let pod_hint: Option<String> = if composing && input_enabled {
            self.compose_target_pod_id().map(|pid| {
                let display = self.pods.get(pid).map(|p| p.name.as_str()).unwrap_or(pid);
                format!("Describe a new thread in `{display}`")
            })
        } else {
            None
        };
        let hint: &str = match (composing, input_enabled, pod_hint.as_deref()) {
            (_, false, _) => "(connecting)",
            (true, true, Some(s)) => s,
            (true, true, None) => "Describe a new thread",
            (false, true, _) => "Message this thread",
        };

        let show_picker =
            (self.composing_new || self.selected.is_none()) && !self.backends.is_empty();
        if show_picker {
            // When the compose target pod changes, drop stale picker
            // overrides so the pickers re-anchor on the new pod's
            // `thread_defaults`. A user who picked gpt-4 in pod A and
            // then clicks "+ Thread" in pod B expects to see pod B's
            // default, not their pick carried over.
            let current_target = self.compose_target_pod_id().map(str::to_owned);
            if current_target != self.last_composed_pod {
                self.picker_backend = None;
                self.picker_model = None;
                self.compose_host_env = None;
                self.last_composed_pod = current_target.clone();
            }
            if let Some(pod_id) = current_target {
                self.ensure_pod_config(&pod_id);
            }
            // The pod config may arrive after the first render, bumping
            // the effective backend to the pod's default. Keep the
            // model list fresh for whichever backend is currently
            // effective so the model combo isn't stuck on "(loading…)".
            // Dedup-guarded by `models_requested`.
            let effective = self.effective_picker_backend().to_string();
            if !effective.is_empty() {
                self.request_models_for(&effective);
            }
        }
        // Scope header text for the composer panel — answers "what am
        // I about to send, and where" at a glance. Built once here so
        // the borrow of `self.tasks`/`self.pods` doesn't fight the
        // `&mut self` borrow inside `Panel::bottom(...).show_inside`.
        let scope_line: String = if composing {
            match self.compose_target_pod_id() {
                Some(pid) => {
                    let display = self.pods.get(pid).map(|p| p.name.as_str()).unwrap_or(pid);
                    format!("New thread in `{display}`")
                }
                None => "New thread".to_string(),
            }
        } else {
            match self.selected.as_deref().and_then(|tid| self.tasks.get(tid)) {
                Some(view) => {
                    let pod_display = self
                        .pods
                        .get(&view.summary.pod_id)
                        .map(|p| p.name.as_str())
                        .unwrap_or(view.summary.pod_id.as_str());
                    let title = view
                        .summary
                        .title
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .unwrap_or("(untitled)");
                    format!("{title}  ·  in `{pod_display}`")
                }
                None => "(no thread selected)".to_string(),
            }
        };
        let mut request_models: Option<String> = None;
        egui::Panel::bottom("input_bar")
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Label::new(
                            RichText::new(scope_line)
                                .small()
                                .color(Color32::from_gray(190)),
                        )
                        .truncate(),
                    );
                });
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

                        // Host env picker: multi-select over the
                        // target pod's `allow.host_env` entries. An
                        // "override" toggle decides whether the
                        // thread inherits the pod's
                        // `thread_defaults.host_env` (None on the
                        // wire) or replaces it with the explicit
                        // list below (Some(vec) — empty means "bare
                        // thread, no host envs"). Hidden entirely
                        // when the pod has no host envs (threads
                        // there run with shared MCPs only) or the
                        // pod config hasn't landed yet.
                        let pod_config = self
                            .compose_target_pod_id()
                            .and_then(|id| self.pod_configs.get(id));
                        if let Some(pod_config) = pod_config
                            && !pod_config.allow.host_env.is_empty()
                        {
                            ui.separator();
                            let default_names = pod_config.thread_defaults.host_env.join(", ");
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new("host env")
                                        .small()
                                        .color(Color32::from_gray(180)),
                                );
                                let mut override_on = self.compose_host_env.is_some();
                                if ui
                                    .checkbox(
                                        &mut override_on,
                                        format!(
                                            "override (else inherit pod default: {default_names})"
                                        ),
                                    )
                                    .changed()
                                {
                                    self.compose_host_env =
                                        if override_on { Some(Vec::new()) } else { None };
                                }
                            });
                            if let Some(selected) = self.compose_host_env.as_mut() {
                                ui.horizontal_wrapped(|ui| {
                                    for nh in &pod_config.allow.host_env {
                                        let mut on = selected.iter().any(|n| n == &nh.name);
                                        if ui.checkbox(&mut on, &nh.name).changed() {
                                            if on {
                                                if !selected.iter().any(|n| n == &nh.name) {
                                                    selected.push(nh.name.clone());
                                                }
                                            } else {
                                                selected.retain(|n| n != &nh.name);
                                            }
                                        }
                                    }
                                });
                            }
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
                        if response.changed() {
                            self.last_input_change_at = Some(ui.input(|i| i.time));
                        }
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

        // (msg_index, seed_text) — paired with `selected` after the
        // render closure since render_item doesn't see thread_id.
        let mut fork_request: Option<(usize, String)> = None;
        // Sudo-banner decisions the user triggered this frame.
        // Collected here and dispatched after the central-panel closure
        // so the closure doesn't need `&mut self` for `send`.
        let mut sudo_decisions: Vec<(
            u64,
            whisper_agent_protocol::permission::SudoDecision,
            Option<String>,
        )> = Vec::new();
        // Pre-split borrows: the closure needs `&mut self.tasks` (via
        // get_mut), `&mut self.md_cache`, and read/write access to the
        // escalation state simultaneously. Destructuring up front lets
        // the closure capture each field independently instead of
        // re-borrowing through `&mut self`.
        let selected = self.selected.clone();
        let tasks = &mut self.tasks;
        let md_cache = &mut self.md_cache;
        let pending_sudos = &self.pending_sudos;
        let sudo_reject_drafts = &mut self.sudo_reject_drafts;
        egui::CentralPanel::default().show_inside(ui, |ui| match selected.clone() {
            None => {
                ui.vertical_centered(|ui| {
                    ui.add_space(60.0);
                    ui.label(
                        RichText::new(
                            "no thread selected — type a prompt below to create a new thread",
                        )
                        .color(Color32::from_gray(140)),
                    );
                });
            }
            Some(thread_id) => match tasks.get_mut(&thread_id) {
                None => {
                    ui.label(
                        RichText::new(format!("thread {thread_id} not found"))
                            .color(Color32::from_rgb(220, 120, 120)),
                    );
                }
                Some(view) => {
                    render_thread_context_inspector(ui, &thread_id, view);
                    render_failure_banner(ui, view);
                    render_sudo_banners(
                        ui,
                        &thread_id,
                        pending_sudos,
                        sudo_reject_drafts,
                        &mut sudo_decisions,
                    );
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
                            for (idx, item) in view.items.iter().enumerate() {
                                if let Some(event) = render_item(ui, md_cache, idx, item) {
                                    match event {
                                        ChatItemEvent::ForkRequested {
                                            msg_index,
                                            seed_text,
                                        } => {
                                            fork_request = Some((msg_index, seed_text));
                                        }
                                    }
                                }
                                ui.add_space(6.0);
                            }
                        }
                        // Transient prefill-progress bar. Only populated
                        // by llamacpp-backed threads while the backend
                        // is ingesting the prompt; cleared on first
                        // delta. Sits at the tail of the chat log so
                        // stick_to_bottom keeps it in view.
                        if let Some((processed, total)) = view.prefill_progress {
                            render_prefill_progress(ui, processed, total);
                        }
                    });
                }
            },
        });
        if let (Some((msg_index, seed_text)), Some(thread_id)) = (fork_request, selected.clone()) {
            // Archive-by-default: fork is almost always "I want to try
            // something different from here"; the original typically
            // becomes noise in the sidebar. User can untick.
            self.fork_modal = Some(ForkModalState {
                thread_id,
                from_message_index: msg_index,
                archive_original: true,
                reset_capabilities: false,
                seed_text,
            });
        }

        // Dispatch any sudo approve/reject the user clicked this frame.
        // Optimistically drop the entry from `pending_sudos` — the
        // server's `SudoResolved` broadcast will land later and be a
        // no-op. Keeping the banner mounted until the server echoed
        // back was too flickery on the approve path, where the parent
        // thread's next turn starts immediately on grant.
        for (function_id, decision, reason) in sudo_decisions {
            self.send(ClientToServer::ResolveSudo {
                function_id,
                decision,
                reason,
            });
            self.pending_sudos.remove(&function_id);
            self.sudo_reject_drafts.remove(&function_id);
        }

        let ctx = ui.ctx().clone();
        self.render_fork_modal(&ctx);
        self.render_new_pod_modal(&ctx);
        self.render_pod_editor_modal(&ctx);
        self.render_new_behavior_modal(&ctx);
        self.render_behavior_editor_modal(&ctx);
        self.render_file_tree_modal(&ctx);
        self.render_file_viewer_modal(&ctx);
        self.render_json_viewer_modal(&ctx);
        self.render_provider_editor_modal(&ctx);
        self.render_settings_modal(&ctx);

        // Draft debounce tick. Schedule a repaint at the deadline so
        // the flush fires even if nothing else is driving frames.
        if let Some(changed_at) = self.last_input_change_at {
            let now = ctx.input(|i| i.time);
            let elapsed_secs = (now - changed_at).max(0.0);
            let debounce_secs = DRAFT_DEBOUNCE.as_secs_f64();
            if elapsed_secs >= debounce_secs {
                self.flush_pending_draft();
            } else {
                ctx.request_repaint_after(Duration::from_secs_f64(debounce_secs - elapsed_secs));
            }
        }
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
    fn render_file_tree_modal(&mut self, ctx: &egui::Context) {
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

    /// Render the fork-from-here confirm dialog. On confirm, fires
    /// `ForkThread` with a correlation_id the `ThreadCreated`
    /// handler matches against `pending_fork_seed` to seed the new
    /// thread's draft.
    fn render_fork_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.fork_modal.take() else {
            return;
        };
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
            // The new thread id is minted server-side, so the seed
            // text has to ride the correlation_id into the
            // `ThreadCreated` handler rather than the `ForkThread`
            // payload itself.
            let correlation_id = self.next_correlation_id();
            self.pending_fork_seed = Some((correlation_id.clone(), modal.seed_text.clone()));
            self.send(ClientToServer::ForkThread {
                thread_id: modal.thread_id.clone(),
                from_message_index: modal.from_message_index,
                archive_original: modal.archive_original,
                reset_capabilities: modal.reset_capabilities,
                correlation_id: Some(correlation_id),
            });
        } else if cancel_clicked || !open {
            // Dropped.
        } else {
            self.fork_modal = Some(modal);
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

    /// Left-panel Providers tab. Lists registered host-env providers
    /// with origin + reachability badges and per-row Edit / Remove
    /// actions. "+ Add provider" at the top opens the add modal.
    /// Host-env provider catalog tab. The sandbox daemons threads
    /// can provision isolated MCP hosts against; distinct from the
    /// Shared MCP hosts tab (long-lived endpoints the operator
    /// points us at). Used to live in the left sidebar; moved here
    /// so server-wide settings are all in one place.
    fn render_settings_host_env_providers_tab(&mut self, ui: &mut egui::Ui) {
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
                self.provider_editor_modal = Some(ProviderEditorModalState::new_add());
            }
        });
        ui.add_space(4.0);

        if self.host_env_providers.is_empty() {
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

        // Snapshot the provider list before the row loop so Edit /
        // Remove actions can mutate `self` (modal state, correlation
        // map) without fighting a borrow against the live Vec.
        let providers = self.host_env_providers.clone();
        ScrollArea::vertical().show(ui, |ui| {
            for provider in &providers {
                render_provider_row(ui, provider, self);
                ui.add_space(2.0);
                ui.separator();
            }
        });
    }

    /// Server-settings modal opened from the top-bar cog. Hosts the
    /// "LLM backends" tab (list + Rotate-credentials for
    /// `chatgpt_subscription` backends) plus, on top of it, the paste-
    /// `auth.json` sub-form when a rotation is in progress. Mutation
    /// paths require an admin token; plain client tokens get a polite
    /// error from the server which surfaces in the banner.
    fn render_settings_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.settings_modal.take() else {
            return;
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
                    SettingsTab::Backends => self.render_settings_backends_tab(
                        ui,
                        &modal.codex_rotate_banner,
                        &mut rotate_request,
                    ),
                    SettingsTab::HostEnvProviders => {
                        self.render_settings_host_env_providers_tab(ui);
                    }
                    SettingsTab::SharedMcp => self.render_settings_shared_mcp_tab(
                        ui,
                        &modal.shared_mcp_banner,
                        &mut modal.shared_mcp_remove_armed,
                        &mut shared_mcp_add_request,
                        &mut shared_mcp_edit_request,
                        &mut shared_mcp_remove_request,
                    ),
                    SettingsTab::ServerConfig => {
                        self.render_settings_server_config_tab(ui, &mut modal.server_config);
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
            let correlation = self.next_correlation_id();
            self.send(ClientToServer::RemoveSharedMcpHost {
                correlation_id: Some(correlation),
                name,
            });
        }

        // Paste-auth.json sub-form. Rendered outside the main window so
        // egui stacks it on top; closing it returns to the list.
        let mut keep_main_open = true;
        if modal.codex_rotate.is_some() {
            self.render_codex_rotate_subform(ctx, &mut modal, &mut keep_main_open);
        }
        if modal.shared_mcp_editor.is_some() {
            self.render_shared_mcp_editor_subform(ctx, &mut modal);
        }

        if open && keep_main_open {
            self.settings_modal = Some(modal);
        }
    }

    /// Paste-auth.json sub-form. Stays open while a request is in
    /// flight (disables the Save button). Success / error banners live
    /// on the parent modal, not here — we close on either outcome.
    fn render_codex_rotate_subform(
        &mut self,
        ctx: &egui::Context,
        modal: &mut SettingsModalState,
        keep_main_open: &mut bool,
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
            let correlation = self.next_correlation_id();
            sub.pending_correlation = Some(correlation.clone());
            sub.error = None;
            let msg = ClientToServer::UpdateCodexAuth {
                correlation_id: Some(correlation),
                backend: sub.backend.clone(),
                contents: sub.contents.clone(),
            };
            self.send(msg);
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
        // we never request the main modal to close from here, we just
        // propagate the main `open` state unchanged.
        let _ = keep_main_open;
    }

    /// Backends list inside the server-settings modal. Renders one row
    /// per configured backend (name, kind, default-model, auth-mode)
    /// plus a Rotate button for `chatgpt_subscription` backends that
    /// opens the paste-auth.json sub-form. Credential material is
    /// never displayed.
    ///
    /// `rotate_request` is an out-parameter: when the user clicks a
    /// Rotate button we record the backend name here, and the caller
    /// transitions the sub-form on our behalf (we can't borrow
    /// `settings_modal` here because it's already held mutably).
    fn render_settings_backends_tab(
        &mut self,
        ui: &mut egui::Ui,
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

        if self.backends.is_empty() {
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

        let backends = self.backends.clone();
        ScrollArea::vertical().show(ui, |ui| {
            for b in &backends {
                render_backend_settings_row(ui, b, rotate_request);
                ui.add_space(2.0);
                ui.separator();
            }
        });
    }

    /// Server-config tab: raw TOML editor for `whisper-agent.toml`.
    /// Admin-only; the server rejects the fetch/update calls from
    /// non-admin connections and the error banner surfaces the
    /// refusal. Fetches on first open (lazy) and keeps its working
    /// draft across tab switches.
    fn render_settings_server_config_tab(
        &mut self,
        ui: &mut egui::Ui,
        state_slot: &mut Option<ServerConfigEditorState>,
    ) {
        // Lazy init: first open triggers a FetchServerConfig.
        if state_slot.is_none() {
            let corr = self.next_correlation_id();
            self.send(ClientToServer::FetchServerConfig {
                correlation_id: Some(corr.clone()),
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
                let corr = self.next_correlation_id();
                state.banner = None;
                state.save_correlation = Some(corr.clone());
                self.send(ClientToServer::UpdateServerConfig {
                    correlation_id: Some(corr),
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
    fn render_settings_shared_mcp_tab(
        &mut self,
        ui: &mut egui::Ui,
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

        if self.shared_mcp_hosts.is_empty() {
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

        let hosts = self.shared_mcp_hosts.clone();
        ScrollArea::vertical().show(ui, |ui| {
            for host in &hosts {
                render_shared_mcp_host_row(ui, host, remove_armed, edit_request, remove_request);
                ui.add_space(2.0);
                ui.separator();
            }
        });
    }

    /// Paste-bearer / edit-url sub-form for Shared MCP hosts.
    /// Dispatches `AddSharedMcpHost` or `UpdateSharedMcpHost` on Save
    /// and tracks the correlation so the matching response routes the
    /// form closed (success) or an inline error back into `sub.error`
    /// (failure). The sub-form sits on top of the main modal — egui
    /// stacks windows in open-order.
    fn render_shared_mcp_editor_subform(
        &mut self,
        ctx: &egui::Context,
        modal: &mut SettingsModalState,
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
                        let had_bearer =
                            matches!(sub.auth_kind_on_load, SharedMcpAuthPublic::Bearer);
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
                                    .hint_text(
                                        "optional; space-separated (defaults to AS metadata)",
                                    )
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
            let correlation = self.next_correlation_id();
            sub.pending_correlation = Some(correlation.clone());
            sub.error = None;
            let bearer = sub.bearer.trim().to_string();
            let scope = sub.oauth_scope.trim().to_string();
            let msg = match (sub.mode, sub.auth_choice) {
                (SharedMcpEditorMode::Add, SharedMcpAuthChoice::Oauth2) => {
                    // Grab the origin the webui is served from so the
                    // redirect_uri on the authorization URL matches
                    // what the browser will actually reach our /oauth/callback
                    // on.
                    let redirect_base = webui_origin();
                    ClientToServer::AddSharedMcpHost {
                        correlation_id: Some(correlation),
                        name: sub.name.trim().to_string(),
                        url: sub.url.trim().to_string(),
                        auth: SharedMcpAuthInput::Oauth2Start {
                            scope: if scope.is_empty() { None } else { Some(scope) },
                            redirect_base,
                        },
                    }
                }
                (SharedMcpEditorMode::Add, SharedMcpAuthChoice::Bearer) => {
                    ClientToServer::AddSharedMcpHost {
                        correlation_id: Some(correlation),
                        name: sub.name.trim().to_string(),
                        url: sub.url.trim().to_string(),
                        auth: SharedMcpAuthInput::Bearer { token: bearer },
                    }
                }
                (SharedMcpEditorMode::Add, SharedMcpAuthChoice::Anonymous) => {
                    ClientToServer::AddSharedMcpHost {
                        correlation_id: Some(correlation),
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
                                matches!(sub.auth_kind_on_load, SharedMcpAuthPublic::Bearer,);
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
                    ClientToServer::UpdateSharedMcpHost {
                        correlation_id: Some(correlation),
                        name: sub.name.clone(),
                        url: sub.url.trim().to_string(),
                        auth,
                    }
                }
            };
            self.send(msg);
        }

        if !open {
            let _ = sub;
        } else {
            modal.shared_mcp_editor = Some(sub);
        }
    }

    /// Add / edit provider modal. Dispatches `AddHostEnvProvider` or
    /// `UpdateHostEnvProvider` on Save, tracks the correlation so the
    /// matching response / error routes back here.
    fn render_provider_editor_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.provider_editor_modal.take() else {
            return;
        };
        let mut open = true;
        let mut save_clicked = false;
        let mut cancel_clicked = false;
        let saving = modal.pending_correlation.is_some();
        let title = match modal.mode {
            ProviderEditorMode::Add => "Add host-env provider",
            ProviderEditorMode::Edit => "Edit host-env provider",
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
                    let name_edit = TextEdit::singleline(&mut modal.name)
                        .hint_text("catalog name (e.g. 'landlock-laptop')")
                        .desired_width(f32::INFINITY);
                    // Edit mode: name is immutable — pod bindings
                    // reference providers by name, so renaming would
                    // dangle them. Rebuild by removing and re-adding
                    // if truly needed.
                    let editable = modal.mode == ProviderEditorMode::Add;
                    ui.add_enabled(editable, name_edit);
                });
                if modal.mode == ProviderEditorMode::Edit {
                    ui.label(
                        RichText::new(
                            "Name is fixed once set — pod bindings reference it. \
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
                        TextEdit::singleline(&mut modal.url)
                            .hint_text("http://host:port")
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("token");
                    ui.add(
                        TextEdit::singleline(&mut modal.token)
                            .password(true)
                            .hint_text(if modal.had_token {
                                "leave blank to clear existing token"
                            } else {
                                "control-plane bearer (optional)"
                            })
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.label(
                    RichText::new(
                        "The token must match the daemon's --control-token-file \
                         (or leave blank for a --no-auth dev daemon).",
                    )
                    .small()
                    .color(Color32::from_gray(150)),
                );
                if let Some(err) = &modal.error {
                    ui.add_space(6.0);
                    ui.colored_label(Color32::from_rgb(220, 80, 80), err);
                }
                ui.add_space(8.0);
                ui.separator();
                ui.horizontal(|ui| {
                    let save_enabled =
                        !saving && !modal.name.trim().is_empty() && !modal.url.trim().is_empty();
                    let label = if saving { "Saving…" } else { "Save" };
                    if ui
                        .add_enabled(save_enabled, egui::Button::new(label))
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
            modal.error = None;
            let correlation = self.next_correlation_id();
            modal.pending_correlation = Some(correlation.clone());
            let name = modal.name.trim().to_string();
            let url = modal.url.trim().to_string();
            let token_raw = modal.token.trim().to_string();
            let token = if token_raw.is_empty() {
                None
            } else {
                Some(token_raw)
            };
            match modal.mode {
                ProviderEditorMode::Add => {
                    self.send(ClientToServer::AddHostEnvProvider {
                        correlation_id: Some(correlation),
                        name,
                        url,
                        token,
                    });
                }
                ProviderEditorMode::Edit => {
                    self.send(ClientToServer::UpdateHostEnvProvider {
                        correlation_id: Some(correlation),
                        name,
                        url,
                        token,
                    });
                }
            }
            // Keep the modal open; it closes on the matching response
            // (HostEnvProviderAdded / HostEnvProviderUpdated) or shows
            // the server's error via the Error event handler.
            self.provider_editor_modal = Some(modal);
        } else if cancel_clicked || !open {
            // Modal closes.
        } else {
            self.provider_editor_modal = Some(modal);
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
                scope: Default::default(),
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

    /// Render the generic pod-file text editor. Single-textarea modal
    /// — no tabs, no sub-forms. Save/Revert/Close footer for
    /// writable files; a read-only notice replaces the footer for
    /// paths the server has flagged as runtime state. Content loads
    /// asynchronously (`ReadPodFile`); while in flight the body
    /// renders a "loading…" placeholder.
    /// Render the read-only JSON tree viewer. Scalars render as
    /// `key: value` one-liners; objects and arrays render as
    /// collapsible headers with their sizes in the label, default-open
    /// at the root and default-closed deeper. Strings are shown
    /// in-line with a preview and a hover tooltip carrying the full
    /// text, so long message content doesn't blow out the row height.
    fn render_json_viewer_modal(&mut self, ctx: &egui::Context) {
        let Some(modal) = self.json_viewer_modal.take() else {
            return;
        };

        let title = format!("{} — {}", modal.path, modal.pod_id);
        let screen = ctx.content_rect();
        let max_h = (screen.height() - 60.0).max(280.0);
        let max_w = (screen.width() - 60.0).max(420.0);
        let mut open = true;
        let mut close_clicked = false;

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
                egui::Panel::bottom("json_viewer_footer").show_inside(ui, |ui| {
                    ui.add_space(6.0);
                    if let Some(err) = modal.error.as_deref() {
                        ui.colored_label(Color32::from_rgb(220, 80, 80), err);
                        ui.add_space(4.0);
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("read-only JSON viewer")
                                .small()
                                .color(Color32::from_gray(160)),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Close").clicked() {
                                close_clicked = true;
                            }
                        });
                    });
                });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    if let Some(value) = modal.parsed.as_ref() {
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                render_json_node(ui, "$", "(root)", value, 0);
                            });
                    } else if modal.error.is_none() {
                        ui.add_space(24.0);
                        ui.label(
                            RichText::new("loading…")
                                .italics()
                                .color(Color32::from_gray(160)),
                        );
                    }
                });
            });

        if close_clicked || !open {
            return;
        }
        self.json_viewer_modal = Some(modal);
    }

    fn render_file_viewer_modal(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.file_viewer_modal.take() else {
            return;
        };

        let mut open = true;
        let mut save_clicked = false;
        let mut revert_clicked = false;
        let mut close_clicked = false;

        let title = format!("{} — {}", modal.path, modal.pod_id);
        let screen = ctx.content_rect();
        let max_h = (screen.height() - 60.0).max(280.0);
        let max_w = (screen.width() - 60.0).max(420.0);
        let dirty = modal.is_dirty();
        let has_data = modal.working.is_some();
        // "saving" iff we have content and a correlation is in flight
        // (a correlation-in-flight with no content yet = pending read).
        let saving = modal.pending_correlation.is_some() && has_data;

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
                egui::Panel::bottom("file_viewer_footer").show_inside(ui, |ui| {
                    if modal.readonly {
                        ui.add_space(6.0);
                        if let Some(err) = modal.error.as_deref() {
                            ui.colored_label(Color32::from_rgb(220, 80, 80), err);
                            ui.add_space(4.0);
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("read-only — runtime state owned by the scheduler")
                                    .small()
                                    .color(Color32::from_gray(160)),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("Close").clicked() {
                                        close_clicked = true;
                                    }
                                },
                            );
                        });
                    } else {
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
                    }
                });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    let Some(working) = modal.working.as_mut() else {
                        ui.add_space(24.0);
                        if modal.error.is_none() {
                            ui.label(
                                RichText::new("loading…")
                                    .italics()
                                    .color(Color32::from_gray(160)),
                            );
                        }
                        return;
                    };
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            let mut edit = TextEdit::multiline(working)
                                .code_editor()
                                .desired_width(ui.available_width());
                            if modal.readonly {
                                edit = edit.interactive(false);
                            }
                            ui.add_sized(
                                [ui.available_width(), ui.available_height().max(200.0)],
                                edit,
                            );
                        });
                });
            });

        if revert_clicked && let Some(b) = modal.baseline.clone() {
            modal.working = Some(b);
            modal.error = None;
        }

        if save_clicked
            && modal.is_dirty()
            && let Some(working) = modal.working.clone()
        {
            let correlation = self.next_correlation_id();
            modal.pending_correlation = Some(correlation.clone());
            modal.error = None;
            self.send(ClientToServer::WritePodFile {
                correlation_id: Some(correlation),
                pod_id: modal.pod_id.clone(),
                path: modal.path.clone(),
                content: working,
            });
        }

        if close_clicked || !open {
            // Modal was taken out at the top — dropping = closed.
            return;
        }
        self.file_viewer_modal = Some(modal);
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
        let pod_backend_names: Vec<String> = self
            .pod_configs
            .get(&modal.pod_id)
            .map(|cfg| cfg.allow.backends.clone())
            .unwrap_or_default();
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
                                        &models_by_backend,
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
        let mut backend_names: Vec<String> = self.backends.iter().map(|b| b.name.clone()).collect();
        backend_names.sort();
        let default_backend = backend_names.first().cloned().unwrap_or_default();
        PodConfig {
            name,
            description: None,
            created_at: String::new(),
            allow: PodAllow {
                backends: backend_names,
                mcp_hosts: Vec::new(),
                host_env: Vec::<NamedHostEnv>::new(),
                tools: AllowMap::allow_all(),
                caps: Default::default(),
            },
            thread_defaults: ThreadDefaults {
                backend: default_backend,
                model: String::new(),
                system_prompt_file: "system_prompt.md".into(),
                max_tokens: 16384,
                max_turns: 30,
                host_env: Vec::new(),
                mcp_hosts: Vec::new(),
                compaction: Default::default(),
                caps: Default::default(),
                tool_surface: Default::default(),
            },
            limits: PodLimits::default(),
        }
    }
}

/// One row in the Settings → LLM backends list. Shows name, kind,
/// default model, and auth-mode badge; for `chatgpt_subscription`
/// backends, a right-aligned "Rotate credentials" button that bubbles
/// the backend name up through `rotate_request` so the caller can open
/// the paste-`auth.json` sub-form. Never renders credential material.
fn render_backend_settings_row(
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
fn render_shared_mcp_host_row(
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
fn webui_origin() -> String {
    web_sys::window()
        .and_then(|w| w.location().origin().ok())
        .unwrap_or_default()
}
#[cfg(not(target_arch = "wasm32"))]
fn webui_origin() -> String {
    String::new()
}

/// Open `url` in a new browser tab. Used by the OAuth flow to hand
/// the user off to the authorization server. No-op on desktop —
/// native OAuth would route through the system browser via
/// a crate like `webbrowser`, which we'll add when desktop needs it.
#[cfg(target_arch = "wasm32")]
fn open_in_new_tab(url: &str) {
    if let Some(window) = web_sys::window() {
        let _ = window.open_with_url_and_target(url, "_blank");
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn open_in_new_tab(_url: &str) {}

/// True when OAuth flows are actually usable — the webui is in a
/// browser that can open a new tab + receive a redirect on the
/// server's `/oauth/callback` route. False on desktop (no browser
/// to drive the flow); the UI disables the OAuth radio option.
#[cfg(target_arch = "wasm32")]
const OAUTH_AVAILABLE: bool = true;
#[cfg(not(target_arch = "wasm32"))]
const OAUTH_AVAILABLE: bool = false;

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

/// One row in the Providers tab. Takes `&mut App` so Edit/Remove
/// clicks can mutate modal state and dispatch protocol messages. The
/// provider snapshot is read from a clone so this function doesn't
/// hold a borrow against the live `host_env_providers` Vec.
fn render_provider_row(ui: &mut egui::Ui, info: &HostEnvProviderInfo, app: &mut ChatApp) {
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
    // Clone the relevant bits out so the closures below can re-borrow
    // `app` mutably without fighting a lingering immutable borrow.
    let pending_error = app
        .provider_remove_pending
        .get(&info.name)
        .and_then(|p| p.error.clone());
    let removing = app
        .provider_remove_pending
        .get(&info.name)
        .is_some_and(|p| p.error.is_none());
    let armed = app.provider_remove_armed.contains(&info.name);

    ui.horizontal(|ui| {
        let edit_btn = ui.add_enabled(!is_overlay && !removing, egui::Button::new("Edit"));
        if edit_btn.clicked() {
            app.provider_editor_modal = Some(ProviderEditorModalState::new_edit(info));
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
            if armed {
                let correlation = app.next_correlation_id();
                app.provider_remove_armed.remove(&info.name);
                app.provider_remove_pending.insert(
                    info.name.clone(),
                    ProviderRemovePending {
                        correlation: correlation.clone(),
                        error: None,
                    },
                );
                app.send(ClientToServer::RemoveHostEnvProvider {
                    correlation_id: Some(correlation),
                    name: info.name.clone(),
                });
            } else {
                app.provider_remove_armed.insert(info.name.clone());
            }
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
}

fn provider_origin_chip(origin: HostEnvProviderOrigin) -> (&'static str, Color32) {
    match origin {
        HostEnvProviderOrigin::Seeded => ("seeded", Color32::from_rgb(140, 160, 200)),
        HostEnvProviderOrigin::Manual => ("manual", Color32::from_rgb(140, 200, 160)),
        HostEnvProviderOrigin::RuntimeOverlay => ("cli-overlay", Color32::from_rgb(200, 180, 120)),
    }
}

fn provider_reachability_chip(r: &HostEnvReachability) -> (&'static str, Color32, Option<String>) {
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

fn resource_state_chip(state: ResourceStateLabel) -> (&'static str, Color32) {
    match state {
        ResourceStateLabel::Provisioning => ("provisioning", Color32::from_rgb(180, 160, 90)),
        ResourceStateLabel::Ready => ("ready", Color32::from_rgb(120, 180, 120)),
        ResourceStateLabel::Errored => ("errored", Color32::from_rgb(200, 110, 110)),
        ResourceStateLabel::Lost => ("lost", Color32::from_rgb(210, 140, 90)),
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
fn render_sudo_banners(
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

/// Draw the transient prefill-progress indicator. Shows only while a
/// llamacpp-backed turn is ingesting its prompt, cleared on first delta
/// by the [`ChatApp::handle_wire`] reducers. Formatted like
/// `prefilling 3,200 / 15,000 tokens · 21%` so the user can see both
/// the absolute numbers (helps for "how big is my context?") and the
/// fraction (helps for "how much longer?"). The bar itself carries no
/// text because egui's built-in text rendering inside the bar clashes
/// with the explicit label we already draw above it.
fn render_prefill_progress(ui: &mut egui::Ui, processed: u32, total: u32) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::{
        ContentBlock, Conversation, Message, ThreadStateLabel, ToolResultContent, Usage,
    };

    fn summary_stub() -> ThreadSummary {
        ThreadSummary {
            thread_id: "t".into(),
            pod_id: "p".into(),
            title: None,
            state: ThreadStateLabel::Idle,
            created_at: String::new(),
            last_active: String::new(),
            origin: None,
            continued_from: None,
            dispatched_by: None,
        }
    }

    /// Walk a `TaskView` through the event stream the server emits for
    /// an assistant turn with N sequential tool calls and verify that
    /// `conv_message_count` lines up with the server's authoritative
    /// `Conversation::len()` after each event. The underlying bug:
    /// `ThreadToolCallEnd` doesn't trigger a bump, but the server pushes
    /// a single `Role::ToolResult` message when all calls resolve, so
    /// every tool-heavy turn used to drift the client by +1 and
    /// `DisplayItem::User.msg_index` stamped during streaming ended up
    /// below the real server index (the fork_thread breakage).
    #[test]
    fn conv_message_count_tracks_server_across_sync_tool_batch() {
        let mut view = TaskView::new(summary_stub());
        // Simulate a snapshot landing with setup prefix + first user
        // message: [System, Tools, System(listing), User] → len 4.
        view.conv_message_count = 4;

        // Turn 1: assistant issues 3 sequential tool calls.
        view.flush_pending_tool_batch(); // AssistantBegin: no-op.
        // ... streaming deltas, no counter touch ...
        // AssistantEnd bumps +1 for the assistant message.
        view.flush_pending_tool_batch();
        view.conv_message_count += 1;
        assert_eq!(view.conv_message_count, 5);

        // Three tool calls, each Begin → End. Arm-then-flush happens on
        // Begin only; End keeps the flag armed.
        for _ in 0..3 {
            view.pending_tool_batch = true; // ToolCallBegin
            // ToolCallEnd emits no counter change.
        }
        // Turn 2 starts: AssistantBegin → flush fires once.
        view.flush_pending_tool_batch();
        assert_eq!(
            view.conv_message_count, 6,
            "batched tool_result append should bump count by exactly 1"
        );
        assert!(!view.pending_tool_batch);

        // Assistant of turn 2 — text-only, no tools.
        view.conv_message_count += 1; // AssistantEnd.
        assert_eq!(view.conv_message_count, 7);
        // User replies ("go ahead"). Handler reads `msg_index = count`
        // *before* bumping; msg_index is now correct at 7 (server index).
        let msg_index_for_user = view.conv_message_count;
        view.conv_message_count += 1;
        assert_eq!(msg_index_for_user, 7);
        assert_eq!(view.conv_message_count, 8);
    }

    #[test]
    fn flush_dispatcher_skips_tool_events_and_snapshot() {
        let begin = ServerToClient::ThreadToolCallBegin {
            thread_id: "t".into(),
            tool_use_id: "tu-1".into(),
            name: "bash".into(),
            args_preview: String::new(),
            args: None,
        };
        let end = ServerToClient::ThreadToolCallEnd {
            thread_id: "t".into(),
            tool_use_id: "tu-1".into(),
            result_preview: String::new(),
            is_error: false,
        };
        let assistant_end = ServerToClient::ThreadAssistantEnd {
            thread_id: "t".into(),
            stop_reason: Some("tool_use".into()),
            usage: Usage::default(),
        };
        assert!(pending_tool_batch_flush_thread_id(&begin).is_none());
        assert!(pending_tool_batch_flush_thread_id(&end).is_none());
        assert_eq!(
            pending_tool_batch_flush_thread_id(&assistant_end),
            Some("t"),
        );
    }

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
                replay: None,
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
    fn snapshot_rebuild_fuses_sync_result_into_tool_call() {
        let items = conversation_to_items(&conv_with_tool_call(), &TurnLog::default());
        // The matching sync tool_result should fuse into the ToolCall
        // (populate its `result` slot) rather than producing a
        // standalone ToolResult row.
        let fused = items
            .iter()
            .find_map(|i| match i {
                DisplayItem::ToolCall {
                    tool_use_id,
                    result,
                    ..
                } if tool_use_id == "tu-1" => result.as_ref().map(|r| (r.text.clone(), r.is_error)),
                _ => None,
            })
            .expect("ToolCall with fused result should be present");
        assert_eq!(fused.0, "file1\nfile2\n");
        assert!(!fused.1);
        let standalone = items.iter().any(
            |i| matches!(i, DisplayItem::ToolResult { tool_use_id, .. } if tool_use_id == "tu-1"),
        );
        assert!(
            !standalone,
            "sync result should not also appear as a standalone ToolResult row"
        );
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
        let items = conversation_to_items(&conv, &TurnLog::default());
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
                DisplayItem::ToolCallStreaming { .. } => "tool_call_streaming",
                DisplayItem::ToolResult { .. } => "tool_result",
                DisplayItem::SystemNote { .. } => "system_note",
                DisplayItem::SetupPrompt { .. } => "setup_prompt",
                DisplayItem::SetupTools { .. } => "setup_tools",
                DisplayItem::TurnStats { .. } => "turn_stats",
            })
            .collect();
        assert!(
            tool_call_count > 0,
            "expected at least one ToolCall item, got items={roles_debug:?}"
        );
        // Every ToolCall should have a fused result (the 4 sync
        // tool_result messages land on their originating calls
        // directly via the fusion walk).
        for item in &items {
            if let DisplayItem::ToolCall {
                tool_use_id,
                result,
                ..
            } = item
            {
                assert!(
                    result.is_some(),
                    "expected fused result on ToolCall {tool_use_id}"
                );
            }
        }
        // The 2 async XML callbacks (msg 10/12) land after an
        // intervening assistant turn, so they push standalone
        // ToolResult rows.
        let standalone_count = items
            .iter()
            .filter(|i| matches!(i, DisplayItem::ToolResult { .. }))
            .count();
        assert_eq!(
            standalone_count, 2,
            "expected 2 standalone ToolResult rows for the async callbacks"
        );
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
            replay: None,
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
        let items = conversation_to_items(&conv, &TurnLog::default());
        // The initial sync ack fuses into the ToolCall (no intervening
        // turn at the time it lands); the async XML callback arrives
        // after an assistant turn and pushes a standalone row.
        let fused = items
            .iter()
            .find_map(|i| match i {
                DisplayItem::ToolCall {
                    tool_use_id,
                    result,
                    ..
                } if tool_use_id == "tu-async" => result.as_ref().map(|r| r.text.clone()),
                _ => None,
            })
            .expect("ToolCall should have the sync ack fused onto it");
        assert_eq!(fused, "Dispatched task-X");
        let standalone: Vec<String> = items
            .iter()
            .filter_map(|i| match i {
                DisplayItem::ToolResult {
                    tool_use_id, text, ..
                } if tool_use_id == "tu-async" => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            standalone.len(),
            1,
            "expected only the async callback as a standalone row"
        );
        assert_eq!(standalone[0], "the real final answer with <code> in it");
    }
}
