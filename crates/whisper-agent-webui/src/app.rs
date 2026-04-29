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
mod conversion;
mod editor_render;
mod modals;
mod sidebar;
mod widgets;
mod wire_handler;

use self::chat_render::{ChatItemEvent, render_item};
use self::modals::{
    BehaviorEditorEvent, BucketsEvent, BuildProgressView, FileViewerEvent, ForkEvent,
    NewBehaviorEvent, NewPodEvent, PodEditorEvent, ProviderEditorEvent, SettingsEvent,
    render_behavior_editor_modal, render_buckets_modal, render_file_viewer_modal,
    render_fork_modal, render_image_lightbox_modal, render_json_viewer_modal,
    render_new_behavior_modal, render_new_pod_modal, render_pod_editor_modal,
    render_provider_editor_modal, render_settings_modal,
};
use self::widgets::{
    render_failure_banner, render_prefill_progress, render_resource_list, render_sudo_banners,
    render_thread_context_inspector, state_chip,
};

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::time::Duration;

use egui::{Color32, ComboBox, RichText, ScrollArea, TextEdit};
use egui_commonmark::CommonMarkCache;
use whisper_agent_protocol::sandbox::NetworkPolicy;
use whisper_agent_protocol::{
    AllowMap, Attachment, BackendSummary, BehaviorConfig, BehaviorOrigin, BehaviorSummary,
    BucketSummary, ClientToServer, EmbeddingProviderInfo, FsEntry, FunctionKind, FunctionSummary,
    HostEnvProviderInfo, HostEnvSpec, ImageMime, ImageSource, ModelSummary, NamedHostEnv, PodAllow,
    PodConfig, PodLimits, PodSummary, ResourceSnapshot, ServerToClient, SharedMcpAuthPublic,
    SharedMcpHostInfo, ThreadBindings, ThreadBindingsRequest, ThreadConfigOverride, ThreadDefaults,
    ThreadStateLabel, ThreadSummary, Usage,
};

/// Brand icon shown in the top-bar header. Embedded at compile time so
/// the egui canvas doesn't pay an HTTP round-trip for the mark on first
/// paint; the same PNG is also served as the PWA / apple-touch-icon
/// favicon, so the visual identity is consistent across browser-tab,
/// PWA tile, and in-app header. Refresh by editing
/// `assets/icon/icon_v1.svg` and re-running `assets/icon/generate.py`.
const APP_ICON_PNG: &[u8] = include_bytes!("../assets/favicon-192.png");
/// Stable URI under which the icon bytes get registered with egui's
/// loader chain. Fixed (not content-hashed like the user-side image
/// strip) because there's exactly one app icon, and a stable URI keeps
/// egui's texture cache hit on every frame.
const APP_ICON_URI: &str = "bytes://app-icon-v1";

/// Raw bytes picked up by the compose area before any MIME sniff or
/// capability check has run. Drop-targeted and file-picker-targeted
/// inputs both produce this shape so the downstream staging pipeline
/// is one function, not two. `source_desc` feeds the user-facing hint
/// on rejection so the reason can mention the filename.
pub(crate) struct RawPick {
    pub(crate) bytes: Vec<u8>,
    pub(crate) source_desc: String,
}

/// Cloneable handle for pushing attachments into `ChatApp` from code
/// that doesn't hold `&mut ChatApp`. Used by the wasm drop handlers
/// installed on `document.body` — those run inside JS event callbacks
/// long after `ChatApp::new` has returned, and need a thread-safe way
/// to enqueue picks that also wakes egui so the next frame drains the
/// queue instead of waiting for unrelated user input.
#[derive(Clone)]
pub struct AttachmentIngress {
    queue: std::sync::Arc<std::sync::Mutex<Vec<RawPick>>>,
    ctx: egui::Context,
}

impl AttachmentIngress {
    pub fn push(&self, bytes: Vec<u8>, source_desc: String) {
        if let Ok(mut q) = self.queue.lock() {
            q.push(RawPick { bytes, source_desc });
        }
        self.ctx.request_repaint();
    }
}

/// Tri-state outcome of checking whether the currently-selected model
/// accepts image input. "Unknown" (catalog hasn't replied yet, or the
/// picker model isn't in the loaded list) is treated as permissive at
/// stage-time — the server gets the final say, and a rejected attempt
/// after the bytes are on the wire shows a clearer error than a
/// silently-dropped paste. Known "no" is the only hard reject.
enum ImageAcceptance {
    Yes,
    No,
    Unknown,
}

/// Image attachment staged in the compose area but not yet sent.
/// Holds the lowered protocol `Attachment` plus a unique id so the
/// thumbnail `Image` widget's cache key is stable across frames —
/// egui's image loader keys by URI, and a duplicate pixel payload
/// would otherwise collide on its bytes-hash.
struct StagedAttachment {
    id: u64,
    attachment: Attachment,
}

impl StagedAttachment {
    /// The `bytes://` URI the thumbnail renders through. Unique by id
    /// so adding the same image twice doesn't share a single texture
    /// (users can stage two copies deliberately).
    fn thumbnail_uri(&self) -> String {
        format!("bytes://compose-attachment-{}", self.id)
    }

    /// Raw bytes egui feeds into the loader for the URI above.
    /// URL-source attachments have no bytes yet — a placeholder slot
    /// stays for them until fetch-and-inline lands.
    fn thumbnail_bytes(&self) -> Option<&[u8]> {
        match &self.attachment {
            Attachment::Image {
                source: ImageSource::Bytes { data, .. },
            } => Some(data.as_slice()),
            Attachment::Image {
                source: ImageSource::Url { .. },
            } => None,
        }
    }
}

/// Sniff a `Content-Type` equivalent from the first few bytes of a
/// dropped file. Avoids trusting filename extensions (images dropped
/// from a screenshot tool often arrive without one) and keeps the
/// MIME set tight — only our `ImageMime` variants are accepted,
/// anything else becomes `None` and the drop is rejected at the
/// compose-area boundary with a visible toast.
fn sniff_image_mime(bytes: &[u8]) -> Option<ImageMime> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(ImageMime::Png)
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some(ImageMime::Jpeg)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some(ImageMime::Gif)
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some(ImageMime::Webp)
    } else if bytes.len() >= 12
        && &bytes[4..8] == b"ftyp"
        && matches!(
            &bytes[8..12],
            b"heic" | b"heix" | b"hevc" | b"hevx" | b"heim" | b"heis"
        )
    {
        Some(ImageMime::Heic)
    } else if bytes.len() >= 12
        && &bytes[4..8] == b"ftyp"
        && matches!(&bytes[8..12], b"mif1" | b"msf1")
    {
        Some(ImageMime::Heif)
    } else {
        None
    }
}

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
/// Accent fill for the primary compose-action button (Send).
const SEND_BUTTON_COLOR: Color32 = Color32::from_rgb(80, 140, 220);

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
        /// Image attachments that travelled with this user message,
        /// in the order the compose pushed them onto the content
        /// blocks. Empty for text-only messages and for
        /// server-injected user content (behavior-trigger prompts,
        /// compaction seeds). Rendered as an inline thumbnail strip
        /// under the message text.
        attachments: Vec<Attachment>,
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
        /// Image sources embedded in the tool result. Rendered as
        /// thumbnails under the expanded body. Empty for text-only
        /// tool results (the common case).
        attachments: Vec<ImageSource>,
    },
    /// Model-emitted image attachment — rendered as a standalone row
    /// in chronological order alongside `AssistantText` and `Reasoning`
    /// so a turn that interleaves text + images displays the way the
    /// model produced it. Sourced from `ServerToClient::ThreadAssistantImage`
    /// during streaming and from `ContentBlock::Image` on
    /// `Role::Assistant` during snapshot rebuilds.
    AssistantImage {
        source: ImageSource,
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
    /// Image sources from the tool result, rendered as thumbnails
    /// under the expanded `ToolCall` body. Empty for text-only results.
    pub attachments: Vec<ImageSource>,
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
    /// Image attachments staged for the next send. Populated by the
    /// compose area's drop handler and file-picker button, rendered
    /// as thumbnails below the text input, and drained into the
    /// outbound `SendUserMessage` / `CreateThread` when the user
    /// submits. Kept on `ChatApp` (not per-`TaskView`) because drafts
    /// cross thread boundaries — the staged attachments follow
    /// whatever the user ends up sending on.
    compose_attachments: Vec<StagedAttachment>,
    /// Monotonic counter for the stable id stamped on each staged
    /// attachment so the thumbnail `Image` widget's cache key survives
    /// reorder / removal without colliding on pixel-identical uploads.
    next_attachment_id: u64,
    /// Async-picker handoff queue. The file-picker runs off the UI
    /// thread (native: tokio-driven via a background thread; wasm:
    /// `wasm-bindgen-futures::spawn_local`); when a pick resolves, it
    /// pushes a `RawPick` onto this queue and the UI drains it each
    /// frame via the same staging pipeline that handles drops. Behind
    /// an `Arc<Mutex>` rather than `Rc<RefCell>` because the native
    /// picker thread isn't the UI thread.
    pending_picks: std::sync::Arc<std::sync::Mutex<Vec<RawPick>>>,
    /// Ephemeral status line rendered under the compose thumbnails.
    /// `(message, expires_at_secs)` where the seconds reference egui's
    /// frame time (`input.time`). Used to surface feedback on
    /// attach-attempts that silently failed before — model capability
    /// mismatches, unsupported MIME, read errors. Cleared automatically
    /// once frame time passes `expires_at_secs`.
    compose_hint: Option<(String, f64)>,
    inbound: Inbound,
    send_fn: SendFn,
    list_requested: bool,

    // --- Model-backend catalog ---
    backends: Vec<BackendSummary>,
    backends_requested: bool,
    /// Cached list of `[embedding_providers.*]` entries from the
    /// server. Populated on first `ListEmbeddingProviders` reply and
    /// kept until the connection drops. Drives the bucket-creation
    /// form's `embedder` dropdown so users don't have to remember
    /// the name.
    embedding_providers: Vec<EmbeddingProviderInfo>,
    /// Dedup flag — only one outbound `ListEmbeddingProviders` per
    /// session. Embedding-provider config can change with a server
    /// restart (catalog reload), but mid-session edits aren't
    /// supported anywhere; refreshing per-modal-open is overkill.
    embedding_providers_requested: bool,
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
    /// Open state for the knowledge-buckets modal (toggled from the
    /// 🗃 button in the top bar). `None` = closed.
    buckets_modal: Option<BucketsModalState>,
    /// Server's knowledge-bucket registry snapshot. Populated lazily
    /// on first `ListBuckets` round-trip (triggered when the buckets
    /// modal opens). Sorted by id; an empty list means the server has
    /// no buckets configured.
    buckets: Vec<BucketSummary>,
    /// `true` once we've sent `ListBuckets` for this connection — keeps
    /// the modal-open path from re-firing the request every frame.
    buckets_requested: bool,
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

/// State for the knowledge-buckets management modal.
///
/// Slice 9 added the search surface: `selected_bucket` /
/// `query_input` / `top_k` drive the form, and `query_status` carries
/// the lifecycle of an in-flight or completed query. Slice 8c added
/// `creating` (the +New bucket form), `delete_armed` (two-click delete
/// state), `build_progress` (live counters per in-flight build) and
/// `build_errors` (last-failed-build message per bucket).
#[derive(Default)]
struct BucketsModalState {
    selected_bucket: Option<String>,
    query_input: String,
    top_k: u32,
    query_status: QueryStatus,
    /// Correlation id of the in-flight query, when one exists. Used
    /// to ignore stale `QueryResults` that arrive after the user has
    /// fired a follow-up search.
    pending_query_correlation: Option<String>,
    /// `Some` when the +New bucket form is open; `None` when collapsed.
    creating: Option<CreateBucketForm>,
    /// Bucket id whose Delete button is "armed" — clicked once, waiting
    /// for a confirming second click. Cleared whenever the user starts
    /// a different action.
    delete_armed: Option<String>,
    /// Live build progress, keyed by `(pod_id, bucket_id)`. Inserted
    /// on `BucketBuildStarted`, updated on `BucketBuildProgress`,
    /// removed on `BucketBuildEnded`. The key carries pod scope so a
    /// pod-scope build's progress doesn't land in a same-named
    /// server-scope row (and vice-versa) — same-named buckets across
    /// scopes can coexist after PB3b.
    build_progress: HashMap<BucketRowKey, BuildProgressView>,
    /// Sticky last-failed-build error message, keyed by `(pod_id,
    /// bucket_id)`. Cleared when a successful build for the same
    /// bucket lands or the user dismisses it (no UI for that yet —
    /// it gets overwritten by the next attempt).
    build_errors: HashMap<BucketRowKey, String>,
}

/// `(pod_id, bucket_id)` pair used to key per-bucket UI state across
/// the WebUI's bucket modal. Tuple rather than a named struct because
/// it composes naturally with `HashMap` and the two callers (modal +
/// wire-handler) construct it in only a handful of places.
type BucketRowKey = (Option<String>, String);

impl BucketsModalState {
    fn fresh() -> Self {
        Self {
            top_k: 10,
            ..Default::default()
        }
    }
}

/// In-progress new-bucket form state. Lives inside
/// `BucketsModalState::creating` so it survives re-render frames.
#[derive(Clone)]
struct CreateBucketForm {
    id: String,
    name: String,
    description: String,
    embedder: String,
    /// Owning scope: `None` ⇒ server-scope (`<buckets_root>/<id>/`),
    /// `Some(pod_id)` ⇒ pod-scope (`<pods_root>/<pod_id>/buckets/<id>/`).
    /// Selected from a dropdown of known pods plus a "(server)" entry.
    pod_id: Option<String>,
    source_kind: SourceKindChoice,
    /// Holds either archive_path (stored) or path (linked); ignored
    /// for managed and tracked.
    source_detail: String,
    /// Selected driver for `kind=tracked`. Ignored for other source
    /// kinds. Today only `Wikipedia` exists; future drivers add new
    /// variants and gate which sub-fields the form surfaces.
    tracked_driver: TrackedDriverChoice,
    /// `kind=tracked, driver=wikipedia` — language code (`"en"`,
    /// `"simple"`, `"de"`, …). Empty default: the user must fill it
    /// before submitting.
    tracked_language: String,
    /// `kind=tracked, driver=wikipedia` — optional mirror override.
    /// Empty ⇒ defaults to `https://dumps.wikimedia.org` server-side.
    tracked_mirror: String,
    /// `kind=tracked` — how often the feed worker polls for deltas.
    tracked_delta_cadence: TrackedCadenceChoice,
    /// `kind=tracked` — how often the background resync rebuilds
    /// against a fresh base snapshot. Wikipedia publishes monthly.
    tracked_resync_cadence: TrackedCadenceChoice,
    chunk_tokens: u32,
    overlap_tokens: u32,
    dense_enabled: bool,
    sparse_enabled: bool,
    /// Vectors-bin quantization. f32 default keeps the historical
    /// behavior; f16/int8 trade recall for disk + RAM footprint.
    quantization: QuantizationChoice,
    /// `Some` while a `CreateBucket` is in flight; the renderer hides
    /// the Create button + shows a spinner. The empty-string sentinel
    /// is set by the renderer on click; the parent overwrites with
    /// the real correlation id after dispatching.
    pending_correlation: Option<String>,
    /// Inline error message from a failed create attempt. Sticky
    /// until the next click on Create.
    error: Option<String>,
}

impl Default for CreateBucketForm {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            description: String::new(),
            embedder: String::new(),
            pod_id: None,
            source_kind: SourceKindChoice::Linked,
            source_detail: String::new(),
            tracked_driver: TrackedDriverChoice::default(),
            tracked_language: String::new(),
            tracked_mirror: String::new(),
            tracked_delta_cadence: TrackedCadenceChoice::Daily,
            tracked_resync_cadence: TrackedCadenceChoice::Monthly,
            chunk_tokens: 500,
            overlap_tokens: 50,
            dense_enabled: true,
            sparse_enabled: true,
            quantization: QuantizationChoice::default(),
            pending_correlation: None,
            error: None,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Default)]
enum SourceKindChoice {
    Stored,
    #[default]
    Linked,
    Managed,
    Tracked,
}

/// Wire-shape mirror of `whisper_agent_protocol::TrackedDriverInput`,
/// kept on the form so the user's selection survives re-render
/// frames. The form's driver ComboBox writes into this and
/// `build_create_input` translates back to the wire type.
#[derive(Copy, Clone, Eq, PartialEq, Default)]
enum TrackedDriverChoice {
    #[default]
    Wikipedia,
}

/// UI mirror of `whisper_agent_protocol::TrackedCadenceInput`. Same
/// vocabulary used for both `delta_cadence` and `resync_cadence` —
/// the meaningful values for each are driver-dependent (Wikipedia
/// publishes daily incrementals + monthly bases) but the enum is
/// shared.
#[derive(Copy, Clone, Eq, PartialEq, Default)]
enum TrackedCadenceChoice {
    #[default]
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Manual,
}

impl TrackedCadenceChoice {
    fn label(self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Weekly => "weekly",
            Self::Monthly => "monthly",
            Self::Quarterly => "quarterly",
            Self::Manual => "manual",
        }
    }
}

/// UI mirror of `whisper_agent_protocol::QuantizationInput`. Frozen into
/// the new slot's manifest at build time, so the choice can't be
/// changed without rebuilding the bucket.
#[derive(Copy, Clone, Eq, PartialEq, Default)]
pub(crate) enum QuantizationChoice {
    #[default]
    F32,
    F16,
    Int8,
}

/// Lifecycle of the search form's last-issued query. Renderer reads
/// this to decide between hint / spinner / results / error display.
#[derive(Default, Clone)]
enum QueryStatus {
    #[default]
    Idle,
    InFlight {
        query: String,
    },
    Results {
        query: String,
        hits: Vec<whisper_agent_protocol::QueryHit>,
        /// chunk_ids the user has clicked open. Keyed by chunk_id (not
        /// hit index) so the expanded set survives re-render and stays
        /// stable across hits arriving in different orders. Cleared
        /// implicitly when a fresh `Results` lands.
        expanded: HashSet<String>,
    },
    Error {
        message: String,
    },
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
    /// Obtain a cloneable handle for enqueuing attachments from
    /// outside the egui event loop. Wasm uses this to install JS-level
    /// drop handlers on `document.body` so dropped files don't depend
    /// on eframe's winit-web DnD plumbing (which was observed to
    /// deliver nothing in the browser).
    pub fn attachment_ingress(&self, ctx: egui::Context) -> AttachmentIngress {
        AttachmentIngress {
            queue: self.pending_picks.clone(),
            ctx,
        }
    }

    pub fn new(inbound: Inbound, send_fn: SendFn) -> Self {
        Self {
            conn_status: ConnectionStatus::Connecting,
            conn_detail: None,
            tasks: HashMap::new(),
            task_order: Vec::new(),
            selected: None,
            composing_new: true,
            input: String::new(),
            compose_attachments: Vec::new(),
            next_attachment_id: 1,
            pending_picks: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            compose_hint: None,
            inbound,
            send_fn,
            list_requested: false,
            backends: Vec::new(),
            backends_requested: false,
            embedding_providers: Vec::new(),
            embedding_providers_requested: false,
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
            buckets_modal: None,
            buckets: Vec::new(),
            buckets_requested: false,
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

    /// Tri-state capability check. Hard-reject only when the picker
    /// model is in the loaded catalog AND its `capabilities.input.image`
    /// is empty. "Unknown" (catalog not loaded yet, or picker model
    /// isn't a known entry) is permissive — the server gets the final
    /// say, and a post-send error is clearer than a silently-dropped
    /// attachment during the catalog-fetch race on startup.
    fn current_model_image_acceptance(&self) -> ImageAcceptance {
        let model_id = self.effective_picker_model();
        if model_id.is_empty() {
            return ImageAcceptance::Unknown;
        }
        let backend = self.effective_picker_backend();
        let Some(summary) = self
            .models_by_backend
            .get(backend)
            .and_then(|list| list.iter().find(|m| m.id == model_id))
        else {
            return ImageAcceptance::Unknown;
        };
        if summary.capabilities.input.image.is_empty() {
            ImageAcceptance::No
        } else {
            ImageAcceptance::Yes
        }
    }

    /// Set a transient status line under the compose thumbnails. The
    /// hint self-dismisses after ~4 s of egui frame time; a new hint
    /// replaces any existing one so the latest feedback always wins.
    fn set_compose_hint(&mut self, ctx: &egui::Context, message: impl Into<String>) {
        let now = ctx.input(|i| i.time);
        self.compose_hint = Some((message.into(), now + 4.0));
    }

    /// Single entry point for staging a raw byte-payload as an
    /// attachment. Used by both drag-drop and the file-picker button
    /// so the MIME-sniff / capability-check / hint-feedback paths are
    /// identical across input sources. Surfacing a hint on every
    /// outcome (success included) removes the "did anything happen?"
    /// ambiguity the prior drop-silently-rejected implementation had.
    fn stage_raw_pick(&mut self, pick: RawPick, ctx: &egui::Context) {
        if pick.bytes.is_empty() {
            self.set_compose_hint(ctx, format!("{} was empty; ignored", pick.source_desc));
            return;
        }
        let Some(mime) = sniff_image_mime(&pick.bytes) else {
            self.set_compose_hint(
                ctx,
                format!(
                    "{} doesn't look like a supported image (jpeg / png / webp / gif)",
                    pick.source_desc
                ),
            );
            return;
        };
        match self.current_model_image_acceptance() {
            ImageAcceptance::No => {
                self.set_compose_hint(
                    ctx,
                    format!(
                        "{} rejected — the selected model doesn't accept image input",
                        pick.source_desc
                    ),
                );
                return;
            }
            ImageAcceptance::Unknown | ImageAcceptance::Yes => {}
        }
        let id = self.next_attachment_id;
        self.next_attachment_id += 1;
        self.compose_attachments.push(StagedAttachment {
            id,
            attachment: Attachment::Image {
                source: ImageSource::Bytes {
                    media_type: mime,
                    data: pick.bytes,
                },
            },
        });
        self.set_compose_hint(
            ctx,
            format!("attached {} as {}", pick.source_desc, mime.as_mime_str()),
        );
    }

    /// Spawn the OS/browser file picker. Native runs rfd's
    /// `AsyncFileDialog` on a background thread driven by a
    /// single-threaded tokio runtime; wasm runs it via
    /// `wasm_bindgen_futures::spawn_local`. Both paths push the
    /// resolved bytes onto `pending_picks`, where the per-frame
    /// drain in `update()` feeds them through `stage_raw_pick` (same
    /// pipeline as drag-drop so the two diagnostics don't drift).
    fn spawn_file_picker(&self, ctx: &egui::Context) {
        let queue = self.pending_picks.clone();
        let ctx_for_wake = ctx.clone();
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    log::error!("failed to build tokio current-thread runtime for file picker");
                    return;
                };
                rt.block_on(async move {
                    let Some(handle) = rfd::AsyncFileDialog::new()
                        .set_title("Attach image")
                        .add_filter("Image", &["png", "jpg", "jpeg", "webp", "gif"])
                        .pick_file()
                        .await
                    else {
                        return;
                    };
                    let name = handle.file_name();
                    let bytes = handle.read().await;
                    if let Ok(mut guard) = queue.lock() {
                        guard.push(RawPick {
                            bytes,
                            source_desc: name,
                        });
                    }
                    ctx_for_wake.request_repaint();
                });
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                let Some(handle) = rfd::AsyncFileDialog::new()
                    .set_title("Attach image")
                    .add_filter("Image", &["png", "jpg", "jpeg", "webp", "gif"])
                    .pick_file()
                    .await
                else {
                    return;
                };
                let name = handle.file_name();
                let bytes = handle.read().await;
                if let Ok(mut guard) = queue.lock() {
                    guard.push(RawPick {
                        bytes,
                        source_desc: name,
                    });
                }
                ctx_for_wake.request_repaint();
            });
        }
    }

    /// Drain the file-picker handoff queue into staged attachments.
    /// Called once per UI frame from `ui()`. No-op on the fast path
    /// (empty queue), so the lock-per-frame cost is negligible.
    fn ingest_pending_picks(&mut self, ctx: &egui::Context) {
        let picks: Vec<RawPick> = match self.pending_picks.lock() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            Err(_) => return,
        };
        for pick in picks {
            self.stage_raw_pick(pick, ctx);
        }
    }

    /// Render the ephemeral compose-area status line. Self-dismisses
    /// once the expiry timestamp passes. Rendered immediately under
    /// the thumbnail strip so the feedback sits visually close to the
    /// thing that produced it.
    fn render_compose_hint(&mut self, ui: &mut egui::Ui) {
        let now = ui.ctx().input(|i| i.time);
        let Some((message, expires)) = self.compose_hint.as_ref() else {
            return;
        };
        if now > *expires {
            self.compose_hint = None;
            return;
        }
        ui.label(
            RichText::new(message)
                .color(Color32::from_gray(170))
                .small(),
        );
    }

    /// Render a compact thumbnail row for staged compose
    /// attachments, sitting above the text input. Each thumbnail
    /// carries an `×` overlay for one-click removal. No-op when
    /// nothing is staged — the surrounding layout reclaims the
    /// vertical space.
    ///
    /// The thumbnail uses egui's image loader through a
    /// per-attachment `bytes://` URI; `ctx.include_bytes` registers
    /// the raw payload with the loader once per frame, which is idempotent
    /// on identical (uri, bytes) pairs and lets the texture cache reuse
    /// the decoded image across frames.
    fn render_compose_attachments(&mut self, ui: &mut egui::Ui) {
        if self.compose_attachments.is_empty() {
            return;
        }
        let mut remove_id: Option<u64> = None;
        ui.horizontal_wrapped(|ui| {
            for staged in &self.compose_attachments {
                let uri = staged.thumbnail_uri();
                if let Some(bytes) = staged.thumbnail_bytes() {
                    // Register once per frame — egui's loader keys by URI
                    // so this is cheap after the first frame.
                    ui.ctx().include_bytes(uri.clone(), bytes.to_vec());
                }
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        if staged.thumbnail_bytes().is_some() {
                            ui.add(
                                egui::Image::new(uri)
                                    .max_size(egui::vec2(96.0, 96.0))
                                    .fit_to_exact_size(egui::vec2(96.0, 96.0)),
                            );
                        } else {
                            // URL-source attachment — no bytes to
                            // render locally. Placeholder preserves
                            // the compose layout.
                            ui.add_sized(
                                egui::vec2(96.0, 96.0),
                                egui::Label::new(
                                    RichText::new("🌐 URL").color(Color32::from_gray(160)),
                                ),
                            );
                        }
                        if ui.small_button("× remove").clicked() {
                            remove_id = Some(staged.id);
                        }
                    });
                });
            }
        });
        if let Some(id) = remove_id {
            self.compose_attachments.retain(|s| s.id != id);
        }
    }

    /// Drain egui's per-frame drop queue into staged attachments via
    /// the shared `stage_raw_pick` pipeline. Every outcome surfaces a
    /// compose-area hint (success, wrong MIME, model doesn't accept,
    /// read error) so drag-drop rejections stop looking like silent
    /// failures. On native a dropped file carries `path`; on wasm it
    /// carries `bytes` — cover both.
    fn accept_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<egui::DroppedFile> = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }
        for file in dropped {
            // egui's DroppedFile.name is a plain String (not Option) but
            // is often empty on native drops — fall back to the path's
            // file stem, and finally a generic label.
            let source_desc = if !file.name.is_empty() {
                file.name.clone()
            } else if let Some(name) = file
                .path
                .as_ref()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            {
                name
            } else {
                "dropped file".to_string()
            };
            let bytes = if let Some(b) = file.bytes {
                b.to_vec()
            } else if let Some(path) = &file.path {
                match std::fs::read(path) {
                    Ok(b) => b,
                    Err(e) => {
                        self.set_compose_hint(
                            ctx,
                            format!("couldn't read {}: {}", path.display(), e),
                        );
                        continue;
                    }
                }
            } else {
                self.set_compose_hint(
                    ctx,
                    format!("{source_desc} carried neither bytes nor a path; skipping"),
                );
                continue;
            };
            self.stage_raw_pick(RawPick { bytes, source_desc }, ctx);
        }
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
                if !self.embedding_providers_requested {
                    self.send(ClientToServer::ListEmbeddingProviders {
                        correlation_id: None,
                    });
                    self.embedding_providers_requested = true;
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
                if !self.buckets_requested {
                    self.send(ClientToServer::ListBuckets {
                        correlation_id: None,
                    });
                    self.buckets_requested = true;
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
        // Empty text is only a send-blocker when the user also hasn't
        // staged any attachments. An image-only compose is a valid
        // send (user dropped a screenshot, wants the model to look at
        // it without a caption).
        if trimmed.is_empty() && self.compose_attachments.is_empty() {
            return;
        }
        if self.composing_new || self.selected.is_none() {
            let (config_override, bindings_request) = self.build_creation_override();
            let initial_attachments = self
                .compose_attachments
                .drain(..)
                .map(|s| s.attachment)
                .collect();
            self.send(ClientToServer::CreateThread {
                correlation_id: None,
                pod_id: self.compose_pod_id.clone(),
                initial_message: trimmed.to_string(),
                initial_attachments,
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
            let attachments = self
                .compose_attachments
                .drain(..)
                .map(|s| s.attachment)
                .collect();
            self.send(ClientToServer::SendUserMessage {
                thread_id,
                text: trimmed.to_string(),
                attachments,
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

impl eframe::App for ChatApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_inbound();
        self.accept_dropped_files(ui.ctx());
        self.ingest_pending_picks(ui.ctx());

        egui::Panel::top("status_bar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                // Register the embedded icon bytes once per frame and
                // render it inline before the title. `include_bytes`
                // is first-write-wins on the URI, so this is a no-op
                // after the first frame; `Image::max_height(24.0)`
                // matches the heading's cap-height so the badge sits
                // visually flush with the "whisper-agent" text.
                ui.ctx().include_bytes(APP_ICON_URI, APP_ICON_PNG);
                ui.add(egui::Image::new(APP_ICON_URI).max_height(24.0));
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

                    let bucket_btn = ui
                        .button(RichText::new("🗃").small())
                        .on_hover_text("Knowledge buckets");
                    if bucket_btn.clicked() {
                        self.buckets_modal = match self.buckets_modal.take() {
                            Some(_) => None,
                            None => Some(BucketsModalState::fresh()),
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
                format!("Describe a new thread in `{display}` — Enter sends, Shift+Enter newline")
            })
        } else {
            None
        };
        let hint: &str = match (composing, input_enabled, pod_hint.as_deref()) {
            (_, false, _) => "(connecting)",
            (true, true, Some(s)) => s,
            (true, true, None) => "Describe a new thread — Enter sends, Shift+Enter newline",
            (false, true, _) => "Message this thread — Enter sends, Shift+Enter newline",
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
                // Thread state drives Send ↔ Stop toggle and whether
                // to hide the stand-alone Cancel button (redundant
                // once Stop lives on the primary button).
                let thread_state = self
                    .selected
                    .as_deref()
                    .and_then(|tid| self.tasks.get(tid))
                    .map(|v| v.summary.state);
                let is_working = thread_state == Some(ThreadStateLabel::Working);
                ui.horizontal(|ui| {
                    if let Some(thread_id) = self.selected.clone() {
                        // While Working, the primary button toggles to
                        // Stop and handles cancellation — hide the
                        // redundant side button so there's a single
                        // affordance.
                        if !is_working && ui.button("Cancel").clicked() {
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
                        self.render_compose_attachments(ui);
                        self.render_compose_hint(ui);
                        // Stable id so the focus check below can look up
                        // the widget's state before the widget is added.
                        let input_id = egui::Id::new("compose-input");
                        // Pre-filter plain Enter so the TextEdit never
                        // sees it: plain Enter submits, Shift+Enter (and
                        // any other modifier combo) passes through to
                        // insert a newline. Note: we can't use
                        // `consume_key(Modifiers::NONE, Enter)` because
                        // that uses `matches_logically`, which treats
                        // `NONE` as "no required modifiers" and would
                        // also eat Shift+Enter. Walk events manually
                        // with an exact-modifiers match instead.
                        // Gated on focus so we don't steal Enter from
                        // other focused widgets (modal buttons etc).
                        let has_focus = ui.memory(|m| m.focused() == Some(input_id));
                        let enter_submit = has_focus
                            && ui.input_mut(|i| {
                                let mut found = false;
                                i.events.retain(|ev| {
                                    let hit = matches!(ev, egui::Event::Key {
                                        key: egui::Key::Enter,
                                        pressed: true,
                                        modifiers,
                                        ..
                                    } if modifiers.matches_exact(egui::Modifiers::NONE));
                                    found |= hit;
                                    !hit
                                });
                                found
                            });
                        // Auto-grow multiline: `desired_rows` reports
                        // the widget's natural height to the enclosing
                        // `Panel::bottom`, which uses its content's
                        // min-rect to decide next-frame panel size.
                        // Counting '\n' is a wrap-agnostic approximation
                        // — good enough for code-like input; true
                        // wrap-aware measurement would need the galley
                        // which isn't available until after the widget
                        // is added. Once content exceeds `MAX_ROWS`,
                        // TextEdit scrolls internally.
                        const MIN_ROWS: usize = 2;
                        const MAX_ROWS: usize = 12;
                        let line_count = self.input.chars().filter(|&c| c == '\n').count() + 1;
                        let rows = line_count.clamp(MIN_ROWS, MAX_ROWS);
                        // Send enables on any staged attachment even when
                        // the text box is empty — image-only compose is a
                        // legitimate send.
                        let can_submit =
                            !self.input.trim().is_empty() || !self.compose_attachments.is_empty();
                        // right_to_left + Align::BOTTOM places the
                        // Send/Stop button at the right edge, aligned
                        // to the bottom of the row height; the
                        // TextEdit then fills the remaining width on
                        // the left and drives the row height via its
                        // `desired_rows`.
                        let inner = ui.allocate_ui_with_layout(
                            egui::vec2(ui.available_width(), 0.0),
                            egui::Layout::right_to_left(egui::Align::BOTTOM),
                            |ui| {
                                let (label, fill, tooltip) = if is_working {
                                    (
                                        "Stop",
                                        SIDEBAR_DANGER_COLOR,
                                        "Cancel this turn (Stop generation).",
                                    )
                                } else {
                                    (
                                        "Send",
                                        SEND_BUTTON_COLOR,
                                        "Send (Enter). Shift+Enter for newline.",
                                    )
                                };
                                // The button's internal label inherits
                                // the surrounding layout's alignment
                                // (Align::BOTTOM here), so drop it into
                                // a fixed-size sub-ui with a centered
                                // layout — that keeps the button rect
                                // anchored bottom-right via the outer
                                // layout while centering the text
                                // within the button itself.
                                let btn_size = egui::vec2(72.0, 28.0);
                                let btn_resp = ui
                                    .allocate_ui_with_layout(
                                        btn_size,
                                        egui::Layout::centered_and_justified(
                                            egui::Direction::LeftToRight,
                                        ),
                                        |ui| {
                                            let btn = egui::Button::new(
                                                RichText::new(label).color(Color32::WHITE).strong(),
                                            )
                                            .fill(fill);
                                            if is_working {
                                                ui.add(btn)
                                            } else {
                                                ui.add_enabled(can_submit, btn)
                                            }
                                        },
                                    )
                                    .inner;
                                let button_clicked = btn_resp.on_hover_text(tooltip).clicked();
                                // Attach button sits to the left of Send
                                // in the right-to-left layout, before the
                                // TextEdit fills the remaining width.
                                // Disabled while the thread is working so
                                // users can't queue attachments mid-turn
                                // (cancel first, then attach).
                                let attach_enabled = !is_working;
                                let attach_resp = ui.add_enabled(
                                    attach_enabled,
                                    egui::Button::new(RichText::new("📎").strong()).frame(true),
                                );
                                let attach_tooltip = match self.current_model_image_acceptance() {
                                    ImageAcceptance::No => {
                                        "Attach file (current model doesn't accept images)"
                                    }
                                    ImageAcceptance::Unknown => {
                                        "Attach file (image support unknown — will try)"
                                    }
                                    ImageAcceptance::Yes => "Attach file",
                                };
                                if attach_resp.on_hover_text(attach_tooltip).clicked() {
                                    self.spawn_file_picker(ui.ctx());
                                }
                                let response = ui.add(
                                    TextEdit::multiline(&mut self.input)
                                        .id(input_id)
                                        .desired_rows(rows)
                                        .desired_width(f32::INFINITY)
                                        .hint_text(hint),
                                );
                                (button_clicked, response)
                            },
                        );
                        let (button_clicked, response) = inner.inner;
                        if response.changed() {
                            self.last_input_change_at = Some(ui.input(|i| i.time));
                        }
                        if is_working {
                            if button_clicked && let Some(tid) = self.selected.clone() {
                                self.send(ClientToServer::CancelThread { thread_id: tid });
                            }
                        } else if (button_clicked || enter_submit) && can_submit {
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
        if let Some(event) = render_fork_modal(&ctx, &mut self.fork_modal) {
            match event {
                ForkEvent::Confirmed {
                    thread_id,
                    from_message_index,
                    archive_original,
                    reset_capabilities,
                    seed_text,
                } => {
                    let correlation_id = self.next_correlation_id();
                    self.pending_fork_seed = Some((correlation_id.clone(), seed_text));
                    self.send(ClientToServer::ForkThread {
                        thread_id,
                        from_message_index,
                        archive_original,
                        reset_capabilities,
                        correlation_id: Some(correlation_id),
                    });
                }
            }
        }
        let backends_empty = self.backends.is_empty();
        if let Some(event) =
            render_new_pod_modal(&ctx, &mut self.new_pod_modal, &self.pods, backends_empty)
        {
            match event {
                NewPodEvent::Created { pod_id, name } => {
                    let config = self.fresh_pod_config(name);
                    self.send(ClientToServer::CreatePod {
                        correlation_id: None,
                        pod_id,
                        config,
                    });
                }
            }
        }
        for event in render_pod_editor_modal(
            &ctx,
            &mut self.pod_editor_modal,
            &self.backends,
            &self.resources,
            &self.host_env_providers,
            &self.models_by_backend,
            &self.buckets,
        ) {
            match event {
                PodEditorEvent::RequestModels(backend) => {
                    self.request_models_for(&backend);
                }
                PodEditorEvent::SaveRequested { pod_id, toml_text } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.pod_editor_modal.as_mut() {
                        modal.pending_correlation = Some(correlation.clone());
                    }
                    self.send(ClientToServer::UpdatePodConfig {
                        correlation_id: Some(correlation),
                        pod_id,
                        toml_text,
                    });
                }
                PodEditorEvent::RefreshRequested { pod_id } => {
                    self.send(ClientToServer::GetPod {
                        correlation_id: None,
                        pod_id,
                    });
                }
            }
        }
        if let Some(event) =
            render_new_behavior_modal(&ctx, &mut self.new_behavior_modal, &self.behaviors_by_pod)
        {
            match event {
                NewBehaviorEvent::Created {
                    pod_id,
                    behavior_id,
                    config,
                } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.new_behavior_modal.as_mut() {
                        modal.pending_correlation = Some(correlation.clone());
                    }
                    self.send(ClientToServer::CreateBehavior {
                        correlation_id: Some(correlation),
                        pod_id,
                        behavior_id,
                        config,
                        prompt: String::new(),
                    });
                }
            }
        }
        for event in render_behavior_editor_modal(
            &ctx,
            &mut self.behavior_editor_modal,
            &self.backends,
            &self.pod_configs,
            &self.models_by_backend,
        ) {
            match event {
                BehaviorEditorEvent::RequestModels(backend) => {
                    self.request_models_for(&backend);
                }
                BehaviorEditorEvent::SaveRequested {
                    pod_id,
                    behavior_id,
                    config,
                    prompt,
                } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.behavior_editor_modal.as_mut() {
                        modal.pending_correlation = Some(correlation.clone());
                    }
                    self.send(ClientToServer::UpdateBehavior {
                        correlation_id: Some(correlation),
                        pod_id,
                        behavior_id,
                        config,
                        prompt,
                    });
                }
            }
        }
        self.render_file_tree_modal(&ctx);
        if let Some(event) = render_file_viewer_modal(&ctx, &mut self.file_viewer_modal) {
            match event {
                FileViewerEvent::SaveRequested {
                    pod_id,
                    path,
                    content,
                } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.file_viewer_modal.as_mut() {
                        modal.pending_correlation = Some(correlation.clone());
                    }
                    self.send(ClientToServer::WritePodFile {
                        correlation_id: Some(correlation),
                        pod_id,
                        path,
                        content,
                    });
                }
            }
        }
        render_json_viewer_modal(&ctx, &mut self.json_viewer_modal);
        if let Some(event) = render_provider_editor_modal(&ctx, &mut self.provider_editor_modal) {
            let correlation = self.next_correlation_id();
            let wire = match event {
                ProviderEditorEvent::AddRequested { name, url, token } => {
                    ClientToServer::AddHostEnvProvider {
                        correlation_id: Some(correlation.clone()),
                        name,
                        url,
                        token,
                    }
                }
                ProviderEditorEvent::UpdateRequested { name, url, token } => {
                    ClientToServer::UpdateHostEnvProvider {
                        correlation_id: Some(correlation.clone()),
                        name,
                        url,
                        token,
                    }
                }
            };
            if let Some(modal) = self.provider_editor_modal.as_mut() {
                modal.pending_correlation = Some(correlation);
            }
            self.send(wire);
        }
        // Knowledge-buckets modal — read-only list (slice 8b) plus
        // the search form (slice 9). Renderer emits `RunQuery` when
        // the user submits a search; we mint a correlation, stamp it
        // on the modal, and dispatch `QueryBuckets`.
        let pods_for_modal: Vec<PodSummary> = {
            let mut v: Vec<PodSummary> = self.pods.values().cloned().collect();
            v.sort_by(|a, b| a.pod_id.cmp(&b.pod_id));
            v
        };
        for event in render_buckets_modal(
            &ctx,
            &mut self.buckets_modal,
            &self.buckets,
            &pods_for_modal,
            &self.embedding_providers,
        ) {
            match event {
                BucketsEvent::RunQuery {
                    bucket_id,
                    pod_id,
                    query,
                    top_k,
                } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.buckets_modal.as_mut() {
                        modal.pending_query_correlation = Some(correlation.clone());
                    }
                    self.send(ClientToServer::QueryBuckets {
                        correlation_id: Some(correlation),
                        bucket_ids: vec![bucket_id],
                        pod_id,
                        query,
                        top_k,
                    });
                }
                BucketsEvent::CreateBucket { id, pod_id, config } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.buckets_modal.as_mut()
                        && let Some(form) = modal.creating.as_mut()
                    {
                        form.pending_correlation = Some(correlation.clone());
                    }
                    self.send(ClientToServer::CreateBucket {
                        correlation_id: Some(correlation),
                        id,
                        pod_id,
                        config,
                    });
                }
                BucketsEvent::DeleteBucket { id, pod_id } => {
                    let correlation = self.next_correlation_id();
                    self.send(ClientToServer::DeleteBucket {
                        correlation_id: Some(correlation),
                        id,
                        pod_id,
                    });
                }
                BucketsEvent::StartBuild { id, pod_id } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.buckets_modal.as_mut() {
                        // Optimistically clear any prior error so the
                        // user doesn't see a stale "last build error"
                        // line during the new attempt.
                        modal.build_errors.remove(&(pod_id.clone(), id.clone()));
                    }
                    self.send(ClientToServer::StartBucketBuild {
                        correlation_id: Some(correlation),
                        id,
                        pod_id,
                    });
                }
                BucketsEvent::CancelBuild { id, pod_id } => {
                    let correlation = self.next_correlation_id();
                    self.send(ClientToServer::CancelBucketBuild {
                        correlation_id: Some(correlation),
                        id,
                        pod_id,
                    });
                }
                BucketsEvent::PollFeedNow { id, pod_id } => {
                    // Fire-and-forget — server's trigger buffer is
                    // bounded at 1, so a click during an in-flight
                    // poll coalesces server-side. We don't surface
                    // the server's `FeedPollAccepted` ack in the UI
                    // today; the user infers success by watching
                    // the bucket's stats refresh on the next
                    // ListBuckets.
                    let correlation = self.next_correlation_id();
                    self.send(ClientToServer::PollFeedNow {
                        correlation_id: Some(correlation),
                        id,
                        pod_id,
                    });
                }
                BucketsEvent::ResyncBucket { id, pod_id } => {
                    // Same wire shape as StartBuild — the server
                    // broadcasts BucketBuildStarted/Progress/Ended
                    // through the existing build-progress channels,
                    // so the modal's in-flight UI lights up
                    // unchanged. AlreadyAtLatest short-circuits to
                    // BucketBuildEnded { Success } with empty slot_id.
                    let correlation = self.next_correlation_id();
                    self.send(ClientToServer::ResyncBucket {
                        correlation_id: Some(correlation),
                        id,
                        pod_id,
                    });
                }
            }
        }

        for event in render_settings_modal(
            &ctx,
            &mut self.settings_modal,
            &self.backends,
            &self.shared_mcp_hosts,
            &self.host_env_providers,
            &mut self.provider_remove_armed,
            &self.provider_remove_pending,
        ) {
            match event {
                SettingsEvent::CodexRotateSave { backend, contents } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.settings_modal.as_mut()
                        && let Some(sub) = modal.codex_rotate.as_mut()
                    {
                        sub.pending_correlation = Some(correlation.clone());
                    }
                    self.send(ClientToServer::UpdateCodexAuth {
                        correlation_id: Some(correlation),
                        backend,
                        contents,
                    });
                }
                SettingsEvent::FetchServerConfig { correlation_id } => {
                    self.send(ClientToServer::FetchServerConfig {
                        correlation_id: Some(correlation_id),
                    });
                }
                SettingsEvent::UpdateServerConfig {
                    correlation_id,
                    toml_text,
                } => {
                    self.send(ClientToServer::UpdateServerConfig {
                        correlation_id: Some(correlation_id),
                        toml_text,
                    });
                }
                SettingsEvent::RemoveSharedMcpHost { name } => {
                    let correlation = self.next_correlation_id();
                    self.send(ClientToServer::RemoveSharedMcpHost {
                        correlation_id: Some(correlation),
                        name,
                    });
                }
                SettingsEvent::AddSharedMcpHost { name, url, auth } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.settings_modal.as_mut()
                        && let Some(sub) = modal.shared_mcp_editor.as_mut()
                    {
                        sub.pending_correlation = Some(correlation.clone());
                    }
                    self.send(ClientToServer::AddSharedMcpHost {
                        correlation_id: Some(correlation),
                        name,
                        url,
                        auth,
                        // Editor doesn't expose prefix yet; server
                        // resolves `Unchanged` to `Default` (use the
                        // host name as the prefix) on Add.
                        prefix: whisper_agent_protocol::SharedMcpPrefixInput::Unchanged,
                    });
                }
                SettingsEvent::UpdateSharedMcpHost { name, url, auth } => {
                    let correlation = self.next_correlation_id();
                    if let Some(modal) = self.settings_modal.as_mut()
                        && let Some(sub) = modal.shared_mcp_editor.as_mut()
                    {
                        sub.pending_correlation = Some(correlation.clone());
                    }
                    self.send(ClientToServer::UpdateSharedMcpHost {
                        correlation_id: Some(correlation),
                        name,
                        url,
                        auth,
                        // Editor doesn't expose prefix yet; `Unchanged`
                        // tells the server to leave the catalog's
                        // existing prefix override alone.
                        prefix: whisper_agent_protocol::SharedMcpPrefixInput::Unchanged,
                    });
                }
                SettingsEvent::OpenAddProvider => {
                    self.provider_editor_modal = Some(ProviderEditorModalState::new_add());
                }
                SettingsEvent::OpenEditProvider(info) => {
                    self.provider_editor_modal = Some(ProviderEditorModalState::new_edit(&info));
                }
                SettingsEvent::RemoveHostEnvProvider { name } => {
                    let correlation = self.next_correlation_id();
                    self.provider_remove_pending.insert(
                        name.clone(),
                        ProviderRemovePending {
                            correlation: correlation.clone(),
                            error: None,
                        },
                    );
                    self.send(ClientToServer::RemoveHostEnvProvider {
                        correlation_id: Some(correlation),
                        name,
                    });
                }
            }
        }
        render_image_lightbox_modal(&ctx);

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
                knowledge_buckets: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::wire_handler::pending_tool_batch_flush_thread_id;
    use super::*;
    use whisper_agent_protocol::{ThreadStateLabel, Usage};

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
}
