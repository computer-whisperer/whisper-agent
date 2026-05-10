//! [`ChatApp`] — the platform-agnostic aetna [`App`] for the
//! whisper-agent chat client.
//!
//! Surface so far:
//!
//! - Brand-marked sidebar with draggable resize handle and a pinned
//!   status footer (server URL + connection dot).
//! - Pod tabs above a per-pod threads list (with dispatch nesting and
//!   provenance markers) and behaviors list (run-now / pause / edit /
//!   delete toolbar).
//! - Snapshot + delta-streamed chat log rendered as event-log rows
//!   (3 px role-colored gutter + content). Assistant turns flow
//!   through `aetna_markdown::md`; reasoning and tool rows live in
//!   collapsible `accordion_item`s.
//! - Inline images (PNG / JPEG / WebP / GIF) with click-to-open
//!   fullscreen lightbox.
//! - Per-turn token-stats footer plus a cumulative
//!   `backend/model · ↑in / ↓out / cache r/c` chip in the thread
//!   header. Prefill progress bar above the chat log when active.
//! - Compose box: text area, drag/drop and file-picker attachments,
//!   Enter-to-send, keyboard hint when empty, paperclip + Send / Stop.
//!   Per-thread drafts persisted via `SetThreadDraft`.
//! - Pending-sudo approval banner above the affected thread.
//! - Modals: fork-thread, "+ new pod", "+ new behavior".
//! - Sheets: pod editor (Allow / Defaults / Limits + raw TOML),
//!   behavior editor (Trigger / Thread / Scope / Retention / Prompt
//!   / System Prompt / Raw TOML).
//! - New-thread compose pane (backend / model / pod pickers + text
//!   area) reachable when no thread is selected.
//!
//! Build/on_event split is the load-bearing test of the pivot — every
//! interactive element routes through [`ChatApp::on_event`] via a
//! key, every visual is a function of state read in
//! [`ChatApp::build`].
//!
//! Surfaces still to land (full inventory in
//! `docs/design_aetna_ui.md` § "Migration gaps from egui webui"):
//! Codex auth rotate + Shared MCP CRUD sub-slices of the server
//! settings modal, bucket create form + search-and-query
//! sub-slices of the knowledge-buckets modal.
//!
//! Dispatch model: a single `dispatch_wire` walks `ServerToClient`
//! variants — only the ones the current stage cares about have arms;
//! the rest drop on the floor.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use crate::cron_preview;

use aetna_core::prelude::*;
use aetna_core::widgets::resize_handle::{self, ResizeDrag, Side};
use aetna_core::widgets::select::{SelectAction, classify_event as classify_select_event};
use whisper_agent_protocol::{
    ActivationSurface, AllowMap, Attachment, BackendSummary, BehaviorConfig, BehaviorSummary,
    BehaviorThreadOverride, BucketBuildOutcome, BucketBuildPhase, BucketSummary, CatchUp,
    ClientToServer, ContentBlock, CoreTools, Disposition, FsEntry, ImageMime, ImageSource,
    InitialListing, ModelSummary, NamedHostEnv, Overlap, PodAllow, PodConfig, PodLimits,
    PodSummary, RetentionPolicy, Role, ServerToClient, SlotStateLabel, SystemPromptChoice,
    ThreadBindingsRequest, ThreadConfigOverride, ThreadDefaults, ThreadSummary, TriggerSpec,
    permission::SudoDecision,
};

/// Inbound event variants the host shell pushes into the [`Inbound`]
/// queue. Mirrors `whisper-agent-webui`'s shape so the desktop bridge
/// can be a one-liner replacement on the import side.
// `ServerToClient` dwarfs the connection variants; matching the egui
// sibling keeps the bridge code trivial. `Box` would change every
// enqueue site without saving meaningfully on a shallow queue.
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
    fn label(self) -> &'static str {
        match self {
            Self::Connecting => "connecting…",
            Self::Connected => "connected",
            Self::Closed => "closed",
            Self::Error => "error",
        }
    }
}

/// One row in the chat log.
enum DisplayItem {
    User {
        text: String,
        /// Index in the conversation's `Message` vector this user
        /// turn occupies. Surfaced to the per-row fork affordance so
        /// `ClientToServer::ForkThread { from_message_index }` knows
        /// where to fork from. Mirrors the egui sibling's per-row
        /// msg_index. Live-streamed `ThreadUserMessage` rows take the
        /// next index past the current `view.items`' last seen User
        /// — see `view_user_msg_index_for_streaming`.
        msg_index: usize,
    },
    Assistant {
        text: String,
    },
    Reasoning {
        text: String,
    },
    /// Thread-prefix system prompt — the `Role::System` message at
    /// the head of the conversation. Default-collapsed so a long
    /// prompt doesn't dominate the chat log; expanded body shows
    /// the full text in a code-styled block. Mirrors the egui
    /// sibling's `DisplayItem::SetupPrompt`.
    SetupPrompt {
        text: String,
    },
    /// Thread-prefix tool manifest — the `Role::Tools` message
    /// carrying one `ContentBlock::ToolSchema` per advertised
    /// tool. Default-collapsed; collapsed header shows
    /// `{N} tools`, expanded body shows the per-tool list. We
    /// keep just the schemas the renderer actually needs (name +
    /// description) — the wire shape stays in the snapshot if
    /// future expansion wants typed parameter walking too.
    SetupTools {
        entries: Vec<ToolSchemaSummary>,
    },
    /// A model-emitted tool call. Fuses with its result when the
    /// matching `ContentBlock::ToolResult` (or `ThreadToolCallEnd`)
    /// arrives without an intervening user / assistant turn — the
    /// common case for sync calls. `streaming_output` accumulates
    /// `ThreadToolCallContent` text fragments while the call is in
    /// flight; `result` holds the integrated final text once `End`
    /// lands.
    ToolCall {
        tool_use_id: String,
        name: String,
        /// One-line summary of the call, derived from a relevant
        /// argument. `read_file` / `edit_file` / `write_file` ⇒ the
        /// `path`; `bash` ⇒ the (truncated) `command`; `grep` /
        /// `glob` ⇒ the `pattern`; `list_dir` ⇒ the `path`. `None`
        /// when the args don't carry a recognized field — the
        /// header then falls back to just the tool name. Mirrors
        /// the egui sibling's `tool_summary`.
        summary: Option<String>,
        /// Resolved diff for `edit_file` / `write_file` tool calls.
        /// `Some` when the args carry the expected shape (path +
        /// old_string + new_string for edit_file; path + content
        /// for write_file). When set, the body renders the diff
        /// instead of the raw `args_pretty` JSON.
        diff: Option<DiffPayload>,
        /// Pretty-printed JSON args. `None` when the snapshot path
        /// produced a tool call without typed args (legacy threads).
        args_pretty: Option<String>,
        /// Live `ThreadToolCallContent` chunks accumulated since the
        /// matching `ThreadToolCallBegin`. Empty until the first
        /// chunk lands; cleared once the integrated `result` arrives.
        streaming_output: String,
        result: Option<FusedToolResult>,
    },
    /// Standalone tool result — rendered when the matching tool call
    /// isn't in this thread's view (e.g. an async `dispatch_thread`
    /// callback landing after the conversation has moved on, or a
    /// `Role::ToolResult` message snapshot-walked without a sibling
    /// `ToolUse` in scope).
    ToolResult {
        tool_use_id: String,
        text: String,
        is_error: bool,
    },
    /// Inline image — user-supplied (drag/paste/file picker on the
    /// egui sibling, eventually here) or model-generated (Gemini
    /// native image output, OpenAI Responses `image_generation`).
    /// Decoded once at push time and cached as an `aetna_core::Image`
    /// so rebuilds don't re-decode the bytes. The `is_user` flag
    /// drives the role gutter color in `event_log_row`.
    Image {
        is_user: bool,
        state: ImageRenderState,
    },
    /// Anything else we don't yet render purpose-built (images,
    /// documents, tool-schema entries) — collapsed into a single
    /// annotated row. Future stages break these out.
    GenericPlaceholder {
        label: String,
    },
    /// Per-assistant-turn token usage summary. Pulled from
    /// `ThreadSnapshot::turn_log` and interleaved after each
    /// `Assistant` message in conversation order. No gutter, no
    /// fill — auxiliary, not conversation. Mirrors the egui
    /// sibling's `DisplayItem::TurnStats`.
    TurnStats {
        usage: whisper_agent_protocol::Usage,
    },
}

/// Pull a one-line summary out of a tool call's args based on the
/// tool's name. The well-known set covers the ~5 tools the user
/// sees most often in a working session — file reads/edits, shell
/// commands, search. Anything else returns `None` and the header
/// falls back to just the tool name.
///
/// Mirrors the egui sibling's `tool_summary`. Hard-coding the
/// argument-name knowledge here is the right scope: every tool
/// declares its own parameter names, so a generic "first-string-
/// arg" heuristic would routinely surface noise (e.g. an
/// `encoding` field instead of `path`).
fn tool_summary_from_args(name: &str, args: Option<&serde_json::Value>) -> Option<String> {
    let v = args?;
    let pick = |key: &str| {
        v.get(key)
            .and_then(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    match name {
        "edit_file" | "write_file" | "read_file" => pick("path"),
        "bash" | "run_bash" => pick("command").map(|c| {
            // Truncate long shell commands so the header line
            // doesn't run off the right edge of the chat pane —
            // the full command is still in the expanded body.
            let mut s = c;
            if s.chars().count() > 120 {
                s = s.chars().take(120).collect::<String>();
                s.push('…');
            }
            s
        }),
        "grep" | "glob" => pick("pattern"),
        // `list_dir` is the egui sibling's name; `list_files` is
        // what whisper-agent's own tool catalog uses today. Both
        // map to the same `path` argument shape.
        "list_dir" | "list_files" => pick("path").or_else(|| Some(".".to_string())),
        _ => None,
    }
}

/// Resolve a [`DiffPayload`] from a tool call's args when the tool
/// is one we know how to render as a diff. `edit_file` carries
/// `path` + `old_string` + `new_string` for an in-place
/// substitution; `write_file` carries `path` + `content` and is
/// treated as creating a fresh file (old_text empty,
/// `is_creation = true`). Anything else returns `None`.
fn extract_diff(name: &str, args: Option<&serde_json::Value>) -> Option<DiffPayload> {
    let v = args?;
    let s = |key: &str| v.get(key).and_then(|x| x.as_str()).map(str::to_owned);
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

/// Format a `Usage` block into the muted one-line summary the
/// `TurnStats` row renders. Mirrors the egui sibling's layout —
/// `in 1,234 · cached 800 · created 200 · out 567`. Cache columns
/// drop out when zero so an uncached call doesn't carry visually
/// empty fields.
fn turn_stats_text(usage: &whisper_agent_protocol::Usage) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(4);
    parts.push(format!("in {}", fmt_count(usage.input_tokens)));
    if usage.cache_read_input_tokens > 0 {
        parts.push(format!(
            "cached {}",
            fmt_count(usage.cache_read_input_tokens)
        ));
    }
    if usage.cache_creation_input_tokens > 0 {
        parts.push(format!(
            "created {}",
            fmt_count(usage.cache_creation_input_tokens)
        ));
    }
    parts.push(format!("out {}", fmt_count(usage.output_tokens)));
    parts.join(" · ")
}

/// Right-aligned `backend/model` chip rendered in the thread
/// header sub-line. Mirrors the egui sibling's `format!("{}/{}",
/// backend_label, view.model)` strip. Returns `None` when the
/// thread isn't bound to a backend yet (`bindings.backend` is
/// empty) — the snapshot hasn't landed, the thread was rebound
/// to nothing, or the server's a legacy version that didn't
/// populate the field. The model alone with no backend reads as
/// noise, so we suppress the whole strip rather than show half.
fn thread_model_chip(backend: &str, model: &str) -> Option<El> {
    if backend.is_empty() {
        return None;
    }
    let label = if model.is_empty() {
        backend.to_string()
    } else {
        format!("{backend}/{model}")
    };
    Some(text(label).caption().muted().ellipsis())
}

/// Right-aligned cumulative-usage chip for the thread header.
/// Format: `↑ 12,345 · ↓ 2,345 · cache 800r/200c`. Suppressed
/// when every field is zero — surfacing a dead "↑ 0 · ↓ 0"
/// chip on a fresh thread before the first turn would fight the
/// otherwise-tight chrome. Cache portion drops when both
/// cache columns are zero so a non-cacheing model doesn't carry
/// the noise.
fn thread_usage_chip(usage: &whisper_agent_protocol::Usage) -> Option<El> {
    if usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cache_read_input_tokens == 0
        && usage.cache_creation_input_tokens == 0
    {
        return None;
    }
    let mut parts: Vec<String> = Vec::with_capacity(3);
    parts.push(format!("\u{2191} {}", fmt_count(usage.input_tokens)));
    parts.push(format!("\u{2193} {}", fmt_count(usage.output_tokens)));
    if usage.cache_read_input_tokens > 0 || usage.cache_creation_input_tokens > 0 {
        parts.push(format!(
            "cache {}r/{}c",
            fmt_count(usage.cache_read_input_tokens),
            fmt_count(usage.cache_creation_input_tokens),
        ));
    }
    Some(text(parts.join(" \u{00B7} ")).caption().muted())
}

/// `12345` → `12,345`. Cheap enough per-rebuild that a dedicated
/// helper is fine; keeps the format consistent across all four
/// turn-stats columns.
fn fmt_count(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let bytes = s.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// File-edit tool call's args resolved into a unified-diff shape.
/// Produced by [`extract_diff`] for `edit_file` (old_string →
/// new_string substitution) and `write_file` (full-content
/// creation, treated as old_text="" + new_text=content). Routes
/// through the diff renderer in [`tool_call_body`] instead of the
/// generic JSON args block.
///
/// `is_creation` flips the body's header from the bare `{path}` to
/// `(new) {path}` so the user can tell at a glance whether they're
/// looking at an in-place edit or a fresh file write.
#[derive(Clone)]
struct DiffPayload {
    path: String,
    old_text: String,
    new_text: String,
    is_creation: bool,
}

/// Compact subset of [`whisper_agent_protocol::ToolSchema`] —
/// just the fields the [`DisplayItem::SetupTools`] body shows.
/// Cloned at conversion time so the renderer doesn't carry the
/// full typed-params shape across rebuilds (params don't render
/// today; reserved for the future structured-params expansion).
#[derive(Clone)]
struct ToolSchemaSummary {
    name: String,
    description: String,
}

/// Tool-result body fused onto its originating [`DisplayItem::ToolCall`].
struct FusedToolResult {
    text: String,
    is_error: bool,
}

/// Sniff a `Content-Type` from the first few bytes of a dropped or
/// picked file. Avoids trusting filename extensions (screenshots
/// arrive without one) and keeps the MIME set tight to the
/// protocol's [`ImageMime`] variants — anything else is rejected at
/// the compose-area boundary with a visible hint. Mirrors the egui
/// sibling's helper of the same name.
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

/// Image attachment staged for the next send. Holds both the
/// canonical [`Attachment`] (drained into outbound `SendUserMessage`
/// / `CreateThread` on submit) and a pre-decoded `aetna_core::Image`
/// for the thumbnail. `id` is a monotonic counter so the thumbnail
/// widget's identity is stable across rebuilds — pixel-identical
/// duplicates the user staged on purpose still count as distinct.
struct StagedAttachment {
    id: u64,
    attachment: Attachment,
    /// Decoded thumbnail. `None` when decode failed or when the
    /// source is a URL (no bytes locally yet); the render path
    /// shows a placeholder in that case.
    thumbnail: Option<aetna_core::image::Image>,
    /// Display label under the thumbnail — the file's basename for
    /// drops/picks, `(URL attachment)` for url-source rows.
    source_desc: String,
}

/// Async handoff from the rfd file picker into the UI thread. The
/// picker runs off-thread (rfd is async; we drive it on a tokio
/// current-thread runtime in a background `std::thread::spawn`);
/// when the user picks a file, the handler pushes a `RawPick` onto
/// `ChatApp::pending_picks` and `before_build` drains it on the
/// next frame.
struct RawPick {
    bytes: Vec<u8>,
    source_desc: String,
}

/// Three terminal states for an inline image: successfully decoded
/// (we hold the `aetna_core::Image` and the original dimensions),
/// remote URL (deferred — Stage 7 doesn't fetch yet), or a decode
/// failure (we kept the reason for surfacing in the UI).
#[derive(Clone)]
enum ImageRenderState {
    Decoded {
        image: aetna_core::image::Image,
        width: u32,
        height: u32,
    },
    Url {
        url: String,
    },
    Failed {
        reason: String,
    },
}

/// State backing the fullscreen image-lightbox modal. The decoded
/// image is cloned out of the originating chat-log row so the modal
/// can paint at native dimensions without re-decoding the source
/// bytes; width/height ride alongside for the bottom caption.
#[derive(Clone)]
struct LightboxState {
    image: aetna_core::image::Image,
    width: u32,
    height: u32,
}

/// Generic text-editor modal over one pod file. Opened by clicking
/// a non-specialized file path in the file tree (pod.toml / behavior
/// configs / behavior prompts / `.json` paths route to their own
/// editors). Buffer-and-baseline shape lets Save short-circuit on
/// `working == baseline` and Revert restore without a re-fetch.
/// Mirrors the egui sibling's `FileViewerModalState`.
struct FileViewerModalState {
    pod_id: String,
    path: String,
    /// In-memory edit buffer. `None` until `PodFileContent` lands.
    working: Option<String>,
    /// Last-known server content. Pairs with `working` for the
    /// dirty check and Revert.
    baseline: Option<String>,
    /// Mirrors `is_readonly_path` on the file. Runtime state
    /// (thread JSONs, `pod_state.json`, `behaviors/*/state.json`)
    /// loads with `true` and the Save / Revert buttons disappear.
    readonly: bool,
    /// Inline error: most recent read or write failure. Cleared on
    /// the next successful round-trip or on the user's next edit.
    error: Option<String>,
    /// Outstanding correlation. Distinct from a missing-yet-loaded
    /// modal: `pending_correlation = Some` + `working = None` means
    /// the initial read is in flight; `pending_correlation = Some`
    /// + `working = Some` means a save is in flight.
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

/// `(pod_id, bucket_id)` key for per-row bucket state (build
/// progress, build errors, delete-armed). Tuple key matches the egui
/// sibling's `BucketRowKey` so the two clients have isomorphic state
/// shapes — pod-scope and server-scope buckets with the same id can
/// coexist after PB3b.
type BucketRowKey = (Option<String>, String);

/// Live per-bucket build progress snapshot. Inserted on
/// `BucketBuildStarted`, updated on `BucketBuildProgress`, removed
/// on `BucketBuildEnded`. Mirrors the egui sibling's
/// `BuildProgressView`.
#[derive(Clone)]
struct BuildProgressView {
    phase: BucketBuildPhase,
    source_records: u64,
    chunks: u64,
    /// RFC3339 wall-clock dispatch time forwarded by `BucketBuildStarted`
    /// and every `BucketBuildProgress`. `None` for legacy servers; renders
    /// no elapsed stopwatch when missing.
    started_at: Option<String>,
    /// Resume-path HNSW-rebuild gauge. `Some` only during a resume
    /// rebuild; `None` during fresh builds.
    dense_inserted: Option<u64>,
    dense_total: Option<u64>,
}

/// Knowledge-buckets modal state. v1 ships the read-only catalog
/// surface, the live build-progress display, and the four row
/// actions (Build / Pause build / Poll now / Resync now) plus
/// arm-confirm Delete. Create form + search-and-query are deferred
/// to follow-up sub-slices — `creating`, `query_input`, `top_k`,
/// `query_status` aren't carried here yet.
#[derive(Default)]
struct BucketsModalState {
    /// Bucket id whose Delete button is "armed" — clicked once,
    /// waiting for a confirming second click. Cleared whenever the
    /// user starts a different action, switches modals, or the
    /// matching `BucketDeleted` lands.
    delete_armed: Option<String>,
    /// Live build-progress map. Keyed by `(pod_id, bucket_id)`.
    build_progress: HashMap<BucketRowKey, BuildProgressView>,
    /// Sticky last-failed-build error message per `(pod_id,
    /// bucket_id)`. Overwritten by the next attempt; cleared when a
    /// success lands for the same key.
    build_errors: HashMap<BucketRowKey, String>,
}

/// Server-settings modal state. v1 has two tabs: a read-only LLM
/// backends listing and an admin-only raw editor for the server's
/// `whisper-agent.toml`. The Shared MCP hosts CRUD + Codex Rotate
/// sub-form land in follow-up slices.
#[derive(Default)]
struct SettingsModalState {
    active_tab: SettingsTab,
    /// Editor state for the Server-config tab. `None` until the tab
    /// has been opened at least once; persists across tab switches
    /// so in-progress edits survive.
    server_config: Option<ServerConfigEditorState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SettingsTab {
    #[default]
    Backends,
    /// Raw editor for `whisper-agent.toml`. Admin-only — the server
    /// rejects `FetchServerConfig` / `UpdateServerConfig` from
    /// non-admin connections. On save, the server hot-swaps the
    /// backend catalog and cancels any thread using a
    /// removed / modified backend.
    ServerConfig,
}

impl SettingsTab {
    fn label(self) -> &'static str {
        match self {
            SettingsTab::Backends => "LLM backends",
            SettingsTab::ServerConfig => "Server config",
        }
    }

    fn wire_value(self) -> &'static str {
        match self {
            SettingsTab::Backends => "backends",
            SettingsTab::ServerConfig => "server-config",
        }
    }

    fn from_wire(raw: &str) -> Option<Self> {
        match raw {
            "backends" => Some(SettingsTab::Backends),
            "server-config" => Some(SettingsTab::ServerConfig),
            _ => None,
        }
    }
}

/// Summary of a successful `UpdateServerConfig` — shown as a banner
/// on the server-config editor. Mirrors the egui sibling's
/// `ServerConfigSaveSummary`.
#[derive(Debug, Clone, Default)]
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
    /// fetch round-trip is in flight (renderer shows "loading…").
    original: Option<String>,
    /// Text the user is currently editing. Seeded from `original`
    /// when fetch completes; survives tab switches.
    working: String,
    /// In-flight `FetchServerConfig` correlation, if any.
    fetch_correlation: Option<String>,
    /// In-flight `UpdateServerConfig` correlation, if any.
    save_correlation: Option<String>,
    /// Success banner data from the most recent save. Cleared when
    /// the user starts a new edit or kicks off another save.
    save_summary: Option<ServerConfigSaveSummary>,
    /// Last save / fetch error. Mutually exclusive with
    /// `save_summary` in practice.
    error: Option<String>,
}

impl ServerConfigEditorState {
    fn new() -> Self {
        Self {
            original: None,
            working: String::new(),
            fetch_correlation: None,
            save_correlation: None,
            save_summary: None,
            error: None,
        }
    }

    fn dirty(&self) -> bool {
        self.original
            .as_deref()
            .map(|o| o != self.working)
            .unwrap_or(false)
    }
}

/// Click-dispatch outcome for a path picked out of the file tree.
/// Each known-shaped file routes to its specialized editor; everything
/// else falls through to the generic [`FileViewerModalState`]
/// (which the server-side `is_readonly_path` sniff downgrades to
/// read-only when the path can't be safely overwritten).
///
/// Path strings mirror server-side constants (`pod::POD_TOML`, the
/// `behaviors/<id>/{behavior.toml,prompt.md}` shape, the `.json`
/// extension) — kept in sync by hand because this crate deliberately
/// doesn't depend on the server crate.
enum PodFileDispatch {
    PodConfig,
    BehaviorConfig(String),
    BehaviorPrompt(String),
    JsonViewer(String),
    TextEditor(String),
}

/// Read-only JSON tree viewer state. Opened over a pod file path
/// (typically a thread JSON or server-emitted state blob). Lifecycle:
/// `open_json_viewer` mints a correlation + fires `ReadPodFile`, the
/// `PodFileContent` arm populates `parsed` on success or `error` on
/// parse failure. Mirrors the egui sibling's `JsonViewerModalState`.
struct JsonViewerModalState {
    pod_id: String,
    path: String,
    /// Parsed payload. `None` while the read is in flight OR when the
    /// file failed to parse as JSON — `error` disambiguates.
    parsed: Option<serde_json::Value>,
    /// Parse error or read failure surfaced to the user. Mutually
    /// exclusive with `parsed` in practice — a successful read clears
    /// any prior error.
    error: Option<String>,
    /// Outstanding `ReadPodFile` correlation. `None` once the matching
    /// `PodFileContent` lands (whether the JSON parsed or not).
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

/// One pending `sudo(...)` approval the model is awaiting a decision
/// on. The banner above the chat log surfaces these; the user picks
/// Approve / Remember / Reject and we send `ResolveSudo`. Mirrors
/// the egui sibling's `PendingSudo`.
#[derive(Clone)]
struct PendingSudoState {
    /// Thread the model is running in. Used to gate which banners
    /// render in which pane — only the selected thread's banners show.
    thread_id: String,
    /// Wrapped tool name (e.g. `"bash"`, `"write_file"`). The thing
    /// the model wants to call.
    tool_name: String,
    /// Inner args the model would pass to the wrapped tool. Pretty-
    /// printed for the banner body so the user can see what's about
    /// to run.
    args: serde_json::Value,
    /// Free-form justification the model emitted alongside the call.
    /// Surfaced verbatim — the model is selling the user on running
    /// this, so the user reads it as the case for approval.
    reason: String,
}

/// Decode an [`ImageSource::Bytes`] payload into an `aetna_core::Image`,
/// returning the source-as-display-state so the call site can `match`
/// without owning the decoder. URL sources route to `Url`; bytes
/// failures land in `Failed { reason }` so the row still has a
/// placeholder rather than disappearing silently.
fn decode_image_source(source: &ImageSource) -> ImageRenderState {
    match source {
        ImageSource::Url { url } => ImageRenderState::Url { url: url.clone() },
        ImageSource::Bytes { data, .. } => match image::load_from_memory(data) {
            Ok(decoded) => {
                let rgba = decoded.to_rgba8();
                let (w, h) = rgba.dimensions();
                let pixels = rgba.into_raw();
                ImageRenderState::Decoded {
                    image: aetna_core::image::Image::from_rgba8(w, h, pixels),
                    width: w,
                    height: h,
                }
            }
            Err(err) => ImageRenderState::Failed {
                reason: err.to_string(),
            },
        },
    }
}

/// Per-thread display state. Held only for threads we've subscribed
/// to; the sidebar list works off [`ChatApp::threads`] alone.
struct ThreadView {
    items: Vec<DisplayItem>,
    /// Carried from the snapshot so the title can fall back to the
    /// summary when the snapshot hasn't arrived yet, and so a
    /// resubscribe replaces the live items cleanly.
    title: Option<String>,
    /// `ThreadSnapshot::failure` mirrored locally. `Some` when the
    /// thread is in a failed state (server set the field at the
    /// time of the integration error). The pane renders a
    /// destructive banner above the chat log when this is set
    /// *and* the summary's state is `Failed` — the conjunction
    /// avoids surfacing stale failure text after a recovery.
    failure: Option<String>,
    /// Backend alias the server resolved this thread to (e.g.
    /// `"prod-anthropic"`). Hydrated from
    /// `ThreadSnapshot::bindings.backend`. Empty string means the
    /// thread isn't bound to a backend yet — surfaced as the muted
    /// `—` placeholder in the header. Mirrors the egui sibling's
    /// `TaskView::backend`.
    backend: String,
    /// Model id the thread was created with. Hydrated from
    /// `ThreadSnapshot::config.model`. Pairs with `backend` to form
    /// the thread-header `backend/model` badge.
    model: String,
    /// Running token tally for this thread. Hydrated from
    /// `ThreadSnapshot::total_usage` and incremented on every
    /// `ThreadAssistantEnd { usage }` arm. Mirrors the egui
    /// sibling's `view.total_usage.add(&usage)` pattern. Surfaced
    /// in the thread header as `↑ in · ↓ out · cache r/c`.
    total_usage: whisper_agent_protocol::Usage,
    /// Index a freshly-arriving `ThreadUserMessage` lands at in the
    /// underlying [`whisper_agent_protocol::Conversation`]. Hydrated
    /// from `conv.messages().count()` on snapshot, then incremented
    /// per server-broadcast message arm so per-row fork affordances
    /// can stamp the matching `from_message_index`. Tool-result and
    /// assistant streaming arms also bump this so the count tracks
    /// the conversation, not just user turns.
    next_msg_index: usize,
    /// Inspector-panel projection of snapshot metadata that the
    /// header chips can't fit. Hydrated once on snapshot arrival
    /// alongside `backend` / `model`; the inspector renders these
    /// inline above the chat log when the user toggles it open.
    host_env_labels: Vec<String>,
    mcp_hosts: Vec<String>,
    max_tokens: u32,
    max_turns: u32,
    created_at: String,
    /// `Some(behavior_id)` when this thread was spawned by a behavior
    /// trigger. `BehaviorSummary`-style display name lookup happens
    /// at render time.
    origin_behavior_id: Option<String>,
}

pub struct ChatApp {
    // ----- transport bridge -----
    inbound: Inbound,
    send_fn: SendFn,

    // ----- branding / identity -----
    /// Server URL the host shell connected this client to. Surfaced in
    /// the sidebar footer so the user can see at a glance "which
    /// whisper-agent am I talking to" without opening settings —
    /// typical desktop sessions have one window per server, but
    /// remote tunnels and shared machines benefit from the affordance.
    /// `None` skips the URL line in the footer (e.g. when the host
    /// hasn't supplied one), but the connection-status line still
    /// renders. Set via [`Self::set_server_label`].
    server_label: Option<String>,

    // ----- sidebar resize -----
    /// Current pinned width of the left sidebar in logical pixels.
    /// Initialized to `tokens::SIDEBAR_WIDTH`; mutated by drag /
    /// keyboard nudges on the `sidebar:resize` handle, clamped to
    /// `[SIDEBAR_WIDTH_MIN, SIDEBAR_WIDTH_MAX]` by the helper. The
    /// `sidebar()` widget bakes its own width default in, but a
    /// chained `.width(...)` at the call site overrides it.
    sidebar_width: f32,
    /// Drag-anchor state for the sidebar resize handle. Holds the
    /// pointer x-position at which the user pressed down and the
    /// width at that moment, so each frame's drag delta can derive
    /// an absolute target. Cleared on PointerUp.
    sidebar_drag: ResizeDrag,

    // ----- modal: image lightbox -----
    /// Decoded image currently shown in the fullscreen lightbox modal,
    /// alongside the original source dimensions for the caption. `Some`
    /// while the modal is open — opened by clicking any inline image
    /// row, closed by clicking the scrim or the close affordance.
    /// Mirrors the egui sibling's `lightbox` slot. The decoded
    /// `aetna_core::Image` is shared with the inline row so we don't
    /// re-decode the bytes when the modal opens.
    lightbox: Option<LightboxState>,

    // ----- thread inspector -----
    /// Thread id whose inspector panel is currently expanded. `Some`
    /// while open, `None` when collapsed. Single-slot rather than a
    /// set since only the selected thread renders its inspector;
    /// switching threads collapses the prior one.
    inspector_open: Option<String>,

    // ----- modal: knowledge buckets -----
    /// Knowledge-buckets modal state. `Some` while open. Opened via
    /// the database icon-button in the sidebar footer. Live build
    /// progress (`build_progress`) is hydrated by `BucketBuildStarted`
    /// / `BucketBuildProgress` events that arrive even while the
    /// modal is closed — they're cheap and the next open paints with
    /// fresh counters. Sticky build errors (`build_errors`) follow
    /// the same lifecycle. Both maps live inside `BucketsModalState`
    /// so closing the modal collapses them; that's fine for v1 since
    /// the modal is the only surface that displays them.
    buckets_modal: Option<BucketsModalState>,

    // ----- modal: server settings -----
    /// Server-settings modal state. `Some` while open. Lifecycle:
    /// opened by `open_settings_modal` (cog button in the sidebar
    /// footer). The Server-config tab lazy-fetches its TOML on first
    /// open via `ensure_server_config_fetched`.
    settings_modal: Option<SettingsModalState>,

    // ----- modal: file viewer -----
    /// Generic edit-with-save modal over a pod file (anything not
    /// claimed by the pod / behavior editors or the JSON viewer).
    /// `Some` while open. Lifecycle: `open_file_viewer` stamps
    /// `pending_correlation` and fires `ReadPodFile`; the matching
    /// `PodFileContent` arm populates `working` + `baseline` +
    /// `readonly`. A user Save mints a new correlation and fires
    /// `WritePodFile`; the matching `PodFileWritten` adopts the
    /// working buffer as the new baseline.
    file_viewer_modal: Option<FileViewerModalState>,

    // ----- modal: file tree -----
    /// Pod the file-tree modal is currently open over. `Some` while
    /// open; cleared on dismiss / close. Mirrors the egui sibling's
    /// `file_tree_modal_pod`.
    file_tree_modal_pod: Option<String>,
    /// Cache of `ListPodDir` replies. Keyed by `(pod_id, path)`
    /// where path is pod-relative ("" = root). Children of expanded
    /// dirs are fetched lazily one round-trip at a time.
    pod_files: HashMap<(String, String), Vec<FsEntry>>,
    /// Outstanding `ListPodDir` keys so a re-render doesn't fire a
    /// second request for the same dir while the first is in flight.
    /// Drained when the matching `PodDirListing` lands.
    pod_files_requested: HashSet<(String, String)>,
    /// Per-directory expanded membership for the file-tree modal.
    /// Keyed by `(pod_id, path)`. Toggled by clicks on the dir row;
    /// expanded dirs trigger a lazy `ensure_pod_dir_fetched`.
    pod_dirs_open: HashSet<(String, String)>,

    // ----- modal: JSON viewer -----
    /// Read-only JSON tree viewer over a pod file. `Some` while the
    /// modal is open. Lifecycle: opened by `open_json_viewer`, which
    /// stamps `pending_correlation` and fires `ReadPodFile`; the
    /// matching `PodFileContent` arm populates `parsed` (on parse
    /// success) or `error` (on parse failure). Mirrors the egui
    /// sibling's `json_viewer_modal`.
    json_viewer_modal: Option<JsonViewerModalState>,
    /// Per-node collapsed/expanded membership for the JSON tree.
    /// Routed accordion keys (`json-tree:accordion:<path>`) toggle in
    /// and out of this set, just like the chat-log `open_accordions`
    /// pattern. Cleared when the modal closes so a re-open starts
    /// with the egui sibling's defaults (root-open, deeper-closed).
    json_tree_open: HashSet<String>,

    // ----- in-flight sudo approvals -----
    /// Server-emitted `SudoRequested` events the user hasn't resolved
    /// yet, keyed by `function_id`. Rendered as a banner stack above
    /// the matching thread's chat log. Drained by `SudoResolved`
    /// broadcasts (server-side resolution) and by the user clicking
    /// Approve / Remember / Reject (optimistic — we drop the entry
    /// before the server's echo lands so the banner doesn't flicker).
    /// Mirrors the egui sibling's `pending_sudos`.
    pending_sudos: HashMap<u64, PendingSudoState>,
    /// Per-banner reject-reason draft buffer, keyed by `function_id`.
    /// Populated only when the user types into the inline text input;
    /// flushed alongside the `pending_sudos` entry on resolve. Mirrors
    /// the egui sibling's `sudo_reject_drafts`.
    sudo_reject_drafts: HashMap<u64, String>,

    // ----- connection state -----
    conn_status: ConnectionStatus,
    conn_detail: Option<String>,
    /// Set on the first `ConnectionOpened`. Guards re-firing the
    /// initial-list batch on hot reconnects.
    list_requested: bool,

    // ----- catalog (server-broadcast tier) -----
    pods: HashMap<String, PodSummary>,
    threads: HashMap<String, ThreadSummary>,
    /// Server's "route `CreateThread { pod_id: None }` here" pod —
    /// echoed on every `PodList`. Used as the auto-select default
    /// for the sidebar's pod tabs on first connect, and as the
    /// "Default pod" sentinel resolution target.
    default_pod_id: Option<String>,

    // ----- sidebar nav -----
    /// Pod whose threads the sidebar is currently showing. Single
    /// active pod at a time (tabs idiom) so the sidebar has space
    /// to surface threads + behaviors + dispatch nesting per pod
    /// without competing for vertical real estate.
    pod_tab: Option<String>,
    /// Per-pod sidebar paginations: pods whose thread list is
    /// fully expanded (otherwise capped at [`SIDEBAR_THREAD_PREVIEW`]
    /// rows and a "Show N more" toggle).
    expanded_pod_threads: HashSet<String>,

    // ----- behaviors (per-pod registry) -----
    /// Behaviors per pod, keyed by `BehaviorSummary.pod_id`. Populated
    /// from `BehaviorList` responses (one per pod) and kept in sync by
    /// `BehaviorCreated` / `BehaviorUpdated` / `BehaviorDeleted` /
    /// `BehaviorStateChanged` broadcasts. Order within a pod is the
    /// server's (sorted by behavior_id today).
    behaviors_by_pod: HashMap<String, Vec<BehaviorSummary>>,
    /// Pods we've already issued a `ListBehaviors` for on this
    /// connection. Cleared on reconnect so a fresh `ConnectionOpened`
    /// re-fetches whatever pods come back in `PodList`.
    requested_behaviors_for: HashSet<String>,
    /// Expanded behavior rows in the sidebar: `"{pod_id}::{behavior_id}"`
    /// composite keys whose membership controls whether the behavior's
    /// spawned threads render nested below the row. Single global set
    /// (rather than per-pod) so a tab switch preserves the user's
    /// expand state across pods.
    expanded_behaviors: HashSet<String>,
    /// `(pod_id, behavior_id)` of a Delete button that's been
    /// armed (clicked once). The next click on the same button
    /// fires `DeleteBehavior`; any other click disarms. At most
    /// one armed delete at a time so the UI never has two
    /// "confirm" buttons live simultaneously.
    delete_armed_behavior: Option<(String, String)>,

    // ----- model catalog (request/response tier) -----
    /// Backends advertised by the server. Populated from
    /// `BackendsList`; drives the new-thread `Backend` picker. Entries
    /// arrive in the server's configured order.
    backends: Vec<BackendSummary>,
    /// Shared-MCP-host catalog. Populated from
    /// `SharedMcpHostsList`. Read by the structured pod editor's
    /// Allow tab (multi-check over names) and the behavior editor's
    /// scope tab (resource list narrowing).
    shared_mcp_hosts: Vec<whisper_agent_protocol::SharedMcpHostInfo>,
    /// Knowledge-bucket catalog. Populated from `BucketsList`. Read
    /// by the pod editor's Allow tab (multi-check over names).
    buckets: Vec<whisper_agent_protocol::BucketSummary>,
    /// Models per backend, keyed by `BackendSummary.name`. Populated
    /// lazily — we only fire `ListModels` for a backend when the user
    /// has actually picked it on the new-thread form, and we dedup
    /// repeat picks via [`requested_models_for`].
    models_by_backend: HashMap<String, Vec<ModelSummary>>,
    /// Backends we've already issued a `ListModels` for on this
    /// connection. Cleared on reconnect alongside [`subscribed`] so a
    /// fresh `ConnectionOpened` re-fetches whatever the user picks.
    requested_models_for: HashSet<String>,

    // ----- selection + per-thread state -----
    selected: Option<String>,
    /// Per-thread chat-log state. Populated on `ThreadSnapshot`; we
    /// only keep state for threads the user has actually opened so a
    /// long thread list doesn't bloat memory.
    views: HashMap<String, ThreadView>,
    /// Threads we've already sent `SubscribeToThread` for this
    /// connection. Cleared on reconnect (a fresh `ConnectionOpened`
    /// implies the server lost our subscriptions).
    subscribed: HashSet<String>,

    // ----- compose -----
    /// In-progress text for the *new-thread* compose form (no thread
    /// selected). Per-thread follow-up drafts live in [`drafts`].
    compose_input: String,
    /// Image attachments staged for the next send. Populated by the
    /// `UiEventKind::FileDropped` arm of `on_event` and the
    /// file-picker button on the compose bar; rendered as a
    /// thumbnail strip above the `text_area`; drained into the
    /// outbound `SendUserMessage` / `CreateThread` on submit. Lives
    /// on `ChatApp` (not per-`ThreadView`) because the user can
    /// stage a file before deciding which thread to send it on —
    /// drafts cross thread boundaries the same way.
    compose_attachments: Vec<StagedAttachment>,
    /// Monotonic counter for the stable id stamped on each staged
    /// attachment. Used by the `compose:attach:remove:{id}` route so
    /// pixel-identical drops the user staged twice are still
    /// addressable individually.
    next_attachment_id: u64,
    /// Async-picker handoff queue. The native file picker runs off
    /// the UI thread (rfd is async, driven on a tokio current-
    /// thread runtime in a background `std::thread::spawn`); when a
    /// pick resolves, it pushes a `RawPick` onto this queue, and
    /// `before_build` drains it on the next frame through the same
    /// staging pipeline that handles drops. `Arc<Mutex<...>>`
    /// rather than `Rc<RefCell<...>>` because the picker thread
    /// isn't the UI thread.
    pending_picks: std::sync::Arc<std::sync::Mutex<Vec<RawPick>>>,
    /// Ephemeral status line under the compose thumbnail strip —
    /// surfaces feedback on attach attempts (success, model
    /// rejection, MIME mismatch, read errors) so silent failures
    /// stop being silent. `(message, expires_at)` where
    /// `expires_at` is wall-clock; cleared once the deadline passes.
    compose_hint: Option<(String, std::time::Instant)>,
    /// Per-thread draft text, keyed by `thread_id`. Hydrated from
    /// [`ThreadSnapshot::draft`] on subscribe and from
    /// `ThreadDraftUpdated` broadcasts (other clients editing).
    /// Source of truth for the compose `text_area` whenever a thread
    /// is selected. Server-side persistence is via `SetThreadDraft`,
    /// which we fire on every text-changing edit — bandwidth is
    /// trivial vs. user perception of typing lag.
    drafts: HashMap<String, String>,
    /// Active prefill progress per thread, keyed by `thread_id`.
    /// `(processed, total)` from [`ServerToClient::ThreadPrefillProgress`].
    /// Cleared on first text/reasoning delta — the protocol says the
    /// indicator is ephemeral and disappears once the model starts
    /// emitting output.
    prefill: HashMap<String, (u32, u32)>,
    /// Global text-selection cursor for `text_area` / `text_input`
    /// widgets. Aetna's controlled widgets read it through
    /// `App::selection` and write back via `apply_event`. Reset
    /// on thread switch so the cursor lands at offset 0 of the
    /// newly-bound draft buffer (offsets index a different string).
    selection: Selection,

    // ----- accordion state -----
    /// Open accordion items, keyed by the routed key string the
    /// accordion runtime emits (`{group}:accordion:{value}` —
    /// produced by [`aetna_core::widgets::accordion::accordion_item_key`]).
    /// Reasoning and tool rows go through this; the egui sibling's
    /// per-row `Option<bool>` shape doesn't generalize when we have
    /// many independent rows in a single thread.
    open_accordions: HashSet<String>,

    // ----- new-thread compose pickers -----
    /// Backend picked for the next `CreateThread`. `None` means
    /// inherit the pod's `thread_defaults.backend`. The picker
    /// re-uses the empty string as the inherit sentinel on the wire
    /// — see [`PickerInherit`].
    picker_backend: Option<String>,
    picker_backend_open: bool,
    /// Model id picked for the next `CreateThread`. `None` falls
    /// through to the backend's `default_model` (or the first model
    /// the backend's `/models` returns).
    picker_model: Option<String>,
    picker_model_open: bool,
    /// Pod id the new thread should land in. `None` routes to the
    /// server's `compose_pod_id` default.
    picker_pod: Option<String>,
    picker_pod_open: bool,

    // ----- modals -----
    /// "+ New pod" dialog state. `Some` while the modal is open;
    /// `None` when no creation is in flight or no modal is rendered.
    /// Shape mirrors the egui sibling's `NewPodModalState` — id +
    /// display name, plus a client-side error slot for fast-fail
    /// validation feedback.
    new_pod_modal: Option<NewPodModalState>,
    /// "+ New behavior" dialog state. Opening seeds `pod_id` with
    /// the active sidebar tab so the dialog title can scope to the
    /// right pod and `submit` knows where to send `CreateBehavior`.
    /// `None` when closed.
    new_behavior_modal: Option<NewBehaviorModalState>,
    /// Behavior editor sheet state. `Some` while the right-hand sheet
    /// is open for a particular `(pod_id, behavior_id)` pair; `None`
    /// when closed. The form hydrates from a `GetBehavior` round-trip
    /// — the sheet renders a "loading…" placeholder until
    /// `BehaviorSnapshot` lands and populates `working_config` /
    /// `working_prompt`.
    behavior_editor: Option<BehaviorEditorSheetState>,
    /// Pod editor sheet state. `Some` while the right-hand raw-TOML
    /// editor is open for a pod; `None` when closed. Hydrates from
    /// a `GetPod` round-trip — the snapshot's `toml_text` becomes the
    /// edit buffer. v1 ships exactly the raw-TOML escape hatch; the
    /// structured tabs (allow lists, host_envs, MCP hosts, sandbox)
    /// land in follow-up slices once `sheet`-shaped form composition
    /// has more surface to share with the behavior editor.
    pod_editor: Option<PodEditorSheetState>,
    /// Fork-thread confirm dialog state. `Some` while the dialog is
    /// open; `None` when closed.
    fork_modal: Option<ForkModalState>,
    /// `(correlation_id, seed_text)` for an outstanding `ForkThread`.
    /// On the matching `ThreadCreated` echo the new thread's draft
    /// gets seeded with the original prompt — the user's typical
    /// next-step is "edit slightly and resend," and the draft slot
    /// is exactly where the new thread's compose box reads from.
    /// Mirrors the egui sibling's same-named field.
    pending_fork_seed: Option<(String, String)>,
    /// Monotonic counter for outgoing `correlation_id` strings.
    /// Each modal that sends a request claims an id and waits for
    /// the matching reply (success or `Error`) before closing or
    /// surfacing an error. Counter-based (not UUID) since the wire
    /// only needs uniqueness within this client session.
    correlation_counter: u64,
}

/// Form state for the "+ New pod" modal. Held inside an `Option` on
/// [`ChatApp`] — `Some` while the dialog is open. Reuses the egui
/// sibling's two-field shape (directory id + display name) plus a
/// `pending_correlation` slot so the close happens on the matching
/// `PodCreated` echo (rather than optimistically on click) and a
/// server-side `Error` reply lands in the same `error` field a
/// client-side validation failure does.
#[derive(Default)]
pub(crate) struct NewPodModalState {
    pub(crate) pod_id: String,
    pub(crate) name: String,
    /// Client- or server-side validation message. `None` while the
    /// form is unvalidated or has been edited since the last failure.
    pub(crate) error: Option<String>,
    /// Correlation id of the outstanding `CreatePod` request, if any.
    /// `Some` ⇒ the Create button is disabled (request in flight).
    /// Cleared on `PodCreated` (close) or `Error` (re-enable + error).
    pub(crate) pending_correlation: Option<String>,
}

/// Form state for the "+ New behavior" modal. Same shape as
/// [`NewPodModalState`] (id + display name + error +
/// pending_correlation) plus the `pod_id` the new behavior lands
/// under — mirrors the egui sibling. The new behavior starts as a
/// Manual-trigger stub with an empty prompt; the user fills the
/// rest in the (eventual) editor on the `BehaviorCreated`
/// round-trip.
pub(crate) struct NewBehaviorModalState {
    pub(crate) pod_id: String,
    pub(crate) behavior_id: String,
    pub(crate) name: String,
    pub(crate) error: Option<String>,
    pub(crate) pending_correlation: Option<String>,
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

/// Form state for the per-behavior editor sheet. Held inside an
/// `Option` on [`ChatApp`] — `Some` while the sheet is open. Hydrates
/// asynchronously: opening fires a `GetBehavior` and stashes the
/// correlation in `pending_get`; the matching `BehaviorSnapshot` reply
/// fills `working_config` / `working_prompt` (and the trigger-derived
/// editor fields). Until then the sheet renders a "loading…" body.
///
/// v1 surface mirrors the doc's call-out: name, description, prompt,
/// trigger kind, cron schedule. Thread overrides, scope narrowing,
/// retention policy, system-prompt override, and the raw-TOML escape
/// hatch are deferred — fields not exposed by the form ride through
/// `working_config` unchanged on save, so a v1 edit can't accidentally
/// strip them.
pub(crate) struct BehaviorEditorSheetState {
    pub(crate) pod_id: String,
    pub(crate) behavior_id: String,
    /// Working copy of the parsed config. `None` until the snapshot
    /// lands (or stays `None` permanently if the on-disk TOML failed
    /// to parse — `error` carries the reason and Save is disabled).
    pub(crate) working_config: Option<BehaviorConfig>,
    /// Buffer for the sibling `prompt.md`. Empty until the snapshot
    /// lands or when the behavior has no prompt.
    pub(crate) working_prompt: String,
    /// Selected trigger kind — tracked separately because
    /// [`TriggerSpec`] is a tagged enum, so name+description edits on
    /// `working_config` can't happen alongside a borrowed schedule
    /// string. Save rebuilds the `TriggerSpec` variant from this label
    /// plus [`Self::schedule_buffer`], preserving the existing
    /// variant's non-exposed fields (timezone, overlap, catch_up)
    /// when possible.
    pub(crate) working_kind: TriggerKindLabel,
    /// Cron schedule edit buffer. Meaningful only when `working_kind
    /// == Cron`; preserved across kind switches so toggling away and
    /// back doesn't lose the user's typed schedule.
    pub(crate) schedule_buffer: String,
    /// Cron timezone edit buffer. Meaningful only when `working_kind
    /// == Cron`; same preservation discipline as
    /// [`Self::schedule_buffer`].
    pub(crate) timezone_buffer: String,
    /// Cron `overlap` policy buffer. Reused under Webhook for the
    /// same field, so kind-switching between Cron and Webhook
    /// preserves the choice.
    pub(crate) overlap_buffer: Overlap,
    /// Cron `catch_up` policy buffer.
    pub(crate) catch_up_buffer: CatchUp,
    /// Active editor tab. Sub-tabs are placeholders for now;
    /// follow-up slices fill them in.
    pub(crate) tab: BehaviorEditorTab,
    /// Whether the trigger-kind `select_menu` popover is open. One
    /// menu at a time across the whole app — opening this menu closes
    /// the new-thread compose pickers via `close_other_pickers`.
    pub(crate) trigger_kind_open: bool,
    /// Which `select_menu` (if any) is currently open inside the
    /// editor. Single-active-at-a-time discipline shared with the
    /// pod editor's pickers via `close_other_pickers`.
    pub(crate) open_picker: Option<BehaviorEditorPicker>,
    /// Thread-tab `max_tokens` override numeric buffer. Empty when
    /// the override is `None`. Same parse-on-edit-and-write-back
    /// pattern as the pod editor's `max_tokens_buf`.
    pub(crate) thread_max_tokens_buf: String,
    /// Thread-tab `max_turns` override numeric buffer. Same shape
    /// as [`Self::thread_max_tokens_buf`].
    pub(crate) thread_max_turns_buf: String,
    /// Retention-tab `days` numeric buffer. Meaningful only when
    /// `cfg.on_completion` is `ArchiveAfterDays` or `DeleteAfterDays`;
    /// preserved across kind switches so toggling Keep ↔ Archive
    /// doesn't lose typed days.
    pub(crate) retention_days_buf: String,
    /// SystemPrompt-tab content buffer for the side
    /// `behaviors/<id>/system_prompt.md` file. Hydrated from
    /// `snapshot.system_prompt`. Round-trips on save via
    /// `UpdateBehavior.system_prompt`. `None` ⇒ no buffer to ship
    /// (override is off, or the snapshot didn't carry content).
    pub(crate) working_system_prompt: Option<String>,
    /// Validation message: load error from the snapshot, client-side
    /// validation failure, or server-side `Error` echo.
    pub(crate) error: Option<String>,
    /// Correlation id of the in-flight `GetBehavior` request, if any.
    /// `Some` until the matching `BehaviorSnapshot` arrives.
    pub(crate) pending_get: Option<String>,
    /// Correlation id of the in-flight `UpdateBehavior` request, if
    /// any. `Some` ⇒ the Save button is disabled. Cleared on
    /// `BehaviorUpdated` (close) or `Error` (re-enable + surface
    /// message).
    pub(crate) pending_save: Option<String>,
    /// Pod's parsed config — provides the `[allow]` lists the
    /// host_env / mcp_hosts override rows render against. Hydrated
    /// asynchronously via a parallel `GetPod` round-trip when the
    /// editor opens; `None` until the snapshot arrives, in which case
    /// the override rows render a "(loading pod config…)" placeholder.
    pub(crate) pod_config: Option<PodConfig>,
    /// Correlation id of the in-flight `GetPod` request that fires
    /// alongside `GetBehavior` on editor open. `Some` until the
    /// matching `PodSnapshot` arrives.
    pub(crate) pending_pod_get: Option<String>,
    /// Raw TOML buffer for the Raw tab. Mirrors the pod editor's
    /// `working_toml`: the structured tabs serialize their
    /// `working_config` here on tab-enter, and Raw → structured
    /// reparses this back into `working_config`. Empty until the
    /// snapshot lands (or until the user first visits the Raw tab).
    pub(crate) working_toml: String,
    /// Whether the user has typed into the Raw tab since the last
    /// re-serialize. Save uses this to decide between shipping the
    /// raw buffer's parse and the structured `working_config`.
    pub(crate) raw_dirty: bool,
}

/// Discriminator used by the editor's trigger-kind picker. Mirrors
/// the on-wire `TriggerSpec` tag values (`"manual" | "cron" |
/// "webhook"`) so persisting and reading round-trips through the
/// label string the picker carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TriggerKindLabel {
    Manual,
    Cron,
    Webhook,
}

impl TriggerKindLabel {
    fn wire_value(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Cron => "cron",
            Self::Webhook => "webhook",
        }
    }

    fn display_label(self) -> &'static str {
        match self {
            Self::Manual => "Manual",
            Self::Cron => "Cron",
            Self::Webhook => "Webhook",
        }
    }

    fn from_wire(value: &str) -> Option<Self> {
        match value {
            "manual" => Some(Self::Manual),
            "cron" => Some(Self::Cron),
            "webhook" => Some(Self::Webhook),
            _ => None,
        }
    }

    fn from_trigger(spec: &TriggerSpec) -> Self {
        match spec {
            TriggerSpec::Manual => Self::Manual,
            TriggerSpec::Cron { .. } => Self::Cron,
            TriggerSpec::Webhook { .. } => Self::Webhook,
        }
    }
}

/// Editor tab discriminator for the behavior editor sheet's
/// multi-tab body. Mirrors the egui sibling's `BehaviorEditorTab`
/// (Trigger / Thread / Scope / Retention / Prompt / SystemPrompt /
/// RawToml). Most non-Trigger tabs render placeholder paragraphs
/// in the first slice; follow-up slices fill them in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BehaviorEditorTab {
    Trigger,
    Thread,
    Scope,
    Retention,
    Prompt,
    SystemPrompt,
    RawToml,
}

impl BehaviorEditorTab {
    fn label(self) -> &'static str {
        // "Retain" rather than "Retention" so all 7 labels fit the
        // 600 px sheet's 7-tab strip without TextOverflow lint
        // findings — at the strip's per-tab width budget,
        // "Retention" overflows by ~7 px and ellipsizes ugly.
        match self {
            Self::Trigger => "Trigger",
            Self::Thread => "Thread",
            Self::Scope => "Scope",
            Self::Retention => "Retain",
            Self::Prompt => "Prompt",
            Self::SystemPrompt => "System",
            Self::RawToml => "Raw",
        }
    }

    fn wire_value(self) -> &'static str {
        match self {
            Self::Trigger => "trigger",
            Self::Thread => "thread",
            Self::Scope => "scope",
            Self::Retention => "retention",
            Self::Prompt => "prompt",
            Self::SystemPrompt => "system_prompt",
            Self::RawToml => "raw",
        }
    }

    fn from_wire(value: &str) -> Option<Self> {
        match value {
            "trigger" => Some(Self::Trigger),
            "thread" => Some(Self::Thread),
            "scope" => Some(Self::Scope),
            "retention" => Some(Self::Retention),
            "prompt" => Some(Self::Prompt),
            "system_prompt" => Some(Self::SystemPrompt),
            "raw" => Some(Self::RawToml),
            _ => None,
        }
    }
}

/// Picker discriminator for the behavior editor's `select_trigger`
/// family. Single-active across the whole sheet via
/// `close_other_pickers`. The trigger-kind picker stays on the
/// dedicated `trigger_kind_open: bool` boolean (it pre-dates this
/// enum) — adding it here would require a larger flow rework.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BehaviorEditorPicker {
    /// Cron `overlap` policy picker.
    Overlap,
    /// Cron `catch_up` policy picker.
    CatchUp,
    /// Thread-tab `thread.model` override picker. Options are the
    /// pod's effective backend's model catalog.
    ThreadModel,
    /// Thread-tab `thread.bindings.backend` override picker.
    /// Options are the server-known backend catalog.
    ThreadBackend,
    /// Retention-tab kind picker (Keep / ArchiveAfterDays /
    /// DeleteAfterDays).
    RetentionKind,
    /// Scope-tab `scope.tools.default` Disposition picker
    /// (Allow / Deny). Only meaningful when the tools override is on.
    ScopeToolsDefault,
    /// Scope-tab cap-narrowing pickers — same option sets as the
    /// pod editor's `[allow.caps]` triggers, but the field is
    /// `Option<Cap>` here (None = inherit pod ceiling).
    ScopeCapsPodModify,
    ScopeCapsDispatch,
    ScopeCapsBehaviors,
}

impl BehaviorEditorPicker {
    fn key(self) -> &'static str {
        match self {
            Self::Overlap => BEHAVIOR_EDITOR_OVERLAP_KEY,
            Self::CatchUp => BEHAVIOR_EDITOR_CATCH_UP_KEY,
            Self::ThreadModel => BEHAVIOR_EDITOR_THREAD_MODEL_KEY,
            Self::ThreadBackend => BEHAVIOR_EDITOR_THREAD_BACKEND_KEY,
            Self::RetentionKind => BEHAVIOR_EDITOR_RETENTION_KIND_KEY,
            Self::ScopeToolsDefault => BEHAVIOR_EDITOR_SCOPE_TOOLS_DEFAULT_KEY,
            Self::ScopeCapsPodModify => BEHAVIOR_EDITOR_SCOPE_CAPS_POD_MODIFY_KEY,
            Self::ScopeCapsDispatch => BEHAVIOR_EDITOR_SCOPE_CAPS_DISPATCH_KEY,
            Self::ScopeCapsBehaviors => BEHAVIOR_EDITOR_SCOPE_CAPS_BEHAVIORS_KEY,
        }
    }
}

impl BehaviorEditorSheetState {
    fn new(
        pod_id: String,
        behavior_id: String,
        pending_get: String,
        pending_pod_get: String,
    ) -> Self {
        Self {
            pod_id,
            behavior_id,
            working_config: None,
            working_prompt: String::new(),
            working_kind: TriggerKindLabel::Manual,
            schedule_buffer: String::new(),
            timezone_buffer: "UTC".into(),
            overlap_buffer: Overlap::default(),
            catch_up_buffer: CatchUp::default(),
            tab: BehaviorEditorTab::Trigger,
            trigger_kind_open: false,
            open_picker: None,
            thread_max_tokens_buf: String::new(),
            thread_max_turns_buf: String::new(),
            retention_days_buf: "30".into(),
            working_system_prompt: None,
            error: None,
            pending_get: Some(pending_get),
            pending_save: None,
            pod_config: None,
            pending_pod_get: Some(pending_pod_get),
            working_toml: String::new(),
            raw_dirty: false,
        }
    }

    /// Hydrate the working state from a [`BehaviorSnapshot`]. Called
    /// once when the matching `GetBehavior` round-trip completes. Sets
    /// `working_kind` + the cron buffers from the trigger variant so
    /// the form widgets render against the snapshot's values. A
    /// `load_error` from disk lands in `error` (and `working_config`
    /// stays `None`) so the user sees the parse failure rather than
    /// editing against a phantom default.
    fn hydrate(&mut self, snapshot: whisper_agent_protocol::BehaviorSnapshot) {
        self.pending_get = None;
        self.error = snapshot.load_error.clone();
        if let Some(cfg) = snapshot.config.as_ref() {
            self.working_kind = TriggerKindLabel::from_trigger(&cfg.trigger);
            match &cfg.trigger {
                TriggerSpec::Cron {
                    schedule,
                    timezone,
                    overlap,
                    catch_up,
                } => {
                    self.schedule_buffer = schedule.clone();
                    self.timezone_buffer = timezone.clone();
                    self.overlap_buffer = *overlap;
                    self.catch_up_buffer = *catch_up;
                }
                TriggerSpec::Webhook { overlap } => {
                    // Webhook reuses the overlap buffer; leave the
                    // schedule/timezone/catch_up at their previous
                    // values so kind-switching back to Cron preserves
                    // the user's typing.
                    self.overlap_buffer = *overlap;
                }
                TriggerSpec::Manual => {
                    // Manual carries no scheduling fields. Leave the
                    // buffers untouched for the same kind-switching
                    // continuity reason.
                }
            }
        }
        self.working_config = snapshot.config;
        self.working_prompt = snapshot.prompt;
        self.working_system_prompt = snapshot.system_prompt;
        self.sync_thread_buffers_from_config();
        self.sync_retention_buffer_from_config();
        // Seed the Raw tab's buffer from the structured config so a
        // first-time visit to Raw shows the snapshot's TOML rather
        // than an empty pane. Failures (shouldn't happen — every
        // BehaviorConfig round-trips) leave the buffer at "".
        if let Some(cfg) = self.working_config.as_ref()
            && let Ok(text) = toml::to_string_pretty(cfg)
        {
            self.working_toml = text;
        }
        self.raw_dirty = false;
    }

    /// Tab-switch with the same raw↔structured sync the pod editor
    /// uses (`PodEditorSheetState::switch_tab`). Leaving Raw with
    /// `raw_dirty` reparses the buffer back into `working_config` —
    /// a parse error keeps the user on Raw with the message in
    /// `error` so the typed text isn't lost. Entering Raw re-
    /// serializes from `working_config` so the buffer reflects
    /// whatever structured edits landed since last enter.
    fn switch_tab(&mut self, target: BehaviorEditorTab) {
        if self.tab == target {
            return;
        }
        let leaving_raw = self.tab == BehaviorEditorTab::RawToml;
        let entering_raw = target == BehaviorEditorTab::RawToml;
        if leaving_raw && self.raw_dirty {
            match toml::from_str::<BehaviorConfig>(&self.working_toml) {
                Ok(parsed) => {
                    self.working_config = Some(parsed);
                    self.raw_dirty = false;
                    self.error = None;
                    // Re-derive structured-tab state from the new
                    // parsed config: trigger kind / cron buffers, the
                    // numeric / retention buffers, and the system-
                    // prompt buffer. Same shape as `hydrate` minus the
                    // wire-snapshot bits.
                    if let Some(cfg) = self.working_config.as_ref() {
                        self.working_kind = TriggerKindLabel::from_trigger(&cfg.trigger);
                        if let TriggerSpec::Cron {
                            schedule,
                            timezone,
                            overlap,
                            catch_up,
                        } = &cfg.trigger
                        {
                            self.schedule_buffer = schedule.clone();
                            self.timezone_buffer = timezone.clone();
                            self.overlap_buffer = *overlap;
                            self.catch_up_buffer = *catch_up;
                        }
                    }
                    self.sync_thread_buffers_from_config();
                    self.sync_retention_buffer_from_config();
                }
                Err(e) => {
                    self.error = Some(format!("raw TOML doesn't parse: {e}"));
                    return; // stay on Raw
                }
            }
        }
        if entering_raw && let Some(cfg) = self.working_config.as_ref() {
            // Re-serialize from working_config so the raw buffer
            // reflects every structured edit since last enter.
            // Apply the same trigger-rebuild submit_behavior_editor
            // does so the round-trip captures pending kind / cron
            // buffer edits, not just whatever's parked on the
            // working trigger.
            let mut snapshot = cfg.clone();
            snapshot.trigger = self.resolved_trigger();
            if let Ok(text) = toml::to_string_pretty(&snapshot) {
                self.working_toml = text;
            }
            self.raw_dirty = false;
        }
        self.tab = target;
    }

    /// Pull the Retention tab's `days` buffer out of
    /// `working_config.on_completion`. Called on hydrate. Defaulted
    /// to "30" so a Keep → Archive flip in the live editor lands at
    /// the same default the egui sibling uses.
    fn sync_retention_buffer_from_config(&mut self) {
        if let Some(cfg) = self.working_config.as_ref() {
            self.retention_days_buf = match &cfg.on_completion {
                RetentionPolicy::Keep => "30".to_string(),
                RetentionPolicy::ArchiveAfterDays { days }
                | RetentionPolicy::DeleteAfterDays { days } => days.to_string(),
            };
        }
    }

    /// Pull the Thread-tab numeric override buffers (`max_tokens` /
    /// `max_turns`) out of `working_config.thread`. Called on hydrate
    /// only — buffer mutations happen inline in the on_event handler
    /// (parse-on-edit-and-write-back), so subsequent calls to this
    /// helper would clobber the user's typing.
    fn sync_thread_buffers_from_config(&mut self) {
        if let Some(cfg) = self.working_config.as_ref() {
            self.thread_max_tokens_buf = cfg
                .thread
                .max_tokens
                .map(|n| n.to_string())
                .unwrap_or_default();
            self.thread_max_turns_buf = cfg
                .thread
                .max_turns
                .map(|n| n.to_string())
                .unwrap_or_default();
        }
    }

    /// Resolve the form state into a `TriggerSpec` for `UpdateBehavior`.
    /// Each variant pulls from the live editor buffers — for kinds the
    /// user hasn't touched, the buffers were seeded from the snapshot
    /// at hydrate time, so the round-trip preserves on-disk values.
    fn resolved_trigger(&self) -> TriggerSpec {
        match self.working_kind {
            TriggerKindLabel::Manual => TriggerSpec::Manual,
            TriggerKindLabel::Cron => TriggerSpec::Cron {
                schedule: self.schedule_buffer.trim().to_string(),
                timezone: self.timezone_buffer.trim().to_string(),
                overlap: self.overlap_buffer,
                catch_up: self.catch_up_buffer,
            },
            TriggerKindLabel::Webhook => TriggerSpec::Webhook {
                overlap: self.overlap_buffer,
            },
        }
    }
}

/// State for the "fork from this message" confirm dialog. Opened
/// from a per-User-row fork icon; closes (without sending anything)
/// on cancel or scrim dismiss, or fires `ForkThread` on confirm and
/// closes immediately. We don't track a `pending_correlation` here
/// because the server doesn't echo a `ForkThread`-specific reply —
/// the matching `ThreadCreated` lands through the normal arm and
/// (when correlation matches `pending_fork_seed`'s id) seeds the
/// new thread's draft with the captured prompt text.
pub(crate) struct ForkModalState {
    /// The thread we're forking off of. Captured at click time so
    /// the dialog still routes correctly even if the user re-selects
    /// before clicking Fork.
    pub(crate) thread_id: String,
    /// Wire's `from_message_index` — the conversation index this
    /// fork replays up to (but not including).
    pub(crate) from_message_index: usize,
    /// User-message text the user clicked fork on. Used as the draft
    /// seed for the new thread once `ThreadCreated` lands.
    pub(crate) seed_text: String,
    /// Archive the source thread on confirm. Default `true` —
    /// fork is almost always "I want to try something different
    /// from here," and the original typically becomes noise.
    pub(crate) archive_original: bool,
    /// Re-derive scope / bindings / config from the pod's current
    /// `thread_defaults` instead of inheriting from the source
    /// thread. Default `false` (inherit) — most forks are "continue
    /// where I left off" with the source's settings.
    pub(crate) reset_capabilities: bool,
}

/// Form state for the pod editor sheet. Multi-tab surface mirroring
/// the egui sibling: `Allow` (identity + allowed resources + caps),
/// `Defaults` (thread-defaults pre-fill), `Limits` (pod-level caps),
/// `RawToml` (always-available escape hatch). On hydrate the
/// snapshot's `config` lands as the structured `working_config` and
/// its `toml_text` becomes the raw buffer; the two are kept in sync
/// across tab switches by re-serializing or re-parsing as the user
/// crosses the structured/raw boundary. Save ships the raw buffer
/// when the user last edited the raw tab, otherwise re-serializes
/// `working_config` — same `UpdatePodConfig { toml_text }` wire shape.
pub(crate) struct PodEditorSheetState {
    pub(crate) pod_id: String,
    /// Parsed working copy. `None` only when the snapshot's `config`
    /// failed to parse server-side (rare — the config is loaded from
    /// disk and validated). Mutated by the structured tabs; the raw
    /// tab snapshots from this on entry and projects back on exit.
    pub(crate) working_config: Option<PodConfig>,
    /// Raw `pod.toml` buffer for the RawToml tab. Re-serialized from
    /// `working_config` on tab-enter; re-parsed on tab-leave when
    /// [`Self::raw_dirty`] says the user has typed.
    pub(crate) working_toml: String,
    /// Server-known baseline. Save short-circuits when nothing has
    /// changed (raw text matches AND working_config matches).
    pub(crate) baseline_toml: String,
    /// Snapshot of the original parsed config. Mirrors `baseline_toml`
    /// on the structured side so a future Revert button has both.
    pub(crate) baseline_config: Option<PodConfig>,
    /// Active tab. Tab-switch logic triggers structured/raw sync.
    pub(crate) tab: PodEditorTab,
    /// Set when the raw buffer has diverged from the last
    /// re-serialization. Drives "leave raw → reparse" logic + the
    /// Save path's choice between raw text and re-serialized config.
    pub(crate) raw_dirty: bool,
    /// Validation message: server `Error` echo (parse failure or
    /// validation reject) or the editor's own diagnostics
    /// (raw-tab parse failure on tab-switch). Cleared on the next
    /// edit.
    pub(crate) error: Option<String>,
    /// Correlation id of the in-flight `GetPod` request, if any.
    pub(crate) pending_get: Option<String>,
    /// Correlation id of the in-flight `UpdatePodConfig` request,
    /// if any. Cleared on `PodConfigUpdated` (close) or `Error`
    /// (re-enable + surface message).
    pub(crate) pending_save: Option<String>,
    /// Which `select_menu` (if any) is currently open inside the
    /// editor. Single-active-at-a-time — the new-thread compose
    /// pickers and the behavior editor's trigger-kind picker
    /// coordinate through `close_other_pickers`.
    pub(crate) open_picker: Option<PodEditorPicker>,
    /// Visible string for the Defaults-tab `max_tokens`
    /// `numeric_input`. The widget owns its own buffer (lets the user
    /// type mid-edit states like `"1"` before the trailing digits
    /// arrive); we parse and write back to
    /// `working_config.thread_defaults.max_tokens` on every parseable
    /// edit. Hydrated from the snapshot's parsed config; re-synced
    /// when leaving Raw with `raw_dirty` reparses.
    pub(crate) max_tokens_buf: String,
    /// Same shape as [`Self::max_tokens_buf`], for
    /// `thread_defaults.max_turns`.
    pub(crate) max_turns_buf: String,
    /// Same shape as [`Self::max_tokens_buf`], for
    /// `limits.max_concurrent_threads`.
    pub(crate) max_concurrent_threads_buf: String,
    /// Visible string for the Defaults-tab tool-surface
    /// `core_tools` named-list `text_area`. Holds the multiline
    /// edit buffer (one tool name per line) while the user types;
    /// every keystroke parses + writes back to
    /// `working_config.thread_defaults.tool_surface.core_tools` as
    /// `CoreTools::Named(parsed)`. Only meaningful when
    /// `core_tools` is `Named`; toggling to `All` leaves the buffer
    /// alone so a user's draft survives the round-trip.
    pub(crate) tool_surface_named_buf: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PodEditorPicker {
    /// `allow.caps.pod_modify_thread_pods` cap selector.
    AllowCapsPodModify,
    /// `allow.caps.dispatch_threads_in_pods` cap selector.
    AllowCapsDispatch,
    /// `allow.caps.use_behaviors_in_pods` cap selector.
    AllowCapsBehaviors,
    /// `thread_defaults.backend` picker (Defaults tab).
    DefaultsBackend,
    /// `thread_defaults.model` picker (Defaults tab).
    DefaultsModel,
    /// `allow.tools.default` (Allow / Deny) picker (Defaults tab —
    /// the gate lives under `[allow]` but its default is a
    /// thread-default knob in spirit, so the egui sibling parks
    /// it here).
    DefaultsToolGate,
    /// `thread_defaults.caps.pod_modify` selector.
    DefaultsCapsPodModify,
    /// `thread_defaults.caps.dispatch` selector.
    DefaultsCapsDispatch,
    /// `thread_defaults.caps.behaviors` selector.
    DefaultsCapsBehaviors,
}

impl PodEditorPicker {
    fn key(self) -> &'static str {
        match self {
            Self::AllowCapsPodModify => POD_EDITOR_ALLOW_CAPS_POD_MODIFY_KEY,
            Self::AllowCapsDispatch => POD_EDITOR_ALLOW_CAPS_DISPATCH_KEY,
            Self::AllowCapsBehaviors => POD_EDITOR_ALLOW_CAPS_BEHAVIORS_KEY,
            Self::DefaultsBackend => POD_EDITOR_DEFAULTS_BACKEND_KEY,
            Self::DefaultsModel => POD_EDITOR_DEFAULTS_MODEL_KEY,
            Self::DefaultsToolGate => POD_EDITOR_DEFAULTS_TOOL_GATE_KEY,
            Self::DefaultsCapsPodModify => POD_EDITOR_DEFAULTS_CAPS_POD_MODIFY_KEY,
            Self::DefaultsCapsDispatch => POD_EDITOR_DEFAULTS_CAPS_DISPATCH_KEY,
            Self::DefaultsCapsBehaviors => POD_EDITOR_DEFAULTS_CAPS_BEHAVIORS_KEY,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PodEditorTab {
    Allow,
    Defaults,
    Limits,
    RawToml,
}

impl PodEditorTab {
    fn label(self) -> &'static str {
        match self {
            PodEditorTab::Allow => "Allow",
            PodEditorTab::Defaults => "Defaults",
            PodEditorTab::Limits => "Limits",
            PodEditorTab::RawToml => "Raw",
        }
    }

    fn wire_value(self) -> &'static str {
        match self {
            PodEditorTab::Allow => "allow",
            PodEditorTab::Defaults => "defaults",
            PodEditorTab::Limits => "limits",
            PodEditorTab::RawToml => "raw",
        }
    }

    fn from_wire(value: &str) -> Option<Self> {
        match value {
            "allow" => Some(PodEditorTab::Allow),
            "defaults" => Some(PodEditorTab::Defaults),
            "limits" => Some(PodEditorTab::Limits),
            "raw" => Some(PodEditorTab::RawToml),
            _ => None,
        }
    }
}

impl PodEditorSheetState {
    fn new(pod_id: String, pending_get: String) -> Self {
        Self {
            pod_id,
            working_config: None,
            working_toml: String::new(),
            baseline_toml: String::new(),
            baseline_config: None,
            tab: PodEditorTab::Allow,
            raw_dirty: false,
            error: None,
            pending_get: Some(pending_get),
            pending_save: None,
            open_picker: None,
            max_tokens_buf: String::new(),
            max_turns_buf: String::new(),
            max_concurrent_threads_buf: String::new(),
            tool_surface_named_buf: String::new(),
        }
    }

    fn hydrate(&mut self, snapshot: whisper_agent_protocol::PodSnapshot) {
        self.pending_get = None;
        self.error = None;
        self.working_toml = snapshot.toml_text.clone();
        self.baseline_toml = snapshot.toml_text;
        self.working_config = Some(snapshot.config.clone());
        self.baseline_config = Some(snapshot.config);
        self.raw_dirty = false;
        self.sync_buffers_from_config();
    }

    /// Pull the Defaults-tab numeric buffers (`max_tokens` /
    /// `max_turns`) out of the structured `working_config`. Called on
    /// hydrate and after Raw → structured reparses so the
    /// `numeric_input` widgets always start aligned with the
    /// underlying u32s.
    fn sync_buffers_from_config(&mut self) {
        if let Some(cfg) = self.working_config.as_ref() {
            self.max_tokens_buf = cfg.thread_defaults.max_tokens.to_string();
            self.max_turns_buf = cfg.thread_defaults.max_turns.to_string();
            self.max_concurrent_threads_buf = cfg.limits.max_concurrent_threads.to_string();
            // Seed the tool_surface named buffer from the parsed
            // CoreTools. `All` leaves the buffer at the conventional
            // default (describe_tool / find_tool / request_escalation)
            // so the textarea isn't blank when the user toggles
            // `All` → `Named`.
            self.tool_surface_named_buf = match &cfg.thread_defaults.tool_surface.core_tools {
                CoreTools::All => default_core_tools_text(),
                CoreTools::Named(list) => list.join("\n"),
            };
        }
    }

    /// Has anything diverged from the server-known baseline? Save
    /// uses this to gate the "no changes to save" short-circuit.
    fn dirty(&self) -> bool {
        if self.raw_dirty && self.working_toml != self.baseline_toml {
            return true;
        }
        match (&self.working_config, &self.baseline_config) {
            (Some(w), Some(b)) => w != b,
            _ => false,
        }
    }

    /// Switch tabs with the egui-style sync invariant:
    /// - leaving Raw with `raw_dirty` ⇒ reparse the buffer back into
    ///   `working_config`. Parse failures keep the user on Raw with
    ///   the error in `error` (don't drop edits).
    /// - entering Raw ⇒ re-serialize `working_config` so the buffer
    ///   reflects the structured edits. Reset `raw_dirty`.
    fn switch_tab(&mut self, target: PodEditorTab) {
        if self.tab == target {
            return;
        }
        let leaving_raw = self.tab == PodEditorTab::RawToml;
        let entering_raw = target == PodEditorTab::RawToml;
        if leaving_raw && self.raw_dirty {
            match toml::from_str::<PodConfig>(&self.working_toml) {
                Ok(parsed) => {
                    self.working_config = Some(parsed);
                    self.raw_dirty = false;
                    self.error = None;
                    // Re-sync numeric buffers so the Defaults tab's
                    // max_tokens / max_turns widgets reflect whatever
                    // the user typed into the raw TOML.
                    self.sync_buffers_from_config();
                }
                Err(e) => {
                    self.error = Some(format!("raw TOML doesn't parse: {e}"));
                    return; // stay on Raw
                }
            }
        }
        if entering_raw && let Some(cfg) = self.working_config.as_ref() {
            // Re-serialize from working_config so the raw buffer
            // reflects every structured edit since last enter. Falls
            // back to leaving the buffer alone on serialize error
            // (shouldn't happen — PodConfig round-trips cleanly).
            if let Ok(text) = toml::to_string_pretty(cfg) {
                self.working_toml = text;
            }
            self.raw_dirty = false;
        }
        self.tab = target;
    }

    /// The TOML text Save should ship. If the user last edited the
    /// raw tab and didn't switch back, ship the raw buffer (preserves
    /// any TOML-level shape they typed — comments, key order, etc.).
    /// Otherwise re-serialize the structured config.
    fn resolved_save_toml(&self) -> Result<String, String> {
        if self.tab == PodEditorTab::RawToml && self.raw_dirty {
            return Ok(self.working_toml.clone());
        }
        let Some(cfg) = self.working_config.as_ref() else {
            return Err(
                "no parsed config to save — fix the on-disk pod.toml from the Raw tab".into(),
            );
        };
        toml::to_string_pretty(cfg).map_err(|e| format!("couldn't serialize config: {e}"))
    }
}

impl ChatApp {
    pub fn new(inbound: Inbound, send_fn: SendFn) -> Self {
        Self {
            inbound,
            send_fn,
            server_label: None,
            sidebar_width: tokens::SIDEBAR_WIDTH,
            sidebar_drag: ResizeDrag::default(),
            lightbox: None,
            inspector_open: None,
            buckets_modal: None,
            settings_modal: None,
            file_viewer_modal: None,
            file_tree_modal_pod: None,
            pod_files: HashMap::new(),
            pod_files_requested: HashSet::new(),
            pod_dirs_open: HashSet::new(),
            json_viewer_modal: None,
            json_tree_open: HashSet::new(),
            pending_sudos: HashMap::new(),
            sudo_reject_drafts: HashMap::new(),
            conn_status: ConnectionStatus::Connecting,
            conn_detail: None,
            list_requested: false,
            pods: HashMap::new(),
            threads: HashMap::new(),
            default_pod_id: None,
            pod_tab: None,
            expanded_pod_threads: HashSet::new(),
            behaviors_by_pod: HashMap::new(),
            requested_behaviors_for: HashSet::new(),
            expanded_behaviors: HashSet::new(),
            delete_armed_behavior: None,
            backends: Vec::new(),
            shared_mcp_hosts: Vec::new(),
            buckets: Vec::new(),
            models_by_backend: HashMap::new(),
            requested_models_for: HashSet::new(),
            selected: None,
            views: HashMap::new(),
            subscribed: HashSet::new(),
            compose_input: String::new(),
            compose_attachments: Vec::new(),
            next_attachment_id: 0,
            pending_picks: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            compose_hint: None,
            drafts: HashMap::new(),
            prefill: HashMap::new(),
            selection: Selection::default(),
            open_accordions: HashSet::new(),
            picker_backend: None,
            picker_backend_open: false,
            picker_model: None,
            picker_model_open: false,
            picker_pod: None,
            picker_pod_open: false,
            new_pod_modal: None,
            new_behavior_modal: None,
            behavior_editor: None,
            pod_editor: None,
            fork_modal: None,
            pending_fork_seed: None,
            correlation_counter: 0,
        }
    }

    /// Tell the chat surface which server URL this client is connected
    /// to. Surfaced as a small line in the sidebar footer so users on
    /// machines with multiple windows / saved profiles can see at a
    /// glance which agent they're talking to. The host shell typically
    /// passes the same string the user typed at the login form (or the
    /// `--server` CLI flag).
    pub fn set_server_label(&mut self, label: impl Into<String>) {
        let s = label.into();
        self.server_label = if s.is_empty() { None } else { Some(s) };
    }

    /// Allocate the next correlation id. Counter-based string suffices
    /// for matching modal pendings — only this session's client
    /// generates these, and a u64 won't wrap in any realistic session.
    fn next_correlation_id(&mut self) -> String {
        self.correlation_counter += 1;
        format!("aui-{}", self.correlation_counter)
    }

    fn send(&self, msg: ClientToServer) {
        (self.send_fn)(msg);
    }

    fn drain_inbound(&mut self) {
        // Drain into a local Vec so the borrow on `self.inbound` is
        // released before each handler runs — handlers may want to
        // re-enqueue (today they don't, but future stages with
        // optimistic local state will).
        let drained: Vec<InboundEvent> = self.inbound.borrow_mut().drain(..).collect();
        for event in drained {
            self.handle_event(event);
        }
    }

    fn handle_event(&mut self, event: InboundEvent) {
        match event {
            InboundEvent::ConnectionOpened => {
                self.conn_status = ConnectionStatus::Connected;
                self.conn_detail = None;
                // Server drops our subscriptions on reconnect; reset
                // the local mirror so re-selecting a thread re-asks
                // for its snapshot.
                self.subscribed.clear();
                // Same story for the per-backend `ListModels` dedup —
                // a fresh socket means we have to re-fetch whatever
                // the user picks.
                self.requested_models_for.clear();
                // Per-pod `ListBehaviors` dedup also reset; the next
                // `PodList` will re-fan-out one request per pod.
                self.requested_behaviors_for.clear();
                // File-tree caches die on reconnect — server may have
                // restarted with a different on-disk layout, and the
                // pending-request guard would otherwise wedge.
                self.pod_files.clear();
                self.pod_files_requested.clear();
                if !self.list_requested {
                    self.send(ClientToServer::ListPods {
                        correlation_id: None,
                    });
                    self.send(ClientToServer::ListThreads {
                        correlation_id: None,
                    });
                    self.send(ClientToServer::ListBackends {
                        correlation_id: None,
                    });
                    // Catalog tier consumed by the structured pod
                    // editor (Allow tab) + behavior editor scope
                    // tabs. Cheap on the server side; both replies
                    // are pure registry reads with no I/O.
                    self.send(ClientToServer::ListSharedMcpHosts {
                        correlation_id: None,
                    });
                    self.send(ClientToServer::ListBuckets {
                        correlation_id: None,
                    });
                    self.list_requested = true;
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
            InboundEvent::Wire(msg) => self.dispatch_wire(msg),
        }
    }

    fn dispatch_wire(&mut self, msg: ServerToClient) {
        match msg {
            ServerToClient::PodList {
                pods,
                default_pod_id,
                ..
            } => {
                self.pods = pods.into_iter().map(|p| (p.pod_id.clone(), p)).collect();
                if !default_pod_id.is_empty() {
                    self.default_pod_id = Some(default_pod_id);
                }
                // Seed the sidebar's active pod tab on first PodList
                // (or after a reconnect that cleared it). Prefer the
                // server-declared default; fall back to whichever
                // non-archived pod sorts first by name. If the
                // currently-tabbed pod no longer exists in the list,
                // re-pick.
                let valid = self
                    .pod_tab
                    .as_ref()
                    .is_some_and(|p| self.pods.contains_key(p));
                if !valid {
                    self.pod_tab = self.pick_default_pod_tab();
                }
                // Drop any cached behavior lists for pods that are no
                // longer in the registry, then fan out `ListBehaviors`
                // for the remaining pods (deduped against any earlier
                // request that already covered them).
                let live: HashSet<String> = self.pods.keys().cloned().collect();
                self.behaviors_by_pod.retain(|p, _| live.contains(p));
                let pod_ids: Vec<String> = self.pods.keys().cloned().collect();
                for pod_id in pod_ids {
                    self.ensure_behaviors_requested(&pod_id);
                }
            }
            ServerToClient::PodSnapshot {
                snapshot,
                correlation_id,
            } => {
                // Two senders today: the pod editor (whole-snapshot
                // hydrate) and the behavior editor (pod_config slot
                // for the Thread tab's host_env / mcp_hosts override
                // pickers). Match by correlation; the pod_id check
                // guards against close-then-reopen for a different
                // pod within the round-trip window.
                if let Some(editor) = self.pod_editor.as_mut()
                    && editor.pending_get.as_ref() == correlation_id.as_ref()
                    && editor.pod_id == snapshot.pod_id
                {
                    editor.hydrate(snapshot);
                } else if let Some(editor) = self.behavior_editor.as_mut()
                    && editor.pending_pod_get.as_ref() == correlation_id.as_ref()
                    && editor.pod_id == snapshot.pod_id
                {
                    editor.pending_pod_get = None;
                    editor.pod_config = Some(snapshot.config);
                }
            }
            ServerToClient::PodConfigUpdated {
                pod_id: _,
                toml_text: _,
                parsed: _,
                correlation_id,
            } => {
                // Close the pod editor sheet when its in-flight
                // `UpdatePodConfig` lands. Match by correlation so an
                // update from another client doesn't close a sheet
                // the user is still editing in. We don't refresh any
                // local state from `parsed` here — `pods` is a list
                // of summaries (`PodSummary`, not `PodConfig`), so
                // nothing in the sidebar projection changed; if any
                // visible field did change (e.g. pod display name)
                // a follow-up `PodList` broadcast would carry it.
                if let Some(editor) = self.pod_editor.as_ref()
                    && let Some(pending) = editor.pending_save.as_ref()
                    && correlation_id.as_ref() == Some(pending)
                {
                    self.pod_editor = None;
                }
            }
            ServerToClient::PodCreated {
                pod,
                correlation_id,
            } => {
                let new_pod_id = pod.pod_id.clone();
                self.pods.insert(new_pod_id.clone(), pod);
                if self.pod_tab.is_none() {
                    self.pod_tab = self.pick_default_pod_tab();
                }
                self.ensure_behaviors_requested(&new_pod_id);

                // Close the new-pod modal if its outstanding request
                // just got echoed back. Match by correlation rather
                // than by pod_id so an unrelated `PodCreated` (from
                // another client, or a webhook) can't accidentally
                // close a modal the user is still working in.
                if let Some(modal) = self.new_pod_modal.as_ref()
                    && let Some(pending) = modal.pending_correlation.as_ref()
                    && correlation_id.as_ref() == Some(pending)
                {
                    self.new_pod_modal = None;
                    // Switch the active tab to the freshly-created
                    // pod — the user just made it, that's almost
                    // certainly where they want to land. Mirrors the
                    // egui sibling's behavior.
                    self.pod_tab = Some(new_pod_id);
                }
            }
            ServerToClient::BehaviorList {
                pod_id, behaviors, ..
            } => {
                self.behaviors_by_pod.insert(pod_id, behaviors);
            }
            ServerToClient::BehaviorCreated {
                summary,
                correlation_id,
            } => {
                // Close the new-behavior modal if its outstanding
                // request just got echoed. Match by id, not pod —
                // an unrelated `BehaviorCreated` shouldn't close a
                // modal the user is still working in.
                if let Some(modal) = self.new_behavior_modal.as_ref()
                    && let Some(pending) = modal.pending_correlation.as_ref()
                    && correlation_id.as_ref() == Some(pending)
                {
                    self.new_behavior_modal = None;
                }
                let entries = self
                    .behaviors_by_pod
                    .entry(summary.pod_id.clone())
                    .or_default();
                // Replace any existing entry with the same id (a
                // create after a delete-then-create round trip), else
                // append. Keep the server's lexicographic-by-id order
                // by re-sorting after the insert.
                if let Some(existing) = entries
                    .iter_mut()
                    .find(|e| e.behavior_id == summary.behavior_id)
                {
                    *existing = summary;
                } else {
                    entries.push(summary);
                }
                entries.sort_by(|a, b| a.behavior_id.cmp(&b.behavior_id));
            }
            ServerToClient::BehaviorSnapshot {
                snapshot,
                correlation_id,
            } => {
                // The editor sheet is the only sender of `GetBehavior`
                // today. Match by the editor's `pending_get` slot —
                // an unrelated snapshot (none expected, but be safe)
                // drops on the floor. The behavior_id check guards
                // against the user closing-then-reopening the editor
                // for a different behavior in the round-trip window.
                if let Some(editor) = self.behavior_editor.as_mut()
                    && editor.pending_get.as_ref() == correlation_id.as_ref()
                    && editor.pod_id == snapshot.pod_id
                    && editor.behavior_id == snapshot.behavior_id
                {
                    editor.hydrate(snapshot);
                }
            }
            ServerToClient::BehaviorUpdated {
                snapshot,
                correlation_id,
            } => {
                // Close the editor sheet if its `UpdateBehavior` round-
                // trip just landed. Match by correlation so an update
                // from another client doesn't accidentally close a
                // sheet the user is still working in. The summary
                // refresh below runs regardless of who initiated the
                // update.
                if let Some(editor) = self.behavior_editor.as_ref()
                    && let Some(pending) = editor.pending_save.as_ref()
                    && correlation_id.as_ref() == Some(pending)
                {
                    self.behavior_editor = None;
                }
                // Project the snapshot back onto the summary shape we
                // hold for the list view. Keep `last_fired_at` /
                // `run_count` / `enabled` from the snapshot's `state`
                // (the canonical source) since UpdateBehavior doesn't
                // mutate those — but `BehaviorStateChanged` events
                // run alongside, so we use whichever lands last.
                let pod_entries = self
                    .behaviors_by_pod
                    .entry(snapshot.pod_id.clone())
                    .or_default();
                let new_summary = BehaviorSummary {
                    behavior_id: snapshot.behavior_id.clone(),
                    pod_id: snapshot.pod_id.clone(),
                    name: snapshot
                        .config
                        .as_ref()
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| snapshot.behavior_id.clone()),
                    description: snapshot.config.as_ref().and_then(|c| c.description.clone()),
                    trigger_kind: snapshot
                        .config
                        .as_ref()
                        .map(|c| trigger_kind_label(&c.trigger).to_string()),
                    enabled: snapshot.state.enabled,
                    run_count: snapshot.state.run_count,
                    last_fired_at: snapshot.state.last_fired_at.clone(),
                    load_error: snapshot.load_error.clone(),
                };
                if let Some(slot) = pod_entries
                    .iter_mut()
                    .find(|e| e.behavior_id == new_summary.behavior_id)
                {
                    *slot = new_summary;
                } else {
                    pod_entries.push(new_summary);
                    pod_entries.sort_by(|a, b| a.behavior_id.cmp(&b.behavior_id));
                }
            }
            ServerToClient::BehaviorDeleted {
                pod_id,
                behavior_id,
                ..
            } => {
                if let Some(entries) = self.behaviors_by_pod.get_mut(&pod_id) {
                    entries.retain(|e| e.behavior_id != behavior_id);
                }
            }
            ServerToClient::BehaviorStateChanged {
                pod_id,
                behavior_id,
                state,
            } => {
                if let Some(entries) = self.behaviors_by_pod.get_mut(&pod_id)
                    && let Some(slot) = entries.iter_mut().find(|e| e.behavior_id == behavior_id)
                {
                    slot.enabled = state.enabled;
                    slot.run_count = state.run_count;
                    slot.last_fired_at = state.last_fired_at;
                }
            }
            ServerToClient::ThreadList { tasks, .. } => {
                // Replace wholesale: server's list is authoritative.
                // Drop per-thread view state for threads no longer in
                // the list (archived/deleted) so they don't linger.
                let live: HashSet<String> = tasks.iter().map(|t| t.thread_id.clone()).collect();
                self.threads = tasks
                    .into_iter()
                    .map(|t| (t.thread_id.clone(), t))
                    .collect();
                self.views.retain(|id, _| live.contains(id));
                if let Some(sel) = &self.selected
                    && !live.contains(sel)
                {
                    self.selected = None;
                }
            }
            ServerToClient::ThreadCreated {
                summary,
                correlation_id,
                ..
            } => {
                let new_id = summary.thread_id.clone();
                self.threads.insert(new_id.clone(), summary);
                // Was this `ThreadCreated` the echo for an outstanding
                // `ForkThread`? If so, capture the seed text into the
                // new thread's draft and auto-select it. Match by
                // correlation so an unrelated `ThreadCreated` (e.g.
                // a behavior fire) doesn't accidentally consume the
                // seed.
                let fork_match = matches!(
                    (&self.pending_fork_seed, correlation_id.as_ref()),
                    (Some((corr, _)), Some(echoed)) if corr == echoed
                );
                if fork_match && let Some((_, seed)) = self.pending_fork_seed.take() {
                    self.drafts.insert(new_id.clone(), seed);
                    self.select_thread(new_id.clone());
                }
                // If the user just submitted the new-thread form (no
                // selection currently), auto-select the freshly
                // created thread so the post-create transition lands
                // them in the live conversation. Skipped when another
                // thread is already selected so a background create
                // (e.g. dispatch_thread) doesn't yank focus away.
                if !fork_match && self.selected.is_none() {
                    self.select_thread(new_id);
                    // Pickers and compose buffer were consumed by the
                    // CreateThread send; reset them so the next
                    // "back to no selection" lands on a fresh form
                    // rather than the stale picker state.
                    self.picker_backend = None;
                    self.picker_model = None;
                    self.picker_pod = None;
                    self.compose_input.clear();
                }
            }
            ServerToClient::BackendsList { backends, .. } => {
                self.backends = backends;
            }
            ServerToClient::SharedMcpHostsList { hosts, .. } => {
                // Server is authoritative — replace wholesale on
                // every snapshot. Per-host add/remove broadcasts
                // (`SharedMcpHostAdded` / `Removed`) aren't surfaced
                // here yet; the next reconnect or full-list re-fetch
                // catches up.
                self.shared_mcp_hosts = hosts;
            }
            ServerToClient::BucketsList { buckets, .. } => {
                self.buckets = buckets;
            }
            ServerToClient::BucketCreated { summary, .. }
                // Append iff not already present — server re-broadcasts on
                // every connected client, and a `ListBuckets` echo may
                // race the per-row event during reconnect.
                if !self
                    .buckets
                    .iter()
                    .any(|b| b.id == summary.id && b.pod_id == summary.pod_id) =>
            {
                self.buckets.push(summary);
            }
            ServerToClient::BucketCreated { .. } => {}
            ServerToClient::BucketDeleted { id, pod_id, .. } => {
                self.buckets.retain(|b| !(b.id == id && b.pod_id == pod_id));
                let key = (pod_id, id);
                if let Some(modal) = self.buckets_modal.as_mut() {
                    modal.build_progress.remove(&key);
                    modal.build_errors.remove(&key);
                    if modal.delete_armed.as_deref() == Some(&key.1) {
                        modal.delete_armed = None;
                    }
                }
            }
            ServerToClient::BucketBuildStarted {
                bucket_id,
                pod_id,
                started_at,
                ..
            } => {
                let key = (pod_id.clone(), bucket_id.clone());
                if let Some(modal) = self.buckets_modal.as_mut() {
                    modal.build_progress.insert(
                        key.clone(),
                        BuildProgressView {
                            phase: BucketBuildPhase::Planning,
                            source_records: 0,
                            chunks: 0,
                            started_at,
                            dense_inserted: None,
                            dense_total: None,
                        },
                    );
                    modal.build_errors.remove(&key);
                }
            }
            ServerToClient::BucketBuildProgress {
                bucket_id,
                pod_id,
                phase,
                source_records,
                chunks,
                started_at,
                dense_inserted,
                dense_total,
                ..
            } => {
                let key = (pod_id, bucket_id);
                if let Some(modal) = self.buckets_modal.as_mut() {
                    modal.build_progress.insert(
                        key,
                        BuildProgressView {
                            phase,
                            source_records,
                            chunks,
                            started_at,
                            dense_inserted,
                            dense_total,
                        },
                    );
                }
            }
            ServerToClient::BucketBuildEnded {
                bucket_id,
                pod_id,
                outcome,
                summary,
                ..
            } => {
                let key = (pod_id.clone(), bucket_id.clone());
                if let Some(modal) = self.buckets_modal.as_mut() {
                    modal.build_progress.remove(&key);
                    match outcome {
                        BucketBuildOutcome::Error { message } => {
                            modal.build_errors.insert(key.clone(), message);
                        }
                        BucketBuildOutcome::Success | BucketBuildOutcome::Cancelled => {
                            modal.build_errors.remove(&key);
                        }
                    }
                }
                // Adopt the post-build summary into the catalog so the
                // row's `active_slot` reflects the freshly-promoted slot
                // without a follow-up `ListBuckets`.
                if let Some(summary) = summary
                    && let Some(existing) = self
                        .buckets
                        .iter_mut()
                        .find(|b| b.id == bucket_id && b.pod_id == pod_id)
                {
                    *existing = summary;
                }
            }
            ServerToClient::FeedPollAccepted { .. } => {
                // No-op v1 — the egui sibling also doesn't surface a
                // banner for the ack. The user clicks Poll now, the
                // request fires, and the next `ListBuckets` /
                // `BucketBuildEnded` reflects whatever the poll did.
            }
            ServerToClient::ModelsList {
                backend, models, ..
            } => {
                self.models_by_backend.insert(backend, models);
            }
            ServerToClient::ServerConfigFetched {
                toml_text,
                correlation_id,
            } => {
                if let Some(modal) = self.settings_modal.as_mut()
                    && let Some(editor) = modal.server_config.as_mut()
                    && editor.fetch_correlation == correlation_id
                {
                    editor.fetch_correlation = None;
                    editor.original = Some(toml_text.clone());
                    editor.working = toml_text;
                    editor.error = None;
                }
            }
            ServerToClient::ServerConfigUpdateResult {
                cancelled_threads,
                restart_required_sections,
                pods_with_missing_backends,
                correlation_id,
            } => {
                if let Some(modal) = self.settings_modal.as_mut()
                    && let Some(editor) = modal.server_config.as_mut()
                    && editor.save_correlation == correlation_id
                {
                    // Adopt the working buffer as the new baseline —
                    // the server-side write atomically swapped to
                    // exactly what we sent.
                    editor.original = Some(editor.working.clone());
                    editor.save_correlation = None;
                    editor.error = None;
                    editor.save_summary = Some(ServerConfigSaveSummary {
                        cancelled_threads,
                        restart_required_sections,
                        pods_with_missing_backends,
                    });
                }
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
                    // new baseline so the dirty indicator flips back to
                    // clean. Mirrors the egui sibling.
                    if let Some(w) = modal.working.clone() {
                        modal.baseline = Some(w);
                    }
                    modal.pending_correlation = None;
                    modal.error = None;
                }
            }
            ServerToClient::ThreadArchived { thread_id } => {
                self.threads.remove(&thread_id);
                self.views.remove(&thread_id);
                if self.selected.as_deref() == Some(&thread_id) {
                    self.selected = None;
                }
            }
            ServerToClient::ThreadStateChanged { thread_id, state } => {
                if let Some(t) = self.threads.get_mut(&thread_id) {
                    t.state = state;
                }
            }
            ServerToClient::ThreadTitleUpdated { thread_id, title } => {
                let opt = if title.is_empty() { None } else { Some(title) };
                if let Some(t) = self.threads.get_mut(&thread_id) {
                    t.title = opt.clone();
                }
                if let Some(v) = self.views.get_mut(&thread_id) {
                    v.title = opt;
                }
            }
            ServerToClient::ThreadSnapshot {
                thread_id,
                snapshot,
            } => {
                let items =
                    conversation_to_display_items(&snapshot.conversation, &snapshot.turn_log);
                let next_msg_index = snapshot.conversation.messages().len();
                let host_env_labels = snapshot
                    .bindings
                    .host_env
                    .iter()
                    .map(|b| match b {
                        whisper_agent_protocol::HostEnvBinding::Named {
                            name,
                            workspace_root,
                        } => match workspace_root {
                            Some(p) => format!("{name} (cwd: {})", p.display()),
                            None => name.clone(),
                        },
                        whisper_agent_protocol::HostEnvBinding::Inline { provider, .. } => {
                            format!("(inline, provider = {provider})")
                        }
                    })
                    .collect();
                let mcp_hosts = snapshot.bindings.mcp_hosts.clone();
                let max_tokens = snapshot.config.max_tokens;
                let max_turns = snapshot.config.max_turns;
                let origin_behavior_id = snapshot.origin.as_ref().map(|o| o.behavior_id.clone());
                let view = ThreadView {
                    items,
                    title: snapshot.title,
                    failure: snapshot.failure,
                    backend: snapshot.bindings.backend,
                    model: snapshot.config.model,
                    total_usage: snapshot.total_usage,
                    next_msg_index,
                    host_env_labels,
                    mcp_hosts,
                    max_tokens,
                    max_turns,
                    created_at: snapshot.created_at,
                    origin_behavior_id,
                };
                // Hydrate the local draft from the snapshot — the
                // server's stored draft is authoritative on initial
                // subscribe. Skip empty strings to avoid littering
                // the map.
                if !snapshot.draft.is_empty() {
                    self.drafts.insert(thread_id.clone(), snapshot.draft);
                } else {
                    self.drafts.remove(&thread_id);
                }
                self.views.insert(thread_id, view);
            }
            ServerToClient::ThreadDraftUpdated { thread_id, text } => {
                // Broadcast from another client editing the same
                // thread's draft. The server fans out to every
                // subscriber except the sender, so we never receive
                // our own echo here. Replace wholesale; the egui
                // sibling does the same.
                if text.is_empty() {
                    self.drafts.remove(&thread_id);
                } else {
                    self.drafts.insert(thread_id, text);
                }
            }
            ServerToClient::ThreadPrefillProgress {
                thread_id,
                tokens_processed,
                tokens_total,
            } => {
                self.prefill
                    .insert(thread_id, (tokens_processed, tokens_total));
            }
            // Per-turn streaming append: a user message just landed on
            // the conversation. Server is the source of truth (the
            // egui sibling deliberately doesn't optimistically append
            // its own send), so we wait for this echo.
            ServerToClient::ThreadUserMessage {
                thread_id,
                text,
                attachments,
            } => {
                if let Some(view) = self.views.get_mut(&thread_id) {
                    if !text.is_empty() {
                        view.items.push(DisplayItem::User {
                            text,
                            msg_index: view.next_msg_index,
                        });
                    }
                    // Inline-render any image attachments the user
                    // sent. The snapshot path (conversation_to_display_items)
                    // already does this via `ContentBlock::Image`, but
                    // the live wire arm needs its own decode + push so
                    // a streaming user message looks identical to
                    // what a resubscribe would show. Each attachment
                    // gets its own row so the role gutter colors the
                    // strip the same way the user's text gutter does.
                    for att in attachments {
                        match att {
                            Attachment::Image { source } => {
                                view.items.push(DisplayItem::Image {
                                    is_user: true,
                                    state: decode_image_source(&source),
                                });
                            }
                        }
                    }
                    // Bump regardless of empty-text guard — the
                    // server still records an empty user message in
                    // the conversation, so the next message lands at
                    // the index past this one.
                    view.next_msg_index += 1;
                }
            }
            ServerToClient::ThreadAssistantTextDelta { thread_id, delta } => {
                // First delta clears the prefill indicator — the
                // protocol guarantees prefill events stop once the
                // assistant starts emitting output.
                self.prefill.remove(&thread_id);
                if let Some(view) = self.views.get_mut(&thread_id) {
                    // Coalesce onto the trailing AssistantText if
                    // there is one — same shape as the egui sibling
                    // so a long generation doesn't fan out into one
                    // card per delta.
                    if let Some(DisplayItem::Assistant { text }) = view.items.last_mut() {
                        text.push_str(&delta);
                    } else {
                        view.items.push(DisplayItem::Assistant { text: delta });
                    }
                }
            }
            ServerToClient::ThreadAssistantReasoningDelta { thread_id, delta } => {
                self.prefill.remove(&thread_id);
                if let Some(view) = self.views.get_mut(&thread_id) {
                    if let Some(DisplayItem::Reasoning { text }) = view.items.last_mut() {
                        text.push_str(&delta);
                    } else {
                        view.items.push(DisplayItem::Reasoning { text: delta });
                    }
                }
            }
            ServerToClient::ThreadToolCallBegin {
                thread_id,
                tool_use_id,
                name,
                args,
                ..
            } => {
                if let Some(view) = self.views.get_mut(&thread_id) {
                    let args_pretty = args
                        .as_ref()
                        .and_then(|v| serde_json::to_string_pretty(v).ok());
                    // Replace any stage-streaming placeholder for this
                    // call (`ThreadToolCallStreaming` upserts one
                    // ahead of Begin) before appending; otherwise just
                    // push the new call.
                    let existing = view.items.iter().rposition(|it| match it {
                        DisplayItem::ToolCall {
                            tool_use_id: id,
                            result: None,
                            args_pretty: None,
                            streaming_output,
                            ..
                        } => id == &tool_use_id && streaming_output.is_empty(),
                        _ => false,
                    });
                    let summary = tool_summary_from_args(&name, args.as_ref());
                    let diff = extract_diff(&name, args.as_ref());
                    let entry = DisplayItem::ToolCall {
                        tool_use_id,
                        name,
                        summary,
                        diff,
                        args_pretty,
                        streaming_output: String::new(),
                        result: None,
                    };
                    match existing {
                        Some(i) => view.items[i] = entry,
                        None => view.items.push(entry),
                    }
                }
            }
            ServerToClient::ThreadToolCallContent {
                thread_id,
                tool_use_id,
                block,
            } => {
                if let Some(view) = self.views.get_mut(&thread_id) {
                    let chunk_text = match &block {
                        ContentBlock::Text { text } => text.clone(),
                        // Non-text streaming chunks (images, etc.) get
                        // a placeholder line — the integrated result
                        // arriving via `End` is the source of truth.
                        _ => String::new(),
                    };
                    if !chunk_text.is_empty()
                        && let Some(DisplayItem::ToolCall {
                            tool_use_id: id,
                            streaming_output,
                            ..
                        }) = view.items.iter_mut().rev().find(|it| {
                            matches!(it, DisplayItem::ToolCall { tool_use_id: id, .. } if id == &tool_use_id)
                        })
                    {
                        let _ = id;
                        streaming_output.push_str(&chunk_text);
                    }
                }
            }
            ServerToClient::ThreadToolCallEnd {
                thread_id,
                tool_use_id,
                result_preview,
                is_error,
                ..
            } => {
                if let Some(view) = self.views.get_mut(&thread_id) {
                    if let Some(DisplayItem::ToolCall {
                        result,
                        streaming_output,
                        ..
                    }) = view.items.iter_mut().rev().find(|it| {
                        matches!(it, DisplayItem::ToolCall { tool_use_id: id, .. } if id == &tool_use_id)
                    }) {
                        // The integrated result is authoritative; drop
                        // the streaming buffer once End lands.
                        streaming_output.clear();
                        *result = Some(FusedToolResult {
                            text: result_preview,
                            is_error,
                        });
                    } else {
                        // Orphan result — push as standalone row.
                        view.items.push(DisplayItem::ToolResult {
                            tool_use_id,
                            text: result_preview,
                            is_error,
                        });
                    }
                }
            }
            ServerToClient::ThreadToolCallStreaming {
                thread_id,
                tool_use_id,
                name,
                ..
            } => {
                // Args-still-streaming placeholder. Upserts a ToolCall
                // with no args / result so the row exists in the log
                // before `Begin` resolves the typed args. `Begin`
                // replaces this entry (matched by empty args + empty
                // streaming_output above).
                if let Some(view) = self.views.get_mut(&thread_id) {
                    let already = view.items.iter().any(|it| {
                        matches!(it, DisplayItem::ToolCall { tool_use_id: id, .. } if id == &tool_use_id)
                    });
                    if !already {
                        view.items.push(DisplayItem::ToolCall {
                            tool_use_id,
                            name,
                            // No args yet — they're still streaming.
                            // `Begin` resolves the typed args and
                            // replaces this entry with a summary-
                            // and-diff-bearing one matched by id.
                            summary: None,
                            diff: None,
                            args_pretty: None,
                            streaming_output: String::new(),
                            result: None,
                        });
                    }
                }
            }
            ServerToClient::ThreadAssistantImage { thread_id, source } => {
                // Native image output (Gemini today, OpenAI image_generation
                // soon). Decoded once at receive time and pushed as its
                // own row — coalescing onto the trailing AssistantText
                // would mix the visual layout (markdown body + raster
                // image side by side reads worse than two adjacent
                // log rows).
                if let Some(view) = self.views.get_mut(&thread_id) {
                    view.items.push(DisplayItem::Image {
                        is_user: false,
                        state: decode_image_source(&source),
                    });
                }
            }
            // Append a `TurnStats` row when the assistant finishes
            // a turn — `ThreadAssistantEnd` carries the per-turn
            // `Usage`. Keeps the streaming chat log in sync with
            // what `conversation_to_display_items` would produce
            // from a fresh snapshot.
            ServerToClient::ThreadAssistantEnd {
                thread_id, usage, ..
            } => {
                if let Some(view) = self.views.get_mut(&thread_id) {
                    view.items.push(DisplayItem::TurnStats { usage });
                    // Roll the per-turn usage into the thread total
                    // so the header chip stays in step with the
                    // server's snapshot tally between resubscribes.
                    // Mirrors the egui sibling's
                    // `view.total_usage.add(&usage)`.
                    view.total_usage.add(&usage);
                    // Assistant turn just committed → conversation
                    // grew by one Message. Keeping `next_msg_index`
                    // honest is what makes per-row fork from a
                    // streamed-in user msg target the right index.
                    view.next_msg_index += 1;
                }
            }
            ServerToClient::ThreadToolResultMessage { thread_id, .. } => {
                // Tool-result message landed on the conversation
                // (separate from the optimistic fusion we already
                // do in the ToolResult arm). We don't surface the
                // arm's own content yet, but the count bump still
                // matters for `next_msg_index`.
                if let Some(view) = self.views.get_mut(&thread_id) {
                    view.next_msg_index += 1;
                }
            }
            // Per-turn append events not yet surfaced.
            ServerToClient::ThreadAssistantBegin { .. }
            | ServerToClient::ThreadLoopComplete { .. }
            | ServerToClient::ThreadCompacted { .. } => {}
            // Model called `sudo(...)` — surface as an approval
            // banner above the matching thread's chat log. Stay
            // mounted until either `SudoResolved` echoes or the user
            // picks a decision (the user-side path optimistically
            // drops the entry to keep the banner from flickering
            // through the wire round-trip).
            ServerToClient::SudoRequested {
                function_id,
                thread_id,
                tool_name,
                args,
                reason,
            } => {
                self.pending_sudos.insert(
                    function_id,
                    PendingSudoState {
                        thread_id,
                        tool_name,
                        args,
                        reason,
                    },
                );
            }
            // Server-side resolution (this client's, another
            // client's, or a timeout). Drop the slot so the banner
            // disappears even when we weren't the resolver. Reject
            // draft is dropped alongside so a stale text input
            // doesn't haunt a future request that happens to reuse
            // the function_id (server is monotonic, but the cleanup
            // is cheap and defensive).
            ServerToClient::SudoResolved { function_id, .. } => {
                self.pending_sudos.remove(&function_id);
                self.sudo_reject_drafts.remove(&function_id);
            }
            ServerToClient::Error {
                correlation_id,
                thread_id,
                message,
            } => {
                // Modals first: each modal owns its own
                // `pending_correlation` slot; whichever matches
                // surfaces the message and clears the pending so
                // the form re-enables for retry. `consumed`
                // gates the thread-scoped fallback so a single
                // Error can't double-fire onto both layers.
                let mut consumed = false;
                if let Some(corr) = correlation_id.as_ref() {
                    if let Some(modal) = self.new_pod_modal.as_mut()
                        && modal.pending_correlation.as_ref() == Some(corr)
                    {
                        modal.error = Some(message.clone());
                        modal.pending_correlation = None;
                        consumed = true;
                    } else if let Some(modal) = self.new_behavior_modal.as_mut()
                        && modal.pending_correlation.as_ref() == Some(corr)
                    {
                        modal.error = Some(message.clone());
                        modal.pending_correlation = None;
                        consumed = true;
                    } else if let Some(editor) = self.behavior_editor.as_mut() {
                        // Two correlation slots in flight at any given
                        // time: `pending_get` (the initial snapshot
                        // round-trip) and `pending_save` (an in-flight
                        // UpdateBehavior). Either can fail; either
                        // surfaces the message into the same `error`
                        // slot. Save re-enables on save failure; a get
                        // failure leaves the form in its loading shape
                        // with the error in place.
                        if editor.pending_save.as_ref() == Some(corr) {
                            editor.error = Some(message.clone());
                            editor.pending_save = None;
                            consumed = true;
                        } else if editor.pending_get.as_ref() == Some(corr) {
                            editor.error = Some(message.clone());
                            editor.pending_get = None;
                            consumed = true;
                        }
                    }
                    // Pod editor — same two-slot pattern as the
                    // behavior editor (pending_get / pending_save).
                    // A save failure here is the common case (e.g.
                    // the server rejected the new TOML's parse or
                    // failed validation); the inline alert tells the
                    // user what went wrong without having to switch
                    // contexts.
                    if !consumed && let Some(editor) = self.pod_editor.as_mut() {
                        if editor.pending_save.as_ref() == Some(corr) {
                            editor.error = Some(message.clone());
                            editor.pending_save = None;
                            consumed = true;
                        } else if editor.pending_get.as_ref() == Some(corr) {
                            editor.error = Some(message.clone());
                            editor.pending_get = None;
                            consumed = true;
                        }
                    }
                }
                // Thread-scoped errors land on the view's failure
                // slot so the pane's destructive banner has
                // something to show even before the next
                // snapshot. State flips to `Failed` via a
                // separate `ThreadStateChanged` echo (the
                // banner gate uses the conjunction).
                if !consumed
                    && let Some(tid) = thread_id.as_ref()
                    && let Some(view) = self.views.get_mut(tid)
                {
                    view.failure = Some(message);
                }
            }
            // Catalog tiers we don't yet surface. Quietly accepted so
            // the server's normal broadcast batch doesn't error here.
            _ => {}
        }
    }
}

impl App for ChatApp {
    fn before_build(&mut self) {
        self.drain_inbound();
        self.drain_pending_picks();
        // Expire stale compose hint. Cheap: one `Instant::now()` +
        // a comparison per frame. Idle apps never re-enter this
        // branch since the slot is `None` until staging fires.
        if let Some((_, expires)) = self.compose_hint.as_ref()
            && std::time::Instant::now() > *expires
        {
            self.compose_hint = None;
        }
    }

    fn build(&self, cx: &BuildCx) -> El {
        // Root is an overlay stack: the main row carries the chrome,
        // any open `select_menu` rides above it as a popover layer.
        // Per `widgets::select` doc, the menu must sit at the El
        // tree root so it paints over content and intercepts the
        // dismiss scrim's click — we collect them here.
        //
        // `cx` threads down through `content` → `event_log_row` so
        // the chat log can read `is_hovering_within(row_key)` to
        // surface per-row affordances (e.g. the user-row fork
        // button) without mirroring hover state through `on_event`.
        // Override the sidebar's baked-in `Fixed(SIDEBAR_WIDTH)` with
        // the user-controlled `sidebar_width` so the resize handle
        // can shrink / grow it. The handle sits between the sidebar
        // and the content pane; its drag events fold into
        // `sidebar_width` via `apply_event_fixed` in `on_event`.
        let sidebar_el = self.sidebar().width(Size::Fixed(self.sidebar_width));
        overlays(
            row([
                sidebar_el,
                resize_handle(Axis::Row).key(SIDEBAR_RESIZE_KEY),
                self.content(cx),
            ])
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
            .gap(0.0),
            self.popover_layers(),
        )
    }

    fn on_event(&mut self, event: UiEvent) {
        // Sidebar resize handle — pointer drag / arrow keys / Home /
        // End on `sidebar:resize` fold into `sidebar_width`. Helper
        // returns `true` if the value changed (no-op for events on
        // other keys), so we can short-circuit the rest of the
        // dispatch when this consumed the event.
        if resize_handle::apply_event_fixed(
            &mut self.sidebar_width,
            &mut self.sidebar_drag,
            &event,
            SIDEBAR_RESIZE_KEY,
            Axis::Row,
            Side::Start,
            tokens::SIDEBAR_WIDTH_MIN,
            tokens::SIDEBAR_WIDTH_MAX,
        ) {
            return;
        }

        // Window-level file drops route here regardless of which
        // keyed leaf is under the pointer — the compose attachment
        // strip is the only sink today, so we treat unrouted drops
        // as compose-bar drops. Multi-file drags fire one
        // `FileDropped` per file, so each iteration of `on_event`
        // stages exactly one. Read the file synchronously: drops
        // are typically modest (screenshots, paste-buffer images)
        // and the sync read avoids the picker thread / queue
        // round-trip a streaming staging path would need.
        if matches!(event.kind, UiEventKind::FileDropped)
            && let Some(path) = event.path.as_ref()
        {
            let source_desc = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string_lossy().into_owned());
            match std::fs::read(path) {
                Ok(bytes) => {
                    self.stage_raw_pick(RawPick { bytes, source_desc });
                }
                Err(e) => {
                    self.set_compose_hint(format!("couldn't read {source_desc}: {e}"));
                }
            }
            return;
        }

        // Compose attach button — opens the OS file picker. Native
        // runs rfd off-thread; wasm sets a "not yet" hint until
        // Stage 11 wires the browser path.
        if event.is_click_or_activate(COMPOSE_ATTACH_KEY) {
            self.spawn_file_picker();
            return;
        }

        // Compose thumbnail remove button. Routes are
        // `compose:attach:remove:{id}`; parse the suffix and drop
        // the matching staged attachment.
        if let Some(key) = event.route()
            && let Some(id_str) = key.strip_prefix(COMPOSE_ATTACH_REMOVE_PREFIX)
            && let Ok(id) = id_str.parse::<u64>()
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
        {
            self.compose_attachments.retain(|s| s.id != id);
            return;
        }

        // Auto-disarm a pending behavior delete unless this click is
        // its matching second arm. The sidebar rebuilds every event,
        // so any other click — selecting a thread, opening a modal,
        // toggling a pod tab — is the user's "I changed my mind"
        // gesture. Runs before any handler that would `return`, so
        // every routed click flows through this gate. Bare-text
        // events (focus, hover, key presses without a route) leave
        // the arm intact.
        if matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && self.delete_armed_behavior.is_some()
        {
            let armed = self.delete_armed_behavior.as_ref().unwrap();
            let armed_key = behavior_delete_key(&armed.0, &armed.1);
            if event.route() != Some(armed_key.as_str()) {
                self.delete_armed_behavior = None;
            }
        }

        // Sidebar thread selection: any click on a `thread:{id}` key
        // activates that thread.
        if let Some(key) = event.route()
            && let Some(thread_id) = key.strip_prefix("thread:")
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
        {
            let thread_id = thread_id.to_string();
            self.select_thread(thread_id);
            return;
        }

        // Inline-image click → open lightbox. Suffix is the
        // display-item index in the active thread's view; we look the
        // image back up rather than packing it into the route. Any
        // non-Decoded state (URL placeholder, decode failure) is
        // skipped — those rows aren't given a click key in the first
        // place, but the guard here defends against a stale state
        // change between build and click.
        if let Some(key) = event.route()
            && let Some(idx_str) = key.strip_prefix(CHAT_IMAGE_LIGHTBOX_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            self.open_lightbox(idx);
            return;
        }
        // Lightbox dismiss / close — both keys collapse to clearing
        // the slot. The dialog scrim auto-emits the dismiss key on
        // outside-click; the close button uses the explicit one.
        if event.is_click_or_activate(LIGHTBOX_MODAL_DISMISS_KEY)
            || event.is_click_or_activate(LIGHTBOX_MODAL_CLOSE_KEY)
        {
            self.lightbox = None;
            return;
        }
        // JSON viewer dismiss / close — same shape as the lightbox.
        // Clearing the slot is the close; the per-node accordion
        // open set is dropped alongside so a re-open lands with the
        // egui sibling's default-collapsed shape rather than the
        // user's previous expansion.
        if event.is_click_or_activate(JSON_VIEWER_DISMISS_KEY)
            || event.is_click_or_activate(JSON_VIEWER_CLOSE_KEY)
        {
            self.json_viewer_modal = None;
            self.json_tree_open.clear();
            return;
        }
        // File viewer routing — body text_area first (target_key is
        // the body, not a click route), then save / revert / close /
        // dismiss. Edits clear the inline error so a stale "couldn't
        // save" message doesn't linger after the user starts fixing
        // it.
        if event.target_key() == Some(FILE_VIEWER_BODY_KEY) {
            if let Some(modal) = self.file_viewer_modal.as_mut()
                && let Some(buf) = modal.working.as_mut()
                && !modal.readonly
            {
                text_area::apply_event(buf, &mut self.selection, FILE_VIEWER_BODY_KEY, &event);
                modal.error = None;
            }
            return;
        }
        if event.is_click_or_activate(FILE_VIEWER_SAVE_KEY) {
            self.submit_file_viewer();
            return;
        }
        if event.is_click_or_activate(FILE_VIEWER_REVERT_KEY) {
            if let Some(modal) = self.file_viewer_modal.as_mut()
                && let Some(baseline) = modal.baseline.clone()
            {
                modal.working = Some(baseline);
                modal.error = None;
            }
            return;
        }
        if event.is_click_or_activate(FILE_VIEWER_DISMISS_KEY)
            || event.is_click_or_activate(FILE_VIEWER_CLOSE_KEY)
        {
            self.file_viewer_modal = None;
            return;
        }
        // Server-settings modal routing — tab strip, body text_area,
        // save / revert / close / dismiss. Body edits clear the save
        // summary banner so a fresh edit reads as "in progress" not
        // "just saved."
        if self.settings_modal.is_some() {
            // Tab strip. Read the active tab into a local, apply, then
            // write back. Dropping the modal borrow before calling
            // `ensure_server_config_fetched` (which re-borrows
            // `settings_modal` to populate the editor slot) keeps
            // borrow-check happy.
            let mut tab = self
                .settings_modal
                .as_ref()
                .map(|m| m.active_tab)
                .unwrap_or_default();
            if aetna_core::widgets::tabs::apply_event(
                &mut tab,
                &event,
                SETTINGS_TABS_KEY,
                SettingsTab::from_wire,
            ) {
                if let Some(modal) = self.settings_modal.as_mut() {
                    modal.active_tab = tab;
                }
                if matches!(tab, SettingsTab::ServerConfig) {
                    self.ensure_server_config_fetched();
                }
                return;
            }
            if event.target_key() == Some(SETTINGS_SERVER_CONFIG_BODY_KEY) {
                if let Some(modal) = self.settings_modal.as_mut()
                    && let Some(editor) = modal.server_config.as_mut()
                    && editor.save_correlation.is_none()
                {
                    text_area::apply_event(
                        &mut editor.working,
                        &mut self.selection,
                        SETTINGS_SERVER_CONFIG_BODY_KEY,
                        &event,
                    );
                    editor.save_summary = None;
                    editor.error = None;
                }
                return;
            }
            if event.is_click_or_activate(SETTINGS_SERVER_CONFIG_SAVE_KEY) {
                self.submit_server_config();
                return;
            }
            if event.is_click_or_activate(SETTINGS_SERVER_CONFIG_REVERT_KEY) {
                if let Some(modal) = self.settings_modal.as_mut()
                    && let Some(editor) = modal.server_config.as_mut()
                    && let Some(original) = editor.original.clone()
                {
                    editor.working = original;
                    editor.save_summary = None;
                    editor.error = None;
                }
                return;
            }
            if event.is_click_or_activate(SETTINGS_DISMISS_KEY)
                || event.is_click_or_activate(SETTINGS_CLOSE_KEY)
            {
                self.settings_modal = None;
                return;
            }
        }
        // File tree modal routing — close / dismiss, plus dir toggle
        // and file pick. Both row prefixes carry `{pod}:{path}` so
        // multi-pod state can't alias across opens.
        if event.is_click_or_activate(FILE_TREE_DISMISS_KEY)
            || event.is_click_or_activate(FILE_TREE_CLOSE_KEY)
        {
            self.file_tree_modal_pod = None;
            return;
        }
        if let Some(key) = event.route()
            && let Some(rest) = key.strip_prefix(FILE_TREE_DIR_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Some((pod, path)) = rest.split_once(':')
        {
            let composite = (pod.to_string(), path.to_string());
            if !self.pod_dirs_open.remove(&composite) {
                self.pod_dirs_open.insert(composite);
                // Lazy fetch of the just-expanded dir. Idempotent
                // when the cache or in-flight request already
                // covers it.
                self.ensure_pod_dir_fetched(pod, path);
            }
            return;
        }
        if let Some(key) = event.route()
            && let Some(rest) = key.strip_prefix(FILE_TREE_FILE_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Some((pod, path)) = rest.split_once(':')
        {
            let pod = pod.to_string();
            let path = path.to_string();
            self.handle_file_tree_pick(&pod, &path);
            return;
        }

        // Sudo banner: approve / remember / reject buttons + reject-
        // reason text input. All four routes carry the function_id as
        // suffix; we match on the prefix and parse the tail back.
        // Optimistic resolution — drop the local slot before the
        // server's `SudoResolved` echo lands so the banner doesn't
        // flicker.
        if let Some(key) = event.route()
            && let Some(id_str) = key.strip_prefix(SUDO_APPROVE_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Ok(fn_id) = id_str.parse::<u64>()
        {
            self.resolve_sudo(fn_id, SudoDecision::ApproveOnce, None);
            return;
        }
        if let Some(key) = event.route()
            && let Some(id_str) = key.strip_prefix(SUDO_REMEMBER_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Ok(fn_id) = id_str.parse::<u64>()
        {
            self.resolve_sudo(fn_id, SudoDecision::ApproveRemember, None);
            return;
        }
        if let Some(key) = event.route()
            && let Some(id_str) = key.strip_prefix(SUDO_REJECT_PREFIX)
            && !key.starts_with(SUDO_REJECT_REASON_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Ok(fn_id) = id_str.parse::<u64>()
        {
            let reason = self
                .sudo_reject_drafts
                .get(&fn_id)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            self.resolve_sudo(fn_id, SudoDecision::Reject, reason);
            return;
        }
        // Reject-reason text input. Routed key carries function_id;
        // text_input::apply_event mutates the per-banner draft in
        // place. We have to look the key up indirectly because
        // `apply_event` takes the key by `&str`.
        if let Some(key) = event.target_key()
            && let Some(id_str) = key.strip_prefix(SUDO_REJECT_REASON_PREFIX)
            && let Ok(fn_id) = id_str.parse::<u64>()
        {
            let key_owned = sudo_reject_reason_key(fn_id);
            let draft = self.sudo_reject_drafts.entry(fn_id).or_default();
            text_input::apply_event(draft, &mut self.selection, &key_owned, &event);
            return;
        }

        // Sidebar pod-tab strip. `tabs::apply_event` parses the
        // `pod-tabs:tab:{id}` routed key back to an id; the parse
        // closure rejects ids we no longer know about (defensive
        // against a stale event arriving after a `PodList` shrunk
        // the set).
        let pod_tab_changed = aetna_core::widgets::tabs::apply_event(
            &mut self.pod_tab,
            &event,
            POD_TABS_KEY,
            |raw| {
                if self.pods.contains_key(raw) {
                    Some(Some(raw.to_string()))
                } else {
                    None
                }
            },
        );
        if pod_tab_changed {
            return;
        }

        // "+ New thread" entry point: clear the selection and
        // pre-bind the active pod in the new-thread compose form's
        // pod picker so the compose pane opens scoped to where the
        // user clicked. No wire round-trip — the actual `CreateThread`
        // fires when the user hits Start in the compose card.
        if event.is_click_or_activate(SIDEBAR_NEW_THREAD_KEY) {
            self.selected = None;
            self.picker_pod = self.pod_tab.clone();
            return;
        }

        // "+ New pod" entry point: open the modal with empty fields.
        // Idempotent — clicking twice while open just resets the
        // form, which matches what the user means by "I want to
        // start over".
        if event.is_click_or_activate(SIDEBAR_POD_SETTINGS_KEY) {
            if let Some(pod) = self.pod_tab.clone() {
                self.open_pod_editor(pod);
            }
            return;
        }
        if event.is_click_or_activate(SIDEBAR_POD_FILES_KEY) {
            if let Some(pod) = self.pod_tab.clone() {
                self.open_file_tree_modal(pod);
            }
            return;
        }
        if event.is_click_or_activate(SIDEBAR_SERVER_SETTINGS_KEY) {
            self.open_settings_modal();
            return;
        }
        if event.is_click_or_activate(SIDEBAR_BUCKETS_KEY) {
            self.open_buckets_modal();
            return;
        }
        // Buckets modal routing — close / dismiss + per-row action
        // prefixes. Each row prefix carries `{pod}:{id}` (pod =
        // sentinel for server-scope).
        if event.is_click_or_activate(BUCKETS_MODAL_DISMISS_KEY)
            || event.is_click_or_activate(BUCKETS_MODAL_CLOSE_KEY)
        {
            self.buckets_modal = None;
            return;
        }
        if let Some(key) = event.route()
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
        {
            let parse = |prefix: &str| -> Option<(Option<String>, String)> {
                let rest = key.strip_prefix(prefix)?;
                let (pod, id) = rest.split_once(':')?;
                Some((bucket_scope_from_token(pod), id.to_string()))
            };
            if let Some((pod, id)) = parse(BUCKETS_BUILD_PREFIX) {
                self.start_bucket_build(id, pod);
                return;
            }
            if let Some((pod, id)) = parse(BUCKETS_PAUSE_PREFIX) {
                self.cancel_bucket_build(id, pod);
                return;
            }
            if let Some((pod, id)) = parse(BUCKETS_POLL_PREFIX) {
                self.poll_bucket_feed(id, pod);
                return;
            }
            if let Some((pod, id)) = parse(BUCKETS_RESYNC_PREFIX) {
                self.resync_bucket(id, pod);
                return;
            }
            if let Some((_pod, id)) = parse(BUCKETS_DELETE_PREFIX) {
                if let Some(modal) = self.buckets_modal.as_mut() {
                    modal.delete_armed = Some(id);
                }
                return;
            }
            if let Some((pod, id)) = parse(BUCKETS_DELETE_CONFIRM_PREFIX) {
                if let Some(modal) = self.buckets_modal.as_mut() {
                    modal.delete_armed = None;
                }
                self.delete_bucket(id, pod);
                return;
            }
            if let Some((_pod, _id)) = parse(BUCKETS_DELETE_CANCEL_PREFIX) {
                if let Some(modal) = self.buckets_modal.as_mut() {
                    modal.delete_armed = None;
                }
                return;
            }
        }
        if event.is_click_or_activate(CHAT_INSPECTOR_TOGGLE_KEY) {
            // Toggle the inspector for the active thread. Clicking
            // while open on the same thread collapses; clicking
            // open on a different thread (only possible if the
            // active selection changed between build and click —
            // rare but defended) re-targets.
            let target = self.selected.clone();
            self.inspector_open = match (&self.inspector_open, &target) {
                (Some(open), Some(sel)) if open == sel => None,
                _ => target,
            };
            return;
        }
        if event.is_click_or_activate(SIDEBAR_NEW_POD_KEY) {
            self.new_pod_modal = Some(NewPodModalState::default());
            return;
        }

        // "+ New behavior" entry point: routed key carries the pod
        // id (`sidebar:new-behavior:{pod}`) so the modal scopes to
        // where the user clicked, not just whatever's in the active
        // tab. Defensive against a stale event arriving for a pod
        // that was deleted in flight — drop the event then.
        if let Some(key) = event.route()
            && let Some(pod_id) = key.strip_prefix(SIDEBAR_NEW_BEHAVIOR_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
        {
            if self.pods.contains_key(pod_id) {
                self.new_behavior_modal = Some(NewBehaviorModalState::new(pod_id.to_string()));
            }
            return;
        }

        // "+ New pod" modal routing — text inputs, primary / cancel
        // buttons, and the scrim's auto-emitted dismiss key. Walking
        // the routes top-down (cheapest checks first) before falling
        // through to the rest of the on_event body.
        if self.handle_new_pod_modal_event(&event) {
            return;
        }
        // "+ New behavior" modal routing — same shape as new-pod
        // (text inputs + primary / cancel + scrim dismiss).
        if self.handle_new_behavior_modal_event(&event) {
            return;
        }
        // Behavior editor sheet routing — text inputs (name /
        // description / prompt / cron schedule), the trigger-kind
        // picker, primary Save, secondary Cancel, scrim Dismiss.
        if self.handle_behavior_editor_event(&event) {
            return;
        }
        // Pod editor sheet routing — single text_area + Save /
        // Cancel / Dismiss.
        if self.handle_pod_editor_event(&event) {
            return;
        }
        // Fork dialog routing — switch toggles, primary / cancel /
        // dismiss.
        if self.handle_fork_modal_event(&event) {
            return;
        }
        // Per-User-row fork affordance — opens the fork dialog
        // pre-populated with the clicked message's index + text.
        if let Some(key) = event.route()
            && let Some(suffix) = key.strip_prefix(CHAT_USER_FORK_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Ok(msg_index) = suffix.parse::<usize>()
        {
            self.open_fork_modal(msg_index);
            return;
        }

        // "Show N more" / "Show less" toggle on the active pod's
        // thread list. One toggle per pod (the active one), so we
        // key membership in `expanded_pod_threads` by the active
        // pod_id.
        if event.is_click_or_activate(SIDEBAR_SHOWMORE_KEY) {
            if let Some(pod) = self.pod_tab.clone()
                && !self.expanded_pod_threads.remove(&pod)
            {
                self.expanded_pod_threads.insert(pod);
            }
            return;
        }

        // Behavior row toggle: routed key is `behavior-row:{pod}:{id}`.
        // Toggle membership of the same `(pod, behavior)` pair in
        // `expanded_behaviors`. Click alone activates — drag /
        // hover / pointer-down don't.
        if let Some(key) = event.route()
            && let Some(rest) = key.strip_prefix(BEHAVIOR_ROW_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
        {
            // `{pod_id}:{behavior_id}` — split on the first `:` so a
            // behavior_id containing a colon doesn't break the parse.
            if let Some((pod, beh)) = rest.split_once(':') {
                let composite = behavior_expand_key(pod, beh);
                if !self.expanded_behaviors.remove(&composite) {
                    self.expanded_behaviors.insert(composite);
                }
            }
            return;
        }

        // Behavior Run-now button: fire `RunBehavior` for the
        // `{pod_id}:{behavior_id}` pair. Server replies with
        // `ThreadCreated` carrying the spawned thread; the
        // sidebar's behavior-section then nests it under the row
        // automatically (no special routing needed here).
        if let Some(key) = event.route()
            && let Some(rest) = key.strip_prefix(BEHAVIOR_RUN_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Some((pod, beh)) = rest.split_once(':')
        {
            self.send(ClientToServer::RunBehavior {
                correlation_id: None,
                pod_id: pod.to_string(),
                behavior_id: beh.to_string(),
                payload: None,
            });
            return;
        }

        // Behavior Pause/Resume toggle: send `SetBehaviorEnabled`
        // with the inverse of whatever we currently believe. Don't
        // optimistically mutate local state — the server's
        // `BehaviorStateChanged` echo is sub-millisecond on
        // loopback and the canonical source.
        if let Some(key) = event.route()
            && let Some(rest) = key.strip_prefix(BEHAVIOR_TOGGLE_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Some((pod, beh)) = rest.split_once(':')
        {
            let next_enabled = self
                .behaviors_by_pod
                .get(pod)
                .and_then(|list| list.iter().find(|b| b.behavior_id == beh))
                .map(|b| !b.enabled)
                .unwrap_or(true);
            self.send(ClientToServer::SetBehaviorEnabled {
                correlation_id: None,
                pod_id: pod.to_string(),
                behavior_id: beh.to_string(),
                enabled: next_enabled,
            });
            return;
        }

        // Behavior Delete: two-click arm-confirm. The pre-handler at
        // the top of `on_event` already cleared `delete_armed_behavior`
        // if this click isn't for the armed pair, so by this point
        // either this is the armed key (fire) or no arm is held
        // (set the arm). One key, two outcomes.
        if let Some(key) = event.route()
            && let Some(rest) = key.strip_prefix(BEHAVIOR_DELETE_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Some((pod, beh)) = rest.split_once(':')
        {
            let pair = (pod.to_string(), beh.to_string());
            if self.delete_armed_behavior.as_ref() == Some(&pair) {
                self.delete_armed_behavior = None;
                self.send(ClientToServer::DeleteBehavior {
                    correlation_id: None,
                    pod_id: pod.to_string(),
                    behavior_id: beh.to_string(),
                });
            } else {
                self.delete_armed_behavior = Some(pair);
            }
            return;
        }

        // Behavior Edit: open the editor sheet for the matching
        // `{pod}:{beh}` pair and fire `GetBehavior` to hydrate the
        // form. If the editor is already open for the same pair, this
        // click is a no-op (avoids re-firing the round-trip and
        // discarding partial edits). If it's open for a different
        // behavior, the new pair takes over — the prior pending
        // snapshot will arrive with a stale correlation and be
        // dropped on the floor.
        if let Some(key) = event.route()
            && let Some(rest) = key.strip_prefix(BEHAVIOR_EDIT_PREFIX)
            && matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Some((pod, beh)) = rest.split_once(':')
        {
            self.open_behavior_editor(pod.to_string(), beh.to_string());
            return;
        }

        // Compose `text_area` edits: when the routed target is the
        // compose box, fold the event through aetna's controlled-
        // widget helper. The buffer is per-thread when a thread is
        // selected (so drafts persist across switches and survive
        // reload via `SetThreadDraft`), or `compose_input` for the
        // new-thread form.
        if event.target_key() == Some(COMPOSE_KEY) {
            // Plain Enter (no modifiers) submits — same UX as the
            // egui sibling. Shift+Enter / Ctrl+Enter / etc. fall
            // through to the text_area which inserts a newline.
            // We branch *before* `text_area::apply_event` so the
            // submit Enter never reaches the buffer.
            if event.kind == UiEventKind::KeyDown
                && let Some(kp) = event.key_press.as_ref()
                && matches!(kp.key, UiKey::Enter)
                && !kp.modifiers.shift
                && !kp.modifiers.ctrl
                && !kp.modifiers.alt
                && !kp.modifiers.logo
            {
                self.send_compose();
                return;
            }
            let selected = self.selected.clone();
            // Field-level borrow split: `self.drafts` /
            // `self.compose_input` and `self.selection` are
            // independent fields, so the borrow checker accepts
            // simultaneous `&mut` access through direct field
            // expressions. Capture `before`/`after` to detect a
            // text-changing edit and skip the SetThreadDraft fanout
            // on cursor-only events.
            let (changed, new_text) = if let Some(tid) = selected.as_ref() {
                let buf = self.drafts.entry(tid.clone()).or_default();
                let before = buf.clone();
                text_area::apply_event(buf, &mut self.selection, COMPOSE_KEY, &event);
                (*buf != before, buf.clone())
            } else {
                let before = self.compose_input.clone();
                text_area::apply_event(
                    &mut self.compose_input,
                    &mut self.selection,
                    COMPOSE_KEY,
                    &event,
                );
                (self.compose_input != before, self.compose_input.clone())
            };
            // Persist server-side. Fire on every text-changing edit
            // — the egui sibling debounces, but bandwidth here is
            // tiny and per-keystroke avoids needing a wall-clock
            // timer, which doesn't ride the build/on_event split
            // cleanly.
            if changed && let Some(tid) = selected {
                self.send(ClientToServer::SetThreadDraft {
                    thread_id: tid,
                    text: new_text,
                });
            }
            return;
        }

        // Send button. While the selected thread is Working the
        // primary button is rendered as "Stop" — same key, but the
        // click cancels the thread instead of sending the (empty)
        // compose buffer.
        if event.is_click_or_activate(SEND_KEY) {
            if let Some(thread_id) = self.selected.clone()
                && self.is_selected_working()
            {
                self.send(ClientToServer::CancelThread { thread_id });
                return;
            }
            self.send_compose();
            return;
        }

        // Thread-level action buttons. Each fires its matching wire
        // command for the currently-selected thread; the server's
        // broadcast updates `threads` and the next build reflects
        // the new state. No optimistic mutation — the UI never
        // diverges from server-confirmed state.
        if event.is_click_or_activate(THREAD_CANCEL_KEY)
            && let Some(thread_id) = self.selected.clone()
        {
            self.send(ClientToServer::CancelThread { thread_id });
            return;
        }
        if event.is_click_or_activate(THREAD_ARCHIVE_KEY)
            && let Some(thread_id) = self.selected.clone()
        {
            self.send(ClientToServer::ArchiveThread { thread_id });
            return;
        }
        if event.is_click_or_activate(THREAD_COMPACT_KEY)
            && let Some(thread_id) = self.selected.clone()
        {
            self.send(ClientToServer::CompactThread {
                thread_id,
                correlation_id: None,
            });
            return;
        }

        // New-thread form pickers (backend / model / pod). Each one is
        // a `select_trigger` + conditionally-rendered `select_menu`
        // pair routed at `picker:{kind}`. We use `classify_event`
        // explicitly (rather than the bundled `apply_event`) because
        // picking a backend has a side-effect — clearing the model
        // selection and firing a `ListModels` for the new backend.
        if let Some(action) = classify_select_event(&event, PICKER_BACKEND) {
            self.handle_backend_pick(action);
            return;
        }
        if let Some(action) = classify_select_event(&event, PICKER_MODEL) {
            self.handle_model_pick(action);
            return;
        }
        if let Some(action) = classify_select_event(&event, PICKER_POD) {
            self.handle_pod_pick(action);
            return;
        }

        // Accordion toggles (reasoning + tool rows + JSON tree
        // nodes). The accordion runtime emits routed keys shaped
        // `{group}:accordion:{value}`; we just toggle membership of
        // the routed key in our open set — independent toggles per
        // row, no single-active enforcement. The JSON viewer's tree
        // routes into its own set so closing the modal can drop
        // the per-node state without disturbing chat-log accordions.
        if matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Some(key) = event.route()
            && key.contains(":accordion:")
        {
            let set = if key.starts_with(JSON_TREE_ACCORDION_GROUP) {
                &mut self.json_tree_open
            } else {
                &mut self.open_accordions
            };
            if !set.remove(key) {
                set.insert(key.to_string());
            }
            return;
        }

        // Global selection plumbing — when the runtime synthesizes a
        // SelectionChanged event from pointer drags, mirror it onto
        // our owned `Selection`. Required for the in-text caret
        // movement in the compose box to feel right.
        if let Some(sel) = event.selection.clone() {
            self.selection = sel;
        }
    }

    fn selection(&self) -> Selection {
        self.selection.clone()
    }

    fn theme(&self) -> Theme {
        // Aetna's slate + blue dark palette (Radix Colors). Card and
        // popover are lifted above background out of the box, so the
        // sidebar / new-thread card / behavior editor sheet read as
        // distinct surfaces without any palette overrides on our side.
        Theme::radix_slate_blue_dark()
    }
}

const COMPOSE_KEY: &str = "compose";
/// Thread-level action affordances rendered in a row above the
/// compose text_area when a thread is selected. Each fires the
/// matching `ClientToServer` variant for the active thread; their
/// visibility / enabled state depends on the thread's state
/// (Cancel hides while Working — the Send button toggles to Stop
/// and covers cancellation; Compact disables while Working since
/// the server rejects mid-turn).
const THREAD_CANCEL_KEY: &str = "thread-action:cancel";
const THREAD_ARCHIVE_KEY: &str = "thread-action:archive";
const THREAD_COMPACT_KEY: &str = "thread-action:compact";
/// Compose attach affordance — paperclip icon-button next to the
/// send button. Click spawns the OS file picker; drag-drop on the
/// window stages files independently of this key.
const COMPOSE_ATTACH_KEY: &str = "compose:attach";
/// Per-thumbnail remove route prefix; suffix is the staged
/// attachment's monotonic id. Lets duplicates be addressed
/// individually.
const COMPOSE_ATTACH_REMOVE_PREFIX: &str = "compose:attach:remove:";
const SEND_KEY: &str = "send";

/// Routed key for the sidebar's pod-tab strip. Per-tab option keys
/// are auto-derived as `pod-tabs:tab:{pod_id}` by `tabs_list`.
const POD_TABS_KEY: &str = "pod-tabs";

/// Routed key for the "Show N more" / "Show less" toggle that
/// expands the active pod's thread list past
/// [`SIDEBAR_THREAD_PREVIEW`]. One toggle per pod (the active one)
/// so a single shared key is fine.
const SIDEBAR_SHOWMORE_KEY: &str = "sidebar:showmore";

/// Default-collapsed cap on per-pod thread rows in the sidebar.
/// Mirrors `whisper-agent-webui::THREAD_ROW_PREVIEW_COUNT`. The
/// "Show N more" toggle reveals the full list per pod.
const SIDEBAR_THREAD_PREVIEW: usize = 10;

/// Routed key for the "+" icon-button next to the Threads section
/// header — the primary entry point for starting a new thread in the
/// active pod. Click clears the selection and pre-selects the active
/// pod in the new-thread compose form's pod picker so the user lands
/// on a compose pane already scoped to where they're working.
const SIDEBAR_NEW_THREAD_KEY: &str = "sidebar:new-thread";

/// Routed key for the sidebar's "+ New pod" entry point, sitting in
/// the sidebar header next to the connection badge. Click opens the
/// `NEW_POD_MODAL_KEY`-keyed dialog.
const SIDEBAR_NEW_POD_KEY: &str = "sidebar:new-pod";

/// Routed-key prefix the "+ New pod" dialog uses. Extended forms:
/// `{prefix}:dismiss` (scrim, emitted by `dialog`),
/// `{prefix}:pod-id` (text_input), `{prefix}:name` (text_input),
/// `{prefix}:create` (primary submit), `{prefix}:cancel` (secondary).
const NEW_POD_MODAL_KEY: &str = "new-pod";

/// Routed-key prefix for the per-pod "+" icon-button in the
/// behaviors-section header. Suffix is the pod_id so the dialog
/// can scope to where the user clicked. Mirrors the
/// `behavior-row:` family in pod-keying style.
const SIDEBAR_NEW_BEHAVIOR_PREFIX: &str = "sidebar:new-behavior:";

/// Routed-key prefix the "+ New behavior" dialog uses. Same
/// extended-key family as [`NEW_POD_MODAL_KEY`]:
/// `{prefix}:dismiss`, `{prefix}:behavior-id`, `{prefix}:name`,
/// `{prefix}:create`, `{prefix}:cancel`.
const NEW_BEHAVIOR_MODAL_KEY: &str = "new-behavior";

/// Routed-key prefix for behavior rows in the sidebar (the
/// expandable "show this behavior's runs" toggle). Children of this
/// prefix carry `{pod_id}:{behavior_id}`. Distinct from the
/// `thread:` prefix so the two click-target families don't collide.
const BEHAVIOR_ROW_PREFIX: &str = "behavior-row:";
/// Manual-fire button inside an expanded behavior body. Sends
/// `RunBehavior` for the matching `{pod_id}:{behavior_id}` pair.
const BEHAVIOR_RUN_PREFIX: &str = "behavior-run:";
/// Pause/resume toggle inside an expanded behavior body. Sends
/// `SetBehaviorEnabled` with the inverse of the row's current
/// `enabled` flag for the matching `{pod_id}:{behavior_id}` pair.
const BEHAVIOR_TOGGLE_PREFIX: &str = "behavior-toggle:";

/// Routed-key prefix for the per-behavior Delete button. Suffix
/// is `{pod_id}:{behavior_id}` (split on the first `:`). Two-click
/// arm-confirm: first click sets `delete_armed_behavior`, second
/// click on the same key fires `DeleteBehavior`. Any unrelated
/// click disarms.
const BEHAVIOR_DELETE_PREFIX: &str = "behavior-delete:";

/// Routed-key prefix for the per-behavior Edit button on the
/// expanded behavior toolbar. Suffix carries `{pod_id}:{behavior_id}`.
/// Click opens the [`BEHAVIOR_EDITOR_KEY`] sheet and fires
/// `GetBehavior` to hydrate the form.
const BEHAVIOR_EDIT_PREFIX: &str = "behavior-edit:";

/// Routed-key prefix for the per-User-row fork affordance. Suffix is
/// `{msg_index}` (parsed back to a `usize` in `on_event`). Click
/// opens the fork dialog pre-populated with the clicked message's
/// index + seed text.
const CHAT_USER_FORK_PREFIX: &str = "chat:user-fork:";

/// Key for a User row's hover-detection wrapper. `idx` is the
/// display-item index (stable per build), distinct from `msg_index`
/// (the wire's conversation message index): we use display-item
/// position for keying so streaming-time index drift doesn't move
/// the hover key out from under the cursor.
fn chat_user_row_key(idx: usize) -> String {
    format!("chat:user-row:{idx}")
}

fn chat_user_fork_key(msg_index: usize) -> String {
    format!("{CHAT_USER_FORK_PREFIX}{msg_index}")
}

/// Routed key for clicking an inline image to open the fullscreen
/// lightbox modal. Suffix is the display-item index in the active
/// thread's `view.items` so the click handler can fetch the matching
/// `DisplayItem::Image`. Mirrors the fork prefix's idx-keyed shape.
const CHAT_IMAGE_LIGHTBOX_PREFIX: &str = "chat:image-lightbox:";

fn chat_image_lightbox_key(idx: usize) -> String {
    format!("{CHAT_IMAGE_LIGHTBOX_PREFIX}{idx}")
}

/// Routed-key shapes for the image lightbox modal.
/// `lightbox:dismiss` is the scrim's outside-click route; the
/// explicit close button uses `lightbox:close`.
const LIGHTBOX_MODAL_DISMISS_KEY: &str = "lightbox:dismiss";
const LIGHTBOX_MODAL_CLOSE_KEY: &str = "lightbox:close";

/// Routed-key shapes for the JSON tree viewer modal. The accordion
/// group key is shared across every collapsible node — the per-node
/// `value` carries the JSON pointer-shaped path so routes don't
/// collide between siblings (`json-tree:accordion:$/foo/0`).
const JSON_VIEWER_DISMISS_KEY: &str = "json-viewer:dismiss";
const JSON_VIEWER_CLOSE_KEY: &str = "json-viewer:close";
const JSON_TREE_ACCORDION_GROUP: &str = "json-tree";

/// Routed-key shapes for the edit-with-save file viewer modal.
/// `file-viewer:body` is the text_area's `target_key`; the rest are
/// click-style buttons / scrim.
const FILE_VIEWER_DISMISS_KEY: &str = "file-viewer:dismiss";
const FILE_VIEWER_CLOSE_KEY: &str = "file-viewer:close";
const FILE_VIEWER_BODY_KEY: &str = "file-viewer:body";
const FILE_VIEWER_SAVE_KEY: &str = "file-viewer:save";
const FILE_VIEWER_REVERT_KEY: &str = "file-viewer:revert";

/// Routed-key shapes for the file tree modal. The scrim uses
/// `file-tree:dismiss`; the close button uses `file-tree:close`. Dir
/// and file rows carry pod-prefixed suffixes so multi-pod state
/// doesn't alias across opens (`file-tree:dir:{pod}:{path}` /
/// `file-tree:file:{pod}:{path}`).
const FILE_TREE_DISMISS_KEY: &str = "file-tree:dismiss";
const FILE_TREE_CLOSE_KEY: &str = "file-tree:close";
const FILE_TREE_DIR_PREFIX: &str = "file-tree:dir:";
const FILE_TREE_FILE_PREFIX: &str = "file-tree:file:";

/// Sidebar header folder icon — opens the file-tree modal scoped to
/// the active pod. Rendered alongside the existing `settings` gear
/// only when some pod tab is active. Single key (no pod-id suffix)
/// since exactly one pod is active at any moment.
const SIDEBAR_POD_FILES_KEY: &str = "sidebar:pod-files";

/// Sidebar footer cog — opens the server-settings modal. Always
/// visible (server settings aren't pod-scoped). Lives in the footer
/// next to the server URL so the icon associates with "where am I
/// connected" rather than "what am I working on."
const SIDEBAR_SERVER_SETTINGS_KEY: &str = "sidebar:server-settings";

/// Thread inspector toggle — `info`-icon button on the thread
/// toolbar. Click flips `inspector_open` between the selected
/// thread id and `None`; the renderer paints an inline detail
/// panel between the header and the chat log when set.
const CHAT_INSPECTOR_TOGGLE_KEY: &str = "chat:inspector-toggle";

/// Sidebar footer database icon — opens the knowledge-buckets modal.
/// Always visible, since buckets span both server-scope and pod-
/// scope and aren't pinned to whichever pod is active. Lives in the
/// footer next to the settings cog.
const SIDEBAR_BUCKETS_KEY: &str = "sidebar:buckets";

/// Routed-key shapes for the knowledge-buckets modal. Row actions
/// (build / pause / poll / resync / delete-arm / delete-confirm /
/// delete-cancel) carry `{pod}|{id}` suffixes — `pod` is the literal
/// string `"server"` for server-scope buckets so `split_once(':')`
/// on the `pod:id` boundary stays unambiguous (real pod ids don't
/// match the sentinel).
const BUCKETS_MODAL_DISMISS_KEY: &str = "buckets:dismiss";
const BUCKETS_MODAL_CLOSE_KEY: &str = "buckets:close";
const BUCKETS_BUILD_PREFIX: &str = "buckets:build:";
const BUCKETS_PAUSE_PREFIX: &str = "buckets:pause:";
const BUCKETS_POLL_PREFIX: &str = "buckets:poll:";
const BUCKETS_RESYNC_PREFIX: &str = "buckets:resync:";
const BUCKETS_DELETE_PREFIX: &str = "buckets:delete:";
const BUCKETS_DELETE_CONFIRM_PREFIX: &str = "buckets:delete-confirm:";
const BUCKETS_DELETE_CANCEL_PREFIX: &str = "buckets:delete-cancel:";

/// Sentinel value baked into bucket row-action keys when the bucket
/// is server-scope (no owning pod). Picked so real pod ids
/// (kebab-case alphanumeric per the server's validator) can't
/// collide.
const BUCKET_SERVER_SCOPE_SENTINEL: &str = "__server__";

fn bucket_scope_token(pod_id: &Option<String>) -> &str {
    pod_id.as_deref().unwrap_or(BUCKET_SERVER_SCOPE_SENTINEL)
}

fn bucket_scope_from_token(tok: &str) -> Option<String> {
    if tok == BUCKET_SERVER_SCOPE_SENTINEL {
        None
    } else {
        Some(tok.to_string())
    }
}

/// Routed-key shapes for the server-settings modal. The tab strip
/// auto-derives `settings:tabs:tab:{value}` per trigger; per-tab
/// field keys live under `settings:{tab}:{field}`.
const SETTINGS_TABS_KEY: &str = "settings:tabs";
const SETTINGS_DISMISS_KEY: &str = "settings:dismiss";
const SETTINGS_CLOSE_KEY: &str = "settings:close";
const SETTINGS_SERVER_CONFIG_BODY_KEY: &str = "settings:server-config:body";
const SETTINGS_SERVER_CONFIG_SAVE_KEY: &str = "settings:server-config:save";
const SETTINGS_SERVER_CONFIG_REVERT_KEY: &str = "settings:server-config:revert";

fn file_tree_dir_key(pod_id: &str, path: &str) -> String {
    format!("{FILE_TREE_DIR_PREFIX}{pod_id}:{path}")
}

fn file_tree_file_key(pod_id: &str, path: &str) -> String {
    format!("{FILE_TREE_FILE_PREFIX}{pod_id}:{path}")
}

/// Routed key for the sidebar resize handle. Drag / arrow / Home /
/// End events on this key fold through `resize_handle::apply_event_fixed`
/// to mutate `sidebar_width`.
const SIDEBAR_RESIZE_KEY: &str = "sidebar:resize";

/// Routed-key shapes for the sudo-approval banner. All four routes
/// carry the `function_id` as suffix so multiple banners (one per
/// pending request, ordered by id) coexist without colliding.
const SUDO_APPROVE_PREFIX: &str = "sudo:approve:";
const SUDO_REMEMBER_PREFIX: &str = "sudo:remember:";
const SUDO_REJECT_PREFIX: &str = "sudo:reject:";
const SUDO_REJECT_REASON_PREFIX: &str = "sudo:reject-reason:";

fn sudo_approve_key(fn_id: u64) -> String {
    format!("{SUDO_APPROVE_PREFIX}{fn_id}")
}
fn sudo_remember_key(fn_id: u64) -> String {
    format!("{SUDO_REMEMBER_PREFIX}{fn_id}")
}
fn sudo_reject_key(fn_id: u64) -> String {
    format!("{SUDO_REJECT_PREFIX}{fn_id}")
}
fn sudo_reject_reason_key(fn_id: u64) -> String {
    format!("{SUDO_REJECT_REASON_PREFIX}{fn_id}")
}

/// Routed-key prefix the fork dialog uses. Extended keys:
/// `{prefix}:dismiss`, `{prefix}:archive` (Switch toggle),
/// `{prefix}:reset-caps` (Switch toggle), `{prefix}:confirm`
/// (primary), `{prefix}:cancel` (secondary).
const FORK_MODAL_KEY: &str = "fork";
const FORK_MODAL_ARCHIVE_KEY: &str = "fork:archive";
const FORK_MODAL_RESET_CAPS_KEY: &str = "fork:reset-caps";
const FORK_MODAL_CONFIRM_KEY: &str = "fork:confirm";
const FORK_MODAL_CANCEL_KEY: &str = "fork:cancel";
const FORK_MODAL_DISMISS_KEY: &str = "fork:dismiss";

/// Routed key for the active-pod settings gear in the sidebar header
/// — opens the pod editor sheet for `pod_tab` (rendered only when
/// some pod is active). Single key, not pod-id-suffixed, because at
/// any moment exactly one pod is active and the click target is
/// scoped to whatever's selected.
const SIDEBAR_POD_SETTINGS_KEY: &str = "sidebar:pod-settings";

/// Routed-key prefix the pod editor sheet uses. Extended keys:
/// `{prefix}:dismiss` (scrim), `{prefix}:tabs` (tabs_list — auto-
/// derives `pod-editor:tabs:tab:{value}` per trigger),
/// `{prefix}:save` (primary), `{prefix}:cancel` (secondary). The
/// per-tab field keys live under `{prefix}:{tab}:{field}` (see the
/// individual constants below).
const POD_EDITOR_KEY: &str = "pod-editor";
const POD_EDITOR_TABS_KEY: &str = "pod-editor:tabs";
/// Allow-tab field keys.
const POD_EDITOR_ALLOW_NAME_KEY: &str = "pod-editor:allow:name";
const POD_EDITOR_ALLOW_DESCRIPTION_KEY: &str = "pod-editor:allow:description";
const POD_EDITOR_ALLOW_BACKENDS_KEY: &str = "pod-editor:allow:backends";
const POD_EDITOR_ALLOW_MCP_HOSTS_KEY: &str = "pod-editor:allow:mcp-hosts";
const POD_EDITOR_ALLOW_BUCKETS_KEY: &str = "pod-editor:allow:buckets";
const POD_EDITOR_ALLOW_CAPS_POD_MODIFY_KEY: &str = "pod-editor:allow:caps:pod-modify";
const POD_EDITOR_ALLOW_CAPS_DISPATCH_KEY: &str = "pod-editor:allow:caps:dispatch";
const POD_EDITOR_ALLOW_CAPS_BEHAVIORS_KEY: &str = "pod-editor:allow:caps:behaviors";
/// Defaults-tab field keys.
const POD_EDITOR_DEFAULTS_BACKEND_KEY: &str = "pod-editor:defaults:backend";
const POD_EDITOR_DEFAULTS_MODEL_KEY: &str = "pod-editor:defaults:model";
const POD_EDITOR_DEFAULTS_SYSTEM_PROMPT_FILE_KEY: &str = "pod-editor:defaults:system-prompt-file";
const POD_EDITOR_DEFAULTS_MAX_TOKENS_KEY: &str = "pod-editor:defaults:max-tokens";
const POD_EDITOR_DEFAULTS_MAX_TURNS_KEY: &str = "pod-editor:defaults:max-turns";
const POD_EDITOR_DEFAULTS_TOOL_GATE_KEY: &str = "pod-editor:defaults:tool-gate";
const POD_EDITOR_DEFAULTS_HOST_ENV_KEY: &str = "pod-editor:defaults:host-env";
/// Defaults-tab tool-surface routed keys. The structured editor for
/// `thread_defaults.tool_surface` rides directly under the Defaults
/// form (the field is always present on `ToolSurface`, not Option-
/// shaped — there's no override checkbox here).
const POD_EDITOR_DEFAULTS_TOOL_SURFACE_CORE_TOOLS_KEY: &str =
    "pod-editor:defaults:tool-surface:core-tools";
const POD_EDITOR_DEFAULTS_TOOL_SURFACE_CORE_TOOLS_NAMED_KEY: &str =
    "pod-editor:defaults:tool-surface:core-tools:named";
const POD_EDITOR_DEFAULTS_TOOL_SURFACE_INITIAL_LISTING_KEY: &str =
    "pod-editor:defaults:tool-surface:initial-listing";
const POD_EDITOR_DEFAULTS_TOOL_SURFACE_ACTIVATION_SURFACE_KEY: &str =
    "pod-editor:defaults:tool-surface:activation-surface";
const POD_EDITOR_DEFAULTS_CAPS_POD_MODIFY_KEY: &str = "pod-editor:defaults:caps:pod-modify";
const POD_EDITOR_DEFAULTS_CAPS_DISPATCH_KEY: &str = "pod-editor:defaults:caps:dispatch";
const POD_EDITOR_DEFAULTS_CAPS_BEHAVIORS_KEY: &str = "pod-editor:defaults:caps:behaviors";
const POD_EDITOR_DEFAULTS_MCP_HOSTS_KEY: &str = "pod-editor:defaults:mcp-hosts";
/// Limits-tab field keys.
const POD_EDITOR_LIMITS_MAX_CONCURRENT_THREADS_KEY: &str =
    "pod-editor:limits:max-concurrent-threads";
/// Raw-tab field key (the only knob in that tab).
const POD_EDITOR_TOML_KEY: &str = "pod-editor:raw:toml";
const POD_EDITOR_SAVE_KEY: &str = "pod-editor:save";
const POD_EDITOR_CANCEL_KEY: &str = "pod-editor:cancel";
const POD_EDITOR_DISMISS_KEY: &str = "pod-editor:dismiss";

/// Routed-key prefix the per-behavior editor sheet uses. Extended
/// keys: `{prefix}:dismiss` (scrim, emitted by `sheet`),
/// `{prefix}:name` / `{prefix}:description` / `{prefix}:schedule`
/// (text_input fields), `{prefix}:prompt` (text_area),
/// `{prefix}:trigger-kind` (select_trigger + select_menu pair),
/// `{prefix}:save` (primary button), `{prefix}:cancel` (secondary).
const BEHAVIOR_EDITOR_KEY: &str = "behavior-editor";
const BEHAVIOR_EDITOR_TABS_KEY: &str = "behavior-editor:tabs";
const BEHAVIOR_EDITOR_NAME_KEY: &str = "behavior-editor:name";
const BEHAVIOR_EDITOR_DESCRIPTION_KEY: &str = "behavior-editor:description";
const BEHAVIOR_EDITOR_SCHEDULE_KEY: &str = "behavior-editor:schedule";
const BEHAVIOR_EDITOR_TIMEZONE_KEY: &str = "behavior-editor:timezone";
const BEHAVIOR_EDITOR_OVERLAP_KEY: &str = "behavior-editor:overlap";
const BEHAVIOR_EDITOR_CATCH_UP_KEY: &str = "behavior-editor:catch-up";
const BEHAVIOR_EDITOR_PROMPT_KEY: &str = "behavior-editor:prompt";
/// `select_trigger` key for the trigger-kind picker; also the prefix
/// `widgets::select` derives the per-option key from
/// (`behavior-editor:trigger-kind:option:{value}`).
const BEHAVIOR_EDITOR_TRIGGER_KIND_KEY: &str = "behavior-editor:trigger-kind";
const BEHAVIOR_EDITOR_SAVE_KEY: &str = "behavior-editor:save";
const BEHAVIOR_EDITOR_CANCEL_KEY: &str = "behavior-editor:cancel";
const BEHAVIOR_EDITOR_DISMISS_KEY: &str = "behavior-editor:dismiss";
/// Routed-key prefix the Trigger-tab cron preset chips use. Suffix
/// is the preset's array index (`0`..`CRON_PRESETS.len()-1`); the
/// click handler reads the preset's expression out of
/// [`crate::cron_preview::CRON_PRESETS`] by index. Indexing rather
/// than name-suffixing avoids escaping `* / -` inside routed keys.
const BEHAVIOR_EDITOR_CRON_PRESET_PREFIX: &str = "behavior-editor:cron-preset:";
/// Same shape, for [`crate::cron_preview::COMMON_TIMEZONES`].
const BEHAVIOR_EDITOR_TZ_PRESET_PREFIX: &str = "behavior-editor:tz-preset:";
/// Thread-tab routed keys. Each scalar `BehaviorThreadOverride`
/// field uses an `_OVERRIDE_KEY` checkbox to toggle Some/None and a
/// `_VALUE_KEY` for the actual control. Inherit-only rows render
/// the value control as a muted "(inherit pod default)" paragraph
/// instead.
const BEHAVIOR_EDITOR_THREAD_MODEL_OVERRIDE_KEY: &str = "behavior-editor:thread:model:override";
const BEHAVIOR_EDITOR_THREAD_MODEL_KEY: &str = "behavior-editor:thread:model";
const BEHAVIOR_EDITOR_THREAD_MAX_TOKENS_OVERRIDE_KEY: &str =
    "behavior-editor:thread:max-tokens:override";
const BEHAVIOR_EDITOR_THREAD_MAX_TOKENS_KEY: &str = "behavior-editor:thread:max-tokens";
const BEHAVIOR_EDITOR_THREAD_MAX_TURNS_OVERRIDE_KEY: &str =
    "behavior-editor:thread:max-turns:override";
const BEHAVIOR_EDITOR_THREAD_MAX_TURNS_KEY: &str = "behavior-editor:thread:max-turns";
const BEHAVIOR_EDITOR_THREAD_BACKEND_OVERRIDE_KEY: &str = "behavior-editor:thread:backend:override";
const BEHAVIOR_EDITOR_THREAD_BACKEND_KEY: &str = "behavior-editor:thread:backend";
/// Thread-tab `bindings.host_env` override checkbox + multi-check
/// group prefix. Per-item routes land on
/// `{HOST_ENV_KEY}:item:{name}` via `apply_checkbox_list_to_vec`,
/// matching the pod editor's host_env multi-check shape.
const BEHAVIOR_EDITOR_THREAD_HOST_ENV_OVERRIDE_KEY: &str =
    "behavior-editor:thread:host-env:override";
const BEHAVIOR_EDITOR_THREAD_HOST_ENV_KEY: &str = "behavior-editor:thread:host-env";
/// Same shape for `bindings.mcp_hosts`.
const BEHAVIOR_EDITOR_THREAD_MCP_HOSTS_OVERRIDE_KEY: &str =
    "behavior-editor:thread:mcp-hosts:override";
const BEHAVIOR_EDITOR_THREAD_MCP_HOSTS_KEY: &str = "behavior-editor:thread:mcp-hosts";
/// Scope-tab routed keys. Each `Option<...>` field on
/// `BehaviorScope` carries an `_OVERRIDE_KEY` checkbox toggling
/// Some/None and either a multi-check group prefix (resource
/// sets) or a `select_trigger` key (tools default + caps).
const BEHAVIOR_EDITOR_SCOPE_BACKENDS_OVERRIDE_KEY: &str = "behavior-editor:scope:backends:override";
const BEHAVIOR_EDITOR_SCOPE_BACKENDS_KEY: &str = "behavior-editor:scope:backends";
const BEHAVIOR_EDITOR_SCOPE_HOST_ENVS_OVERRIDE_KEY: &str =
    "behavior-editor:scope:host-envs:override";
const BEHAVIOR_EDITOR_SCOPE_HOST_ENVS_KEY: &str = "behavior-editor:scope:host-envs";
const BEHAVIOR_EDITOR_SCOPE_MCP_HOSTS_OVERRIDE_KEY: &str =
    "behavior-editor:scope:mcp-hosts:override";
const BEHAVIOR_EDITOR_SCOPE_MCP_HOSTS_KEY: &str = "behavior-editor:scope:mcp-hosts";
const BEHAVIOR_EDITOR_SCOPE_TOOLS_OVERRIDE_KEY: &str = "behavior-editor:scope:tools:override";
const BEHAVIOR_EDITOR_SCOPE_TOOLS_DEFAULT_KEY: &str = "behavior-editor:scope:tools:default";
const BEHAVIOR_EDITOR_SCOPE_CAPS_POD_MODIFY_OVERRIDE_KEY: &str =
    "behavior-editor:scope:caps:pod-modify:override";
const BEHAVIOR_EDITOR_SCOPE_CAPS_POD_MODIFY_KEY: &str = "behavior-editor:scope:caps:pod-modify";
const BEHAVIOR_EDITOR_SCOPE_CAPS_DISPATCH_OVERRIDE_KEY: &str =
    "behavior-editor:scope:caps:dispatch:override";
const BEHAVIOR_EDITOR_SCOPE_CAPS_DISPATCH_KEY: &str = "behavior-editor:scope:caps:dispatch";
const BEHAVIOR_EDITOR_SCOPE_CAPS_BEHAVIORS_OVERRIDE_KEY: &str =
    "behavior-editor:scope:caps:behaviors:override";
const BEHAVIOR_EDITOR_SCOPE_CAPS_BEHAVIORS_KEY: &str = "behavior-editor:scope:caps:behaviors";
const BEHAVIOR_EDITOR_SCOPE_TOOL_SURFACE_OVERRIDE_KEY: &str =
    "behavior-editor:scope:tool-surface:override";
/// Retention-tab routed keys.
const BEHAVIOR_EDITOR_RETENTION_KIND_KEY: &str = "behavior-editor:retention:kind";
const BEHAVIOR_EDITOR_RETENTION_DAYS_KEY: &str = "behavior-editor:retention:days";
/// SystemPrompt-tab routed keys. The override checkbox flips
/// `cfg.thread.system_prompt` between `None` and `Some(File {
/// name = conventional_path })`; the editor's
/// `working_system_prompt` buffer round-trips against the file at
/// that path on save (server reads `UpdateBehavior.system_prompt`
/// as the new content).
const BEHAVIOR_EDITOR_SYSTEM_PROMPT_OVERRIDE_KEY: &str = "behavior-editor:system-prompt:override";
const BEHAVIOR_EDITOR_SYSTEM_PROMPT_KEY: &str = "behavior-editor:system-prompt";
/// Raw-tab routed key for the behavior.toml `text_area`. Edits flip
/// `editor.raw_dirty` so a Save (or a tab-switch back to a
/// structured tab) reparses the buffer before shipping the wire
/// message.
const BEHAVIOR_EDITOR_RAW_TOML_KEY: &str = "behavior-editor:raw-toml";

/// Conventional pod-relative path for the system-prompt override
/// associated with a behavior. The toggle in the System Prompt tab
/// flips `config.thread.system_prompt` between `None` and
/// `Some(File { name = behavior_system_prompt_path(id) })`; the
/// editor buffer round-trips against this path through
/// `UpdateBehavior.system_prompt`. Mirrors the pod-level
/// `system_prompt.md` convention.
fn behavior_system_prompt_path(behavior_id: &str) -> String {
    format!("behaviors/{behavior_id}/system_prompt.md")
}

fn behavior_row_key(pod_id: &str, behavior_id: &str) -> String {
    format!("{BEHAVIOR_ROW_PREFIX}{pod_id}:{behavior_id}")
}

fn behavior_run_key(pod_id: &str, behavior_id: &str) -> String {
    format!("{BEHAVIOR_RUN_PREFIX}{pod_id}:{behavior_id}")
}

fn behavior_toggle_key(pod_id: &str, behavior_id: &str) -> String {
    format!("{BEHAVIOR_TOGGLE_PREFIX}{pod_id}:{behavior_id}")
}

fn behavior_delete_key(pod_id: &str, behavior_id: &str) -> String {
    format!("{BEHAVIOR_DELETE_PREFIX}{pod_id}:{behavior_id}")
}

fn behavior_edit_key(pod_id: &str, behavior_id: &str) -> String {
    format!("{BEHAVIOR_EDIT_PREFIX}{pod_id}:{behavior_id}")
}

/// Composite expansion-state key for `expanded_behaviors`. Mirrors
/// the route key but without the prefix — the set is internal,
/// so the leading `behavior-row:` isn't load-bearing.
fn behavior_expand_key(pod_id: &str, behavior_id: &str) -> String {
    format!("{pod_id}::{behavior_id}")
}

/// Per-item route key inside a checkbox-list (multi-check rendered
/// as a column of `[checkbox, label]` rows). Suffix carries the
/// item's wire value.
fn checkbox_list_item_key(prefix: &str, value: &str) -> String {
    format!("{prefix}:item:{value}")
}

/// Apply a click on a checkbox-list item to a `Vec<String>` of
/// selected values. Clicks land on `{prefix}:item:{value}`; the
/// helper toggles the value's membership and keeps the Vec sorted
/// for a stable on-disk layout. Returns whether the set changed.
///
/// Custom rather than reusing `toggle::apply_event_multi` because
/// aetna's `toggle_group_multi` is a non-wrapping row and the pod
/// editor's narrow sheet (360 px) overflows once a few labels add
/// up. Column-of-checkboxes is the wrapping-friendly fallback.
fn apply_checkbox_list_to_vec(vec: &mut Vec<String>, event: &UiEvent, prefix: &str) -> bool {
    let Some(target) = event.route() else {
        return false;
    };
    let item_prefix = format!("{prefix}:item:");
    let Some(value) = target.strip_prefix(&item_prefix) else {
        return false;
    };
    if !matches!(event.kind, UiEventKind::Click | UiEventKind::Activate) {
        return false;
    }
    if let Some(idx) = vec.iter().position(|v| v == value) {
        vec.remove(idx);
    } else {
        vec.push(value.to_string());
        vec.sort();
    }
    true
}

/// Column-of-checkboxes layout for a multi-check resource list.
/// Each option becomes a `[checkbox, text(label)]` row keyed
/// `{group_prefix}:item:{value}`. Use this in narrow surfaces (the
/// pod editor sheet is 360 px) where aetna's non-wrapping
/// `toggle_group_multi` would overflow horizontally.
fn checkbox_column(group_prefix: &str, selected: &[String], options: Vec<(String, String)>) -> El {
    let rows: Vec<El> = options
        .into_iter()
        .map(|(value, label)| {
            let on = selected.iter().any(|s| s == &value);
            let key = checkbox_list_item_key(group_prefix, &value);
            row([
                checkbox(on).key(&key),
                text(label).text_color(tokens::FOREGROUND),
            ])
            .gap(tokens::SPACE_2)
            .align(Align::Center)
            .width(Size::Fill(1.0))
        })
        .collect();
    column(rows).gap(tokens::SPACE_1).width(Size::Fill(1.0))
}

fn pod_modify_cap_label(cap: whisper_agent_protocol::PodModifyCap) -> &'static str {
    use whisper_agent_protocol::PodModifyCap as C;
    match cap {
        C::None => "none",
        C::Memories => "memories",
        C::Content => "content",
        C::ModifyAllow => "modify_allow",
    }
}

fn pod_modify_cap_from_wire(s: &str) -> Option<whisper_agent_protocol::PodModifyCap> {
    use whisper_agent_protocol::PodModifyCap as C;
    match s {
        "none" => Some(C::None),
        "memories" => Some(C::Memories),
        "content" => Some(C::Content),
        "modify_allow" => Some(C::ModifyAllow),
        _ => None,
    }
}

fn dispatch_cap_label(cap: whisper_agent_protocol::DispatchCap) -> &'static str {
    use whisper_agent_protocol::DispatchCap as C;
    match cap {
        C::None => "none",
        C::WithinScope => "within_scope",
    }
}

fn dispatch_cap_from_wire(s: &str) -> Option<whisper_agent_protocol::DispatchCap> {
    use whisper_agent_protocol::DispatchCap as C;
    match s {
        "none" => Some(C::None),
        "within_scope" => Some(C::WithinScope),
        _ => None,
    }
}

fn behaviors_cap_label(cap: whisper_agent_protocol::BehaviorOpsCap) -> &'static str {
    use whisper_agent_protocol::BehaviorOpsCap as C;
    match cap {
        C::None => "none",
        C::Read => "read",
        C::AuthorNarrower => "author_narrower",
        C::AuthorAny => "author_any",
    }
}

fn behaviors_cap_from_wire(s: &str) -> Option<whisper_agent_protocol::BehaviorOpsCap> {
    use whisper_agent_protocol::BehaviorOpsCap as C;
    match s {
        "none" => Some(C::None),
        "read" => Some(C::Read),
        "author_narrower" => Some(C::AuthorNarrower),
        "author_any" => Some(C::AuthorAny),
        _ => None,
    }
}

fn tool_gate_label(d: Disposition) -> &'static str {
    match d {
        Disposition::Allow => "allow_all",
        Disposition::Deny => "deny_all",
    }
}

fn tool_gate_from_wire(s: &str) -> Option<Disposition> {
    match s {
        "allow_all" => Some(Disposition::Allow),
        "deny_all" => Some(Disposition::Deny),
        _ => None,
    }
}

/// Bare `Disposition` label for the Scope tab's `tools.default`
/// picker. Different from `tool_gate_label` (which uses the
/// `_all` suffix for the pod-level "tool gate" framing) because
/// the Scope tools default isn't a pod-wide gate; it's the
/// fallback for un-overridden tools in this behavior's narrowed
/// allow map.
fn disposition_label(d: Disposition) -> &'static str {
    match d {
        Disposition::Allow => "allow",
        Disposition::Deny => "deny",
    }
}

fn disposition_from_wire(s: &str) -> Option<Disposition> {
    match s {
        "allow" => Some(Disposition::Allow),
        "deny" => Some(Disposition::Deny),
        _ => None,
    }
}

/// Default seed text for the tool-surface `core_tools` named buffer
/// when toggling `All` → `Named`. Mirrors the egui sibling's
/// `default_core_tools_text`. The actual `CoreTools::Named(default)`
/// at protocol level is the same three names.
fn default_core_tools_text() -> String {
    "describe_tool\nfind_tool\nrequest_escalation".to_string()
}

/// Parse the multi-line named-tools buffer back into a sorted-by-
/// user-order `Vec<String>`. Empty lines / whitespace-only lines
/// drop. Mirrors the egui sibling's parse loop.
fn parse_core_tools_named(buf: &str) -> Vec<String> {
    buf.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn initial_listing_label(l: InitialListing) -> &'static str {
    match l {
        InitialListing::None => "none",
        InitialListing::AllNames => "all_names",
        InitialListing::CoreOnly => "core_only",
    }
}

fn initial_listing_from_wire(s: &str) -> Option<InitialListing> {
    match s {
        "none" => Some(InitialListing::None),
        "all_names" => Some(InitialListing::AllNames),
        "core_only" => Some(InitialListing::CoreOnly),
        _ => None,
    }
}

fn activation_surface_label(a: ActivationSurface) -> &'static str {
    match a {
        ActivationSurface::Announce => "announce",
        ActivationSurface::InjectSchema => "inject_schema",
    }
}

fn activation_surface_from_wire(s: &str) -> Option<ActivationSurface> {
    match s {
        "announce" => Some(ActivationSurface::Announce),
        "inject_schema" => Some(ActivationSurface::InjectSchema),
        _ => None,
    }
}

fn overlap_label(o: Overlap) -> &'static str {
    match o {
        Overlap::Skip => "skip",
        Overlap::QueueOne => "queue_one",
        Overlap::Allow => "allow",
    }
}

fn overlap_from_wire(s: &str) -> Option<Overlap> {
    match s {
        "skip" => Some(Overlap::Skip),
        "queue_one" => Some(Overlap::QueueOne),
        "allow" => Some(Overlap::Allow),
        _ => None,
    }
}

fn catch_up_label(c: CatchUp) -> &'static str {
    match c {
        CatchUp::None => "none",
        CatchUp::One => "one",
        CatchUp::All => "all",
    }
}

fn catch_up_from_wire(s: &str) -> Option<CatchUp> {
    match s {
        "none" => Some(CatchUp::None),
        "one" => Some(CatchUp::One),
        "all" => Some(CatchUp::All),
        _ => None,
    }
}

fn retention_kind_label(p: &RetentionPolicy) -> &'static str {
    match p {
        RetentionPolicy::Keep => "keep",
        RetentionPolicy::ArchiveAfterDays { .. } => "archive_after_days",
        RetentionPolicy::DeleteAfterDays { .. } => "delete_after_days",
    }
}

// Routed keys for the new-thread compose form's three `select_trigger`
// pickers. Each `select_menu` in the build pass shares its trigger key
// so the popover anchors below the trigger; the same key is what we
// classify on in `on_event`.
const PICKER_BACKEND: &str = "picker:backend";
const PICKER_MODEL: &str = "picker:model";
const PICKER_POD: &str = "picker:pod";

/// Sentinel option value emitted by the "inherit / auto / default"
/// menu row each picker prepends. Empty string can't collide with a
/// real backend / model id / pod id. Picker pick handlers map this
/// back to `None` on the corresponding picker field.
const PICKER_INHERIT: &str = "";

impl ChatApp {
    /// MIME-sniff a raw byte payload and stage it as a compose
    /// attachment, surfacing a hint on every outcome (success and
    /// failure). Used by both the drop handler and the file-picker
    /// drain so the pipeline is identical regardless of input
    /// source — silently-rejected drops were the prior pain point
    /// the egui sibling fixed by always surfacing a hint.
    fn stage_raw_pick(&mut self, pick: RawPick) {
        if pick.bytes.is_empty() {
            self.set_compose_hint(format!("{} was empty; ignored", pick.source_desc));
            return;
        }
        let Some(mime) = sniff_image_mime(&pick.bytes) else {
            self.set_compose_hint(format!(
                "{} doesn't look like a supported image (jpeg / png / webp / gif)",
                pick.source_desc
            ));
            return;
        };
        // Pre-decode for the thumbnail. HEIC/HEIF the protocol
        // accepts but the `image` crate doesn't decode without an
        // extra C library — so the staged row carries no thumbnail
        // (placeholder), but the bytes still ride to the server
        // (Gemini accepts them; other adapters reject at dispatch).
        let thumbnail = image::load_from_memory(&pick.bytes).ok().map(|decoded| {
            let rgba = decoded.to_rgba8();
            let (w, h) = rgba.dimensions();
            aetna_core::image::Image::from_rgba8(w, h, rgba.into_raw())
        });
        let id = self.next_attachment_id;
        self.next_attachment_id += 1;
        let source_desc = pick.source_desc.clone();
        self.compose_attachments.push(StagedAttachment {
            id,
            attachment: Attachment::Image {
                source: ImageSource::Bytes {
                    media_type: mime,
                    data: pick.bytes,
                },
            },
            thumbnail,
            source_desc: pick.source_desc,
        });
        self.set_compose_hint(format!(
            "attached {} as {}",
            source_desc,
            mime.as_mime_str()
        ));
    }

    /// Set the ephemeral compose hint with a fixed expiration
    /// window. 4 seconds is enough to read the line without
    /// lingering when the user moves on; matches the egui sibling's
    /// timing.
    fn set_compose_hint(&mut self, msg: String) {
        let expires = std::time::Instant::now() + std::time::Duration::from_secs(4);
        self.compose_hint = Some((msg, expires));
    }

    /// Drain the file-picker handoff queue into staged attachments.
    /// Called once per frame from `before_build`; on the empty fast
    /// path the lock cost is negligible, so the always-on poll
    /// matches the egui sibling's `ingest_pending_picks` shape.
    fn drain_pending_picks(&mut self) {
        let picks: Vec<RawPick> = match self.pending_picks.lock() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            Err(_) => return,
        };
        for pick in picks {
            self.stage_raw_pick(pick);
        }
    }

    /// Spawn the OS file picker. Native runs rfd's `AsyncFileDialog`
    /// on a background thread driven by a current-thread tokio
    /// runtime; the resolved bytes land on `pending_picks` and the
    /// next `before_build` drains them through the same staging
    /// pipeline as drag-drop. Wasm gets a no-op + hint until the
    /// browser-entry slice lands (the rfd wasm path needs the same
    /// `wasm-bindgen-futures::spawn_local` glue Stage 11 will set up).
    #[cfg(not(target_arch = "wasm32"))]
    fn spawn_file_picker(&self) {
        let queue = self.pending_picks.clone();
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
            });
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn spawn_file_picker(&mut self) {
        // Stage 11 wasm entry will wire rfd's wasm path; until then
        // surface a hint so the user knows the click registered.
        self.set_compose_hint("file picker not available in this build (drag-drop works)".into());
    }

    fn send_compose(&mut self) {
        let text = self.active_compose_text().trim().to_string();
        // Allow attachments-only messages (text empty + at least
        // one staged image). Mirrors egui's gate; an entirely empty
        // message wastes a turn either way.
        if text.is_empty() && self.compose_attachments.is_empty() {
            return;
        }
        // Drain staged attachments into the wire's `Vec<Attachment>`.
        // Same order they were staged — the user expects "first
        // dropped, first delivered" so the model sees a stable
        // sequence.
        let attachments: Vec<Attachment> = self
            .compose_attachments
            .drain(..)
            .map(|s| s.attachment)
            .collect();
        if let Some(thread_id) = self.selected.clone() {
            // Clear the per-thread draft both locally and on the
            // server. `SetThreadDraft { text: "" }` doubles as the
            // "we're submitting this draft" signal so other
            // subscribers stop seeing the in-progress text.
            self.drafts.remove(&thread_id);
            self.send(ClientToServer::SetThreadDraft {
                thread_id: thread_id.clone(),
                text: String::new(),
            });
            self.send(ClientToServer::SendUserMessage {
                thread_id,
                text,
                attachments,
            });
        } else {
            // No selection -> the compose form is in new-thread
            // mode. Materialize the picker state into a
            // `CreateThread` request. `ThreadCreated` will land in
            // `dispatch_wire` and auto-select the result.
            let (config_override, bindings_request, pod_id) = self.build_creation_request();
            self.compose_input.clear();
            self.send(ClientToServer::CreateThread {
                correlation_id: None,
                pod_id,
                initial_message: text,
                initial_attachments: attachments,
                config_override,
                bindings_request,
            });
        }
        // Reset selection inside the now-empty compose box so the
        // caret lands at offset 0 on the next frame.
        self.selection = Selection::default();
    }

    /// Active compose buffer for the current selection state. When a
    /// thread is selected this is the per-thread draft; otherwise
    /// it's the new-thread form's `compose_input`. Used by both the
    /// `compose_box` and `new_thread_pane` `text_area` builders.
    fn active_compose_text(&self) -> &str {
        match self.selected.as_ref() {
            Some(tid) => self.drafts.get(tid).map(String::as_str).unwrap_or(""),
            None => self.compose_input.as_str(),
        }
    }

    /// Pick the best `(config_override, bindings_request, pod_id)` triple
    /// for the next `CreateThread` from the compose-form picker state.
    /// Mirrors the egui sibling's `build_creation_override` shape so the
    /// two clients send identical wire requests.
    ///
    /// Critical: when the user picks a backend, the matching
    /// `ThreadBindingsRequest { backend, .. }` MUST ride alongside the
    /// `ThreadConfigOverride { model, .. }` — otherwise the server pairs
    /// the user-picked model with the pod's `thread_defaults.backend`
    /// and the upstream API rejects the request as "unknown model".
    fn build_creation_request(
        &self,
    ) -> (
        Option<ThreadConfigOverride>,
        Option<ThreadBindingsRequest>,
        Option<String>,
    ) {
        let backend = self.picker_backend.clone();
        // If the user picked a backend but didn't touch the model
        // dropdown, pin down the model explicitly so the server
        // doesn't fall back to the *default backend's* default_model
        // (which would be wrong for the picked backend). Prefer the
        // picked backend's `default_model`; else the first model the
        // backend's `/models` returned; else `None`.
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
        let config_override = if model.is_some() {
            Some(ThreadConfigOverride {
                model,
                ..Default::default()
            })
        } else {
            None
        };
        let bindings_request = backend.as_ref().map(|_| ThreadBindingsRequest {
            backend: backend.clone(),
            host_env: None,
            mcp_hosts: None,
        });
        (config_override, bindings_request, self.picker_pod.clone())
    }

    fn handle_backend_pick(&mut self, action: SelectAction) {
        match action {
            SelectAction::Toggle => {
                self.close_other_pickers(PICKER_BACKEND);
                self.picker_backend_open = !self.picker_backend_open;
            }
            SelectAction::Dismiss => self.picker_backend_open = false,
            SelectAction::Pick(value) => {
                let new_value = if value == PICKER_INHERIT {
                    None
                } else {
                    Some(value)
                };
                if self.picker_backend != new_value {
                    // Picking a new backend invalidates the model
                    // selection — model ids are backend-scoped.
                    self.picker_model = None;
                    if let Some(b) = new_value.as_ref() {
                        self.ensure_models_requested(b);
                    }
                    self.picker_backend = new_value;
                }
                self.picker_backend_open = false;
            }
            // `SelectAction` is `#[non_exhaustive]` — future variants
            // (e.g. keyboard navigation) drop on the floor here until
            // the picker grows a corresponding handler.
            _ => {}
        }
    }

    fn handle_model_pick(&mut self, action: SelectAction) {
        match action {
            SelectAction::Toggle => {
                self.close_other_pickers(PICKER_MODEL);
                self.picker_model_open = !self.picker_model_open;
            }
            SelectAction::Dismiss => self.picker_model_open = false,
            SelectAction::Pick(value) => {
                self.picker_model = if value == PICKER_INHERIT {
                    None
                } else {
                    Some(value)
                };
                self.picker_model_open = false;
            }
            _ => {}
        }
    }

    fn handle_pod_pick(&mut self, action: SelectAction) {
        match action {
            SelectAction::Toggle => {
                self.close_other_pickers(PICKER_POD);
                self.picker_pod_open = !self.picker_pod_open;
            }
            SelectAction::Dismiss => self.picker_pod_open = false,
            SelectAction::Pick(value) => {
                self.picker_pod = if value == PICKER_INHERIT {
                    None
                } else {
                    Some(value)
                };
                self.picker_pod_open = false;
            }
            _ => {}
        }
    }

    /// Close every picker except the one identified by `keep_open`.
    /// Two open menus at once would visually overlap and confuse the
    /// click-outside dismiss behavior; aetna's popover scrim is per
    /// menu, so we enforce single-open-at-a-time at the app layer.
    fn close_other_pickers(&mut self, keep_open: &str) {
        if keep_open != PICKER_BACKEND {
            self.picker_backend_open = false;
        }
        if keep_open != PICKER_MODEL {
            self.picker_model_open = false;
        }
        if keep_open != PICKER_POD {
            self.picker_pod_open = false;
        }
        // The behavior editor's trigger-kind picker shares the same
        // single-open-at-a-time discipline. Its key isn't one of the
        // new-thread pickers, so it always closes when something else
        // toggles.
        if keep_open != BEHAVIOR_EDITOR_TRIGGER_KIND_KEY
            && let Some(editor) = self.behavior_editor.as_mut()
        {
            editor.trigger_kind_open = false;
        }
        // Pod editor pickers — every variant maps 1:1 to a routed
        // `select_trigger` key via `PodEditorPicker::key`. If the
        // open picker's key isn't `keep_open`, close it.
        if let Some(editor) = self.pod_editor.as_mut()
            && let Some(open) = editor.open_picker
            && keep_open != open.key()
        {
            editor.open_picker = None;
        }
        // Behavior editor's overlap / catch_up pickers — same
        // single-active discipline. The trigger-kind picker is
        // closed via the dedicated `trigger_kind_open` boolean
        // above; everything else lives under `open_picker`.
        if let Some(editor) = self.behavior_editor.as_mut()
            && let Some(open) = editor.open_picker
            && keep_open != open.key()
        {
            editor.open_picker = None;
        }
    }

    /// Fire a `ListModels { backend }` if we haven't already on this
    /// connection. Idempotent — repeated picks of the same backend
    /// hit the local `models_by_backend` cache.
    fn ensure_models_requested(&mut self, backend: &str) {
        if self.requested_models_for.contains(backend) {
            return;
        }
        self.send(ClientToServer::ListModels {
            correlation_id: None,
            backend: backend.to_string(),
        });
        self.requested_models_for.insert(backend.to_string());
    }

    /// Fire a `ListBehaviors { pod_id }` if we haven't already on
    /// this connection. Subsequent `Behavior*` broadcasts keep the
    /// cache fresh; the dedup means repeated `PodList` echoes don't
    /// cause N×M traffic.
    fn ensure_behaviors_requested(&mut self, pod_id: &str) {
        if self.requested_behaviors_for.contains(pod_id) {
            return;
        }
        self.send(ClientToServer::ListBehaviors {
            correlation_id: None,
            pod_id: pod_id.to_string(),
        });
        self.requested_behaviors_for.insert(pod_id.to_string());
    }

    fn select_thread(&mut self, thread_id: String) {
        // No-op when re-selecting the already-selected thread —
        // avoids resetting the selection cursor when the user
        // re-clicks the row.
        if self.selected.as_deref() == Some(thread_id.as_str()) {
            return;
        }
        self.selected = Some(thread_id.clone());
        // Selection offsets indexed into the previous buffer; reset
        // so the cursor lands at offset 0 of the newly-bound draft.
        self.selection = Selection::default();
        // Subscribe lazily — once per thread per connection. The
        // server replies with a `ThreadSnapshot` we render in
        // `dispatch_wire`.
        if !self.subscribed.contains(&thread_id) {
            self.send(ClientToServer::SubscribeToThread {
                thread_id: thread_id.clone(),
            });
            self.subscribed.insert(thread_id);
        }
    }

    fn sidebar(&self) -> El {
        // The redesign: pod tabs at the top, single active pod whose
        // threads (and eventually behaviors) fill the rest. The
        // egui sibling's pod-as-collapsible-section idiom doesn't
        // scale when one pod dominates and the others are empty —
        // tabs put the focus on the workspace the user is in.
        //
        // Sidebar header: brand mark + title fill the leading edge;
        // the "+ new pod" affordance and the per-pod settings gear
        // hug the trailing edge. The connection badge moved out of
        // the header to the sidebar footer so the header reads as
        // pure identity / actions, not "where am I connected." The
        // "+" is the global pod-creation affordance (the threads-
        // section "+" is per-pod), so it stays in the header to keep
        // the two scopes visually distinct.
        let mut header_row: Vec<El> = vec![
            // Sparkles brand mark — single-glyph identity that
            // distinguishes whisper-agent from a generic shadcn
            // shell. Sized just under the title so the two read as
            // a single chip rather than the icon dominating.
            icon(crate::icons::ICON_SPARKLES.clone())
                .icon_size(tokens::ICON_SM)
                .text_color(tokens::PRIMARY),
            // Title eats the slack and ellipses if the right-hand
            // chrome grows past the remaining width. The app name
            // is fixed and contextual; clipping it is fine.
            text("whisper-agent")
                .title()
                .ellipsis()
                .width(Size::Fill(1.0)),
            icon_button("plus")
                .key(SIDEBAR_NEW_POD_KEY)
                .ghost()
                .icon_size(tokens::ICON_XS),
        ];
        if self.pod_tab.is_some() {
            header_row.push(
                icon_button("folder")
                    .key(SIDEBAR_POD_FILES_KEY)
                    .ghost()
                    .icon_size(tokens::ICON_XS),
            );
            header_row.push(
                icon_button("settings")
                    .key(SIDEBAR_POD_SETTINGS_KEY)
                    .ghost()
                    .icon_size(tokens::ICON_XS),
            );
        }
        let header_el = sidebar_header([row(header_row)
            .align(Align::Center)
            .gap(tokens::SPACE_2)
            .width(Size::Fill(1.0))]);

        // Inner scroll viewport keeps the header pinned at the top and
        // the footer pinned at the bottom while the body (pod tabs,
        // threads, behaviors) scrolls independently. Without this,
        // long thread lists push the footer past the bottom edge of
        // the viewport. Keyed so the offset persists across rebuilds.
        let mut body_entries: Vec<El> = Vec::new();

        if self.pods.is_empty() {
            // Empty workspace: small inline icon + body so the
            // sidebar doesn't read as a single dead "no pods yet"
            // line. The "+" in the header above is the actionable
            // affordance — this block just makes the empty state
            // legible at a glance.
            body_entries.push(self.sidebar_inline_empty(
                crate::icons::ICON_INBOX.clone(),
                "No pods yet",
                "Use \u{2018}+\u{2019} above to create one.",
            ));
        } else {
            body_entries.push(self.pod_tabs());

            if let Some(active) = self.pod_tab.as_deref() {
                // Threads (interactive — `origin == None`) come first;
                // they're the day-to-day reading order. Behavior-
                // spawned threads (`origin == Some(behavior_id)`) nest
                // under their parent behavior in the section below,
                // since grouping per-behavior runs is more useful than
                // mixing them into the interactive list. Mirrors the
                // egui sibling's partition.
                body_entries.push(self.threads_section(active));

                // Render the section once the per-pod `BehaviorList`
                // round-trip has landed — even if the list is empty,
                // since the header carries the "+ New behavior"
                // affordance the user still needs. Skip entirely
                // before the round-trip arrives so a "just connected"
                // sidebar doesn't flash an empty section that
                // immediately repopulates a frame later.
                if let Some(behaviors) = self.behaviors_by_pod.get(active) {
                    body_entries.push(self.behaviors_section(active, behaviors));
                }
            }
        }

        // 2 px of horizontal padding gives the inner items' focus
        // rings room to paint inside the scroll's clip rect (rings
        // extend `tokens::RING_WIDTH` outside the row's bounding box).
        // Without it the lint flags every keyboard-focusable thread
        // row as clipped.
        let body = scroll(body_entries)
            .key("sidebar-content")
            .gap(tokens::SPACE_4)
            .padding(Sides {
                left: tokens::RING_WIDTH,
                right: tokens::RING_WIDTH,
                top: 0.0,
                bottom: 0.0,
            })
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0));

        sidebar([header_el, body, self.sidebar_footer()])
    }

    /// Sidebar footer block — server URL + connection status badge,
    /// rendered as a thin top-bordered strip at the bottom of the
    /// sidebar. The server URL line is suppressed when the host
    /// hasn't supplied one (CLI flag missing, login-skip path);
    /// the connection status line always renders.
    ///
    /// Both lines use the same small / muted typography with a
    /// leading 8 px chrome dot — the server-URL row's lucide-server
    /// icon and the status row's colored circle. Treating them as
    /// a single visual family keeps the footer reading as one
    /// quiet metadata strip rather than "icon-and-text plus a
    /// floating pill."
    fn sidebar_footer(&self) -> El {
        let mut rows: Vec<El> = Vec::new();
        let buckets = icon_button(crate::icons::ICON_DATABASE.clone())
            .key(SIDEBAR_BUCKETS_KEY)
            .ghost()
            .icon_size(tokens::ICON_XS);
        let cog = icon_button("settings")
            .key(SIDEBAR_SERVER_SETTINGS_KEY)
            .ghost()
            .icon_size(tokens::ICON_XS);
        if let Some(label) = self.server_label.as_deref() {
            rows.push(
                row([
                    icon(crate::icons::ICON_SERVER.clone())
                        .icon_size(tokens::ICON_XS)
                        .text_color(tokens::MUTED_FOREGROUND),
                    text(label)
                        .small()
                        .muted()
                        .ellipsis()
                        .width(Size::Fill(1.0)),
                    buckets,
                    cog,
                ])
                .gap(tokens::SPACE_2)
                .align(Align::Center)
                .width(Size::Fill(1.0)),
            );
        } else {
            // No server-URL row to host the chrome — drop the
            // affordances onto a standalone right-aligned row.
            rows.push(
                row([buckets, cog])
                    .gap(tokens::SPACE_2)
                    .justify(Justify::End)
                    .width(Size::Fill(1.0)),
            );
        }
        rows.push(self.connection_status_line());
        column(rows)
            .gap(tokens::SPACE_2)
            .padding(Sides {
                left: tokens::SPACE_2,
                right: tokens::SPACE_2,
                top: tokens::SPACE_3,
                bottom: tokens::SPACE_1,
            })
            .width(Size::Fill(1.0))
            .stroke(tokens::BORDER)
    }

    /// Status-dot + label row used in the sidebar footer. The dot's
    /// color carries the semantic (`success` / `muted` / `warning` /
    /// `destructive`) and the label is muted text — same shape as
    /// the server-URL row above it. `Connecting` reads as `…`-
    /// suffixed; the others are single words. No colored fill on
    /// the text — putting the color on the dot lets the row sit
    /// quietly next to the server-URL line instead of competing.
    fn connection_status_line(&self) -> El {
        let dot_color = match self.conn_status {
            ConnectionStatus::Connected => tokens::SUCCESS,
            ConnectionStatus::Connecting => tokens::MUTED_FOREGROUND,
            ConnectionStatus::Closed => tokens::WARNING,
            ConnectionStatus::Error => tokens::DESTRUCTIVE,
        };
        // 8 px circle aligned to the same gutter the lucide-server
        // icon occupies in the row above. `radius(4)` on an 8 px
        // square gives a true circle on integer pixel boundaries.
        let dot = El::new(Kind::Custom("status-dot"))
            .width(Size::Fixed(8.0))
            .height(Size::Fixed(8.0))
            .fill(dot_color)
            .radius(4.0);
        // Center the dot inside the same 14 px box `tokens::ICON_XS`
        // produces, so the dot's left edge lines up with the server
        // icon's left edge — the two footer rows then share a clean
        // leading gutter.
        let dot_box = column([dot])
            .width(Size::Fixed(tokens::ICON_XS))
            .height(Size::Fixed(tokens::ICON_XS))
            .align(Align::Center)
            .justify(Justify::Center);

        row([
            dot_box,
            text(self.conn_status.label())
                .small()
                .muted()
                .width(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .width(Size::Fill(1.0))
    }

    /// Compact inline empty-state strip used inside the sidebar (where
    /// the chat-pane's full `empty_state(…)` medallion would dwarf the
    /// 224 px column). Icon + headline on one line, muted body on a
    /// second.
    fn sidebar_inline_empty(
        &self,
        icon_src: aetna_core::SvgIcon,
        headline: &str,
        body: &str,
    ) -> El {
        column([
            row([
                icon(icon_src)
                    .icon_size(tokens::ICON_SM)
                    .text_color(tokens::MUTED_FOREGROUND),
                text(headline).small().semibold(),
            ])
            .gap(tokens::SPACE_2)
            .align(Align::Center),
            text(body).xsmall().muted(),
        ])
        .gap(tokens::SPACE_1)
        .padding(Sides::xy(tokens::SPACE_2, tokens::SPACE_2))
        .width(Size::Fill(1.0))
    }

    /// Behaviors subsection for the active pod: header + per-behavior
    /// expandable `item`-shaped rows. Each behavior's spawned threads
    /// (those whose `origin.behavior_id` matches) nest below the row
    /// when expanded — mirrors the egui sibling's grouping so per-run
    /// history reads as the behavior's own log instead of polluting
    /// the interactive thread list.
    fn behaviors_section(&self, pod_id: &str, behaviors: &[BehaviorSummary]) -> El {
        let total = behaviors.len();
        let header_label = format!("Behaviors ({total})");
        // Mirrors the threads-section "+" pattern (slice δ): per-pod
        // entry point for `CreateBehavior`. The pod id rides on the
        // routed-key suffix so opening the modal can scope to the
        // right pod without inferring it from the active tab —
        // future drag-and-drop could fire this against a non-active
        // pod and the routing would still be correct.
        let new_key = format!("{SIDEBAR_NEW_BEHAVIOR_PREFIX}{pod_id}");
        let header = row([
            sidebar_group_label(header_label).width(Size::Fill(1.0)),
            icon_button("plus")
                .key(new_key)
                .ghost()
                .icon_size(tokens::ICON_XS),
        ])
        .align(Align::Center)
        .padding(Sides::xy(tokens::SPACE_2, tokens::SPACE_1))
        .width(Size::Fill(1.0));

        // Pre-bucket the active pod's behavior-origin threads by
        // behavior_id so each row can reach in for its own runs
        // without a per-row scan of the threads map.
        let mut threads_by_behavior: HashMap<&str, Vec<&ThreadSummary>> = HashMap::new();
        for t in self.threads.values().filter(|t| t.pod_id == pod_id) {
            if let Some(origin) = &t.origin {
                threads_by_behavior
                    .entry(origin.behavior_id.as_str())
                    .or_default()
                    .push(t);
            }
        }
        for runs in threads_by_behavior.values_mut() {
            runs.sort_by(|a, b| b.last_active.cmp(&a.last_active));
        }

        // Empty pod: section header (with the "+") still renders so
        // the affordance is always reachable. Mirrors the
        // threads-section "no threads in this pod yet" empty state.
        if behaviors.is_empty() {
            return sidebar_group(vec![
                header,
                self.sidebar_inline_empty(
                    crate::icons::ICON_ZAP.clone(),
                    "No behaviors yet",
                    "Triggered runs will appear here.",
                ),
            ]);
        }

        let mut rows: Vec<El> = Vec::new();
        for b in behaviors {
            let runs = threads_by_behavior
                .get(b.behavior_id.as_str())
                .cloned()
                .unwrap_or_default();
            let expand_key = behavior_expand_key(pod_id, &b.behavior_id);
            let expanded = self.expanded_behaviors.contains(&expand_key);
            rows.push(self.behavior_item_row(pod_id, b, runs.len(), expanded));
            if expanded {
                rows.push(self.behavior_actions_row(pod_id, b));
                if runs.is_empty() {
                    rows.push(text("no runs yet").muted().small().padding(Sides {
                        left: tokens::SPACE_3 + tokens::SPACE_3,
                        right: tokens::SPACE_3,
                        top: tokens::SPACE_1,
                        bottom: tokens::SPACE_1,
                    }));
                } else {
                    for run in runs {
                        // Nested under its behavior — the parent
                        // row already names the origin, so suppress
                        // the per-row "via" marker.
                        rows.push(self.thread_item_row(run, 1, false));
                    }
                }
            }
        }

        sidebar_group(vec![header, item_group(rows)])
    }

    /// One behavior row in the sidebar — visually item-shaped with a
    /// chevron indicator in the actions slot. Click anywhere on the
    /// row toggles `expanded_behaviors` membership, which controls
    /// whether the behavior's spawned threads render nested below
    /// in [`behaviors_section`]. Title is the display name; description
    /// is `"{kind} · {status}"` (or `"errored: {reason}"` for
    /// load-error rows). Run-count suffixed when nonzero so the user
    /// sees the run history without expanding.
    fn behavior_item_row(
        &self,
        pod_id: &str,
        b: &BehaviorSummary,
        run_count_in_view: usize,
        expanded: bool,
    ) -> El {
        let title = if b.load_error.is_some() {
            format!("⚠ {}", b.name)
        } else {
            b.name.clone()
        };
        // Errored behaviors surface the parse / validation reason
        // directly — there's no useful trigger_kind to compose with
        // and the message is the actionable detail. Healthy rows
        // compose `{kind} · {status}` so the kind tag is always the
        // first thing the eye lands on.
        let secondary = if let Some(err) = b.load_error.as_deref() {
            // Truncate aggressively — the sidebar is narrow and the
            // full message belongs in the (eventual) editor modal.
            let preview: String = err.chars().take(60).collect();
            if preview.len() < err.len() {
                format!("errored: {preview}…")
            } else {
                format!("errored: {preview}")
            }
        } else {
            let kind = b.trigger_kind.as_deref().unwrap_or("manual");
            let status = if !b.enabled {
                "paused".to_string()
            } else if let Some(last) = b.last_fired_at.as_deref() {
                format!("last fired {}", format_relative(last))
            } else {
                "no runs yet".to_string()
            };
            format!("{kind} · {status}")
        };

        let mut content_blocks: Vec<El> = vec![item_title(title), item_description(secondary)];
        // Show a small run-count badge on the row when there are
        // spawned threads in view — same affordance the user
        // expands into. Skipped at zero so quiet behaviors stay
        // visually quiet.
        if run_count_in_view > 0 {
            content_blocks.push(
                text(format!(
                    "{run_count_in_view} run{}",
                    if run_count_in_view == 1 { "" } else { "s" }
                ))
                .caption()
                .muted(),
            );
        }
        let content = item_content(content_blocks);

        // Chevron in the actions slot tracks expansion. Click on
        // the row routes via `behavior:{pod}:{id}`.
        let chevron_name = if expanded {
            "chevron-down"
        } else {
            "chevron-right"
        };
        let actions = item_actions([icon(chevron_name)
            .icon_size(tokens::ICON_SM)
            .text_color(tokens::MUTED_FOREGROUND)]);

        let key = behavior_row_key(pod_id, &b.behavior_id);
        item([content, actions]).key(key)
    }

    /// Inline action toolbar shown inside an expanded behavior body.
    /// Single row of four icon-buttons (lucide `zap` / `pause` or
    /// `play` / `square-pen` / `trash`) clustered side-by-side. With
    /// every action shrunk to a 32 px icon the whole strip is ~152 px
    /// wide — well under the sidebar's 224 px — so the two-row split
    /// the text-labelled version needed is gone.
    ///
    /// Run-now uses `zap` (lightning = "fire now") rather than `play`
    /// so it doesn't visually collide with the Resume `play` icon
    /// when the behavior is paused. Pause and Resume share the
    /// `behavior-toggle:` route key — same button, different icon
    /// based on `enabled`.
    ///
    /// Delete is two-click arm-confirm: idle is `trash.ghost()`,
    /// armed is the same `trash` icon under `.destructive()` (solid
    /// red fill, since `.ghost().destructive()` doesn't compose —
    /// `destructive()` writes back to `fill` regardless of ghost).
    /// The color flip is the arm signal: a quiet gray icon goes loud
    /// red on the user's first click, and on a normal-tempo
    /// double-click the red has time to render before the second
    /// click lands. Pre-handler at the top of `on_event` clears the
    /// arm if any other click happens first.
    ///
    /// Edit opens the [`BEHAVIOR_EDITOR_KEY`] sheet (right-attached
    /// `SheetSide::Right`); load-errored rows can still open it so the
    /// user can fix the broken `behavior.toml` from the form (or fall
    /// back to the eventual raw-TOML tab).
    fn behavior_actions_row(&self, pod_id: &str, b: &BehaviorSummary) -> El {
        let mut run = icon_button(crate::icons::ICON_ZAP.clone())
            .key(behavior_run_key(pod_id, &b.behavior_id))
            .ghost();
        if b.load_error.is_some() {
            run = run.disabled();
        }
        let toggle_icon = if b.enabled {
            crate::icons::ICON_PAUSE.clone()
        } else {
            crate::icons::ICON_PLAY.clone()
        };
        let toggle = icon_button(toggle_icon)
            .key(behavior_toggle_key(pod_id, &b.behavior_id))
            .ghost();
        let edit = icon_button(crate::icons::ICON_SQUARE_PEN.clone())
            .key(behavior_edit_key(pod_id, &b.behavior_id))
            .ghost();

        let armed = self
            .delete_armed_behavior
            .as_ref()
            .map(|(p, bb)| p == pod_id && bb == &b.behavior_id)
            .unwrap_or(false);
        let delete = if armed {
            icon_button(crate::icons::ICON_TRASH.clone())
                .key(behavior_delete_key(pod_id, &b.behavior_id))
                .destructive()
        } else {
            icon_button(crate::icons::ICON_TRASH.clone())
                .key(behavior_delete_key(pod_id, &b.behavior_id))
                .ghost()
        };

        // Indent matches the nested-thread depth=1 left-pad so the
        // toolbar visually belongs to the expanded body.
        let pad = Sides {
            left: tokens::SPACE_3 + tokens::SPACE_3,
            right: tokens::SPACE_3,
            top: 0.0,
            bottom: tokens::SPACE_1,
        };
        row([run, toggle, edit, delete])
            .gap(tokens::SPACE_2)
            .padding(pad)
            .width(Size::Fill(1.0))
    }

    /// Build the pod selector row. Single-active selection keyed
    /// `POD_TABS_KEY`; per-tab option keys auto-derived as
    /// `pod-tabs:tab:{pod_id}` by `tabs_list`.
    fn pod_tabs(&self) -> El {
        let pods = self.sorted_pods();
        // Tabs default to the empty string when nothing is active —
        // `tabs_list` reads the `current` value to mark the
        // selected trigger and wouldn't match any real pod_id.
        let current = self.pod_tab.clone().unwrap_or_default();
        let options: Vec<(String, String)> = pods
            .iter()
            .map(|p| (p.pod_id.clone(), pod_tab_label(p)))
            .collect();
        tabs_list(POD_TABS_KEY, &current, options)
    }

    /// Threads section for the active pod: header + paginated list
    /// of `item` rows. Interactive threads only (origin == None);
    /// behavior-spawned threads nest under their behavior in
    /// [`behaviors_section`]. Orphan-origin threads (their behavior
    /// is no longer in the registry — likely deleted while spawned
    /// threads survived) fall through here so they don't disappear
    /// silently, with a `via {behavior_id}` description marker.
    fn threads_section(&self, pod_id: &str) -> El {
        let known_behavior_ids: HashSet<&str> = self
            .behaviors_by_pod
            .get(pod_id)
            .map(|v| v.iter().map(|b| b.behavior_id.as_str()).collect())
            .unwrap_or_default();
        let mut threads: Vec<&ThreadSummary> = self
            .threads
            .values()
            .filter(|t| t.pod_id == pod_id)
            .filter(|t| match &t.origin {
                None => true,
                Some(origin) => !known_behavior_ids.contains(origin.behavior_id.as_str()),
            })
            .collect();
        // Newest activity floats to the top; ties broken by created_at
        // (stable across rebuilds since both fields are server-side
        // monotonic).
        threads.sort_by(|a, b| b.last_active.cmp(&a.last_active));

        let total = threads.len();
        let header_label = format!("Threads ({total})");
        // The "+" entry point lives in the section header so it's
        // the first thing the eye lands on when scanning the
        // sidebar. Sticking it in the right slot of the header row
        // keeps it adjacent to the label that scopes its meaning
        // ("new thread *here*"). Render it `ghost` so it visually
        // belongs to the chrome rather than competing with row
        // content.
        let header = row([
            sidebar_group_label(header_label).width(Size::Fill(1.0)),
            icon_button("plus")
                .key(SIDEBAR_NEW_THREAD_KEY)
                .ghost()
                .icon_size(tokens::ICON_XS),
        ])
        .align(Align::Center)
        .padding(Sides::xy(tokens::SPACE_2, tokens::SPACE_1))
        .width(Size::Fill(1.0));

        let mut group_children: Vec<El> = vec![header];

        if threads.is_empty() {
            group_children.push(self.sidebar_inline_empty(
                crate::icons::ICON_MESSAGE_SQUARE.clone(),
                "No threads in this pod",
                "Hit \u{2018}+\u{2019} to start one.",
            ));
            return sidebar_group(group_children);
        }

        let ordered = order_threads_by_dispatch(&threads);
        let expanded = self.expanded_pod_threads.contains(pod_id);
        let shown = if expanded {
            ordered.len()
        } else {
            ordered.len().min(SIDEBAR_THREAD_PREVIEW)
        };

        let rows: Vec<El> = ordered[..shown]
            .iter()
            // Threads list is the place where orphan-origin rows
            // surface — show the via-marker so their provenance is
            // visible.
            .map(|(t, depth)| self.thread_item_row(t, *depth, true))
            .collect();
        group_children.push(item_group(rows));

        let hidden = ordered.len().saturating_sub(shown);
        if hidden > 0 {
            group_children.push(
                button(format!("Show {hidden} more"))
                    .ghost()
                    .key(SIDEBAR_SHOWMORE_KEY)
                    .width(Size::Fill(1.0)),
            );
        } else if expanded && ordered.len() > SIDEBAR_THREAD_PREVIEW {
            group_children.push(
                button("Show less")
                    .ghost()
                    .key(SIDEBAR_SHOWMORE_KEY)
                    .width(Size::Fill(1.0)),
            );
        }

        sidebar_group(group_children)
    }

    /// One thread row in the sidebar. Uses aetna's `item` widget
    /// (the canonical "object/action list row" affordance), keyed
    /// by `thread:{id}` so the existing on_event routing stays
    /// intact. Title sits above a muted second line of
    /// `state · relative_time`. Dispatch / behavior children are
    /// left-padded by `depth * SPACE_3` so chains read top-down
    /// without needing inline marker glyphs on every row.
    ///
    /// `show_via_origin` controls the orphan-origin marker. The
    /// caller passes `true` only when the row is rendered
    /// standalone in the threads list because its behavior was
    /// deleted — then the suffix `· via {behavior}` keeps the
    /// provenance visible. Rows nested under their behavior in
    /// [`behaviors_section`] pass `false` since the parent row
    /// already supplies that context.
    fn thread_item_row(&self, t: &ThreadSummary, depth: usize, show_via_origin: bool) -> El {
        let key = format!("thread:{}", t.thread_id);
        let is_current = self.selected.as_deref() == Some(t.thread_id.as_str());

        let title = t.title.as_deref().unwrap_or(&t.thread_id).to_string();
        let mut secondary = format!(
            "{} · {}",
            state_label(t.state),
            format_relative(&t.last_active),
        );
        if show_via_origin && let Some(origin) = &t.origin {
            secondary.push_str(&format!(" · via {}", origin.behavior_id));
        }

        let content = item_content([item_title(title), item_description(secondary)]);
        let mut row_el = item([content]).key(key);
        if is_current {
            row_el = row_el.current();
        }
        if depth > 0 {
            // Item's default left padding is SPACE_3. Add an extra
            // SPACE_3 per dispatch level to surface parentage.
            row_el = row_el.pl(tokens::SPACE_3 + tokens::SPACE_3 * depth as f32);
        }
        row_el
    }

    /// Pods sorted for display: non-archived first, then by name
    /// (display) for stability. Returns slice borrows into
    /// [`Self::pods`] so the caller can read fields without
    /// cloning.
    fn sorted_pods(&self) -> Vec<&PodSummary> {
        let mut pods: Vec<&PodSummary> = self.pods.values().collect();
        pods.sort_by(|a, b| {
            a.archived
                .cmp(&b.archived)
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| a.pod_id.cmp(&b.pod_id))
        });
        pods
    }

    /// Pick a sensible default pod for the sidebar's active tab.
    /// Prefers the server-declared `default_pod_id`; falls back to
    /// the first non-archived pod by display name. Returns `None`
    /// only when there are no pods yet (sidebar shows "no pods").
    fn pick_default_pod_tab(&self) -> Option<String> {
        if let Some(id) = &self.default_pod_id
            && self.pods.contains_key(id)
        {
            return Some(id.clone());
        }
        self.sorted_pods()
            .into_iter()
            .find(|p| !p.archived)
            .map(|p| p.pod_id.clone())
    }

    fn content(&self, cx: &BuildCx) -> El {
        let inner = match self.selected.as_ref() {
            None => self.new_thread_pane(),
            Some(thread_id) => self.thread_pane(thread_id, cx),
        };
        // Connection alerts surface above the per-thread pane so
        // they're visible regardless of which thread (if any) is
        // selected. Only render when there's actually something to
        // say — Connecting and Connected are quiet states.
        let mut layers: Vec<El> = Vec::new();
        if let Some(banner) = self.connection_alert() {
            layers.push(banner);
        }
        layers.push(inner);
        column(layers)
            .gap(0.0)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
    }

    fn connection_alert(&self) -> Option<El> {
        let detail = self.conn_detail.as_deref();
        match self.conn_status {
            ConnectionStatus::Closed => Some(
                alert([
                    alert_title("connection closed"),
                    alert_description(detail.unwrap_or("the server closed the connection")),
                ])
                .warning(),
            ),
            ConnectionStatus::Error => Some(
                alert([
                    alert_title("connection error"),
                    alert_description(detail.unwrap_or("see logs for details")),
                ])
                .destructive(),
            ),
            ConnectionStatus::Connecting | ConnectionStatus::Connected => None,
        }
    }

    /// Build the prefill-progress row for the given thread, if a
    /// `ThreadPrefillProgress` event is currently active for it. The
    /// row is a thin `progress(...)` bar above a small muted token
    /// count; the egui sibling renders the same shape.
    ///
    /// Returns `None` when no prefill is active or the total is zero
    /// (defensive — a zero total would NaN out the fraction).
    /// Thread inspector panel — inline key/value grid rendered above
    /// the chat log when the user toggles the toolbar's info button.
    /// Mirrors the egui sibling's `render_thread_context_box`. Skips
    /// fields the user can already see in the header (title, state),
    /// surfaces the rest:
    /// - identity: thread_id, pod_id, created_at, last_active
    /// - resolved bindings: host_env, mcp_hosts
    /// - thread config: max_tokens, max_turns
    /// - cumulative usage: total in / out / cache
    /// - trigger origin: behavior_id (when spawned by a behavior)
    ///
    /// Scope rendering, OAuth-aware MCP host details, and the
    /// trigger payload pretty-print are deferred — they add
    /// substantial surface for a v1 inspector.
    fn render_inspector_panel(
        &self,
        thread_id: &str,
        summary: Option<&ThreadSummary>,
        view: Option<&ThreadView>,
    ) -> Option<El> {
        let view = view?;
        let mut rows: Vec<El> = Vec::new();
        let mut push = |label: &str, value: String| {
            rows.push(
                row([
                    text(label.to_string())
                        .caption()
                        .muted()
                        .width(Size::Fixed(140.0)),
                    text(value).caption().width(Size::Fill(1.0)),
                ])
                .gap(tokens::SPACE_3)
                .align(Align::Start)
                .width(Size::Fill(1.0)),
            );
        };

        push("thread_id", thread_id.to_string());
        if let Some(s) = summary {
            push("pod_id", s.pod_id.clone());
            if !s.last_active.is_empty() {
                push("last_active", s.last_active.clone());
            }
        }
        if !view.created_at.is_empty() {
            push("created_at", view.created_at.clone());
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
        push("backend", backend_val);
        push("model", model_val);
        if view.max_tokens > 0 {
            push("max_tokens", view.max_tokens.to_string());
        }
        if view.max_turns > 0 {
            push("max_turns", view.max_turns.to_string());
        }
        let host_env_val = if view.host_env_labels.is_empty() {
            "(none \u{2014} shared MCPs only)".to_string()
        } else {
            view.host_env_labels.join(", ")
        };
        push("host_env", host_env_val);
        let mcp_val = if view.mcp_hosts.is_empty() {
            "(none)".to_string()
        } else {
            view.mcp_hosts.join(", ")
        };
        push("mcp_hosts", mcp_val);
        push("total_in", view.total_usage.input_tokens.to_string());
        push("total_out", view.total_usage.output_tokens.to_string());
        if view.total_usage.cache_read_input_tokens > 0
            || view.total_usage.cache_creation_input_tokens > 0
        {
            push(
                "cache r/w",
                format!(
                    "{}/{}",
                    view.total_usage.cache_read_input_tokens,
                    view.total_usage.cache_creation_input_tokens,
                ),
            );
        }
        if let Some(behavior_id) = view.origin_behavior_id.as_deref() {
            push("origin behavior", behavior_id.to_string());
        }

        Some(
            column(rows)
                .gap(tokens::SPACE_1)
                .padding(Sides {
                    left: tokens::SPACE_2,
                    right: tokens::SPACE_2,
                    top: tokens::SPACE_2,
                    bottom: tokens::SPACE_2,
                })
                .width(Size::Fill(1.0))
                .fill(tokens::MUTED.with_alpha(40))
                .radius(tokens::RADIUS_SM),
        )
    }

    fn prefill_indicator(&self, thread_id: &str) -> Option<El> {
        let &(processed, total) = self.prefill.get(thread_id)?;
        if total == 0 {
            return None;
        }
        let frac = (processed as f32 / total as f32).clamp(0.0, 1.0);
        let bar = progress(frac, tokens::INFO)
            .width(Size::Fill(1.0))
            .height(Size::Fixed(4.0));
        let label = text(format!("prefilling {processed} / {total} tokens"))
            .caption()
            .muted();
        Some(
            column([bar, label])
                .gap(tokens::SPACE_1)
                .width(Size::Fill(1.0)),
        )
    }

    fn thread_pane(&self, thread_id: &str, cx: &BuildCx) -> El {
        let summary = self.threads.get(thread_id);
        let view = self.views.get(thread_id);

        let title = view
            .and_then(|v| v.title.as_deref())
            .or_else(|| summary.and_then(|s| s.title.as_deref()))
            .unwrap_or("untitled")
            .to_string();

        // `toolbar([title, spacer, badge])` is the canonical thread
        // header recipe — compact, center-aligned, single line. The
        // thread id rides as a muted second line so the toolbar stays
        // tight; an inspector affordance can pin it back into a popover
        // later when modals land.
        let mut toolbar_children: Vec<El> = vec![toolbar_title(title), spacer()];
        // Provenance chips between title and state badge: behavior
        // origin first (most load-bearing — tells the user this
        // thread fires on a schedule / webhook), then continuation
        // (forked from another thread), then dispatch parent
        // (spawned by `dispatch_thread` from another thread).
        // Muted styling so they read as metadata, not chrome.
        if let Some(s) = summary {
            if let Some(origin) = &s.origin {
                toolbar_children.push(badge(format!("via {}", origin.behavior_id)).muted());
            }
            if let Some(prev) = &s.continued_from {
                toolbar_children.push(badge(format!("forked from {}", short_id(prev))).muted());
            }
            if let Some(parent) = &s.dispatched_by {
                toolbar_children
                    .push(badge(format!("dispatched from {}", short_id(parent))).muted());
            }
            toolbar_children.push(state_badge(s.state));
            // Inspector toggle — small info icon on the trailing
            // edge of the toolbar. Always rendered (the inspector
            // panel itself is conditional on `inspector_open`).
            toolbar_children.push(
                icon_button("info")
                    .key(CHAT_INSPECTOR_TOGGLE_KEY)
                    .ghost()
                    .icon_size(tokens::ICON_XS),
            );
        }
        // Sub-line under the toolbar: thread id (left) and the
        // backend/model + cumulative-usage chrome (right). Both
        // pieces are caption-sized muted metadata that doesn't
        // compete with the title; right-aligning the model + usage
        // mirrors the egui sibling's status bar where this lived.
        let mut sub_line: Vec<El> = vec![text(thread_id).muted().xsmall().width(Size::Fill(1.0))];
        if let Some(v) = view {
            if let Some(model_chip) = thread_model_chip(&v.backend, &v.model) {
                sub_line.push(model_chip);
            }
            if let Some(usage_chip) = thread_usage_chip(&v.total_usage) {
                sub_line.push(usage_chip);
            }
        }
        let mut header_rows: Vec<El> = vec![
            toolbar(toolbar_children),
            row(sub_line)
                .gap(tokens::SPACE_3)
                .align(Align::Center)
                .width(Size::Fill(1.0)),
        ];
        // Inspector panel — rendered between the toolbar / sub-line
        // and the prefill indicator when expanded. Stays part of the
        // header column so it shares the bottom-stroke separator
        // with the rest of the chrome and scrolls *with* the
        // header (not the chat log) when the dialog gets tall.
        if self.inspector_open.as_deref() == Some(thread_id)
            && let Some(panel) = self.render_inspector_panel(thread_id, summary, view)
        {
            header_rows.push(panel);
        }
        // Prefill progress: a thin progress bar plus a muted token
        // count, rendered while the model is ingesting the prompt.
        // Cleared automatically once the first text/reasoning delta
        // arrives (see the `dispatch_wire` arms for those events).
        if let Some(prefill_row) = self.prefill_indicator(thread_id) {
            header_rows.push(prefill_row);
        }
        let header = column(header_rows)
            .gap(tokens::SPACE_1)
            .padding(tokens::SPACE_4)
            .width(Size::Fill(1.0))
            .stroke(tokens::BORDER);

        // Scroll key derived from the thread id so the offset
        // persists across rebuilds *for this thread*; switching
        // threads gets a fresh offset.
        let scroll_key = format!("chat-scroll:{thread_id}");

        let body: El = match view {
            None => empty_state(
                crate::icons::ICON_MESSAGE_SQUARE.clone(),
                "Loading thread",
                "Catching up on this conversation\u{2026}",
            ),
            Some(v) if v.items.is_empty() => empty_state(
                crate::icons::ICON_MESSAGE_SQUARE.clone(),
                "No messages yet",
                "Type below and hit Enter to start the conversation.",
            ),
            Some(v) => {
                // Thread the row index in as the accordion `value` so
                // each reasoning / tool row has an independent open
                // state stable across rebuilds.
                let rows: Vec<El> = v
                    .items
                    .iter()
                    .enumerate()
                    .map(|(idx, item)| self.event_log_row(idx, item, cx))
                    .collect();
                // Inter-row breathing room. Upstream's README example
                // has no explicit gap; with the row content sitting
                // flush, adjacent unfilled rows visually blur into one
                // text flow. SPACE_2 (8px) reads as a log-style line
                // gap rather than card-isolation.
                //
                // Right padding lives on the *content* column rather
                // than the scroll itself so the scrollbar thumb sits
                // in a reserved gutter outside the focusable rows
                // (lint flags otherwise — `ScrollbarObscuresFocusable`
                // — once the body overflows, which the inspector
                // panel can trigger by shrinking available height).
                let content = column(rows)
                    .gap(tokens::SPACE_2)
                    .padding(Sides {
                        left: tokens::SPACE_2,
                        right: tokens::SPACE_2 + tokens::SCROLLBAR_THUMB_WIDTH,
                        top: tokens::SPACE_2,
                        bottom: tokens::SPACE_2,
                    })
                    .width(Size::Fill(1.0))
                    .height(Size::Hug);
                scroll([content])
                    .key(scroll_key)
                    .width(Size::Fill(1.0))
                    .height(Size::Fill(1.0))
            }
        };

        // Destructive banner above the chat log when this thread
        // is in `Failed` state and the view carries a failure
        // detail. The conjunction matches egui — we don't surface
        // a stale failure message on a thread that's recovered
        // (its `state` will have flipped back) even if `failure`
        // hasn't been cleared yet by a fresh snapshot.
        let mut layers: Vec<El> = vec![header];
        if let Some(banner) = self.thread_failure_banner(thread_id) {
            layers.push(banner);
        }
        // Pending sudo banners ride above the chat log so the user
        // can't miss an approval request hidden under a long stream
        // of streaming output. Only this thread's banners surface —
        // other threads' pending sudos stay scoped to their own pane.
        if let Some(banners) = self.render_pending_sudo_banners(thread_id) {
            layers.push(banners);
        }
        layers.push(body);
        layers.push(self.compose_box());
        column(layers)
            .gap(0.0)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
    }

    /// Return a destructive `alert` describing the thread's
    /// recorded failure, when the thread is currently in
    /// `Failed` state and the view has a failure detail. `None`
    /// when either guard fails.
    fn thread_failure_banner(&self, thread_id: &str) -> Option<El> {
        let summary = self.threads.get(thread_id)?;
        if summary.state != whisper_agent_protocol::ThreadStateLabel::Failed {
            return None;
        }
        let view = self.views.get(thread_id)?;
        let detail = view.failure.as_deref()?;
        Some(
            alert([
                alert_title("thread failed"),
                alert_description(detail.to_string()),
            ])
            .destructive(),
        )
    }

    /// New-thread compose form. Rendered in the content pane when no
    /// thread is selected — the equivalent of the egui sibling's
    /// `composing_new` mode. Three `select_trigger` pickers
    /// (backend / model / pod) sit above a `text_area` and a "Start"
    /// button. Open menus are emitted as overlay layers from
    /// [`popover_layers`] so they paint above the rest of the UI.
    fn new_thread_pane(&self) -> El {
        let backend_trigger = select_trigger(PICKER_BACKEND, self.backend_label());
        let model_trigger = select_trigger(PICKER_MODEL, self.model_label());
        let pod_trigger = select_trigger(PICKER_POD, self.pod_label_for_picker());

        let buf = self.active_compose_text();
        let editor = text_area(buf, &self.selection, COMPOSE_KEY).height(Size::Fixed(140.0));

        let can_send = !buf.trim().is_empty();
        let mut send = button("Start").key(SEND_KEY).primary();
        if !can_send {
            send = send.disabled();
        }

        // Right-aligned action row inside `card_footer`.
        let footer = row([spacer(), send])
            .width(Size::Fill(1.0))
            .align(Align::Center);

        let body = card([
            card_header([
                card_title("Start a new conversation"),
                card_description(
                    "Pick a backend and pod, then type a message to begin. \
                     Leaving a picker on its inherit row falls through to the \
                     pod's defaults.",
                ),
            ]),
            card_content([form([
                form_item([
                    form_label("Backend"),
                    form_control(backend_trigger),
                    form_description(self.backend_hint()),
                ]),
                form_item([
                    form_label("Model"),
                    form_control(model_trigger),
                    form_description(self.model_hint()),
                ]),
                form_item([form_label("Pod"), form_control(pod_trigger)]),
                form_item([form_label("Message"), form_control(editor)]),
            ])]),
            card_footer([footer]),
        ])
        .width(Size::Fixed(640.0));

        // Center the card in the pane; padding keeps it off the
        // pane edges on small windows.
        column([body])
            .padding(tokens::SPACE_6)
            .align(Align::Center)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
    }

    fn backend_label(&self) -> String {
        self.picker_backend
            .clone()
            .unwrap_or_else(|| "inherit pod default".to_string())
    }

    fn backend_hint(&self) -> String {
        match self.backends.len() {
            0 => "loading backends…".to_string(),
            n => format!("{n} backend{} configured", if n == 1 { "" } else { "s" }),
        }
    }

    fn model_label(&self) -> String {
        if let Some(m) = self.picker_model.as_deref() {
            return m.to_string();
        }
        // Mirror `build_creation_request` so the trigger reflects
        // exactly what the server would resolve if the user hit
        // Start now — no surprises after submit.
        if let Some(b) = self.picker_backend.as_ref() {
            if let Some(default) = self
                .backends
                .iter()
                .find(|bs| &bs.name == b)
                .and_then(|bs| bs.default_model.clone())
            {
                return format!("{default}  (backend default)");
            }
            if let Some(first) = self
                .models_by_backend
                .get(b)
                .and_then(|list| list.first())
                .map(|m| m.id.clone())
            {
                return format!("{first}  (first available)");
            }
        }
        "auto (backend default)".to_string()
    }

    fn model_hint(&self) -> String {
        let Some(b) = self.picker_backend.as_ref() else {
            return "pick a backend first to load its models".to_string();
        };
        match self.models_by_backend.get(b) {
            None => "loading models…".to_string(),
            Some(list) => format!(
                "{} model{} available",
                list.len(),
                if list.len() == 1 { "" } else { "s" }
            ),
        }
    }

    fn pod_label_for_picker(&self) -> String {
        match self.picker_pod.as_deref() {
            None => "default pod".to_string(),
            Some(id) => self
                .pods
                .get(id)
                .map(|p| p.name.clone())
                .unwrap_or_else(|| id.to_string()),
        }
    }

    /// Build the popover layer list for [`overlays`]. Exactly one
    /// menu can be open at a time (enforced by `close_other_pickers`),
    /// but the iterator shape lets us extend this to e.g. a settings
    /// modal or fork dialog without reshaping `build`.
    fn popover_layers(&self) -> Vec<Option<El>> {
        let mut out: Vec<Option<El>> = Vec::new();
        if self.picker_backend_open {
            out.push(Some(self.backend_menu()));
        }
        if self.picker_model_open {
            out.push(Some(self.model_menu()));
        }
        if self.picker_pod_open {
            out.push(Some(self.pod_menu()));
        }
        // Modals layer last so they paint above the picker
        // popovers — though in practice the two are mutually
        // exclusive on screen, the layering choice is harmless.
        if let Some(modal_el) = self.render_new_pod_modal() {
            out.push(Some(modal_el));
        }
        if let Some(modal_el) = self.render_new_behavior_modal() {
            out.push(Some(modal_el));
        }
        if let Some(modal_el) = self.render_behavior_editor_modal() {
            out.push(Some(modal_el));
            // Trigger-kind menu rides above the dialog itself —
            // `select_menu` paints as its own layer; ordering it last
            // makes it the topmost so it floats above the dialog panel.
            if let Some(editor) = self.behavior_editor.as_ref()
                && editor.trigger_kind_open
            {
                out.push(Some(self.behavior_editor_trigger_kind_menu()));
            }
            // Overlap / catch_up pickers — single-active per
            // `editor.open_picker`. Same topmost-layer discipline as
            // the trigger-kind menu above.
            if let Some(editor) = self.behavior_editor.as_ref()
                && let Some(open) = editor.open_picker
            {
                out.push(Some(self.behavior_editor_picker_menu(open)));
            }
        }
        if let Some(modal_el) = self.render_pod_editor_modal() {
            out.push(Some(modal_el));
            // Pod editor pickers (caps + Defaults backend/model/tool
            // gate) ride above the dialog — same single-active
            // discipline as the behavior editor's trigger-kind
            // picker. Topmost so it floats above the dialog panel.
            if let Some(editor) = self.pod_editor.as_ref()
                && let Some(open) = editor.open_picker
            {
                out.push(Some(self.pod_editor_picker_menu(open)));
            }
        }
        if let Some(modal_el) = self.render_fork_modal() {
            out.push(Some(modal_el));
        }
        if let Some(modal_el) = self.render_settings_modal() {
            out.push(Some(modal_el));
        }
        if let Some(modal_el) = self.render_buckets_modal() {
            out.push(Some(modal_el));
        }
        if let Some(modal_el) = self.render_file_tree_modal() {
            out.push(Some(modal_el));
        }
        if let Some(modal_el) = self.render_file_viewer_modal() {
            out.push(Some(modal_el));
        }
        if let Some(modal_el) = self.render_json_viewer_modal() {
            out.push(Some(modal_el));
        }
        // Lightbox last so it paints on top of any other layer —
        // clicking an image in the chat log is a deliberate "show me
        // this big," and a stray modal underneath shouldn't clip it.
        if let Some(modal_el) = self.render_lightbox_modal() {
            out.push(Some(modal_el));
        }
        out
    }

    /// Routing for the "+ New pod" modal. Returns `true` when the
    /// event was consumed so the outer `on_event` can `return` and
    /// avoid double-handling. All routes live under the
    /// [`NEW_POD_MODAL_KEY`] prefix — bail early if the modal is
    /// closed (its routed events are nonsensical without state).
    fn handle_new_pod_modal_event(&mut self, event: &UiEvent) -> bool {
        if self.new_pod_modal.is_none() {
            return false;
        }
        let pod_id_key = format!("{NEW_POD_MODAL_KEY}:pod-id");
        let name_key = format!("{NEW_POD_MODAL_KEY}:name");
        let create_key = format!("{NEW_POD_MODAL_KEY}:create");
        let cancel_key = format!("{NEW_POD_MODAL_KEY}:cancel");
        let dismiss_key = format!("{NEW_POD_MODAL_KEY}:dismiss");

        // Text inputs go through `text_input::apply_event` so cursor
        // updates / IME / paste behavior matches every other text
        // input in the app.
        if event.target_key() == Some(pod_id_key.as_str()) {
            if let Some(modal) = self.new_pod_modal.as_mut() {
                text_input::apply_event(&mut modal.pod_id, &mut self.selection, &pod_id_key, event);
                // Edits clear stale errors so the form doesn't keep
                // showing a previous validation message after the
                // user starts fixing it.
                modal.error = None;
            }
            return true;
        }
        if event.target_key() == Some(name_key.as_str()) {
            if let Some(modal) = self.new_pod_modal.as_mut() {
                text_input::apply_event(&mut modal.name, &mut self.selection, &name_key, event);
                modal.error = None;
            }
            return true;
        }

        // Cancel + scrim-dismiss: drop the modal entirely. State is
        // ephemeral — the next open starts fresh.
        if event.is_click_or_activate(&cancel_key) || event.is_click_or_activate(&dismiss_key) {
            self.new_pod_modal = None;
            return true;
        }

        // Create: client-validate, then send `CreatePod` with a
        // correlation id so we can match the server's reply. We do
        // not optimistically close — the modal stays up with a
        // disabled button until `PodCreated` (close) or `Error`
        // (re-enable + error message) lands.
        if event.is_click_or_activate(&create_key) {
            self.submit_new_pod_modal();
            return true;
        }

        false
    }

    /// Validate + dispatch the "+ New pod" form. Bails early on
    /// client-side failures (empty fields, illegal id chars,
    /// duplicate id, no backends) by setting `modal.error`. On
    /// success, allocates a correlation id, stamps it onto the
    /// modal, and sends `CreatePod` — keeping the modal open and
    /// disabled until the wire round-trip finishes.
    fn submit_new_pod_modal(&mut self) {
        let Some(modal) = self.new_pod_modal.as_ref() else {
            return;
        };
        // Already in flight — guard against a stray double-click
        // racing past the `disabled()` styling.
        if modal.pending_correlation.is_some() {
            return;
        }
        let pod_id = modal.pod_id.trim().to_string();
        let name = modal.name.trim().to_string();
        if let Err(msg) = validate_pod_id_client(&pod_id) {
            self.set_new_pod_error(msg.to_string());
            return;
        }
        if self.pods.contains_key(&pod_id) {
            self.set_new_pod_error(format!("pod `{pod_id}` already exists"));
            return;
        }
        if self.backends.is_empty() {
            self.set_new_pod_error("no backends configured on the server".into());
            return;
        }

        let backend_names: Vec<String> = self.backends.iter().map(|b| b.name.clone()).collect();
        let config = fresh_pod_config(name, backend_names);
        let correlation_id = self.next_correlation_id();
        if let Some(modal) = self.new_pod_modal.as_mut() {
            modal.pending_correlation = Some(correlation_id.clone());
            modal.error = None;
        }
        self.send(ClientToServer::CreatePod {
            correlation_id: Some(correlation_id),
            pod_id,
            config,
        });
    }

    /// Helper: surface a client-side validation message without
    /// touching `pending_correlation` (these failures don't go to
    /// the wire so there's no in-flight request to track).
    fn set_new_pod_error(&mut self, msg: String) {
        if let Some(modal) = self.new_pod_modal.as_mut() {
            modal.error = Some(msg);
        }
    }

    /// Routing for the "+ New behavior" modal. Mirrors
    /// [`Self::handle_new_pod_modal_event`] — bail when closed,
    /// route text inputs, route cancel / dismiss, route create.
    fn handle_new_behavior_modal_event(&mut self, event: &UiEvent) -> bool {
        if self.new_behavior_modal.is_none() {
            return false;
        }
        let behavior_id_key = format!("{NEW_BEHAVIOR_MODAL_KEY}:behavior-id");
        let name_key = format!("{NEW_BEHAVIOR_MODAL_KEY}:name");
        let create_key = format!("{NEW_BEHAVIOR_MODAL_KEY}:create");
        let cancel_key = format!("{NEW_BEHAVIOR_MODAL_KEY}:cancel");
        let dismiss_key = format!("{NEW_BEHAVIOR_MODAL_KEY}:dismiss");

        if event.target_key() == Some(behavior_id_key.as_str()) {
            if let Some(modal) = self.new_behavior_modal.as_mut() {
                text_input::apply_event(
                    &mut modal.behavior_id,
                    &mut self.selection,
                    &behavior_id_key,
                    event,
                );
                modal.error = None;
            }
            return true;
        }
        if event.target_key() == Some(name_key.as_str()) {
            if let Some(modal) = self.new_behavior_modal.as_mut() {
                text_input::apply_event(&mut modal.name, &mut self.selection, &name_key, event);
                modal.error = None;
            }
            return true;
        }

        if event.is_click_or_activate(&cancel_key) || event.is_click_or_activate(&dismiss_key) {
            self.new_behavior_modal = None;
            return true;
        }

        if event.is_click_or_activate(&create_key) {
            self.submit_new_behavior_modal();
            return true;
        }

        false
    }

    /// Validate + dispatch the "+ New behavior" form. Same control
    /// flow as [`Self::submit_new_pod_modal`] — client-side checks
    /// first, then mint a correlation id and fire `CreateBehavior`
    /// while keeping the modal open and disabled. The wire
    /// round-trip closes the modal (`BehaviorCreated`) or surfaces
    /// the rejection message (`Error`).
    fn submit_new_behavior_modal(&mut self) {
        let Some(modal) = self.new_behavior_modal.as_ref() else {
            return;
        };
        if modal.pending_correlation.is_some() {
            return;
        }
        let pod_id = modal.pod_id.clone();
        let behavior_id = modal.behavior_id.trim().to_string();
        let name = modal.name.trim().to_string();
        if let Err(msg) = validate_behavior_id_client(&behavior_id) {
            self.set_new_behavior_error(msg.to_string());
            return;
        }
        // Defensive: the modal could have been opened against a
        // pod that was since deleted (rare but cheap to check).
        if !self.pods.contains_key(&pod_id) {
            self.set_new_behavior_error(format!("pod `{pod_id}` no longer exists"));
            return;
        }
        let exists = self
            .behaviors_by_pod
            .get(&pod_id)
            .map(|list| list.iter().any(|b| b.behavior_id == behavior_id))
            .unwrap_or(false);
        if exists {
            self.set_new_behavior_error(format!("behavior `{behavior_id}` already exists"));
            return;
        }
        if name.is_empty() {
            self.set_new_behavior_error("name is empty".into());
            return;
        }

        // Manual-trigger stub with empty prompt — the user fills
        // the rest in the (eventual) editor on the
        // `BehaviorCreated` round-trip. Mirrors the egui sibling.
        let config = BehaviorConfig {
            name,
            description: None,
            trigger: TriggerSpec::Manual,
            thread: BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
            scope: Default::default(),
        };
        let correlation_id = self.next_correlation_id();
        if let Some(modal) = self.new_behavior_modal.as_mut() {
            modal.pending_correlation = Some(correlation_id.clone());
            modal.error = None;
        }
        self.send(ClientToServer::CreateBehavior {
            correlation_id: Some(correlation_id),
            pod_id,
            behavior_id,
            config,
            prompt: String::new(),
            system_prompt: None,
        });
    }

    fn set_new_behavior_error(&mut self, msg: String) {
        if let Some(modal) = self.new_behavior_modal.as_mut() {
            modal.error = Some(msg);
        }
    }

    /// Open (or re-target) the behavior editor sheet for the given
    /// `(pod, behavior)` pair. No-op if the sheet is already open for
    /// the same pair (avoids re-firing `GetBehavior` and trampling the
    /// user's pending edits). Mints a correlation id and stashes it
    /// in `pending_get` so the matching `BehaviorSnapshot` reply
    /// hydrates the form.
    fn open_behavior_editor(&mut self, pod_id: String, behavior_id: String) {
        if let Some(existing) = self.behavior_editor.as_ref()
            && existing.pod_id == pod_id
            && existing.behavior_id == behavior_id
        {
            return;
        }
        let behavior_correlation = self.next_correlation_id();
        let pod_correlation = self.next_correlation_id();
        self.behavior_editor = Some(BehaviorEditorSheetState::new(
            pod_id.clone(),
            behavior_id.clone(),
            behavior_correlation.clone(),
            pod_correlation.clone(),
        ));
        self.send(ClientToServer::GetBehavior {
            correlation_id: Some(behavior_correlation),
            pod_id: pod_id.clone(),
            behavior_id,
        });
        // Parallel `GetPod` so the Thread tab's host_env / mcp_hosts
        // override rows know what allow lists the pod declares. Both
        // round-trips race; whichever lands first hydrates its slice
        // of editor state and the other lands when ready.
        self.send(ClientToServer::GetPod {
            correlation_id: Some(pod_correlation),
            pod_id,
        });
    }

    /// Routing for the per-behavior editor sheet. Same shape as
    /// [`Self::handle_new_behavior_modal_event`] — bail when closed,
    /// route text inputs, route the trigger-kind picker, route
    /// primary / cancel / dismiss. Returns `true` when the event was
    /// consumed so the outer `on_event` can short-circuit. Per-field
    /// edits clear the `error` slot so a stale validation message
    /// doesn't linger after the user starts fixing it.
    fn handle_behavior_editor_event(&mut self, event: &UiEvent) -> bool {
        if self.behavior_editor.is_none() {
            return false;
        }

        // Tabs strip — reuses aetna's `widgets::tabs` event apply
        // through a String → enum bridge. Identical to the pod
        // editor's tab-switch wiring.
        let mut next_tab: Option<String> = None;
        let _ = aetna_core::widgets::tabs::apply_event(
            &mut next_tab,
            event,
            BEHAVIOR_EDITOR_TABS_KEY,
            |raw| Some(Some(raw.to_string())),
        );
        if let Some(value) = next_tab
            && let Some(target) = BehaviorEditorTab::from_wire(&value)
            && let Some(editor) = self.behavior_editor.as_mut()
        {
            // `switch_tab` runs the raw↔structured sync; a Raw → struct
            // hop with an unparseable buffer keeps the user on Raw and
            // surfaces the parse error in `editor.error`.
            editor.switch_tab(target);
            return true;
        }

        // Trigger-kind picker: classified through `widgets::select` so
        // toggle / dismiss / pick map to the same shape the new-thread
        // pickers use. Routed first because its sub-keys
        // (`...:trigger-kind:option:{value}`) overlap with a
        // `target_key`-based prefix check.
        if let Some(action) = classify_select_event(event, BEHAVIOR_EDITOR_TRIGGER_KIND_KEY) {
            self.handle_behavior_editor_trigger_kind_pick(action);
            return true;
        }

        // Overlap / catch_up pickers — Cron and Webhook share Overlap.
        for which in [
            BehaviorEditorPicker::Overlap,
            BehaviorEditorPicker::CatchUp,
            BehaviorEditorPicker::ThreadModel,
            BehaviorEditorPicker::ThreadBackend,
            BehaviorEditorPicker::RetentionKind,
            BehaviorEditorPicker::ScopeToolsDefault,
            BehaviorEditorPicker::ScopeCapsPodModify,
            BehaviorEditorPicker::ScopeCapsDispatch,
            BehaviorEditorPicker::ScopeCapsBehaviors,
        ] {
            if let Some(action) = classify_select_event(event, which.key()) {
                self.handle_behavior_editor_picker(which, action);
                return true;
            }
        }

        if event.target_key() == Some(BEHAVIOR_EDITOR_NAME_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                text_input::apply_event(
                    &mut cfg.name,
                    &mut self.selection,
                    BEHAVIOR_EDITOR_NAME_KEY,
                    event,
                );
                editor.error = None;
            }
            return true;
        }
        if event.target_key() == Some(BEHAVIOR_EDITOR_DESCRIPTION_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                // `description` is `Option<String>` on the wire — keep
                // a String buffer for the multi-line text_area and
                // project back to `None` when the user empties it
                // (matches the serde `skip_serializing_if =
                // "Option::is_none"` round-trip on disk).
                let mut buf = cfg.description.clone().unwrap_or_default();
                text_area::apply_event(
                    &mut buf,
                    &mut self.selection,
                    BEHAVIOR_EDITOR_DESCRIPTION_KEY,
                    event,
                );
                cfg.description = if buf.trim().is_empty() {
                    None
                } else {
                    Some(buf)
                };
                editor.error = None;
            }
            return true;
        }
        if event.target_key() == Some(BEHAVIOR_EDITOR_SCHEDULE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut() {
                text_input::apply_event(
                    &mut editor.schedule_buffer,
                    &mut self.selection,
                    BEHAVIOR_EDITOR_SCHEDULE_KEY,
                    event,
                );
                editor.error = None;
            }
            return true;
        }
        if event.target_key() == Some(BEHAVIOR_EDITOR_TIMEZONE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut() {
                text_input::apply_event(
                    &mut editor.timezone_buffer,
                    &mut self.selection,
                    BEHAVIOR_EDITOR_TIMEZONE_KEY,
                    event,
                );
                editor.error = None;
            }
            return true;
        }
        if event.target_key() == Some(BEHAVIOR_EDITOR_PROMPT_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut() {
                text_area::apply_event(
                    &mut editor.working_prompt,
                    &mut self.selection,
                    BEHAVIOR_EDITOR_PROMPT_KEY,
                    event,
                );
                editor.error = None;
            }
            return true;
        }

        // Thread-tab override-checkbox toggles. Each click flips the
        // matching `Option<...>` field on `cfg.thread`. Defaults
        // when transitioning to `Some` mirror the egui sibling's
        // pre-fill values (16384 tokens, 30 turns).
        if event.is_click_or_activate(BEHAVIOR_EDITOR_THREAD_MODEL_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.thread.model = if cfg.thread.model.is_some() {
                    None
                } else {
                    Some(String::new())
                };
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_THREAD_MAX_TOKENS_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                if cfg.thread.max_tokens.is_some() {
                    cfg.thread.max_tokens = None;
                    editor.thread_max_tokens_buf.clear();
                } else {
                    cfg.thread.max_tokens = Some(16384);
                    editor.thread_max_tokens_buf = "16384".to_string();
                }
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_THREAD_MAX_TURNS_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                if cfg.thread.max_turns.is_some() {
                    cfg.thread.max_turns = None;
                    editor.thread_max_turns_buf.clear();
                } else {
                    cfg.thread.max_turns = Some(30);
                    editor.thread_max_turns_buf = "30".to_string();
                }
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_THREAD_BACKEND_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.thread.bindings.backend = if cfg.thread.bindings.backend.is_some() {
                    None
                } else {
                    Some(String::new())
                };
                editor.error = None;
            }
            return true;
        }

        // host_env / mcp_hosts override checkboxes. Toggling on
        // seeds an empty `Some(vec![])` (the multi-check column
        // populates entries as the user clicks); toggling off
        // drops to `None` (inherit pod default).
        if event.is_click_or_activate(BEHAVIOR_EDITOR_THREAD_HOST_ENV_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.thread.bindings.host_env = if cfg.thread.bindings.host_env.is_some() {
                    None
                } else {
                    Some(Vec::new())
                };
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_THREAD_MCP_HOSTS_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.thread.bindings.mcp_hosts = if cfg.thread.bindings.mcp_hosts.is_some() {
                    None
                } else {
                    Some(Vec::new())
                };
                editor.error = None;
            }
            return true;
        }

        // host_env / mcp_hosts multi-check item clicks. Routes land
        // on `{group}:item:{name}`; the helper toggles membership
        // in the override `Vec<String>`. Only meaningful when the
        // override is on (`Some(vec)`); when `None`, the rendering
        // skipped the multi-check entirely so no events fire.
        if let Some(editor) = self.behavior_editor.as_mut()
            && let Some(cfg) = editor.working_config.as_mut()
            && let Some(vec) = cfg.thread.bindings.host_env.as_mut()
            && apply_checkbox_list_to_vec(vec, event, BEHAVIOR_EDITOR_THREAD_HOST_ENV_KEY)
        {
            editor.error = None;
            return true;
        }
        if let Some(editor) = self.behavior_editor.as_mut()
            && let Some(cfg) = editor.working_config.as_mut()
            && let Some(vec) = cfg.thread.bindings.mcp_hosts.as_mut()
            && apply_checkbox_list_to_vec(vec, event, BEHAVIOR_EDITOR_THREAD_MCP_HOSTS_KEY)
        {
            editor.error = None;
            return true;
        }

        // Scope-tab override checkboxes. Resource sets (backends /
        // host_envs / mcp_hosts) toggle Some(Vec::new()) ↔ None;
        // tools toggles `AllowMap::allow_all()` ↔ None; caps toggle
        // a sensible default ↔ None; tool_surface toggles a default
        // ToolSurface ↔ None.
        if event.is_click_or_activate(BEHAVIOR_EDITOR_SCOPE_BACKENDS_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.scope.backends = if cfg.scope.backends.is_some() {
                    None
                } else {
                    Some(Vec::new())
                };
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_SCOPE_HOST_ENVS_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.scope.host_envs = if cfg.scope.host_envs.is_some() {
                    None
                } else {
                    Some(Vec::new())
                };
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_SCOPE_MCP_HOSTS_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.scope.mcp_hosts = if cfg.scope.mcp_hosts.is_some() {
                    None
                } else {
                    Some(Vec::new())
                };
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_SCOPE_TOOLS_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.scope.tools = if cfg.scope.tools.is_some() {
                    None
                } else {
                    // `allow_all` is the most-permissive identity —
                    // narrowing to `pod.allow.tools.narrow(allow_all)`
                    // leaves the pod ceiling unchanged. Per-tool
                    // overrides ride through Raw TOML for v1.
                    Some(AllowMap::allow_all())
                };
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_SCOPE_CAPS_POD_MODIFY_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.scope.caps.pod_modify = if cfg.scope.caps.pod_modify.is_some() {
                    None
                } else {
                    // ModifyAllow is the most-permissive cap; narrowing
                    // against the pod ceiling collapses to whatever the
                    // pod allows. Same identity-under-narrow trick.
                    Some(whisper_agent_protocol::PodModifyCap::ModifyAllow)
                };
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_SCOPE_CAPS_DISPATCH_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.scope.caps.dispatch = if cfg.scope.caps.dispatch.is_some() {
                    None
                } else {
                    Some(whisper_agent_protocol::DispatchCap::WithinScope)
                };
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_SCOPE_CAPS_BEHAVIORS_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.scope.caps.behaviors = if cfg.scope.caps.behaviors.is_some() {
                    None
                } else {
                    Some(whisper_agent_protocol::BehaviorOpsCap::AuthorAny)
                };
                editor.error = None;
            }
            return true;
        }
        if event.is_click_or_activate(BEHAVIOR_EDITOR_SCOPE_TOOL_SURFACE_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                cfg.scope.tool_surface = if cfg.scope.tool_surface.is_some() {
                    None
                } else {
                    Some(whisper_agent_protocol::ToolSurface::default())
                };
                editor.error = None;
            }
            return true;
        }

        // Scope-tab resource-set multi-check item clicks. Same shape
        // as the Thread-tab bindings handlers — only meaningful when
        // the override is on (`Some(vec)`).
        if let Some(editor) = self.behavior_editor.as_mut()
            && let Some(cfg) = editor.working_config.as_mut()
            && let Some(vec) = cfg.scope.backends.as_mut()
            && apply_checkbox_list_to_vec(vec, event, BEHAVIOR_EDITOR_SCOPE_BACKENDS_KEY)
        {
            editor.error = None;
            return true;
        }
        if let Some(editor) = self.behavior_editor.as_mut()
            && let Some(cfg) = editor.working_config.as_mut()
            && let Some(vec) = cfg.scope.host_envs.as_mut()
            && apply_checkbox_list_to_vec(vec, event, BEHAVIOR_EDITOR_SCOPE_HOST_ENVS_KEY)
        {
            editor.error = None;
            return true;
        }
        if let Some(editor) = self.behavior_editor.as_mut()
            && let Some(cfg) = editor.working_config.as_mut()
            && let Some(vec) = cfg.scope.mcp_hosts.as_mut()
            && apply_checkbox_list_to_vec(vec, event, BEHAVIOR_EDITOR_SCOPE_MCP_HOSTS_KEY)
        {
            editor.error = None;
            return true;
        }

        // SystemPrompt-tab override checkbox. Toggling on binds the
        // config to the conventional pod-relative path and seeds an
        // empty buffer if none exists; toggling off drops the config
        // reference but leaves the buffer alone (so a re-toggle
        // doesn't lose the user's draft).
        if event.is_click_or_activate(BEHAVIOR_EDITOR_SYSTEM_PROMPT_OVERRIDE_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                if cfg.thread.system_prompt.is_some() {
                    cfg.thread.system_prompt = None;
                } else {
                    let conv_path = behavior_system_prompt_path(&editor.behavior_id);
                    cfg.thread.system_prompt = Some(SystemPromptChoice::File { name: conv_path });
                    if editor.working_system_prompt.is_none() {
                        editor.working_system_prompt = Some(String::new());
                    }
                }
                editor.error = None;
            }
            return true;
        }

        // SystemPrompt-tab body text_area. Routes to either
        // `working_system_prompt` (File variant — saved to the side
        // file) or `cfg.thread.system_prompt.Text.text` (inline).
        if event.target_key() == Some(BEHAVIOR_EDITOR_SYSTEM_PROMPT_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                match cfg.thread.system_prompt.as_mut() {
                    Some(SystemPromptChoice::File { .. }) => {
                        let buf = editor.working_system_prompt.get_or_insert_with(String::new);
                        text_area::apply_event(
                            buf,
                            &mut self.selection,
                            BEHAVIOR_EDITOR_SYSTEM_PROMPT_KEY,
                            event,
                        );
                    }
                    Some(SystemPromptChoice::Text { text }) => {
                        text_area::apply_event(
                            text,
                            &mut self.selection,
                            BEHAVIOR_EDITOR_SYSTEM_PROMPT_KEY,
                            event,
                        );
                    }
                    None => {}
                }
                editor.error = None;
            }
            return true;
        }

        // Raw-tab text_area. Each edit flips `raw_dirty` so the
        // tab-switch reparse / Save round-trip pick up the change;
        // see `BehaviorEditorSheetState::switch_tab` for the
        // `working_toml` ↔ `working_config` contract.
        if event.target_key() == Some(BEHAVIOR_EDITOR_RAW_TOML_KEY) {
            if let Some(editor) = self.behavior_editor.as_mut() {
                let before = editor.working_toml.clone();
                text_area::apply_event(
                    &mut editor.working_toml,
                    &mut self.selection,
                    BEHAVIOR_EDITOR_RAW_TOML_KEY,
                    event,
                );
                if editor.working_toml != before {
                    editor.raw_dirty = true;
                }
                editor.error = None;
            }
            return true;
        }

        // Retention-tab `days` numeric input. Only meaningful when
        // `cfg.on_completion` is a timed variant; we still parse and
        // write back when the kind happens to be Keep so the buffer
        // stays well-defined across kind switches.
        if let Some(editor) = self.behavior_editor.as_mut() {
            let opts = NumericInputOpts::default().min(1.0).max(3650.0).step(1.0);
            if numeric_input::apply_event(
                &mut editor.retention_days_buf,
                &mut self.selection,
                BEHAVIOR_EDITOR_RETENTION_DAYS_KEY,
                &opts,
                event,
            ) {
                if let Ok(v) = editor.retention_days_buf.parse::<u32>()
                    && let Some(cfg) = editor.working_config.as_mut()
                {
                    let clamped = v.clamp(1, 3650);
                    match &mut cfg.on_completion {
                        RetentionPolicy::Keep => {}
                        RetentionPolicy::ArchiveAfterDays { days }
                        | RetentionPolicy::DeleteAfterDays { days } => {
                            *days = clamped;
                        }
                    }
                }
                editor.error = None;
                return true;
            }
        }

        // Thread-tab numeric inputs (max_tokens / max_turns). Same
        // buffer-then-parse-back pattern as the pod editor's
        // Defaults numeric inputs.
        if let Some(editor) = self.behavior_editor.as_mut() {
            let max_tokens_opts = NumericInputOpts::default()
                .min(1.0)
                .max(200_000.0)
                .step(50.0);
            if numeric_input::apply_event(
                &mut editor.thread_max_tokens_buf,
                &mut self.selection,
                BEHAVIOR_EDITOR_THREAD_MAX_TOKENS_KEY,
                &max_tokens_opts,
                event,
            ) {
                if let Ok(v) = editor.thread_max_tokens_buf.parse::<u32>()
                    && let Some(cfg) = editor.working_config.as_mut()
                {
                    cfg.thread.max_tokens = Some(v.clamp(1, 200_000));
                }
                editor.error = None;
                return true;
            }
            let max_turns_opts = NumericInputOpts::default().min(1.0).max(10_000.0).step(1.0);
            if numeric_input::apply_event(
                &mut editor.thread_max_turns_buf,
                &mut self.selection,
                BEHAVIOR_EDITOR_THREAD_MAX_TURNS_KEY,
                &max_turns_opts,
                event,
            ) {
                if let Ok(v) = editor.thread_max_turns_buf.parse::<u32>()
                    && let Some(cfg) = editor.working_config.as_mut()
                {
                    cfg.thread.max_turns = Some(v.clamp(1, 10_000));
                }
                editor.error = None;
                return true;
            }
        }

        // Cron / timezone preset chips. The route's suffix is the
        // preset's array index; we look the expression up directly so
        // routed keys never carry `* / -` characters.
        if matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Some(route) = event.route()
        {
            if let Some(idx_str) = route.strip_prefix(BEHAVIOR_EDITOR_CRON_PRESET_PREFIX)
                && let Ok(idx) = idx_str.parse::<usize>()
                && let Some((_, expr)) = cron_preview::CRON_PRESETS.get(idx)
                && let Some(editor) = self.behavior_editor.as_mut()
            {
                editor.schedule_buffer = (*expr).to_string();
                self.selection = Selection::default();
                if let Some(e) = self.behavior_editor.as_mut() {
                    e.error = None;
                }
                return true;
            }
            if let Some(idx_str) = route.strip_prefix(BEHAVIOR_EDITOR_TZ_PRESET_PREFIX)
                && let Ok(idx) = idx_str.parse::<usize>()
                && let Some(name) = cron_preview::COMMON_TIMEZONES.get(idx)
                && let Some(editor) = self.behavior_editor.as_mut()
            {
                editor.timezone_buffer = (*name).to_string();
                self.selection = Selection::default();
                if let Some(e) = self.behavior_editor.as_mut() {
                    e.error = None;
                }
                return true;
            }
        }

        if event.is_click_or_activate(BEHAVIOR_EDITOR_CANCEL_KEY)
            || event.is_click_or_activate(BEHAVIOR_EDITOR_DISMISS_KEY)
        {
            self.behavior_editor = None;
            return true;
        }

        if event.is_click_or_activate(BEHAVIOR_EDITOR_SAVE_KEY) {
            self.submit_behavior_editor();
            return true;
        }

        false
    }

    /// Pick handler for the behavior editor's overlap / catch_up
    /// `select_trigger`s. Same Toggle / Dismiss / Pick flow as the
    /// pod editor's picker handler — `which` identifies which slot
    /// to write back; the parsed wire value lands in the matching
    /// editor buffer.
    fn handle_behavior_editor_picker(&mut self, which: BehaviorEditorPicker, action: SelectAction) {
        match action {
            SelectAction::Toggle => {
                self.close_other_pickers(which.key());
                if let Some(editor) = self.behavior_editor.as_mut() {
                    editor.open_picker = if editor.open_picker == Some(which) {
                        None
                    } else {
                        Some(which)
                    };
                }
            }
            SelectAction::Dismiss => {
                if let Some(editor) = self.behavior_editor.as_mut() {
                    editor.open_picker = None;
                }
            }
            SelectAction::Pick(value) => {
                if let Some(editor) = self.behavior_editor.as_mut() {
                    match which {
                        BehaviorEditorPicker::Overlap => {
                            if let Some(v) = overlap_from_wire(&value) {
                                editor.overlap_buffer = v;
                            }
                        }
                        BehaviorEditorPicker::CatchUp => {
                            if let Some(v) = catch_up_from_wire(&value) {
                                editor.catch_up_buffer = v;
                            }
                        }
                        BehaviorEditorPicker::ThreadModel => {
                            if let Some(cfg) = editor.working_config.as_mut() {
                                cfg.thread.model = Some(value);
                            }
                        }
                        BehaviorEditorPicker::ThreadBackend => {
                            if let Some(cfg) = editor.working_config.as_mut() {
                                cfg.thread.bindings.backend = Some(value);
                                // Backend changed — clear the model
                                // override (if any) since the prior
                                // model id is almost certainly invalid
                                // for the new backend. Same shape as
                                // the pod editor's Defaults backend
                                // pick.
                                cfg.thread.model = cfg.thread.model.as_ref().map(|_| String::new());
                            }
                        }
                        BehaviorEditorPicker::RetentionKind => {
                            // Re-apply the live `retention_days_buf`
                            // when transitioning into a timed variant
                            // so the user's typed days survive a
                            // Keep → Archive ↔ Delete flip.
                            let days = editor.retention_days_buf.parse::<u32>().unwrap_or(30);
                            if let Some(cfg) = editor.working_config.as_mut() {
                                cfg.on_completion = match value.as_str() {
                                    "keep" => RetentionPolicy::Keep,
                                    "archive_after_days" => {
                                        RetentionPolicy::ArchiveAfterDays { days }
                                    }
                                    "delete_after_days" => {
                                        RetentionPolicy::DeleteAfterDays { days }
                                    }
                                    _ => cfg.on_completion.clone(),
                                };
                            }
                        }
                        BehaviorEditorPicker::ScopeToolsDefault => {
                            if let Some(cfg) = editor.working_config.as_mut()
                                && let Some(map) = cfg.scope.tools.as_mut()
                                && let Some(v) = disposition_from_wire(&value)
                            {
                                map.default = v;
                            }
                        }
                        BehaviorEditorPicker::ScopeCapsPodModify => {
                            if let Some(cfg) = editor.working_config.as_mut()
                                && let Some(v) = pod_modify_cap_from_wire(&value)
                            {
                                cfg.scope.caps.pod_modify = Some(v);
                            }
                        }
                        BehaviorEditorPicker::ScopeCapsDispatch => {
                            if let Some(cfg) = editor.working_config.as_mut()
                                && let Some(v) = dispatch_cap_from_wire(&value)
                            {
                                cfg.scope.caps.dispatch = Some(v);
                            }
                        }
                        BehaviorEditorPicker::ScopeCapsBehaviors => {
                            if let Some(cfg) = editor.working_config.as_mut()
                                && let Some(v) = behaviors_cap_from_wire(&value)
                            {
                                cfg.scope.caps.behaviors = Some(v);
                            }
                        }
                    }
                    editor.open_picker = None;
                    editor.error = None;
                }
            }
            _ => {}
        }
    }

    /// Build the select_menu for whichever overlap / catch_up picker
    /// is open. Single-active across the editor; rendered as the
    /// topmost popover layer when `editor.open_picker` matches.
    fn behavior_editor_picker_menu(&self, which: BehaviorEditorPicker) -> El {
        match which {
            BehaviorEditorPicker::Overlap => {
                let options: Vec<(String, String)> =
                    [Overlap::Skip, Overlap::QueueOne, Overlap::Allow]
                        .into_iter()
                        .map(|o| {
                            let lbl = overlap_label(o);
                            (lbl.to_string(), lbl.to_string())
                        })
                        .collect();
                select_menu(which.key(), options)
            }
            BehaviorEditorPicker::CatchUp => {
                let options: Vec<(String, String)> = [CatchUp::None, CatchUp::One, CatchUp::All]
                    .into_iter()
                    .map(|c| {
                        let lbl = catch_up_label(c);
                        (lbl.to_string(), lbl.to_string())
                    })
                    .collect();
                select_menu(which.key(), options)
            }
            BehaviorEditorPicker::ThreadModel => {
                // Model menu options come from
                // `models_by_backend[bindings.backend]` when the
                // override is set. Empty fallback ⇒ a single
                // "(no models)" sentinel so the popover isn't blank.
                let mut options: Vec<(String, String)> = Vec::new();
                if let Some(editor) = self.behavior_editor.as_ref()
                    && let Some(cfg) = editor.working_config.as_ref()
                {
                    let backend = cfg
                        .thread
                        .bindings
                        .backend
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .unwrap_or("");
                    if let Some(models) = self.models_by_backend.get(backend) {
                        for m in models {
                            options.push((m.id.clone(), m.id.clone()));
                        }
                    }
                }
                if options.is_empty() {
                    options.push((String::new(), "(no models)".to_string()));
                }
                select_menu(which.key(), options)
            }
            BehaviorEditorPicker::ThreadBackend => {
                // Server-known backend catalog. Behaviors must bind
                // to a backend the pod's `[allow.backends]` permits;
                // showing the full list mirrors the pod editor's
                // Defaults backend menu (and the server validates
                // out-of-allow picks on save).
                let options: Vec<(String, String)> = self
                    .backends
                    .iter()
                    .map(|b| (b.name.clone(), b.name.clone()))
                    .collect();
                select_menu(which.key(), options)
            }
            BehaviorEditorPicker::RetentionKind => {
                let options: Vec<(String, String)> = [
                    ("keep", "keep"),
                    ("archive_after_days", "archive_after_days"),
                    ("delete_after_days", "delete_after_days"),
                ]
                .into_iter()
                .map(|(v, l)| (v.to_string(), l.to_string()))
                .collect();
                select_menu(which.key(), options)
            }
            BehaviorEditorPicker::ScopeToolsDefault => {
                let options: Vec<(String, String)> = [Disposition::Allow, Disposition::Deny]
                    .into_iter()
                    .map(|d| {
                        let lbl = disposition_label(d);
                        (lbl.to_string(), lbl.to_string())
                    })
                    .collect();
                select_menu(which.key(), options)
            }
            BehaviorEditorPicker::ScopeCapsPodModify => {
                use whisper_agent_protocol::PodModifyCap as C;
                let options: Vec<(String, String)> =
                    [C::None, C::Memories, C::Content, C::ModifyAllow]
                        .into_iter()
                        .map(|c| {
                            let lbl = pod_modify_cap_label(c);
                            (lbl.to_string(), lbl.to_string())
                        })
                        .collect();
                select_menu(which.key(), options)
            }
            BehaviorEditorPicker::ScopeCapsDispatch => {
                use whisper_agent_protocol::DispatchCap as C;
                let options: Vec<(String, String)> = [C::None, C::WithinScope]
                    .into_iter()
                    .map(|c| {
                        let lbl = dispatch_cap_label(c);
                        (lbl.to_string(), lbl.to_string())
                    })
                    .collect();
                select_menu(which.key(), options)
            }
            BehaviorEditorPicker::ScopeCapsBehaviors => {
                use whisper_agent_protocol::BehaviorOpsCap as C;
                let options: Vec<(String, String)> =
                    [C::None, C::Read, C::AuthorNarrower, C::AuthorAny]
                        .into_iter()
                        .map(|c| {
                            let lbl = behaviors_cap_label(c);
                            (lbl.to_string(), lbl.to_string())
                        })
                        .collect();
                select_menu(which.key(), options)
            }
        }
    }

    /// Pick handler for the behavior editor's trigger-kind menu.
    /// Mirrors the new-thread pickers' classify-then-mutate shape.
    fn handle_behavior_editor_trigger_kind_pick(&mut self, action: SelectAction) {
        match action {
            SelectAction::Toggle => {
                self.close_other_pickers(BEHAVIOR_EDITOR_TRIGGER_KIND_KEY);
                if let Some(editor) = self.behavior_editor.as_mut() {
                    editor.trigger_kind_open = !editor.trigger_kind_open;
                }
            }
            SelectAction::Dismiss => {
                if let Some(editor) = self.behavior_editor.as_mut() {
                    editor.trigger_kind_open = false;
                }
            }
            SelectAction::Pick(value) => {
                if let Some(editor) = self.behavior_editor.as_mut() {
                    if let Some(kind) = TriggerKindLabel::from_wire(&value) {
                        editor.working_kind = kind;
                        editor.error = None;
                    }
                    editor.trigger_kind_open = false;
                }
            }
            // `SelectAction` is `#[non_exhaustive]`.
            _ => {}
        }
    }

    /// Validate + dispatch the behavior editor's Save. Same
    /// correlation-stamping pattern as the create modals: mint an id,
    /// stash it in `pending_save`, fire `UpdateBehavior`. The wire
    /// echo (`BehaviorUpdated` or `Error`) closes or re-enables the
    /// sheet from the inbound handler.
    fn submit_behavior_editor(&mut self) {
        let Some(editor) = self.behavior_editor.as_ref() else {
            return;
        };
        if editor.pending_save.is_some() {
            return;
        }

        // Raw-current with edits ⇒ ship the buffer's parse, not the
        // structured `working_config`. A parse failure surfaces in
        // `error` and stays on Raw so the typed text isn't lost
        // (mirrors the pod editor's `resolved_save_toml` path).
        let raw_save = editor.tab == BehaviorEditorTab::RawToml && editor.raw_dirty;
        let config = if raw_save {
            match toml::from_str::<BehaviorConfig>(&editor.working_toml) {
                Ok(cfg) => cfg,
                Err(e) => {
                    self.set_behavior_editor_error(format!("raw TOML doesn't parse: {e}"));
                    return;
                }
            }
        } else {
            let Some(working) = editor.working_config.as_ref() else {
                // Snapshot hasn't landed yet (or load_error). Save isn't
                // meaningful until we have a config to mutate.
                return;
            };
            let mut snapshot = working.clone();
            snapshot.trigger = editor.resolved_trigger();
            snapshot
        };
        if config.name.trim().is_empty() {
            self.set_behavior_editor_error("name is empty".into());
            return;
        }
        if !raw_save
            && matches!(editor.working_kind, TriggerKindLabel::Cron)
            && editor.schedule_buffer.trim().is_empty()
        {
            self.set_behavior_editor_error("cron schedule is empty".into());
            return;
        }

        let pod_id = editor.pod_id.clone();
        let behavior_id = editor.behavior_id.clone();
        let prompt = editor.working_prompt.clone();
        // Ship the side-file buffer when the override is on (File
        // variant). For the Text variant the content is already in
        // `config.thread.system_prompt`, so no side-file write is
        // needed — `system_prompt = None` leaves the on-disk
        // `system_prompt.md` alone. For override-off, also `None`.
        let system_prompt = match config.thread.system_prompt.as_ref() {
            Some(SystemPromptChoice::File { .. }) => editor.working_system_prompt.clone(),
            _ => None,
        };
        let correlation_id = self.next_correlation_id();
        if let Some(editor) = self.behavior_editor.as_mut() {
            editor.pending_save = Some(correlation_id.clone());
            editor.error = None;
        }
        self.send(ClientToServer::UpdateBehavior {
            correlation_id: Some(correlation_id),
            pod_id,
            behavior_id,
            config,
            prompt,
            system_prompt,
        });
    }

    fn set_behavior_editor_error(&mut self, msg: String) {
        if let Some(editor) = self.behavior_editor.as_mut() {
            editor.error = Some(msg);
        }
    }

    /// Render the "+ New behavior" dialog if open. Same shadcn
    /// scaffolding as the new-pod dialog; the only difference is
    /// the field labels and the description text (manual-trigger
    /// stub vs pod template).
    fn render_new_behavior_modal(&self) -> Option<El> {
        let modal = self.new_behavior_modal.as_ref()?;

        let behavior_id_key = format!("{NEW_BEHAVIOR_MODAL_KEY}:behavior-id");
        let name_key = format!("{NEW_BEHAVIOR_MODAL_KEY}:name");
        let create_key = format!("{NEW_BEHAVIOR_MODAL_KEY}:create");
        let cancel_key = format!("{NEW_BEHAVIOR_MODAL_KEY}:cancel");

        let behavior_id_input = text_input(&modal.behavior_id, &self.selection, &behavior_id_key);
        let name_input = text_input(&modal.name, &self.selection, &name_key);

        let pending = modal.pending_correlation.is_some();
        let create_enabled =
            !modal.behavior_id.trim().is_empty() && !modal.name.trim().is_empty() && !pending;

        let mut create = button(if pending { "Creating…" } else { "Create" })
            .key(&create_key)
            .primary();
        if !create_enabled {
            create = create.disabled();
        }
        let cancel = button("Cancel").key(&cancel_key);

        // Compose the title with the pod scope so the user knows
        // which pod the new behavior lands in (a power user can
        // have several open at once via the per-pod sidebar
        // entry-point fanout).
        let pod_label = self
            .pods
            .get(&modal.pod_id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| modal.pod_id.clone());
        let title = format!("New behavior — {pod_label}");

        let mut children: Vec<El> = vec![
            dialog_header([
                dialog_title(title),
                dialog_description(
                    "Starts as a manually-triggered behavior with an empty \
                     prompt. Edit in the (eventual) full editor to add a \
                     trigger, override thread settings, or write the prompt.",
                ),
            ]),
            form([
                form_item([
                    form_label("behavior_id"),
                    form_control(behavior_id_input),
                    form_description(
                        "Becomes the behavior's directory name under the pod; \
                         immutable after creation.",
                    ),
                ]),
                form_item([
                    form_label("name"),
                    form_control(name_input),
                    form_description("Display name (free text)."),
                ]),
            ]),
        ];
        if let Some(err) = modal.error.as_deref() {
            children.push(
                alert([
                    alert_title("couldn't create behavior"),
                    alert_description(err.to_string()),
                ])
                .destructive(),
            );
        }
        children.push(dialog_footer([cancel, create]));

        Some(dialog(NEW_BEHAVIOR_MODAL_KEY, children))
    }

    /// Render the "+ New pod" dialog if its modal state is open.
    /// Children pass through `dialog` directly as siblings — the
    /// shadcn-shaped `dialog_header` / `dialog_footer` are just
    /// styled column / row wrappers, the body items in between
    /// (form, alerts) inherit `dialog_content`'s default gap.
    fn render_new_pod_modal(&self) -> Option<El> {
        let modal = self.new_pod_modal.as_ref()?;

        let pod_id_key = format!("{NEW_POD_MODAL_KEY}:pod-id");
        let name_key = format!("{NEW_POD_MODAL_KEY}:name");
        let create_key = format!("{NEW_POD_MODAL_KEY}:create");
        let cancel_key = format!("{NEW_POD_MODAL_KEY}:cancel");

        let pod_id_input = text_input(&modal.pod_id, &self.selection, &pod_id_key);
        let name_input = text_input(&modal.name, &self.selection, &name_key);

        let backends_empty = self.backends.is_empty();
        let pending = modal.pending_correlation.is_some();
        // Mirror the egui sibling's enable rule: both fields trimmed
        // non-empty, at least one backend known (so `fresh_pod_config`
        // produces a usable thread_defaults.backend), and no request
        // already in flight.
        let create_enabled = !modal.pod_id.trim().is_empty()
            && !modal.name.trim().is_empty()
            && !backends_empty
            && !pending;

        let mut create = button(if pending { "Creating…" } else { "Create" })
            .key(&create_key)
            .primary();
        if !create_enabled {
            create = create.disabled();
        }
        let cancel = button("Cancel").key(&cancel_key);

        let mut children: Vec<El> = vec![
            dialog_header([
                dialog_title("New pod"),
                dialog_description(
                    "The new pod starts with the configured backends, an \
                     empty MCP allow-list, and the default tool surface. \
                     Use the (eventual) pod editor to add MCPs / host \
                     envs / tool overrides afterwards.",
                ),
            ]),
            form([
                form_item([
                    form_label("pod_id"),
                    form_control(pod_id_input),
                    form_description(
                        "Becomes the pod's directory name on disk; immutable \
                         after creation. Letters, numbers, dashes, underscores.",
                    ),
                ]),
                form_item([
                    form_label("name"),
                    form_control(name_input),
                    form_description("Display name (free text)."),
                ]),
            ]),
        ];

        if backends_empty {
            children.push(
                alert([
                    alert_title("no backends configured"),
                    alert_description(
                        "Add at least one backend on the server before creating a pod — \
                         the new pod's `thread_defaults.backend` would be empty otherwise.",
                    ),
                ])
                .warning(),
            );
        }
        if let Some(err) = modal.error.as_deref() {
            children.push(
                alert([
                    alert_title("couldn't create pod"),
                    alert_description(err.to_string()),
                ])
                .destructive(),
            );
        }

        children.push(dialog_footer([cancel, create]));

        Some(dialog(NEW_POD_MODAL_KEY, children))
    }

    /// Render the per-behavior editor modal if its state slot is
    /// open. Centered modal dialog (mirroring egui's
    /// `Window::new(...).anchor(CENTER_CENTER)`) because the 7-tab
    /// strip + the Defaults / Thread / Scope tabs need substantially
    /// more horizontal room than a right-attached sheet can comfortably
    /// give. Body composes shadcn-shaped form items: the same
    /// `form_item` (label / control / optional description) the create
    /// modals use, just inside a wider dialog shell.
    ///
    /// Body shape:
    /// - dialog_header: title (`Edit behavior — {pod}/{id}`) + a one-
    ///   liner reminding the user that the v1 form covers a subset
    ///   of the on-disk config (everything else rides through
    ///   unchanged on save).
    /// - loading placeholder while `pending_get` is in flight and no
    ///   `working_config` has landed yet — prevents accidentally
    ///   serializing a default `BehaviorConfig` and overwriting the
    ///   on-disk one.
    /// - form: name / description / trigger-kind / [cron schedule] /
    ///   prompt. Cron schedule slot only appears when
    ///   `working_kind == Cron`.
    /// - alert: load error or save failure. Destructive when present.
    /// - dialog_footer: Cancel + Save (Save disabled until snapshot
    ///   lands and during a pending save).
    fn render_behavior_editor_modal(&self) -> Option<El> {
        let editor = self.behavior_editor.as_ref()?;

        let pod_label = self
            .pods
            .get(&editor.pod_id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| editor.pod_id.clone());
        let title = format!("Edit behavior — {pod_label}/{}", editor.behavior_id);
        let header = dialog_header([
            dialog_title(title),
            dialog_description(
                "Trigger / Thread / Scope / Retention / Prompt / System / Raw \
                 tabs split the per-behavior surface up. Trigger and Prompt \
                 are wired in this slice; the other tabs land in follow-up \
                 commits.",
            ),
        ]);

        let tab_value = editor.tab.wire_value().to_string();
        let tabs_strip = tabs_list(
            BEHAVIOR_EDITOR_TABS_KEY,
            &tab_value,
            [
                (
                    BehaviorEditorTab::Trigger.wire_value(),
                    BehaviorEditorTab::Trigger.label(),
                ),
                (
                    BehaviorEditorTab::Thread.wire_value(),
                    BehaviorEditorTab::Thread.label(),
                ),
                (
                    BehaviorEditorTab::Scope.wire_value(),
                    BehaviorEditorTab::Scope.label(),
                ),
                (
                    BehaviorEditorTab::Retention.wire_value(),
                    BehaviorEditorTab::Retention.label(),
                ),
                (
                    BehaviorEditorTab::Prompt.wire_value(),
                    BehaviorEditorTab::Prompt.label(),
                ),
                (
                    BehaviorEditorTab::SystemPrompt.wire_value(),
                    BehaviorEditorTab::SystemPrompt.label(),
                ),
                (
                    BehaviorEditorTab::RawToml.wire_value(),
                    BehaviorEditorTab::RawToml.label(),
                ),
            ],
        );

        let body: El = match editor.working_config.as_ref() {
            None => {
                // Either the snapshot hasn't landed yet (pending_get
                // active) or it landed with a load_error and no
                // parsed config to render against. The error path
                // surfaces the message via the alert below; the
                // pending path just reads "loading…".
                let label = if editor.pending_get.is_some() {
                    "loading behavior…"
                } else {
                    "no parsed config — fix the on-disk TOML to edit here"
                };
                paragraph(label).muted()
            }
            Some(cfg) => match editor.tab {
                BehaviorEditorTab::Trigger => self.render_behavior_editor_trigger_tab(editor, cfg),
                BehaviorEditorTab::Prompt => self.render_behavior_editor_prompt_tab(editor),
                BehaviorEditorTab::Thread => self.render_behavior_editor_thread_tab(editor, cfg),
                BehaviorEditorTab::Scope => self.render_behavior_editor_scope_tab(editor, cfg),
                BehaviorEditorTab::Retention => {
                    self.render_behavior_editor_retention_tab(editor, cfg)
                }
                BehaviorEditorTab::SystemPrompt => {
                    self.render_behavior_editor_system_prompt_tab(editor, cfg)
                }
                BehaviorEditorTab::RawToml => self.render_behavior_editor_raw_tab(editor),
            },
        };

        let pending_save = editor.pending_save.is_some();
        let mut save = button(if pending_save { "Saving…" } else { "Save" })
            .key(BEHAVIOR_EDITOR_SAVE_KEY)
            .primary();
        if editor.working_config.is_none() || pending_save {
            save = save.disabled();
        }
        let cancel = button("Cancel").key(BEHAVIOR_EDITOR_CANCEL_KEY);

        // Body + optional error alert ride inside a `scroll` so the
        // dialog panel never overflows vertically — the prompt
        // text_area alone is 160 px and the form has ~6 form_items, so
        // the natural height regularly exceeds the dialog's fixed
        // height. Header and footer stay fixed; the scroll grabs the
        // leftover height. `.key` so the scroll offset survives
        // rebuilds across edit clicks.
        let mut body_children: Vec<El> = vec![tabs_strip, body];
        if let Some(err) = editor.error.as_deref() {
            body_children.push(
                alert([
                    alert_title("couldn't save behavior"),
                    alert_description(err.to_string()),
                ])
                .destructive(),
            );
        }
        let scroll_body = scroll(body_children)
            .key("behavior-editor:scroll")
            .gap(tokens::SPACE_4);

        let children: Vec<El> = vec![header, scroll_body, dialog_footer([cancel, save])];

        // Inline `dialog()` shape so we can override
        // `dialog_content`'s 420 px stock width: the 7-tab strip and
        // the Defaults / Thread / Scope rows need substantially more
        // room. 720 px matches egui's `Window::default_width(720)` and
        // leaves a comfortable margin on a 1280 px viewport. Height
        // pinned at 640 px so the inner scroll has a bounded extent;
        // mirrors egui's `default_height(560)` plus a little headroom
        // for the larger form_item gaps aetna uses.
        let panel = dialog_content(children)
            .block_pointer()
            .width(Size::Fixed(720.0))
            .height(Size::Fixed(640.0));
        let layer = overlay([scrim(format!("{BEHAVIOR_EDITOR_KEY}:dismiss")), panel])
            .align(Align::Center)
            .justify(Justify::Center);
        Some(layer)
    }

    /// Trigger tab body. Identity (name + description) + trigger kind
    /// picker + the cron / webhook / manual sub-form. Cron preview
    /// (next-firings render) is deferred to a follow-up slice.
    fn render_behavior_editor_trigger_tab(
        &self,
        editor: &BehaviorEditorSheetState,
        cfg: &BehaviorConfig,
    ) -> El {
        let name_input = text_input(&cfg.name, &self.selection, BEHAVIOR_EDITOR_NAME_KEY);
        let description_input = text_area(
            cfg.description.as_deref().unwrap_or(""),
            &self.selection,
            BEHAVIOR_EDITOR_DESCRIPTION_KEY,
        )
        .height(Size::Fixed(80.0));
        let kind_trigger = select_trigger(
            BEHAVIOR_EDITOR_TRIGGER_KIND_KEY,
            editor.working_kind.display_label(),
        );

        let mut items: Vec<El> = vec![
            form_item([
                form_label("name"),
                form_control(name_input),
                form_description("Display name; freely editable."),
            ]),
            form_item([
                form_label("description"),
                form_control(description_input),
                form_description(
                    "Short summary surfaced in the sidebar's behavior rows. \
                     Empty clears the field on save.",
                ),
            ]),
            form_item([
                form_label("trigger kind"),
                form_control(kind_trigger),
                form_description(
                    "Manual = fires only on RunBehavior. Cron = scheduled. \
                     Webhook = HTTP POST to the endpoint shown.",
                ),
            ]),
        ];

        match editor.working_kind {
            TriggerKindLabel::Manual => {
                items.push(form_item([
                    form_label("manual"),
                    form_control(
                        paragraph(
                            "No further configuration. Use the Run icon in \
                             the sidebar (or RunBehavior on the wire) to \
                             fire.",
                        )
                        .muted()
                        .small(),
                    ),
                ]));
            }
            TriggerKindLabel::Cron => {
                let schedule_input = text_input(
                    &editor.schedule_buffer,
                    &self.selection,
                    BEHAVIOR_EDITOR_SCHEDULE_KEY,
                );
                let timezone_input = text_input(
                    &editor.timezone_buffer,
                    &self.selection,
                    BEHAVIOR_EDITOR_TIMEZONE_KEY,
                );
                let overlap_trigger = select_trigger(
                    BEHAVIOR_EDITOR_OVERLAP_KEY,
                    overlap_label(editor.overlap_buffer),
                );
                let catch_up_trigger = select_trigger(
                    BEHAVIOR_EDITOR_CATCH_UP_KEY,
                    catch_up_label(editor.catch_up_buffer),
                );

                let cron_chips = self.render_preset_chips(
                    BEHAVIOR_EDITOR_CRON_PRESET_PREFIX,
                    cron_preview::CRON_PRESETS
                        .iter()
                        .enumerate()
                        .map(|(idx, (label, _))| (idx, (*label).to_string())),
                    4,
                );
                let tz_chips = self.render_preset_chips(
                    BEHAVIOR_EDITOR_TZ_PRESET_PREFIX,
                    cron_preview::COMMON_TIMEZONES
                        .iter()
                        .enumerate()
                        .map(|(idx, name)| (idx, (*name).to_string())),
                    3,
                );

                items.push(form_item([
                    form_label("schedule"),
                    form_control(schedule_input),
                    form_description(
                        "Five-field cron expression (minute hour day-of-month \
                         month day-of-week). Example: `0 9 * * *` for daily \
                         9am.",
                    ),
                ]));
                items.push(form_item([
                    form_label("schedule presets"),
                    form_control(cron_chips),
                ]));
                items.push(form_item([
                    form_label("timezone"),
                    form_control(timezone_input),
                    form_description("IANA name, e.g. `America/Los_Angeles`."),
                ]));
                items.push(form_item([
                    form_label("timezone presets"),
                    form_control(tz_chips),
                ]));
                items.push(form_item([
                    form_label("overlap"),
                    form_control(overlap_trigger),
                    form_description(
                        "What happens if a fire arrives while the previous \
                         run is still in flight.",
                    ),
                ]));
                items.push(form_item([
                    form_label("catch_up"),
                    form_control(catch_up_trigger),
                    form_description(
                        "What to do about cron fires missed while the server \
                         was down.",
                    ),
                ]));
                items.push(form_item([
                    form_label("preview"),
                    form_control(
                        self.render_cron_preview(&editor.schedule_buffer, &editor.timezone_buffer),
                    ),
                ]));
            }
            TriggerKindLabel::Webhook => {
                let overlap_trigger = select_trigger(
                    BEHAVIOR_EDITOR_OVERLAP_KEY,
                    overlap_label(editor.overlap_buffer),
                );
                items.push(form_item([
                    form_label("overlap"),
                    form_control(overlap_trigger),
                    form_description(
                        "Same semantics as Cron: Skip drops, QueueOne parks \
                         one, Allow spawns concurrent runs.",
                    ),
                ]));
                items.push(form_item([
                    form_label("endpoint"),
                    form_control(
                        paragraph(format!(
                            "POST /triggers/{}/{} (relative to server). \
                             Empty body → null payload; non-empty bodies \
                             must be valid JSON.",
                            editor.pod_id, editor.behavior_id
                        ))
                        .muted()
                        .small(),
                    ),
                ]));
            }
        }
        form(items)
    }

    /// Build a wrapping-style row of preset chips. Aetna's `row`
    /// doesn't flex-wrap, so we approximate by splitting the input
    /// list into multiple `row(...)`s under one `column`. `chunk` is
    /// the per-row count — picked by the caller to fit the sheet's
    /// content width.
    fn render_preset_chips<I>(&self, key_prefix: &str, presets: I, chunk: usize) -> El
    where
        I: IntoIterator<Item = (usize, String)>,
    {
        let presets: Vec<(usize, String)> = presets.into_iter().collect();
        let rows: Vec<El> = presets
            .chunks(chunk)
            .map(|group| {
                let buttons: Vec<El> = group
                    .iter()
                    .map(|(idx, label)| {
                        button(label.clone())
                            .key(format!("{key_prefix}{idx}"))
                            .ghost()
                    })
                    .collect();
                row(buttons).gap(tokens::SPACE_2)
            })
            .collect();
        column(rows).gap(tokens::SPACE_1)
    }

    /// Cron-preview body: validate the schedule + timezone, then
    /// either show a destructive alert with the parse error or a
    /// small column of next-firing rows. Uses the same parsers the
    /// server's scheduler uses, so the preview's "next 5 fires"
    /// agrees with what the cron job will actually fire on.
    fn render_cron_preview(&self, schedule_str: &str, tz_str: &str) -> El {
        use chrono::Utc;
        match (
            cron_preview::parse_schedule(schedule_str),
            cron_preview::parse_tz(tz_str),
        ) {
            (Err(e), _) => paragraph(format!("schedule: {e}")).destructive().small(),
            (_, Err(e)) => paragraph(e).destructive().small(),
            (Ok(schedule), Ok(tz)) => {
                let upcoming = cron_preview::next_firings(&schedule, tz);
                if upcoming.is_empty() {
                    return paragraph("schedule has no upcoming fires")
                        .destructive()
                        .small();
                }
                let now_utc = Utc::now();
                let header = paragraph(format!("next {} firings ({})", upcoming.len(), tz.name()))
                    .muted()
                    .small();
                let rows: Vec<El> = upcoming
                    .iter()
                    .map(|fire| {
                        let when = fire.format("%Y-%m-%d %H:%M %Z").to_string();
                        let rel = cron_preview::format_relative(now_utc, fire.with_timezone(&Utc));
                        row([
                            mono(when).text_color(tokens::FOREGROUND),
                            text(rel).muted().small(),
                        ])
                        .gap(tokens::SPACE_4)
                    })
                    .collect();
                column([header].into_iter().chain(rows).collect::<Vec<_>>()).gap(tokens::SPACE_1)
            }
        }
    }

    /// Prompt tab body. The behavior's `prompt.md` text — the user
    /// turn the spawned thread sees first. `{{payload}}` is the only
    /// templated substitution.
    fn render_behavior_editor_prompt_tab(&self, editor: &BehaviorEditorSheetState) -> El {
        let prompt_input = text_area(
            &editor.working_prompt,
            &self.selection,
            BEHAVIOR_EDITOR_PROMPT_KEY,
        )
        .height(Size::Fixed(360.0));
        form([form_item([
            form_label("prompt"),
            form_control(prompt_input),
            form_description(
                "The behavior's `prompt.md` body. `{{payload}}` is \
                 substituted with the trigger's payload at fire time.",
            ),
        ])])
    }

    /// Thread tab body. Scalar-override knobs on
    /// `BehaviorThreadOverride`: `model`, `max_tokens`, `max_turns`.
    /// Each row is a `[checkbox, control-or-inherit-hint]` pair —
    /// checking the override flips `Some(default) ↔ None`. The
    /// bindings sub-struct (backend / host_env / mcp_hosts) is
    /// deferred to a follow-up sub-slice so this commit stays focused
    /// on scalars.
    fn render_behavior_editor_thread_tab(
        &self,
        editor: &BehaviorEditorSheetState,
        cfg: &BehaviorConfig,
    ) -> El {
        // Effective backend for the model picker. The pod's effective
        // backend (its `thread_defaults.backend`) lives in the pod's
        // full config, not the summary — the bindings sub-slice that
        // wires `bindings.backend` will also wire that lookup. For
        // now: prefer the binding override when set; otherwise leave
        // empty and let the model menu render "(no models)".
        let effective_backend = cfg
            .thread
            .bindings
            .backend
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("");

        // Model row.
        let model_override = cfg.thread.model.is_some();
        let model_value: El = if let Some(model) = cfg.thread.model.as_ref() {
            let label = if model.is_empty() {
                "(none)".to_string()
            } else {
                model.clone()
            };
            select_trigger(BEHAVIOR_EDITOR_THREAD_MODEL_KEY, label)
        } else {
            paragraph("(inherit pod default)").muted().small()
        };
        let model_row = row([
            checkbox(model_override).key(BEHAVIOR_EDITOR_THREAD_MODEL_OVERRIDE_KEY),
            text("override").muted().small(),
            model_value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .width(Size::Fill(1.0));

        // Numeric override row helper inlined twice — splitting it
        // out into a method would carry a bag of arguments (label,
        // override_key, value_key, buf, opts) and only buy a small
        // dedup, so keep it inline here.
        let max_tokens_override = cfg.thread.max_tokens.is_some();
        let max_tokens_value: El = if max_tokens_override {
            numeric_input(
                &editor.thread_max_tokens_buf,
                &self.selection,
                BEHAVIOR_EDITOR_THREAD_MAX_TOKENS_KEY,
                NumericInputOpts::default()
                    .min(1.0)
                    .max(200_000.0)
                    .step(50.0),
            )
        } else {
            paragraph("(inherit pod default)").muted().small()
        };
        let max_tokens_row = row([
            checkbox(max_tokens_override).key(BEHAVIOR_EDITOR_THREAD_MAX_TOKENS_OVERRIDE_KEY),
            text("override").muted().small(),
            max_tokens_value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .width(Size::Fill(1.0));

        let max_turns_override = cfg.thread.max_turns.is_some();
        let max_turns_value: El = if max_turns_override {
            numeric_input(
                &editor.thread_max_turns_buf,
                &self.selection,
                BEHAVIOR_EDITOR_THREAD_MAX_TURNS_KEY,
                NumericInputOpts::default().min(1.0).max(10_000.0).step(1.0),
            )
        } else {
            paragraph("(inherit pod default)").muted().small()
        };
        let max_turns_row = row([
            checkbox(max_turns_override).key(BEHAVIOR_EDITOR_THREAD_MAX_TURNS_OVERRIDE_KEY),
            text("override").muted().small(),
            max_turns_value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .width(Size::Fill(1.0));

        let backend_hint = format!(
            "Effective backend for the model picker: `{}`. Use the bindings \
             override below to switch the spawned thread to a different \
             backend.",
            if effective_backend.is_empty() {
                "(inherit pod default)"
            } else {
                effective_backend
            }
        );

        // Bindings.backend row (Optional<String>).
        let backend_override = cfg.thread.bindings.backend.is_some();
        let backend_value: El = if let Some(name) = cfg.thread.bindings.backend.as_ref() {
            let label = if name.is_empty() {
                "(none)".to_string()
            } else {
                name.clone()
            };
            select_trigger(BEHAVIOR_EDITOR_THREAD_BACKEND_KEY, label)
        } else {
            paragraph("(inherit pod default)").muted().small()
        };
        let backend_row = row([
            checkbox(backend_override).key(BEHAVIOR_EDITOR_THREAD_BACKEND_OVERRIDE_KEY),
            text("override").muted().small(),
            backend_value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .width(Size::Fill(1.0));

        // host_env / mcp_hosts override rows. When override is OFF
        // we render the standard inherit hint; when ON we render a
        // multi-check column over the pod's allow list. The pod
        // config may not have landed yet (parallel `GetPod` is in
        // flight), so an unhydrated state surfaces a "(loading…)"
        // hint rather than a blank list.
        let host_env_override = cfg.thread.bindings.host_env.is_some();
        let host_env_value: El = if !host_env_override {
            paragraph("(inherit pod default)").muted().small()
        } else if editor.pending_pod_get.is_some() {
            paragraph("(loading pod allow list…)").muted().small()
        } else if let Some(pod_cfg) = editor.pod_config.as_ref() {
            self.render_behavior_editor_thread_host_env_check(cfg, pod_cfg)
        } else {
            paragraph("(pod config unavailable)").muted().small()
        };
        let host_env_row = row([
            checkbox(host_env_override).key(BEHAVIOR_EDITOR_THREAD_HOST_ENV_OVERRIDE_KEY),
            text("override").muted().small(),
            host_env_value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Start)
        .width(Size::Fill(1.0));

        let mcp_hosts_override = cfg.thread.bindings.mcp_hosts.is_some();
        let mcp_hosts_value: El = if !mcp_hosts_override {
            paragraph("(inherit pod default)").muted().small()
        } else if editor.pending_pod_get.is_some() {
            paragraph("(loading pod allow list…)").muted().small()
        } else if let Some(pod_cfg) = editor.pod_config.as_ref() {
            self.render_behavior_editor_thread_mcp_hosts_check(cfg, pod_cfg)
        } else {
            paragraph("(pod config unavailable)").muted().small()
        };
        let mcp_hosts_row = row([
            checkbox(mcp_hosts_override).key(BEHAVIOR_EDITOR_THREAD_MCP_HOSTS_OVERRIDE_KEY),
            text("override").muted().small(),
            mcp_hosts_value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Start)
        .width(Size::Fill(1.0));

        form([
            form_item([
                form_label("model"),
                form_control(model_row),
                form_description(backend_hint),
            ]),
            form_item([
                form_label("max tokens"),
                form_control(max_tokens_row),
                form_description(
                    "Per-response output cap for the spawned thread. Inherits \
                     `thread_defaults.max_tokens` when off.",
                ),
            ]),
            form_item([
                form_label("max turns"),
                form_control(max_turns_row),
                form_description(
                    "Per-cycle assistant-turn cap for the spawned thread. \
                     Inherits `thread_defaults.max_turns` when off.",
                ),
            ]),
            form_item([
                form_label("backend binding"),
                form_control(backend_row),
                form_description(
                    "Server-known backend catalog. The pod's `[allow.backends]` \
                     is authoritative — picks outside it are rejected on save.",
                ),
            ]),
            form_item([
                form_label("host_env binding"),
                form_control(host_env_row),
                form_description(
                    "Replace the pod-default host_env list for this behavior's \
                     spawned threads. Options come from the pod's \
                     `[allow.host_env]`.",
                ),
            ]),
            form_item([
                form_label("mcp_hosts binding"),
                form_control(mcp_hosts_row),
                form_description(
                    "Replace the pod-default mcp_hosts list for this behavior's \
                     spawned threads. Options come from the pod's \
                     `[allow.mcp_hosts]`.",
                ),
            ]),
        ])
    }

    /// Multi-check column over the pod's `allow.host_env` names.
    /// Returns a "(no host envs in pod allow list)" hint when the
    /// pod declares none — distinct from the parent's
    /// "(loading…)" / "(pod config unavailable)" placeholders.
    fn render_behavior_editor_thread_host_env_check(
        &self,
        cfg: &BehaviorConfig,
        pod_cfg: &PodConfig,
    ) -> El {
        if pod_cfg.allow.host_env.is_empty() {
            return paragraph("(no host envs in pod [allow])").muted().small();
        }
        let options: Vec<(String, String)> = pod_cfg
            .allow
            .host_env
            .iter()
            .map(|e| (e.name.clone(), e.name.clone()))
            .collect();
        let selected = cfg.thread.bindings.host_env.as_deref().unwrap_or(&[]);
        checkbox_column(BEHAVIOR_EDITOR_THREAD_HOST_ENV_KEY, selected, options)
    }

    fn render_behavior_editor_thread_mcp_hosts_check(
        &self,
        cfg: &BehaviorConfig,
        pod_cfg: &PodConfig,
    ) -> El {
        if pod_cfg.allow.mcp_hosts.is_empty() {
            return paragraph("(no shared MCP hosts in pod [allow])")
                .muted()
                .small();
        }
        let options: Vec<(String, String)> = pod_cfg
            .allow
            .mcp_hosts
            .iter()
            .map(|n| (n.clone(), n.clone()))
            .collect();
        let selected = cfg.thread.bindings.mcp_hosts.as_deref().unwrap_or(&[]);
        checkbox_column(BEHAVIOR_EDITOR_THREAD_MCP_HOSTS_KEY, selected, options)
    }

    /// Scope tab body. Per-behavior allow narrowing — at fire time
    /// the spawned thread runs with `pod.allow.narrow(scope)`, so
    /// every field on `BehaviorScope` is `Option`-shaped: `None`
    /// means inherit the pod ceiling, `Some(...)` narrows further.
    /// Mirrors the egui sibling's `render_behavior_editor_scope_tab`,
    /// minus the per-tool overrides (those are an unbounded
    /// `String → Disposition` map; the egui sibling defers them to
    /// Raw TOML and we follow suit) and the structured tool-surface
    /// editor (deferred for the same reason as the pod editor's
    /// Defaults `tool_surface` sub-slice).
    fn render_behavior_editor_scope_tab(
        &self,
        editor: &BehaviorEditorSheetState,
        cfg: &BehaviorConfig,
    ) -> El {
        // Resource set rows: backends / host_envs / mcp_hosts. Each
        // row has an override checkbox; when on, a multi-check column
        // over the corresponding pod allow list. Pod config may not
        // have landed yet, so render a "(loading…)" hint in flight.
        let backends_row = self.render_behavior_editor_scope_resource_row(
            editor,
            cfg.scope.backends.as_deref(),
            BEHAVIOR_EDITOR_SCOPE_BACKENDS_OVERRIDE_KEY,
            BEHAVIOR_EDITOR_SCOPE_BACKENDS_KEY,
            |pod_cfg| pod_cfg.allow.backends.clone(),
            "(no backends in pod [allow])",
        );
        let host_envs_row = self.render_behavior_editor_scope_resource_row(
            editor,
            cfg.scope.host_envs.as_deref(),
            BEHAVIOR_EDITOR_SCOPE_HOST_ENVS_OVERRIDE_KEY,
            BEHAVIOR_EDITOR_SCOPE_HOST_ENVS_KEY,
            |pod_cfg| {
                pod_cfg
                    .allow
                    .host_env
                    .iter()
                    .map(|e| e.name.clone())
                    .collect()
            },
            "(no host envs in pod [allow])",
        );
        let mcp_hosts_row = self.render_behavior_editor_scope_resource_row(
            editor,
            cfg.scope.mcp_hosts.as_deref(),
            BEHAVIOR_EDITOR_SCOPE_MCP_HOSTS_OVERRIDE_KEY,
            BEHAVIOR_EDITOR_SCOPE_MCP_HOSTS_KEY,
            |pod_cfg| pod_cfg.allow.mcp_hosts.clone(),
            "(no shared MCP hosts in pod [allow])",
        );

        // Tools row: override on ⇒ Disposition select_trigger +
        // override-count text; off ⇒ inherit hint. Per-tool
        // overrides defer to Raw TOML (matches egui).
        let tools_override = cfg.scope.tools.is_some();
        let tools_value: El = if let Some(map) = cfg.scope.tools.as_ref() {
            let trigger = select_trigger(
                BEHAVIOR_EDITOR_SCOPE_TOOLS_DEFAULT_KEY,
                disposition_label(map.default),
            );
            let override_count = map.overrides.len();
            let count_label = if override_count == 0 {
                "(no per-tool overrides)".to_string()
            } else {
                format!("{override_count} per-tool override(s) — edit via Raw TOML")
            };
            row([trigger, text(count_label).muted().small()])
                .gap(tokens::SPACE_2)
                .align(Align::Center)
                .width(Size::Fill(1.0))
        } else {
            paragraph("(inherit pod allow.tools)").muted().small()
        };
        let tools_row = row([
            checkbox(tools_override).key(BEHAVIOR_EDITOR_SCOPE_TOOLS_OVERRIDE_KEY),
            text("override").muted().small(),
            tools_value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .width(Size::Fill(1.0));

        // Cap rows: each cap is `Option<Cap>`; override on ⇒
        // select_trigger over the cap's variants; off ⇒ inherit
        // hint. Mirrors the egui sibling's per-cap helpers.
        let pod_modify_row = self.render_behavior_editor_scope_cap_row(
            cfg.scope.caps.pod_modify.map(pod_modify_cap_label),
            BEHAVIOR_EDITOR_SCOPE_CAPS_POD_MODIFY_OVERRIDE_KEY,
            BEHAVIOR_EDITOR_SCOPE_CAPS_POD_MODIFY_KEY,
        );
        let dispatch_row = self.render_behavior_editor_scope_cap_row(
            cfg.scope.caps.dispatch.map(dispatch_cap_label),
            BEHAVIOR_EDITOR_SCOPE_CAPS_DISPATCH_OVERRIDE_KEY,
            BEHAVIOR_EDITOR_SCOPE_CAPS_DISPATCH_KEY,
        );
        let behaviors_row = self.render_behavior_editor_scope_cap_row(
            cfg.scope.caps.behaviors.map(behaviors_cap_label),
            BEHAVIOR_EDITOR_SCOPE_CAPS_BEHAVIORS_OVERRIDE_KEY,
            BEHAVIOR_EDITOR_SCOPE_CAPS_BEHAVIORS_KEY,
        );

        // Tool-surface row: override checkbox + inherit/edit hint.
        // Structured editor lands later (same pattern as the pod
        // editor's Defaults `tool_surface` sub-slice).
        let tool_surface_override = cfg.scope.tool_surface.is_some();
        let tool_surface_value: El = if tool_surface_override {
            paragraph(
                "(override on — structured editor coming in a follow-up; \
                 use Raw TOML to edit fields)",
            )
            .muted()
            .small()
        } else {
            paragraph("(inherit pod thread_defaults.tool_surface)")
                .muted()
                .small()
        };
        let tool_surface_row = row([
            checkbox(tool_surface_override).key(BEHAVIOR_EDITOR_SCOPE_TOOL_SURFACE_OVERRIDE_KEY),
            text("override").muted().small(),
            tool_surface_value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .width(Size::Fill(1.0));

        form([
            form_item([
                form_label("backends"),
                form_control(backends_row),
                form_description(
                    "Restrict the spawned thread to a subset of pod-allowed \
                     backends. Empty list = none.",
                ),
            ]),
            form_item([
                form_label("host_envs"),
                form_control(host_envs_row),
                form_description(
                    "Restrict the spawned thread to a subset of pod-allowed \
                     host envs.",
                ),
            ]),
            form_item([
                form_label("mcp_hosts"),
                form_control(mcp_hosts_row),
                form_description(
                    "Restrict the spawned thread to a subset of pod-allowed \
                     shared MCP hosts.",
                ),
            ]),
            form_item([
                form_label("tools default"),
                form_control(tools_row),
                form_description(
                    "Per-tool default for un-overridden tools. Per-tool \
                     overrides ride through Raw TOML in v1.",
                ),
            ]),
            form_item([
                form_label("pod_modify"),
                form_control(pod_modify_row),
                form_description(
                    "Cap on pod-directory writes for the spawned thread. \
                     Inherits the pod ceiling when off.",
                ),
            ]),
            form_item([
                form_label("dispatch"),
                form_control(dispatch_row),
                form_description(
                    "Cap on `dispatch_thread` calls. `none` blocks even \
                     children with strictly narrower scopes.",
                ),
            ]),
            form_item([
                form_label("behaviors"),
                form_control(behaviors_row),
                form_description(
                    "Cap on the behavior-management tools. `read` lets the \
                     thread read configs; `author_*` opens write paths.",
                ),
            ]),
            form_item([
                form_label("tool surface"),
                form_control(tool_surface_row),
                form_description(
                    "Optional override of the pod's `thread_defaults.\
                     tool_surface`. Replaces wholesale; this is a \
                     presentation knob, not a permission gate.",
                ),
            ]),
        ])
    }

    /// Helper for the Scope tab's resource-set rows (backends /
    /// host_envs / mcp_hosts). All three share the same shape: an
    /// override checkbox plus a multi-check column over the pod's
    /// matching allow list when override is on. Pulled out so the
    /// three call sites stay readable; pod-config-loading and empty
    /// states route through here too.
    fn render_behavior_editor_scope_resource_row(
        &self,
        editor: &BehaviorEditorSheetState,
        selected: Option<&[String]>,
        override_key: &str,
        group_key: &str,
        pod_options: impl Fn(&PodConfig) -> Vec<String>,
        empty_hint: &str,
    ) -> El {
        let on = selected.is_some();
        let value: El = if !on {
            paragraph("(inherit pod ceiling)").muted().small()
        } else if editor.pending_pod_get.is_some() {
            paragraph("(loading pod allow list…)").muted().small()
        } else if let Some(pod_cfg) = editor.pod_config.as_ref() {
            let names = pod_options(pod_cfg);
            if names.is_empty() {
                paragraph(empty_hint.to_string()).muted().small()
            } else {
                let options: Vec<(String, String)> =
                    names.into_iter().map(|n| (n.clone(), n)).collect();
                checkbox_column(group_key, selected.unwrap_or(&[]), options)
            }
        } else {
            paragraph("(pod config unavailable)").muted().small()
        };
        row([
            checkbox(on).key(override_key),
            text("override").muted().small(),
            value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Start)
        .width(Size::Fill(1.0))
    }

    /// Helper for the Scope tab's cap rows. Override checkbox + a
    /// `select_trigger` (when on) showing the current cap label; an
    /// inherit hint otherwise.
    fn render_behavior_editor_scope_cap_row(
        &self,
        current_label: Option<&'static str>,
        override_key: &str,
        trigger_key: &str,
    ) -> El {
        let on = current_label.is_some();
        let value: El = if let Some(lbl) = current_label {
            select_trigger(trigger_key, lbl)
        } else {
            paragraph("(inherit pod ceiling)").muted().small()
        };
        row([
            checkbox(on).key(override_key),
            text("override").muted().small(),
            value,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .width(Size::Fill(1.0))
    }

    /// Retention tab body. Mirrors the egui sibling's
    /// `render_behavior_editor_retention_tab`: a 3-way kind picker
    /// (Keep / ArchiveAfterDays / DeleteAfterDays) plus a `days`
    /// `numeric_input` that's only shown for the timed variants.
    fn render_behavior_editor_retention_tab(
        &self,
        editor: &BehaviorEditorSheetState,
        cfg: &BehaviorConfig,
    ) -> El {
        let kind_label = retention_kind_label(&cfg.on_completion);
        let kind_trigger = select_trigger(BEHAVIOR_EDITOR_RETENTION_KIND_KEY, kind_label);

        let mut items: Vec<El> = vec![form_item([
            form_label("policy"),
            form_control(kind_trigger),
            form_description(
                "Applies only to behavior-spawned threads — interactive \
                 threads always Keep. Sweep runs hourly server-side.",
            ),
        ])];

        if matches!(
            cfg.on_completion,
            RetentionPolicy::ArchiveAfterDays { .. } | RetentionPolicy::DeleteAfterDays { .. }
        ) {
            let days_widget = numeric_input(
                &editor.retention_days_buf,
                &self.selection,
                BEHAVIOR_EDITOR_RETENTION_DAYS_KEY,
                NumericInputOpts::default().min(1.0).max(3650.0).step(1.0),
            );
            let hint = match cfg.on_completion {
                RetentionPolicy::ArchiveAfterDays { .. } => {
                    "Days past `last_active` before the JSON moves to \
                     `<pod>/.archived/threads/`."
                }
                RetentionPolicy::DeleteAfterDays { .. } => {
                    "Days past `last_active` before the JSON is rm'd. \
                     No forensic copy."
                }
                _ => "",
            };
            items.push(form_item([
                form_label("days"),
                form_control(days_widget),
                form_description(hint),
            ]));
        }

        form(items)
    }

    /// SystemPrompt tab body. Mirrors the egui sibling's
    /// `render_behavior_editor_system_prompt_tab`. An override
    /// checkbox flips `cfg.thread.system_prompt` between `None`
    /// (inherit pod default) and `Some(File { name = conventional
    /// path })`. When on, a text_area binds to
    /// `working_system_prompt` (File variant) or directly to the
    /// inline text (Text variant).
    fn render_behavior_editor_system_prompt_tab(
        &self,
        editor: &BehaviorEditorSheetState,
        cfg: &BehaviorConfig,
    ) -> El {
        let conv_path = behavior_system_prompt_path(&editor.behavior_id);
        let override_on = cfg.thread.system_prompt.is_some();

        let mut items: Vec<El> = vec![form_item([
            form_label("override"),
            form_control(
                row([
                    checkbox(override_on).key(BEHAVIOR_EDITOR_SYSTEM_PROMPT_OVERRIDE_KEY),
                    text("override pod default system prompt").muted().small(),
                ])
                .gap(tokens::SPACE_2)
                .align(Align::Center),
            ),
            form_description(
                "Off = inherit the pod's default system prompt. On = use \
                 the text below as the agent-personality preamble for \
                 spawned threads. Stored as a sibling file in the \
                 behavior's directory.",
            ),
        ])];

        if !override_on {
            items.push(form_item([
                form_label("body"),
                form_control(
                    paragraph("(inheriting pod default — toggle override on to edit)")
                        .muted()
                        .small(),
                ),
            ]));
            return form(items);
        }

        match cfg.thread.system_prompt.as_ref() {
            Some(SystemPromptChoice::File { name }) => {
                let non_conv = name != &conv_path;
                let body = editor.working_system_prompt.as_deref().unwrap_or("");
                let body_input =
                    text_area(body, &self.selection, BEHAVIOR_EDITOR_SYSTEM_PROMPT_KEY)
                        .height(Size::Fixed(280.0));
                items.push(form_item([
                    form_label("file"),
                    form_control(mono(name.clone()).muted().small()),
                ]));
                if non_conv {
                    items.push(form_item([
                        form_label(""),
                        form_control(
                            paragraph(
                                "Non-conventional path — `UpdateBehavior` only writes \
                                 the conventional `behaviors/<id>/system_prompt.md`, so \
                                 edits to this body land there. Use Raw TOML to retarget \
                                 the pointer if you really want a different path.",
                            )
                            .muted()
                            .small(),
                        ),
                    ]));
                }
                items.push(form_item([
                    form_label("body"),
                    form_control(body_input),
                    form_description(
                        "Side-file content for `behaviors/<id>/system_prompt.md`. \
                         Saved alongside the config on the next Save.",
                    ),
                ]));
            }
            Some(SystemPromptChoice::Text { text: inline_text }) => {
                let inline_input = text_area(
                    inline_text,
                    &self.selection,
                    BEHAVIOR_EDITOR_SYSTEM_PROMPT_KEY,
                )
                .height(Size::Fixed(280.0));
                items.push(form_item([
                    form_label("inline"),
                    form_control(paragraph("inline text (no side file)").muted().small()),
                ]));
                items.push(form_item([
                    form_label("body"),
                    form_control(inline_input),
                    form_description(
                        "Inline override text. Edits land directly on the config — no \
                         side file involved.",
                    ),
                ]));
            }
            None => {}
        }
        form(items)
    }

    /// Raw TOML tab body. A hint paragraph above a monospace
    /// multi-line `text_area` over `working_toml`. Round-trip with
    /// the structured tabs is handled in `switch_tab`: leaving Raw
    /// with `raw_dirty` reparses; entering Raw re-serializes from
    /// the structured config so structural edits the user made
    /// since last visit show up here.
    ///
    /// The prompt text is edited in the Prompt tab, not here —
    /// `behavior.toml` doesn't carry the prompt body, only the
    /// trigger / thread / scope / retention config.
    fn render_behavior_editor_raw_tab(&self, editor: &BehaviorEditorSheetState) -> El {
        let hint = paragraph(
            "Raw `behavior.toml`. Edits here override the structured tabs on save. \
             Prompt text is edited in the Prompt tab, not here. Switching back \
             to a structured tab tries to parse this text first; a parse error \
             keeps you here so the edit isn't lost.",
        )
        .muted()
        .small();
        // Hug height (no fixed) lets the textarea grow with the
        // TOML's line count; the editor modal's outer `scroll`
        // handles overflow. Same shape the pod editor's Raw tab
        // uses. The column gets `tokens::RING_WIDTH` of horizontal
        // padding so the textarea's focus ring isn't clipped by
        // the scroll's scissor (lint catches the bare-edge case).
        let body = text_area(
            &editor.working_toml,
            &self.selection,
            BEHAVIOR_EDITOR_RAW_TOML_KEY,
        )
        .mono();
        column([hint, body])
            .gap(tokens::SPACE_2)
            .padding(Sides::xy(tokens::RING_WIDTH, 0.0))
            .width(Size::Fill(1.0))
    }

    /// `select_menu` for the behavior editor's trigger-kind picker.
    /// Three fixed options matching the wire's `TriggerSpec` tag
    /// values; rendered as a popover layer when
    /// `editor.trigger_kind_open` is set.
    fn behavior_editor_trigger_kind_menu(&self) -> El {
        let options: Vec<(String, String)> = [
            TriggerKindLabel::Manual,
            TriggerKindLabel::Cron,
            TriggerKindLabel::Webhook,
        ]
        .into_iter()
        .map(|k| (k.wire_value().to_string(), k.display_label().to_string()))
        .collect();
        select_menu(BEHAVIOR_EDITOR_TRIGGER_KIND_KEY, options)
    }

    /// Open (or re-target) the pod editor sheet for `pod_id`. Mints
    /// a correlation, fires `GetPod`, hangs the sheet on
    /// `pod_editor`. No-op if the sheet is already open for the same
    /// pod (avoids re-firing the round-trip and trampling pending
    /// edits). A different pod takes over wholesale; any in-flight
    /// snapshot for the previous pod will arrive with a stale
    /// correlation and be dropped on the floor.
    fn open_pod_editor(&mut self, pod_id: String) {
        if let Some(existing) = self.pod_editor.as_ref()
            && existing.pod_id == pod_id
        {
            return;
        }
        let correlation_id = self.next_correlation_id();
        self.pod_editor = Some(PodEditorSheetState::new(
            pod_id.clone(),
            correlation_id.clone(),
        ));
        self.send(ClientToServer::GetPod {
            correlation_id: Some(correlation_id),
            pod_id,
        });
    }

    /// Routing for the pod editor sheet. Mirrors the behavior editor
    /// shape: text_area edits, primary Save, secondary Cancel, scrim
    /// Dismiss. Returns `true` when consumed.
    fn handle_pod_editor_event(&mut self, event: &UiEvent) -> bool {
        if self.pod_editor.is_none() {
            return false;
        }

        // Tab strip: route through `tabs::apply_event`, then call
        // `switch_tab` so the structured/raw sync invariants run.
        // The widget mutates a String via the closure; we map it back
        // to a `PodEditorTab`.
        let mut next_tab: Option<String> = None;
        let _ = aetna_core::widgets::tabs::apply_event(
            &mut next_tab,
            event,
            POD_EDITOR_TABS_KEY,
            |raw| Some(Some(raw.to_string())),
        );
        if let Some(value) = next_tab
            && let Some(target) = PodEditorTab::from_wire(&value)
            && let Some(editor) = self.pod_editor.as_mut()
        {
            editor.switch_tab(target);
            return true;
        }

        // Picker triggers / menus — every editor `select_trigger` is
        // routed by its variant's [`PodEditorPicker::key`]. Match each
        // before the text-input handlers below (whose target_key
        // checks would otherwise shadow the picker option keys).
        for which in [
            PodEditorPicker::AllowCapsPodModify,
            PodEditorPicker::AllowCapsDispatch,
            PodEditorPicker::AllowCapsBehaviors,
            PodEditorPicker::DefaultsBackend,
            PodEditorPicker::DefaultsModel,
            PodEditorPicker::DefaultsToolGate,
            PodEditorPicker::DefaultsCapsPodModify,
            PodEditorPicker::DefaultsCapsDispatch,
            PodEditorPicker::DefaultsCapsBehaviors,
        ] {
            if let Some(action) = classify_select_event(event, which.key()) {
                self.handle_pod_editor_picker(which, action);
                return true;
            }
        }

        // Allow tab text fields.
        if event.target_key() == Some(POD_EDITOR_ALLOW_NAME_KEY) {
            if let Some(editor) = self.pod_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                text_input::apply_event(
                    &mut cfg.name,
                    &mut self.selection,
                    POD_EDITOR_ALLOW_NAME_KEY,
                    event,
                );
                editor.error = None;
            }
            return true;
        }
        if event.target_key() == Some(POD_EDITOR_ALLOW_DESCRIPTION_KEY) {
            if let Some(editor) = self.pod_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                let mut buf = cfg.description.clone().unwrap_or_default();
                text_area::apply_event(
                    &mut buf,
                    &mut self.selection,
                    POD_EDITOR_ALLOW_DESCRIPTION_KEY,
                    event,
                );
                cfg.description = if buf.trim().is_empty() {
                    None
                } else {
                    Some(buf)
                };
                editor.error = None;
            }
            return true;
        }

        // Allow tab multi-checks (backends / mcp_hosts / buckets).
        // Each item is a `[checkbox, label]` row keyed
        // `{group_prefix}:item:{value}` — column layout instead of
        // a non-wrapping toggle_group so narrow sheet widths don't
        // overflow.
        if let Some(editor) = self.pod_editor.as_mut()
            && let Some(cfg) = editor.working_config.as_mut()
        {
            if apply_checkbox_list_to_vec(
                &mut cfg.allow.backends,
                event,
                POD_EDITOR_ALLOW_BACKENDS_KEY,
            ) {
                editor.error = None;
                return true;
            }
            if apply_checkbox_list_to_vec(
                &mut cfg.allow.mcp_hosts,
                event,
                POD_EDITOR_ALLOW_MCP_HOSTS_KEY,
            ) {
                editor.error = None;
                return true;
            }
            if apply_checkbox_list_to_vec(
                &mut cfg.allow.knowledge_buckets,
                event,
                POD_EDITOR_ALLOW_BUCKETS_KEY,
            ) {
                editor.error = None;
                return true;
            }
        }

        // Defaults tab text fields.
        if event.target_key() == Some(POD_EDITOR_DEFAULTS_SYSTEM_PROMPT_FILE_KEY) {
            if let Some(editor) = self.pod_editor.as_mut()
                && let Some(cfg) = editor.working_config.as_mut()
            {
                text_input::apply_event(
                    &mut cfg.thread_defaults.system_prompt_file,
                    &mut self.selection,
                    POD_EDITOR_DEFAULTS_SYSTEM_PROMPT_FILE_KEY,
                    event,
                );
                editor.error = None;
            }
            return true;
        }

        // Defaults tab numeric inputs (max_tokens / max_turns). The
        // widget owns a String buffer so mid-edit states like `"1"`
        // survive between keystrokes; we mirror parseable u32 values
        // back into `working_config` so `dirty()` and the save round
        // trip stay accurate.
        if let Some(editor) = self.pod_editor.as_mut() {
            let max_tokens_opts = NumericInputOpts::default()
                .min(1.0)
                .max(200_000.0)
                .step(50.0);
            if numeric_input::apply_event(
                &mut editor.max_tokens_buf,
                &mut self.selection,
                POD_EDITOR_DEFAULTS_MAX_TOKENS_KEY,
                &max_tokens_opts,
                event,
            ) {
                if let Ok(v) = editor.max_tokens_buf.parse::<u32>()
                    && let Some(cfg) = editor.working_config.as_mut()
                {
                    cfg.thread_defaults.max_tokens = v.clamp(1, 200_000);
                }
                editor.error = None;
                return true;
            }
            let max_turns_opts = NumericInputOpts::default().min(1.0).max(10_000.0).step(1.0);
            if numeric_input::apply_event(
                &mut editor.max_turns_buf,
                &mut self.selection,
                POD_EDITOR_DEFAULTS_MAX_TURNS_KEY,
                &max_turns_opts,
                event,
            ) {
                if let Ok(v) = editor.max_turns_buf.parse::<u32>()
                    && let Some(cfg) = editor.working_config.as_mut()
                {
                    cfg.thread_defaults.max_turns = v.clamp(1, 10_000);
                }
                editor.error = None;
                return true;
            }
        }

        // Defaults tab multi-checks (host_env / default mcp_hosts).
        if let Some(editor) = self.pod_editor.as_mut()
            && let Some(cfg) = editor.working_config.as_mut()
        {
            if apply_checkbox_list_to_vec(
                &mut cfg.thread_defaults.host_env,
                event,
                POD_EDITOR_DEFAULTS_HOST_ENV_KEY,
            ) {
                editor.error = None;
                return true;
            }
            if apply_checkbox_list_to_vec(
                &mut cfg.thread_defaults.mcp_hosts,
                event,
                POD_EDITOR_DEFAULTS_MCP_HOSTS_KEY,
            ) {
                editor.error = None;
                return true;
            }
        }

        // Defaults tab tool_surface — three radio groups + one
        // text_area for the named-tools buffer when core_tools is
        // Named. The radio click for core_tools rebuilds the
        // CoreTools variant from the live buffer (so an All → Named
        // toggle commits the conventional default text without a
        // separate keystroke); the text_area's `apply_event` parses
        // and writes back on every keystroke so `dirty()` stays
        // accurate mid-edit.
        if let Some(editor) = self.pod_editor.as_mut() {
            let mut core_tools_pick: Option<String> = None;
            if radio::apply_event(
                &mut core_tools_pick,
                event,
                POD_EDITOR_DEFAULTS_TOOL_SURFACE_CORE_TOOLS_KEY,
                |raw| Some(Some(raw.to_string())),
            ) {
                if let Some(pick) = core_tools_pick.as_deref()
                    && let Some(cfg) = editor.working_config.as_mut()
                {
                    cfg.thread_defaults.tool_surface.core_tools = match pick {
                        "all" => CoreTools::All,
                        "named" => {
                            CoreTools::Named(parse_core_tools_named(&editor.tool_surface_named_buf))
                        }
                        _ => return true,
                    };
                }
                editor.error = None;
                return true;
            }
        }
        if let Some(editor) = self.pod_editor.as_mut()
            && event.target_key() == Some(POD_EDITOR_DEFAULTS_TOOL_SURFACE_CORE_TOOLS_NAMED_KEY)
        {
            text_area::apply_event(
                &mut editor.tool_surface_named_buf,
                &mut self.selection,
                POD_EDITOR_DEFAULTS_TOOL_SURFACE_CORE_TOOLS_NAMED_KEY,
                event,
            );
            if let Some(cfg) = editor.working_config.as_mut() {
                cfg.thread_defaults.tool_surface.core_tools =
                    CoreTools::Named(parse_core_tools_named(&editor.tool_surface_named_buf));
            }
            editor.error = None;
            return true;
        }
        if let Some(editor) = self.pod_editor.as_mut()
            && let Some(cfg) = editor.working_config.as_mut()
        {
            if radio::apply_event(
                &mut cfg.thread_defaults.tool_surface.initial_listing,
                event,
                POD_EDITOR_DEFAULTS_TOOL_SURFACE_INITIAL_LISTING_KEY,
                initial_listing_from_wire,
            ) {
                editor.error = None;
                return true;
            }
            if radio::apply_event(
                &mut cfg.thread_defaults.tool_surface.activation_surface,
                event,
                POD_EDITOR_DEFAULTS_TOOL_SURFACE_ACTIVATION_SURFACE_KEY,
                activation_surface_from_wire,
            ) {
                editor.error = None;
                return true;
            }
        }

        // Limits tab numeric input — `limits.max_concurrent_threads`.
        // Same buffer-then-parse-back pattern as the Defaults numeric
        // inputs above; see the comment there.
        if let Some(editor) = self.pod_editor.as_mut() {
            let opts = NumericInputOpts::default().min(1.0).max(1000.0).step(1.0);
            if numeric_input::apply_event(
                &mut editor.max_concurrent_threads_buf,
                &mut self.selection,
                POD_EDITOR_LIMITS_MAX_CONCURRENT_THREADS_KEY,
                &opts,
                event,
            ) {
                if let Ok(v) = editor.max_concurrent_threads_buf.parse::<u32>()
                    && let Some(cfg) = editor.working_config.as_mut()
                {
                    cfg.limits.max_concurrent_threads = v.clamp(1, 1000);
                }
                editor.error = None;
                return true;
            }
        }

        // Raw tab text_area.
        if event.target_key() == Some(POD_EDITOR_TOML_KEY) {
            if let Some(editor) = self.pod_editor.as_mut() {
                let before = editor.working_toml.clone();
                text_area::apply_event(
                    &mut editor.working_toml,
                    &mut self.selection,
                    POD_EDITOR_TOML_KEY,
                    event,
                );
                if editor.working_toml != before {
                    editor.raw_dirty = true;
                }
                editor.error = None;
            }
            return true;
        }

        if event.is_click_or_activate(POD_EDITOR_CANCEL_KEY)
            || event.is_click_or_activate(POD_EDITOR_DISMISS_KEY)
        {
            self.pod_editor = None;
            return true;
        }

        if event.is_click_or_activate(POD_EDITOR_SAVE_KEY) {
            self.submit_pod_editor();
            return true;
        }

        false
    }

    /// Pick handler for the pod editor's `select_trigger` family.
    /// `which` identifies the slot driving the action; on `Pick`, the
    /// value is parsed into whichever cap / disposition / catalog
    /// name the slot edits and written back into `working_config`.
    /// Toggle opens or closes the menu after closing any other open
    /// picker (single-active invariant).
    fn handle_pod_editor_picker(&mut self, which: PodEditorPicker, action: SelectAction) {
        match action {
            SelectAction::Toggle => {
                self.close_other_pickers(which.key());
                if let Some(editor) = self.pod_editor.as_mut() {
                    editor.open_picker = if editor.open_picker == Some(which) {
                        None
                    } else {
                        Some(which)
                    };
                }
            }
            SelectAction::Dismiss => {
                if let Some(editor) = self.pod_editor.as_mut() {
                    editor.open_picker = None;
                }
            }
            SelectAction::Pick(value) => {
                if let Some(editor) = self.pod_editor.as_mut()
                    && let Some(cfg) = editor.working_config.as_mut()
                {
                    match which {
                        PodEditorPicker::AllowCapsPodModify => {
                            if let Some(v) = pod_modify_cap_from_wire(&value) {
                                cfg.allow.caps.pod_modify = v;
                            }
                        }
                        PodEditorPicker::AllowCapsDispatch => {
                            if let Some(v) = dispatch_cap_from_wire(&value) {
                                cfg.allow.caps.dispatch = v;
                            }
                        }
                        PodEditorPicker::AllowCapsBehaviors => {
                            if let Some(v) = behaviors_cap_from_wire(&value) {
                                cfg.allow.caps.behaviors = v;
                            }
                        }
                        PodEditorPicker::DefaultsBackend => {
                            cfg.thread_defaults.backend = value;
                            // Backend changed — clear the model since the
                            // pre-existing string almost certainly isn't
                            // valid for the new backend. The user picks
                            // a model next; the menu only shows valid
                            // options.
                            cfg.thread_defaults.model = String::new();
                        }
                        PodEditorPicker::DefaultsModel => {
                            cfg.thread_defaults.model = value;
                        }
                        PodEditorPicker::DefaultsToolGate => {
                            if let Some(v) = tool_gate_from_wire(&value) {
                                cfg.allow.tools.default = v;
                            }
                        }
                        PodEditorPicker::DefaultsCapsPodModify => {
                            if let Some(v) = pod_modify_cap_from_wire(&value) {
                                cfg.thread_defaults.caps.pod_modify = v;
                            }
                        }
                        PodEditorPicker::DefaultsCapsDispatch => {
                            if let Some(v) = dispatch_cap_from_wire(&value) {
                                cfg.thread_defaults.caps.dispatch = v;
                            }
                        }
                        PodEditorPicker::DefaultsCapsBehaviors => {
                            if let Some(v) = behaviors_cap_from_wire(&value) {
                                cfg.thread_defaults.caps.behaviors = v;
                            }
                        }
                    }
                    editor.open_picker = None;
                    editor.error = None;
                }
            }
            _ => {}
        }
    }

    /// Validate + dispatch the pod editor's Save. Server does the
    /// heavy lifting (parse + validate); the client only short-
    /// circuits the no-op case where the buffer matches the
    /// baseline. Mints a correlation, stashes it in `pending_save`,
    /// ships `UpdatePodConfig`. Wire echo (`PodConfigUpdated` or
    /// `Error`) closes or re-enables the sheet via the inbound arm.
    fn submit_pod_editor(&mut self) {
        let Some(editor) = self.pod_editor.as_ref() else {
            return;
        };
        if editor.pending_save.is_some() {
            return;
        }
        if editor.pending_get.is_some() {
            // Snapshot still in flight — saving an empty buffer
            // would clobber the on-disk pod with whitespace. Refuse
            // until hydration completes.
            return;
        }
        if !editor.dirty() {
            self.set_pod_editor_error("no changes to save".into());
            return;
        }
        let pod_id = editor.pod_id.clone();
        let toml_text = match editor.resolved_save_toml() {
            Ok(text) => text,
            Err(msg) => {
                self.set_pod_editor_error(msg);
                return;
            }
        };
        let correlation_id = self.next_correlation_id();
        if let Some(editor) = self.pod_editor.as_mut() {
            editor.pending_save = Some(correlation_id.clone());
            editor.error = None;
        }
        self.send(ClientToServer::UpdatePodConfig {
            correlation_id: Some(correlation_id),
            pod_id,
            toml_text,
        });
    }

    fn set_pod_editor_error(&mut self, msg: String) {
        if let Some(editor) = self.pod_editor.as_mut() {
            editor.error = Some(msg);
        }
    }

    /// Render the pod editor modal if open. Same dialog + `scroll`
    /// shape as the behavior editor; the only meaningful difference
    /// is the body — one big monospace `text_area` over the raw TOML
    /// instead of a structured form. The Allow tab is structured;
    /// Defaults / Limits land in follow-up slices. RawToml is the
    /// always-available escape hatch with the full text_area.
    fn render_pod_editor_modal(&self) -> Option<El> {
        let editor = self.pod_editor.as_ref()?;

        let pod_label = self
            .pods
            .get(&editor.pod_id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| editor.pod_id.clone());
        let header = dialog_header([
            dialog_title(format!("Pod settings — {pod_label}")),
            dialog_description(
                "Edit the pod's allow lists, thread defaults, and limits. \
                 The Raw TOML tab is always available; the structured \
                 tabs round-trip through it on save.",
            ),
        ]);

        // Tab strip. Width-fill so the strip spans the dialog.
        let tab_value = editor.tab.wire_value().to_string();
        let tabs_strip = tabs_list(
            POD_EDITOR_TABS_KEY,
            &tab_value,
            [
                (
                    PodEditorTab::Allow.wire_value(),
                    PodEditorTab::Allow.label(),
                ),
                (
                    PodEditorTab::Defaults.wire_value(),
                    PodEditorTab::Defaults.label(),
                ),
                (
                    PodEditorTab::Limits.wire_value(),
                    PodEditorTab::Limits.label(),
                ),
                (
                    PodEditorTab::RawToml.wire_value(),
                    PodEditorTab::RawToml.label(),
                ),
            ],
        );

        let body: El = if editor.pending_get.is_some() {
            paragraph("loading pod…").muted()
        } else {
            match editor.tab {
                PodEditorTab::Allow => self.render_pod_editor_allow_tab(editor),
                PodEditorTab::Defaults => self.render_pod_editor_defaults_tab(editor),
                PodEditorTab::Limits => self.render_pod_editor_limits_tab(editor),
                PodEditorTab::RawToml => {
                    text_area(&editor.working_toml, &self.selection, POD_EDITOR_TOML_KEY).mono()
                }
            }
        };

        let pending_save = editor.pending_save.is_some();
        let dirty = editor.dirty();
        let mut save = button(if pending_save { "Saving…" } else { "Save" })
            .key(POD_EDITOR_SAVE_KEY)
            .primary();
        if editor.pending_get.is_some() || pending_save || !dirty {
            save = save.disabled();
        }
        let cancel = button("Cancel").key(POD_EDITOR_CANCEL_KEY);

        let mut body_children: Vec<El> = vec![tabs_strip, body];
        if let Some(err) = editor.error.as_deref() {
            body_children.push(
                alert([
                    alert_title("couldn't save pod"),
                    alert_description(err.to_string()),
                ])
                .destructive(),
            );
        }
        let scroll_body = scroll(body_children)
            .key("pod-editor:scroll")
            .gap(tokens::SPACE_4);

        let children: Vec<El> = vec![header, scroll_body, dialog_footer([cancel, save])];

        // Inline `dialog()` shape so we can override
        // `dialog_content`'s 420 px stock width: the Allow tab's
        // multi-checks and the Defaults tab's wide rows need more room.
        // Sized to match the behavior editor (720 × 640) for a
        // consistent editor footprint.
        let panel = dialog_content(children)
            .block_pointer()
            .width(Size::Fixed(720.0))
            .height(Size::Fixed(640.0));
        let layer = overlay([scrim(format!("{POD_EDITOR_KEY}:dismiss")), panel])
            .align(Align::Center)
            .justify(Justify::Center);
        Some(layer)
    }

    /// Allow tab body. Identity (name + description) + multi-checks
    /// for the three resource lists (backends / shared MCP hosts /
    /// knowledge buckets) over the server-known catalogs + cap
    /// ceilings (3 select_triggers). Host-envs editor is deferred —
    /// it needs a sandbox-entry sub-modal that's a separate slice.
    /// Per-tool overrides defer to the Raw TOML tab (the egui
    /// sibling does the same).
    fn render_pod_editor_allow_tab(&self, editor: &PodEditorSheetState) -> El {
        let Some(cfg) = editor.working_config.as_ref() else {
            return paragraph("no parsed config — fix the on-disk pod.toml from the Raw tab")
                .muted();
        };

        let name_input = text_input(&cfg.name, &self.selection, POD_EDITOR_ALLOW_NAME_KEY);
        let description_input = text_area(
            cfg.description.as_deref().unwrap_or(""),
            &self.selection,
            POD_EDITOR_ALLOW_DESCRIPTION_KEY,
        )
        .height(Size::Fixed(64.0));

        // Multi-check rows over server-known catalogs.
        let backends_widget = self.render_pod_editor_backends_check(cfg);
        let mcp_hosts_widget = self.render_pod_editor_mcp_hosts_check(cfg);
        let buckets_widget = self.render_pod_editor_buckets_check(cfg);

        // Cap select_triggers — the trigger label always shows the
        // current value; the menu rides on `popover_layers()` when
        // `editor.open_picker` matches.
        let pod_modify_trigger = select_trigger(
            POD_EDITOR_ALLOW_CAPS_POD_MODIFY_KEY,
            pod_modify_cap_label(cfg.allow.caps.pod_modify),
        );
        let dispatch_trigger = select_trigger(
            POD_EDITOR_ALLOW_CAPS_DISPATCH_KEY,
            dispatch_cap_label(cfg.allow.caps.dispatch),
        );
        let behaviors_trigger = select_trigger(
            POD_EDITOR_ALLOW_CAPS_BEHAVIORS_KEY,
            behaviors_cap_label(cfg.allow.caps.behaviors),
        );

        form([
            form_item([
                form_label("name"),
                form_control(name_input),
                form_description("Display name for the pod."),
            ]),
            form_item([
                form_label("description"),
                form_control(description_input),
                form_description("Optional — surfaced in the pod list."),
            ]),
            form_item([
                form_label("Allowed backends"),
                form_control(backends_widget),
                form_description(
                    "Threads in this pod may bind to any backend listed here. \
                     `thread_defaults.backend` must be one of these.",
                ),
            ]),
            form_item([
                form_label("Allowed shared MCP hosts"),
                form_control(mcp_hosts_widget),
                form_description(
                    "Singleton MCP hosts the pod can use. `thread_defaults.mcp_hosts` \
                     must reference these by name.",
                ),
            ]),
            form_item([
                form_label("Allowed knowledge buckets"),
                form_control(buckets_widget),
                form_description(
                    "Bucket ids that threads in this pod may query through the \
                     `knowledge_query` tool. Empty list = no buckets reachable.",
                ),
            ]),
            form_item([
                form_label("pod_modify ceiling"),
                form_control(pod_modify_trigger),
            ]),
            form_item([
                form_label("dispatch ceiling"),
                form_control(dispatch_trigger),
            ]),
            form_item([
                form_label("behaviors ceiling"),
                form_control(behaviors_trigger),
            ]),
        ])
    }

    fn render_pod_editor_backends_check(&self, cfg: &PodConfig) -> El {
        if self.backends.is_empty() {
            return paragraph("(no backends — server hasn't reported any yet)")
                .muted()
                .small();
        }
        let options: Vec<(String, String)> = self
            .backends
            .iter()
            .map(|b| (b.name.clone(), b.name.clone()))
            .collect();
        checkbox_column(POD_EDITOR_ALLOW_BACKENDS_KEY, &cfg.allow.backends, options)
    }

    fn render_pod_editor_mcp_hosts_check(&self, cfg: &PodConfig) -> El {
        if self.shared_mcp_hosts.is_empty() {
            return paragraph("(no shared MCP hosts configured server-side)")
                .muted()
                .small();
        }
        let options: Vec<(String, String)> = self
            .shared_mcp_hosts
            .iter()
            .map(|h| (h.name.clone(), h.name.clone()))
            .collect();
        checkbox_column(
            POD_EDITOR_ALLOW_MCP_HOSTS_KEY,
            &cfg.allow.mcp_hosts,
            options,
        )
    }

    fn render_pod_editor_buckets_check(&self, cfg: &PodConfig) -> El {
        if self.buckets.is_empty() {
            return paragraph("(no buckets exist on this server yet)")
                .muted()
                .small();
        }
        let options: Vec<(String, String)> = self
            .buckets
            .iter()
            .map(|b| (b.id.clone(), b.id.clone()))
            .collect();
        checkbox_column(
            POD_EDITOR_ALLOW_BUCKETS_KEY,
            &cfg.allow.knowledge_buckets,
            options,
        )
    }

    /// Defaults tab body. Mirrors the egui sibling's
    /// `render_pod_editor_defaults_tab`: backend / model select_triggers
    /// (catalog-driven), system-prompt path text input, max_tokens /
    /// max_turns numeric inputs, tool-gate default + per-cap defaults
    /// pickers, and host_env / mcp_hosts multi-checks scoped to the
    /// pod's `allow` lists. Per-tool overrides defer to the Raw tab
    /// (egui sibling does the same — they're an unbounded
    /// String → Disposition map; a structured editor would balloon
    /// the sheet).
    fn render_pod_editor_defaults_tab(&self, editor: &PodEditorSheetState) -> El {
        let Some(cfg) = editor.working_config.as_ref() else {
            return paragraph("no parsed config — fix the on-disk pod.toml from the Raw tab")
                .muted();
        };

        let backend_label = if cfg.thread_defaults.backend.is_empty() {
            "(none)".to_string()
        } else {
            cfg.thread_defaults.backend.clone()
        };
        let backend_trigger = select_trigger(POD_EDITOR_DEFAULTS_BACKEND_KEY, backend_label);

        let model_label = if cfg.thread_defaults.model.is_empty() {
            "(none)".to_string()
        } else {
            cfg.thread_defaults.model.clone()
        };
        let model_trigger = select_trigger(POD_EDITOR_DEFAULTS_MODEL_KEY, model_label);

        let system_prompt_input = text_input(
            &cfg.thread_defaults.system_prompt_file,
            &self.selection,
            POD_EDITOR_DEFAULTS_SYSTEM_PROMPT_FILE_KEY,
        );

        let max_tokens_widget = numeric_input(
            &editor.max_tokens_buf,
            &self.selection,
            POD_EDITOR_DEFAULTS_MAX_TOKENS_KEY,
            NumericInputOpts::default()
                .min(1.0)
                .max(200_000.0)
                .step(50.0),
        );
        let max_turns_widget = numeric_input(
            &editor.max_turns_buf,
            &self.selection,
            POD_EDITOR_DEFAULTS_MAX_TURNS_KEY,
            NumericInputOpts::default().min(1.0).max(10_000.0).step(1.0),
        );

        let tool_gate_trigger = select_trigger(
            POD_EDITOR_DEFAULTS_TOOL_GATE_KEY,
            tool_gate_label(cfg.allow.tools.default),
        );
        let override_count = cfg.allow.tools.overrides.len();
        let overrides_label: El = if override_count == 0 {
            paragraph("(none — edit per-tool overrides via the Raw tab)")
                .muted()
                .small()
        } else {
            paragraph(format!(
                "{override_count} override(s) — edit via the Raw tab"
            ))
            .muted()
            .small()
        };

        let host_env_widget = self.render_pod_editor_defaults_host_env_check(cfg);
        let mcp_hosts_widget = self.render_pod_editor_defaults_mcp_hosts_check(cfg);

        let pod_modify_trigger = select_trigger(
            POD_EDITOR_DEFAULTS_CAPS_POD_MODIFY_KEY,
            pod_modify_cap_label(cfg.thread_defaults.caps.pod_modify),
        );
        let dispatch_trigger = select_trigger(
            POD_EDITOR_DEFAULTS_CAPS_DISPATCH_KEY,
            dispatch_cap_label(cfg.thread_defaults.caps.dispatch),
        );
        let behaviors_trigger = select_trigger(
            POD_EDITOR_DEFAULTS_CAPS_BEHAVIORS_KEY,
            behaviors_cap_label(cfg.thread_defaults.caps.behaviors),
        );

        form([
            form_item([
                form_label("backend"),
                form_control(backend_trigger),
                form_description(
                    "Default model backend for new threads. Must be in `allow.backends`.",
                ),
            ]),
            form_item([
                form_label("model"),
                form_control(model_trigger),
                form_description(
                    "Default model id for new threads. The list filters to the picked \
                     backend's catalog.",
                ),
            ]),
            form_item([
                form_label("system prompt file"),
                form_control(system_prompt_input),
                form_description(
                    "Path relative to the pod directory. Empty = no pod-level system \
                     prompt.",
                ),
            ]),
            form_item([
                form_label("max tokens"),
                form_control(max_tokens_widget),
                form_description(
                    "Per-response output cap. Threads inherit this and can override \
                     at create-time.",
                ),
            ]),
            form_item([
                form_label("max turns"),
                form_control(max_turns_widget),
                form_description(
                    "Per-cycle assistant-turn cap. The thread halts the loop once \
                     reached.",
                ),
            ]),
            form_item([
                form_label("tool gate default"),
                form_control(tool_gate_trigger),
                form_description(
                    "Disposition for tools not listed in `allow.tools.overrides`. \
                     `allow_all` matches the legacy AutoApproveAll preset.",
                ),
            ]),
            form_item([form_label("tool overrides"), form_control(overrides_label)]),
            form_item([
                form_label("tool surface"),
                form_control(self.render_pod_editor_defaults_tool_surface(editor)),
                form_description(
                    "Pod baseline for the wire `tools:` payload + system-prompt \
                     listing + mid-conversation activation. Behaviors can wholesale-\
                     replace this via `[scope.tool_surface]`.",
                ),
            ]),
            form_item([
                form_label("default host envs"),
                form_control(host_env_widget),
                form_description(
                    "Names from `allow.host_env` that new threads bind to by default. \
                     Empty = threads run with no host-env MCPs (shared MCPs only).",
                ),
            ]),
            form_item([
                form_label("default caps — pod_modify"),
                form_control(pod_modify_trigger),
                form_description("Starting `pod_modify` cap on a new thread. Must be ≤ allow."),
            ]),
            form_item([
                form_label("default caps — dispatch"),
                form_control(dispatch_trigger),
                form_description("Starting `dispatch` cap on a new thread. Must be ≤ allow."),
            ]),
            form_item([
                form_label("default caps — behaviors"),
                form_control(behaviors_trigger),
                form_description("Starting `behaviors` cap on a new thread. Must be ≤ allow."),
            ]),
            form_item([
                form_label("default mcp hosts"),
                form_control(mcp_hosts_widget),
                form_description(
                    "Subset of `allow.mcp_hosts` new threads subscribe to by default.",
                ),
            ]),
        ])
    }

    fn render_pod_editor_defaults_host_env_check(&self, cfg: &PodConfig) -> El {
        if cfg.allow.host_env.is_empty() {
            return paragraph("(no host envs in [allow] — threads here run with shared MCPs only)")
                .muted()
                .small();
        }
        let options: Vec<(String, String)> = cfg
            .allow
            .host_env
            .iter()
            .map(|e| (e.name.clone(), e.name.clone()))
            .collect();
        checkbox_column(
            POD_EDITOR_DEFAULTS_HOST_ENV_KEY,
            &cfg.thread_defaults.host_env,
            options,
        )
    }

    fn render_pod_editor_defaults_mcp_hosts_check(&self, cfg: &PodConfig) -> El {
        if cfg.allow.mcp_hosts.is_empty() {
            return paragraph("(no shared MCP hosts in [allow])")
                .muted()
                .small();
        }
        let options: Vec<(String, String)> = cfg
            .allow
            .mcp_hosts
            .iter()
            .map(|n| (n.clone(), n.clone()))
            .collect();
        checkbox_column(
            POD_EDITOR_DEFAULTS_MCP_HOSTS_KEY,
            &cfg.thread_defaults.mcp_hosts,
            options,
        )
    }

    /// Structured editor for `thread_defaults.tool_surface`. Three
    /// sections (matching the egui sibling's
    /// `render_tool_surface_editor`):
    /// - core_tools: 2-option `radio_group` (All / Named) + a
    ///   conditional `text_area` when Named is selected.
    /// - initial_listing: 3-option `radio_group`.
    /// - activation_surface: 2-option `radio_group`.
    ///
    /// Each section's hint paragraph below the controls echoes the
    /// egui sibling's prose verbatim — it's load-bearing UX, not
    /// decoration. The named buffer lives on `PodEditorSheetState`
    /// (mid-edit states like a trailing newline must survive between
    /// keystrokes).
    fn render_pod_editor_defaults_tool_surface(&self, editor: &PodEditorSheetState) -> El {
        let Some(cfg) = editor.working_config.as_ref() else {
            return paragraph("(unavailable — config not parsed)")
                .muted()
                .small();
        };
        let surface = &cfg.thread_defaults.tool_surface;

        // core_tools radio + conditional textarea.
        let is_all = matches!(surface.core_tools, CoreTools::All);
        let core_tools_radio = radio_group(
            POD_EDITOR_DEFAULTS_TOOL_SURFACE_CORE_TOOLS_KEY,
            &if is_all { "all" } else { "named" },
            [
                ("all", "All admissible tools (\"all\")"),
                ("named", "Only these tools"),
            ],
        );
        let core_tools_section: El = if is_all {
            column([
                core_tools_radio,
                paragraph(
                    "Every admitted tool lands on the wire with full description. \
                     Pre-rework behavior — useful when the pod has few tools or \
                     the author wants no summarization.",
                )
                .muted()
                .small(),
            ])
            .gap(tokens::SPACE_2)
            .width(Size::Fill(1.0))
        } else {
            let named_input = text_area(
                &editor.tool_surface_named_buf,
                &self.selection,
                POD_EDITOR_DEFAULTS_TOOL_SURFACE_CORE_TOOLS_NAMED_KEY,
            )
            .height(Size::Fixed(96.0))
            .mono();
            column([
                core_tools_radio,
                paragraph(
                    "One name per line. Listed tools keep full descriptions; \
                     every other admitted tool carries a first-line-only \
                     summary. Names not admitted to the thread are silently \
                     dropped.",
                )
                .muted()
                .small(),
                named_input,
            ])
            .gap(tokens::SPACE_2)
            .width(Size::Fill(1.0))
        };

        let initial_listing_radio = radio_group(
            POD_EDITOR_DEFAULTS_TOOL_SURFACE_INITIAL_LISTING_KEY,
            &initial_listing_label(surface.initial_listing),
            [
                ("none", "None (discovery-first)"),
                ("all_names", "All names"),
                ("core_only", "Core only + counts"),
            ],
        );
        let initial_listing_section = column([
            initial_listing_radio,
            paragraph(
                "What gets appended to the system prompt at thread seed. \
                 `All names` lists every admissible tool (plus a trailing \
                 escalation-available section). `Core only` shows counts \
                 per group. `None` requires `find_tool`.",
            )
            .muted()
            .small(),
        ])
        .gap(tokens::SPACE_2)
        .width(Size::Fill(1.0));

        let activation_surface_radio = radio_group(
            POD_EDITOR_DEFAULTS_TOOL_SURFACE_ACTIVATION_SURFACE_KEY,
            &activation_surface_label(surface.activation_surface),
            [
                ("announce", "Announce names (default)"),
                ("inject_schema", "Inject schemas"),
            ],
        );
        let activation_surface_section = column([
            activation_surface_radio,
            paragraph(
                "Mid-conversation tool addition (escalation, late MCP \
                 attach). `Announce` lists names — the model fetches \
                 schemas via `describe_tool`. `Inject schemas` inlines \
                 them, trading tokens for zero round-trips.",
            )
            .muted()
            .small(),
        ])
        .gap(tokens::SPACE_2)
        .width(Size::Fill(1.0));

        column([
            text("Wire `tools:` core set").label(),
            core_tools_section,
            text("System-prompt listing").label(),
            initial_listing_section,
            text("Mid-conversation activation").label(),
            activation_surface_section,
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
    }

    /// Limits tab body. Pod-level resource ceilings — currently a
    /// single `max_concurrent_threads` knob, since the protocol's
    /// [`PodLimits`] struct only carries that one field. As the
    /// schema grows (rate caps, spend ceilings, …) new form items
    /// drop in alongside.
    fn render_pod_editor_limits_tab(&self, editor: &PodEditorSheetState) -> El {
        let Some(_cfg) = editor.working_config.as_ref() else {
            return paragraph("no parsed config — fix the on-disk pod.toml from the Raw tab")
                .muted();
        };

        let max_concurrent_widget = numeric_input(
            &editor.max_concurrent_threads_buf,
            &self.selection,
            POD_EDITOR_LIMITS_MAX_CONCURRENT_THREADS_KEY,
            NumericInputOpts::default().min(1.0).max(1000.0).step(1.0),
        );

        form([form_item([
            form_label("max concurrent threads"),
            form_control(max_concurrent_widget),
            form_description(
                "Ceiling on simultaneously-running threads in this pod. New threads \
                 above the cap queue rather than dispatch.",
            ),
        ])])
    }

    /// Build the select_menu for whichever pod-editor picker is open.
    /// Single-active — at most one picker open at a time. The menu
    /// options match the underlying cap / disposition enum, except
    /// for backend / model which list the hydrated catalogs.
    fn pod_editor_picker_menu(&self, which: PodEditorPicker) -> El {
        use whisper_agent_protocol::{BehaviorOpsCap, DispatchCap, PodModifyCap};
        let editor = self.pod_editor.as_ref();
        let cfg = editor.and_then(|e| e.working_config.as_ref());
        match which {
            PodEditorPicker::AllowCapsPodModify | PodEditorPicker::DefaultsCapsPodModify => {
                let options: Vec<(String, String)> = [
                    PodModifyCap::None,
                    PodModifyCap::Memories,
                    PodModifyCap::Content,
                    PodModifyCap::ModifyAllow,
                ]
                .into_iter()
                .map(|c| {
                    let lbl = pod_modify_cap_label(c);
                    (lbl.to_string(), lbl.to_string())
                })
                .collect();
                select_menu(which.key(), options)
            }
            PodEditorPicker::AllowCapsDispatch | PodEditorPicker::DefaultsCapsDispatch => {
                let options: Vec<(String, String)> = [DispatchCap::None, DispatchCap::WithinScope]
                    .into_iter()
                    .map(|c| {
                        let lbl = dispatch_cap_label(c);
                        (lbl.to_string(), lbl.to_string())
                    })
                    .collect();
                select_menu(which.key(), options)
            }
            PodEditorPicker::AllowCapsBehaviors | PodEditorPicker::DefaultsCapsBehaviors => {
                let options: Vec<(String, String)> = [
                    BehaviorOpsCap::None,
                    BehaviorOpsCap::Read,
                    BehaviorOpsCap::AuthorNarrower,
                    BehaviorOpsCap::AuthorAny,
                ]
                .into_iter()
                .map(|c| {
                    let lbl = behaviors_cap_label(c);
                    (lbl.to_string(), lbl.to_string())
                })
                .collect();
                select_menu(which.key(), options)
            }
            PodEditorPicker::DefaultsBackend => {
                // Mirror the egui sibling: the menu offers every
                // entry in `allow.backends` first, then the rest of
                // the server-known catalog as "not-in-allow" picks
                // (so a user can preview names server-side without
                // needing to leave the editor). The catalog branch's
                // labels carry no special styling here — aetna's
                // current select_menu is text-only — but the option
                // value still round-trips through the same wire.
                let mut options: Vec<(String, String)> = Vec::new();
                if let Some(cfg) = cfg {
                    for name in &cfg.allow.backends {
                        options.push((name.clone(), name.clone()));
                    }
                    for b in &self.backends {
                        if !cfg.allow.backends.iter().any(|x| x == &b.name) {
                            options.push((b.name.clone(), format!("{} (not in allow)", b.name)));
                        }
                    }
                } else {
                    for b in &self.backends {
                        options.push((b.name.clone(), b.name.clone()));
                    }
                }
                select_menu(which.key(), options)
            }
            PodEditorPicker::DefaultsModel => {
                // Models are scoped by the currently-picked backend.
                // Empty list → render a single sentinel option so the
                // popover isn't blank. The menu is hover-coalesced via
                // its key so it pops up anchored under the trigger.
                let mut options: Vec<(String, String)> = Vec::new();
                if let Some(cfg) = cfg {
                    let backend = &cfg.thread_defaults.backend;
                    if let Some(models) = self.models_by_backend.get(backend) {
                        for m in models {
                            options.push((m.id.clone(), m.id.clone()));
                        }
                    }
                }
                if options.is_empty() {
                    options.push((String::new(), "(no models)".to_string()));
                }
                select_menu(which.key(), options)
            }
            PodEditorPicker::DefaultsToolGate => {
                let options: Vec<(String, String)> = [Disposition::Allow, Disposition::Deny]
                    .into_iter()
                    .map(|d| {
                        let lbl = tool_gate_label(d);
                        (lbl.to_string(), lbl.to_string())
                    })
                    .collect();
                select_menu(which.key(), options)
            }
        }
    }

    /// Open the fork-thread dialog for the current thread's
    /// `msg_index`-th message. Pulls the seed text out of the
    /// matching User display item — the dialog uses it to label the
    /// confirm and the wire round-trip uses it to seed the new
    /// thread's draft. No-op if no thread is selected (the fork
    /// affordance is only rendered on a selected thread's User
    /// rows, but defend against the race anyway) or if the
    /// `msg_index` doesn't resolve to a User item in the live view
    /// (e.g. the snapshot rolled while the click was in flight).
    fn open_fork_modal(&mut self, msg_index: usize) {
        let Some(thread_id) = self.selected.clone() else {
            return;
        };
        let Some(view) = self.views.get(&thread_id) else {
            return;
        };
        let Some(seed_text) = view.items.iter().find_map(|it| match it {
            DisplayItem::User { text, msg_index: m } if *m == msg_index => Some(text.clone()),
            _ => None,
        }) else {
            return;
        };
        // Archive-by-default per the egui sibling: a fork is almost
        // always "I want to try something different from here," so
        // the default leaves the original off the sidebar list.
        self.fork_modal = Some(ForkModalState {
            thread_id,
            from_message_index: msg_index,
            seed_text,
            archive_original: true,
            reset_capabilities: false,
        });
    }

    /// Routing for the fork dialog. Switch toggles flip in place
    /// (no wire round-trip until confirm); cancel / dismiss close;
    /// confirm fires `ClientToServer::ForkThread` and stamps
    /// `pending_fork_seed` so the resulting `ThreadCreated` echo
    /// can seed the new thread's draft. Returns `true` when
    /// consumed.
    fn handle_fork_modal_event(&mut self, event: &UiEvent) -> bool {
        if self.fork_modal.is_none() {
            return false;
        }

        // Switches use `widgets::switch` — its `apply_event` flips
        // the bool when the routed click lands on the switch's key.
        if let Some(modal) = self.fork_modal.as_mut() {
            switch::apply_event(&mut modal.archive_original, event, FORK_MODAL_ARCHIVE_KEY);
            switch::apply_event(
                &mut modal.reset_capabilities,
                event,
                FORK_MODAL_RESET_CAPS_KEY,
            );
        }

        if event.is_click_or_activate(FORK_MODAL_CANCEL_KEY)
            || event.is_click_or_activate(FORK_MODAL_DISMISS_KEY)
        {
            self.fork_modal = None;
            return true;
        }

        if event.is_click_or_activate(FORK_MODAL_CONFIRM_KEY) {
            self.confirm_fork_modal();
            return true;
        }

        // Switch clicks land on their own keys; consume those so we
        // don't fall through to other handlers.
        if event.target_key() == Some(FORK_MODAL_ARCHIVE_KEY)
            || event.target_key() == Some(FORK_MODAL_RESET_CAPS_KEY)
        {
            return true;
        }

        false
    }

    /// Fire `ForkThread` and stash the correlation + seed text so
    /// the matching `ThreadCreated` arm can seed the new thread's
    /// draft. Closes the modal immediately (server-side fork is
    /// quick on loopback; if it fails the `Error` arm surfaces a
    /// thread-scoped error rather than a modal-side one — the modal
    /// state is gone by then).
    fn confirm_fork_modal(&mut self) {
        let Some(modal) = self.fork_modal.take() else {
            return;
        };
        let correlation_id = self.next_correlation_id();
        self.pending_fork_seed = Some((correlation_id.clone(), modal.seed_text));
        self.send(ClientToServer::ForkThread {
            correlation_id: Some(correlation_id),
            thread_id: modal.thread_id,
            from_message_index: modal.from_message_index,
            archive_original: modal.archive_original,
            reset_capabilities: modal.reset_capabilities,
        });
    }

    /// Render the fork dialog if open. Same shadcn-shaped scaffold
    /// the create modals use: `dialog([dialog_header, body,
    /// dialog_footer])`. Body composes the explainer text + two
    /// `switch`-shaped form items (Archive original, Reset
    /// capabilities) with form_descriptions clarifying the
    /// non-obvious choice.
    /// Optimistically resolve a pending sudo: drop the local entry
    /// (so the banner clears immediately) and ship `ResolveSudo`. The
    /// server's `SudoResolved` echo lands later as a no-op since the
    /// slot is already gone. Mirrors the egui sibling's optimistic
    /// drop pattern — keeping the banner mounted through the round-
    /// trip flickered noticeably on the approve path, where the
    /// thread's next turn fires immediately.
    fn resolve_sudo(&mut self, function_id: u64, decision: SudoDecision, reason: Option<String>) {
        self.pending_sudos.remove(&function_id);
        self.sudo_reject_drafts.remove(&function_id);
        self.send(ClientToServer::ResolveSudo {
            function_id,
            decision,
            reason,
        });
    }

    /// Render one approval banner per pending sudo whose `thread_id`
    /// matches `thread_id`. Returns `None` when there are no banners
    /// to surface — caller can short-circuit the section. Each banner
    /// shows: tool name, pretty-printed args, model justification,
    /// and three actions (Approve once / Remember / Reject) with an
    /// optional reject-reason text input.
    fn render_pending_sudo_banners(&self, thread_id: &str) -> Option<El> {
        let mut ids: Vec<u64> = self
            .pending_sudos
            .iter()
            .filter_map(|(id, s)| (s.thread_id == thread_id).then_some(*id))
            .collect();
        if ids.is_empty() {
            return None;
        }
        ids.sort_unstable();

        let mut banners: Vec<El> = Vec::with_capacity(ids.len());
        for fn_id in ids {
            let Some(s) = self.pending_sudos.get(&fn_id) else {
                continue;
            };
            banners.push(self.sudo_banner(fn_id, s));
        }
        Some(
            column(banners)
                .gap(tokens::SPACE_2)
                .padding(Sides::xy(tokens::SPACE_4, tokens::SPACE_2))
                .width(Size::Fill(1.0)),
        )
    }

    /// One sudo-approval banner. Built as an `alert(...).warning()`
    /// so it visually announces "model is asking to do something"
    /// without the destructive red used for failures. The body is
    /// a column carrying the tool name + pretty-printed args + the
    /// model's justification + the approve/remember/reject row + an
    /// optional reject-reason text input.
    fn sudo_banner(&self, fn_id: u64, s: &PendingSudoState) -> El {
        let args_pretty =
            serde_json::to_string_pretty(&s.args).unwrap_or_else(|_| s.args.to_string());

        let approve = button("Approve").key(sudo_approve_key(fn_id)).primary();
        let remember = button("Remember").key(sudo_remember_key(fn_id));
        let reject = button("Reject").key(sudo_reject_key(fn_id)).destructive();
        let reason_key = sudo_reject_reason_key(fn_id);
        let reason_buf = self
            .sudo_reject_drafts
            .get(&fn_id)
            .cloned()
            .unwrap_or_default();
        let reason_input = text_input_with(
            &reason_buf,
            &self.selection,
            &reason_key,
            TextInputOpts::default().placeholder("reject reason (optional)"),
        );

        let body = column([
            text(format!("tool: {}", s.tool_name)).code(),
            text(args_pretty).code().small(),
            text(if s.reason.trim().is_empty() {
                String::new()
            } else {
                format!("reason: {}", s.reason)
            })
            .small()
            .muted(),
            row([approve, remember, reject])
                .gap(tokens::SPACE_2)
                .align(Align::Center),
            reason_input,
        ])
        .gap(tokens::SPACE_2)
        .width(Size::Fill(1.0));

        alert([
            alert_title(format!("sudo requested · fn #{fn_id}")),
            alert_description(""),
            body,
        ])
        .warning()
    }

    /// Promote the inline `DisplayItem::Image` at `idx` into the
    /// lightbox slot. No-op when the active view's item at `idx`
    /// isn't an Image, or its state isn't Decoded — the click
    /// affordance is only attached to Decoded rows in the first
    /// place, but the guards defend against a snapshot rolling
    /// between build and click.
    fn open_lightbox(&mut self, idx: usize) {
        let Some(thread_id) = self.selected.as_deref() else {
            return;
        };
        let Some(view) = self.views.get(thread_id) else {
            return;
        };
        let Some(item) = view.items.get(idx) else {
            return;
        };
        if let DisplayItem::Image {
            state:
                ImageRenderState::Decoded {
                    image,
                    width,
                    height,
                },
            ..
        } = item
        {
            self.lightbox = Some(LightboxState {
                image: image.clone(),
                width: *width,
                height: *height,
            });
        }
    }

    /// Fullscreen image-lightbox modal. The default `dialog(...)`
    /// helper hard-codes `width = Fixed(420)` on its modal panel,
    /// which would clip a wide screenshot to a narrow column — so we
    /// reach for `dialog_content(...)` directly (still gets the
    /// shadcn-shaped surface chrome, `Popover` role, padding, gap,
    /// shadow) and override the width to `Hug` so it sizes to the
    /// image's display dimensions instead. The `overlay + scrim`
    /// frame is the same shape `dialog(...)` would have built.
    fn render_lightbox_modal(&self) -> Option<El> {
        let lb = self.lightbox.as_ref()?;
        // Cap larger than the inline `MAX_DISPLAY_HEIGHT` so the
        // lightbox is meaningfully bigger than the in-row preview;
        // 720 fits comfortably inside a 1080-tall viewport once the
        // dialog chrome + caption are accounted for.
        const LIGHTBOX_MAX_W: f32 = 1100.0;
        const LIGHTBOX_MAX_H: f32 = 720.0;
        let (w, h) = lightbox_display_dims(lb.width, lb.height, LIGHTBOX_MAX_W, LIGHTBOX_MAX_H);

        let body = column([
            image(lb.image.clone())
                .image_fit(ImageFit::Contain)
                .width(Size::Fixed(w))
                .height(Size::Fixed(h))
                .radius(tokens::RADIUS_SM),
            text(format!("{} \u{00D7} {}", lb.width, lb.height))
                .caption()
                .muted(),
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .width(Size::Hug);

        let close = button("Close").key(LIGHTBOX_MODAL_CLOSE_KEY);
        let children: Vec<El> = vec![body, dialog_footer([close])];

        Some(overlay([
            scrim(LIGHTBOX_MODAL_DISMISS_KEY),
            dialog_content(children).width(Size::Hug).block_pointer(),
        ]))
    }

    /// Open the generic text-editor modal on `<pod_id>/<path>`. Mints
    /// a correlation, stamps the slot, and fires `ReadPodFile`; the
    /// matching `PodFileContent` arm hydrates the working buffer,
    /// the saved baseline, and the read-only flag. Mirrors the egui
    /// sibling's `open_file_viewer`. Public so the future file-tree
    /// slice can route generic-file clicks here directly.
    pub fn open_file_viewer(&mut self, pod_id: String, path: String) {
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

    /// Save the file viewer's working buffer. Mints a fresh
    /// correlation, ships `WritePodFile`, and stamps the modal so the
    /// matching `PodFileWritten` arm clears the pending state. No-ops
    /// when the buffer hasn't loaded, is unchanged, is read-only, or
    /// a save is already in flight — the rendered button is disabled
    /// in those cases but a stray keyboard route still routes here.
    fn submit_file_viewer(&mut self) {
        let Some(modal) = self.file_viewer_modal.as_mut() else {
            return;
        };
        if modal.readonly || modal.pending_correlation.is_some() || !modal.is_dirty() {
            return;
        }
        let Some(content) = modal.working.clone() else {
            return;
        };
        let pod_id = modal.pod_id.clone();
        let path = modal.path.clone();
        let correlation = self.next_correlation_id();
        if let Some(modal) = self.file_viewer_modal.as_mut() {
            modal.pending_correlation = Some(correlation.clone());
            modal.error = None;
        }
        self.send(ClientToServer::WritePodFile {
            correlation_id: Some(correlation),
            pod_id,
            path,
            content,
        });
    }

    /// Edit-with-save modal over one pod file. Same `dialog_content`
    /// shape as the JSON viewer; differs in the body (a single
    /// `text_area` over `working`) and the footer (Save / Revert
    /// when editable, Close-only when read-only). Pending reads
    /// render "loading…"; an in-flight save renders the same panel
    /// with a "Saving…" label on the primary button.
    fn render_file_viewer_modal(&self) -> Option<El> {
        let modal = self.file_viewer_modal.as_ref()?;
        const FILE_VIEWER_W: f32 = 720.0;
        const FILE_VIEWER_H: f32 = 560.0;

        let title = format!("{} — {}", modal.path, modal.pod_id);
        let has_data = modal.working.is_some();
        let dirty = modal.is_dirty();
        let saving = modal.pending_correlation.is_some() && has_data;

        let body: El = if let Some(working) = modal.working.as_ref() {
            text_area(working, &self.selection, FILE_VIEWER_BODY_KEY)
                .mono()
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0))
        } else if let Some(err) = modal.error.as_deref() {
            paragraph(err)
                .color(tokens::DESTRUCTIVE)
                .width(Size::Fill(1.0))
        } else {
            paragraph("loading\u{2026}").muted().width(Size::Fill(1.0))
        };

        let description = if modal.readonly {
            "Read-only — runtime state owned by the scheduler"
        } else {
            "Edits write back through WritePodFile on Save"
        };

        let mut footer_children: Vec<El> = Vec::new();
        footer_children.push(button("Close").key(FILE_VIEWER_CLOSE_KEY));
        if !modal.readonly {
            let mut revert = button("Revert").key(FILE_VIEWER_REVERT_KEY);
            if !dirty || saving {
                revert = revert.disabled();
            }
            footer_children.push(revert);
            let mut save = button(if saving { "Saving\u{2026}" } else { "Save" })
                .key(FILE_VIEWER_SAVE_KEY)
                .primary();
            if !has_data || !dirty || saving {
                save = save.disabled();
            }
            footer_children.push(save);
        }

        let mut body_children: Vec<El> = vec![body];
        if let Some(err) = modal.error.as_deref()
            && has_data
        {
            body_children.push(
                alert([
                    alert_title("couldn't save file"),
                    alert_description(err.to_string()),
                ])
                .destructive(),
            );
        }

        let children: Vec<El> = vec![
            dialog_header([dialog_title(title), dialog_description(description)]),
            column(body_children)
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0)),
            dialog_footer(footer_children),
        ];

        Some(overlay([
            scrim(FILE_VIEWER_DISMISS_KEY),
            dialog_content(children)
                .width(Size::Fixed(FILE_VIEWER_W))
                .height(Size::Fixed(FILE_VIEWER_H))
                .block_pointer(),
        ]))
    }

    /// Open the server-settings modal. Idempotent: re-opening keeps
    /// the active tab + any in-flight server-config edit alive (the
    /// user likely wants to switch tabs and come back). The
    /// Server-config tab's lazy fetch is deferred to the tab-switch
    /// handler so that a fresh open on the Backends tab doesn't
    /// fire an admin-only request the operator might not have
    /// privileges for.
    pub fn open_settings_modal(&mut self) {
        if self.settings_modal.is_none() {
            self.settings_modal = Some(SettingsModalState::default());
        }
    }

    /// Open the knowledge-buckets modal. Idempotent. Bucket-list
    /// data already lives on `self.buckets` (hydrated by
    /// `BucketsList` at connect), so no fetch is fired here — the
    /// modal renders whatever the catalog already shows. Live build
    /// progress arrives via `BucketBuildStarted` /
    /// `BucketBuildProgress` whether the modal is open or closed;
    /// keeping the maps inside `BucketsModalState` means closing
    /// the modal drops them, which is fine for v1 (the modal is
    /// the only consumer).
    pub fn open_buckets_modal(&mut self) {
        if self.buckets_modal.is_none() {
            self.buckets_modal = Some(BucketsModalState::default());
        }
    }

    /// Fire `StartBucketBuild`. The server picks up the work and
    /// broadcasts `BucketBuildStarted`; that arm hydrates the
    /// per-row progress entry. Idempotent at the server: clicking
    /// twice when a build is already in flight is a no-op.
    fn start_bucket_build(&mut self, id: String, pod_id: Option<String>) {
        self.send(ClientToServer::StartBucketBuild {
            correlation_id: None,
            id,
            pod_id,
        });
    }

    /// Fire `CancelBucketBuild`. The server stops the build worker
    /// and broadcasts `BucketBuildEnded { outcome: Cancelled }`.
    /// Build state on disk is preserved (server resumes from
    /// `build.state` on the next Build click).
    fn cancel_bucket_build(&mut self, id: String, pod_id: Option<String>) {
        self.send(ClientToServer::CancelBucketBuild {
            correlation_id: None,
            id,
            pod_id,
        });
    }

    /// Fire `DeleteBucket`. The server validates + broadcasts
    /// `BucketDeleted`; the wire arm removes the row + clears any
    /// per-row state. Two-click arm-confirm gates this call site.
    fn delete_bucket(&mut self, id: String, pod_id: Option<String>) {
        self.send(ClientToServer::DeleteBucket {
            correlation_id: None,
            id,
            pod_id,
        });
    }

    /// Fire `PollFeedNow` for a tracked bucket. The server's
    /// trigger channel is bounded at 1 so multiple rapid clicks
    /// coalesce server-side; we fire-and-forget.
    fn poll_bucket_feed(&mut self, id: String, pod_id: Option<String>) {
        self.send(ClientToServer::PollFeedNow {
            correlation_id: None,
            id,
            pod_id,
        });
    }

    /// Fire `ResyncBucket`. Server short-circuits if the recorded
    /// base is already at latest, so a stray click is cheap.
    fn resync_bucket(&mut self, id: String, pod_id: Option<String>) {
        self.send(ClientToServer::ResyncBucket {
            correlation_id: None,
            id,
            pod_id,
        });
    }

    /// Lazy-fetch the server config TOML on first Server-config tab
    /// open. Idempotent: a populated `original` or in-flight
    /// correlation short-circuits.
    fn ensure_server_config_fetched(&mut self) {
        let needs_fetch = match self.settings_modal.as_ref() {
            Some(modal) => match modal.server_config.as_ref() {
                Some(editor) => editor.original.is_none() && editor.fetch_correlation.is_none(),
                None => true,
            },
            None => return,
        };
        if !needs_fetch {
            return;
        }
        let correlation = self.next_correlation_id();
        if let Some(modal) = self.settings_modal.as_mut() {
            let editor = modal
                .server_config
                .get_or_insert_with(ServerConfigEditorState::new);
            editor.fetch_correlation = Some(correlation.clone());
        }
        self.send(ClientToServer::FetchServerConfig {
            correlation_id: Some(correlation),
        });
    }

    /// Save the server-config working buffer. No-ops while a save is
    /// already in flight, while a fetch hasn't landed, or when the
    /// buffer is unchanged. Mints a fresh correlation; the matching
    /// `ServerConfigUpdateResult` arm populates `save_summary` and
    /// adopts the working buffer as the new baseline.
    fn submit_server_config(&mut self) {
        // Phase 1: read the snapshot we'd ship and validate gates.
        // Phase 2: mint a correlation (re-borrows `self`).
        // Phase 3: stamp the editor and send.
        let toml_text = match self
            .settings_modal
            .as_ref()
            .and_then(|m| m.server_config.as_ref())
        {
            Some(editor) if editor.save_correlation.is_none() && editor.dirty() => {
                editor.working.clone()
            }
            _ => return,
        };
        let correlation = self.next_correlation_id();
        if let Some(modal) = self.settings_modal.as_mut()
            && let Some(editor) = modal.server_config.as_mut()
        {
            editor.save_correlation = Some(correlation.clone());
            editor.save_summary = None;
            editor.error = None;
        }
        self.send(ClientToServer::UpdateServerConfig {
            correlation_id: Some(correlation),
            toml_text,
        });
    }

    /// Server-settings modal renderer. Centered `dialog_content`
    /// (720 × 640, matching the pod / behavior editors) over a tab
    /// strip + per-tab body + Close footer. Save / Revert affordances
    /// for the Server-config tab live in the body, not the dialog
    /// footer, since they're tab-scoped — switching to Backends
    /// shouldn't leave a Save button hanging at the bottom of the
    /// dialog with no live edit to ship.
    fn render_settings_modal(&self) -> Option<El> {
        let modal = self.settings_modal.as_ref()?;
        const SETTINGS_W: f32 = 720.0;
        const SETTINGS_H: f32 = 640.0;

        let tab_value = modal.active_tab.wire_value().to_string();
        let tabs_strip = tabs_list(
            SETTINGS_TABS_KEY,
            &tab_value,
            [
                (
                    SettingsTab::Backends.wire_value(),
                    SettingsTab::Backends.label(),
                ),
                (
                    SettingsTab::ServerConfig.wire_value(),
                    SettingsTab::ServerConfig.label(),
                ),
            ],
        );

        let body: El = match modal.active_tab {
            SettingsTab::Backends => self.render_settings_backends_tab(),
            SettingsTab::ServerConfig => self.render_settings_server_config_tab(modal),
        };

        let close = button("Close").key(SETTINGS_CLOSE_KEY);

        let children: Vec<El> = vec![
            dialog_header([
                dialog_title("Server settings"),
                dialog_description(
                    "LLM backends are read-only here; edit the underlying \
                     whisper-agent.toml on the Server config tab. Shared MCP \
                     hosts CRUD and Codex auth rotation land in a follow-up.",
                ),
            ]),
            tabs_strip,
            body,
            dialog_footer([close]),
        ];

        Some(overlay([
            scrim(SETTINGS_DISMISS_KEY),
            dialog_content(children)
                .width(Size::Fixed(SETTINGS_W))
                .height(Size::Fixed(SETTINGS_H))
                .block_pointer(),
        ]))
    }

    /// Backends tab — read-only catalog of configured LLM backends.
    /// One card per backend: alias + kind, then the default model
    /// and auth-mode badges (or a muted placeholder when the
    /// backend declares neither). Codex rotation lives behind a
    /// follow-up sub-form; nothing on this tab is wire-actionable
    /// yet.
    fn render_settings_backends_tab(&self) -> El {
        if self.backends.is_empty() {
            return paragraph(
                "No backends configured. Seed them via [backends.*] in \
                 whisper-agent.toml on the Server config tab.",
            )
            .muted();
        }
        let mut rows: Vec<El> = Vec::new();
        for b in &self.backends {
            let mut chips: Vec<El> = Vec::new();
            if let Some(model) = b.default_model.as_deref() {
                chips.push(text(format!("default: {model}")).caption().muted());
            }
            if let Some(auth) = b.auth_mode.as_deref() {
                chips.push(text(format!("auth: {auth}")).caption().muted());
            }
            let chip_row: El = if chips.is_empty() {
                text("(no default model / auth mode declared)")
                    .caption()
                    .muted()
            } else {
                row(chips).gap(tokens::SPACE_3).align(Align::Center)
            };
            rows.push(
                card([
                    row([
                        text(b.name.clone()).label().bold(),
                        text(format!("\u{00B7} {}", b.kind)).caption().muted(),
                    ])
                    .gap(tokens::SPACE_2)
                    .align(Align::Center),
                    chip_row,
                ])
                .gap(tokens::SPACE_1)
                .width(Size::Fill(1.0)),
            );
        }
        scroll(rows)
            .gap(tokens::SPACE_2)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
    }

    /// Server-config tab — admin-only raw TOML editor over
    /// `whisper-agent.toml`. Lazy-loaded on first tab open; renders
    /// a "loading…" placeholder while the fetch is in flight.
    /// Save / Revert sit directly above the text_area (tab-scoped).
    fn render_settings_server_config_tab(&self, modal: &SettingsModalState) -> El {
        let editor = match modal.server_config.as_ref() {
            Some(e) => e,
            None => {
                return paragraph(
                    "Server config tab not yet primed — switch back to it once \
                     the fetch lands.",
                )
                .muted();
            }
        };
        if editor.fetch_correlation.is_some() && editor.original.is_none() {
            return paragraph("loading whisper-agent.toml\u{2026}").muted();
        }
        if editor.original.is_none()
            && let Some(err) = editor.error.as_deref()
        {
            return paragraph(err).color(tokens::DESTRUCTIVE);
        }
        let dirty = editor.dirty();
        let saving = editor.save_correlation.is_some();

        let body = text_area(
            &editor.working,
            &self.selection,
            SETTINGS_SERVER_CONFIG_BODY_KEY,
        )
        .mono()
        .width(Size::Fill(1.0))
        .height(Size::Fill(1.0));

        let mut save = button(if saving { "Saving\u{2026}" } else { "Save" })
            .key(SETTINGS_SERVER_CONFIG_SAVE_KEY)
            .primary();
        if !dirty || saving {
            save = save.disabled();
        }
        let mut revert = button("Revert").key(SETTINGS_SERVER_CONFIG_REVERT_KEY);
        if !dirty || saving {
            revert = revert.disabled();
        }

        let mut entries: Vec<El> = Vec::new();
        if let Some(summary) = editor.save_summary.as_ref() {
            entries.push(self.render_server_config_summary(summary));
        }
        if let Some(err) = editor.error.as_deref() {
            entries.push(
                alert([
                    alert_title("couldn't save server config"),
                    alert_description(err.to_string()),
                ])
                .destructive(),
            );
        }
        entries.push(body);
        entries.push(
            row([revert, save])
                .gap(tokens::SPACE_2)
                .align(Align::Center),
        );

        column(entries)
            .gap(tokens::SPACE_3)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
    }

    fn render_server_config_summary(&self, s: &ServerConfigSaveSummary) -> El {
        let mut lines: Vec<El> = vec![text("Saved.").bold()];
        if !s.cancelled_threads.is_empty() {
            lines.push(text(format!(
                "Cancelled {} thread(s): {}",
                s.cancelled_threads.len(),
                s.cancelled_threads.join(", "),
            )));
        }
        if !s.restart_required_sections.is_empty() {
            lines.push(text(format!(
                "Restart required for: {}",
                s.restart_required_sections.join(", "),
            )));
        }
        if !s.pods_with_missing_backends.is_empty() {
            lines.push(text(format!(
                "Pods referencing removed backends: {}",
                s.pods_with_missing_backends.join(", "),
            )));
        }
        alert(lines)
    }

    /// Knowledge-buckets modal renderer. Centered `dialog_content`
    /// (720 × 640, matching the other editor modals) over a scrolled
    /// column of per-bucket cards. Each card shows the catalog
    /// metadata (name + id, description, scope / source / embedder
    /// chips, slot info when present), live build progress when one
    /// is in flight, the last build error if any, and a row of
    /// action buttons. Create form + search-and-query land in
    /// follow-up sub-slices.
    fn render_buckets_modal(&self) -> Option<El> {
        let modal = self.buckets_modal.as_ref()?;
        const BUCKETS_W: f32 = 720.0;
        const BUCKETS_H: f32 = 640.0;

        let body: El = if self.buckets.is_empty() {
            paragraph(
                "No buckets yet. Create one via the wire (server-side \
                 catalog config or a future +New bucket affordance \
                 here).",
            )
            .muted()
        } else {
            let mut rows: Vec<El> = Vec::new();
            for b in &self.buckets {
                rows.push(self.render_bucket_row(modal, b));
            }
            scroll(rows)
                .gap(tokens::SPACE_3)
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0))
        };

        let close = button("Close").key(BUCKETS_MODAL_CLOSE_KEY);

        let children: Vec<El> = vec![
            dialog_header([
                dialog_title("Knowledge buckets"),
                dialog_description(
                    "Catalog of indexed-text buckets the agent can query. \
                     Build / Pause / Delete actions per row. Tracked \
                     buckets (Wikipedia today) carry Poll-now and \
                     Resync-now affordances. Create form + search land \
                     in a follow-up.",
                ),
            ]),
            body,
            dialog_footer([close]),
        ];

        Some(overlay([
            scrim(BUCKETS_MODAL_DISMISS_KEY),
            dialog_content(children)
                .width(Size::Fixed(BUCKETS_W))
                .height(Size::Fixed(BUCKETS_H))
                .block_pointer(),
        ]))
    }

    /// One bucket card. Name + id (mono) at the top, optional
    /// description, then a row of metadata chips (scope, source,
    /// embedder, dense/sparse flags), then slot info when present
    /// (`<dim>-d · <chunks> chunks · <bytes>` + model + built-at),
    /// then live progress / error / row actions.
    fn render_bucket_row(&self, modal: &BucketsModalState, b: &BucketSummary) -> El {
        let row_key = (b.pod_id.clone(), b.id.clone());
        let in_flight_build = modal.build_progress.contains_key(&row_key);

        let mut card_children: Vec<El> = Vec::new();

        // Title row: bold name + muted id.
        card_children.push(
            row([
                text(b.name.clone()).label().bold(),
                text(format!("({})", b.id)).caption().muted(),
            ])
            .gap(tokens::SPACE_2)
            .align(Align::Center),
        );

        if let Some(desc) = b.description.as_deref()
            && !desc.is_empty()
        {
            card_children.push(paragraph(desc.to_string()).muted());
        }

        // Metadata chip row. `scope` carries server/pod, `source`
        // the bucket.toml source-kind tag.
        let mut chips: Vec<El> = vec![
            self.render_bucket_chip("scope", &b.scope),
            self.render_bucket_chip("source", &b.source_kind),
            self.render_bucket_chip("embedder", &b.embedder_provider),
        ];
        chips.push(
            text(if b.dense_enabled {
                "dense \u{2713}"
            } else {
                "dense \u{2717}"
            })
            .caption()
            .muted(),
        );
        chips.push(
            text(if b.sparse_enabled {
                "sparse \u{2713}"
            } else {
                "sparse \u{2717}"
            })
            .caption()
            .muted(),
        );
        if let Some(detail) = b.source_detail.as_deref() {
            chips.push(text(detail.to_string()).caption().muted());
        }
        card_children.push(
            row(chips)
                .gap(tokens::SPACE_3)
                .align(Align::Center)
                .width(Size::Fill(1.0)),
        );

        // Slot info row or "no active slot" hint.
        match &b.active_slot {
            None if !in_flight_build => {
                card_children.push(
                    text("no active slot \u{2014} bucket has not been built yet")
                        .caption()
                        .muted(),
                );
            }
            None => {}
            Some(slot) => {
                card_children.push(
                    row([
                        text("slot").caption().muted(),
                        text(short_slot(&slot.slot_id)).caption().code(),
                        slot_state_chip(slot.state),
                        text(format!(
                            "{}-d \u{00B7} {} chunks \u{00B7} {}",
                            slot.dimension,
                            format_count(slot.chunk_count),
                            format_bytes(slot.disk_size_bytes),
                        ))
                        .caption()
                        .muted(),
                    ])
                    .gap(tokens::SPACE_3)
                    .align(Align::Center)
                    .width(Size::Fill(1.0)),
                );
                let mut model_row: Vec<El> = vec![
                    text(format!("model: {}", slot.embedder_model))
                        .caption()
                        .muted(),
                ];
                if let Some(built) = slot.built_at.as_deref() {
                    model_row.push(text(format!("built: {built}")).caption().muted());
                }
                card_children.push(
                    row(model_row)
                        .gap(tokens::SPACE_3)
                        .align(Align::Center)
                        .width(Size::Fill(1.0)),
                );
            }
        }

        // Live build-progress block (spinner-shaped row).
        if let Some(progress) = modal.build_progress.get(&row_key) {
            card_children.push(self.render_build_progress_row(progress));
        }

        // Sticky last-failed-build error.
        if let Some(err) = modal.build_errors.get(&row_key) {
            card_children.push(
                alert([
                    alert_title("last build error"),
                    alert_description(err.to_string()),
                ])
                .destructive(),
            );
        }

        // Action row.
        card_children.push(self.render_bucket_row_actions(modal, b, in_flight_build));

        card(card_children)
            .gap(tokens::SPACE_2)
            .width(Size::Fill(1.0))
    }

    fn render_bucket_chip(&self, label: &str, value: &str) -> El {
        row([
            text(format!("{label}:")).caption().muted(),
            text(value.to_string()).caption(),
        ])
        .gap(tokens::SPACE_1)
        .align(Align::Center)
    }

    fn render_build_progress_row(&self, p: &BuildProgressView) -> El {
        let phase = match p.phase {
            BucketBuildPhase::Downloading => "downloading",
            BucketBuildPhase::Planning => "planning",
            BucketBuildPhase::Indexing => "indexing",
            BucketBuildPhase::BuildingDense => "building HNSW",
            BucketBuildPhase::Finalizing => "finalizing",
        };
        let counts_body = match (p.dense_inserted, p.dense_total) {
            (Some(inserted), Some(total)) if total > 0 => format!(
                "{} / {} HNSW inserts",
                format_count(inserted),
                format_count(total),
            ),
            _ => format!(
                "{} pages \u{00B7} {} chunks",
                format_count(p.source_records),
                format_count(p.chunks),
            ),
        };
        let elapsed_str = p
            .started_at
            .as_deref()
            .map(format_build_elapsed)
            .filter(|s| !s.is_empty());
        let body = match elapsed_str {
            Some(elapsed) => format!("{phase} \u{00B7} {counts_body} \u{00B7} {elapsed}"),
            None => format!("{phase} \u{00B7} {counts_body}"),
        };
        text(body).caption().color(tokens::WARNING)
    }

    fn render_bucket_row_actions(
        &self,
        modal: &BucketsModalState,
        b: &BucketSummary,
        in_flight_build: bool,
    ) -> El {
        let scope = bucket_scope_token(&b.pod_id);
        let armed = modal.delete_armed.as_deref() == Some(b.id.as_str());

        let mut buttons: Vec<El> = Vec::new();
        if in_flight_build {
            buttons
                .push(button("Pause build").key(format!("{BUCKETS_PAUSE_PREFIX}{scope}:{}", b.id)));
        } else {
            let mut build = button("Build").key(format!("{BUCKETS_BUILD_PREFIX}{scope}:{}", b.id));
            // Managed buckets have no external source — `Build` would
            // bounce server-side. Gate at render so the affordance
            // reads as "not applicable."
            if b.source_kind == "managed" {
                build = build.disabled();
            }
            buttons.push(build);
        }
        if !in_flight_build && b.source_kind == "tracked" {
            buttons.push(button("Poll now").key(format!("{BUCKETS_POLL_PREFIX}{scope}:{}", b.id)));
            buttons
                .push(button("Resync now").key(format!("{BUCKETS_RESYNC_PREFIX}{scope}:{}", b.id)));
        }
        if armed {
            buttons.push(
                button("Confirm delete")
                    .key(format!("{BUCKETS_DELETE_CONFIRM_PREFIX}{scope}:{}", b.id))
                    .destructive(),
            );
            buttons.push(
                button("Cancel").key(format!("{BUCKETS_DELETE_CANCEL_PREFIX}{scope}:{}", b.id)),
            );
        } else {
            buttons.push(button("Delete").key(format!("{BUCKETS_DELETE_PREFIX}{scope}:{}", b.id)));
        }

        row(buttons)
            .gap(tokens::SPACE_2)
            .align(Align::Center)
            .width(Size::Fill(1.0))
            // Inset by a ring-width so the buttons' focus rings
            // don't clip against the card / scroll scissor on the
            // leading edge (lint catches the bare-edge case).
            .padding(Sides {
                left: tokens::RING_WIDTH,
                right: tokens::RING_WIDTH,
                top: 0.0,
                bottom: 0.0,
            })
    }

    /// Classify a pod-relative file path for the file-tree click
    /// dispatcher. Strings mirror server-side constants (`pod::POD_TOML`,
    /// the `behaviors/<id>/{behavior.toml,prompt.md}` shape) — kept in
    /// sync by hand because this crate deliberately doesn't depend on
    /// the server crate. `.json` paths route to the JSON viewer; all
    /// other files fall through to the generic text editor, which the
    /// server-side `is_readonly_path` sniff downgrades to read-only
    /// when the path can't be safely overwritten.
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
    /// have a cached listing or a request in flight. `path` is the
    /// pod-relative directory ("" = pod root). Shallow: children of
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

    /// Open the file-tree modal scoped to `pod_id`. Stamps the slot
    /// (the renderer reads it next frame) and primes the root-dir
    /// fetch so the body has entries to paint by the time the dialog
    /// is on screen.
    pub fn open_file_tree_modal(&mut self, pod_id: String) {
        self.ensure_pod_dir_fetched(&pod_id, "");
        self.file_tree_modal_pod = Some(pod_id);
    }

    /// Recursive emitter for one pod directory into `rows`. Dirs
    /// render as a chevron + name row; clicking toggles
    /// `pod_dirs_open` membership and (on the next frame) the
    /// `ensure_pod_dir_fetched` call below kicks the fetch. Files
    /// render as a row keyed by the dispatch shape — clicks land in
    /// `handle_file_tree_pick` for routing.
    ///
    /// The renderer is `&self`; lazy-fetch + expansion writes happen
    /// downstream in `on_event` (click-driven) and during build via
    /// the side-effect-free `is_open` check.
    fn render_pod_dir(&self, rows: &mut Vec<El>, pod_id: &str, path: &str, depth: usize) {
        let indent = tokens::SPACE_3 * depth as f32;
        let key = (pod_id.to_string(), path.to_string());
        let Some(entries) = self.pod_files.get(&key) else {
            rows.push(text("loading\u{2026}").caption().muted().padding(Sides {
                left: indent,
                ..Sides::default()
            }));
            return;
        };
        if entries.is_empty() {
            rows.push(text("(empty)").caption().muted().padding(Sides {
                left: indent,
                ..Sides::default()
            }));
            return;
        }
        for entry in entries {
            let child_path = if path.is_empty() {
                entry.name.clone()
            } else {
                format!("{path}/{}", entry.name)
            };
            if entry.is_dir {
                let dir_open = self
                    .pod_dirs_open
                    .contains(&(pod_id.to_string(), child_path.clone()));
                let chevron = if dir_open {
                    "chevron-down"
                } else {
                    "chevron-right"
                };
                rows.push(
                    row([
                        icon(chevron)
                            .icon_size(tokens::ICON_XS)
                            .color(tokens::MUTED_FOREGROUND),
                        text(format!("{}/", entry.name)).small().semibold(),
                    ])
                    .key(file_tree_dir_key(pod_id, &child_path))
                    .cursor(Cursor::Pointer)
                    .gap(tokens::SPACE_1)
                    .align(Align::Center)
                    .width(Size::Fill(1.0))
                    .focusable()
                    .padding(Sides {
                        left: indent,
                        ..Sides::default()
                    }),
                );
                if dir_open {
                    self.render_pod_dir(rows, pod_id, &child_path, depth + 1);
                }
            } else {
                let mut label = text(&entry.name).small();
                if entry.readonly {
                    label = label.muted();
                }
                rows.push(
                    row([label])
                        .key(file_tree_file_key(pod_id, &child_path))
                        .cursor(Cursor::Pointer)
                        .align(Align::Center)
                        .width(Size::Fill(1.0))
                        .focusable()
                        .padding(Sides {
                            left: indent + tokens::SPACE_3,
                            ..Sides::default()
                        }),
                );
            }
        }
    }

    /// File-tree modal. Centered `dialog_content` (520 × 600) over a
    /// scrollable column of recursively-rendered entries rooted at
    /// the pod's directory. Lazy: each expanded dir fires a single
    /// `ListPodDir`; the renderer reads cached entries when present
    /// and renders a "loading…" placeholder otherwise. Click on a
    /// file routes through `classify_pod_file_path` to the right
    /// editor.
    fn render_file_tree_modal(&self) -> Option<El> {
        let pod_id = self.file_tree_modal_pod.as_deref()?;
        const FILE_TREE_W: f32 = 520.0;
        const FILE_TREE_H: f32 = 600.0;

        let pod_label = self
            .pods
            .get(pod_id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| pod_id.to_string());

        let mut rows: Vec<El> = Vec::new();
        self.render_pod_dir(&mut rows, pod_id, "", 0);
        let body = scroll(rows)
            .gap(tokens::SPACE_1)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0));

        let close = button("Close").key(FILE_TREE_CLOSE_KEY);

        let children: Vec<El> = vec![
            dialog_header([
                dialog_title(format!("Files \u{2014} {pod_label}")),
                dialog_description(
                    "Click a file to edit it. Specialized files (pod.toml, \
                     behavior configs, prompts) open their own editors; \
                     .json files open the read-only tree viewer.",
                ),
            ]),
            body,
            dialog_footer([close]),
        ];

        Some(overlay([
            scrim(FILE_TREE_DISMISS_KEY),
            dialog_content(children)
                .width(Size::Fixed(FILE_TREE_W))
                .height(Size::Fixed(FILE_TREE_H))
                .block_pointer(),
        ]))
    }

    /// Handle a click on a file row. Looks up the dispatch shape via
    /// `classify_pod_file_path` and opens the right editor. The file-
    /// tree modal stays open underneath — multi-file editing sessions
    /// don't have to bounce through the tree icon between opens.
    fn handle_file_tree_pick(&mut self, pod_id: &str, path: &str) {
        match Self::classify_pod_file_path(path) {
            PodFileDispatch::PodConfig => {
                self.open_pod_editor(pod_id.to_string());
            }
            PodFileDispatch::BehaviorConfig(behavior_id)
            | PodFileDispatch::BehaviorPrompt(behavior_id) => {
                // v1 opens the behavior editor on its default tab —
                // the egui sibling deep-links Prompt clicks to the
                // Prompt tab, which we'll wire when `open_behavior_editor`
                // grows a per-tab variant.
                self.open_behavior_editor(pod_id.to_string(), behavior_id);
            }
            PodFileDispatch::JsonViewer(p) => {
                self.open_json_viewer(pod_id.to_string(), p);
            }
            PodFileDispatch::TextEditor(p) => {
                self.open_file_viewer(pod_id.to_string(), p);
            }
        }
    }

    /// Open the read-only JSON tree viewer on `<pod_id>/<path>`. Mints
    /// a correlation, stamps the slot, and fires `ReadPodFile`; the
    /// matching `PodFileContent` arm hydrates `parsed` (or `error`).
    /// Mirrors the egui sibling's `open_json_viewer`. Public so the
    /// future file-tree slice can route `.json` clicks here without
    /// reaching across the modal-state boundary.
    pub fn open_json_viewer(&mut self, pod_id: String, path: String) {
        let correlation = self.next_correlation_id();
        let mut state = JsonViewerModalState::new(pod_id.clone(), path.clone());
        state.pending_correlation = Some(correlation.clone());
        self.json_tree_open.clear();
        self.json_viewer_modal = Some(state);
        self.send(ClientToServer::ReadPodFile {
            correlation_id: Some(correlation),
            pod_id,
            path,
        });
    }

    /// Read-only JSON tree viewer. Reaches for `dialog_content`
    /// directly (same reasoning as the lightbox — the default
    /// `dialog` width-locks to 420 px, too narrow for a JSON column).
    /// The body is a scrollable column of recursive nodes; the
    /// footer is a single Close button. Pending reads render
    /// "loading…"; parse failures render a destructive paragraph.
    fn render_json_viewer_modal(&self) -> Option<El> {
        let modal = self.json_viewer_modal.as_ref()?;
        const JSON_VIEWER_W: f32 = 720.0;
        const JSON_VIEWER_H: f32 = 560.0;

        let title = format!("{} — {}", modal.path, modal.pod_id);

        let body: El = if let Some(value) = modal.parsed.as_ref() {
            let mut rows: Vec<El> = Vec::new();
            self.render_json_node(&mut rows, "$", "(root)", value, 0);
            scroll(rows)
                .gap(tokens::SPACE_1)
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0))
        } else if let Some(err) = modal.error.as_deref() {
            paragraph(err)
                .color(tokens::DESTRUCTIVE)
                .width(Size::Fill(1.0))
        } else {
            paragraph("loading\u{2026}").muted().width(Size::Fill(1.0))
        };

        let close = button("Close").key(JSON_VIEWER_CLOSE_KEY);

        let children: Vec<El> = vec![
            dialog_header([
                dialog_title(title),
                dialog_description("Read-only JSON viewer"),
            ]),
            body,
            dialog_footer([close]),
        ];

        Some(overlay([
            scrim(JSON_VIEWER_DISMISS_KEY),
            dialog_content(children)
                .width(Size::Fixed(JSON_VIEWER_W))
                .height(Size::Fixed(JSON_VIEWER_H))
                .block_pointer(),
        ]))
    }

    /// Recursive emitter for one JSON node into `rows`. Mirrors the
    /// egui sibling's `render_json_node`: scalars become one-line
    /// monospace labels; objects and arrays become collapsible rows
    /// whose body indents children by `depth + 1`. Long strings render
    /// as `"…"`-suffixed previews so wide payloads don't blow up the
    /// row height. (The egui sibling exposes the full text on hover —
    /// a follow-up once aetna grows a row-level tooltip primitive.)
    ///
    /// `path` is a JSON-pointer-style string used as the accordion
    /// value so sibling collapsibles don't share state (per-node
    /// membership in `json_tree_open` is keyed on the routed form,
    /// `json-tree:accordion:<path>`).
    fn render_json_node(
        &self,
        rows: &mut Vec<El>,
        path: &str,
        label: &str,
        value: &serde_json::Value,
        depth: usize,
    ) {
        const STRING_PREVIEW_BYTES: usize = 80;
        let indent = tokens::SPACE_3 * depth as f32;
        match value {
            serde_json::Value::Null => {
                rows.push(
                    text(format!("{label}: null"))
                        .code()
                        .small()
                        .muted()
                        .padding(Sides {
                            left: indent,
                            ..Sides::default()
                        }),
                );
            }
            serde_json::Value::Bool(b) => {
                rows.push(text(format!("{label}: {b}")).code().small().padding(Sides {
                    left: indent,
                    ..Sides::default()
                }));
            }
            serde_json::Value::Number(n) => {
                rows.push(text(format!("{label}: {n}")).code().small().padding(Sides {
                    left: indent,
                    ..Sides::default()
                }));
            }
            serde_json::Value::String(s) => {
                let full = format!("{s:?}");
                let preview = if full.len() > STRING_PREVIEW_BYTES {
                    let mut cut = STRING_PREVIEW_BYTES;
                    while !full.is_char_boundary(cut) && cut > 0 {
                        cut -= 1;
                    }
                    format!("{}\u{2026}", &full[..cut])
                } else {
                    full
                };
                rows.push(
                    text(format!("{label}: {preview}"))
                        .code()
                        .small()
                        .padding(Sides {
                            left: indent,
                            ..Sides::default()
                        }),
                );
            }
            serde_json::Value::Array(arr) => {
                let header = format!(
                    "{label}: [{} item{}]",
                    arr.len(),
                    if arr.len() == 1 { "" } else { "s" }
                );
                let routed = accordion_item_key(JSON_TREE_ACCORDION_GROUP, &path);
                let open = depth == 0 || self.json_tree_open.contains(&routed);
                rows.push(json_node_trigger(path, &header, open).padding(Sides {
                    left: indent,
                    ..Sides::default()
                }));
                if open {
                    for (i, item) in arr.iter().enumerate() {
                        let child_path = format!("{path}/{i}");
                        let child_label = format!("[{i}]");
                        self.render_json_node(rows, &child_path, &child_label, item, depth + 1);
                    }
                }
            }
            serde_json::Value::Object(obj) => {
                let header = format!(
                    "{label}: {{ {} key{} }}",
                    obj.len(),
                    if obj.len() == 1 { "" } else { "s" }
                );
                let routed = accordion_item_key(JSON_TREE_ACCORDION_GROUP, &path);
                let open = depth == 0 || self.json_tree_open.contains(&routed);
                rows.push(json_node_trigger(path, &header, open).padding(Sides {
                    left: indent,
                    ..Sides::default()
                }));
                if open {
                    for (k, v) in obj.iter() {
                        let child_path = format!("{path}/{k}");
                        self.render_json_node(rows, &child_path, k, v, depth + 1);
                    }
                }
            }
        }
    }

    fn render_fork_modal(&self) -> Option<El> {
        let modal = self.fork_modal.as_ref()?;

        let archive_switch = switch(modal.archive_original).key(FORK_MODAL_ARCHIVE_KEY);
        let reset_switch = switch(modal.reset_capabilities).key(FORK_MODAL_RESET_CAPS_KEY);

        let confirm = button("Fork").key(FORK_MODAL_CONFIRM_KEY).primary();
        let cancel = button("Cancel").key(FORK_MODAL_CANCEL_KEY);

        let body = column([
            paragraph(
                "Forks this thread at the selected user message. The new \
                 thread shares the pod, bindings, config, and tool allowlist, \
                 and starts with the conversation up to (but not including) \
                 that message — ready for you to retype the prompt.",
            )
            .muted(),
            form([
                form_item([
                    form_label("Archive original"),
                    form_control(archive_switch),
                    form_description(
                        "Archived threads drop off the sidebar list but stay \
                         on disk; still readable from the server's pod \
                         directory.",
                    ),
                ]),
                form_item([
                    form_label("Reset capabilities"),
                    form_control(reset_switch),
                    form_description(
                        "Off (default): inherit the source thread's live \
                         bindings, scope, and config. On: re-derive from the \
                         pod's current defaults — pick this up to use newly-\
                         added MCP hosts, sandbox bindings, or cap changes \
                         since the source was created.",
                    ),
                ]),
            ]),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0));

        let children: Vec<El> = vec![
            dialog_header([
                dialog_title("Fork from this message"),
                dialog_description(format!(
                    "Forking at message #{} of the current thread.",
                    modal.from_message_index
                )),
            ]),
            body,
            dialog_footer([cancel, confirm]),
        ];
        Some(dialog(FORK_MODAL_KEY, children))
    }

    fn backend_menu(&self) -> El {
        let mut options: Vec<(String, String)> = vec![(
            PICKER_INHERIT.to_string(),
            "Inherit pod default".to_string(),
        )];
        for b in &self.backends {
            // `kind` (e.g. "anthropic" / "openai_chat") is useful
            // context — surface it after the alias so the option
            // reads as `prod-anthropic — anthropic`.
            options.push((b.name.clone(), format!("{} — {}", b.name, b.kind)));
        }
        select_menu(PICKER_BACKEND, options)
    }

    fn model_menu(&self) -> El {
        let mut options: Vec<(String, String)> = vec![(
            PICKER_INHERIT.to_string(),
            "Auto (backend default)".to_string(),
        )];
        if let Some(b) = self.picker_backend.as_ref()
            && let Some(list) = self.models_by_backend.get(b)
        {
            for m in list {
                let label = m
                    .display_name
                    .clone()
                    .map(|d| format!("{d}  ({})", m.id))
                    .unwrap_or_else(|| m.id.clone());
                options.push((m.id.clone(), label));
            }
        }
        select_menu(PICKER_MODEL, options)
    }

    fn pod_menu(&self) -> El {
        let mut options: Vec<(String, String)> =
            vec![(PICKER_INHERIT.to_string(), "Default pod".to_string())];
        let mut pods: Vec<&PodSummary> = self.pods.values().filter(|p| !p.archived).collect();
        pods.sort_by(|a, b| a.name.cmp(&b.name));
        for p in pods {
            options.push((p.pod_id.clone(), p.name.clone()));
        }
        select_menu(PICKER_POD, options)
    }

    fn compose_box(&self) -> El {
        // Allow attachments-only sends: the gate matches send_compose's
        // mirror (text-or-attachments). While the selected thread is
        // Working, the primary button toggles to "Stop" (always
        // enabled) and clicks cancel the thread instead.
        let buf = self.active_compose_text();
        let working = self.is_selected_working();
        let can_send = !buf.trim().is_empty() || !self.compose_attachments.is_empty();
        let send = if working {
            // Cancel-shaped affordance — destructive styling so the
            // user reads it as a stop, not a send.
            button("Stop").key(SEND_KEY).destructive()
        } else {
            let mut b = button("Send").key(SEND_KEY).primary();
            if !can_send {
                b = b.disabled();
            }
            b
        };
        let attach = icon_button(crate::icons::ICON_PAPERCLIP.clone())
            .key(COMPOSE_ATTACH_KEY)
            .ghost();
        let editor = text_area(buf, &self.selection, COMPOSE_KEY).height(Size::Fixed(120.0));

        // The compose row sits at the bottom of the pane: text_area
        // takes the leftover width, attach + send hug to the right.
        // We lift its fill to `tokens::CARD` so it reads as an
        // elevated bottom panel against the deeper chat-log
        // background — the panel stops feeling continuous with the
        // scroll area and the user has a clear "compose surface" to
        // land on. Thread-action row + thumbnail strip + hint sit
        // above when relevant.
        let mut children: Vec<El> = Vec::new();
        if let Some(actions) = self.thread_actions_row() {
            children.push(actions);
        }
        if let Some(strip) = self.render_compose_attachments() {
            children.push(strip);
        }
        if let Some(hint_el) = self.render_compose_hint() {
            children.push(hint_el);
        }
        children.push(
            row([editor, attach, send])
                .gap(tokens::SPACE_3)
                .align(Align::End)
                .width(Size::Fill(1.0)),
        );
        // Static keyboard-shortcut footer beneath the compose row.
        // Only visible when the buffer is empty + no attachments are
        // staged — the moment the user starts typing the cue is no
        // longer load-bearing. Mirrors how a placeholder would behave
        // if `text_area` had one (it doesn't yet).
        if buf.is_empty() && self.compose_attachments.is_empty() {
            children.push(self.compose_kbd_hint());
        }
        column(children)
            .gap(tokens::SPACE_2)
            .padding(tokens::SPACE_3)
            .width(Size::Fill(1.0))
            // Panel role disambiguates this from a hand-rolled card —
            // the bottom compose strip is a docked surface (no rounded
            // corners, full bleed) that doesn't fit the `card([…])`
            // recipe but does want the elevated CARD fill.
            .surface_role(SurfaceRole::Panel)
            .stroke(tokens::BORDER)
            .fill(tokens::CARD)
    }

    /// Small caption row beneath the compose textarea that surfaces the
    /// Enter-to-send convention. Two muted chips:
    /// `↵ to send` and `⇧+↵ for newline`. Built as a single right-
    /// aligned row so it lives just under the Send button without
    /// stealing focus from anything else on the bar.
    fn compose_kbd_hint(&self) -> El {
        let return_chip = row([
            icon(crate::icons::ICON_RETURN.clone())
                .icon_size(tokens::ICON_XS)
                .text_color(tokens::MUTED_FOREGROUND),
            text("to send").xsmall().muted(),
        ])
        .gap(tokens::SPACE_1)
        .align(Align::Center);

        let shift_chip = text("Shift+\u{21B5} for newline").xsmall().muted();

        row([spacer(), return_chip, shift_chip])
            .gap(tokens::SPACE_3)
            .align(Align::Center)
            .width(Size::Fill(1.0))
    }

    /// Whether the currently-selected thread is in `Working` state.
    /// `false` for the no-selection (compose-new) path. Used by the
    /// compose box to toggle Send → Stop and by the Compact button
    /// to gate enabled state.
    fn is_selected_working(&self) -> bool {
        self.selected
            .as_deref()
            .and_then(|tid| self.threads.get(tid))
            .map(|s| s.state == whisper_agent_protocol::ThreadStateLabel::Working)
            .unwrap_or(false)
    }

    /// Thread-level action row above the compose text_area:
    /// Cancel / Archive / Compact. `None` when no thread is
    /// selected (compose-new mode has no thread to act on).
    /// Cancel hides while Working — the primary Send/Stop button
    /// already covers cancellation. Compact disables while
    /// Working since the server rejects mid-turn compaction.
    fn thread_actions_row(&self) -> Option<El> {
        let _tid = self.selected.as_ref()?;
        let working = self.is_selected_working();
        let mut buttons: Vec<El> = Vec::new();
        if !working {
            buttons.push(button("Cancel").key(THREAD_CANCEL_KEY).ghost());
        }
        buttons.push(button("Archive").key(THREAD_ARCHIVE_KEY).ghost());
        let mut compact = button("Compact").key(THREAD_COMPACT_KEY).ghost();
        if working {
            compact = compact.disabled();
        }
        buttons.push(compact);
        Some(
            row(buttons)
                .gap(tokens::SPACE_2)
                .align(Align::Center)
                .justify(Justify::End)
                .width(Size::Fill(1.0)),
        )
    }

    /// Thumbnail strip above the compose row. Each staged image
    /// renders as a 96×96 thumbnail with a small `×` overlay
    /// keyed `compose:attach:remove:{id}`. Hugs height when no
    /// attachments are staged (callers omit the strip entirely
    /// in that case).
    fn render_compose_attachments(&self) -> Option<El> {
        if self.compose_attachments.is_empty() {
            return None;
        }
        let tiles: Vec<El> = self
            .compose_attachments
            .iter()
            .map(compose_thumbnail_tile)
            .collect();
        Some(
            row(tiles)
                .gap(tokens::SPACE_2)
                .align(Align::Start)
                .width(Size::Fill(1.0)),
        )
    }

    /// Render the ephemeral compose hint paragraph if one is live.
    /// Self-expires through `before_build`'s deadline check; the
    /// renderer only mirrors the slot's current contents.
    fn render_compose_hint(&self) -> Option<El> {
        let (msg, _expires) = self.compose_hint.as_ref()?;
        Some(paragraph(msg.clone()).muted().small())
    }
}

/// Build a single compose-thumbnail tile: 96×96 image (or
/// "(URL)" placeholder), small file label below it, and an `×`
/// remove button overlaid in the top-right corner. The `id`
/// rides into the route key so duplicates the user staged
/// deliberately are still individually addressable.
fn compose_thumbnail_tile(s: &StagedAttachment) -> El {
    let remove_key = format!("{COMPOSE_ATTACH_REMOVE_PREFIX}{}", s.id);
    let preview: El = if let Some(img) = s.thumbnail.as_ref() {
        El::new(Kind::Custom("compose-thumbnail"))
            .image(img.clone())
            .image_fit(ImageFit::Cover)
            .width(Size::Fixed(96.0))
            .height(Size::Fixed(96.0))
            .radius(tokens::RADIUS_SM)
            .clip()
    } else {
        // Decode failed (HEIC/HEIF without C lib, or a corrupt
        // payload) — placeholder keeps layout stable. Bytes still
        // ride to the server.
        column([text("(no preview)").muted().small()])
            .width(Size::Fixed(96.0))
            .height(Size::Fixed(96.0))
            .align(Align::Center)
            .justify(Justify::Center)
            .fill(tokens::MUTED)
            .radius(tokens::RADIUS_SM)
    };
    let remove = icon_button(crate::icons::ICON_X.clone())
        .key(&remove_key)
        .ghost();
    let label = text(truncate_filename(&s.source_desc, 20)).muted().small();
    column([
        row([preview, remove])
            .gap(tokens::SPACE_1)
            .align(Align::Start),
        label,
    ])
    .gap(tokens::SPACE_1)
    .width(Size::Hug)
}

/// Truncate a filename for display under a 96 px thumbnail tile.
/// Preserves the start so the user can identify the file at a
/// glance; ellipsizes the middle when needed.
fn truncate_filename(name: &str, max_chars: usize) -> String {
    if name.chars().count() <= max_chars {
        return name.to_string();
    }
    let half = max_chars.saturating_sub(1) / 2;
    let prefix: String = name.chars().take(half).collect();
    let suffix: String = name
        .chars()
        .rev()
        .take(half)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{prefix}…{suffix}")
}

/// Centered empty-state block for the chat pane. Shows a soft-tint
/// circular icon medallion above a headline and a muted body line.
/// Lighter than `card([…])` (no border, no fill) so it doesn't compete
/// with adjacent chrome; the icon medallion is the only colored
/// element so the eye still has something to land on instead of bare
/// text floating in the middle of the pane.
fn empty_state(icon_src: aetna_core::SvgIcon, headline: &str, body: &str) -> El {
    let medallion = column([icon(icon_src)
        .icon_size(28.0)
        .text_color(tokens::MUTED_FOREGROUND)])
    .width(Size::Fixed(64.0))
    .height(Size::Fixed(64.0))
    .align(Align::Center)
    .justify(Justify::Center)
    .fill(tokens::MUTED)
    .radius(32.0);

    column([
        medallion,
        text(headline).label().semibold(),
        paragraph(body).muted().small(),
    ])
    .padding(tokens::SPACE_6)
    .gap(tokens::SPACE_3)
    .align(Align::Center)
    .justify(Justify::Center)
    .width(Size::Fill(1.0))
    .height(Size::Fill(1.0))
}

/// Map a `TriggerSpec` to the same short discriminator the server
/// puts in `BehaviorSummary.trigger_kind` (`"manual" | "cron" |
/// "webhook"`). Keeps both populated paths consistent — `BehaviorList`
/// rides `trigger_kind` directly; `BehaviorUpdated` carries a full
/// `BehaviorSnapshot` and we have to re-derive it.
fn trigger_kind_label(spec: &whisper_agent_protocol::TriggerSpec) -> &'static str {
    use whisper_agent_protocol::TriggerSpec as T;
    match spec {
        T::Manual => "manual",
        T::Cron { .. } => "cron",
        T::Webhook { .. } => "webhook",
    }
}

/// Compact label for the sidebar's pod tab pill. Shows the pod name
/// with a thread-count suffix so an empty pod is visible at a glance;
/// archived pods get a trailing marker so the tab still reads sensibly
/// when the user has un-hidden them.
fn pod_tab_label(pod: &PodSummary) -> String {
    let suffix = if pod.archived { " (archived)" } else { "" };
    format!("{}{suffix}", pod.name)
}

/// Map a `ThreadStateLabel` to a styled badge. Color mirrors the
/// shadcn / Radix semantic palette: success for done, info-ish for
/// in-flight, destructive for failures.
fn state_badge(state: whisper_agent_protocol::ThreadStateLabel) -> El {
    use whisper_agent_protocol::ThreadStateLabel as S;
    let b = badge(state_label(state));
    match state {
        S::Idle => b.muted(),
        S::Working => b, // default info palette
        S::Completed => b.success(),
        S::Failed => b.destructive(),
        S::Cancelled => b.muted(),
    }
}

fn state_label(state: whisper_agent_protocol::ThreadStateLabel) -> &'static str {
    use whisper_agent_protocol::ThreadStateLabel as S;
    match state {
        S::Idle => "idle",
        S::Working => "working",
        S::Completed => "done",
        S::Failed => "failed",
        S::Cancelled => "cancelled",
    }
}

/// Format an RFC3339 timestamp as a compact relative duration:
/// `"12m"`, `"3h"`, `"1d"`, `"2w"`, `"5mo"`, `"2y"`. Mirrors the
/// Client-side pre-flight check on a pod_id about to be sent to
/// `CreatePod`. The server runs the same checks (and a few more)
/// before writing to disk, but rejecting locally lets us surface a
/// hint inline without a wire round-trip. Mirrors the egui sibling's
/// `validate_pod_id_client`.
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

/// Same rules as [`validate_pod_id_client`] — behavior ids become
/// directory names under `<pod>/behaviors/`, so the constraint set
/// is identical. Kept as a separate function so error messages can
/// say "behavior_id" instead of "pod_id".
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

/// Build a minimal `PodConfig` from the user's display name plus the
/// known backends. Lighter than the egui sibling's
/// `fresh_pod_config` — that path also clones the server-default
/// pod's template (system prompt, allowed MCPs, host envs) which we
/// haven't wired the `GetPod` round-trip for yet. The user can
/// edit any of these afterwards via the (eventual) pod editor; for
/// now this is enough to land a working pod the server will accept.
///
/// `backend_names` should be the full list of configured backends —
/// the new pod is allowed to use any of them. The first by sort
/// order becomes the `thread_defaults.backend`.
fn fresh_pod_config(name: String, mut backend_names: Vec<String>) -> PodConfig {
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

/// Truncate a thread id for inline use in chips ("forked from {id}",
/// "dispatched from {id}"). Thread ids are typically `t-{ulid}`
/// shaped — long enough to make a chip noisy if rendered in full.
/// First 12 chars + `…` is enough to recognize the thread when
/// hovering matches against the sidebar.
fn short_id(id: &str) -> String {
    const HEAD: usize = 12;
    let chars = id.chars().count();
    if chars <= HEAD {
        return id.to_string();
    }
    let mut head: String = id.chars().take(HEAD).collect();
    head.push('…');
    head
}

/// egui sibling's `format_relative_time`. Returns `"just now"` for
/// sub-minute deltas and `""` for parse failures (callers can drop
/// the row's secondary line entirely if they care).
fn format_relative(rfc3339: &str) -> String {
    let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(rfc3339) else {
        return String::new();
    };
    let now = chrono::Utc::now();
    let delta = now.signed_duration_since(parsed.with_timezone(&chrono::Utc));
    let secs = delta.num_seconds();
    if secs < 60 {
        return "just now".to_string();
    }
    let mins = secs / 60;
    if mins < 60 {
        return format!("{mins}m");
    }
    let hours = mins / 60;
    if hours < 24 {
        return format!("{hours}h");
    }
    let days = hours / 24;
    if days < 14 {
        return format!("{days}d");
    }
    let weeks = days / 7;
    if weeks < 8 {
        return format!("{weeks}w");
    }
    let months = days / 30;
    if months < 24 {
        return format!("{months}mo");
    }
    let years = days / 365;
    format!("{years}y")
}

/// Reorder a flat list of threads into DFS-by-dispatch order. Each
/// root is followed by its dispatched children (transitively); rare
/// in production, but when a thread fans out to N subthreads the
/// indent makes the parentage scan-able. Children whose parent isn't
/// in the input list are promoted to roots so nothing is lost; cycles
/// are broken by a visited set (the scheduler enforces a depth cap,
/// but defensive code is cheap).
fn order_threads_by_dispatch<'a>(flat: &[&'a ThreadSummary]) -> Vec<(&'a ThreadSummary, usize)> {
    let in_set: HashSet<&str> = flat.iter().map(|t| t.thread_id.as_str()).collect();
    let by_id: HashMap<&str, &ThreadSummary> =
        flat.iter().map(|t| (t.thread_id.as_str(), *t)).collect();

    // Sibling order: walk `flat` in input order so the newest-first
    // sort is preserved within each bucket.
    let mut children_of: HashMap<&str, Vec<&ThreadSummary>> = HashMap::new();
    let mut roots: Vec<&ThreadSummary> = Vec::new();
    for t in flat {
        match t.dispatched_by.as_deref() {
            Some(parent) if in_set.contains(parent) => {
                children_of.entry(parent).or_default().push(t);
            }
            _ => roots.push(t),
        }
    }

    let mut visited: HashSet<&str> = HashSet::new();
    let mut out: Vec<(&ThreadSummary, usize)> = Vec::new();
    let mut stack: Vec<(&ThreadSummary, usize)> =
        roots.into_iter().rev().map(|t| (t, 0_usize)).collect();
    while let Some((t, depth)) = stack.pop() {
        if !visited.insert(t.thread_id.as_str()) {
            continue;
        }
        out.push((t, depth));
        if let Some(kids) = children_of.get(t.thread_id.as_str()) {
            // Reverse-push so the original order is preserved on pop.
            for kid in kids.iter().rev() {
                stack.push((kid, depth + 1));
            }
        }
    }
    // Promote any unreached children (ones whose parent landed in
    // `flat` but the BFS missed — defensive against cycles).
    for t in flat {
        if visited.insert(t.thread_id.as_str()) {
            out.push((t, 0));
        }
    }
    let _ = by_id; // reserved for parent lookup if we add hover tooltips later
    out
}

/// Take up to `n` chars from a string, respecting char boundaries.
#[allow(dead_code)]
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

impl ChatApp {
    /// One row in the chat log. Per upstream guidance
    /// (`aetna-core/README.md` "Conversation / event-log row"), we
    /// don't render every message as a `card` — cards isolate
    /// objects, but a transcript should scan as one continuous
    /// record with small role markers. Each row is a 3px role-colored
    /// gutter beside the message content; user rows get a faint
    /// info-tinted fill so user turns stand out from a long
    /// assistant stream.
    ///
    /// Reasoning + tool rows nest an `accordion_item` so their
    /// bodies collapse — typical thread reads are noisy with these
    /// otherwise.
    fn event_log_row(&self, idx: usize, item: &DisplayItem, cx: &BuildCx) -> El {
        match item {
            DisplayItem::User { text: t, msg_index } => {
                // Wrap the body in a keyed container so
                // `is_hovering_within` can answer "is the cursor on
                // this row?" — the fork affordance fades in based on
                // the predicate. The key uses `idx` (display-item
                // position, stable per build pass) rather than
                // msg_index so that two User items at the same
                // msg_index — possible during streaming if anything
                // weird happens — don't collide. Behavior is purely
                // visual; routing for the fork click goes through
                // `fork:{msg_index}` (msg_index is what the wire
                // needs).
                let row_key = chat_user_row_key(idx);
                let hovered = cx.is_hovering_within(&row_key);
                // Plain paragraph for the unhovered case keeps the
                // wrap calculation identical to the pre-fork-
                // affordance render — wrapping a paragraph in a
                // multi-child row caused Y-overflow because the
                // row's cross-axis sizing didn't propagate the
                // wrap-width hint correctly. Only diverge into a
                // row layout when the affordance is actually
                // present.
                let inner = if hovered {
                    // `git-branch` ships in aetna's built-in icon
                    // registry — close enough to a "fork" semantic
                    // for v1 without bundling a `git-fork` SVG.
                    let fork = icon_button("git-branch")
                        .key(chat_user_fork_key(*msg_index))
                        .ghost()
                        .icon_size(tokens::ICON_XS);
                    row([paragraph(t.clone()).width(Size::Fill(1.0)), fork])
                        .gap(tokens::SPACE_2)
                        .align(Align::Start)
                        .width(Size::Fill(1.0))
                        .key(row_key)
                } else {
                    paragraph(t.clone()).width(Size::Fill(1.0)).key(row_key)
                };
                // Upstream README's worked example uses
                // `with_alpha(18)` (~7%) for the user fill — too low
                // to be visible against the dark zinc-950 background.
                // Bump to ~25% so user turns actually break up a long
                // assistant stream. Subtle enough to still read as
                // "log entry, not card".
                log_row(tokens::INFO, Some(tokens::INFO.with_alpha(64)), inner)
            }
            DisplayItem::Assistant { text: t } => {
                // Markdown rendering for assistant content. The egui
                // sibling renders user input verbatim and assistant
                // output as markdown — same pattern here.
                let body = aetna_markdown::md(t);
                log_row(tokens::SUCCESS, None, body)
            }
            DisplayItem::Reasoning { text: t } => {
                let preview = first_line_preview(t, 80);
                let key = "reasoning";
                let value = format!("{idx}");
                let routed = accordion_item_key(key, &value);
                let open = self.open_accordions.contains(&routed);
                let item_el = accordion_item(key, value, preview, open, [paragraph(t.clone())]);
                log_row(tokens::MUTED_FOREGROUND, None, item_el)
            }
            DisplayItem::SetupPrompt { text: t } => {
                // Default-collapsed accordion: header is "SYSTEM"
                // tag + a one-line preview of the prompt; body
                // is the full text in a code-styled block. The
                // muted gutter reads as "thread metadata" rather
                // than a participant turn.
                let preview = first_line_preview(t, 80);
                let key = "setup-prompt";
                let value = format!("{idx}");
                let routed = accordion_item_key(key, &value);
                let open = self.open_accordions.contains(&routed);
                let header = format!("SYSTEM · {preview}");
                let item_el = accordion_item(key, value, header, open, [code_block(t.clone())]);
                log_row(tokens::MUTED_FOREGROUND, None, item_el)
            }
            DisplayItem::SetupTools { entries } => {
                // Default-collapsed accordion: header is the count
                // ("TOOLS · 23 tools"); body lists per-tool
                // name + description as muted text. Future work:
                // walk the typed params for each tool inline (the
                // egui sibling does this; we kept the wire shape
                // off the `DisplayItem` for now).
                let count = entries.len();
                let key = "setup-tools";
                let value = format!("{idx}");
                let routed = accordion_item_key(key, &value);
                let open = self.open_accordions.contains(&routed);
                let header = format!("TOOLS · {count} tool{}", if count == 1 { "" } else { "s" });
                let body_blocks: Vec<El> = entries
                    .iter()
                    .map(|t| {
                        column([
                            text(t.name.clone()).label().bold(),
                            text(t.description.clone()).muted().small().wrap_text(),
                        ])
                        .gap(tokens::SPACE_1)
                        .width(Size::Fill(1.0))
                    })
                    .collect();
                let item_el = accordion_item(key, value, header, open, body_blocks);
                log_row(tokens::MUTED_FOREGROUND, None, item_el)
            }
            DisplayItem::ToolCall {
                tool_use_id,
                name,
                summary,
                diff,
                args_pretty,
                streaming_output,
                result,
            } => {
                let key = "tool";
                let value = format!("{idx}");
                let routed = accordion_item_key(key, &value);
                // Auto-expand while content is streaming so the
                // user sees bash-style output scroll without
                // clicking each running call. Collapses back to
                // default-closed once `End` lands and the
                // streaming buffer clears. The user's explicit
                // `open_accordions` membership keeps a finished
                // call open if they manually expanded it.
                let streaming = result.is_none() && !streaming_output.is_empty();
                let open = streaming || self.open_accordions.contains(&routed);
                let header = tool_call_header(name, summary.as_deref(), result.as_ref());
                let body_blocks = tool_call_body(
                    diff.as_ref(),
                    args_pretty.as_deref(),
                    streaming_output,
                    result.as_ref(),
                );
                let item_el = accordion_item(key, value, header, open, body_blocks);
                let gutter = match result {
                    Some(r) if r.is_error => tokens::DESTRUCTIVE,
                    _ => tokens::WARNING,
                };
                let _ = tool_use_id; // reserved for future fork / re-run affordances
                log_row(gutter, None, item_el)
            }
            DisplayItem::ToolResult {
                tool_use_id,
                text: t,
                is_error,
            } => {
                let key = "tool-result";
                let value = format!("{idx}");
                let routed = accordion_item_key(key, &value);
                let open = self.open_accordions.contains(&routed);
                let header = if *is_error {
                    "tool result (error)".to_string()
                } else {
                    "tool result".to_string()
                };
                let item_el = accordion_item(key, value, header, open, [code_block(t.clone())]);
                let _ = tool_use_id;
                let gutter = if *is_error {
                    tokens::DESTRUCTIVE
                } else {
                    tokens::WARNING
                };
                log_row(gutter, None, item_el)
            }
            DisplayItem::GenericPlaceholder { label } => {
                let key = "generic";
                let value = format!("{idx}");
                let routed = accordion_item_key(key, &value);
                let open = self.open_accordions.contains(&routed);
                let item_el =
                    accordion_item(key, value, label.clone(), open, [paragraph(label.clone())]);
                log_row(tokens::MUTED_FOREGROUND, None, item_el)
            }
            DisplayItem::TurnStats { usage } => {
                // No gutter, no fill — auxiliary metadata, not
                // conversation. A right-aligned single-line caption
                // sits where the assistant's gutter ends so it
                // visually attaches to the turn it summarizes
                // without dominating the chat.
                row([spacer(), text(turn_stats_text(usage)).caption().muted()])
                    .align(Align::Center)
                    .padding(Sides::xy(tokens::SPACE_3, 0.0))
                    .width(Size::Fill(1.0))
            }
            DisplayItem::Image { is_user, state } => {
                let gutter = if *is_user {
                    tokens::INFO
                } else {
                    tokens::SUCCESS
                };
                let body = image_body(state);
                // Clickable wrapper opens the fullscreen lightbox.
                // Only Decoded states get the click affordance — a
                // URL placeholder and a decode-failure row are
                // already non-interactive blocks. The route key
                // carries the row idx so `on_event` can pull the
                // matching display item out of the active view.
                let is_clickable = matches!(state, ImageRenderState::Decoded { .. });
                let body_keyed = if is_clickable {
                    body.key(chat_image_lightbox_key(idx))
                        .focusable()
                        .cursor(Cursor::Pointer)
                } else {
                    body
                };
                if *is_user {
                    log_row(gutter, Some(tokens::INFO.with_alpha(64)), body_keyed)
                } else {
                    log_row(gutter, None, body_keyed)
                }
            }
        }
    }
}

/// Render the body of a [`DisplayItem::Image`]. Decoded images go
/// through aetna's [`image()`] widget at a capped display height
/// (so a 4K screenshot doesn't dominate the chat log); URL sources
/// and decode failures fall through to small annotated placeholders.
fn image_body(state: &ImageRenderState) -> El {
    /// Cap on the displayed height of a single image row. Width is
    /// chosen by `ImageFit::Contain` to preserve aspect — a wide
    /// screenshot stays full-width but shrinks vertically; a tall
    /// portrait shrinks to the cap and lets width auto-fit.
    const MAX_DISPLAY_HEIGHT: f32 = 320.0;

    match state {
        ImageRenderState::Decoded {
            image: img,
            width,
            height,
        } => {
            // Choose displayed dimensions: never up-scale, never
            // exceed the cap. Width is derived from the aspect ratio
            // of the source so long-aspect images keep their shape.
            let (w, h) = display_dims(*width, *height, MAX_DISPLAY_HEIGHT);
            let caption = text(format!("{width}×{height}")).caption().muted();
            column([
                image(img.clone())
                    .image_fit(ImageFit::Contain)
                    .width(Size::Fixed(w))
                    .height(Size::Fixed(h))
                    .radius(tokens::RADIUS_SM),
                caption,
            ])
            .gap(tokens::SPACE_1)
        }
        ImageRenderState::Url { url } => {
            // No fetch yet — surface the URL so the user can at least
            // know what was meant to render. Stage 7 follow-up: fetch
            // + cache via the host shell.
            column([
                text("[image at remote URL — fetch deferred]").muted(),
                text(url.clone()).caption().muted(),
            ])
            .gap(tokens::SPACE_1)
        }
        ImageRenderState::Failed { reason } => column([
            text("[image decode failed]").destructive(),
            text(reason.clone()).caption().muted(),
        ])
        .gap(tokens::SPACE_1),
    }
}

/// Compute display `(width, height)` for an inline image given the
/// source dims and a max display height. Never scales up; preserves
/// aspect ratio. Used by `image_body` so a tall portrait shrinks to
/// the cap and a tiny avatar stays at native size.
fn display_dims(src_w: u32, src_h: u32, max_h: f32) -> (f32, f32) {
    let src_w = src_w.max(1) as f32;
    let src_h = src_h.max(1) as f32;
    if src_h <= max_h {
        return (src_w, src_h);
    }
    let scale = max_h / src_h;
    (src_w * scale, max_h)
}

/// Collapsible row for the JSON tree viewer — tighter than
/// `accordion_trigger` (which bakes a 40 px height + card chrome
/// suited to a list of full-width items). A deeply-nested tree
/// needs single-line rows: a chevron, a monospace header label, and
/// no fill. The row carries the accordion-routed key so the existing
/// `:accordion:` event handler picks it up.
#[track_caller]
fn json_node_trigger(path: &str, header: &str, open: bool) -> El {
    let chevron = if open {
        "chevron-down"
    } else {
        "chevron-right"
    };
    row([
        icon(chevron)
            .icon_size(tokens::ICON_XS)
            .color(tokens::MUTED_FOREGROUND),
        text(header.to_string()).code().small().semibold(),
    ])
    .key(accordion_item_key(JSON_TREE_ACCORDION_GROUP, &path))
    .cursor(Cursor::Pointer)
    .gap(tokens::SPACE_1)
    .align(Align::Center)
    .width(Size::Fill(1.0))
    .focusable()
}

/// Same as [`display_dims`] but with a separate width cap. The
/// lightbox modal needs both axes constrained: a wide ultrawide
/// screenshot at native dims would punch out of the dialog, and a
/// tall portrait would push the close button off-screen. Aspect is
/// preserved by picking the smaller of the two scale factors.
/// Slot-state colored caption — `ready` green, `failed` destructive,
/// `building` / `planning` warning, `archived` muted. Used in the
/// bucket-row slot-info line. Mirrors the egui sibling's
/// `state_chip` color choices.
fn slot_state_chip(state: SlotStateLabel) -> El {
    let (label, color) = match state {
        SlotStateLabel::Planning => ("planning", tokens::WARNING),
        SlotStateLabel::Building => ("building", tokens::WARNING),
        SlotStateLabel::Ready => ("ready", tokens::SUCCESS),
        SlotStateLabel::Failed => ("failed", tokens::DESTRUCTIVE),
        SlotStateLabel::Archived => ("archived", tokens::MUTED_FOREGROUND),
    };
    text(label).caption().color(color)
}

/// Shorten a slot id to its leading 8 chars + ellipsis. Slot ids are
/// ~30-char timestamps + random suffix; the 8-char prefix sorts and
/// disambiguates without dominating the row.
fn short_slot(slot_id: &str) -> String {
    if slot_id.len() > 8 {
        format!("{}\u{2026}", &slot_id[..8])
    } else {
        slot_id.to_string()
    }
}

/// Compact human count: `1.5M` / `820k` / `42`. Used in the bucket
/// chunk-count + per-page progress counters.
fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

/// Compact human size: `1.23 GiB` / `45.6 MiB` / `7.8 KiB` / `123 B`.
/// Powers-of-2 units so the displayed value matches the on-disk
/// reservation.
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Build-elapsed formatter — `2d 14h` / `3h 22m` / `5m 13s` / `42s`.
/// Multi-day buckets (Wikipedia-scale builds) want days-and-hours;
/// sub-minute builds want raw seconds. Returns empty string on
/// unparseable input so the caller's `.filter(|s| !s.is_empty())`
/// suppresses the elapsed segment.
fn format_build_elapsed(started_at_rfc3339: &str) -> String {
    let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(started_at_rfc3339) else {
        return String::new();
    };
    let secs = (chrono::Utc::now() - parsed.with_timezone(&chrono::Utc)).num_seconds();
    if secs < 0 {
        // Clock skew between server and client — surface as 0s.
        return "0s elapsed".to_string();
    }
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let d = secs / 86400;
    let body = if d > 0 {
        format!("{d}d {h}h")
    } else if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    };
    format!("{body} elapsed")
}

fn lightbox_display_dims(src_w: u32, src_h: u32, max_w: f32, max_h: f32) -> (f32, f32) {
    let src_w = src_w.max(1) as f32;
    let src_h = src_h.max(1) as f32;
    let scale_w = max_w / src_w;
    let scale_h = max_h / src_h;
    let scale = scale_w.min(scale_h).min(1.0);
    (src_w * scale, src_h * scale)
}

/// Collapsed-header label for a tool call. Three pieces compose
/// into a single line: status glyph + name + optional argument
/// summary + optional result preview. The summary lands in
/// `[brackets]` so it reads as a parameter even when an in-flight
/// row has no result yet (`⏳ read_file [/path/to/file.rs]`).
/// Once a result arrives, the preview suffix lets a closed row
/// communicate pass / fail *and* the outcome at a glance —
/// `✓ read_file [src/lib.rs] · pub mod foo;`.
fn tool_call_header(name: &str, summary: Option<&str>, result: Option<&FusedToolResult>) -> String {
    let glyph = match result {
        None => "⏳",
        Some(r) if r.is_error => "✗",
        Some(_) => "✓",
    };
    let mut out = format!("{glyph} {name}");
    if let Some(s) = summary {
        out.push_str(" [");
        out.push_str(s);
        out.push(']');
    }
    if let Some(r) = result {
        let preview = first_line_preview(&r.text, 80);
        if !preview.is_empty() {
            out.push_str(" · ");
            out.push_str(&preview);
        }
    }
    out
}

/// Expanded-body content for a tool-call accordion. When a `diff`
/// is present (`edit_file` / `write_file` paths the renderer
/// recognizes), the body shows a unified-diff view rooted at the
/// path; otherwise args render in a generic `code_block`. Then the
/// streaming stdout/stderr buffer if any chunks have arrived, then
/// the integrated result body once `End` has landed. Sections are
/// separated by small muted captions so the structure reads at a
/// glance.
fn tool_call_body(
    diff: Option<&DiffPayload>,
    args_pretty: Option<&str>,
    streaming_output: &str,
    result: Option<&FusedToolResult>,
) -> Vec<El> {
    let mut blocks: Vec<El> = Vec::new();
    if let Some(d) = diff {
        // The diff replaces the args block — `path` + the
        // unified-diff body already say everything the raw JSON
        // would, plus the visual signal (red/green) of what's
        // actually changing.
        let header_text = if d.is_creation {
            format!("(new) {}", d.path)
        } else {
            d.path.clone()
        };
        blocks.push(text(header_text).caption().muted().bold());
        blocks.push(diff_body(&d.old_text, &d.new_text));
    } else if let Some(args) = args_pretty
        && !args.trim().is_empty()
    {
        blocks.push(text("args").caption().muted());
        blocks.push(code_block(args.to_string()));
    }
    if !streaming_output.trim().is_empty() {
        blocks.push(text("output").caption().muted());
        blocks.push(code_block(streaming_output.to_string()));
    }
    if let Some(r) = result {
        let label = if r.is_error { "error" } else { "result" };
        blocks.push(text(label).caption().muted());
        blocks.push(code_block(r.text.clone()));
    }
    if blocks.is_empty() {
        blocks.push(text("(no detail)").muted());
    }
    blocks
}

/// Mid-saturation diff-line text colors. The destructive /
/// success tokens read poorly on the dark `MUTED` body fill —
/// `DESTRUCTIVE` is dark maroon (intended as a *background* color
/// behind white text), and `SUCCESS` runs hot enough that long
/// stretches of `+` lines visually vibrate. These hues sit closer
/// to GitHub's diff palette: rosy for deletes, mint for adds.
/// Declared as named tokens (rather than raw `Color::rgb`) so the
/// bundle linter accepts them as theme-registered.
const DIFF_DEL_FG: Color = Color::token("diff-del-foreground", 230, 130, 130, 255);
const DIFF_ADD_FG: Color = Color::token("diff-add-foreground", 150, 220, 170, 255);

/// Render a unified line-granularity diff body. Each `similar`
/// change becomes one mono row prefixed with `+` / `-` / ` ` and
/// colored to match the change kind. Equal lines stay muted so
/// the eye lands on the additions / deletions.
fn diff_body(old_text: &str, new_text: &str) -> El {
    use similar::{ChangeTag, TextDiff};

    let text_diff = TextDiff::from_lines(old_text, new_text);
    let mut rows: Vec<El> = Vec::new();
    for change in text_diff.iter_all_changes() {
        let (prefix, color) = match change.tag() {
            ChangeTag::Equal => (' ', tokens::MUTED_FOREGROUND),
            ChangeTag::Delete => ('-', DIFF_DEL_FG),
            ChangeTag::Insert => ('+', DIFF_ADD_FG),
        };
        let raw = change.value();
        let trimmed = raw.strip_suffix('\n').unwrap_or(raw);
        rows.push(
            mono(format!("{prefix}{trimmed}"))
                .small()
                .text_color(color)
                .width(Size::Fill(1.0)),
        );
    }
    column(rows)
        .gap(0.0)
        .padding(Sides::xy(tokens::SPACE_3, tokens::SPACE_2))
        .fill(tokens::MUTED)
        .radius(tokens::RADIUS_MD)
        .width(Size::Fill(1.0))
}

/// Event-log row — narrow role-colored gutter + content with
/// internal padding. Optional `faint_fill` tints the row's
/// background for emphasis (we use it for user turns so they're
/// distinguishable in a long assistant stream). Mirrors the
/// upstream `aetna-core/README.md` "Conversation / event-log row"
/// recipe.
fn log_row(role_color: Color, faint_fill: Option<Color>, content: El) -> El {
    let gutter = El::new(Kind::Custom("log_gutter"))
        .fill(role_color)
        .width(Size::Fixed(3.0))
        .height(Size::Fill(1.0));
    let body = content
        .padding(Sides {
            left: tokens::SPACE_3,
            right: tokens::SPACE_2,
            top: tokens::SPACE_2,
            bottom: tokens::SPACE_2,
        })
        .width(Size::Fill(1.0));
    let row_el = row([gutter, body]).gap(0.0).width(Size::Fill(1.0));
    if let Some(fill) = faint_fill {
        row_el.fill(fill)
    } else {
        row_el
    }
}

/// First non-empty line of `s`, truncated to `max_chars` chars
/// (char-boundary aware). Used as the collapsed accordion label for
/// reasoning rows so the log header gives a hint of what's inside
/// without unfolding the body.
fn first_line_preview(s: &str, max_chars: usize) -> String {
    let line = s
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let chars: Vec<char> = line.chars().collect();
    if chars.len() <= max_chars {
        line.to_string()
    } else {
        let head: String = chars.iter().take(max_chars).collect();
        format!("{head}…")
    }
}

/// Walk a [`Conversation`] into the [`DisplayItem`] shape, fusing
/// `ContentBlock::ToolResult` blocks onto their originating
/// `ContentBlock::ToolUse` when the result is "near" the call (no
/// intervening user / assistant text turn). Mirrors how the egui
/// sibling reads its conversation, just with our smaller item set.
///
/// `turn_log` is interleaved by walking its entries in temporal
/// order alongside the conversation's `Assistant`-role messages —
/// the runtime pushes exactly one entry per
/// `integrate_model_response`, so entry N corresponds to the Nth
/// assistant turn. Older threads (pre-turn-log) load with empty
/// entries; trailing turns past the entries vec just don't get a
/// stats row, which is the right fallback.
fn conversation_to_display_items(
    conv: &whisper_agent_protocol::Conversation,
    turn_log: &whisper_agent_protocol::TurnLog,
) -> Vec<DisplayItem> {
    let mut out: Vec<DisplayItem> = Vec::new();
    let mut entry_iter = turn_log.entries.iter();
    for (msg_index, msg) in conv.messages().iter().enumerate() {
        match msg.role {
            Role::System => {
                // System prompt at the head of the conversation —
                // render as a default-collapsed `SetupPrompt`
                // accordion. Empty prompts produce no row so the
                // log doesn't start with a meaningless "(empty)"
                // entry. Mirrors the egui sibling.
                if let Some(t) = first_text(&msg.content)
                    && !t.is_empty()
                {
                    out.push(DisplayItem::SetupPrompt { text: t });
                }
            }
            Role::Tools => {
                // Tool manifest — fold every advertised
                // `ContentBlock::ToolSchema` into a single
                // default-collapsed `SetupTools` row whose header
                // counts entries and whose body lists per-tool
                // name + description. Empty manifests skipped for
                // the same reason as empty system prompts.
                let entries: Vec<ToolSchemaSummary> = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolSchema {
                            name, description, ..
                        } => Some(ToolSchemaSummary {
                            name: name.clone(),
                            description: description.clone(),
                        }),
                        _ => None,
                    })
                    .collect();
                if !entries.is_empty() {
                    out.push(DisplayItem::SetupTools { entries });
                }
            }
            Role::User => {
                for block in &msg.content {
                    push_block(block, true, msg_index, &mut out);
                }
            }
            Role::Assistant => {
                for block in &msg.content {
                    push_block(block, false, msg_index, &mut out);
                }
                // Pull the next turn-log entry — one per assistant
                // response — and append the usage row so it lands
                // directly under the turn's content. A short log
                // (older threads, mid-migration) leaves trailing
                // turns without a stats row rather than crashing.
                if let Some(entry) = entry_iter.next() {
                    out.push(DisplayItem::TurnStats { usage: entry.usage });
                }
            }
            Role::ToolResult => {
                for block in &msg.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } = block
                    {
                        let text = tool_result_text_summary(content);
                        if !try_fuse_tool_result(&mut out, tool_use_id, &text, *is_error) {
                            out.push(DisplayItem::ToolResult {
                                tool_use_id: tool_use_id.clone(),
                                text,
                                is_error: *is_error,
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

fn push_block(block: &ContentBlock, user_role: bool, msg_index: usize, out: &mut Vec<DisplayItem>) {
    match block {
        ContentBlock::Text { text } => {
            if !text.is_empty() {
                if user_role {
                    out.push(DisplayItem::User {
                        text: text.clone(),
                        msg_index,
                    });
                } else {
                    out.push(DisplayItem::Assistant { text: text.clone() });
                }
            }
        }
        ContentBlock::Thinking { thinking, .. } => {
            if !thinking.is_empty() {
                out.push(DisplayItem::Reasoning {
                    text: thinking.clone(),
                });
            }
        }
        ContentBlock::ToolUse {
            id, name, input, ..
        } => {
            let args_pretty = serde_json::to_string_pretty(input).ok();
            let summary = tool_summary_from_args(name, Some(input));
            let diff = extract_diff(name, Some(input));
            out.push(DisplayItem::ToolCall {
                tool_use_id: id.clone(),
                name: name.clone(),
                summary,
                diff,
                args_pretty,
                streaming_output: String::new(),
                result: None,
            });
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            let text = tool_result_text_summary(content);
            if !try_fuse_tool_result(out, tool_use_id, &text, *is_error) {
                out.push(DisplayItem::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    text,
                    is_error: *is_error,
                });
            }
        }
        ContentBlock::Image { source, .. } => {
            out.push(DisplayItem::Image {
                is_user: user_role,
                state: decode_image_source(source),
            });
        }
        ContentBlock::Document { .. } => {
            out.push(DisplayItem::GenericPlaceholder {
                label: "[document]".into(),
            });
        }
        ContentBlock::ToolSchema { name, .. } => {
            out.push(DisplayItem::GenericPlaceholder {
                label: format!("tool schema: {name}"),
            });
        }
    }
}

/// Try to attach a tool result to its originating call, walking
/// backward from the end of `out`. Returns `true` if the result was
/// fused (caller should not push a standalone `ToolResult`); `false`
/// if no matching open call was found and the caller should fall
/// through to a standalone row.
///
/// "Backward" rather than "any" because ordering matches the
/// conversation log: the most recent unresolved call is the right
/// match. We stop at user / assistant text rows to keep an async
/// dispatch_thread callback that arrives across a turn boundary
/// from being incorrectly fused with an ancient call.
fn try_fuse_tool_result(
    out: &mut [DisplayItem],
    tool_use_id: &str,
    text: &str,
    is_error: bool,
) -> bool {
    for item in out.iter_mut().rev() {
        match item {
            DisplayItem::ToolCall {
                tool_use_id: id,
                result,
                ..
            } if id == tool_use_id && result.is_none() => {
                *result = Some(FusedToolResult {
                    text: text.to_string(),
                    is_error,
                });
                return true;
            }
            DisplayItem::User { .. } | DisplayItem::Assistant { .. } => break,
            _ => {}
        }
    }
    false
}

fn first_text(blocks: &[ContentBlock]) -> Option<String> {
    blocks.iter().find_map(|b| match b {
        ContentBlock::Text { text } => Some(text.clone()),
        _ => None,
    })
}

fn tool_result_text_summary(content: &whisper_agent_protocol::ToolResultContent) -> String {
    use whisper_agent_protocol::ToolResultContent;
    match content {
        ToolResultContent::Text(text) => preview(text, 200),
        ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .find_map(|b| match b {
                ContentBlock::Text { text } => Some(preview(text, 200)),
                _ => None,
            })
            .unwrap_or_default(),
    }
}

fn preview(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else {
        let head: String = chars.iter().take(max_chars).collect();
        format!("{head}…")
    }
}
