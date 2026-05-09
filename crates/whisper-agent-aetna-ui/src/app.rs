//! [`ChatApp`] — the platform-agnostic aetna [`App`] for the
//! whisper-agent chat client.
//!
//! Surface so far: connection-status banner, sidebar with pods +
//! threads, snapshot + delta-streamed chat log rendered as event-log
//! rows (3px role-colored gutter + content), assistant turns through
//! `aetna_markdown::md`, reasoning + tool rows in collapsible
//! `accordion_item`s, decoded inline images (PNG/JPEG/WebP/GIF)
//! rendered through aetna's `image()` widget, compose box that
//! sends follow-up turns to the selected thread, per-thread compose
//! drafts persisted via `SetThreadDraft`, prefill-progress indicator,
//! and a new-thread compose form (card + form + `select_trigger`
//! pickers for backend / model / pod) reachable when no thread is
//! selected.
//! Build/on_event split is the load-bearing test of the pivot — every
//! interactive element routes through [`ChatApp::on_event`] via a
//! key, every visual is a function of state read in
//! [`ChatApp::build`].
//!
//! Things deliberately deferred:
//! - inline diff rendering for `edit_file` / `write_file` tool calls
//!   (currently shown as raw JSON args)
//! - bindings surface (knowledge-db / behavior / shared MCP hosts) on
//!   the new-thread form
//! - attachment staging input side (drag/drop/paste/filepicker) —
//!   blocked on aetna-winit-wgpu exposing winit drop / clipboard
//!   image events; rendering inbound images already works
//! - URL-source image fetching (currently a muted placeholder)
//! - all settings / behavior / pod / bucket / fork modals
//!
//! Dispatch model: a single `dispatch_wire` walks `ServerToClient`
//! variants — only the ones the current stage cares about have arms;
//! the rest drop on the floor.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use aetna_core::prelude::*;
use aetna_core::widgets::select::{SelectAction, classify_event as classify_select_event};
use whisper_agent_protocol::{
    AllowMap, BackendSummary, BehaviorConfig, BehaviorSummary, BehaviorThreadOverride,
    ClientToServer, ContentBlock, ImageSource, ModelSummary, NamedHostEnv, PodAllow, PodConfig,
    PodLimits, PodSummary, RetentionPolicy, Role, ServerToClient, ThreadConfigOverride,
    ThreadDefaults, ThreadSummary, TriggerSpec,
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

/// Three terminal states for an inline image: successfully decoded
/// (we hold the `aetna_core::Image` and the original dimensions),
/// remote URL (deferred — Stage 7 doesn't fetch yet), or a decode
/// failure (we kept the reason for surfacing in the UI).
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
}

pub struct ChatApp {
    // ----- transport bridge -----
    inbound: Inbound,
    send_fn: SendFn,

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

impl ChatApp {
    pub fn new(inbound: Inbound, send_fn: SendFn) -> Self {
        Self {
            inbound,
            send_fn,
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
            models_by_backend: HashMap::new(),
            requested_models_for: HashSet::new(),
            selected: None,
            views: HashMap::new(),
            subscribed: HashSet::new(),
            compose_input: String::new(),
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
            correlation_counter: 0,
        }
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
            ServerToClient::BehaviorUpdated { snapshot, .. } => {
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
            ServerToClient::ThreadCreated { summary, .. } => {
                let new_id = summary.thread_id.clone();
                self.threads.insert(new_id.clone(), summary);
                // If the user just submitted the new-thread form (no
                // selection currently), auto-select the freshly
                // created thread so the post-create transition lands
                // them in the live conversation. Skipped when another
                // thread is already selected so a background create
                // (e.g. dispatch_thread) doesn't yank focus away.
                if self.selected.is_none() {
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
            ServerToClient::ModelsList {
                backend, models, ..
            } => {
                self.models_by_backend.insert(backend, models);
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
                let view = ThreadView {
                    items,
                    title: snapshot.title,
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
                thread_id, text, ..
            } => {
                if let Some(view) = self.views.get_mut(&thread_id)
                    && !text.is_empty()
                {
                    view.items.push(DisplayItem::User { text });
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
                }
            }
            // Per-turn append events not yet surfaced.
            ServerToClient::ThreadToolResultMessage { .. }
            | ServerToClient::ThreadAssistantBegin { .. }
            | ServerToClient::ThreadLoopComplete { .. }
            | ServerToClient::ThreadCompacted { .. } => {}
            ServerToClient::Error {
                correlation_id,
                message,
                ..
            } => {
                // Route the error to whichever modal claimed the
                // matching correlation. Each modal owns its own
                // `pending_correlation` slot; whoever it matches
                // surfaces the message and clears the pending so
                // the form re-enables for retry.
                if let Some(corr) = correlation_id.as_ref() {
                    if let Some(modal) = self.new_pod_modal.as_mut()
                        && modal.pending_correlation.as_ref() == Some(corr)
                    {
                        modal.error = Some(message);
                        modal.pending_correlation = None;
                    } else if let Some(modal) = self.new_behavior_modal.as_mut()
                        && modal.pending_correlation.as_ref() == Some(corr)
                    {
                        modal.error = Some(message);
                        modal.pending_correlation = None;
                    }
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
    }

    fn build(&self, _cx: &BuildCx) -> El {
        // Root is an overlay stack: the main row carries the chrome,
        // any open `select_menu` rides above it as a popover layer.
        // Per `widgets::select` doc, the menu must sit at the El
        // tree root so it paints over content and intercepts the
        // dismiss scrim's click — we collect them here.
        overlays(
            row([self.sidebar(), self.content()])
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0))
                .gap(0.0),
            self.popover_layers(),
        )
    }

    fn on_event(&mut self, event: UiEvent) {
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

        // Compose `text_area` edits: when the routed target is the
        // compose box, fold the event through aetna's controlled-
        // widget helper. The buffer is per-thread when a thread is
        // selected (so drafts persist across switches and survive
        // reload via `SetThreadDraft`), or `compose_input` for the
        // new-thread form.
        if event.target_key() == Some(COMPOSE_KEY) {
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

        // Send button.
        if event.is_click_or_activate(SEND_KEY) {
            self.send_compose();
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

        // Accordion toggles (reasoning + tool rows). The accordion
        // runtime emits routed keys shaped `{group}:accordion:{value}`;
        // we just toggle membership of the routed key in our open set
        // — independent toggles per row, no single-active enforcement.
        if matches!(event.kind, UiEventKind::Click | UiEventKind::Activate)
            && let Some(key) = event.route()
            && key.contains(":accordion:")
        {
            if !self.open_accordions.remove(key) {
                self.open_accordions.insert(key.to_string());
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
        Theme::aetna_dark()
    }
}

const COMPOSE_KEY: &str = "compose";
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

/// Composite expansion-state key for `expanded_behaviors`. Mirrors
/// the route key but without the prefix — the set is internal,
/// so the leading `behavior-row:` isn't load-bearing.
fn behavior_expand_key(pod_id: &str, behavior_id: &str) -> String {
    format!("{pod_id}::{behavior_id}")
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
    fn send_compose(&mut self) {
        let text = self.active_compose_text().trim().to_string();
        if text.is_empty() {
            return;
        }
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
                attachments: Vec::new(),
            });
        } else {
            // No selection -> the compose form is in new-thread
            // mode. Materialize the picker state into a
            // `CreateThread` request. `ThreadCreated` will land in
            // `dispatch_wire` and auto-select the result.
            let (config_override, pod_id) = self.build_creation_request();
            self.compose_input.clear();
            self.send(ClientToServer::CreateThread {
                correlation_id: None,
                pod_id,
                initial_message: text,
                initial_attachments: Vec::new(),
                config_override,
                bindings_request: None,
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

    /// Pick the best (`config_override`, `pod_id`) pair for the next
    /// `CreateThread` from the compose-form picker state. Mirrors the
    /// egui sibling's `build_creation_override` shape so server-side
    /// behavior stays identical between the two clients.
    fn build_creation_request(&self) -> (Option<ThreadConfigOverride>, Option<String>) {
        // If the user picked a backend but didn't touch the model
        // dropdown, pin down the model explicitly so the server
        // doesn't fall back to the *default backend's* default_model
        // (which would be wrong for the picked backend). Prefer the
        // picked backend's `default_model`; else the first model the
        // backend's `/models` returned; else `None`.
        let model = self.picker_model.clone().or_else(|| {
            let b = self.picker_backend.as_ref()?;
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
        (config_override, self.picker_pod.clone())
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
        // Sidebar header: title fills available width, "+" entry
        // point for new pods sits inline before the connection
        // badge so both pieces of chrome cluster on the right edge.
        // The "+" is the global pod-creation affordance — it's
        // pod-tab-agnostic (the threads-section "+" is per-pod), so
        // putting it in the sidebar header keeps the two scopes
        // visually distinct.
        let mut entries: Vec<El> = vec![sidebar_header([row([
            // Title eats the slack and ellipses if the right-hand
            // chrome grows past the remaining width (e.g. the wider
            // "connecting…" badge during early connect). The app
            // name is fixed and contextual; clipping it is fine.
            text("whisper-agent")
                .title()
                .ellipsis()
                .width(Size::Fill(1.0)),
            icon_button("plus")
                .key(SIDEBAR_NEW_POD_KEY)
                .ghost()
                .icon_size(tokens::ICON_XS),
            self.connection_badge(),
        ])
        .align(Align::Center)
        .gap(tokens::SPACE_2)
        .width(Size::Fill(1.0))])];

        if self.pods.is_empty() {
            entries.push(text("no pods yet").muted().small());
            return sidebar(entries);
        }

        entries.push(self.pod_tabs());

        if let Some(active) = self.pod_tab.as_deref() {
            // Threads (interactive — `origin == None`) come first;
            // they're the day-to-day reading order. Behavior-spawned
            // threads (`origin == Some(behavior_id)`) nest under
            // their parent behavior in the section below, since
            // grouping per-behavior runs is more useful than mixing
            // them into the interactive list. Mirrors the egui
            // sibling's partition.
            entries.push(self.threads_section(active));

            // Render the section once the per-pod `BehaviorList`
            // round-trip has landed — even if the list is empty,
            // since the header carries the "+ New behavior"
            // affordance the user still needs. Skip entirely
            // before the round-trip arrives so a "just connected"
            // sidebar doesn't flash an empty section that
            // immediately repopulates a frame later.
            if let Some(behaviors) = self.behaviors_by_pod.get(active) {
                entries.push(self.behaviors_section(active, behaviors));
            }
        }

        sidebar(entries)
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
                text("no behaviors in this pod yet")
                    .muted()
                    .small()
                    .padding(Sides::xy(tokens::SPACE_2, tokens::SPACE_1)),
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
    /// Two-row layout: the top row carries the everyday Run-now and
    /// Pause/Resume buttons; the bottom row carries a destructive
    /// Delete with two-click arm-confirm. First click flips the
    /// label to "Confirm delete?" and the styling to destructive;
    /// second click on the same button fires `DeleteBehavior`. Any
    /// unrelated click disarms (handled at the top of `on_event`).
    /// Two rows because the sidebar's 224 px isn't wide enough to
    /// hold all three buttons side-by-side once the armed Delete
    /// label expands. The visual split also signals "this is the
    /// danger zone" without needing an extra divider. Edit still
    /// pending (needs the full behavior editor).
    fn behavior_actions_row(&self, pod_id: &str, b: &BehaviorSummary) -> El {
        let mut run = button("Run now")
            .key(behavior_run_key(pod_id, &b.behavior_id))
            .ghost();
        if b.load_error.is_some() {
            run = run.disabled();
        }
        let toggle_label = if b.enabled { "Pause" } else { "Resume" };
        let toggle = button(toggle_label)
            .key(behavior_toggle_key(pod_id, &b.behavior_id))
            .ghost();

        let armed = self
            .delete_armed_behavior
            .as_ref()
            .map(|(p, bb)| p == pod_id && bb == &b.behavior_id)
            .unwrap_or(false);
        // The color shift is the load-bearing arm signal: idle is a
        // muted ghost button that visually matches Run / Pause, so
        // the user doesn't accidentally hit a loud "Delete" when
        // they meant to scroll. Click once → solid destructive
        // fill plus the wider "Confirm delete?" label, which makes
        // the second click an obvious deliberate gesture.
        // (`.ghost().destructive()` doesn't compose the way it
        // looks — `destructive()` runs `tint()` which writes back
        // to `fill` regardless of ghost, so the idle styling has
        // to be plain ghost.)
        let delete = if armed {
            button("Confirm delete?")
                .key(behavior_delete_key(pod_id, &b.behavior_id))
                .destructive()
        } else {
            button("Delete")
                .key(behavior_delete_key(pod_id, &b.behavior_id))
                .ghost()
        };

        // Indent matches the nested-thread depth=1 left-pad so the
        // toolbar visually belongs to the expanded body. Two rows:
        // actions left-aligned on top, destructive right-aligned
        // below. Tight `gap(0)` between the sub-rows so the toolbar
        // doesn't eat extra vertical space — the row backgrounds
        // are transparent, the visual separation is the spacer +
        // right-aligned destructive cluster.
        let pad = Sides {
            left: tokens::SPACE_3 + tokens::SPACE_3,
            right: tokens::SPACE_3,
            top: 0.0,
            bottom: tokens::SPACE_1,
        };
        column([
            row([run, toggle])
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            row([spacer(), delete])
                .align(Align::Center)
                .width(Size::Fill(1.0)),
        ])
        .gap(0.0)
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
            group_children.push(
                text("no threads in this pod yet")
                    .muted()
                    .small()
                    .padding(Sides::xy(tokens::SPACE_2, tokens::SPACE_1)),
            );
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

    fn connection_badge(&self) -> El {
        let label = self.conn_status.label();
        let b = badge(label);
        match self.conn_status {
            ConnectionStatus::Connected => b.success(),
            ConnectionStatus::Connecting => b.muted(),
            ConnectionStatus::Closed => b.warning(),
            ConnectionStatus::Error => b.destructive(),
        }
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

    fn content(&self) -> El {
        let inner = match self.selected.as_ref() {
            None => self.new_thread_pane(),
            Some(thread_id) => self.thread_pane(thread_id),
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

    fn thread_pane(&self, thread_id: &str) -> El {
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
        }
        let mut header_rows: Vec<El> =
            vec![toolbar(toolbar_children), text(thread_id).muted().xsmall()];
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
            None => empty_state("loading…"),
            Some(v) if v.items.is_empty() => empty_state("no messages yet"),
            Some(v) => {
                // Thread the row index in as the accordion `value` so
                // each reasoning / tool row has an independent open
                // state stable across rebuilds.
                let rows: Vec<El> = v
                    .items
                    .iter()
                    .enumerate()
                    .map(|(idx, item)| self.event_log_row(idx, item))
                    .collect();
                // Inter-row breathing room. Upstream's README example
                // has no explicit gap; with the row content sitting
                // flush, adjacent unfilled rows visually blur into one
                // text flow. SPACE_2 (8px) reads as a log-style line
                // gap rather than card-isolation.
                scroll(rows)
                    .key(scroll_key)
                    .gap(tokens::SPACE_2)
                    .padding(tokens::SPACE_2)
                    .width(Size::Fill(1.0))
                    .height(Size::Fill(1.0))
            }
        };

        column([header, body, self.compose_box()])
            .gap(0.0)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
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
        // Disable the send button while the input is whitespace-only —
        // sending an empty message wastes an LLM turn.
        let buf = self.active_compose_text();
        let can_send = !buf.trim().is_empty();
        let mut send = button("Send").key(SEND_KEY).primary();
        if !can_send {
            // Aetna's button widget honors `.disabled()` to gray out
            // the surface and skip routing.
            send = send.disabled();
        }
        let editor = text_area(buf, &self.selection, COMPOSE_KEY).height(Size::Fixed(120.0));

        // The compose row sits at the bottom of the pane: text_area
        // takes the leftover width, the send button hugs to the
        // right. `card([...])` would over-style this — it's part of
        // the same content surface — so we use a thin top stroke as
        // the visual divider from the chat log.
        row([editor, send])
            .gap(tokens::SPACE_3)
            .padding(tokens::SPACE_3)
            .align(Align::End)
            .width(Size::Fill(1.0))
            .stroke(tokens::BORDER)
            .fill(tokens::BACKGROUND)
    }
}

fn empty_state(label: &str) -> El {
    column([text(label).muted()])
        .padding(tokens::SPACE_6)
        .align(Align::Center)
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
    fn event_log_row(&self, idx: usize, item: &DisplayItem) -> El {
        match item {
            DisplayItem::User { text: t } => {
                let body = paragraph(t.clone());
                // Upstream README's worked example uses
                // `with_alpha(18)` (~7%) for the user fill — too low
                // to be visible against the dark zinc-950 background.
                // Bump to ~25% so user turns actually break up a long
                // assistant stream. Subtle enough to still read as
                // "log entry, not card".
                log_row(tokens::INFO, Some(tokens::INFO.with_alpha(64)), body)
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
                if *is_user {
                    log_row(gutter, Some(tokens::INFO.with_alpha(64)), body)
                } else {
                    log_row(gutter, None, body)
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
    for msg in conv.messages() {
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
                    push_block(block, true, &mut out);
                }
            }
            Role::Assistant => {
                for block in &msg.content {
                    push_block(block, false, &mut out);
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

fn push_block(block: &ContentBlock, user_role: bool, out: &mut Vec<DisplayItem>) {
    match block {
        ContentBlock::Text { text } => {
            if !text.is_empty() {
                if user_role {
                    out.push(DisplayItem::User { text: text.clone() });
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
