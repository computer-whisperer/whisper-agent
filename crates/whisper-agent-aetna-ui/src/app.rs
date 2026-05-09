//! [`ChatApp`] — the platform-agnostic aetna [`App`] for the
//! whisper-agent chat client.
//!
//! Surface so far: connection-status banner, sidebar with pods +
//! threads, snapshot + delta-streamed chat log rendered as event-log
//! rows (3px role-colored gutter + content), assistant turns through
//! `aetna_markdown::md`, reasoning + tool rows in collapsible
//! `accordion_item`s, compose box that sends follow-up turns to the
//! selected thread.
//! Build/on_event split is the load-bearing test of the pivot — every
//! interactive element routes through [`ChatApp::on_event`] via a
//! key, every visual is a function of state read in
//! [`ChatApp::build`].
//!
//! Things deliberately deferred:
//! - inline diff rendering for `edit_file` / `write_file` tool calls
//!   (currently shown as raw JSON args)
//! - prefill progress + image output streaming
//! - "compose into a fresh thread" (needs the model picker / pod
//!   target / bindings surface)
//! - attachment staging + drag/drop/paste/filepicker
//! - per-thread draft persistence (`SetThreadDraft` debounce)
//! - all settings / behavior / pod / bucket / fork modals
//!
//! Dispatch model: a single `dispatch_wire` walks `ServerToClient`
//! variants — only the ones the current stage cares about have arms;
//! the rest drop on the floor.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use aetna_core::prelude::*;
use whisper_agent_protocol::{
    ClientToServer, ContentBlock, PodSummary, Role, ServerToClient, ThreadSummary,
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
    SystemNote {
        text: String,
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
    /// Anything else we don't yet render purpose-built (images,
    /// documents, tool-schema entries) — collapsed into a single
    /// annotated row. Future stages break these out.
    GenericPlaceholder {
        label: String,
    },
}

/// Tool-result body fused onto its originating [`DisplayItem::ToolCall`].
struct FusedToolResult {
    text: String,
    is_error: bool,
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
    /// In-progress text in the compose box. Cleared on send. Stage 3
    /// keeps a single buffer rather than per-thread drafts; the egui
    /// sibling persists drafts via `SetThreadDraft` debounce — that's
    /// a stage-4 concern once we've validated the basic send path.
    compose_input: String,
    /// Global text-selection cursor for `text_area` / `text_input`
    /// widgets. Aetna's controlled widgets read it through
    /// `App::selection` and write back via `apply_event`. Stage 3
    /// only has the compose `text_area`, but the same cursor will
    /// own selection in the chat log when `.selectable()` lands.
    selection: Selection,

    // ----- accordion state -----
    /// Open accordion items, keyed by the routed key string the
    /// accordion runtime emits (`{group}:accordion:{value}` —
    /// produced by [`aetna_core::widgets::accordion::accordion_item_key`]).
    /// Reasoning and tool rows go through this; the egui sibling's
    /// per-row `Option<bool>` shape doesn't generalize when we have
    /// many independent rows in a single thread.
    open_accordions: HashSet<String>,
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
            selected: None,
            views: HashMap::new(),
            subscribed: HashSet::new(),
            compose_input: String::new(),
            selection: Selection::default(),
            open_accordions: HashSet::new(),
        }
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
                if !self.list_requested {
                    self.send(ClientToServer::ListPods {
                        correlation_id: None,
                    });
                    self.send(ClientToServer::ListThreads {
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
            ServerToClient::PodList { pods, .. } => {
                self.pods = pods.into_iter().map(|p| (p.pod_id.clone(), p)).collect();
            }
            ServerToClient::PodCreated { pod, .. } => {
                self.pods.insert(pod.pod_id.clone(), pod);
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
                self.threads.insert(summary.thread_id.clone(), summary);
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
                let items = conversation_to_display_items(&snapshot.conversation);
                let view = ThreadView {
                    items,
                    title: snapshot.title,
                };
                self.views.insert(thread_id, view);
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
                    let entry = DisplayItem::ToolCall {
                        tool_use_id,
                        name,
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
                            args_pretty: None,
                            streaming_output: String::new(),
                            result: None,
                        });
                    }
                }
            }
            // Per-turn append events not yet surfaced.
            ServerToClient::ThreadToolResultMessage { .. }
            | ServerToClient::ThreadAssistantBegin { .. }
            | ServerToClient::ThreadAssistantImage { .. }
            | ServerToClient::ThreadAssistantEnd { .. }
            | ServerToClient::ThreadLoopComplete { .. }
            | ServerToClient::ThreadDraftUpdated { .. }
            | ServerToClient::ThreadCompacted { .. }
            | ServerToClient::ThreadPrefillProgress { .. } => {}
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
        // Root is an overlay stack so future stages can append
        // toast / tooltip / popover layers without revisiting the
        // root shape.
        overlays(
            row([self.sidebar(), self.content()])
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0))
                .gap(0.0),
            std::iter::empty::<Option<El>>(),
        )
    }

    fn on_event(&mut self, event: UiEvent) {
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

        // Compose `text_area` edits: when the routed target is the
        // compose box, fold the event through aetna's controlled-
        // widget helper, which handles caret/selection/clipboard.
        if event.target_key() == Some(COMPOSE_KEY) {
            text_area::apply_event(
                &mut self.compose_input,
                &mut self.selection,
                COMPOSE_KEY,
                &event,
            );
            return;
        }

        // Send button.
        if event.is_click_or_activate(SEND_KEY) {
            self.send_compose();
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

impl ChatApp {
    fn send_compose(&mut self) {
        // Stage 3 only supports follow-up turns on an already-existing
        // thread. Creating a fresh thread from the compose box (the
        // egui sibling's `composing_new` mode) is a stage-4 concern
        // since it needs the model picker / pod target / bindings
        // surface. For now: no thread selected = nothing to send.
        let Some(thread_id) = self.selected.clone() else {
            return;
        };
        let text = std::mem::take(&mut self.compose_input).trim().to_string();
        if text.is_empty() {
            return;
        }
        self.send(ClientToServer::SendUserMessage {
            thread_id,
            text,
            attachments: Vec::new(),
        });
        // Reset selection inside the now-empty compose box so the
        // caret lands at offset 0 on the next frame.
        self.selection = Selection::default();
    }

    fn select_thread(&mut self, thread_id: String) {
        self.selected = Some(thread_id.clone());
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
        // `sidebar([...])` bundles the canonical Panel-surface
        // recipe: width = `tokens::SIDEBAR_WIDTH`, fill/stroke from
        // theme tokens, default padding + gap. We compose its
        // children from `sidebar_header` + per-pod `sidebar_group`.
        let mut entries: Vec<El> = vec![sidebar_header([row([
            text("whisper-agent").title().width(Size::Fill(1.0)),
            self.connection_badge(),
        ])
        .align(Align::Center)
        .gap(tokens::SPACE_2)
        .width(Size::Fill(1.0))])];

        // Pods sorted: non-archived first, then by pod_id for stability.
        // Within each pod, threads sorted by last_active descending so
        // the most recently touched thread floats to the top.
        let mut pods: Vec<&PodSummary> = self.pods.values().collect();
        pods.sort_by(|a, b| {
            a.archived
                .cmp(&b.archived)
                .then_with(|| a.pod_id.cmp(&b.pod_id))
        });

        if pods.is_empty() {
            entries.push(text("no pods yet").muted().small());
        }

        for pod in pods {
            let mut group_children: Vec<El> = vec![sidebar_group_label(pod_label(pod))];

            let mut pod_threads: Vec<&ThreadSummary> = self
                .threads
                .values()
                .filter(|t| t.pod_id == pod.pod_id)
                .collect();
            pod_threads.sort_by(|a, b| b.last_active.cmp(&a.last_active));

            if pod_threads.is_empty() {
                group_children.push(text("no threads").muted().xsmall());
            } else {
                let menu_items: Vec<El> = pod_threads
                    .iter()
                    .map(|t| sidebar_menu_item(self.thread_row(t)))
                    .collect();
                group_children.push(sidebar_menu(menu_items));
            }

            entries.push(sidebar_group(group_children));
        }

        sidebar(entries)
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

    fn thread_row(&self, t: &ThreadSummary) -> El {
        let key = format!("thread:{}", t.thread_id);
        let current = self.selected.as_deref() == Some(t.thread_id.as_str());
        // `sidebar_menu_button` handles the `current` styling itself
        // (Surface/Current vs Ghost). We tack the routing key on
        // after construction. The displayed label is trimmed to keep
        // the fixed sidebar width — `ellipsis()` would also work but
        // a hard char cut keeps the dump deterministic.
        sidebar_menu_button(thread_label(t), current).key(key)
    }

    fn content(&self) -> El {
        let inner = match self.selected.as_ref() {
            None => column([
                h2("no thread selected"),
                text("pick a thread from the sidebar to view its log").muted(),
            ])
            .gap(tokens::SPACE_3)
            .padding(tokens::SPACE_6)
            .align(Align::Start),
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
        if let Some(s) = summary {
            toolbar_children.push(state_badge(s.state));
        }
        let header = column([toolbar(toolbar_children), text(thread_id).muted().xsmall()])
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

    fn compose_box(&self) -> El {
        // Disable the send button while the input is whitespace-only —
        // sending an empty message wastes an LLM turn.
        let can_send = !self.compose_input.trim().is_empty();
        let mut send = button("Send").key(SEND_KEY).primary();
        if !can_send {
            // Aetna's button widget honors `.disabled()` to gray out
            // the surface and skip routing.
            send = send.disabled();
        }
        let editor =
            text_area(&self.compose_input, &self.selection, COMPOSE_KEY).height(Size::Fixed(120.0));

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

fn pod_label(pod: &PodSummary) -> String {
    if pod.archived {
        format!("{} (archived)", pod.name)
    } else {
        pod.name.clone()
    }
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

fn thread_label(t: &ThreadSummary) -> String {
    let title = t.title.as_deref().unwrap_or("untitled");
    // Sidebar width is fixed; trim long titles so they don't push
    // the row wider and force the button to ellipsize awkwardly.
    let trimmed = take_chars(title, 32);
    if trimmed.len() < title.len() {
        format!("{trimmed}…")
    } else {
        trimmed
    }
}

/// Take up to `n` chars from a string, respecting char boundaries.
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
            DisplayItem::SystemNote { text: t } => {
                let body = paragraph(t.clone());
                log_row(tokens::WARNING, None, body)
            }
            DisplayItem::ToolCall {
                tool_use_id,
                name,
                args_pretty,
                streaming_output,
                result,
            } => {
                let key = "tool";
                let value = format!("{idx}");
                let routed = accordion_item_key(key, &value);
                let open = self.open_accordions.contains(&routed);
                let header = tool_call_header(name, result.as_ref());
                let body_blocks =
                    tool_call_body(args_pretty.as_deref(), streaming_output, result.as_ref());
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
        }
    }
}

/// Collapsed-header label for a tool call. When the result has
/// arrived we show a compact status hint so a closed row
/// communicates pass / fail at a glance.
fn tool_call_header(name: &str, result: Option<&FusedToolResult>) -> String {
    match result {
        None => format!("⏳ {name}"),
        Some(r) if r.is_error => format!("✗ {name}"),
        Some(_) => format!("✓ {name}"),
    }
}

/// Expanded-body content for a tool-call accordion. Args render in a
/// `code_block` (mono, scrollable horizontally), then the streaming
/// stdout/stderr buffer if any chunks have arrived, then the
/// integrated result body once `End` has landed. Sections are
/// separated by small muted captions so the structure reads at a
/// glance.
fn tool_call_body(
    args_pretty: Option<&str>,
    streaming_output: &str,
    result: Option<&FusedToolResult>,
) -> Vec<El> {
    let mut blocks: Vec<El> = Vec::new();
    if let Some(args) = args_pretty
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
fn conversation_to_display_items(conv: &whisper_agent_protocol::Conversation) -> Vec<DisplayItem> {
    let mut out: Vec<DisplayItem> = Vec::new();
    for msg in conv.messages() {
        match msg.role {
            Role::System => {
                if let Some(t) = first_text(&msg.content)
                    && !t.is_empty()
                {
                    out.push(DisplayItem::SystemNote { text: t });
                }
            }
            Role::Tools => {
                let names: Vec<String> = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolSchema { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect();
                if !names.is_empty() {
                    out.push(DisplayItem::SystemNote {
                        text: format!("tools available: {}", names.join(", ")),
                    });
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
            out.push(DisplayItem::ToolCall {
                tool_use_id: id.clone(),
                name: name.clone(),
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
        ContentBlock::Image { .. } => {
            out.push(DisplayItem::GenericPlaceholder {
                label: "[image]".into(),
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
