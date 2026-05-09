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
//! - tool calls + tool results — currently a single accordion row
//!   per non-text block (placeholder text); stage 6 breaks them
//!   into purpose-built rows with diff rendering, streaming output,
//!   etc.
//! - tool-streaming, prefill progress, image output streaming
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

/// One row in the chat log. Stage 2 only fills the text variants;
/// `ToolCallPlaceholder` is the catch-all that collapses every
/// non-text content block down to a single annotated row so a thread
/// with tool calls still renders as something instead of as a hole.
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
    /// Any non-text block (ToolUse, ToolResult, Image, Document,
    /// ToolSchema, …) collapsed to a single annotated row. Stage 3+
    /// breaks these out into purpose-built variants.
    ToolCallPlaceholder {
        label: String,
    },
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
            // Tool-call streaming, prefill progress, image output, and
            // the begin/end bookends remain stage-4 concerns. They
            // arrive but don't surface yet.
            ServerToClient::ThreadToolResultMessage { .. }
            | ServerToClient::ThreadAssistantBegin { .. }
            | ServerToClient::ThreadAssistantImage { .. }
            | ServerToClient::ThreadAssistantEnd { .. }
            | ServerToClient::ThreadToolCallStreaming { .. }
            | ServerToClient::ThreadToolCallBegin { .. }
            | ServerToClient::ThreadToolCallContent { .. }
            | ServerToClient::ThreadToolCallEnd { .. }
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
                scroll(rows)
                    .key(scroll_key)
                    .gap(0.0)
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
                log_row(tokens::INFO, Some(tokens::INFO.with_alpha(18)), body)
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
            DisplayItem::ToolCallPlaceholder { label } => {
                let key = "tool";
                let value = format!("{idx}");
                let routed = accordion_item_key(key, &value);
                let open = self.open_accordions.contains(&routed);
                let item_el =
                    accordion_item(key, value, label.clone(), open, [paragraph(label.clone())]);
                log_row(tokens::WARNING, None, item_el)
            }
        }
    }
}

/// Event-log row — narrow role-colored gutter + content with
/// internal padding. Optional `faint_fill` tints the row's
/// background for emphasis (we use it for user turns so they're
/// distinguishable in a long assistant stream).
fn log_row(role_color: Color, faint_fill: Option<Color>, content: El) -> El {
    let gutter = El::new(Kind::Custom("log_gutter"))
        .fill(role_color)
        .width(Size::Fixed(3.0))
        .height(Size::Fill(1.0));
    let padded_content = content
        .padding(Sides {
            left: tokens::SPACE_3,
            right: tokens::SPACE_2,
            top: tokens::SPACE_2,
            bottom: tokens::SPACE_2,
        })
        .width(Size::Fill(1.0));
    let row_el = row([gutter, padded_content])
        .gap(0.0)
        .width(Size::Fill(1.0));
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

/// Walk a [`Conversation`] into the lightweight stage-2 [`DisplayItem`]
/// shape. One item per content block (matching the egui sibling); we
/// don't yet fuse tool calls with their results.
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
                // Surface tool-result text under a generic "tool"
                // label until stage 3's tool-call fusion lands.
                for block in &msg.content {
                    if let ContentBlock::ToolResult { content, .. } = block {
                        let text = tool_result_text_summary(content);
                        if !text.is_empty() {
                            out.push(DisplayItem::ToolCallPlaceholder {
                                label: format!("tool result: {text}"),
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
        ContentBlock::ToolUse { name, .. } => {
            out.push(DisplayItem::ToolCallPlaceholder {
                label: format!("tool call: {name}"),
            });
        }
        ContentBlock::ToolResult { .. } => {
            // ToolResult inside an Assistant message is unusual but
            // surface it generically; the Role::ToolResult path
            // covers the common case.
            out.push(DisplayItem::ToolCallPlaceholder {
                label: "tool result".into(),
            });
        }
        ContentBlock::Image { .. } => {
            out.push(DisplayItem::ToolCallPlaceholder {
                label: "[image]".into(),
            });
        }
        ContentBlock::Document { .. } => {
            out.push(DisplayItem::ToolCallPlaceholder {
                label: "[document]".into(),
            });
        }
        ContentBlock::ToolSchema { name, .. } => {
            out.push(DisplayItem::ToolCallPlaceholder {
                label: format!("tool schema: {name}"),
            });
        }
    }
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
