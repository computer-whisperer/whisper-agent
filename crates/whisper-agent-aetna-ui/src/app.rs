//! [`ChatApp`] — the platform-agnostic aetna [`App`] for the
//! whisper-agent chat client.
//!
//! Stage 2 surface: connection-status banner, sidebar with pods +
//! threads, snapshot rendering of the selected thread's
//! conversation as user/assistant text rows. Build/on_event split is
//! the load-bearing test of the pivot — every interactive element
//! routes through [`ChatApp::on_event`] via a key, every visual is a
//! function of state read in [`ChatApp::build`].
//!
//! Things stage 2 deliberately defers (planned for later stages):
//! - markdown / code blocks (pending an upstream aetna parser)
//! - reasoning-block collapsing
//! - tool calls + tool results (rendered as a generic placeholder for now)
//! - streaming (text-delta / reasoning-delta / tool-streaming) — only
//!   the snapshot path is implemented; live deltas are queued but
//!   ignored
//! - compose box, attachments, model picker, modals
//! - per-frame draft persistence (`SetThreadDraft` debounce)
//!
//! Dispatch model: a single `dispatch_wire` walks `ServerToClient`
//! variants — only the few stage-2 cares about have arms; the rest
//! drop on the floor with a debug log so we can see what we're
//! ignoring without spamming the user.

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
            // Streaming + per-turn append events: stage 2 only renders
            // snapshots, so we drop these. Resubscribing forces a fresh
            // snapshot if the user wants to see what arrived. Stage 3
            // wires these into incremental DisplayItem appends.
            ServerToClient::ThreadUserMessage { .. }
            | ServerToClient::ThreadToolResultMessage { .. }
            | ServerToClient::ThreadAssistantBegin { .. }
            | ServerToClient::ThreadAssistantTextDelta { .. }
            | ServerToClient::ThreadAssistantReasoningDelta { .. }
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
        }
    }

    fn theme(&self) -> Theme {
        Theme::aetna_dark()
    }
}

impl ChatApp {
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

        let mut title_row: Vec<El> = vec![card_title(title).width(Size::Fill(1.0))];
        if let Some(s) = summary {
            title_row.push(state_badge(s.state));
        }
        let header_children: Vec<El> = vec![
            row(title_row)
                .align(Align::Center)
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            card_description(thread_id.to_string()),
        ];
        // The card_header surface gives the title bar a subtle muted
        // strip; we drop the surrounding card chrome (no shadow, no
        // radius) since this header sits flush at the top of the
        // pane and would otherwise float visibly inside the wider
        // content rect.
        let header = card_header(header_children)
            .fill(tokens::MUTED)
            .stroke(tokens::BORDER)
            .width(Size::Fill(1.0));

        // Scroll key derived from the thread id so the offset
        // persists across rebuilds *for this thread*; switching
        // threads gets a fresh offset.
        let scroll_key = format!("chat-scroll:{thread_id}");

        let body: El = match view {
            None => empty_state("loading…"),
            Some(v) if v.items.is_empty() => empty_state("no messages yet"),
            Some(v) => {
                let rows: Vec<El> = v.items.iter().map(display_item_row).collect();
                scroll(rows)
                    .key(scroll_key)
                    .gap(tokens::SPACE_3)
                    .padding(tokens::SPACE_4)
                    .width(Size::Fill(1.0))
                    .height(Size::Fill(1.0))
            }
        };

        column([header, body])
            .gap(0.0)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
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

fn display_item_row(item: &DisplayItem) -> El {
    match item {
        DisplayItem::User { text: t } => chat_row("user", t, RoleVariant::User),
        DisplayItem::Assistant { text: t } => chat_row("assistant", t, RoleVariant::Assistant),
        DisplayItem::Reasoning { text: t } => chat_row("reasoning", t, RoleVariant::Reasoning),
        DisplayItem::SystemNote { text: t } => chat_row("system", t, RoleVariant::System),
        DisplayItem::ToolCallPlaceholder { label } => chat_row("tool", label, RoleVariant::Tool),
    }
}

#[derive(Clone, Copy)]
enum RoleVariant {
    User,
    Assistant,
    Reasoning,
    System,
    Tool,
}

/// One chat row: a card with a role badge in the header and the
/// message body in the content slot. Reasoning / system / tool
/// variants tint the badge so a long log scans at a glance.
fn chat_row(role_label: &str, body: &str, variant: RoleVariant) -> El {
    let role_chip = match variant {
        RoleVariant::User => badge(role_label).info(),
        RoleVariant::Assistant => badge(role_label).success(),
        RoleVariant::Reasoning => badge(role_label).muted(),
        RoleVariant::System => badge(role_label).warning(),
        RoleVariant::Tool => badge(role_label).muted(),
    };
    card([
        card_header([role_chip]),
        card_content([paragraph(body.to_string())]),
    ])
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
