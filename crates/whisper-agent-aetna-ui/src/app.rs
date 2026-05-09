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
                .height(Size::Fill(1.0)),
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
        let mut entries: Vec<El> = vec![
            text("whisper-agent").bold().font_size(18.0),
            text(self.conn_status_line()).muted().small(),
            divider(),
        ];

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
            entries.push(text("(no pods yet)").muted().small());
        }

        for pod in pods {
            entries.push(spacer().height(Size::Fixed(tokens::SPACE_2)));
            let header = if pod.archived {
                format!("{} (archived)", pod.name)
            } else {
                pod.name.clone()
            };
            entries.push(text(header).bold().small());

            let mut pod_threads: Vec<&ThreadSummary> = self
                .threads
                .values()
                .filter(|t| t.pod_id == pod.pod_id)
                .collect();
            pod_threads.sort_by(|a, b| b.last_active.cmp(&a.last_active));

            if pod_threads.is_empty() {
                entries.push(text("(no threads)").muted().xsmall());
            } else {
                for t in pod_threads {
                    entries.push(self.thread_row(t));
                }
            }
        }

        column(entries)
            .gap(tokens::SPACE_2)
            .padding(tokens::SPACE_4)
            .width(Size::Fixed(260.0))
            .height(Size::Fill(1.0))
            .fill(tokens::CARD)
            .stroke(tokens::BORDER)
    }

    fn conn_status_line(&self) -> String {
        match &self.conn_detail {
            Some(detail) => format!("{} — {}", self.conn_status.label(), detail),
            None => self.conn_status.label().to_string(),
        }
    }

    fn thread_row(&self, t: &ThreadSummary) -> El {
        let label = thread_label(t);
        let key = format!("thread:{}", t.thread_id);
        let selected = self.selected.as_deref() == Some(t.thread_id.as_str());
        let mut b = button(label).key(key);
        if selected {
            b = b.primary();
        } else {
            b = b.ghost();
        }
        b
    }

    fn content(&self) -> El {
        let inner = match self.selected.as_ref() {
            None => column([
                text("no thread selected").muted(),
                text("pick a thread from the sidebar to view its log")
                    .muted()
                    .small(),
            ])
            .gap(tokens::SPACE_2)
            .padding(tokens::SPACE_6),
            Some(thread_id) => self.thread_pane(thread_id),
        };
        column([inner])
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
    }

    fn thread_pane(&self, thread_id: &str) -> El {
        let summary = self.threads.get(thread_id);
        let view = self.views.get(thread_id);

        let title = view
            .and_then(|v| v.title.as_deref())
            .or_else(|| summary.and_then(|s| s.title.as_deref()))
            .unwrap_or("untitled");

        let header = column([
            text(title).bold().font_size(16.0),
            text(thread_id).muted().xsmall(),
        ])
        .gap(tokens::SPACE_1)
        .padding(tokens::SPACE_4)
        .width(Size::Fill(1.0))
        .fill(tokens::CARD)
        .stroke(tokens::BORDER);

        // Scroll key derived from the thread id so the offset
        // persists across rebuilds *for this thread*; switching
        // threads gets a fresh offset.
        let scroll_key = format!("chat-scroll:{thread_id}");

        let body: El = match view {
            None => column([text("loading…").muted()])
                .padding(tokens::SPACE_6)
                .width(Size::Fill(1.0)),
            Some(v) if v.items.is_empty() => column([text("(no messages)").muted()])
                .padding(tokens::SPACE_6)
                .width(Size::Fill(1.0)),
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
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
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
        DisplayItem::User { text: t } => labeled_block("user", t, tokens::PRIMARY),
        DisplayItem::Assistant { text: t } => labeled_block("assistant", t, tokens::FOREGROUND),
        DisplayItem::Reasoning { text: t } => {
            labeled_block("reasoning", t, tokens::MUTED_FOREGROUND)
        }
        DisplayItem::SystemNote { text: t } => labeled_block("system", t, tokens::MUTED_FOREGROUND),
        DisplayItem::ToolCallPlaceholder { label } => {
            labeled_block("tool", label, tokens::MUTED_FOREGROUND)
        }
    }
}

/// One chat-row card with a small role label above a wrapping body.
/// `_label_color` is currently unused — kept on the signature so we
/// can color-code the role chip in stage 3 without churning every
/// call site.
fn labeled_block(label: &str, body: &str, _label_color: Color) -> El {
    column([text(label).small().muted(), paragraph(body)])
        .gap(tokens::SPACE_1)
        .padding(tokens::SPACE_3)
        .width(Size::Fill(1.0))
        .fill(tokens::CARD)
        .stroke(tokens::BORDER)
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
