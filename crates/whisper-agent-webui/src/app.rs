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
//! selection we send `SubscribeToTask`; the server's `TaskSnapshot` response rebuilds
//! the task's display items. Subsequent turn events append to them.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use egui::{Color32, RichText, ScrollArea, TextEdit};
use whisper_agent_protocol::{
    ClientToServer, ContentBlock, Conversation, Message, Role, ServerToClient, TaskStateLabel,
    TaskSummary, ToolResultContent, Usage,
};

/// Events pushed into [`Inbound`]. In addition to decoded wire messages we pipe in
/// connection-level signals (open/close/error) so the UI can show a connection status
/// distinct from per-task state.
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

fn state_chip(state: TaskStateLabel) -> (&'static str, Color32) {
    match state {
        TaskStateLabel::Idle => ("idle", Color32::from_gray(160)),
        TaskStateLabel::Working => ("working", Color32::from_rgb(120, 180, 240)),
        TaskStateLabel::AwaitingApproval => ("approval", Color32::from_rgb(240, 180, 90)),
        TaskStateLabel::Completed => ("completed", Color32::from_rgb(120, 200, 140)),
        TaskStateLabel::Failed => ("failed", Color32::from_rgb(220, 110, 110)),
        TaskStateLabel::Cancelled => ("cancelled", Color32::from_rgb(180, 140, 140)),
    }
}

enum DisplayItem {
    User {
        text: String,
    },
    AssistantText {
        text: String,
    },
    ToolCall {
        tool_use_id: String,
        name: String,
        args_preview: String,
        result: Option<String>,
        is_error: bool,
    },
    SystemNote {
        text: String,
        is_error: bool,
    },
}

struct TaskView {
    summary: TaskSummary,
    items: Vec<DisplayItem>,
    total_usage: Usage,
    subscribed: bool,
}

impl TaskView {
    fn new(summary: TaskSummary) -> Self {
        Self {
            summary,
            items: Vec::new(),
            total_usage: Usage::default(),
            subscribed: false,
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
        }
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
                    self.send(ClientToServer::ListTasks { correlation_id: None });
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
            InboundEvent::Wire(msg) => self.handle_wire(msg),
        }
    }

    fn handle_wire(&mut self, msg: ServerToClient) {
        match msg {
            ServerToClient::TaskCreated { task_id, summary, correlation_id: _ } => {
                self.upsert_task(summary);
                self.recompute_order();
                // Auto-select newly created tasks.
                if self.selected.as_deref() != Some(&task_id) {
                    self.select_task(task_id);
                }
            }
            ServerToClient::TaskStateChanged { task_id, state } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.summary.state = state;
                }
            }
            ServerToClient::TaskTitleUpdated { task_id, title } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.summary.title = Some(title);
                }
            }
            ServerToClient::TaskArchived { task_id } => {
                self.tasks.remove(&task_id);
                self.task_order.retain(|id| id != &task_id);
                if self.selected.as_deref() == Some(&task_id) {
                    self.selected = None;
                    self.composing_new = true;
                }
            }
            ServerToClient::TaskList { tasks, .. } => {
                self.tasks.retain(|id, _| tasks.iter().any(|t| &t.task_id == id));
                for summary in tasks {
                    self.upsert_task(summary);
                }
                self.recompute_order();
            }
            ServerToClient::TaskSnapshot { task_id, snapshot } => {
                let items = conversation_to_items(&snapshot.conversation);
                let view = self
                    .tasks
                    .entry(task_id.clone())
                    .or_insert_with(|| TaskView::new(snapshot_summary(&snapshot)));
                view.summary.state = snapshot.state;
                view.summary.title = snapshot.title;
                view.total_usage = snapshot.total_usage;
                view.items = items;
                view.subscribed = true;
            }
            ServerToClient::TaskAssistantBegin { .. } => {}
            ServerToClient::TaskAssistantText { task_id, text } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.items.push(DisplayItem::AssistantText { text });
                }
            }
            ServerToClient::TaskAssistantTextDelta { task_id, delta } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    if let Some(DisplayItem::AssistantText { text }) = view.items.last_mut() {
                        text.push_str(&delta);
                    } else {
                        view.items.push(DisplayItem::AssistantText { text: delta });
                    }
                }
            }
            ServerToClient::TaskToolCallBegin { task_id, tool_use_id, name, args_preview } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.items.push(DisplayItem::ToolCall {
                        tool_use_id,
                        name,
                        args_preview,
                        result: None,
                        is_error: false,
                    });
                }
            }
            ServerToClient::TaskToolCallEnd {
                task_id,
                tool_use_id,
                result_preview,
                is_error,
            } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    for item in view.items.iter_mut().rev() {
                        if let DisplayItem::ToolCall { tool_use_id: id, result, is_error: err, .. } =
                            item
                        {
                            if id == &tool_use_id {
                                *result = Some(result_preview);
                                *err = is_error;
                                break;
                            }
                        }
                    }
                }
            }
            ServerToClient::TaskAssistantEnd { task_id, usage, .. } => {
                if let Some(view) = self.tasks.get_mut(&task_id) {
                    view.total_usage.add(&usage);
                }
            }
            ServerToClient::TaskLoopComplete { .. } => {}
            ServerToClient::Error { task_id, message, .. } => {
                if let Some(tid) = task_id.as_ref()
                    && let Some(view) = self.tasks.get_mut(tid)
                {
                    view.items.push(DisplayItem::SystemNote { text: message, is_error: true });
                } else {
                    // No task scope — surface via conn detail so the banner reflects it.
                    self.conn_detail = Some(message);
                }
            }
        }
    }

    fn upsert_task(&mut self, summary: TaskSummary) {
        let id = summary.task_id.clone();
        self.tasks
            .entry(id)
            .and_modify(|v| v.summary = summary.clone())
            .or_insert_with(|| TaskView::new(summary));
    }

    fn recompute_order(&mut self) {
        let mut ids: Vec<String> = self.tasks.keys().cloned().collect();
        ids.sort_by(|a, b| {
            let ta = self.tasks.get(a).map(|v| v.summary.created_at.clone()).unwrap_or_default();
            let tb = self.tasks.get(b).map(|v| v.summary.created_at.clone()).unwrap_or_default();
            tb.cmp(&ta)
        });
        self.task_order = ids;
    }

    fn select_task(&mut self, task_id: String) {
        if self.selected.as_deref() == Some(&task_id) {
            return;
        }
        self.selected = Some(task_id.clone());
        self.composing_new = false;
        let need_subscribe = self
            .tasks
            .get(&task_id)
            .map(|v| !v.subscribed)
            .unwrap_or(true);
        if need_subscribe {
            self.send(ClientToServer::SubscribeToTask { task_id });
        }
    }

    fn submit(&mut self) {
        let text = std::mem::take(&mut self.input);
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.composing_new || self.selected.is_none() {
            self.send(ClientToServer::CreateTask {
                correlation_id: None,
                initial_message: trimmed.to_string(),
                config_override: None,
            });
        } else if let Some(task_id) = self.selected.clone() {
            if let Some(view) = self.tasks.get_mut(&task_id) {
                view.items.push(DisplayItem::User { text: trimmed.to_string() });
            }
            self.send(ClientToServer::SendUserMessage {
                task_id,
                text: trimmed.to_string(),
            });
        }
    }
}

fn snapshot_summary(s: &whisper_agent_protocol::TaskSnapshot) -> TaskSummary {
    TaskSummary {
        task_id: s.task_id.clone(),
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
                    ContentBlock::Text { text } => out.push(DisplayItem::User { text: text.clone() }),
                    ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                        let text = tool_result_text(content);
                        // Backfill the result onto a matching ToolCall item, or push a system note.
                        let mut matched = false;
                        for existing in out.iter_mut().rev() {
                            if let DisplayItem::ToolCall { tool_use_id: id, result, is_error: err, .. }
                                = existing
                            {
                                if id == tool_use_id && result.is_none() {
                                    *result = Some(text.clone());
                                    *err = *is_error;
                                    matched = true;
                                    break;
                                }
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
                        out.push(DisplayItem::ToolCall {
                            tool_use_id: id.clone(),
                            name: name.clone(),
                            args_preview: truncate(
                                serde_json::to_string(input).unwrap_or_default(),
                                200,
                            ),
                            result: None,
                            is_error: false,
                        });
                    }
                    ContentBlock::Thinking { thinking, .. } => {
                        out.push(DisplayItem::SystemNote {
                            text: format!("thinking: {thinking}"),
                            is_error: false,
                        });
                    }
                    _ => {}
                }
            }
        }
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
                if ui.button("+ New task").clicked() {
                    self.selected = None;
                    self.composing_new = true;
                    self.input.clear();
                }
                ui.separator();
                ScrollArea::vertical().show(ui, |ui| {
                    let order = self.task_order.clone();
                    for task_id in order {
                        let Some(view) = self.tasks.get(&task_id) else { continue };
                        let is_selected = self.selected.as_deref() == Some(&task_id);
                        let title = view
                            .summary
                            .title
                            .clone()
                            .unwrap_or_else(|| task_id[..task_id.len().min(14)].to_string());
                        let (chip, chip_color) = state_chip(view.summary.state);
                        let row = ui.add_sized(
                            [ui.available_width(), 0.0],
                            egui::Button::selectable(
                                is_selected,
                                RichText::new(format!("{title}  [{chip}]")).color(
                                    if is_selected { Color32::WHITE } else { chip_color },
                                ),
                            ),
                        );
                        if row.clicked() {
                            self.select_task(task_id.clone());
                        }
                    }
                });
            });

        let input_enabled = matches!(self.conn_status, ConnectionStatus::Connected);
        let hint = if self.composing_new || self.selected.is_none() {
            if input_enabled { "Describe a new task" } else { "(connecting)" }
        } else {
            if input_enabled { "Message this task" } else { "(connecting)" }
        };

        egui::TopBottomPanel::bottom("input_bar")
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if let Some(task_id) = self.selected.clone() {
                        if ui.button("Cancel").clicked() {
                            self.send(ClientToServer::CancelTask { task_id: task_id.clone() });
                        }
                        if ui.button("Archive").clicked() {
                            self.send(ClientToServer::ArchiveTask { task_id });
                        }
                        ui.separator();
                    }
                    ui.add_enabled_ui(input_enabled, |ui| {
                        let send_pressed = ui.button("Send").clicked();
                        let response = ui.add_sized(
                            [ui.available_width(), 28.0],
                            TextEdit::singleline(&mut self.input).hint_text(hint),
                        );
                        let enter_pressed = response.lost_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        if (send_pressed || enter_pressed) && input_enabled {
                            self.submit();
                            response.request_focus();
                        }
                    });
                });
                ui.add_space(4.0);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.selected.clone() {
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
                Some(task_id) => match self.tasks.get(&task_id) {
                    None => {
                        ui.label(
                            RichText::new(format!("task {task_id} not found"))
                                .color(Color32::from_rgb(220, 120, 120)),
                        );
                    }
                    Some(view) => {
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
            }
        });
    }
}

fn render_item(ui: &mut egui::Ui, item: &DisplayItem) {
    match item {
        DisplayItem::User { text } => {
            ui.horizontal_top(|ui| {
                ui.label(RichText::new("you").color(Color32::from_rgb(120, 180, 240)).strong());
                ui.label(text);
            });
        }
        DisplayItem::AssistantText { text } => {
            ui.horizontal_top(|ui| {
                ui.label(RichText::new("agent").color(Color32::from_rgb(160, 220, 160)).strong());
                ui.label(text);
            });
        }
        DisplayItem::ToolCall { name, args_preview, result, is_error, .. } => {
            ui.horizontal_top(|ui| {
                ui.label(RichText::new("tool").color(Color32::from_rgb(220, 180, 100)).strong());
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(format!("{name}({args_preview})"))
                            .color(Color32::from_gray(200))
                            .monospace(),
                    );
                    match result {
                        None => {
                            ui.label(
                                RichText::new("running…")
                                    .color(Color32::from_gray(140))
                                    .italics(),
                            );
                        }
                        Some(text) => {
                            let color = if *is_error {
                                Color32::from_rgb(220, 120, 120)
                            } else {
                                Color32::from_gray(180)
                            };
                            ui.label(RichText::new(text).color(color).monospace().small());
                        }
                    }
                });
            });
        }
        DisplayItem::SystemNote { text, is_error } => {
            let color = if *is_error {
                Color32::from_rgb(220, 120, 120)
            } else {
                Color32::from_gray(160)
            };
            ui.label(RichText::new(text).color(color).italics());
        }
    }
}
