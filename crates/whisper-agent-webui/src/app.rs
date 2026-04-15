//! Chat UI.
//!
//! Pure-egui rendering — compiles on both native and wasm. Networking lives in the
//! wasm-only `web_entry` module in `lib.rs`. The two communicate through:
//!   - `inbound`: a shared queue of [`ServerToClient`] events the WebSocket pushes
//!     into and `update()` drains each frame.
//!   - `send_fn`: a closure provided at construction time. On wasm it serializes to
//!     CBOR and calls `WebSocket::send_with_u8_array`; on native it's a no-op stub.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use egui::{Color32, RichText, ScrollArea, TextEdit};
use whisper_agent_protocol::{ClientToServer, ServerToClient, SessionState, Usage};

pub type Inbound = Rc<RefCell<VecDeque<ServerToClient>>>;
pub type SendFn = Box<dyn Fn(ClientToServer)>;

pub struct ChatApp {
    input: String,
    items: Vec<DisplayItem>,
    status: ConnectionStatus,
    last_usage: Option<Usage>,
    inbound: Inbound,
    send_fn: SendFn,
}

impl ChatApp {
    pub fn new(inbound: Inbound, send_fn: SendFn) -> Self {
        Self {
            input: String::new(),
            items: Vec::new(),
            status: ConnectionStatus::Connecting,
            last_usage: None,
            inbound,
            send_fn,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConnectionStatus {
    Connecting,
    Connected,
    Ready,
    Working,
    Error,
}

impl ConnectionStatus {
    fn label(self) -> (&'static str, Color32) {
        match self {
            Self::Connecting => ("connecting…", Color32::from_rgb(200, 180, 60)),
            Self::Connected => ("connected", Color32::from_rgb(80, 180, 100)),
            Self::Ready => ("ready", Color32::from_rgb(80, 200, 120)),
            Self::Working => ("working…", Color32::from_rgb(120, 180, 240)),
            Self::Error => ("error", Color32::from_rgb(220, 90, 90)),
        }
    }

    fn from_session(s: SessionState) -> Self {
        match s {
            SessionState::Connected => Self::Connected,
            SessionState::Ready => Self::Ready,
            SessionState::Working => Self::Working,
            SessionState::Error => Self::Error,
        }
    }

    fn input_enabled(self) -> bool {
        matches!(self, Self::Ready | Self::Connected)
    }
}

enum DisplayItem {
    User { text: String },
    AssistantText { text: String },
    ToolCall {
        id: String,
        name: String,
        args_preview: String,
        result: Option<String>,
        is_error: bool,
    },
    SystemNote { text: String, is_error: bool },
}

impl ChatApp {
    fn drain_inbound(&mut self) {
        let events: Vec<ServerToClient> = self.inbound.borrow_mut().drain(..).collect();
        for event in events {
            self.handle_event(event);
        }
    }

    fn handle_event(&mut self, event: ServerToClient) {
        match event {
            ServerToClient::Status { state, detail } => {
                self.status = ConnectionStatus::from_session(state);
                if let Some(d) = detail {
                    self.items.push(DisplayItem::SystemNote { text: d, is_error: false });
                }
            }
            ServerToClient::AssistantBegin => {
                // Mark a turn boundary; nothing to render yet.
            }
            ServerToClient::AssistantText { text } => {
                self.items.push(DisplayItem::AssistantText { text });
            }
            ServerToClient::ToolCallBegin { id, name, args_preview } => {
                self.items.push(DisplayItem::ToolCall {
                    id,
                    name,
                    args_preview,
                    result: None,
                    is_error: false,
                });
            }
            ServerToClient::ToolCallEnd { id, result_preview, is_error } => {
                if let Some(DisplayItem::ToolCall {
                    id: existing,
                    result,
                    is_error: existing_err,
                    ..
                }) = self.items.iter_mut().rev().find(|item| matches!(item, DisplayItem::ToolCall { id: i, .. } if i == &id))
                {
                    if *existing == id {
                        *result = Some(result_preview);
                        *existing_err = is_error;
                    }
                }
            }
            ServerToClient::AssistantEnd { stop_reason: _, usage } => {
                self.last_usage = Some(usage);
            }
            ServerToClient::LoopComplete => {}
            ServerToClient::Error { message } => {
                self.items.push(DisplayItem::SystemNote { text: message, is_error: true });
            }
        }
    }

    fn submit(&mut self) {
        let text = std::mem::take(&mut self.input);
        if text.trim().is_empty() {
            return;
        }
        self.items.push(DisplayItem::User { text: text.clone() });
        (self.send_fn)(ClientToServer::UserMessage { text });
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_inbound();

        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("whisper-agent");
                ui.separator();
                let (label, color) = self.status.label();
                ui.label(RichText::new(label).color(color));
                if let Some(u) = self.last_usage {
                    ui.separator();
                    ui.label(
                        RichText::new(format!(
                            "tokens: {}↑ {}↓  cache: {}r/{}c",
                            u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_creation_input_tokens
                        ))
                        .color(Color32::from_gray(160))
                        .small(),
                    );
                }
            });
        });

        let input_enabled = self.status.input_enabled();
        egui::TopBottomPanel::bottom("input_bar")
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(input_enabled, |ui| {
                        let send_pressed = ui.button("Send").clicked();
                        let response = ui.add_sized(
                            [ui.available_width(), 28.0],
                            TextEdit::singleline(&mut self.input).hint_text(if input_enabled {
                                "Message the agent"
                            } else {
                                "(connecting / agent busy)"
                            }),
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
            if self.items.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(60.0);
                    ui.label(
                        RichText::new("conversation will appear here")
                            .color(Color32::from_gray(140)),
                    );
                });
                return;
            }
            ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for item in &self.items {
                    render_item(ui, item);
                    ui.add_space(6.0);
                }
            });
        });
    }
}

fn render_item(ui: &mut egui::Ui, item: &DisplayItem) {
    match item {
        DisplayItem::User { text } => {
            ui.horizontal_top(|ui| {
                ui.label(
                    RichText::new("you").color(Color32::from_rgb(120, 180, 240)).strong(),
                );
                ui.label(text);
            });
        }
        DisplayItem::AssistantText { text } => {
            ui.horizontal_top(|ui| {
                ui.label(
                    RichText::new("agent").color(Color32::from_rgb(160, 220, 160)).strong(),
                );
                ui.label(text);
            });
        }
        DisplayItem::ToolCall { name, args_preview, result, is_error, .. } => {
            ui.horizontal_top(|ui| {
                ui.label(
                    RichText::new("tool").color(Color32::from_rgb(220, 180, 100)).strong(),
                );
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
