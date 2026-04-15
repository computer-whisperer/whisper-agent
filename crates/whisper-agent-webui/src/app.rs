//! Chat UI scaffold.
//!
//! Layout: status bar at the top, scrollable message list in the middle, input row at
//! the bottom. No networking yet — the input button currently appends a fake echo so
//! the layout exercises the message list rendering.

use egui::{Color32, RichText, ScrollArea, TextEdit};

#[derive(Default)]
pub struct ChatApp {
    input: String,
    messages: Vec<ChatMessage>,
    status: ConnectionStatus,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum ConnectionStatus {
    #[default]
    Disconnected,
    #[allow(dead_code)] // wired up in step 6
    Connecting,
    #[allow(dead_code)]
    Connected,
}

impl ConnectionStatus {
    fn label(self) -> (&'static str, Color32) {
        match self {
            Self::Disconnected => ("disconnected", Color32::from_rgb(180, 80, 80)),
            Self::Connecting => ("connecting…", Color32::from_rgb(200, 180, 60)),
            Self::Connected => ("connected", Color32::from_rgb(80, 180, 100)),
        }
    }
}

#[derive(Clone, Debug)]
struct ChatMessage {
    role: Role,
    text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    User,
    #[allow(dead_code)] // wired up in step 6
    Assistant,
    System,
}

impl Role {
    fn label(self) -> &'static str {
        match self {
            Role::User => "you",
            Role::Assistant => "agent",
            Role::System => "system",
        }
    }

    fn color(self) -> Color32 {
        match self {
            Role::User => Color32::from_rgb(120, 180, 240),
            Role::Assistant => Color32::from_rgb(160, 220, 160),
            Role::System => Color32::from_gray(160),
        }
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("whisper-agent");
                ui.separator();
                let (label, color) = self.status.label();
                ui.label(RichText::new(label).color(color));
            });
        });

        egui::TopBottomPanel::bottom("input_bar")
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let send_pressed = ui.button("Send").clicked();
                    let response = ui.add_sized(
                        [ui.available_width(), 28.0],
                        TextEdit::singleline(&mut self.input)
                            .hint_text("Message the agent (step 5 scaffold — not wired to a server yet)"),
                    );
                    let enter_pressed = response.lost_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if (send_pressed || enter_pressed) && !self.input.trim().is_empty() {
                        self.submit_local_only();
                    }
                });
                ui.add_space(4.0);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.messages.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(60.0);
                    ui.label(
                        RichText::new("conversation will appear here").color(Color32::from_gray(140)),
                    );
                });
                return;
            }
            ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for msg in &self.messages {
                    ui.horizontal_top(|ui| {
                        ui.label(RichText::new(msg.role.label()).color(msg.role.color()).strong());
                        ui.label(&msg.text);
                    });
                    ui.add_space(4.0);
                }
            });
        });
    }
}

impl ChatApp {
    /// Local-only submit handler used while the WebSocket isn't wired up. Pushes the
    /// user's message + a stub echo so the layout has something to render.
    fn submit_local_only(&mut self) {
        let text = std::mem::take(&mut self.input);
        self.messages.push(ChatMessage { role: Role::User, text: text.clone() });
        self.messages.push(ChatMessage {
            role: Role::System,
            text: format!("(echo, no server connected yet) you said: {text}"),
        });
    }
}
