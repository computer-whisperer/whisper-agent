//! Login form shown before the chat UI connects.
//!
//! Mirrors the webui's HTML `/login.html` form in spirit: one box, a
//! token field (password-masked), a connect button. The webui only
//! needs a token because the browser's URL already encodes the host;
//! we add a server URL field since the desktop binary has no
//! implicit host.

use egui::{Color32, Key, Ui};

pub struct LoginForm {
    server: String,
    token: String,
    remember: bool,
    err: Option<String>,
}

pub struct Submission {
    pub server: String,
    pub token: String,
    pub remember: bool,
}

impl LoginForm {
    pub fn new(server: Option<String>, token: Option<String>) -> Self {
        Self {
            server: server.unwrap_or_default(),
            token: token.unwrap_or_default(),
            remember: true,
            err: None,
        }
    }

    /// Replace the error banner (e.g. after a connection attempt
    /// bounced us back to the form).
    pub fn set_error(&mut self, err: impl Into<String>) {
        self.err = Some(err.into());
    }

    /// Render the form into the given ui. Returns `Some(Submission)`
    /// on the frame the user clicks Connect (or hits Enter) with a
    /// non-empty server URL; `None` otherwise.
    pub fn show(&mut self, ui: &mut Ui) -> Option<Submission> {
        let ctx = ui.ctx().clone();
        let mut clicked = false;
        let mut enter_pressed = false;

        ui.vertical_centered(|ui| {
            ui.add_space(80.0);
            ui.heading("whisper-agent");
            ui.add_space(24.0);

            ui.allocate_ui_with_layout(
                egui::vec2(360.0, 0.0),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.label("Server URL");
                    let s_resp = ui.add(
                        egui::TextEdit::singleline(&mut self.server)
                            .hint_text("https://host:port")
                            .desired_width(f32::INFINITY),
                    );
                    ui.add_space(8.0);

                    ui.label("Client token");
                    let t_resp = ui.add(
                        egui::TextEdit::singleline(&mut self.token)
                            .password(true)
                            .hint_text("(blank is OK for loopback servers)")
                            .desired_width(f32::INFINITY),
                    );
                    ui.add_space(10.0);

                    ui.checkbox(&mut self.remember, "Remember on this device");
                    ui.add_space(10.0);

                    if ui
                        .add_sized([f32::INFINITY, 28.0], egui::Button::new("Connect"))
                        .clicked()
                    {
                        clicked = true;
                    }

                    // Enter submits from either text field.
                    if (s_resp.lost_focus() || t_resp.lost_focus())
                        && ctx.input(|i| i.key_pressed(Key::Enter))
                    {
                        enter_pressed = true;
                    }

                    if let Some(err) = &self.err {
                        ui.add_space(8.0);
                        ui.colored_label(Color32::from_rgb(0xd0, 0x70, 0x70), err);
                    }
                },
            );
        });

        if !(clicked || enter_pressed) {
            return None;
        }

        let server = self.server.trim().to_string();
        if server.is_empty() {
            self.err = Some("server URL is required".into());
            return None;
        }
        self.err = None;
        Some(Submission {
            server,
            token: self.token.trim().to_string(),
            remember: self.remember,
        })
    }
}
