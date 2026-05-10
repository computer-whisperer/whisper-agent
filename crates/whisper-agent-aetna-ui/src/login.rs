//! [`LoginApp`] — the aetna [`App`] shown before the chat client has
//! credentials.
//!
//! Mirrors the egui sibling's `LoginForm` shape: a centered card with
//! a server URL field, a password-masked token field, a "remember"
//! checkbox, and a "Connect" button. Submission is fanned out through
//! a host-supplied callback (`SubmitFn`) so the desktop binary can
//! persist the credentials and swap to the chat phase.
//!
//! This crate is the same lib the wasm browser entry will eventually
//! consume, so the form lives here (not in the desktop binary). Auth
//! persistence and the WS-task swap are host concerns and stay on the
//! caller side of the callback.
//!
//! Routing keys:
//! - `login:server` — server URL `text_input`
//! - `login:token` — bearer token `text_input` (password-masked)
//! - `login:remember` — "Remember on this device" checkbox
//! - `login:connect` — primary button; submits the form
//!
//! The host can update the form's error banner from outside (e.g.
//! after a connection attempt bounces the user back here) via
//! [`LoginApp::set_error`].

use aetna_core::prelude::*;

const KEY_SERVER: &str = "login:server";
const KEY_TOKEN: &str = "login:token";
const KEY_REMEMBER: &str = "login:remember";
const KEY_CONNECT: &str = "login:connect";

/// Submitted credentials. The host serializes whatever fields it
/// wants to persist (the egui sibling honors `remember` to
/// gate writing the token to disk; the aetna binary mirrors that).
#[derive(Clone, Debug)]
pub struct LoginInput {
    pub server: String,
    pub token: String,
    pub remember: bool,
}

/// Callback the host registers to receive a successful submission.
/// The `LoginApp` validates the server URL is non-empty before
/// firing — the host doesn't need to defensively re-check.
pub type SubmitFn = Box<dyn Fn(LoginInput)>;

/// The login form as an aetna [`App`]. Owns its own buffers and
/// selection cursor so it can be wrapped in a host's phase-switch
/// app without sharing state with the chat-side `ChatApp`.
pub struct LoginApp {
    server: String,
    token: String,
    remember: bool,
    error: Option<String>,
    selection: Selection,
    submit: SubmitFn,
}

impl LoginApp {
    /// Build a login form, optionally pre-filled with values from a
    /// config file or CLI flags. Empty strings render as empty
    /// fields with their placeholder hints.
    pub fn new(
        initial_server: Option<String>,
        initial_token: Option<String>,
        submit: SubmitFn,
    ) -> Self {
        Self {
            server: initial_server.unwrap_or_default(),
            token: initial_token.unwrap_or_default(),
            // Default to ticked — the egui sibling does the same.
            // First-time users almost always want their creds saved;
            // power users can untick.
            remember: true,
            error: None,
            selection: Selection::default(),
            submit,
        }
    }

    /// Replace the form's error banner. Called by the host after a
    /// failed connection attempt so the user sees why the previous
    /// submission didn't stick.
    pub fn set_error(&mut self, err: impl Into<String>) {
        self.error = Some(err.into());
    }

    /// Validate the form and fire the host callback. Empty server
    /// URL fails inline with an error banner; the callback only
    /// receives well-formed submissions.
    fn submit(&mut self) {
        let server = self.server.trim().to_string();
        if server.is_empty() {
            self.error = Some("server URL is required".into());
            return;
        }
        self.error = None;
        (self.submit)(LoginInput {
            server,
            token: self.token.trim().to_string(),
            remember: self.remember,
        });
    }
}

impl App for LoginApp {
    fn build(&self, _cx: &BuildCx) -> El {
        let server_input = text_input(&self.server, &self.selection, KEY_SERVER);
        // `password()` lives on `TextInputOpts` (it controls both the
        // visual mask and the copy/cut suppression at apply_event time),
        // so the password field has to go through `text_input_with`.
        let token_input = text_input_with(
            &self.token,
            &self.selection,
            KEY_TOKEN,
            TextInputOpts::default().password(),
        );
        let connect = button("Connect").key(KEY_CONNECT).primary();

        let remember_row = row([
            checkbox(self.remember).key(KEY_REMEMBER),
            text("Remember on this device"),
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center);

        let mut content_blocks: Vec<El> = vec![form([
            form_item([
                form_label("Server URL"),
                form_control(server_input),
                form_description("e.g. https://host:port — http://127.0.0.1:8080 for loopback"),
            ]),
            form_item([
                form_label("Client token"),
                form_control(token_input),
                form_description("blank is OK for loopback servers"),
            ]),
            form_item([form_control(remember_row)]),
        ])];

        if let Some(err) = &self.error {
            content_blocks.push(
                alert([
                    alert_title("connection failed"),
                    alert_description(err.clone()),
                ])
                .destructive(),
            );
        }

        let card_el = card([
            card_header([
                card_title("whisper-agent"),
                card_description("Sign in to a whisper-agent server"),
            ]),
            card_content(content_blocks),
            card_footer([row([spacer(), connect]).width(Size::Fill(1.0))]),
        ])
        .width(Size::Fixed(440.0));

        column([card_el])
            .padding(tokens::SPACE_6)
            .align(Align::Center)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
    }

    fn on_event(&mut self, event: UiEvent) {
        // Server URL field.
        if event.target_key() == Some(KEY_SERVER) {
            text_input::apply_event(&mut self.server, &mut self.selection, KEY_SERVER, &event);
            return;
        }
        // Token field. Mirrors the build-side `password()` opt so
        // `apply_event_with` can suppress copy/cut on the masked
        // value (paste in is still allowed — password managers).
        if event.target_key() == Some(KEY_TOKEN) {
            text_input::apply_event_with(
                &mut self.token,
                &mut self.selection,
                KEY_TOKEN,
                &event,
                &TextInputOpts::default().password(),
            );
            return;
        }
        // Remember checkbox.
        if checkbox::apply_event(&mut self.remember, &event, KEY_REMEMBER) {
            return;
        }
        // Connect button.
        if event.is_click_or_activate(KEY_CONNECT) {
            self.submit();
            return;
        }
        // Mirror runtime selection changes onto our owned cursor —
        // same plumbing the chat app uses.
        if let Some(sel) = event.selection.clone() {
            self.selection = sel;
        }
    }

    fn selection(&self) -> Selection {
        self.selection.clone()
    }

    fn theme(&self) -> Theme {
        // Match the chat phase so the post-login transition doesn't
        // flash a different palette.
        Theme::radix_slate_blue_dark()
    }
}
