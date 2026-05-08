//! Shared auth primitives.
//!
//! Two pieces:
//! - [`Auth`] — the static config-file shape (`api_key` / `chatgpt_subscription`
//!   / `google_oauth`), used to deserialize `[backends.X].auth` from
//!   `whisper-agent.toml`.
//! - [`ClientAuth`] — the live, runtime view used by HTTP request code. Holds
//!   either a static API key or an `Arc<Mutex<CodexAuth>>` that refreshes
//!   itself in place. Built from a config [`Auth`] via
//!   [`ClientAuth::from_config`].
//!
//! [`CodexAuth`] is the OAuth-token reader/refresher for the Codex CLI's
//! `~/.codex/auth.json`.

mod auth;
mod client;
mod codex;

pub use auth::{Auth, ChatgptSubscriptionSource, GoogleOauthSource};
pub use client::ClientAuth;
pub use codex::CodexAuth;
