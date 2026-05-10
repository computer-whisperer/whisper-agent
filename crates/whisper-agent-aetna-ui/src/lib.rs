//! whisper-agent-aetna-ui — chat UI built on aetna.
//!
//! Single ui implementation across browser + native targets:
//!   - Browser: `web_entry` is the `#[wasm_bindgen(start)]` cdylib
//!     entry that the agent embeds via rust-embed and serves at
//!     `/pkg/whisper_agent_aetna_ui.js`. Owns the wgpu+winit host
//!     shell + WebSocket transport for the browser.
//!   - Native: `whisper-agent-desktop-aetna` consumes this crate as
//!     an rlib, wraps it in `aetna-winit-wgpu`, and runs its own
//!     tokio+tungstenite WS bridge.
//!
//! The crate owns two platform-agnostic aetna [`App`]s: [`ChatApp`]
//! (post-auth chat surface) and [`LoginApp`] (pre-connection
//! credentials form, used by the native client; the browser path
//! takes login through a static HTML form before the wasm bundle
//! loads). Hosts wire them through a phase-switching wrapper of
//! their choosing.

mod app;
mod branding;
mod cron_preview;
mod icons;
mod login;
#[cfg(target_arch = "wasm32")]
mod web_entry;

pub use app::{AttachmentIngress, ChatApp, Inbound, InboundEvent, SendFn};
pub use login::{LoginApp, LoginInput, SubmitFn};

pub use aetna_core;
