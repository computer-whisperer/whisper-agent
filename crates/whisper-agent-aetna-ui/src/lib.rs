//! whisper-agent-aetna-ui — chat UI built on aetna.
//!
//! Sibling to `whisper-agent-webui` (egui). The host shell — wasm
//! browser entry / native window / WebSocket transport — lives in
//! `whisper-agent-desktop-aetna` and (eventually) a wasm entry beside
//! this module. This crate owns two platform-agnostic aetna [`App`]s:
//! [`ChatApp`] (post-auth chat surface) and [`LoginApp`]
//! (pre-connection credentials form). Hosts wire them through a
//! phase-switching wrapper of their choosing.

mod app;
mod cron_preview;
mod icons;
mod login;
#[cfg(target_arch = "wasm32")]
mod web_entry;

pub use app::{AttachmentIngress, ChatApp, Inbound, InboundEvent, SendFn};
pub use login::{LoginApp, LoginInput, SubmitFn};

pub use aetna_core;
