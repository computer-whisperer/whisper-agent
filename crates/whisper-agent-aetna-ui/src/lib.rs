//! whisper-agent-aetna-ui — chat UI built on aetna.
//!
//! Sibling to `whisper-agent-webui` (egui). The host shell — wasm
//! browser entry / native window / WebSocket transport — lives in
//! `whisper-agent-desktop-aetna` and (eventually) a wasm entry beside
//! this module. This crate owns the platform-agnostic [`ChatApp`]
//! that implements [`aetna_core::App`].

mod app;

pub use app::{ChatApp, Inbound, InboundEvent, SendFn};

pub use aetna_core;
