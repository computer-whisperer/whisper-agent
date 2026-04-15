//! whisper-agent core: task-scheduler runtime, provider client, MCP client, HTTP server.
//!
//! Canonical conversation types live in the `whisper-agent-protocol` crate so the webui
//! can render them from wire snapshots without a server round-trip per event.

pub mod anthropic;
pub mod audit;
pub mod config;
pub mod mcp;
pub mod model;
pub mod openai_chat;
pub mod persist;
pub mod scheduler;
pub mod server;
pub mod task;
pub mod turn;
