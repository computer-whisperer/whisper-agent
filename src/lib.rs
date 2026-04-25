//! whisper-agent core: task-scheduler runtime, provider client, MCP client, HTTP server.
//!
//! Canonical conversation types live in the `whisper-agent-protocol` crate so the webui
//! can render them from wire snapshots without a server round-trip per event.

pub mod functions;
pub mod knowledge;
pub mod mcp_oauth;
pub mod permission;
pub mod pod;
pub mod providers;
pub mod runtime;
pub mod server;
pub mod tools;
