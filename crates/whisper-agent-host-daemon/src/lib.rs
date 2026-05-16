//! Host-side daemon for the v2 host-env protocol.
//!
//! Dials whisper-agent over an authenticated WebSocket
//! (`/v1/host_env_link`), advertises a tool catalog discovered by
//! probing the bundled worker binary, then provisions and proxies a
//! per-session sandbox per incoming [`whisper_agent_host_proto::Frame::OpenSession`].
//!
//! See `docs/design_host_env_protocol.md`.
//!
//! The connection direction (daemon → scheduler) is the whole point:
//! whisper-agent terminates HTTPS at a stable address, daemons live on
//! arbitrary hosts (laptops, k8s pods, edge boxes) where inbound
//! reachability and TLS termination are someone else's problem.
//!
//! # Phase 3 scope
//!
//! - One WebSocket dial per `Daemon::run`. Reconnect with exponential
//!   backoff is at the binary level (`main.rs`), not the library level —
//!   library callers get a clean "this connection is done" signal and
//!   decide whether to dial again.
//! - Per-`OpenSession` worker spawn via `whisper-agent-mcp-host` over
//!   loopback HTTP + per-session bearer (the same machinery
//!   `whisper-agent-sandbox` uses today, lifted into `worker.rs`).
//! - `InvokeTool` is proxied as MCP `tools/call` and the result is
//!   sent back as a single `ToolFinal`. Streaming chunks are not yet
//!   surfaced (mirrors the server-side phase 2b cut).
//! - Background tasks, hooks, `UpdateSession` — accepted on the wire
//!   (the protocol crate has the variants) but not yet implemented;
//!   they land in phases 5/6.

pub mod catalog;
pub mod config;
pub mod connection;
pub mod credentials;
pub mod sessions;
pub mod worker;

pub use connection::{ConnectError, run_connection};
