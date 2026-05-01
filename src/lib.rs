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

/// Pin rustls's process-level CryptoProvider to `ring` before any TLS
/// code runs. reqwest 0.13's `rustls-no-provider` feature deliberately
/// leaves provider selection to the application: `Client::new()` panics
/// with "No provider set" until something installs one. `main()` calls
/// this once at startup, and unit tests that build a `reqwest::Client`
/// call it from their setup; idempotent via `Once`, so repeat calls are
/// cheap.
pub fn ensure_default_crypto_provider() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // `.ok()` swallows the "already installed" error if some other
        // library beat us to it.
        rustls::crypto::ring::default_provider()
            .install_default()
            .ok();
    });
}
