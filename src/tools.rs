//! Tool infrastructure invoked from inside a step.
//!
//! [`mcp`] is the HTTP client that talks to external MCP hosts.
//! [`builtin_tools`] is the in-process tool surface that lets a task
//! edit its own pod configuration.
//! [`host_env_link`] is the v2 daemon link — server-side WebSocket
//! endpoint and live registry.
//! [`shared_mcp_catalog`] is the durable store for shared MCP hosts
//! (third-party endpoints with their own auth).

pub mod about_docs;
pub mod builtin_tools;
pub mod host_env_link;
pub mod mcp;
pub mod shared_mcp_catalog;
