//! Tool infrastructure invoked from inside a step.
//!
//! [`mcp`] is the HTTP client that talks to external MCP hosts.
//! [`builtin_tools`] is the in-process tool surface that lets a task
//! edit its own pod configuration. [`sandbox`] provisions the host-env
//! jails that MCP hosts and other subprocesses run inside.
//! [`host_env_catalog`] is the durable store sandbox provider entries
//! persist into across restarts.
//! [`shared_mcp_catalog`] is the sibling durable store for shared
//! MCP hosts (third-party endpoints with their own auth).

pub mod builtin_tools;
pub mod host_env_catalog;
pub mod mcp;
pub mod pod_about_docs;
pub mod sandbox;
pub mod shared_mcp_catalog;
