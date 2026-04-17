//! Tool infrastructure invoked from inside a step.
//!
//! [`mcp`] is the HTTP client that talks to external MCP hosts.
//! [`builtin_tools`] is the in-process tool surface that lets a task
//! edit its own pod configuration. [`sandbox`] provisions the host-env
//! jails that MCP hosts and other subprocesses run inside.

pub mod builtin_tools;
pub mod mcp;
pub mod sandbox;
