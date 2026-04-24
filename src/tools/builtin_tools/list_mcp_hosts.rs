//! `list_mcp_hosts` — scheduler intercept that enumerates the
//! shared MCP hosts registered on this server, alongside a per-entry
//! marker for whether the current pod's `[allow.mcp_hosts]` admits
//! each name.
//!
//! Cheap, synchronous — reads the scheduler's in-memory catalog.
//! Tokens and OAuth state are never included: the public
//! `SharedMcpAuthPublic` classification (`none` / `bearer` / `oauth2`)
//! is all the agent sees.

use serde::Deserialize;
use serde_json::{Value, json};

use super::LIST_MCP_HOSTS;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: LIST_MCP_HOSTS.into(),
        description: "List the shared MCP hosts this server is configured with. Each entry is \
                      `(name, url, auth, connected, origin, in_pod_scope)`. `in_pod_scope` \
                      is true when the host appears in this pod's `[allow].mcp_hosts` — \
                      those hosts are the ones threads in this pod can actually bind to. \
                      Hosts with `in_pod_scope=false` exist at the server level but adding \
                      them to the pod requires editing `pod.toml`, which needs the \
                      `pod_modify` capability. `auth` is a classification only — token and \
                      OAuth state are never exposed to tools. Use this to discover which \
                      MCP endpoints are available before binding a thread or dispatching."
            .into(),
        input_schema: json!({ "type": "object", "properties": {} }),
        annotations: ToolAnnotations {
            title: Some("List shared MCP hosts".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            // Catalog is server-local state — no external world.
            open_world_hint: Some(false),
        },
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ListMcpHostsArgs {}

pub fn parse_args(value: Value) -> Result<ListMcpHostsArgs, String> {
    serde_json::from_value::<ListMcpHostsArgs>(value)
        .map_err(|e| format!("invalid list_mcp_hosts arguments: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_ok() {
        assert!(parse_args(json!({})).is_ok());
    }

    #[test]
    fn parse_tolerates_unknown_fields() {
        assert!(parse_args(json!({ "extra": true })).is_ok());
    }
}
