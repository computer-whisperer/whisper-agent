//! `list_host_env_providers` — scheduler intercept that enumerates the
//! host-env (sandbox) providers registered on this server, alongside a
//! per-entry marker for whether the current pod's `[[allow.host_env]]`
//! table references each provider.
//!
//! Cheap, synchronous — reads the scheduler's in-memory registry.
//! Control-plane tokens are never exposed; `has_token` is a boolean
//! presence flag only.

use serde::Deserialize;
use serde_json::{Value, json};

use super::LIST_HOST_ENV_PROVIDERS;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: LIST_HOST_ENV_PROVIDERS.into(),
        description: "List the host-env (sandbox) providers this server is configured with. \
                      Each entry is `(name, url, has_token, reachability, origin, \
                      in_pod_scope)`. `in_pod_scope` is true when at least one of the pod's \
                      `[[allow.host_env]]` entries has `provider = <name>` — those are the \
                      providers the pod's sandbox bindings actually draw from. A provider \
                      with `in_pod_scope=false` is registered on the server but no pod \
                      sandbox references it; using it requires adding an `[[allow.host_env]]` \
                      entry to `pod.toml`, which needs the `pod_modify` capability. \
                      `has_token` is a boolean presence flag; the token itself is never \
                      surfaced to tools. Use this to see which sandbox providers exist and \
                      which are already usable from this pod."
            .into(),
        input_schema: json!({ "type": "object", "properties": {} }),
        annotations: ToolAnnotations {
            title: Some("List host-env providers".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            // Registry is server-local state.
            open_world_hint: Some(false),
        },
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ListHostEnvProvidersArgs {}

pub fn parse_args(value: Value) -> Result<ListHostEnvProvidersArgs, String> {
    serde_json::from_value::<ListHostEnvProvidersArgs>(value)
        .map_err(|e| format!("invalid list_host_env_providers arguments: {e}"))
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
