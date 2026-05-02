//! Tool-catalog discovery.
//!
//! On daemon startup we spawn one short-lived probe worker, run MCP
//! `tools/list`, translate the descriptors into the host-proto shape,
//! and cache the result for use in [`whisper_agent_host_proto::Frame::Hello`].
//! The probe worker is then killed (its `kill_on_drop` does the work).
//!
//! Why probe instead of baking a `const` in the daemon binary? The
//! design doc says "tools change rarely (compiled into the daemon)" —
//! both true here, since the daemon ships in lock-step with the
//! worker. But the worker binary is the actual source of truth, so
//! probing eliminates a second copy of the schema that could drift
//! from the worker's behavior. The probe runs once at startup so it's
//! not on the hot path; if the worker can't be spawned, daemon
//! startup fails loudly — which is what we want.

use std::net::IpAddr;

use serde_json::Value;
use tracing::{debug, info};
use whisper_agent_host_proto::{ToolAnnotations, ToolDescriptor};
use whisper_agent_protocol::sandbox::{NetworkPolicy, PathAccess};

use crate::worker::{Worker, WorkerError, spawn};

/// Errors from [`probe_tool_catalog`]. Distinct from [`WorkerError`]
/// so the caller can tell "daemon startup couldn't even spawn a
/// worker" from "daemon startup spawned the worker but couldn't get
/// a tool list out of it."
#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("probe worker failed to spawn: {0}")]
    Spawn(#[from] WorkerError),
    #[error("MCP request to probe worker failed: {0}")]
    McpRequest(String),
    #[error("MCP response from probe worker malformed: {0}")]
    McpResponse(String),
}

/// Spawn a probe worker against `probe_workspace`, ask it for its
/// tool list, return the host-proto-shaped descriptors. The probe
/// worker is killed on return.
///
/// `probe_workspace` is an arbitrary directory the daemon can grant
/// the probe read+write access to. The actual filesystem doesn't
/// matter — the probe never reads or writes a file, it only answers
/// `tools/list`. A tempdir is fine.
pub async fn probe_tool_catalog(
    mcp_host_bin: &str,
    bind_ip: IpAddr,
    probe_workspace: &str,
) -> Result<Vec<ToolDescriptor>, CatalogError> {
    let spec = whisper_agent_host_proto::HostEnvSpec::Landlock {
        allowed_paths: vec![PathAccess::read_write(probe_workspace)],
        network: NetworkPolicy::Isolated,
    };
    debug!(
        %probe_workspace,
        "spawning probe worker for tool-catalog discovery"
    );
    let worker = spawn(&spec, mcp_host_bin, bind_ip, None).await?;
    let descriptors = list_tools(&worker).await?;
    info!(count = descriptors.len(), "probed worker tool catalog");
    // `worker` drops here → child gets killed via `kill_on_drop`.
    Ok(descriptors)
}

async fn list_tools(worker: &Worker) -> Result<Vec<ToolDescriptor>, CatalogError> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
    });
    let resp = reqwest::Client::new()
        .post(&worker.mcp_url)
        .bearer_auth(&worker.mcp_token)
        .json(&body)
        .send()
        .await
        .map_err(|e| CatalogError::McpRequest(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(CatalogError::McpRequest(format!(
            "tools/list returned HTTP {}",
            resp.status()
        )));
    }
    let value: Value = resp
        .json()
        .await
        .map_err(|e| CatalogError::McpResponse(format!("body decode: {e}")))?;
    let tools = value
        .get("result")
        .and_then(|r| r.get("tools"))
        .and_then(|t| t.as_array())
        .ok_or_else(|| CatalogError::McpResponse("response missing result.tools array".into()))?;
    tools
        .iter()
        .map(translate_tool)
        .collect::<Result<Vec<_>, _>>()
}

/// Translate one MCP `tools/list` entry into a host-proto
/// [`ToolDescriptor`]. The two shapes are nearly identical; the only
/// real work is moving the camelCase MCP-wire field names onto the
/// snake_case host-proto fields, and pulling annotation hints
/// out of an optional sibling object.
fn translate_tool(value: &Value) -> Result<ToolDescriptor, CatalogError> {
    let name = value
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| CatalogError::McpResponse("tool missing `name`".into()))?
        .to_string();
    let description = value
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let input_schema = value
        .get("inputSchema")
        .cloned()
        .ok_or_else(|| CatalogError::McpResponse(format!("tool `{name}` missing `inputSchema`")))?;
    let annotations = match value.get("annotations") {
        None => ToolAnnotations::default(),
        Some(a) => translate_annotations(a),
    };
    Ok(ToolDescriptor {
        name,
        description,
        input_schema,
        annotations,
    })
}

fn translate_annotations(value: &Value) -> ToolAnnotations {
    let s = |key: &str| value.get(key).and_then(|v| v.as_str()).map(str::to_string);
    let b = |key: &str| value.get(key).and_then(|v| v.as_bool());
    ToolAnnotations {
        title: s("title"),
        read_only_hint: b("readOnlyHint"),
        destructive_hint: b("destructiveHint"),
        idempotent_hint: b("idempotentHint"),
        open_world_hint: b("openWorldHint"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn translate_minimal_tool() {
        let v = json!({
            "name": "read_file",
            "description": "Read a file.",
            "inputSchema": { "type": "object" },
        });
        let t = translate_tool(&v).unwrap();
        assert_eq!(t.name, "read_file");
        assert_eq!(t.description, "Read a file.");
        assert_eq!(t.input_schema, json!({"type":"object"}));
        assert_eq!(t.annotations, ToolAnnotations::default());
    }

    #[test]
    fn translate_tool_with_annotations() {
        let v = json!({
            "name": "bash",
            "description": "Run a shell command.",
            "inputSchema": { "type": "object" },
            "annotations": {
                "title": "Shell",
                "readOnlyHint": false,
                "destructiveHint": true,
                "idempotentHint": false,
                "openWorldHint": true,
            }
        });
        let t = translate_tool(&v).unwrap();
        assert_eq!(t.annotations.title.as_deref(), Some("Shell"));
        assert_eq!(t.annotations.read_only_hint, Some(false));
        assert_eq!(t.annotations.destructive_hint, Some(true));
        assert_eq!(t.annotations.idempotent_hint, Some(false));
        assert_eq!(t.annotations.open_world_hint, Some(true));
    }

    #[test]
    fn translate_rejects_missing_name() {
        let v = json!({
            "description": "no name",
            "inputSchema": {},
        });
        let e = translate_tool(&v).unwrap_err();
        assert!(matches!(e, CatalogError::McpResponse(_)));
    }

    #[test]
    fn translate_rejects_missing_input_schema() {
        let v = json!({
            "name": "x",
            "description": "no schema",
        });
        let e = translate_tool(&v).unwrap_err();
        match e {
            CatalogError::McpResponse(msg) => assert!(msg.contains("inputSchema"), "got: {msg}"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn translate_accepts_missing_description() {
        let v = json!({
            "name": "x",
            "inputSchema": {},
        });
        let t = translate_tool(&v).unwrap();
        assert_eq!(t.description, "");
    }

    #[test]
    fn translate_accepts_partial_annotations() {
        let v = json!({
            "name": "x",
            "inputSchema": {},
            "annotations": { "readOnlyHint": true }
        });
        let t = translate_tool(&v).unwrap();
        assert_eq!(t.annotations.read_only_hint, Some(true));
        assert_eq!(t.annotations.destructive_hint, None);
        assert_eq!(t.annotations.title, None);
    }
}
