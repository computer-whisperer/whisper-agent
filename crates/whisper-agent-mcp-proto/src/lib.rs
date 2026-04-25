//! MCP and JSON-RPC types shared by whisper-agent's MCP servers.
//!
//! Spec: <https://modelcontextprotocol.io/specification/2025-06-18>
//!
//! Hand-rolled for the subset whisper-agent actually exercises. Both
//! `whisper-agent-mcp-host` (per-task filesystem tools) and
//! `whisper-agent-mcp-fetch` (singleton web-fetch) speak this wire format,
//! so the types live here rather than being duplicated.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROTOCOL_VERSION: &str = "2025-06-18";

// ---------- JSON-RPC envelope ----------

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    /// Missing for notifications.
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// Standard JSON-RPC error codes (the subset we currently emit).
pub mod error_codes {
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
}

// ---------- MCP: initialize ----------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: String,
    // Extra fields (capabilities, clientInfo, etc.) are silently accepted but unused for MVP.
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Implementation {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: &'static str,
    pub capabilities: ServerCapabilities,
    pub server_info: Implementation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct ServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
}

#[derive(Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    pub list_changed: bool,
}

// ---------- MCP: tools/list ----------

#[derive(Debug, Serialize)]
pub struct ListToolsResult {
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<ToolAnnotations>,
}

#[derive(Debug, Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolAnnotations {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_only_hint: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destructive_hint: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotent_hint: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_world_hint: Option<bool>,
}

// ---------- MCP: tools/call ----------

#[derive(Debug, Deserialize)]
pub struct CallToolParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolResult {
    pub content: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl CallToolResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: text.into() }],
            is_error: None,
        }
    }

    pub fn error_text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: text.into() }],
            is_error: Some(true),
        }
    }

    /// Wrap a single base64-encoded image as a successful tool result.
    /// `mime_type` is the IANA media type (`image/png`, `image/jpeg`, …);
    /// the agent runtime decodes and normalizes against its accepted
    /// MIME set on the way in.
    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Image {
                data: data.into(),
                mime_type: mime_type.into(),
            }],
            is_error: None,
        }
    }

    /// Wrap a single base64-encoded binary resource (e.g. a PDF) as a
    /// successful tool result. `uri` should be a stable identifier
    /// (typically `file:///<path>`); `mime_type` is the IANA media
    /// type (`application/pdf`).
    pub fn resource_blob(
        uri: impl Into<String>,
        mime_type: impl Into<String>,
        blob: impl Into<String>,
    ) -> Self {
        Self {
            content: vec![ContentBlock::Resource {
                resource: EmbeddedResource {
                    uri: uri.into(),
                    mime_type: Some(mime_type.into()),
                    text: None,
                    blob: Some(blob.into()),
                },
            }],
            is_error: None,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// MCP 2025-06-18 image content. `data` is base64-encoded bytes;
    /// `mime_type` carries the IANA media type. Field renamed to
    /// `mimeType` on the wire to match the spec.
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// MCP 2025-06-18 embedded-resource content. Used for binary
    /// payloads that aren't images — today, PDFs from `view_pdf`. The
    /// inner [`EmbeddedResource`] carries either inline `text` or
    /// base64-encoded `blob` bytes plus a stable `uri` identifier.
    Resource {
        resource: EmbeddedResource,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddedResource {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Inline text content. Set when the resource is text; mutually
    /// exclusive with `blob` per the MCP spec.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded bytes. Set when the resource is binary; mutually
    /// exclusive with `text`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}
