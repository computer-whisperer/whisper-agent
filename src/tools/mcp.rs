//! MCP HTTP client.
//!
//! Models a single MCP host as a stateful session that owns a [`reqwest::Client`] and
//! an `Mcp-Session-Id` once the server assigns one. Tool invocations return a
//! `Stream<ToolEvent>` even when the underlying HTTP response is a single JSON object —
//! the streaming shape is the API's contract; the transport is an implementation detail.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use async_stream::stream;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tracing::{debug, warn};

const PROTOCOL_VERSION: &str = "2025-06-18";

#[derive(Debug, Error)]
pub enum McpError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("transport error {status}: {body}")]
    Transport { status: u16, body: String },
    #[error("rpc error {code}: {message}")]
    Rpc { code: i32, message: String },
    #[error("malformed response: {0}")]
    Malformed(String),
}

#[derive(Deserialize, Debug, Clone)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
    /// Server-declared safety/capability hints. All fields are tri-state (unset / true /
    /// false) because the spec lets servers be explicit about defaults.
    #[serde(default)]
    pub annotations: ToolAnnotations,
}

/// MCP 2025-06-18 tool annotations. See [`ToolDescriptor::annotations`].
#[derive(Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolAnnotations {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub read_only_hint: Option<bool>,
    #[serde(default)]
    pub destructive_hint: Option<bool>,
    #[serde(default)]
    pub idempotent_hint: Option<bool>,
    #[serde(default)]
    pub open_world_hint: Option<bool>,
}

impl ToolAnnotations {
    /// Server declared this tool is read-only (no side effects).
    pub fn is_read_only(&self) -> bool {
        self.read_only_hint == Some(true)
    }

    /// Server declared this tool is destructive.
    pub fn is_destructive(&self) -> bool {
        self.destructive_hint == Some(true)
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpContentBlock {
    Text { text: String },
}

#[derive(Deserialize, Debug)]
pub struct CallToolResult {
    pub content: Vec<McpContentBlock>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

#[derive(Debug)]
pub enum ToolEvent {
    /// Final outcome of a tool call. The MVP transport delivers exactly one of these per call.
    Completed(CallToolResult),
}

#[derive(Serialize, Debug)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'static str,
    id: u64,
    method: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Deserialize, Debug)]
struct JsonRpcResponse {
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<JsonRpcErrorObj>,
}

#[derive(Deserialize, Debug)]
struct JsonRpcErrorObj {
    code: i32,
    message: String,
}

#[derive(Debug)]
pub struct McpSession {
    http: reqwest::Client,
    url: String,
    /// Bearer token attached as `Authorization: Bearer <token>` on
    /// every request. `None` when the server was configured for
    /// anonymous access (dev / shared-host loopback usage).
    bearer: Option<String>,
    next_id: AtomicU64,
}

impl McpSession {
    /// Open a session against the given MCP server URL and complete
    /// the `initialize` handshake. `bearer`, when set, is presented as
    /// `Authorization: Bearer <token>` on every subsequent request —
    /// per-sandbox MCP hosts issue this token at provision time.
    pub async fn connect(
        url: impl Into<String>,
        bearer: Option<String>,
    ) -> Result<Self, McpError> {
        let session = Self {
            http: reqwest::Client::new(),
            url: url.into(),
            bearer,
            next_id: AtomicU64::new(1),
        };
        session.initialize().await?;
        Ok(session)
    }

    async fn initialize(&self) -> Result<(), McpError> {
        let _: Value = self
            .call(
                "initialize",
                Some(json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": env!("CARGO_PKG_NAME"),
                        "version": env!("CARGO_PKG_VERSION"),
                    }
                })),
            )
            .await?;
        // Per spec, the client should send `notifications/initialized` after handshake. The MVP
        // server treats notifications as no-ops and 202s them; we send it to be well-behaved.
        if let Err(e) = self.notify("notifications/initialized", None).await {
            warn!(error = ?e, "notifications/initialized failed (non-fatal)");
        }
        Ok(())
    }

    pub async fn list_tools(&self) -> Result<Vec<ToolDescriptor>, McpError> {
        let result = self.call("tools/list", None).await?;
        let tools = result
            .get("tools")
            .ok_or_else(|| McpError::Malformed("tools/list missing 'tools'".into()))?
            .clone();
        serde_json::from_value(tools).map_err(|e| McpError::Malformed(e.to_string()))
    }

    /// Invoke a tool. Returns a stream that emits exactly one [`ToolEvent::Completed`]
    /// for the MVP transport (no streaming output yet); the streaming shape is in place
    /// so adding multi-event tools later is purely a server-side change.
    pub async fn invoke(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<Pin<Box<dyn Stream<Item = ToolEvent> + Send>>, McpError> {
        let result = self
            .call(
                "tools/call",
                Some(json!({ "name": name, "arguments": arguments })),
            )
            .await?;
        let parsed: CallToolResult =
            serde_json::from_value(result).map_err(|e| McpError::Malformed(e.to_string()))?;
        let s = stream! { yield ToolEvent::Completed(parsed); };
        Ok(Box::pin(s))
    }

    async fn call(&self, method: &str, params: Option<Value>) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };
        debug!(method, "rpc out");
        let mut builder = self.http.post(&self.url).json(&req);
        if let Some(tok) = &self.bearer {
            builder = builder.bearer_auth(tok);
        }
        let resp = builder.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(McpError::Transport {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: JsonRpcResponse = resp.json().await?;
        if let Some(e) = parsed.error {
            return Err(McpError::Rpc {
                code: e.code,
                message: e.message,
            });
        }
        parsed
            .result
            .ok_or_else(|| McpError::Malformed("response missing both result and error".into()))
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
        // Notifications carry no id and expect no body.
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let mut builder = self.http.post(&self.url).json(&body);
        if let Some(tok) = &self.bearer {
            builder = builder.bearer_auth(tok);
        }
        let resp = builder.send().await?;
        let status = resp.status();
        if !status.is_success() && status.as_u16() != 202 {
            let body = resp.text().await.unwrap_or_default();
            return Err(McpError::Transport {
                status: status.as_u16(),
                body,
            });
        }
        Ok(())
    }
}
