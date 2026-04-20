//! MCP HTTP client.
//!
//! Models a single MCP host as a stateful session that owns a [`reqwest::Client`] and
//! an `Mcp-Session-Id` once the server assigns one. Tool invocations return a
//! `Stream<ToolEvent>` regardless of transport: the server may respond either
//! with a single JSON body (non-streaming tools) or with
//! `text/event-stream` interleaving `notifications/tools/output` chunks
//! before the final response (streaming tools like `bash`). The streaming
//! shape is the API's contract; which wire transport the server picks is
//! an implementation detail.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use async_stream::stream;
use futures::{Stream, StreamExt};
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

#[derive(Deserialize, Debug, Clone)]
pub struct CallToolResult {
    pub content: Vec<McpContentBlock>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

#[derive(Debug)]
pub enum ToolEvent {
    /// Streaming content fragment — a text chunk, image, or structured
    /// payload produced mid-call. Emitted over the SSE transport as
    /// `notifications/tools/output` notifications arrive (bash output
    /// lines, today). Non-streaming tools never emit this variant.
    Content(McpContentBlock),
    /// Final outcome of a tool call. Every tool call produces exactly
    /// one of these; it carries the full result the model will see.
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
    pub async fn connect(url: impl Into<String>, bearer: Option<String>) -> Result<Self, McpError> {
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

    /// Invoke a tool. The returned stream emits zero or more
    /// [`ToolEvent::Content`]s as output arrives (SSE transport only) and
    /// terminates with exactly one [`ToolEvent::Completed`]. A non-streaming
    /// tool yields only the `Completed` event.
    pub async fn invoke(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<Pin<Box<dyn Stream<Item = ToolEvent> + Send>>, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: "tools/call",
            params: Some(json!({ "name": name, "arguments": arguments })),
        };
        debug!(method = "tools/call", tool = name, "rpc out");

        let mut builder = self
            .http
            .post(&self.url)
            // Advertise both transports; the server chooses per-call. MVP
            // tools get JSON, streaming tools (bash) get SSE.
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .json(&req);
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
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        if content_type.starts_with("text/event-stream") {
            Ok(Box::pin(sse_tool_stream(resp)))
        } else {
            let parsed: JsonRpcResponse = resp.json().await?;
            if let Some(e) = parsed.error {
                return Err(McpError::Rpc {
                    code: e.code,
                    message: e.message,
                });
            }
            let result = parsed
                .result
                .ok_or_else(|| McpError::Malformed("response missing result".into()))?;
            let tool_result: CallToolResult =
                serde_json::from_value(result).map_err(|e| McpError::Malformed(e.to_string()))?;
            Ok(Box::pin(
                stream! { yield ToolEvent::Completed(tool_result); },
            ))
        }
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

/// Parse the `text/event-stream` body of a streaming `tools/call` into a
/// stream of [`ToolEvent`]s.
///
/// Each SSE `data:` payload is a JSON-RPC message. Notifications
/// (`method == "notifications/tools/output"`) become `ToolEvent::Content`.
/// The first response with a `result` field becomes `ToolEvent::Completed`
/// and terminates the stream. Malformed frames and unrelated methods are
/// logged and skipped — we never leak partial parsing errors into the
/// tool-output stream the UI consumes.
fn sse_tool_stream(resp: reqwest::Response) -> impl Stream<Item = ToolEvent> + Send {
    stream! {
        let mut bytes = resp.bytes_stream();
        let mut buf = String::new();
        'outer: while let Some(next) = bytes.next().await {
            let chunk = match next {
                Ok(b) => b,
                Err(e) => {
                    warn!(error = %e, "SSE byte stream errored mid-tool-call");
                    return;
                }
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(frame) = take_sse_frame(&mut buf) {
                let Some(data) = extract_sse_data(&frame) else {
                    continue;
                };
                match serde_json::from_str::<Value>(&data) {
                    Ok(Value::Object(msg)) => {
                        // Response: has `id` + (`result` or `error`).
                        if msg.contains_key("id") {
                            if let Some(err) = msg.get("error") {
                                warn!(error = %err, "SSE final response carried JSON-RPC error");
                                return;
                            }
                            if let Some(result) = msg.get("result") {
                                match serde_json::from_value::<CallToolResult>(result.clone()) {
                                    Ok(parsed) => {
                                        yield ToolEvent::Completed(parsed);
                                        return;
                                    }
                                    Err(e) => {
                                        warn!(error = %e, "SSE final result malformed");
                                        return;
                                    }
                                }
                            }
                        }
                        // Notification: carries `method` but no `id`.
                        if let Some(method) = msg.get("method").and_then(|m| m.as_str())
                            && method == "notifications/tools/output"
                            && let Some(content) = msg.get("params").and_then(|p| p.get("content"))
                            && let Ok(block) =
                                serde_json::from_value::<McpContentBlock>(content.clone())
                        {
                            yield ToolEvent::Content(block);
                            continue 'outer;
                        }
                    }
                    Ok(_) => {
                        warn!(data = %data, "SSE frame was not a JSON object");
                    }
                    Err(e) => {
                        warn!(error = %e, data = %data, "SSE frame JSON parse failed");
                    }
                }
            }
        }
        // Connection closed without a final response — log and let the
        // caller see an empty tail. The dispatch layer surfaces "no
        // Completed event in tool stream" in that case.
        warn!("SSE stream ended without final tools/call response");
    }
}

/// Extract the next complete SSE event frame from `buf`. A frame is the text
/// preceding the first `\n\n` (or `\r\n\r\n`) boundary; processed bytes are
/// drained from `buf`. Returns `None` while no full frame is buffered.
fn take_sse_frame(buf: &mut String) -> Option<String> {
    let idx_nn = buf.find("\n\n");
    let idx_rnrn = buf.find("\r\n\r\n");
    let (pos, boundary_len) = match (idx_nn, idx_rnrn) {
        (Some(a), Some(b)) if a <= b => (a, 2),
        (Some(_), Some(b)) => (b, 4),
        (Some(a), None) => (a, 2),
        (None, Some(b)) => (b, 4),
        (None, None) => return None,
    };
    let frame: String = buf.drain(..pos).collect();
    buf.drain(..boundary_len);
    Some(frame)
}

/// Join the `data:` lines of an SSE frame. Per the spec, multiple
/// `data:` lines concatenate with `\n`; the leading single space after
/// the colon is stripped if present. Non-`data:` fields (`event:`,
/// `id:`, `retry:`, comments) are ignored.
fn extract_sse_data(frame: &str) -> Option<String> {
    let mut out = String::new();
    for line in frame.lines() {
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(rest);
    }
    if out.is_empty() { None } else { Some(out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_sse_frame_splits_on_double_newline() {
        let mut buf = String::from("data: a\n\ndata: b\n\npartial");
        assert_eq!(take_sse_frame(&mut buf).as_deref(), Some("data: a"));
        assert_eq!(take_sse_frame(&mut buf).as_deref(), Some("data: b"));
        assert_eq!(take_sse_frame(&mut buf), None);
        assert_eq!(buf, "partial");
    }

    #[test]
    fn take_sse_frame_handles_crlf() {
        let mut buf = String::from("data: hi\r\n\r\n");
        assert_eq!(take_sse_frame(&mut buf).as_deref(), Some("data: hi"));
        assert!(buf.is_empty());
    }

    #[test]
    fn extract_sse_data_joins_multiple_data_lines() {
        let frame = "event: msg\ndata: first\ndata: second\nid: 1";
        assert_eq!(extract_sse_data(frame).as_deref(), Some("first\nsecond"));
    }

    #[test]
    fn extract_sse_data_handles_missing_leading_space() {
        let frame = "data:nospace";
        assert_eq!(extract_sse_data(frame).as_deref(), Some("nospace"));
    }

    #[test]
    fn extract_sse_data_none_when_frame_has_no_data_line() {
        let frame = "event: ping\nid: 5";
        assert!(extract_sse_data(frame).is_none());
    }
}
