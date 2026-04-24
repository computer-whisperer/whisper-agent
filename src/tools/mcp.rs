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
use tokio_util::sync::CancellationToken;
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
    /// The caller's per-thread [`CancellationToken`] fired before or
    /// during the RPC. Distinct from transport / RPC errors so the
    /// scheduler can tell a user cancel apart from a genuine failure
    /// and avoid marking the host-env `Lost`.
    #[error("cancelled")]
    Cancelled,
}

impl McpError {
    /// Whether this error looks like "the MCP host is gone" — i.e. the
    /// scheduler should mark the owning host-env `Lost` rather than
    /// surface it as a normal tool-returned-an-error. Covers:
    /// - connect failures and timeouts (daemon host unreachable),
    /// - 5xx from the MCP (MCP host crashed; daemon may or may not be),
    /// - 401/403 (daemon restarted and MCP host doesn't recognize our
    ///   per-sandbox bearer anymore — effectively session-gone),
    /// - 404 on the MCP URL (daemon restarted and the URL points
    ///   nowhere usable).
    ///
    /// Rpc / Malformed are NOT transport: the MCP host *responded*, so
    /// the session is alive — the tool just errored or the response
    /// was garbage. Those flow as a normal tool-call failure.
    pub fn is_transport_lost(&self) -> bool {
        match self {
            McpError::Http(e) => e.is_connect() || e.is_timeout() || e.is_request(),
            McpError::Transport { status, .. } => matches!(
                *status,
                401 | 403 | 404 | 408 | 500 | 502 | 503 | 504 | 522 | 523 | 524
            ),
            McpError::Rpc { .. } | McpError::Malformed(_) | McpError::Cancelled => false,
        }
    }
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
    ///
    /// `cancel` is the caller's per-thread cancellation signal. It
    /// aborts the call at three points:
    /// 1. Before `send()` resolves — returns [`McpError::Cancelled`] after
    ///    firing a best-effort `notifications/cancelled`.
    /// 2. While reading the response body (non-streaming path).
    /// 3. While the caller is driving the returned stream — the wrapper
    ///    ends the stream early and fires `notifications/cancelled`.
    pub async fn invoke(
        &self,
        name: &str,
        arguments: Value,
        cancel: &CancellationToken,
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
        let send = builder.send();
        let resp = tokio::select! {
            r = send => r?,
            _ = cancel.cancelled() => {
                self.best_effort_notify_cancelled(id, "client cancelled").await;
                return Err(McpError::Cancelled);
            }
        };
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
            // Wrap the SSE body so the stream ends — and
            // `notifications/cancelled` is sent best-effort — when the
            // cancel token fires. Dropping the inner reqwest stream
            // closes the TCP connection; the cancel notification is
            // graceful belt-and-suspenders for hosts that watch it.
            let inner = sse_tool_stream(resp);
            let cancel = cancel.clone();
            let http = self.http.clone();
            let url = self.url.clone();
            let bearer = self.bearer.clone();
            Ok(Box::pin(stream! {
                futures::pin_mut!(inner);
                loop {
                    tokio::select! {
                        next = inner.next() => {
                            match next {
                                Some(ev) => yield ev,
                                None => return,
                            }
                        }
                        _ = cancel.cancelled() => {
                            // Drop `inner` (closes the HTTP stream)
                            // and fire notifications/cancelled so the
                            // server can stop work promptly.
                            best_effort_cancel(&http, &url, bearer.as_deref(), id).await;
                            return;
                        }
                    }
                }
            }))
        } else {
            let parsed: JsonRpcResponse = tokio::select! {
                r = resp.json() => r?,
                _ = cancel.cancelled() => {
                    self.best_effort_notify_cancelled(id, "client cancelled").await;
                    return Err(McpError::Cancelled);
                }
            };
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

    /// Fire `notifications/cancelled` for an in-flight RPC id. Silently
    /// swallows any failure — the host may already be gone, and the
    /// cancel path must not fail regardless.
    async fn best_effort_notify_cancelled(&self, id: u64, reason: &str) {
        let _ = self
            .notify(
                "notifications/cancelled",
                Some(json!({ "requestId": id, "reason": reason })),
            )
            .await;
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
        // Accept both transports: MCP streamable-HTTP lets the server
        // pick JSON or SSE. In-tree whisper-agent-mcp-host always
        // returns JSON for one-shot RPCs; third-party servers (e.g.
        // GitHub's Copilot MCP) pick SSE even for `initialize` and
        // `tools/list`. Advertising both means we work with either.
        let mut builder = self
            .http
            .post(&self.url)
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
            read_json_rpc_from_sse(resp, id).await
        } else {
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

/// Free-function variant of `best_effort_notify_cancelled` that
/// doesn't borrow `McpSession` — used by the cancellation-aware stream
/// wrapper returned from `invoke`, which outlives the `&self` borrow.
async fn best_effort_cancel(http: &reqwest::Client, url: &str, bearer: Option<&str>, id: u64) {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": { "requestId": id, "reason": "client cancelled" },
    });
    let mut builder = http.post(url).json(&body);
    if let Some(tok) = bearer {
        builder = builder.bearer_auth(tok);
    }
    let _ = builder.send().await;
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

/// Read a single JSON-RPC response from an SSE body. Thin wrapper
/// around [`SseRpcParser`]: drives the reqwest byte-stream and feeds
/// each chunk into the pure parser. Splitting them lets the parser
/// be table-driven tested without spinning up a real HTTP server.
async fn read_json_rpc_from_sse(
    resp: reqwest::Response,
    expected_id: u64,
) -> Result<Value, McpError> {
    let mut bytes = resp.bytes_stream();
    let mut parser = SseRpcParser::new(expected_id);
    while let Some(next) = bytes.next().await {
        let chunk = next.map_err(|e| McpError::Malformed(format!("SSE byte stream: {e}")))?;
        if let Some(outcome) = parser.push(&chunk) {
            return outcome;
        }
    }
    Err(McpError::Malformed(
        "SSE stream ended before a matching JSON-RPC response arrived".into(),
    ))
}

/// Stateful SSE-RPC parser. Buffers byte chunks until complete event
/// frames arrive, then checks each frame against the expected id.
/// Held open across multiple `push` calls so a response split mid-
/// frame across reqwest chunks still parses cleanly.
///
/// `push` returns `Some` the moment a decision is final:
/// `Some(Ok(result))` on the matching response, `Some(Err(...))` on
/// an explicit JSON-RPC error frame. It returns `None` while the
/// caller should keep feeding chunks — notifications, unmatched-id
/// frames, and half-read frames all land there. Used by
/// [`McpSession::call`] for one-shot RPCs (`initialize`,
/// `tools/list`) when the server opted to respond via SSE instead
/// of inline JSON.
struct SseRpcParser {
    buf: String,
    expected_id: u64,
}

impl SseRpcParser {
    fn new(expected_id: u64) -> Self {
        Self {
            buf: String::new(),
            expected_id,
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Option<Result<Value, McpError>> {
        self.buf.push_str(&String::from_utf8_lossy(chunk));
        while let Some(frame) = take_sse_frame(&mut self.buf) {
            let Some(data) = extract_sse_data(&frame) else {
                continue;
            };
            let msg: Value = match serde_json::from_str(&data) {
                Ok(v) => v,
                Err(e) => {
                    warn!(error = %e, data = %data, "SSE RPC frame JSON parse failed");
                    continue;
                }
            };
            let Some(obj) = msg.as_object() else {
                continue;
            };
            let id_match = obj
                .get("id")
                .and_then(|v| v.as_u64())
                .map(|n| n == self.expected_id)
                .unwrap_or(false);
            if !id_match {
                continue;
            }
            if let Some(err) = obj.get("error") {
                let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(0) as i32;
                let message = err
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("<no message>")
                    .to_string();
                return Some(Err(McpError::Rpc { code, message }));
            }
            return Some(
                obj.get("result")
                    .cloned()
                    .ok_or_else(|| McpError::Malformed("SSE RPC response missing result".into())),
            );
        }
        None
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

    // ---------- SseRpcParser ----------

    fn push(parser: &mut SseRpcParser, s: &str) -> Option<Result<Value, McpError>> {
        parser.push(s.as_bytes())
    }

    #[test]
    fn sse_rpc_parser_returns_result_on_matching_id() {
        let mut p = SseRpcParser::new(7);
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":true}}\n\n";
        let outcome = push(&mut p, body).expect("should resolve after first frame");
        let result = outcome.expect("expected Ok");
        assert_eq!(result, serde_json::json!({"ok": true}));
    }

    #[test]
    fn sse_rpc_parser_resumes_across_chunk_boundary() {
        // Frame split mid-data across two `push` calls — common with
        // real reqwest reads since SSE traffic tends to be short
        // frames but chunk boundaries don't respect them.
        let mut p = SseRpcParser::new(1);
        assert!(push(&mut p, "data: {\"jsonrpc\":\"2.0\",\"id\":1,").is_none());
        let outcome = push(&mut p, "\"result\":{\"v\":42}}\n\n")
            .expect("second push should deliver the frame");
        let result = outcome.unwrap();
        assert_eq!(result, serde_json::json!({"v": 42}));
    }

    #[test]
    fn sse_rpc_parser_skips_unmatched_id_then_returns_match() {
        // Some servers send unrelated notifications / responses on the
        // same SSE channel. We must scan past them and only act on the
        // frame whose id matches.
        let mut p = SseRpcParser::new(9);
        let body = concat!(
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{\"pct\":50}}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":4,\"result\":{\"stale\":true}}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":9,\"result\":{\"match\":true}}\n\n",
        );
        let outcome = push(&mut p, body).expect("should find the id=9 frame");
        assert_eq!(outcome.unwrap(), serde_json::json!({"match": true}));
    }

    #[test]
    fn sse_rpc_parser_maps_rpc_error() {
        let mut p = SseRpcParser::new(3);
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":3,\"error\":{\"code\":-32000,\"message\":\"nope\"}}\n\n";
        let outcome = push(&mut p, body).expect("should resolve");
        let err = outcome.expect_err("expected Err");
        match err {
            McpError::Rpc { code, message } => {
                assert_eq!(code, -32000);
                assert_eq!(message, "nope");
            }
            other => panic!("expected Rpc err, got {other:?}"),
        }
    }

    #[test]
    fn sse_rpc_parser_malformed_when_result_field_missing() {
        // Malformed final frame (id matches, but neither result nor
        // error is present) surfaces as Malformed.
        let mut p = SseRpcParser::new(2);
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":2}\n\n";
        let outcome = push(&mut p, body).expect("should resolve");
        let err = outcome.expect_err("expected Err");
        assert!(
            matches!(err, McpError::Malformed(ref m) if m.contains("missing result")),
            "got {err:?}"
        );
    }

    #[test]
    fn sse_rpc_parser_ignores_garbage_frames() {
        // Non-JSON data: lines shouldn't abort — we just log and keep
        // looking, because servers may emit comments or keepalives
        // the spec doesn't categorize for us.
        let mut p = SseRpcParser::new(5);
        let body = concat!(
            "data: not valid json at all\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":5,\"result\":\"yes\"}\n\n",
        );
        let outcome = push(&mut p, body).expect("should skip garbage, land on match");
        assert_eq!(outcome.unwrap(), serde_json::json!("yes"));
    }

    #[test]
    fn sse_rpc_parser_returns_none_until_frame_boundary() {
        // Data without the terminating double-newline isn't a frame
        // yet — parser must wait for it. This lets the async wrapper
        // know whether to keep reading.
        let mut p = SseRpcParser::new(1);
        assert!(
            push(
                &mut p,
                "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":true}"
            )
            .is_none()
        );
    }
}
