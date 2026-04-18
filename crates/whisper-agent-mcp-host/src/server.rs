//! HTTP server: single POST endpoint at `/mcp` dispatching JSON-RPC method names.
//!
//! Streamable-HTTP allows the server to respond with either a single JSON body or an SSE
//! stream. For MVP we always respond with JSON since none of the tools stream output yet.
//! No GET endpoint, no `Mcp-Session-Id` enforcement: the server is stateless across requests.
//!
//! Auth: when `AppState::expected_token` is `Some`, every request must present
//! `Authorization: Bearer <token>` matching it (constant-time compared). The token
//! is a per-sandbox secret the dispatcher hands the MCP host on stdin at startup,
//! then returns to the scheduler in the provision response — it scopes what any
//! local process with TCP reach can actually drive on the MCP wire.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use serde_json::{Value, json};
use subtle::ConstantTimeEq;
use tracing::{debug, warn};

use whisper_agent_mcp_proto::{
    CallToolParams, Implementation, InitializeParams, InitializeResult, JsonRpcRequest,
    JsonRpcResponse, ListToolsResult, PROTOCOL_VERSION, ServerCapabilities, ToolsCapability,
    error_codes,
};

use crate::tools;
use crate::workspace::Workspace;

#[derive(Clone)]
pub struct AppState {
    pub workspace: Arc<Workspace>,
    /// When `Some`, every `/mcp` request must present this token as
    /// `Authorization: Bearer <token>`. `None` means auth is explicitly
    /// disabled (`--no-auth`).
    pub expected_token: Option<Arc<String>>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/mcp", post(handle_post))
        .with_state(state)
}

/// Extract the bearer token from an Authorization header, if present.
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let h = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    h.strip_prefix("Bearer ")
}

async fn handle_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<JsonRpcRequest>,
) -> Response {
    if let Some(expected) = &state.expected_token {
        let presented = bearer_token(&headers).unwrap_or("");
        if presented.as_bytes().ct_eq(expected.as_bytes()).unwrap_u8() != 1 {
            return (StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response();
        }
    }
    debug!(method = %req.method, "rpc");

    if req.jsonrpc != "2.0" {
        let id = req.id.unwrap_or(Value::Null);
        return Json(JsonRpcResponse::error(
            id,
            error_codes::INVALID_REQUEST,
            "jsonrpc must be \"2.0\"",
        ))
        .into_response();
    }

    // Notifications carry no id and expect no response (RFC). Return 202 Accepted.
    if req.id.is_none() {
        debug!(method = %req.method, "notification (no response)");
        return StatusCode::ACCEPTED.into_response();
    }

    let id = req.id.unwrap();
    let response = match req.method.as_str() {
        "initialize" => initialize(id, req.params),
        "tools/list" => list_tools(id, &state),
        "tools/call" => call_tool(id, &state, req.params).await,
        "ping" => JsonRpcResponse::success(id, json!({})),
        other => {
            warn!(method = %other, "unknown method");
            JsonRpcResponse::error(
                id,
                error_codes::METHOD_NOT_FOUND,
                format!("method not found: {other}"),
            )
        }
    };
    Json(response).into_response()
}

fn initialize(id: Value, params: Option<Value>) -> JsonRpcResponse {
    let parsed: Result<InitializeParams, _> = serde_json::from_value(params.unwrap_or(Value::Null));
    let _client_protocol = match parsed {
        Ok(p) => p.protocol_version,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("initialize params: {e}"),
            );
        }
    };

    let result = InitializeResult {
        protocol_version: PROTOCOL_VERSION,
        capabilities: ServerCapabilities {
            tools: Some(ToolsCapability { list_changed: false }),
        },
        server_info: Implementation {
            name: env!("CARGO_PKG_NAME").into(),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        instructions: Some(
            "Workspace-confined POSIX tools: read_file, write_file, bash. Paths are relative to the configured workspace root.".into(),
        ),
    };
    JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
}

fn list_tools(id: Value, _state: &AppState) -> JsonRpcResponse {
    let result = ListToolsResult {
        tools: tools::descriptors(),
        next_cursor: None,
    };
    JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
}

async fn call_tool(id: Value, state: &AppState, params: Option<Value>) -> JsonRpcResponse {
    let parsed: CallToolParams = match serde_json::from_value(params.unwrap_or(Value::Null)) {
        Ok(p) => p,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("tools/call params: {e}"),
            );
        }
    };
    match tools::call(&state.workspace, &parsed.name, parsed.arguments).await {
        Ok(result) => JsonRpcResponse::success(id, serde_json::to_value(result).unwrap()),
        Err(tools::ToolDispatchError::UnknownTool(name)) => JsonRpcResponse::error(
            id,
            error_codes::METHOD_NOT_FOUND,
            format!("unknown tool: {name}"),
        ),
    }
}
