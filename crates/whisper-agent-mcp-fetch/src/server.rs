//! HTTP server: single POST endpoint at `/mcp` dispatching JSON-RPC method names.
//! Mirrors the shape of `whisper-agent-mcp-host` so the scheduler can talk to
//! both with the same client.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde_json::{Value, json};
use tracing::{debug, warn};

use whisper_agent_mcp_proto::{
    CallToolParams, Implementation, InitializeParams, InitializeResult, JsonRpcRequest,
    JsonRpcResponse, ListToolsResult, PROTOCOL_VERSION, ServerCapabilities, ToolsCapability,
    error_codes,
};

use crate::tools::{self, FetchConfig};

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<FetchConfig>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/mcp", post(handle_post))
        .with_state(state)
}

async fn handle_post(State(state): State<AppState>, Json(req): Json<JsonRpcRequest>) -> Response {
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

    if req.id.is_none() {
        debug!(method = %req.method, "notification (no response)");
        return StatusCode::ACCEPTED.into_response();
    }

    let id = req.id.unwrap();
    let response = match req.method.as_str() {
        "initialize" => initialize(id, req.params),
        "tools/list" => list_tools(id),
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
    let parsed: Result<InitializeParams, _> =
        serde_json::from_value(params.unwrap_or(Value::Null));
    if let Err(e) = parsed {
        return JsonRpcResponse::error(
            id,
            error_codes::INVALID_PARAMS,
            format!("initialize params: {e}"),
        );
    }
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
            "Web fetch over http(s) with SSRF, size, timeout, and redirect caps.".into(),
        ),
    };
    JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
}

fn list_tools(id: Value) -> JsonRpcResponse {
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
    match tools::call(&state.cfg, &parsed.name, parsed.arguments).await {
        Ok(result) => JsonRpcResponse::success(id, serde_json::to_value(result).unwrap()),
        Err(tools::ToolDispatchError::UnknownTool(name)) => JsonRpcResponse::error(
            id,
            error_codes::METHOD_NOT_FOUND,
            format!("unknown tool: {name}"),
        ),
    }
}
