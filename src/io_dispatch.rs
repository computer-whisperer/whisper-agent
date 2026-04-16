//! I/O future construction for the scheduler.
//!
//! Translates a [`task::IoRequest`] into a `Future<Output = IoCompletion>` that the
//! scheduler can drop into its `FuturesUnordered`. The dispatcher reads scheduler-
//! owned state (sandbox provider, backend catalog, MCP session pools) through the
//! narrow `pub(crate)` accessors on [`Scheduler`]; it does not mutate.
//!
//! Per-IO-type knowledge (how to provision a sandbox, how to merge tool lists across
//! sessions, how to assemble a model request) lives here so adding a new
//! [`IoRequest`] variant only touches this module plus the task state machine — not
//! the scheduler's run loop or event-routing layer.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use tracing::{debug, warn};

use crate::mcp::{McpSession, ToolDescriptor as McpTool};
use crate::model::{CacheBreakpoint, ModelRequest, ToolSpec, default_cache_policy};
use crate::scheduler::Scheduler;
use crate::thread::{IoRequest, IoResult, OpId};

/// Completion message delivered by every I/O future the scheduler dispatches.
pub(crate) struct IoCompletion {
    pub(crate) task_id: String,
    pub(crate) op_id: OpId,
    pub(crate) result: IoResult,
}

pub(crate) type IoFuture = Pin<Box<dyn Future<Output = IoCompletion> + Send>>;

/// Build a future that executes one I/O op and yields an [`IoCompletion`].
pub(crate) fn build_io_future(scheduler: &Scheduler, task_id: String, req: IoRequest) -> IoFuture {
    match req {
        IoRequest::McpConnect { op_id } => mcp_connect(scheduler, task_id, op_id),
        IoRequest::ListTools { op_id } => list_tools(scheduler, task_id, op_id),
        IoRequest::ModelCall { op_id } => model_call(scheduler, task_id, op_id),
        IoRequest::ToolCall {
            op_id,
            tool_use_id,
            name,
            input,
        } => tool_call(scheduler, task_id, op_id, tool_use_id, name, input),
    }
}

fn mcp_connect(scheduler: &Scheduler, task_id: String, op_id: OpId) -> IoFuture {
    let task = scheduler.task(&task_id).expect("task exists");
    let fallback_url = task.config.mcp_host_url.clone();
    let sandbox_spec = task.config.sandbox.clone();
    let provider = scheduler.sandbox_provider().clone();

    Box::pin(async move {
        // Provision sandbox if the task has a non-None spec.
        let (mcp_url, sandbox_handle) = match provider.provision(&task_id, &sandbox_spec).await {
            Ok(handle) => {
                let url = handle
                    .mcp_url()
                    .map(|u| u.to_string())
                    .unwrap_or(fallback_url);
                (url, Some(handle))
            }
            Err(e) => {
                return IoCompletion {
                    task_id,
                    op_id,
                    result: IoResult::McpConnect(Err(format!("sandbox provision failed: {e}"))),
                };
            }
        };

        match McpSession::connect(&mcp_url).await {
            Ok(s) => IoCompletion {
                task_id,
                op_id,
                result: IoResult::McpConnectSuccess {
                    session: Arc::new(s),
                    sandbox_handle,
                },
            },
            Err(e) => IoCompletion {
                task_id,
                op_id,
                result: IoResult::McpConnect(Err(e.to_string())),
            },
        }
    })
}

fn list_tools(scheduler: &Scheduler, task_id: String, op_id: OpId) -> IoFuture {
    let sessions = scheduler
        .pool_sessions(&task_id)
        .expect("mcp pool present before ListTools");
    Box::pin(async move {
        let mut all_tools: Vec<McpTool> = Vec::new();
        let mut routing: HashMap<String, usize> = HashMap::new();
        for (idx, session) in sessions.iter().enumerate() {
            match session.list_tools().await {
                Ok(tools) => {
                    for tool in tools {
                        if let Some(prev) = routing.get(&tool.name) {
                            // Earlier session wins (primary first).
                            warn!(
                                %task_id, op_id, name = %tool.name,
                                kept_idx = prev, dropped_idx = idx,
                                "tool name collision across MCP sessions; dropping later"
                            );
                            continue;
                        }
                        routing.insert(tool.name.clone(), idx);
                        all_tools.push(tool);
                    }
                }
                Err(e) => {
                    return IoCompletion {
                        task_id,
                        op_id,
                        result: IoResult::ListTools(Err(format!("session {idx}: {e}"))),
                    };
                }
            }
        }
        IoCompletion {
            task_id,
            op_id,
            result: IoResult::ListToolsSuccess {
                tools: all_tools,
                routing,
            },
        }
    })
}

fn model_call(scheduler: &Scheduler, task_id: String, op_id: OpId) -> IoFuture {
    let (owned_req, model_name, backend_name) = build_model_request(scheduler, &task_id);
    let provider = match scheduler.backend(&backend_name) {
        Some(entry) => entry.provider.clone(),
        None => {
            // Unknown backend — surface as a model-call failure so the
            // task transitions to Failed via the existing path.
            let msg = format!("unknown backend `{backend_name}`");
            return Box::pin(async move {
                IoCompletion {
                    task_id,
                    op_id,
                    result: IoResult::ModelCall(Err(msg)),
                }
            });
        }
    };
    Box::pin(async move {
        debug!(%task_id, op_id, backend = %backend_name, model = %model_name, "dispatching model call");
        let req = ModelRequest {
            model: &owned_req.model,
            max_tokens: owned_req.max_tokens,
            system_prompt: &owned_req.system_prompt,
            tools: &owned_req.tools,
            messages: &owned_req.messages,
            cache_breakpoints: &owned_req.cache_breakpoints,
        };
        match provider.create_message(&req).await {
            Ok(resp) => IoCompletion {
                task_id,
                op_id,
                result: IoResult::ModelCall(Ok(resp)),
            },
            Err(e) => IoCompletion {
                task_id,
                op_id,
                result: IoResult::ModelCall(Err(e.to_string())),
            },
        }
    })
}

fn tool_call(
    scheduler: &Scheduler,
    task_id: String,
    op_id: OpId,
    tool_use_id: String,
    name: String,
    input: serde_json::Value,
) -> IoFuture {
    // Route to whichever pool session contributed this tool. Falls back to the
    // primary if routing wasn't filled in (shouldn't happen because ToolCall
    // always follows a successful ListTools, but we'd rather try than panic).
    let mcp = scheduler
        .route_tool(&task_id, &name)
        .expect("mcp pool present before ToolCall");
    Box::pin(async move {
        let result = match mcp.invoke(&name, input).await {
            Ok(mut stream) => {
                let mut last = None;
                while let Some(event) = stream.next().await {
                    match event {
                        crate::mcp::ToolEvent::Completed(r) => last = Some(r),
                    }
                }
                match last {
                    Some(r) => Ok(r),
                    None => Err("no Completed event in tool stream".to_string()),
                }
            }
            Err(e) => Err(e.to_string()),
        };
        IoCompletion {
            task_id,
            op_id,
            result: IoResult::ToolCall {
                tool_use_id,
                result,
            },
        }
    })
}

/// Owned-by-value model request payload — the dispatched future outlives the
/// scheduler borrow that produced it, so it can't hold references back.
struct OwnedModelRequest {
    model: String,
    max_tokens: u32,
    system_prompt: String,
    tools: Vec<ToolSpec>,
    messages: Vec<whisper_agent_protocol::Message>,
    cache_breakpoints: Vec<CacheBreakpoint>,
}

fn build_model_request(
    scheduler: &Scheduler,
    task_id: &str,
) -> (OwnedModelRequest, String, String) {
    let task = scheduler.task(task_id).expect("task exists");
    let tools: Vec<ToolSpec> = scheduler
        .tool_descriptors(task_id)
        .map(|tools| tools.iter().map(mcp_tool_to_spec).collect())
        .unwrap_or_default();
    let messages = task.conversation.messages().to_vec();
    let backend_name = scheduler.resolve_backend_name(task).to_string();
    // Empty task.config.model → consult the backend's default_model. If that's
    // also None (common for single-model local endpoints), pass empty through.
    let model = if task.config.model.is_empty() {
        scheduler
            .backend(&backend_name)
            .and_then(|e| e.default_model.clone())
            .unwrap_or_default()
    } else {
        task.config.model.clone()
    };
    let max_tokens = task.config.max_tokens;
    let cache_breakpoints = default_cache_policy(&messages, 2);
    (
        OwnedModelRequest {
            model: model.clone(),
            max_tokens,
            system_prompt: task.config.system_prompt.clone(),
            tools,
            messages,
            cache_breakpoints,
        },
        model,
        backend_name,
    )
}

fn mcp_tool_to_spec(t: &McpTool) -> ToolSpec {
    ToolSpec {
        name: t.name.clone(),
        description: t.description.clone(),
        input_schema: t.input_schema.clone(),
    }
}
