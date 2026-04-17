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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use tracing::debug;

use crate::mcp::{McpSession, ToolDescriptor as McpTool};
use crate::model::{CacheBreakpoint, ModelRequest, ToolSpec, default_cache_policy};
use crate::scheduler::Scheduler;
use crate::thread::{IoRequest, IoResult, OpId};

/// Completion message delivered by every I/O future the scheduler dispatches.
pub(crate) struct IoCompletion {
    pub(crate) thread_id: String,
    pub(crate) op_id: OpId,
    pub(crate) result: IoResult,
}

pub(crate) type IoFuture = Pin<Box<dyn Future<Output = IoCompletion> + Send>>;

/// Build a future that executes one I/O op and yields an [`IoCompletion`].
pub(crate) fn build_io_future(scheduler: &Scheduler, thread_id: String, req: IoRequest) -> IoFuture {
    match req {
        IoRequest::McpConnect { op_id } => mcp_connect(scheduler, thread_id, op_id),
        IoRequest::ListTools { op_id } => list_tools(scheduler, thread_id, op_id),
        IoRequest::ModelCall { op_id } => model_call(scheduler, thread_id, op_id),
        IoRequest::ToolCall {
            op_id,
            tool_use_id,
            name,
            input,
        } => tool_call(scheduler, thread_id, op_id, tool_use_id, name, input),
    }
}

fn mcp_connect(scheduler: &Scheduler, thread_id: String, op_id: OpId) -> IoFuture {
    // Phase 3d.i: the sandbox spec lives on the registry entry that
    // `pre_register_sandbox` inserted at create_task time. Read it back via
    // `bindings.sandbox`; if the entry is already Ready (another thread
    // with the matching spec finished provisioning before we got here),
    // skip provision and reuse the cached URL.
    let fallback_url = scheduler.default_mcp_url().to_string();
    let provider = scheduler.sandbox_provider().clone();
    let action = match scheduler.sandbox_for_thread(&thread_id) {
        Some(e) if e.state.is_ready() => SandboxAction::Skip {
            mcp_url: e.mcp_url.clone().unwrap_or_else(|| fallback_url.clone()),
        },
        Some(e) => SandboxAction::Provision {
            spec: e.spec.clone(),
        },
        None => SandboxAction::Skip {
            mcp_url: fallback_url.clone(),
        },
    };

    Box::pin(async move {
        let (mcp_url, sandbox_handle) = match action {
            SandboxAction::Skip { mcp_url } => (mcp_url, None),
            SandboxAction::Provision { spec } => {
                match provider.provision(&thread_id, &spec).await {
                    Ok(handle) => {
                        let url = handle
                            .mcp_url()
                            .map(|u| u.to_string())
                            .unwrap_or(fallback_url);
                        (url, Some(handle))
                    }
                    Err(e) => {
                        return IoCompletion {
                            thread_id,
                            op_id,
                            result: IoResult::McpConnect(Err(format!(
                                "sandbox provision failed: {e}"
                            ))),
                        };
                    }
                }
            }
        };

        match McpSession::connect(&mcp_url).await {
            Ok(s) => IoCompletion {
                thread_id,
                op_id,
                result: IoResult::McpConnectSuccess {
                    session: Arc::new(s),
                    sandbox_handle,
                },
            },
            Err(e) => IoCompletion {
                thread_id,
                op_id,
                result: IoResult::McpConnect(Err(e.to_string())),
            },
        }
    })
}

/// Decision the dispatcher made about the bound sandbox before crossing
/// into the async block. Computed against the registry while we still hold
/// the synchronous borrow; the future itself only needs the resolved
/// pieces.
enum SandboxAction {
    /// Sandbox is already Ready (or no sandbox is bound) — connect MCP at
    /// `mcp_url` directly, no provision call.
    Skip { mcp_url: String },
    /// Sandbox needs provisioning; call into the provider with this spec.
    Provision {
        spec: whisper_agent_protocol::SandboxSpec,
    },
}

fn list_tools(scheduler: &Scheduler, thread_id: String, op_id: OpId) -> IoFuture {
    // Phase 3c: only the primary needs a per-thread `tools/list` — shared
    // hosts published their catalogs at startup. The accessor returns
    // sessions in pool order (primary first); we just need the head.
    let sessions = scheduler
        .pool_sessions(&thread_id)
        .expect("primary MCP host present before ListTools");
    let primary = sessions.into_iter().next().expect("primary present");
    Box::pin(async move {
        match primary.list_tools().await {
            Ok(tools) => IoCompletion {
                thread_id,
                op_id,
                result: IoResult::ListToolsSuccess { tools },
            },
            Err(e) => IoCompletion {
                thread_id,
                op_id,
                result: IoResult::ListTools(Err(e.to_string())),
            },
        }
    })
}

fn model_call(scheduler: &Scheduler, thread_id: String, op_id: OpId) -> IoFuture {
    let (owned_req, model_name, backend_name) = build_model_request(scheduler, &thread_id);
    let provider = match scheduler.backend(&backend_name) {
        Some(entry) => entry.provider.clone(),
        None => {
            // Unknown backend — surface as a model-call failure so the
            // task transitions to Failed via the existing path.
            let msg = format!("unknown backend `{backend_name}`");
            return Box::pin(async move {
                IoCompletion {
                    thread_id,
                    op_id,
                    result: IoResult::ModelCall(Err(msg)),
                }
            });
        }
    };
    Box::pin(async move {
        debug!(%thread_id, op_id, backend = %backend_name, model = %model_name, "dispatching model call");
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
                thread_id,
                op_id,
                result: IoResult::ModelCall(Ok(resp)),
            },
            Err(e) => IoCompletion {
                thread_id,
                op_id,
                result: IoResult::ModelCall(Err(e.to_string())),
            },
        }
    })
}

fn tool_call(
    scheduler: &Scheduler,
    thread_id: String,
    op_id: OpId,
    tool_use_id: String,
    name: String,
    input: serde_json::Value,
) -> IoFuture {
    // Route to whichever pool session contributed this tool. Falls back to the
    // primary if routing wasn't filled in (shouldn't happen because ToolCall
    // always follows a successful ListTools, but we'd rather try than panic).
    let mcp = scheduler
        .route_tool(&thread_id, &name)
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
            thread_id,
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
    thread_id: &str,
) -> (OwnedModelRequest, String, String) {
    let task = scheduler.task(thread_id).expect("task exists");
    let tools: Vec<ToolSpec> = scheduler
        .tool_descriptors(thread_id)
        .iter()
        .map(mcp_tool_to_spec)
        .collect();
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
