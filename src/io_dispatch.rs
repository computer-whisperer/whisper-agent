//! I/O future construction for the scheduler.
//!
//! Two flavors of completion ride the same `FuturesUnordered`:
//!
//! - [`SchedulerCompletion::Io`] — per-thread state-machine ops (model
//!   call, tool call). Built from a [`task::IoRequest`] the thread itself
//!   requested via `step()`. Routed back through `Thread::apply_io_result`.
//! - [`SchedulerCompletion::Provision`] — per-thread resource provisioning
//!   (sandbox + primary MCP host + initial `tools/list`). Dispatched
//!   eagerly by the scheduler at thread-creation time, *not* at the
//!   thread's request — the resource registry owns this lifecycle now.
//!   Routed via `Scheduler::apply_provision_completion` which updates the
//!   registry and nudges any threads parked in `WaitingOnResources`.
//!
//! Per-IO-type knowledge (how to provision a sandbox, how to assemble a
//! model request) lives here so adding a new variant only touches this
//! module plus the consumer (thread state machine for `Io`, scheduler
//! resource handling for `Provision`).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use tracing::debug;

use crate::mcp::{McpSession, ToolDescriptor as McpTool};
use crate::model::{CacheBreakpoint, ModelRequest, ToolSpec, default_cache_policy};
use crate::resources::SandboxId;
use crate::sandbox::SandboxHandle;
use crate::scheduler::Scheduler;
use crate::thread::{IoRequest, IoResult, OpId};

/// Completion message delivered by every per-thread I/O future the
/// scheduler dispatches via `step_until_blocked`.
pub(crate) struct IoCompletion {
    pub(crate) thread_id: String,
    pub(crate) op_id: OpId,
    pub(crate) result: IoResult,
}

/// Completion message delivered by every resource provisioning future.
/// Carries the requesting thread (so the scheduler knows whose primary
/// MCP entry to attach the session to) plus the resource-side payload.
pub(crate) struct ProvisionCompletion {
    pub(crate) thread_id: String,
    pub(crate) result: ProvisionResult,
}

pub(crate) enum ProvisionResult {
    /// Sandbox + MCP connect + initial `tools/list` all completed. The
    /// scheduler attaches everything to the registry: `sandbox_handle` to
    /// the sandbox entry (or drops it if dedup already wrote one), and
    /// `mcp_session` + `tools` to the per-thread primary MCP entry.
    PrimaryMcpReady {
        /// Bound sandbox id, when one was bound. None for threads with no
        /// sandbox (`SandboxSpec::None` still pre-registers a sb-… entry,
        /// so this is None only if the binding itself was None).
        sandbox_id: Option<SandboxId>,
        /// Provisioned sandbox handle. None on dedup hit (someone else
        /// completed the sandbox before us) or when the spec produces no
        /// runtime artifact (e.g. `SandboxSpec::None`).
        sandbox_handle: Option<Box<dyn SandboxHandle>>,
        mcp_url: String,
        mcp_session: Arc<McpSession>,
        tools: Vec<crate::mcp::ToolDescriptor>,
    },
    /// Provisioning failed somewhere in the chain (sandbox, MCP connect,
    /// or initial `tools/list`). The carried `phase` tells the scheduler
    /// which Errored state to mark on the affected registry entries.
    PrimaryMcpFailed {
        phase: ProvisionPhase,
        message: String,
        sandbox_id: Option<SandboxId>,
    },
}

/// Which phase of `provision_primary_mcp` failed — drives where the
/// scheduler marks Errored.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProvisionPhase {
    Sandbox,
    McpConnect,
    ListTools,
}

impl ProvisionPhase {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ProvisionPhase::Sandbox => "sandbox",
            ProvisionPhase::McpConnect => "mcp_connect",
            ProvisionPhase::ListTools => "list_tools",
        }
    }
}

/// Unified completion type the scheduler's `FuturesUnordered` carries.
pub(crate) enum SchedulerCompletion {
    Io(IoCompletion),
    Provision(ProvisionCompletion),
}

pub(crate) type SchedulerFuture =
    Pin<Box<dyn Future<Output = SchedulerCompletion> + Send>>;

/// Build a future that executes one per-thread I/O op and yields a
/// [`SchedulerCompletion::Io`].
pub(crate) fn build_io_future(
    scheduler: &Scheduler,
    thread_id: String,
    req: IoRequest,
) -> SchedulerFuture {
    match req {
        IoRequest::ModelCall { op_id } => model_call(scheduler, thread_id, op_id),
        IoRequest::ToolCall {
            op_id,
            tool_use_id,
            name,
            input,
        } => tool_call(scheduler, thread_id, op_id, tool_use_id, name, input),
    }
}

/// Build a future that provisions a thread's primary MCP host (sandbox +
/// MCP connect + initial `tools/list`) and yields a
/// [`SchedulerCompletion::Provision`]. Called by the scheduler at
/// thread-creation time, not from the thread's state machine.
pub(crate) fn provision_primary_mcp(
    scheduler: &Scheduler,
    thread_id: String,
) -> SchedulerFuture {
    let fallback_url = scheduler.default_mcp_url().to_string();
    let provider = scheduler.sandbox_provider().clone();

    // Snapshot the sandbox decision while we have the sync borrow. If the
    // entry is already Ready (a sibling thread provisioned the same spec
    // before we got here), skip provision and reuse the cached URL.
    let sandbox_action = match scheduler.sandbox_for_thread(&thread_id) {
        Some(e) if e.state.is_ready() => SandboxAction::Skip {
            mcp_url: e.mcp_url.clone().unwrap_or_else(|| fallback_url.clone()),
            sandbox_id: Some(e.id.clone()),
        },
        Some(e) => SandboxAction::Provision {
            spec: e.spec.clone(),
            sandbox_id: e.id.clone(),
        },
        None => SandboxAction::Skip {
            mcp_url: fallback_url.clone(),
            sandbox_id: None,
        },
    };

    Box::pin(async move {
        // --- Phase 1: sandbox ---
        let (mcp_url, sandbox_handle, sandbox_id) = match sandbox_action {
            SandboxAction::Skip {
                mcp_url,
                sandbox_id,
            } => (mcp_url, None, sandbox_id),
            SandboxAction::Provision { spec, sandbox_id } => {
                match provider.provision(&thread_id, &spec).await {
                    Ok(handle) => {
                        let url = handle
                            .mcp_url()
                            .map(|u| u.to_string())
                            .unwrap_or_else(|| fallback_url.clone());
                        (url, Some(handle), Some(sandbox_id))
                    }
                    Err(e) => {
                        return SchedulerCompletion::Provision(ProvisionCompletion {
                            thread_id,
                            result: ProvisionResult::PrimaryMcpFailed {
                                phase: ProvisionPhase::Sandbox,
                                message: format!("sandbox provision failed: {e}"),
                                sandbox_id: Some(sandbox_id),
                            },
                        });
                    }
                }
            }
        };

        // --- Phase 2: MCP connect ---
        let session = match McpSession::connect(&mcp_url).await {
            Ok(s) => Arc::new(s),
            Err(e) => {
                return SchedulerCompletion::Provision(ProvisionCompletion {
                    thread_id,
                    result: ProvisionResult::PrimaryMcpFailed {
                        phase: ProvisionPhase::McpConnect,
                        message: format!("mcp connect failed: {e}"),
                        sandbox_id,
                    },
                });
            }
        };

        // --- Phase 3: initial tools/list ---
        let tools = match session.list_tools().await {
            Ok(t) => t,
            Err(e) => {
                return SchedulerCompletion::Provision(ProvisionCompletion {
                    thread_id,
                    result: ProvisionResult::PrimaryMcpFailed {
                        phase: ProvisionPhase::ListTools,
                        message: format!("list_tools failed: {e}"),
                        sandbox_id,
                    },
                });
            }
        };

        SchedulerCompletion::Provision(ProvisionCompletion {
            thread_id,
            result: ProvisionResult::PrimaryMcpReady {
                sandbox_id,
                sandbox_handle,
                mcp_url,
                mcp_session: session,
                tools,
            },
        })
    })
}

/// Decision the dispatcher made about the bound sandbox before crossing
/// into the async block.
enum SandboxAction {
    /// Sandbox is already Ready (or no sandbox is bound) — connect MCP at
    /// `mcp_url` directly, no provision call.
    Skip {
        mcp_url: String,
        sandbox_id: Option<SandboxId>,
    },
    /// Sandbox needs provisioning; call into the provider with this spec.
    Provision {
        spec: whisper_agent_protocol::SandboxSpec,
        sandbox_id: SandboxId,
    },
}

fn model_call(scheduler: &Scheduler, thread_id: String, op_id: OpId) -> SchedulerFuture {
    let (owned_req, model_name, backend_name) = build_model_request(scheduler, &thread_id);
    let provider = match scheduler.backend(&backend_name) {
        Some(entry) => entry.provider.clone(),
        None => {
            // Unknown backend — surface as a model-call failure so the
            // task transitions to Failed via the existing path.
            let msg = format!("unknown backend `{backend_name}`");
            return Box::pin(async move {
                SchedulerCompletion::Io(IoCompletion {
                    thread_id,
                    op_id,
                    result: IoResult::ModelCall(Err(msg)),
                })
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
        let result = match provider.create_message(&req).await {
            Ok(resp) => IoResult::ModelCall(Ok(resp)),
            Err(e) => IoResult::ModelCall(Err(e.to_string())),
        };
        SchedulerCompletion::Io(IoCompletion {
            thread_id,
            op_id,
            result,
        })
    })
}

fn tool_call(
    scheduler: &Scheduler,
    thread_id: String,
    op_id: OpId,
    tool_use_id: String,
    name: String,
    input: serde_json::Value,
) -> SchedulerFuture {
    // Route to whichever pool session contributed this tool. Falls back to the
    // primary if routing wasn't filled in (shouldn't happen because ToolCall
    // always follows a Ready primary, but we'd rather try than panic).
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
        SchedulerCompletion::Io(IoCompletion {
            thread_id,
            op_id,
            result: IoResult::ToolCall {
                tool_use_id,
                result,
            },
        })
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
