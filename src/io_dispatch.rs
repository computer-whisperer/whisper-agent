//! I/O future construction for the scheduler.
//!
//! Two flavors of completion ride the same `FuturesUnordered`:
//!
//! - [`SchedulerCompletion::Io`] — per-thread state-machine ops (model
//!   call, tool call). Built from a [`task::IoRequest`] the thread itself
//!   requested via `step()`. Routed back through `Thread::apply_io_result`.
//! - [`SchedulerCompletion::Provision`] — per-thread host-env
//!   provisioning (env + MCP connect + initial `tools/list`).
//!   Dispatched eagerly by the scheduler at thread-creation time for
//!   threads that have a host_env binding. Routed via
//!   `Scheduler::apply_provision_completion` which updates the
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
use crate::resources::HostEnvId;
use crate::sandbox::HostEnvHandle;
use crate::scheduler::Scheduler;
use crate::thread::{IoRequest, IoResult, OpId};

/// Completion message delivered by every per-thread I/O future the
/// scheduler dispatches via `step_until_blocked`.
pub(crate) struct IoCompletion {
    pub(crate) thread_id: String,
    pub(crate) op_id: OpId,
    pub(crate) result: IoResult,
}

/// Completion message delivered by every host-env provisioning future.
pub(crate) struct ProvisionCompletion {
    pub(crate) thread_id: String,
    pub(crate) result: ProvisionResult,
}

pub(crate) enum ProvisionResult {
    /// Host env + MCP connect + initial `tools/list` all completed.
    /// Both the host-env entry and the host-env MCP entry get
    /// attached; dedup races drop redundant handles.
    HostEnvMcpReady {
        host_env_id: HostEnvId,
        host_env_handle: Box<dyn HostEnvHandle>,
        mcp_url: String,
        mcp_session: Arc<McpSession>,
        tools: Vec<crate::mcp::ToolDescriptor>,
    },
    /// Provisioning failed somewhere in the chain (host env, MCP
    /// connect, or initial `tools/list`). The carried `phase` tells
    /// the scheduler which Errored state to mark on the affected
    /// registry entries.
    HostEnvMcpFailed {
        phase: ProvisionPhase,
        message: String,
        host_env_id: HostEnvId,
    },
}

/// Which phase of `provision_host_env_mcp` failed — drives where the
/// scheduler marks Errored.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProvisionPhase {
    HostEnv,
    McpConnect,
    ListTools,
}

impl ProvisionPhase {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ProvisionPhase::HostEnv => "host_env",
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

/// Build a future that provisions a thread's host env (env + MCP
/// connect + initial `tools/list`) and yields a
/// [`SchedulerCompletion::Provision`]. Only called for threads that
/// have a host_env binding — threads bound only to shared MCPs don't
/// need host-env provisioning.
pub(crate) fn provision_host_env_mcp(
    scheduler: &Scheduler,
    thread_id: String,
) -> SchedulerFuture {
    // Snapshot the host-env decision while we have the sync borrow.
    // The entry must exist — ensure_host_env_provisioning pre-registered
    // it before dispatching. If the entry is already Ready (dedup hit),
    // skip provision and reuse the cached URL.
    let host_env_action = match scheduler.host_env_for_thread(&thread_id) {
        Some(e) if e.state.is_ready() => {
            let mcp_url = e.mcp_url.clone().unwrap_or_default();
            HostEnvAction::Skip {
                mcp_url,
                host_env_id: e.id.clone(),
            }
        }
        Some(e) => match scheduler.host_env_provider(&e.provider) {
            Some(provider) => HostEnvAction::Provision {
                provider: provider.clone(),
                spec: e.spec.clone(),
                host_env_id: e.id.clone(),
            },
            None => HostEnvAction::UnknownProvider {
                host_env_id: e.id.clone(),
                provider: e.provider.clone(),
            },
        },
        None => {
            // Shouldn't happen: only called when
            // ensure_host_env_provisioning saw a binding.
            return Box::pin(async move {
                SchedulerCompletion::Provision(ProvisionCompletion {
                    thread_id,
                    result: ProvisionResult::HostEnvMcpFailed {
                        phase: ProvisionPhase::HostEnv,
                        message: "thread has no host_env binding".into(),
                        host_env_id: HostEnvId(String::new()),
                    },
                })
            });
        }
    };

    Box::pin(async move {
        // --- Phase 1: host env ---
        let (mcp_url, host_env_handle, host_env_id) = match host_env_action {
            HostEnvAction::Skip {
                mcp_url,
                host_env_id,
            } => {
                // Dedup hit — another thread already completed the
                // provision. Connect to the cached URL directly; we
                // don't carry a handle (the dedup winner owns it).
                // This path is a no-op from the registry's
                // perspective: the entry is already Ready.
                (mcp_url, None, host_env_id)
            }
            HostEnvAction::UnknownProvider {
                host_env_id,
                provider,
            } => {
                return SchedulerCompletion::Provision(ProvisionCompletion {
                    thread_id,
                    result: ProvisionResult::HostEnvMcpFailed {
                        phase: ProvisionPhase::HostEnv,
                        message: format!(
                            "host env provider `{provider}` is not in the server catalog"
                        ),
                        host_env_id,
                    },
                });
            }
            HostEnvAction::Provision {
                provider,
                spec,
                host_env_id,
            } => match provider.provision(&thread_id, &spec).await {
                Ok(handle) => {
                    let url = handle.mcp_url().to_string();
                    (url, Some(handle), host_env_id)
                }
                Err(e) => {
                    return SchedulerCompletion::Provision(ProvisionCompletion {
                        thread_id,
                        result: ProvisionResult::HostEnvMcpFailed {
                            phase: ProvisionPhase::HostEnv,
                            message: format!("host env provision failed: {e}"),
                            host_env_id,
                        },
                    });
                }
            },
        };

        // --- Phase 2: MCP connect ---
        let session = match McpSession::connect(&mcp_url).await {
            Ok(s) => Arc::new(s),
            Err(e) => {
                return SchedulerCompletion::Provision(ProvisionCompletion {
                    thread_id,
                    result: ProvisionResult::HostEnvMcpFailed {
                        phase: ProvisionPhase::McpConnect,
                        message: format!("mcp connect failed: {e}"),
                        host_env_id,
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
                    result: ProvisionResult::HostEnvMcpFailed {
                        phase: ProvisionPhase::ListTools,
                        message: format!("list_tools failed: {e}"),
                        host_env_id,
                    },
                });
            }
        };

        // Dedup winners carry a handle; dedup losers don't. The
        // scheduler drops a None handle cleanly via the
        // AlreadyCompleted outcome.
        let handle = host_env_handle.unwrap_or_else(|| Box::new(DedupNoopHandle { mcp_url: mcp_url.clone() }));

        SchedulerCompletion::Provision(ProvisionCompletion {
            thread_id,
            result: ProvisionResult::HostEnvMcpReady {
                host_env_id,
                host_env_handle: handle,
                mcp_url,
                mcp_session: session,
                tools,
            },
        })
    })
}

/// Placeholder handle used when a dedup loser doesn't own the real
/// handle but still needs to fit the `HostEnvHandle` type expected by
/// the completion payload. `complete_host_env_provisioning` observes
/// the `AlreadyCompleted` outcome and drops this handle without ever
/// calling `teardown`; `mcp_url` is present only so `HostEnvHandle`'s
/// contract holds.
struct DedupNoopHandle {
    mcp_url: String,
}

impl crate::sandbox::HostEnvHandle for DedupNoopHandle {
    fn mcp_url(&self) -> &str {
        &self.mcp_url
    }
    fn teardown(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), crate::sandbox::HostEnvError>> + Send + '_>>
    {
        Box::pin(async { Ok(()) })
    }
}

/// Decision the dispatcher made about the bound host env before
/// crossing into the async block.
enum HostEnvAction {
    /// Already Ready (dedup hit) — connect MCP at `mcp_url` directly,
    /// no provision call.
    Skip {
        mcp_url: String,
        host_env_id: HostEnvId,
    },
    /// Needs provisioning; call into the resolved provider with this spec.
    Provision {
        provider: Arc<dyn crate::sandbox::HostEnvProvider>,
        spec: whisper_agent_protocol::HostEnvSpec,
        host_env_id: HostEnvId,
    },
    /// The named provider isn't in the catalog (catalog edited after
    /// the entry was registered, or pod config never validated). The
    /// async block fails fast so the thread parks in `Errored`.
    UnknownProvider {
        host_env_id: HostEnvId,
        provider: String,
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
    // Route to whichever bound host advertises this tool. If nothing
    // does, the thread's model is calling a tool we never exposed —
    // surface it as a tool-call error rather than panicking.
    let Some(mcp) = scheduler.route_tool(&thread_id, &name) else {
        return Box::pin(async move {
            SchedulerCompletion::Io(IoCompletion {
                thread_id,
                op_id,
                result: IoResult::ToolCall {
                    tool_use_id,
                    result: Err(format!(
                        "no bound MCP host advertises tool `{name}`",
                    )),
                },
            })
        });
    };
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
