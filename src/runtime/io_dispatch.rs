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

use crate::pod::resources::HostEnvId;
use crate::providers::model::{CacheBreakpoint, ModelRequest, ToolSpec, default_cache_policy};
use crate::runtime::scheduler::Scheduler;
use crate::runtime::thread::{IoRequest, IoResult, OpId};
use crate::tools::mcp::{McpSession, ToolDescriptor as McpTool};
use crate::tools::sandbox::HostEnvHandle;

/// Completion message delivered by every per-thread I/O future the
/// scheduler dispatches via `step_until_blocked`.
///
/// `pod_update`, when set, is a side effect produced by a builtin tool
/// (pod_write_file / pod_edit_file) that the scheduler applies to
/// in-memory pod state after handing the tool result to the task —
/// keeping the broadcast and on-disk write atomic from subscribers'
/// perspective.
///
/// `scheduler_command`, when set, is a runtime action the tool wants
/// the scheduler to perform (pause / run a behavior). Applied in the
/// same completion step as `pod_update`, with scheduler access.
pub(crate) struct IoCompletion {
    pub(crate) thread_id: String,
    pub(crate) op_id: OpId,
    pub(crate) result: IoResult,
    pub(crate) pod_update: Option<crate::tools::builtin_tools::PodUpdate>,
    pub(crate) scheduler_command: Option<crate::tools::builtin_tools::SchedulerCommand>,
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
        mcp_token: Option<String>,
        mcp_session: Arc<McpSession>,
        tools: Vec<crate::tools::mcp::ToolDescriptor>,
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

pub(crate) type SchedulerFuture = Pin<Box<dyn Future<Output = SchedulerCompletion> + Send>>;

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
pub(crate) fn provision_host_env_mcp(scheduler: &Scheduler, thread_id: String) -> SchedulerFuture {
    // Snapshot the host-env decision while we have the sync borrow.
    // The entry must exist — ensure_host_env_provisioning pre-registered
    // it before dispatching. If the entry is already Ready (dedup hit),
    // skip provision and reuse the cached URL.
    let host_env_action = match scheduler.host_env_for_thread(&thread_id) {
        Some(e) if e.state.is_ready() => {
            let mcp_url = e.mcp_url.clone().unwrap_or_default();
            let mcp_token = e.mcp_token.clone();
            HostEnvAction::Skip {
                mcp_url,
                mcp_token,
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
        let (mcp_url, mcp_token, host_env_handle, host_env_id) = match host_env_action {
            HostEnvAction::Skip {
                mcp_url,
                mcp_token,
                host_env_id,
            } => {
                // Dedup hit — another thread already completed the
                // provision. Connect to the cached URL directly; we
                // don't carry a handle (the dedup winner owns it).
                // This path is a no-op from the registry's
                // perspective: the entry is already Ready.
                (mcp_url, mcp_token, None, host_env_id)
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
                    let tok = handle.mcp_token().map(|s| s.to_string());
                    (url, tok, Some(handle), host_env_id)
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
        let session = match McpSession::connect(&mcp_url, mcp_token.clone()).await {
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
        let handle = host_env_handle.unwrap_or_else(|| {
            Box::new(DedupNoopHandle {
                mcp_url: mcp_url.clone(),
            })
        });

        SchedulerCompletion::Provision(ProvisionCompletion {
            thread_id,
            result: ProvisionResult::HostEnvMcpReady {
                host_env_id,
                host_env_handle: handle,
                mcp_url,
                mcp_token,
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

impl crate::tools::sandbox::HostEnvHandle for DedupNoopHandle {
    fn mcp_url(&self) -> &str {
        &self.mcp_url
    }
    fn mcp_token(&self) -> Option<&str> {
        // The dedup loser is handed the registry-cached token by the
        // scheduler at completion time; the noop handle itself never
        // holds one (and is dropped unused).
        None
    }
    fn teardown(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), crate::tools::sandbox::HostEnvError>> + Send + '_>>
    {
        Box::pin(async { Ok(()) })
    }
}

/// Decision the dispatcher made about the bound host env before
/// crossing into the async block.
enum HostEnvAction {
    /// Already Ready (dedup hit) — connect MCP at `mcp_url` directly,
    /// no provision call. `mcp_token` is the same per-sandbox bearer
    /// the original provisioner received; a dedup hit needs it to
    /// authenticate its own MCP session against the shared host.
    Skip {
        mcp_url: String,
        mcp_token: Option<String>,
        host_env_id: HostEnvId,
    },
    /// Needs provisioning; call into the resolved provider with this spec.
    Provision {
        provider: Arc<dyn crate::tools::sandbox::HostEnvProvider>,
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
                    pod_update: None,
                    scheduler_command: None,
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
        let result =
            match call_with_rate_limit_retry(provider.as_ref(), &req, &thread_id, op_id).await {
                Ok(resp) => IoResult::ModelCall(Ok(resp)),
                Err(e) => IoResult::ModelCall(Err(e.to_string())),
            };
        SchedulerCompletion::Io(IoCompletion {
            thread_id,
            op_id,
            result,
            pod_update: None,
            scheduler_command: None,
        })
    })
}

/// Upper bound on how long we're willing to sleep through a single
/// rate-limit signal before giving up. Short Code-Assist 429s (say ~6s)
/// flow through transparently; long quota-reset waits surface as
/// failures so the user isn't left waiting on a minutes-long sleep.
const MAX_RATE_LIMIT_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

async fn call_with_rate_limit_retry(
    provider: &dyn crate::providers::model::ModelProvider,
    req: &ModelRequest<'_>,
    thread_id: &str,
    op_id: OpId,
) -> Result<crate::providers::model::ModelResponse, crate::providers::model::ModelError> {
    match provider.create_message(req).await {
        Ok(resp) => Ok(resp),
        Err(crate::providers::model::ModelError::RateLimited { retry_after, body }) => {
            if retry_after > MAX_RATE_LIMIT_WAIT {
                return Err(crate::providers::model::ModelError::RateLimited { retry_after, body });
            }
            tracing::warn!(
                %thread_id,
                op_id,
                wait_ms = retry_after.as_millis() as u64,
                "model backend rate-limited; waiting before single retry"
            );
            tokio::time::sleep(retry_after).await;
            // One retry; any further rate-limit bubbles up as a terminal
            // failure so the thread parks rather than spinning indefinitely.
            provider.create_message(req).await
        }
        Err(other) => Err(other),
    }
}

fn tool_call(
    scheduler: &Scheduler,
    thread_id: String,
    op_id: OpId,
    tool_use_id: String,
    name: String,
    input: serde_json::Value,
) -> SchedulerFuture {
    use crate::runtime::scheduler::ToolRoute;
    match scheduler.route_tool(&thread_id, &name) {
        Some(ToolRoute::Builtin { pod_id }) => {
            // Snapshot pod dir + config now while we have the sync
            // borrow. If the pod has vanished between route_tool and
            // here, surface that as a tool error (shouldn't happen in
            // practice — pods don't drop mid-turn).
            let snapshot = match scheduler.pod_snapshot(&pod_id) {
                Some(s) => s,
                None => {
                    return Box::pin(async move {
                        SchedulerCompletion::Io(IoCompletion {
                            thread_id,
                            op_id,
                            result: IoResult::ToolCall {
                                tool_use_id,
                                result: Err(format!("unknown pod `{pod_id}`")),
                            },
                            pod_update: None,
                            scheduler_command: None,
                        })
                    });
                }
            };
            Box::pin(async move {
                let outcome = crate::tools::builtin_tools::dispatch(
                    snapshot.pod_dir,
                    snapshot.config,
                    snapshot.behavior_ids,
                    &name,
                    input,
                )
                .await;
                let pod_update = outcome.pod_update;
                let scheduler_command = outcome.scheduler_command;
                let result = if outcome.result.is_error {
                    // Surface tool-level errors as Err in the task-facing
                    // IoResult — the approval-audit path uses the Err
                    // variant to decide how to record the turn. Builtin
                    // tool errors already carry a text block we can
                    // pass through.
                    Err(join_text(&outcome.result.content))
                } else {
                    Ok(outcome.result)
                };
                SchedulerCompletion::Io(IoCompletion {
                    thread_id,
                    op_id,
                    result: IoResult::ToolCall {
                        tool_use_id,
                        result,
                    },
                    pod_update,
                    scheduler_command,
                })
            })
        }
        Some(ToolRoute::Mcp(mcp)) => Box::pin(async move {
            let result = match mcp.invoke(&name, input).await {
                Ok(mut stream) => {
                    let mut last = None;
                    while let Some(event) = stream.next().await {
                        match event {
                            crate::tools::mcp::ToolEvent::Completed(r) => last = Some(r),
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
                pod_update: None,
                scheduler_command: None,
            })
        }),
        None => Box::pin(async move {
            SchedulerCompletion::Io(IoCompletion {
                thread_id,
                op_id,
                result: IoResult::ToolCall {
                    tool_use_id,
                    result: Err(format!("no bound host advertises tool `{name}`")),
                },
                pod_update: None,
                scheduler_command: None,
            })
        }),
    }
}

fn join_text(blocks: &[crate::tools::mcp::McpContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        match b {
            crate::tools::mcp::McpContentBlock::Text { text } => out.push_str(text),
        }
    }
    out
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
