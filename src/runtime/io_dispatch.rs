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
use whisper_agent_protocol::ServerToClient;

use crate::pod::resources::HostEnvId;
use crate::providers::model::{
    CacheBreakpoint, ModelEvent, ModelRequest, ToolSpec, default_cache_policy,
};
use crate::runtime::scheduler::{Scheduler, StreamUpdate};
use crate::runtime::thread::{IoRequest, IoResult, OpId};
use crate::tools::mcp::McpSession;
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
    /// When set, the dispatched tool call hit a transport-level
    /// failure against a host-env MCP — the daemon session is gone.
    /// The scheduler marks the named host env + its MCP entry `Lost`
    /// *after* applying the tool result, so the thread sees a normal
    /// tool-call error and keeps running. A later provision attempt
    /// (same thread or another) re-provisions the env under the same
    /// `(provider, spec)` key.
    pub(crate) host_env_lost: Option<HostEnvLostSignal>,
}

#[derive(Clone)]
pub(crate) struct HostEnvLostSignal {
    pub(crate) mcp_host_id: crate::pod::resources::McpHostId,
    pub(crate) message: String,
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
    Probe(ProbeCompletion),
    SharedMcp(SharedMcpCompletion),
    OauthStart(OauthStartCompletion),
    OauthComplete(OauthCompleteCompletion),
    OauthRefresh(OauthRefreshCompletion),
}

/// Which Add/Update operation the completion applies to. The
/// scheduler uses this to decide the reply event and whether the
/// catalog write is an `insert` or `update`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SharedMcpOp {
    Add,
    Update,
}

/// Result delivered to the scheduler loop after a runtime
/// `AddSharedMcpHost` / `UpdateSharedMcpHost` finishes its connect +
/// tools/list. Validation already ran synchronously on the loop
/// before this future was dispatched — if we got here, the only
/// failure mode left is the network one, surfaced as
/// `Err(message)`.
pub(crate) struct SharedMcpCompletion {
    pub(crate) conn_id: crate::runtime::scheduler::ConnId,
    pub(crate) correlation_id: Option<String>,
    pub(crate) name: String,
    pub(crate) url: String,
    pub(crate) auth: crate::tools::shared_mcp_catalog::SharedMcpAuth,
    pub(crate) op: SharedMcpOp,
    pub(crate) result: Result<(Arc<McpSession>, Vec<crate::tools::mcp::ToolDescriptor>), String>,
}

// ---------- OAuth flow completions ----------
//
// The shared-MCP OAuth flow has two off-thread phases: (1) discover
// AS metadata + DCR + build the authorization URL (runs after the
// Add handler), and (2) exchange the authorization code + connect
// MCP with the new access token (runs after the /oauth/callback
// axum handler feeds a SchedulerMsg::OauthCallback into the inbox).
// Keeping each as its own completion type means the scheduler loop
// dispatches by match arm, not by a flag on a shared struct.

/// Result of the pre-authorization phase. On Ok, the scheduler
/// stashes the pending flow (keyed by `state`) and pushes
/// `SharedMcpOauthFlowStarted` to the caller; on Err, an `Error`.
pub(crate) struct OauthStartCompletion {
    pub(crate) conn_id: crate::runtime::scheduler::ConnId,
    pub(crate) correlation_id: Option<String>,
    pub(crate) name: String,
    pub(crate) url: String,
    pub(crate) redirect_uri: String,
    pub(crate) scope: Option<String>,
    pub(crate) result: Result<OauthStartData, String>,
}

/// Bundle of everything produced by the pre-authorization phase that
/// the post-authorization phase will need. Moved into the pending-
/// flow map by the scheduler.
pub(crate) struct OauthStartData {
    pub(crate) authorization_url: String,
    pub(crate) state: String,
    pub(crate) pkce_verifier: String,
    pub(crate) issuer: String,
    pub(crate) token_endpoint: String,
    pub(crate) registration_endpoint: Option<String>,
    pub(crate) client_id: String,
    pub(crate) client_secret: Option<String>,
    pub(crate) resource: String,
}

/// Result of the post-authorization phase: token exchange + MCP
/// connect + tools/list. On Ok, the scheduler writes the catalog
/// entry (OAuth2 variant) and inserts into the resource registry; on
/// Err, surfaces the failure to the originating webui connection.
pub(crate) struct OauthCompleteCompletion {
    pub(crate) conn_id: crate::runtime::scheduler::ConnId,
    pub(crate) correlation_id: Option<String>,
    pub(crate) name: String,
    pub(crate) url: String,
    /// Echoed so the scheduler can build the `SharedMcpAuth::Oauth2`
    /// entry without a second lookup into the pending-flow state
    /// (which has already been removed by this point).
    pub(crate) issuer: String,
    pub(crate) token_endpoint: String,
    pub(crate) registration_endpoint: Option<String>,
    pub(crate) client_id: String,
    pub(crate) client_secret: Option<String>,
    pub(crate) resource: String,
    pub(crate) scope: Option<String>,
    pub(crate) result: Result<OauthCompleteData, String>,
}

pub(crate) struct OauthCompleteData {
    pub(crate) session: Arc<McpSession>,
    pub(crate) tools: Vec<crate::tools::mcp::ToolDescriptor>,
    pub(crate) access_token: String,
    pub(crate) refresh_token: Option<String>,
    pub(crate) expires_at: Option<i64>,
    pub(crate) scope_granted: Option<String>,
}

/// Discovery + DCR + authz URL build. Runs off-thread so a slow AS
/// can't stall the scheduler loop. `redirect_base` comes from the
/// webui (its own origin); we append `/oauth/callback`.
pub(crate) fn build_oauth_start_future(
    conn_id: crate::runtime::scheduler::ConnId,
    correlation_id: Option<String>,
    name: String,
    url: String,
    scope: Option<String>,
    redirect_base: String,
) -> SchedulerFuture {
    let redirect_uri = format!("{}/oauth/callback", redirect_base.trim_end_matches('/'));
    Box::pin(async move {
        let http = match crate::mcp_oauth::http_client() {
            Ok(c) => c,
            Err(e) => {
                return SchedulerCompletion::OauthStart(OauthStartCompletion {
                    conn_id,
                    correlation_id,
                    name,
                    url,
                    redirect_uri,
                    scope,
                    result: Err(format!("init http client: {e}")),
                });
            }
        };

        let start = async {
            let discovered = crate::mcp_oauth::discover(&http, &url)
                .await
                .map_err(|e| format!("discover `{name}` ({url}): {e}"))?;
            let registration_endpoint =
                discovered.registration_endpoint.clone().ok_or_else(|| {
                    format!(
                        "AS {} does not support dynamic client registration \
                         (no registration_endpoint); pre-registered clients \
                         aren't supported yet",
                        discovered.issuer
                    )
                })?;
            // Honour caller-supplied scope, fall back to the
            // server-advertised list, or omit entirely.
            let effective_scope = scope.clone().or_else(|| {
                if discovered.scopes_supported.is_empty() {
                    None
                } else {
                    Some(discovered.scopes_supported.join(" "))
                }
            });
            let client = crate::mcp_oauth::dynamic_client_register(
                &http,
                &registration_endpoint,
                &redirect_uri,
                &format!("whisper-agent shared MCP: {name}"),
                effective_scope.as_deref(),
            )
            .await
            .map_err(|e| format!("DCR at {registration_endpoint}: {e}"))?;
            let pkce = crate::mcp_oauth::pkce_pair();
            let state = crate::mcp_oauth::random_state();
            let authorization_url = crate::mcp_oauth::build_authorization_url(
                &crate::mcp_oauth::AuthorizationUrlArgs {
                    authorization_endpoint: &discovered.authorization_endpoint,
                    client_id: &client.client_id,
                    redirect_uri: &redirect_uri,
                    state: &state,
                    code_challenge: &pkce.challenge,
                    scope: effective_scope.as_deref(),
                    resource: &discovered.resource,
                },
            )
            .map_err(|e| format!("build authz URL: {e}"))?;
            Ok::<OauthStartData, String>(OauthStartData {
                authorization_url,
                state,
                pkce_verifier: pkce.verifier,
                issuer: discovered.issuer,
                token_endpoint: discovered.token_endpoint,
                registration_endpoint: Some(registration_endpoint),
                client_id: client.client_id,
                client_secret: client.client_secret,
                resource: discovered.resource,
            })
        }
        .await;

        SchedulerCompletion::OauthStart(OauthStartCompletion {
            conn_id,
            correlation_id,
            name,
            url,
            redirect_uri,
            scope,
            result: start,
        })
    })
}

/// Post-authorization phase: exchange the code at the token
/// endpoint, open an MCP session with the fresh access token, run
/// `tools/list`. The scheduler's pending-flow entry has been
/// consumed into the arguments by now; this future can't look
/// anything up, it just does the I/O.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_oauth_complete_future(
    conn_id: crate::runtime::scheduler::ConnId,
    correlation_id: Option<String>,
    name: String,
    url: String,
    issuer: String,
    token_endpoint: String,
    registration_endpoint: Option<String>,
    client_id: String,
    client_secret: Option<String>,
    resource: String,
    scope: Option<String>,
    code: String,
    code_verifier: String,
    redirect_uri: String,
) -> SchedulerFuture {
    Box::pin(async move {
        let http = match crate::mcp_oauth::http_client() {
            Ok(c) => c,
            Err(e) => {
                return SchedulerCompletion::OauthComplete(OauthCompleteCompletion {
                    conn_id,
                    correlation_id,
                    name,
                    url,
                    issuer,
                    token_endpoint,
                    registration_endpoint,
                    client_id,
                    client_secret,
                    resource,
                    scope,
                    result: Err(format!("init http client: {e}")),
                });
            }
        };

        let complete = async {
            let token = crate::mcp_oauth::exchange_code(
                &http,
                &token_endpoint,
                &code,
                &redirect_uri,
                &client_id,
                client_secret.as_deref(),
                &code_verifier,
                &resource,
            )
            .await
            .map_err(|e| format!("token exchange: {e}"))?;
            let session = Arc::new(
                McpSession::connect(&url, Some(token.access_token.clone()))
                    .await
                    .map_err(|e| format!("mcp connect with new access token: {e}"))?,
            );
            let tools = session
                .list_tools()
                .await
                .map_err(|e| format!("tools/list: {e}"))?;
            let expires_at = token
                .expires_in
                .map(|secs| chrono::Utc::now().timestamp() + secs);
            Ok::<OauthCompleteData, String>(OauthCompleteData {
                session,
                tools,
                access_token: token.access_token,
                refresh_token: token.refresh_token,
                expires_at,
                scope_granted: token.scope,
            })
        }
        .await;

        SchedulerCompletion::OauthComplete(OauthCompleteCompletion {
            conn_id,
            correlation_id,
            name,
            url,
            issuer,
            token_endpoint,
            registration_endpoint,
            client_id,
            client_secret,
            resource,
            scope,
            result: complete,
        })
    })
}

// ---------- OAuth refresh ----------
//
// Runs off the scheduler thread because it calls the token endpoint
// (HTTP) and then re-handshakes with the MCP server. The scheduler
// feeds all refreshable entries into this pipeline on the 60s
// refresh tick; completions write the rotated tokens back to the
// catalog and swap the session in the registry.

pub(crate) struct OauthRefreshCompletion {
    /// Name of the catalog entry being refreshed. The completion
    /// handler re-looks up the entry to detect the rare
    /// "entry removed mid-flight" case.
    pub(crate) name: String,
    /// MCP URL captured at dispatch time. Used to re-connect after
    /// the token rotates. The catalog's URL doesn't change during a
    /// refresh so passing it through the future keeps the
    /// completion self-contained.
    pub(crate) url: String,
    pub(crate) result: Result<OauthRefreshData, String>,
}

pub(crate) struct OauthRefreshData {
    /// Fresh session authenticated with the rotated access_token.
    pub(crate) session: Arc<McpSession>,
    pub(crate) access_token: String,
    pub(crate) refresh_token: Option<String>,
    pub(crate) expires_at: Option<i64>,
    pub(crate) scope: Option<String>,
}

/// Arguments for `build_oauth_refresh_future`. The refresh future is
/// stateless — it does the HTTP exchange + an MCP re-connect and
/// returns the outcome; the scheduler then performs all catalog +
/// registry mutation on its own thread. Grouped into a struct
/// because the flat parameter list hit 8 positional args, which
/// clippy flags and which tend to rot as OAuth fields evolve.
pub(crate) struct OauthRefreshArgs {
    pub(crate) name: String,
    pub(crate) url: String,
    pub(crate) token_endpoint: String,
    pub(crate) client_id: String,
    pub(crate) client_secret: Option<String>,
    pub(crate) refresh_token: String,
    pub(crate) resource: String,
    pub(crate) scope: Option<String>,
}

pub(crate) fn build_oauth_refresh_future(args: OauthRefreshArgs) -> SchedulerFuture {
    let OauthRefreshArgs {
        name,
        url,
        token_endpoint,
        client_id,
        client_secret,
        refresh_token,
        resource,
        scope,
    } = args;
    Box::pin(async move {
        let http = match crate::mcp_oauth::http_client() {
            Ok(c) => c,
            Err(e) => {
                return SchedulerCompletion::OauthRefresh(OauthRefreshCompletion {
                    name,
                    url,
                    result: Err(format!("init http client: {e}")),
                });
            }
        };
        let result = async {
            let token = crate::mcp_oauth::refresh_access_token(
                &http,
                &token_endpoint,
                &refresh_token,
                &client_id,
                client_secret.as_deref(),
                &resource,
                scope.as_deref(),
            )
            .await
            .map_err(|e| format!("refresh_access_token: {e}"))?;
            // Re-open the MCP session with the new access token. The
            // MCP `initialize` RPC is cheap — one round-trip — and
            // keeps the session state machine clean. If this ever
            // becomes hot, we can swap `McpSession.bearer` for an
            // interior-mutable field instead.
            let session = Arc::new(
                McpSession::connect(&url, Some(token.access_token.clone()))
                    .await
                    .map_err(|e| format!("mcp reconnect with rotated token: {e}"))?,
            );
            let expires_at = token
                .expires_in
                .map(|secs| chrono::Utc::now().timestamp() + secs);
            Ok::<OauthRefreshData, String>(OauthRefreshData {
                session,
                access_token: token.access_token,
                refresh_token: token.refresh_token,
                expires_at,
                scope: token.scope,
            })
        }
        .await;
        SchedulerCompletion::OauthRefresh(OauthRefreshCompletion { name, url, result })
    })
}

/// Build a future that connects to a shared-MCP-host URL, runs
/// `tools/list`, and yields a [`SchedulerCompletion::SharedMcp`].
/// The scheduler pushes this into `pending_io` from the Add/Update
/// handlers; the completion's `apply_shared_mcp_completion` performs
/// the catalog + registry mutation on the scheduler thread.
pub(crate) fn build_shared_mcp_connect_future(
    conn_id: crate::runtime::scheduler::ConnId,
    correlation_id: Option<String>,
    name: String,
    url: String,
    auth: crate::tools::shared_mcp_catalog::SharedMcpAuth,
    op: SharedMcpOp,
) -> SchedulerFuture {
    Box::pin(async move {
        let bearer = auth.bearer().map(str::to_string);
        let result = async {
            let session = Arc::new(
                McpSession::connect(&url, bearer)
                    .await
                    .map_err(|e| format!("connect `{name}` ({url}): {e}"))?,
            );
            let tools = session
                .list_tools()
                .await
                .map_err(|e| format!("tools/list on `{name}`: {e}"))?;
            Ok::<_, String>((session, tools))
        }
        .await;
        SchedulerCompletion::SharedMcp(SharedMcpCompletion {
            conn_id,
            correlation_id,
            name,
            url,
            auth,
            op,
            result,
        })
    })
}

/// Result of a host-env provider reachability probe (`GET /health`).
/// Produced by the background ticker; consumed in
/// `apply_probe_completion` where it updates the registry entry's
/// `reachability` and emits a push if the state changed.
pub(crate) struct ProbeCompletion {
    pub(crate) name: String,
    pub(crate) observed_at: chrono::DateTime<chrono::Utc>,
    /// `Ok(())` on successful probe; `Err(message)` on any failure
    /// (connect timeout, non-2xx, I/O error). The scheduler maps the
    /// error message to the `last_error` string on the registry.
    pub(crate) result: Result<(), String>,
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

/// Build a future that provisions a specific `host_env_id` (env +
/// MCP connect + initial `tools/list`) and yields a
/// [`SchedulerCompletion::Provision`]. Called once per (thread, id)
/// pair; threads with multiple host-env bindings get one future per
/// binding. `thread_id` is carried through only so the completion
/// can be tied back to who originally dispatched — the per-id guard
/// deduplicates across threads sharing the same binding.
pub(crate) fn provision_host_env_mcp(
    scheduler: &Scheduler,
    thread_id: String,
    host_env_id: HostEnvId,
) -> SchedulerFuture {
    // Snapshot the host-env decision while we have the sync borrow.
    // The entry must exist — ensure_host_env_provisioning pre-registered
    // it before dispatching. If the entry is already Ready (dedup hit),
    // skip provision and reuse the cached URL.
    let host_env_action = match scheduler.resources().host_envs.get(&host_env_id) {
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
                        message: "host_env id not registered at dispatch time".into(),
                        host_env_id,
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
                    host_env_lost: None,
                })
            });
        }
    };
    let stream_tx = scheduler.stream_sender();
    Box::pin(async move {
        debug!(%thread_id, op_id, backend = %backend_name, model = %model_name, "dispatching streaming model call");
        let req = ModelRequest {
            model: &owned_req.model,
            max_tokens: owned_req.max_tokens,
            system_prompt: &owned_req.system_prompt,
            tools: &owned_req.tools,
            messages: &owned_req.messages,
            cache_breakpoints: &owned_req.cache_breakpoints,
        };
        let result =
            match stream_with_rate_limit_retry(provider.as_ref(), &req, &thread_id, &stream_tx)
                .await
            {
                Ok(resp) => IoResult::ModelCall(Ok(resp)),
                Err(e) => IoResult::ModelCall(Err(e.to_string())),
            };
        SchedulerCompletion::Io(IoCompletion {
            thread_id,
            op_id,
            result,
            pod_update: None,
            scheduler_command: None,
            host_env_lost: None,
        })
    })
}

/// Upper bound on how long we're willing to sleep through a single
/// rate-limit signal before giving up. Short Code-Assist 429s (say ~6s)
/// flow through transparently; long quota-reset waits surface as
/// failures so the user isn't left waiting on a minutes-long sleep.
const MAX_RATE_LIMIT_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// Drive a streaming model call to completion, pushing live deltas into
/// `stream_tx` as they arrive and returning the assembled `ModelResponse`.
///
/// Rate-limit retry only applies when the *first* stream item is the
/// RateLimited error — once any delta has been broadcast, a mid-stream
/// failure is terminal (we can't un-send events to subscribers).
async fn stream_with_rate_limit_retry(
    provider: &dyn crate::providers::model::ModelProvider,
    req: &ModelRequest<'_>,
    thread_id: &str,
    stream_tx: &tokio::sync::mpsc::UnboundedSender<StreamUpdate>,
) -> Result<crate::providers::model::ModelResponse, crate::providers::model::ModelError> {
    match consume_stream(provider, req, thread_id, stream_tx).await {
        Ok(resp) => Ok(resp),
        Err(FirstEventError::RateLimited {
            retry_after,
            body: _,
        }) if retry_after <= MAX_RATE_LIMIT_WAIT => {
            tracing::warn!(
                %thread_id,
                wait_ms = retry_after.as_millis() as u64,
                "model backend rate-limited on first event; waiting before single retry"
            );
            tokio::time::sleep(retry_after).await;
            match consume_stream(provider, req, thread_id, stream_tx).await {
                Ok(resp) => Ok(resp),
                Err(e) => Err(e.into_model_error()),
            }
        }
        Err(e) => Err(e.into_model_error()),
    }
}

/// Error shape that distinguishes "error before any delta was emitted"
/// (safe to retry) from "error after streaming started" (terminal). The
/// `RateLimited` variant preserves the retry hint so the caller can
/// decide whether to sleep and try again.
enum FirstEventError {
    /// First item was `Err(RateLimited {..})` — retryable.
    RateLimited {
        retry_after: std::time::Duration,
        body: String,
    },
    /// Any other error, or mid-stream failure — terminal.
    Fatal(crate::providers::model::ModelError),
}

impl FirstEventError {
    fn into_model_error(self) -> crate::providers::model::ModelError {
        match self {
            FirstEventError::RateLimited { retry_after, body } => {
                crate::providers::model::ModelError::RateLimited { retry_after, body }
            }
            FirstEventError::Fatal(e) => e,
        }
    }
}

async fn consume_stream(
    provider: &dyn crate::providers::model::ModelProvider,
    req: &ModelRequest<'_>,
    thread_id: &str,
    stream_tx: &tokio::sync::mpsc::UnboundedSender<StreamUpdate>,
) -> Result<crate::providers::model::ModelResponse, FirstEventError> {
    let mut stream = provider.create_message_streaming(req);
    let mut any_delta_emitted = false;
    while let Some(event) = stream.next().await {
        match event {
            Ok(ModelEvent::TextDelta { text }) => {
                any_delta_emitted = true;
                let _ = stream_tx.send(StreamUpdate {
                    thread_id: thread_id.to_string(),
                    event: ServerToClient::ThreadAssistantTextDelta {
                        thread_id: thread_id.to_string(),
                        delta: text,
                    },
                });
            }
            Ok(ModelEvent::ThinkingDelta { text }) => {
                any_delta_emitted = true;
                let _ = stream_tx.send(StreamUpdate {
                    thread_id: thread_id.to_string(),
                    event: ServerToClient::ThreadAssistantReasoningDelta {
                        thread_id: thread_id.to_string(),
                        delta: text,
                    },
                });
            }
            Ok(ModelEvent::ToolCall { .. }) => {
                // ToolCall events aren't broadcast as wire deltas — the
                // thread state machine emits ThreadToolCallBegin when
                // it actually dispatches each tool (after approval
                // evaluation). Letting the event through before that
                // would duplicate the tool-call row in the webui.
                any_delta_emitted = true;
            }
            Ok(ModelEvent::Completed {
                content,
                stop_reason,
                usage,
            }) => {
                return Ok(crate::providers::model::ModelResponse {
                    content,
                    stop_reason,
                    usage,
                });
            }
            Err(crate::providers::model::ModelError::RateLimited { retry_after, body })
                if !any_delta_emitted =>
            {
                return Err(FirstEventError::RateLimited { retry_after, body });
            }
            Err(e) => return Err(FirstEventError::Fatal(e)),
        }
    }
    Err(FirstEventError::Fatal(
        crate::providers::model::ModelError::Transport(
            "model stream ended without Completed event".into(),
        ),
    ))
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
                            host_env_lost: None,
                        })
                    });
                }
            };
            let pod_modify = scheduler.thread_pod_modify_cap(&thread_id);
            let behaviors_cap = scheduler.thread_behaviors_cap(&thread_id);
            Box::pin(async move {
                let outcome = crate::tools::builtin_tools::dispatch(
                    snapshot.pod_dir,
                    snapshot.config,
                    snapshot.behavior_ids,
                    pod_modify,
                    behaviors_cap,
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
                    host_env_lost: None,
                })
            })
        }
        Some(ToolRoute::Mcp {
            session: mcp,
            real_name,
            host,
        }) => {
            let stream_tx = scheduler.stream_sender();
            let stream_thread_id = thread_id.clone();
            let stream_tool_use_id = tool_use_id.clone();
            let mcp_host_id = crate::pod::resources::McpHostId(host);
            Box::pin(async move {
                // `real_name` is the tool name the MCP host advertises
                // server-side — the public name the model called minus
                // any `{env_name}_` prefix route_tool stripped off.
                //
                // If the invoke call itself fails at the transport
                // layer AND the MCP is host-env-backed, signal the
                // scheduler to mark the env `Lost` — the daemon
                // session is gone and no amount of retry against this
                // URL + session_id will bring it back. Shared MCPs
                // just surface the error as-is; Lost is specifically
                // a host-env concept (they're the thing that gets
                // re-provisioned per thread).
                let mut host_env_lost: Option<HostEnvLostSignal> = None;
                let result = match mcp.invoke(&real_name, input).await {
                    Ok(mut stream) => {
                        let mut last = None;
                        while let Some(event) = stream.next().await {
                            match event {
                                crate::tools::mcp::ToolEvent::Content(block) => {
                                    let wire_block = mcp_block_to_wire(&block);
                                    let _ = stream_tx.send(StreamUpdate {
                                        thread_id: stream_thread_id.clone(),
                                        event: ServerToClient::ThreadToolCallContent {
                                            thread_id: stream_thread_id.clone(),
                                            tool_use_id: stream_tool_use_id.clone(),
                                            block: wire_block,
                                        },
                                    });
                                }
                                crate::tools::mcp::ToolEvent::Completed(r) => last = Some(r),
                            }
                        }
                        match last {
                            Some(r) => Ok(r),
                            None => Err("no Completed event in tool stream".to_string()),
                        }
                    }
                    Err(e) => {
                        if mcp_host_id.is_host_env() && e.is_transport_lost() {
                            host_env_lost = Some(HostEnvLostSignal {
                                mcp_host_id: mcp_host_id.clone(),
                                message: e.to_string(),
                            });
                        }
                        Err(e.to_string())
                    }
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
                    host_env_lost,
                })
            })
        }
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
                host_env_lost: None,
            })
        }),
    }
}

/// Translate an MCP transport content block into the protocol's
/// conversational `ContentBlock` shape. The transport today only carries
/// `Text`; additional MCP content types (image, audio, resource) map to
/// equivalents here as they're added to both sides.
fn mcp_block_to_wire(
    b: &crate::tools::mcp::McpContentBlock,
) -> whisper_agent_protocol::ContentBlock {
    match b {
        crate::tools::mcp::McpContentBlock::Text { text } => {
            whisper_agent_protocol::ContentBlock::Text { text: text.clone() }
        }
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
    // System prompt + tool manifest live at the head of the
    // conversation (captured there at thread creation / MCP-rebind
    // time). Adapters keep their existing `req.system_prompt` +
    // `req.tools` expectations — we just source those fields from
    // the conversation here instead of a parallel config slot.
    let system_prompt = task.conversation.system_prompt_text().to_string();
    let tools: Vec<ToolSpec> = task
        .conversation
        .tool_schemas()
        .map(|(name, description, input_schema)| ToolSpec {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: input_schema.clone(),
        })
        .collect();
    let setup_end = task.conversation.setup_prefix_end();
    let messages = task.conversation.messages()[setup_end..].to_vec();
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
    let cache_breakpoints = default_cache_policy(&messages);
    (
        OwnedModelRequest {
            model: model.clone(),
            max_tokens,
            system_prompt,
            tools,
            messages,
            cache_breakpoints,
        },
        model,
        backend_name,
    )
}
