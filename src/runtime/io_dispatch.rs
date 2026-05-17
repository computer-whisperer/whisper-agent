//! I/O future construction for the scheduler.
//!
//! [`SchedulerCompletion::Io`] is the per-thread state-machine completion
//! (model call, tool call) built from a [`task::IoRequest`] the thread
//! requested via `step()`. Routed back through `Thread::apply_io_result`.
//!
//! Per-IO-type knowledge (how to assemble a model request, etc.) lives
//! here so adding a new variant only touches this module plus the
//! consumer (thread state machine).

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::StreamExt;
use tracing::debug;
use whisper_agent_protocol::{ServerToClient, TunableValue};

use crate::providers::model::{
    CacheBreakpoint, ModelEvent, ModelRequest, ToolSpec, default_cache_policy,
};
use crate::runtime::scheduler::{BackendUsageUpdate, Scheduler, StreamUpdate};
use crate::runtime::thread::{IoRequest, IoResult, OpId};
use crate::tools::mcp::McpSession;

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
    /// Knowledge-hit keys surfaced by this IO completion. Explicit
    /// `knowledge_query` tool calls populate this so opportunistic
    /// autoquery can avoid repeating the same records later in the
    /// thread.
    pub(crate) knowledge_hit_keys: Vec<String>,
}

/// Unified completion type the scheduler's `FuturesUnordered` carries.
pub(crate) enum SchedulerCompletion {
    Io(IoCompletion),
    KnowledgeAutoquery(KnowledgeAutoqueryCompletion),
    SharedMcp(SharedMcpCompletion),
    OauthStart(OauthStartCompletion),
    OauthComplete(OauthCompleteCompletion),
    OauthRefresh(OauthRefreshCompletion),
    /// A `sudo` approval completed its wrapped tool call; the scheduler
    /// loop routes this to `complete_function` on the Function::Sudo
    /// entry, firing the parked parent tool call's oneshot delivery.
    SudoInner(SudoInnerCompletion),
}

pub(crate) struct KnowledgeAutoqueryCompletion {
    pub(crate) thread_id: String,
    pub(crate) result: Result<KnowledgeAutoqueryResult, String>,
}

pub(crate) struct KnowledgeAutoqueryResult {
    pub(crate) query: String,
    pub(crate) labels: Vec<String>,
    pub(crate) hits: Vec<crate::knowledge::RerankedCandidate>,
}

/// Result of a sudo'd inner tool invocation. `result` is the usual
/// builtin/MCP outcome: `Ok(CallToolResult)` on a reachable handler
/// (which may still carry `is_error = true`) or `Err(message)` when
/// routing / transport failed before we could run it.
///
/// `pod_update` / `scheduler_command` mirror the same fields on
/// `IoCompletion` — a successful sudo'd builtin tool produces the
/// same side effects (`pod.toml` reparse, behavior register, etc.)
/// as a direct call, and the scheduler applies them before the
/// Function's terminal fires so subsequent sudo calls observe the
/// refreshed state.
pub(crate) struct SudoInnerCompletion {
    pub(crate) function_id: crate::functions::FunctionId,
    pub(crate) thread_id: String,
    pub(crate) decision: crate::permission::SudoDecision,
    pub(crate) result: Result<crate::tools::mcp::CallToolResult, String>,
    pub(crate) pod_update: Option<crate::tools::builtin_tools::PodUpdate>,
    pub(crate) scheduler_command: Option<crate::tools::builtin_tools::SchedulerCommand>,
}

/// Which Add/Update operation the completion applies to. The
/// scheduler uses this to decide the reply event and whether the
/// catalog write is an `insert` or `update`. Also carries the prefix
/// intent: `Add` always names a concrete value (resolved from the
/// wire's `SharedMcpPrefixInput::Unchanged` to `Default = None`) so
/// the insert is unambiguous; `Update` uses outer-`None` to mean
/// "leave the catalog's prefix untouched" (mirroring how `auth` is
/// optional on `UpdateSharedMcpHost`).
#[derive(Debug, Clone)]
pub(crate) enum SharedMcpOp {
    Add { prefix: Option<String> },
    Update { prefix: Option<Option<String>> },
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
    /// Echoed verbatim from the originating `AddSharedMcpHost` so the
    /// scheduler can stash it on the pending flow without an extra
    /// lookup. Three-state per `CatalogEntry::prefix`.
    pub(crate) prefix: Option<String>,
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
    /// Operator-selected tool-name prefix for the catalog entry.
    /// Three-state per `CatalogEntry::prefix`.
    pub(crate) prefix: Option<String>,
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_oauth_start_future(
    conn_id: crate::runtime::scheduler::ConnId,
    correlation_id: Option<String>,
    name: String,
    url: String,
    scope: Option<String>,
    redirect_base: String,
    prefix: Option<String>,
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
                    prefix,
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
            prefix,
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
    prefix: Option<String>,
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
                    prefix,
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
            prefix,
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

pub(crate) type SchedulerFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = SchedulerCompletion> + Send>>;

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
                    knowledge_hit_keys: Vec::new(),
                })
            });
        }
    };
    let stream_tx = scheduler.stream_sender();
    let usage_tx = scheduler.usage_sender();
    let forensic_sink = scheduler.forensic_sink();
    let cancel = scheduler.cancel_token_or_default(&thread_id);
    Box::pin(async move {
        debug!(%thread_id, op_id, backend = %backend_name, model = %model_name, "dispatching streaming model call");
        let req = ModelRequest {
            model: &owned_req.model,
            max_tokens: owned_req.max_tokens,
            system_prompt: &owned_req.system_prompt,
            tools: &owned_req.tools,
            messages: &owned_req.messages,
            cache_breakpoints: &owned_req.cache_breakpoints,
            tunables: &owned_req.tunables,
            request_cache_key: Some(owned_req.request_cache_key.as_str()),
            session_id: Some(owned_req.session_id.as_str()),
            installation_id: Some(owned_req.installation_id.as_str()),
        };
        let result = match stream_with_retry(
            provider.as_ref(),
            &req,
            &thread_id,
            &backend_name,
            &stream_tx,
            &usage_tx,
            &cancel,
        )
        .await
        {
            Ok(resp) => IoResult::ModelCall(Ok(resp)),
            Err(e) => {
                // Structured trace + (optional) disk dump on terminal
                // failure. The structured fields land in `kubectl logs`
                // immediately; the disk dump preserves the request +
                // response bytes for offline replay if the operator
                // wants to reproduce the failure against the provider.
                let detail = e.provider_detail().cloned().unwrap_or_default();
                tracing::warn!(
                    %thread_id,
                    op_id,
                    backend = %backend_name,
                    model = %model_name,
                    error.code = detail.code.as_deref().unwrap_or(""),
                    error.kind = detail.kind.as_deref().unwrap_or(""),
                    error.response_id = detail.response_id.as_deref().unwrap_or(""),
                    error.message = detail.message.as_deref().unwrap_or(""),
                    forensic_captured = e.forensic().is_some(),
                    error = %e,
                    "model call failed terminally"
                );
                forensic_sink
                    .record_terminal_failure(&thread_id, &backend_name, op_id, &e)
                    .await;
                IoResult::ModelCall(Err(e.to_string()))
            }
        };
        SchedulerCompletion::Io(IoCompletion {
            thread_id,
            op_id,
            result,
            pod_update: None,
            scheduler_command: None,
            knowledge_hit_keys: Vec::new(),
        })
    })
}

/// Upper bound on how long we're willing to sleep through a single
/// rate-limit signal before giving up. Short Code-Assist 429s (say ~6s)
/// flow through transparently; long quota-reset waits surface as
/// failures so the user isn't left waiting on a minutes-long sleep.
const MAX_RATE_LIMIT_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// Maximum number of retry attempts (not counting the initial try)
/// for first-event failures. Shared budget across rate-limit and
/// transient-fatal (transport / 5xx / 408) errors.
const MAX_FIRST_EVENT_RETRIES: u32 = 3;

/// Drive a streaming model call to completion, pushing live deltas into
/// `stream_tx` as they arrive and returning the assembled `ModelResponse`.
///
/// Retries only apply when the *first* stream item is the error — once
/// any delta has been broadcast, a mid-stream failure is terminal (we
/// can't un-send events to subscribers). Retryable first-event errors:
/// server-signalled rate limits with a short `retry_after`, and any
/// `ModelError::is_transient()` (transport glitch, 408, or 5xx). The
/// retry budget is shared; transient errors use a fixed exponential
/// backoff (1s, 2s, 4s).
async fn stream_with_retry(
    provider: &dyn crate::providers::model::ModelProvider,
    req: &ModelRequest<'_>,
    thread_id: &str,
    backend_name: &str,
    stream_tx: &tokio::sync::mpsc::UnboundedSender<StreamUpdate>,
    usage_tx: &tokio::sync::mpsc::UnboundedSender<BackendUsageUpdate>,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<crate::providers::model::ModelResponse, crate::providers::model::ModelError> {
    let mut attempt = 0u32;
    loop {
        // Fast-path: caller cancelled before (or between) attempts.
        if cancel.is_cancelled() {
            return Err(crate::providers::model::ModelError::Cancelled);
        }
        match consume_stream(
            provider,
            req,
            thread_id,
            backend_name,
            stream_tx,
            usage_tx,
            cancel,
        )
        .await
        {
            Ok(resp) => return Ok(resp),
            Err(FirstEventError::RateLimited { retry_after, body })
                if retry_after <= MAX_RATE_LIMIT_WAIT && attempt < MAX_FIRST_EVENT_RETRIES =>
            {
                tracing::warn!(
                    %thread_id,
                    attempt = attempt + 1,
                    wait_ms = retry_after.as_millis() as u64,
                    "model backend rate-limited on first event; waiting before retry"
                );
                // Honour cancel during the backoff sleep — the user
                // shouldn't have to wait out a 30s rate-limit wait
                // before their cancel takes effect.
                tokio::select! {
                    _ = tokio::time::sleep(retry_after) => {}
                    _ = cancel.cancelled() => {
                        return Err(crate::providers::model::ModelError::Cancelled);
                    }
                }
                attempt += 1;
                // Body is only needed when we give up — on retry
                // success / further retries we discard it.
                drop(body);
            }
            Err(FirstEventError::Fatal(e))
                if e.is_transient() && attempt < MAX_FIRST_EVENT_RETRIES =>
            {
                let wait = transient_backoff(attempt);
                let detail = e.provider_detail().cloned().unwrap_or_default();
                tracing::warn!(
                    %thread_id,
                    backend = %backend_name,
                    attempt = attempt + 1,
                    wait_ms = wait.as_millis() as u64,
                    error.code = detail.code.as_deref().unwrap_or(""),
                    error.kind = detail.kind.as_deref().unwrap_or(""),
                    error.response_id = detail.response_id.as_deref().unwrap_or(""),
                    error = %e,
                    "transient model-call error on first event; retrying after backoff"
                );
                tokio::select! {
                    _ = tokio::time::sleep(wait) => {}
                    _ = cancel.cancelled() => {
                        return Err(crate::providers::model::ModelError::Cancelled);
                    }
                }
                attempt += 1;
            }
            Err(e) => return Err(e.into_model_error()),
        }
    }
}

/// Backoff for transient first-event failures (non-rate-limit).
/// Sequence: 1s, 2s, 4s. `attempt` is zero-indexed — 0 is the wait
/// taken *before* the second attempt.
fn transient_backoff(attempt: u32) -> std::time::Duration {
    let secs = 1u64 << attempt.min(3);
    std::time::Duration::from_secs(secs)
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
    backend_name: &str,
    stream_tx: &tokio::sync::mpsc::UnboundedSender<StreamUpdate>,
    usage_tx: &tokio::sync::mpsc::UnboundedSender<BackendUsageUpdate>,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<crate::providers::model::ModelResponse, FirstEventError> {
    let mut stream = provider.create_message_streaming(req, cancel);
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
            Ok(ModelEvent::ImageBlock { source }) => {
                // Live-update channel for model-emitted images.
                // Persisted state still rides on `ModelEvent::Completed`
                // — clients receiving this delta render the thumbnail
                // immediately so the user sees the image as soon as
                // the wire frame arrives, not after the next snapshot.
                any_delta_emitted = true;
                let _ = stream_tx.send(StreamUpdate {
                    thread_id: thread_id.to_string(),
                    event: ServerToClient::ThreadAssistantImage {
                        thread_id: thread_id.to_string(),
                        source,
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
            Ok(ModelEvent::ToolCallStreaming {
                id,
                name,
                args_chars,
            }) => {
                // Placeholder events while the model streams args JSON.
                // Forwarded verbatim so clients can frame the row
                // early. Does not set `any_delta_emitted` — a
                // rate-limit error right after these is still first
                // event and retry-eligible.
                let _ = stream_tx.send(StreamUpdate {
                    thread_id: thread_id.to_string(),
                    event: ServerToClient::ThreadToolCallStreaming {
                        thread_id: thread_id.to_string(),
                        tool_use_id: id,
                        name,
                        args_chars,
                    },
                });
            }
            Ok(ModelEvent::ProviderWarning { code, message }) => {
                // Provider signalled a non-fatal degradation alongside an
                // otherwise-successful call (Gemini SAFETY, OpenAI
                // content_filter, …). The Completed event still lands;
                // this fires beforehand so the warning shows up in
                // `kubectl logs` adjacent to the turn it came from.
                // Does not set `any_delta_emitted` — adapters emit this
                // either before or after deltas, and either way the
                // retry classifier should key off the actual content
                // events, not the warning.
                tracing::warn!(
                    %thread_id,
                    backend = %backend_name,
                    code = %code,
                    message = %message,
                    "model provider warning"
                );
            }
            Ok(ModelEvent::ProviderUsage { snapshot }) => {
                // Per-backend account/quota snapshot from the provider
                // (today: Codex header scrape). Forward off-thread to
                // the scheduler so it can update the registry entry and
                // broadcast `ResourceUpdated`. Does not set
                // `any_delta_emitted` — the snapshot rides the response
                // headers, no real model output has happened yet, so a
                // rate-limit error landing right after is still
                // retry-eligible. Channel send is fire-and-forget: if
                // the scheduler is mid-shutdown the receiver is closed,
                // and there's nothing useful to surface to the model
                // call about that.
                let _ = usage_tx.send(BackendUsageUpdate {
                    backend_name: backend_name.to_string(),
                    usage: snapshot,
                });
            }
            Ok(ModelEvent::PrefillProgress {
                tokens_processed,
                tokens_total,
            }) => {
                // Purely decorative heartbeat; does not set
                // `any_delta_emitted` because the provider hasn't yet
                // produced real output. If a rate-limit error arrives
                // after one of these, we still treat it as first-event
                // (retryable) rather than mid-stream (fatal).
                let _ = stream_tx.send(StreamUpdate {
                    thread_id: thread_id.to_string(),
                    event: ServerToClient::ThreadPrefillProgress {
                        thread_id: thread_id.to_string(),
                        tokens_processed,
                        tokens_total,
                    },
                });
            }
            Ok(ModelEvent::OutputTokensProgress { output_tokens }) => {
                // Decorative heartbeat for the live tokens-per-second
                // indicator. Arrives after the first real output token,
                // so `any_delta_emitted` is set — a rate-limit error
                // after this is mid-stream and fatal (matches the
                // TextDelta / ThinkingDelta arms).
                any_delta_emitted = true;
                let _ = stream_tx.send(StreamUpdate {
                    thread_id: thread_id.to_string(),
                    event: ServerToClient::ThreadOutputTokensProgress {
                        thread_id: thread_id.to_string(),
                        output_tokens,
                    },
                });
            }
            Ok(ModelEvent::Completed {
                content,
                stop_reason,
                usage,
            }) => {
                // Route `ThreadAssistantEnd` through `stream_tx`
                // *here*, immediately after any trailing delta events
                // we just forwarded for the same turn. The natural
                // place to fire this wire event would be from
                // [`thread_router::dispatch_events`] when it processes
                // the `ThreadEvent::AssistantEnd` that
                // `integrate_model_response` pushes — but that path
                // runs from `pending_io` in the scheduler's main
                // select!, which races with `stream_rx`. When both
                // are ready (model finished AND a trailing
                // `ThreadOutputTokensProgress` is sitting in the
                // queue), `tokio::select!` picks randomly, so the
                // wire could see `AssistantEnd` before its preceding
                // `OutputTokensProgress` — which puts the UI's live
                // tok/s chip into a phantom-live state. Sending End
                // through `stream_tx` ourselves means it rides the
                // *same* FIFO channel as the deltas, so the wire
                // order matches the emission order. The
                // `ThreadEvent::AssistantEnd` is still produced for
                // persistence + internal bookkeeping; only the wire
                // broadcast moves channels (see the matching no-op
                // in `thread_router::dispatch_events`).
                let _ = stream_tx.send(StreamUpdate {
                    thread_id: thread_id.to_string(),
                    event: ServerToClient::ThreadAssistantEnd {
                        thread_id: thread_id.to_string(),
                        stop_reason: stop_reason.clone(),
                        usage,
                    },
                });
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
    let cancel = scheduler.cancel_token_or_default(&thread_id);
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
                            knowledge_hit_keys: Vec::new(),
                        })
                    });
                }
            };
            let pod_modify = scheduler.thread_pod_modify_cap(&thread_id);
            let behaviors_cap = scheduler.thread_behaviors_cap(&thread_id);
            // Authoring gate: a write to `behaviors/<id>/behavior.toml`
            // is denied if the caller's `BehaviorOpsCap` doesn't admit
            // the declared scope. Runs BEFORE the builtin dispatch so
            // a partially-applied state is never reached — if the gate
            // denies, we never hit `prepare_update` or `PodUpdate`.
            if let Some(denial) = scheduler.check_behavior_authoring(&thread_id, &name, &input) {
                return Box::pin(async move {
                    SchedulerCompletion::Io(IoCompletion {
                        thread_id,
                        op_id,
                        result: IoResult::ToolCall {
                            tool_use_id,
                            result: Err(denial),
                        },
                        pod_update: None,
                        scheduler_command: None,
                        knowledge_hit_keys: Vec::new(),
                    })
                });
            }
            Box::pin(async move {
                let dispatch_fut = crate::tools::builtin_tools::dispatch(
                    snapshot.pod_dir,
                    snapshot.config,
                    snapshot.behavior_ids,
                    pod_modify,
                    behaviors_cap,
                    &name,
                    input,
                );
                // Builtin tools don't talk to an external service we
                // need to notify on cancel — dropping the future mid-
                // await is enough. Most builtins are fast (file ops);
                // for the few that do awaited work this interrupts at
                // the next `.await` boundary.
                let outcome = tokio::select! {
                    o = dispatch_fut => o,
                    _ = cancel.cancelled() => {
                        return SchedulerCompletion::Io(IoCompletion {
                            thread_id,
                            op_id,
                            result: IoResult::ToolCall {
                                tool_use_id,
                                result: Err("cancelled".into()),
                            },
                            pod_update: None,
                            scheduler_command: None,
                            knowledge_hit_keys: Vec::new(),
                        });
                    }
                };
                let pod_update = outcome.pod_update;
                let scheduler_command = outcome.scheduler_command;
                let result = if outcome.result.is_error {
                    // Surface tool-level errors as Err in the task-facing
                    // IoResult — the approval-audit path uses the Err
                    // variant to decide how to record the turn. Builtin
                    // tool errors already carry a text block we can
                    // pass through.
                    Err(crate::tools::mcp::mcp_blocks_text_preview(
                        &outcome.result.content,
                    ))
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
                    knowledge_hit_keys: Vec::new(),
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
            let _ = host;
            Box::pin(async move {
                // `real_name` is the tool name the MCP host advertises
                // server-side — the public name the model called minus
                // any `{env_name}_` prefix route_tool stripped off.
                let result = match mcp.invoke(&real_name, input, &cancel).await {
                    Ok(mut stream) => {
                        let mut last = None;
                        while let Some(event) = stream.next().await {
                            match event {
                                crate::tools::mcp::ToolEvent::Content(block) => {
                                    let wire_block = block.to_content_block();
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
                            // Stream ended without Completed: either
                            // the host closed mid-call, or the cancel
                            // wrapper inside `invoke` ended the stream
                            // after firing `notifications/cancelled`.
                            // Either way the scheduler drops this
                            // result when the thread is Cancelled.
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
                    knowledge_hit_keys: Vec::new(),
                })
            })
        }
        Some(ToolRoute::V2HostEnv {
            daemon_handle,
            binding_name,
            spec,
            real_name,
        }) => {
            let sessions = scheduler.v2_session_store();
            let contexts = scheduler.v2_context_store();
            let registry = scheduler.v2_daemon_registry();
            let policy = scheduler.v2_policy_store().get(&thread_id);
            // Resolve x-content-ref params from conversation state
            // before dispatch — bytes ride to the daemon as an
            // attachments sidecar so the tool's input_schema can
            // declare an honest `index: integer` parameter while the
            // worker actually receives the resolved bytes. Resolution
            // failures (out-of-range index, malformed args) short-
            // circuit to a tool-call error without ever opening a
            // worker session.
            let attachments = match daemon_handle
                .capabilities()
                .tools
                .iter()
                .find(|t| t.name == real_name)
            {
                Some(tool) => match crate::runtime::v2_dispatch::resolve_content_refs(
                    &tool.input_schema,
                    &input,
                    &scheduler
                        .task(&thread_id)
                        .expect("task exists")
                        .conversation,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        return Box::pin(async move {
                            SchedulerCompletion::Io(IoCompletion {
                                thread_id,
                                op_id,
                                result: IoResult::ToolCall {
                                    tool_use_id,
                                    result: Err(e),
                                },
                                pod_update: None,
                                scheduler_command: None,
                                knowledge_hit_keys: Vec::new(),
                            })
                        });
                    }
                },
                None => Vec::new(),
            };
            Box::pin(async move {
                let result = crate::runtime::v2_dispatch::dispatch_v2_tool(
                    sessions,
                    contexts,
                    registry,
                    daemon_handle,
                    policy,
                    thread_id.clone(),
                    binding_name,
                    spec,
                    real_name,
                    input,
                    attachments,
                    cancel,
                )
                .await;
                let result = match result {
                    Ok(r) if r.is_error => {
                        Err(crate::tools::mcp::mcp_blocks_text_preview(&r.content))
                    }
                    Ok(r) => Ok(r),
                    Err(e) => Err(e),
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
                    knowledge_hit_keys: Vec::new(),
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
                knowledge_hit_keys: Vec::new(),
            })
        }),
    }
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
    tunables: BTreeMap<String, TunableValue>,
    /// Per-thread routing-affinity key — surfaced to providers as
    /// [`ModelRequest::request_cache_key`]. Sourced from `thread_id`
    /// so every request for the same thread shares it.
    request_cache_key: String,
    /// Server-process lifetime identifier. Surfaced to providers as
    /// [`ModelRequest::session_id`]. Cloned from the scheduler's
    /// startup-minted UUID; same value for every thread served by
    /// this process.
    session_id: String,
    /// Per-deployment identifier persisted under the data dir.
    /// Surfaced to providers as [`ModelRequest::installation_id`].
    /// Cloned from the scheduler's load-or-create startup value.
    installation_id: String,
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
        .map(|t| ToolSpec {
            name: t.name.to_string(),
            description: t.description.to_string(),
            params: t.params.to_vec(),
            kind: t.kind,
        })
        .collect();
    let setup_end = task.conversation.setup_prefix_end();
    let messages = task.conversation.messages()[setup_end..].to_vec();
    let backend_name = task.bindings.backend.clone();
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
    let tunables = task.config.tunables.clone();
    let request_cache_key = thread_id.to_string();
    let session_id = scheduler.session_id().to_string();
    let installation_id = scheduler.installation_id().to_string();
    (
        OwnedModelRequest {
            model: model.clone(),
            max_tokens,
            system_prompt,
            tools,
            messages,
            cache_breakpoints,
            tunables,
            request_cache_key,
            session_id,
            installation_id,
        },
        model,
        backend_name,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_backoff_doubles_then_caps() {
        use std::time::Duration;
        assert_eq!(transient_backoff(0), Duration::from_secs(1));
        assert_eq!(transient_backoff(1), Duration::from_secs(2));
        assert_eq!(transient_backoff(2), Duration::from_secs(4));
        // Clamp keeps the sequence bounded even if an accidental
        // off-by-one walks past MAX_FIRST_EVENT_RETRIES.
        assert_eq!(transient_backoff(3), Duration::from_secs(8));
        assert_eq!(transient_backoff(99), Duration::from_secs(8));
    }
}
