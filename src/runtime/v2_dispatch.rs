//! v2 host-env tool dispatch.
//!
//! Mirror of the MCP-call dispatcher in [`crate::runtime::io_dispatch`]
//! but for v2 host-env daemons: look up an existing
//! [`crate::tools::host_env_link::SessionHandle`] for the
//! `(thread_id, binding_name)` pair, opening one if missing, and
//! dispatch a [`whisper_agent_host_proto::Frame::InvokeTool`] through
//! it. Returns the result in the [`crate::tools::mcp::CallToolResult`]
//! shape the rest of the runtime expects (translating from the
//! host-proto shape, which is identical-modulo-names).
//!
//! Phase 4c only awaits the terminal `ToolFinal` — streaming
//! `ToolChunk`s are not surfaced (mirrors phase 2b's cut on the
//! server side). Hooking chunks into the existing `StreamUpdate`
//! mechanism is a follow-up.

use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;
use whisper_agent_host_proto::{ContentBlock, HostEnvSpec};

use crate::runtime::scheduler::{V2ContextStore, V2DisconnectPolicy, V2SessionStore};
use crate::tools::host_env_link::{LinkError, LiveDaemonHandle, LiveDaemonRegistry};
use crate::tools::mcp::{CallToolResult, McpContentBlock, McpEmbeddedResource};

/// Invoke a tool against the daemon for `(thread_id, binding_name)`.
///
/// Lookup-or-open semantics:
/// 1. If [`V2SessionStore::get`] returns a handle, use it directly.
/// 2. Otherwise read the current `ThreadContext` from
///    [`V2ContextStore`] (defaulting on miss) and call
///    [`LiveDaemonHandle::open_session`] with it. The daemon then
///    enforces context fields (denylist, runas, etc.) per phase 5a.
/// 3. [`V2SessionStore::insert_or_keep`] races safely with concurrent
///    opens for the same key — the loser's handle drops here, fires
///    `CloseSession`, and the winner's handle is returned to all
///    callers.
///
/// Disconnect handling is governed by `policy`:
/// - [`V2DisconnectPolicy::ContinueWithWarning`] returns the
///   `LinkError::Disconnected` immediately as a tool error.
/// - [`V2DisconnectPolicy::PauseUntilReconnect`] drops the stale
///   session, awaits a fresh handle from the registry via
///   [`LiveDaemonRegistry::wait_for_connection`], then loops to retry
///   open + invoke transparently. The cancel token is the escape
///   hatch — drop it (or the thread cancels) to break the wait.
///
/// Cancellation is awaited alongside dispatch via `select!`. On cancel
/// the in-flight future is dropped (the daemon-side call may continue
/// and emit a ToolFinal for it — the scheduler discards the result
/// for a Cancelled thread anyway).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn dispatch_v2_tool(
    sessions: V2SessionStore,
    contexts: V2ContextStore,
    registry: Arc<LiveDaemonRegistry>,
    initial_handle: Arc<LiveDaemonHandle>,
    policy: V2DisconnectPolicy,
    thread_id: String,
    binding_name: String,
    spec: HostEnvSpec,
    real_name: String,
    arguments: Value,
    cancel: CancellationToken,
) -> Result<CallToolResult, String> {
    let mut handle = initial_handle;
    loop {
        let session = match sessions.get(&thread_id, &binding_name) {
            Some(s) => s,
            None => {
                let context = contexts.get_or_default(&thread_id, &binding_name);
                let open = handle.open_session(thread_id.clone(), spec.clone(), context);
                let opened = tokio::select! {
                    r = open => r,
                    _ = cancel.cancelled() => return Err("cancelled".to_string()),
                };
                match opened {
                    Ok(o) => sessions.insert_or_keep(
                        thread_id.clone(),
                        binding_name.clone(),
                        Arc::new(o),
                    ),
                    Err(LinkError::Disconnected(name))
                        if policy == V2DisconnectPolicy::PauseUntilReconnect =>
                    {
                        match wait_for_reconnect(&registry, &name, &cancel).await {
                            Some(fresh) => {
                                handle = fresh;
                                continue;
                            }
                            None => return Err("cancelled".to_string()),
                        }
                    }
                    Err(e) => return Err(open_session_error_msg(e)),
                }
            }
        };

        let invoke = session.invoke_tool(real_name.clone(), arguments.clone(), vec![]);
        let result = tokio::select! {
            r = invoke => r,
            _ = cancel.cancelled() => return Err("cancelled".to_string()),
        };

        match result {
            Ok(host_result) => return Ok(translate_call_result(host_result)),
            Err(LinkError::Disconnected(name))
                if policy == V2DisconnectPolicy::PauseUntilReconnect =>
            {
                // Drop the stale session before swapping in a fresh
                // handle: its cmd_tx is dead, and leaving it in the
                // store would let a parallel dispatch race on the
                // same broken handle.
                drop(session);
                sessions.remove_session(&thread_id, &binding_name);
                match wait_for_reconnect(&registry, &name, &cancel).await {
                    Some(fresh) => {
                        handle = fresh;
                        // Loop iterates: next pass opens a fresh
                        // session via `handle` and re-invokes.
                    }
                    None => return Err("cancelled".to_string()),
                }
            }
            Err(e) => return Err(invoke_error_msg(e)),
        }
    }
}

/// Wait for `name` to come back online, escaping on cancel. Returns
/// `Some(handle)` on reconnect or `None` if cancelled. Pulled out of
/// the dispatcher so the same select-on-cancel shape covers both the
/// open-session and invoke disconnect paths.
async fn wait_for_reconnect(
    registry: &LiveDaemonRegistry,
    name: &str,
    cancel: &CancellationToken,
) -> Option<Arc<LiveDaemonHandle>> {
    tokio::select! {
        h = registry.wait_for_connection(name) => Some(h),
        _ = cancel.cancelled() => None,
    }
}

/// Translate a `LinkError` from the open-session phase into the
/// scheduler-side string an MCP-call failure would produce. Keeps
/// the dispatcher's error-channel shape consistent so retry / Lost-
/// state heuristics in `io_dispatch` can treat both wires uniformly.
fn open_session_error_msg(err: LinkError) -> String {
    format!("v2 session open failed: {err}")
}

fn invoke_error_msg(err: LinkError) -> String {
    format!("v2 invoke failed: {err}")
}

/// `whisper_agent_host_proto::CallToolResult` → `crate::tools::mcp::CallToolResult`.
/// Same fields, distinct types — the host-proto crate uses snake_case
/// throughout (it's internal CBOR, not the MCP HTTP wire), the mcp
/// types are camelCase-deserialized for HTTP wire compat. The
/// runtime types downstream of the dispatcher only know the mcp
/// shape, so we translate at the boundary.
fn translate_call_result(src: whisper_agent_host_proto::CallToolResult) -> CallToolResult {
    CallToolResult {
        content: src
            .content
            .into_iter()
            .map(translate_content_block)
            .collect(),
        is_error: src.is_error.unwrap_or(false),
    }
}

fn translate_content_block(src: ContentBlock) -> McpContentBlock {
    match src {
        ContentBlock::Text { text } => McpContentBlock::Text { text },
        ContentBlock::Image { data, mime_type } => McpContentBlock::Image { data, mime_type },
        ContentBlock::Resource { resource } => McpContentBlock::Resource {
            resource: McpEmbeddedResource {
                uri: resource.uri,
                mime_type: resource.mime_type,
                text: resource.text,
                blob: resource.blob,
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::Duration;
    use whisper_agent_host_proto::{
        DaemonCapabilities, EmbeddedResource, HostEnvSpecKind, ToolDescriptor,
    };
    use whisper_agent_protocol::sandbox::{NetworkPolicy, PathAccess};

    use crate::tools::host_env_link::test_fixtures::{
        FakeDaemonRig, InvokeOutcome, spawn_fake_daemon,
    };

    #[test]
    fn translate_text_block() {
        let src = ContentBlock::Text { text: "hi".into() };
        let McpContentBlock::Text { text } = translate_content_block(src) else {
            panic!("expected text");
        };
        assert_eq!(text, "hi");
    }

    #[test]
    fn translate_image_block() {
        let src = ContentBlock::Image {
            data: "AAAA".into(),
            mime_type: "image/png".into(),
        };
        let McpContentBlock::Image { data, mime_type } = translate_content_block(src) else {
            panic!("expected image");
        };
        assert_eq!(data, "AAAA");
        assert_eq!(mime_type, "image/png");
    }

    #[test]
    fn translate_resource_blob_block() {
        let src = ContentBlock::Resource {
            resource: EmbeddedResource {
                uri: "file:///tmp/x.pdf".into(),
                mime_type: Some("application/pdf".into()),
                text: None,
                blob: Some("BBBB".into()),
            },
        };
        let McpContentBlock::Resource { resource } = translate_content_block(src) else {
            panic!("expected resource");
        };
        assert_eq!(resource.uri, "file:///tmp/x.pdf");
        assert_eq!(resource.mime_type.as_deref(), Some("application/pdf"));
        assert_eq!(resource.blob.as_deref(), Some("BBBB"));
        assert_eq!(resource.text, None);
    }

    #[test]
    fn translate_call_result_with_error_flag() {
        let src = whisper_agent_host_proto::CallToolResult {
            content: vec![ContentBlock::Text {
                text: "boom".into(),
            }],
            is_error: Some(true),
        };
        let r = translate_call_result(src);
        assert!(r.is_error);
        assert_eq!(r.content.len(), 1);
    }

    #[test]
    fn translate_call_result_default_is_error_false() {
        let src = whisper_agent_host_proto::CallToolResult {
            content: vec![],
            is_error: None,
        };
        let r = translate_call_result(src);
        assert!(!r.is_error);
    }

    fn sample_capabilities() -> DaemonCapabilities {
        DaemonCapabilities {
            tools: vec![ToolDescriptor {
                name: "echo".into(),
                description: "echo a message".into(),
                input_schema: serde_json::json!({"type":"object"}),
                annotations: Default::default(),
            }],
            spec_kinds: vec![HostEnvSpecKind::Landlock],
            max_concurrent_sessions: None,
            supports_background_tasks: false,
        }
    }

    fn sample_spec() -> HostEnvSpec {
        HostEnvSpec::Landlock {
            allowed_paths: vec![PathAccess::read_write("/tmp/v2-dispatch-test")],
            network: NetworkPolicy::Isolated,
        }
    }

    fn dispatch_args() -> Value {
        serde_json::json!({"msg":"hi"})
    }

    /// Drop the rig's handle and wait for the fake-daemon task to
    /// exit. Once the last `Arc<LiveDaemonHandle>` is dropped (and any
    /// derived `SessionHandle`s — they each clone the cmd_tx), the
    /// command channel closes and the task's `recv` returns `None`.
    async fn shutdown_and_join(rig: FakeDaemonRig, sessions: V2SessionStore, thread_ids: &[&str]) {
        for tid in thread_ids {
            sessions.remove_thread(tid);
        }
        drop(sessions);
        drop(rig.handle);
        tokio::time::timeout(Duration::from_secs(2), rig.task)
            .await
            .expect("fake daemon task hung")
            .expect("fake daemon task panicked");
    }

    #[tokio::test]
    async fn dispatch_opens_session_and_returns_translated_result() {
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |tool, args| {
            assert_eq!(tool, "echo");
            assert_eq!(args, &dispatch_args());
            InvokeOutcome::Reply(Ok(whisper_agent_host_proto::CallToolResult {
                content: vec![ContentBlock::Text {
                    text: "echoed: hi".into(),
                }],
                is_error: None,
            }))
        });
        let sessions = V2SessionStore::default();

        let result = dispatch_v2_tool(
            sessions.clone(),
            V2ContextStore::default(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            CancellationToken::new(),
        )
        .await
        .expect("dispatch failed");

        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
        let McpContentBlock::Text { text } = &result.content[0] else {
            panic!("expected text content");
        };
        assert_eq!(text, "echoed: hi");
        assert_eq!(rig.state.opens.load(Ordering::Relaxed), 1);
        assert_eq!(rig.state.invokes.load(Ordering::Relaxed), 1);

        shutdown_and_join(rig, sessions, &["thread-1"]).await;
    }

    #[tokio::test]
    async fn dispatch_reuses_session_within_same_thread_binding() {
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_tool, _args| {
            InvokeOutcome::Reply(Ok(whisper_agent_host_proto::CallToolResult {
                content: vec![],
                is_error: None,
            }))
        });
        let sessions = V2SessionStore::default();

        for _ in 0..3 {
            dispatch_v2_tool(
                sessions.clone(),
                V2ContextStore::default(),
                Arc::new(LiveDaemonRegistry::new()),
                rig.handle.clone(),
                V2DisconnectPolicy::ContinueWithWarning,
                "thread-1".into(),
                "alpha".into(),
                sample_spec(),
                "echo".into(),
                dispatch_args(),
                CancellationToken::new(),
            )
            .await
            .expect("dispatch failed");
        }

        assert_eq!(
            rig.state.opens.load(Ordering::Relaxed),
            1,
            "session should be opened once and reused"
        );
        assert_eq!(rig.state.invokes.load(Ordering::Relaxed), 3);

        shutdown_and_join(rig, sessions, &["thread-1"]).await;
    }

    #[tokio::test]
    async fn dispatch_opens_distinct_sessions_for_distinct_threads() {
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_tool, _args| {
            InvokeOutcome::Reply(Ok(whisper_agent_host_proto::CallToolResult {
                content: vec![],
                is_error: None,
            }))
        });
        let sessions = V2SessionStore::default();

        for tid in ["thread-1", "thread-2"] {
            dispatch_v2_tool(
                sessions.clone(),
                V2ContextStore::default(),
                Arc::new(LiveDaemonRegistry::new()),
                rig.handle.clone(),
                V2DisconnectPolicy::ContinueWithWarning,
                tid.into(),
                "alpha".into(),
                sample_spec(),
                "echo".into(),
                dispatch_args(),
                CancellationToken::new(),
            )
            .await
            .expect("dispatch failed");
        }

        assert_eq!(rig.state.opens.load(Ordering::Relaxed), 2);
        assert_eq!(rig.state.invokes.load(Ordering::Relaxed), 2);

        shutdown_and_join(rig, sessions, &["thread-1", "thread-2"]).await;
    }

    #[tokio::test]
    async fn dispatch_remove_thread_drops_session_and_fires_close() {
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_tool, _args| {
            InvokeOutcome::Reply(Ok(whisper_agent_host_proto::CallToolResult {
                content: vec![],
                is_error: None,
            }))
        });
        let sessions = V2SessionStore::default();

        dispatch_v2_tool(
            sessions.clone(),
            V2ContextStore::default(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            CancellationToken::new(),
        )
        .await
        .expect("dispatch failed");
        assert_eq!(rig.state.opens.load(Ordering::Relaxed), 1);

        // Sweep the thread; SessionHandle::Drop should fire CloseSession.
        sessions.remove_thread("thread-1");
        // CloseSession is fire-and-forget; give the task a beat to receive it.
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if rig.state.closes.load(Ordering::Relaxed) >= 1 {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("CloseSession never delivered");

        // Next dispatch on the same key opens a fresh session.
        dispatch_v2_tool(
            sessions.clone(),
            V2ContextStore::default(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            CancellationToken::new(),
        )
        .await
        .expect("dispatch failed");
        assert_eq!(rig.state.opens.load(Ordering::Relaxed), 2);

        shutdown_and_join(rig, sessions, &["thread-1"]).await;
    }

    #[tokio::test]
    async fn dispatch_returns_cancelled_when_token_fires_mid_invoke() {
        let rig = spawn_fake_daemon(
            "alpha".into(),
            sample_capabilities(),
            // Stash the InvokeTool oneshot so the dispatch future stays pending.
            |_tool, _args| InvokeOutcome::Stash,
        );
        let sessions = V2SessionStore::default();
        let cancel = CancellationToken::new();

        let cancel_for_task = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel_for_task.cancel();
        });

        let err = dispatch_v2_tool(
            sessions.clone(),
            V2ContextStore::default(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            cancel,
        )
        .await
        .expect_err("expected cancellation error");
        assert_eq!(err, "cancelled");

        // Drain the stashed senders so the receiver side doesn't keep
        // them alive past the rig task's lifetime.
        rig.state.pending_invokes.lock().await.clear();
        shutdown_and_join(rig, sessions, &["thread-1"]).await;
    }

    #[tokio::test]
    async fn dispatch_sends_configured_context_at_open_session() {
        use std::collections::BTreeSet;
        use whisper_agent_host_proto::ThreadContext;
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_tool, _args| {
            InvokeOutcome::Reply(Ok(whisper_agent_host_proto::CallToolResult {
                content: vec![],
                is_error: None,
            }))
        });
        let sessions = V2SessionStore::default();
        let contexts = V2ContextStore::default();
        let configured = ThreadContext {
            runas: Some("worker".into()),
            tool_denylist: BTreeSet::from(["bash".into()]),
            ..ThreadContext::default()
        };
        contexts.set("thread-1".into(), "alpha".into(), configured.clone());

        dispatch_v2_tool(
            sessions.clone(),
            contexts.clone(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            CancellationToken::new(),
        )
        .await
        .expect("dispatch failed");

        let opened = rig.state.opened_contexts.lock().await;
        assert_eq!(opened.len(), 1);
        assert_eq!(opened[0], configured);

        // A second dispatch on the same key should reuse the session
        // — and thus NOT re-open with a context. Verify only one
        // OpenSession was emitted.
        drop(opened);
        dispatch_v2_tool(
            sessions.clone(),
            contexts.clone(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            CancellationToken::new(),
        )
        .await
        .expect("dispatch failed");
        assert_eq!(rig.state.opened_contexts.lock().await.len(), 1);

        shutdown_and_join(rig, sessions, &["thread-1"]).await;
    }

    #[tokio::test]
    async fn dispatch_uses_default_context_when_unset() {
        use whisper_agent_host_proto::ThreadContext;
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_tool, _args| {
            InvokeOutcome::Reply(Ok(whisper_agent_host_proto::CallToolResult {
                content: vec![],
                is_error: None,
            }))
        });
        let sessions = V2SessionStore::default();

        dispatch_v2_tool(
            sessions.clone(),
            V2ContextStore::default(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            CancellationToken::new(),
        )
        .await
        .expect("dispatch failed");

        let opened = rig.state.opened_contexts.lock().await;
        assert_eq!(opened.len(), 1);
        assert_eq!(opened[0], ThreadContext::default());
        drop(opened);

        shutdown_and_join(rig, sessions, &["thread-1"]).await;
    }

    #[tokio::test]
    async fn session_handle_update_context_reaches_daemon() {
        use std::collections::BTreeSet;
        use whisper_agent_host_proto::ThreadContext;
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_tool, _args| {
            InvokeOutcome::Reply(Ok(whisper_agent_host_proto::CallToolResult {
                content: vec![],
                is_error: None,
            }))
        });
        let sessions = V2SessionStore::default();

        // Open a session via dispatch_v2_tool so we have a SessionHandle
        // in the store.
        dispatch_v2_tool(
            sessions.clone(),
            V2ContextStore::default(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            CancellationToken::new(),
        )
        .await
        .expect("dispatch failed");

        // Pull the session handle back out and push an UpdateSession.
        let session = sessions.get("thread-1", "alpha").expect("session missing");
        let delta = ThreadContext::default().diff_to(&ThreadContext {
            tool_denylist: BTreeSet::from(["bash".into()]),
            ..ThreadContext::default()
        });
        session
            .update_context(delta.clone())
            .expect("update failed");
        drop(session);

        // Wire send is fire-and-forget; spin briefly until the fake
        // daemon records it.
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if !rig.state.updates.lock().await.is_empty() {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("UpdateSession never delivered");

        let updates = rig.state.updates.lock().await;
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0], delta);
        drop(updates);

        shutdown_and_join(rig, sessions, &["thread-1"]).await;
    }

    #[tokio::test]
    async fn session_handle_update_context_after_disconnect_returns_disconnected() {
        use whisper_agent_host_proto::ThreadContextDelta;
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_tool, _args| {
            InvokeOutcome::Reply(Ok(whisper_agent_host_proto::CallToolResult {
                content: vec![],
                is_error: None,
            }))
        });
        let sessions = V2SessionStore::default();

        dispatch_v2_tool(
            sessions.clone(),
            V2ContextStore::default(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            CancellationToken::new(),
        )
        .await
        .expect("dispatch failed");

        let session = sessions.get("thread-1", "alpha").expect("session missing");

        // Tear the daemon side down by aborting the fake-daemon task —
        // that drops the receiver end of the command channel, which
        // makes try_send return Err. We can't wait for the task to
        // exit by dropping all senders, because `session` itself holds
        // a sender clone (and we still need it to call update_context).
        rig.task.abort();
        let _ = rig.task.await; // surfaces JoinError::Cancelled, fine

        let err = session
            .update_context(ThreadContextDelta::default())
            .expect_err("expected disconnected");
        assert!(
            matches!(err, crate::tools::host_env_link::LinkError::Disconnected(ref n) if n == "alpha"),
            "unexpected error: {err}",
        );
        drop(session);
        drop(rig.handle);
        sessions.remove_thread("thread-1");
    }

    #[tokio::test]
    async fn context_store_remove_thread_drops_only_that_thread() {
        use whisper_agent_host_proto::ThreadContext;
        let store = V2ContextStore::default();
        let ctx_a = ThreadContext {
            runas: Some("a".into()),
            ..ThreadContext::default()
        };
        let ctx_b = ThreadContext {
            runas: Some("b".into()),
            ..ThreadContext::default()
        };
        store.set("thread-1".into(), "alpha".into(), ctx_a.clone());
        store.set("thread-2".into(), "alpha".into(), ctx_b.clone());

        store.remove_thread("thread-1");
        assert_eq!(
            store.get_or_default("thread-1", "alpha"),
            ThreadContext::default(),
            "thread-1's entry should be cleared"
        );
        assert_eq!(
            store.get_or_default("thread-2", "alpha"),
            ctx_b,
            "thread-2's entry must survive"
        );
    }

    #[tokio::test]
    async fn dispatch_propagates_invoke_failure() {
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_tool, _args| {
            InvokeOutcome::Reply(Err("backend exploded".into()))
        });
        let sessions = V2SessionStore::default();

        let err = dispatch_v2_tool(
            sessions.clone(),
            V2ContextStore::default(),
            Arc::new(LiveDaemonRegistry::new()),
            rig.handle.clone(),
            V2DisconnectPolicy::ContinueWithWarning,
            "thread-1".into(),
            "alpha".into(),
            sample_spec(),
            "echo".into(),
            dispatch_args(),
            CancellationToken::new(),
        )
        .await
        .expect_err("expected invoke error");
        assert!(
            err.contains("v2 invoke failed") && err.contains("backend exploded"),
            "unexpected error message: {err}",
        );

        shutdown_and_join(rig, sessions, &["thread-1"]).await;
    }

    #[tokio::test]
    async fn pause_until_reconnect_retries_after_daemon_swap() {
        // Wire path under test: invoke against handle1 fails with
        // Disconnected → dispatcher waits for "alpha" to reappear
        // in the registry → handle2 inserted → dispatcher opens a
        // fresh session against handle2 and the retry succeeds.
        let registry = Arc::new(LiveDaemonRegistry::new());
        let rig1 = spawn_fake_daemon(
            "alpha".into(),
            sample_capabilities(),
            // Stash invokes so the call hangs until we abort the rig.
            |_, _| InvokeOutcome::Stash,
        );
        registry.insert_for_test("alpha".into(), rig1.handle.clone());
        let rig2 = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_, _| {
            InvokeOutcome::Reply(Ok(whisper_agent_host_proto::CallToolResult {
                content: vec![ContentBlock::Text {
                    text: "from-rig2".into(),
                }],
                is_error: None,
            }))
        });

        let sessions = V2SessionStore::default();
        let cancel = CancellationToken::new();

        let dispatch = tokio::spawn({
            let sessions = sessions.clone();
            let registry = registry.clone();
            let handle1 = rig1.handle.clone();
            let cancel = cancel.clone();
            async move {
                dispatch_v2_tool(
                    sessions,
                    V2ContextStore::default(),
                    registry,
                    handle1,
                    V2DisconnectPolicy::PauseUntilReconnect,
                    "thread-1".into(),
                    "alpha".into(),
                    sample_spec(),
                    "echo".into(),
                    dispatch_args(),
                    cancel,
                )
                .await
            }
        });

        // Wait until rig1 has stashed the invoke (proves dispatch_v2_tool
        // got past open_session and is awaiting the result oneshot).
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if rig1.state.invokes.load(Ordering::Relaxed) >= 1 {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("rig1 never received the invoke");

        // Simulate disconnect: abort rig1's task and drain any stashed
        // result oneshots from FakeDaemonState. The state struct is
        // Arc'd, so abort alone doesn't drop the pending_invokes Vec
        // — the test has to drain it explicitly to surface
        // LinkError::Disconnected to invoke_tool's awaiting result_rx.
        registry.remove_for_test("alpha");
        rig1.task.abort();
        rig1.state.pending_invokes.lock().await.clear();

        // Reconnect: register rig2 under the same name. The
        // dispatcher's wait_for_reconnect resolves to rig2's handle
        // and the retry opens a fresh session against it.
        registry.insert_for_test("alpha".into(), rig2.handle.clone());

        let result = tokio::time::timeout(Duration::from_secs(5), dispatch)
            .await
            .expect("dispatch hung after reconnect")
            .expect("dispatch task panicked")
            .expect("dispatch returned error");
        assert_eq!(result.content.len(), 1);
        let McpContentBlock::Text { text } = &result.content[0] else {
            panic!("expected text content");
        };
        assert_eq!(text, "from-rig2");
        assert_eq!(rig2.state.opens.load(Ordering::Relaxed), 1);
        assert_eq!(rig2.state.invokes.load(Ordering::Relaxed), 1);

        cancel.cancel();
        // Pull the rig2 handle clone the registry is still holding;
        // otherwise the cmd_tx stays alive and the daemon task can't
        // exit when shutdown_and_join drops the rig's own handle.
        registry.remove_for_test("alpha");
        drop(registry);
        shutdown_and_join(rig2, sessions, &["thread-1"]).await;
        // rig1 was aborted; just drop the handle.
        drop(rig1.handle);
    }

    #[tokio::test]
    async fn continue_with_warning_returns_error_immediately_on_disconnect() {
        let registry = Arc::new(LiveDaemonRegistry::new());
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_, _| {
            InvokeOutcome::Stash
        });
        registry.insert_for_test("alpha".into(), rig.handle.clone());

        let sessions = V2SessionStore::default();
        let cancel = CancellationToken::new();

        let dispatch = tokio::spawn({
            let sessions = sessions.clone();
            let registry = registry.clone();
            let handle = rig.handle.clone();
            let cancel = cancel.clone();
            async move {
                dispatch_v2_tool(
                    sessions,
                    V2ContextStore::default(),
                    registry,
                    handle,
                    V2DisconnectPolicy::ContinueWithWarning,
                    "thread-1".into(),
                    "alpha".into(),
                    sample_spec(),
                    "echo".into(),
                    dispatch_args(),
                    cancel,
                )
                .await
            }
        });

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if rig.state.invokes.load(Ordering::Relaxed) >= 1 {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("rig never received the invoke");

        rig.task.abort();
        // Drain stashed oneshots — see pause_until_reconnect test for
        // the why. Without this the result_rx in invoke_tool stays
        // awaiting forever even after the task aborts.
        rig.state.pending_invokes.lock().await.clear();
        // No new daemon registered — ContinueWithWarning shouldn't
        // wait for one, it just returns the error.

        let err = tokio::time::timeout(Duration::from_secs(2), dispatch)
            .await
            .expect("dispatch hung")
            .expect("dispatch panicked")
            .expect_err("expected error");
        assert!(
            err.contains("v2 invoke failed") && err.contains("disconnect"),
            "unexpected error: {err}",
        );

        cancel.cancel();
        sessions.remove_thread("thread-1");
        drop(rig.handle);
    }

    #[tokio::test]
    async fn pause_until_reconnect_cancellable_during_wait() {
        let registry = Arc::new(LiveDaemonRegistry::new());
        let rig = spawn_fake_daemon("alpha".into(), sample_capabilities(), |_, _| {
            InvokeOutcome::Stash
        });
        registry.insert_for_test("alpha".into(), rig.handle.clone());

        let sessions = V2SessionStore::default();
        let cancel = CancellationToken::new();

        let dispatch = tokio::spawn({
            let sessions = sessions.clone();
            let registry = registry.clone();
            let handle = rig.handle.clone();
            let cancel = cancel.clone();
            async move {
                dispatch_v2_tool(
                    sessions,
                    V2ContextStore::default(),
                    registry,
                    handle,
                    V2DisconnectPolicy::PauseUntilReconnect,
                    "thread-1".into(),
                    "alpha".into(),
                    sample_spec(),
                    "echo".into(),
                    dispatch_args(),
                    cancel,
                )
                .await
            }
        });

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if rig.state.invokes.load(Ordering::Relaxed) >= 1 {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("rig never received the invoke");

        // Disconnect → dispatcher enters wait_for_reconnect (no daemon
        // will reappear). Cancel the token to escape.
        registry.remove_for_test("alpha");
        rig.task.abort();
        rig.state.pending_invokes.lock().await.clear();
        // Give the dispatcher a beat to enter wait_for_reconnect.
        tokio::task::yield_now().await;
        cancel.cancel();

        let err = tokio::time::timeout(Duration::from_secs(2), dispatch)
            .await
            .expect("dispatch hung after cancel")
            .expect("dispatch panicked")
            .expect_err("expected cancelled error");
        assert_eq!(err, "cancelled");

        sessions.remove_thread("thread-1");
        drop(rig.handle);
    }

    #[tokio::test]
    async fn policy_store_default_on_miss_and_set_get_round_trip() {
        let store = crate::runtime::scheduler::V2PolicyStore::new();
        assert_eq!(
            store.get("never-set"),
            V2DisconnectPolicy::ContinueWithWarning
        );
        let prior = store.set("t1".into(), V2DisconnectPolicy::PauseUntilReconnect);
        assert!(prior.is_none());
        assert_eq!(store.get("t1"), V2DisconnectPolicy::PauseUntilReconnect);
        store.remove_thread("t1");
        assert_eq!(store.get("t1"), V2DisconnectPolicy::ContinueWithWarning);
    }
}
