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
use whisper_agent_host_proto::{ContentBlock, HostEnvSpec, ThreadContext};

use crate::runtime::scheduler::V2SessionStore;
use crate::tools::host_env_link::{LinkError, LiveDaemonHandle};
use crate::tools::mcp::{CallToolResult, McpContentBlock, McpEmbeddedResource};

/// Invoke a tool against the daemon for `(thread_id, binding_name)`.
///
/// Lookup-or-open semantics:
/// 1. If [`V2SessionStore::get`] returns a handle, use it directly.
/// 2. Otherwise call [`LiveDaemonHandle::open_session`] (with the
///    binding's spec + an empty [`ThreadContext`] today; per-thread
///    context lands in phase 5 alongside `UpdateSession`).
/// 3. [`V2SessionStore::insert_or_keep`] races safely with concurrent
///    opens for the same key — the loser's handle drops here, fires
///    `CloseSession`, and the winner's handle is returned to all
///    callers.
///
/// Cancellation is awaited alongside the dispatch via `select!`. On
/// cancel the in-flight `invoke_tool` future is dropped (the channel
/// to the connection task carries no cancellation message yet — the
/// daemon will eventually emit `ToolFinal` with whatever it has, the
/// scheduler discards the result for a Cancelled thread anyway).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn dispatch_v2_tool(
    sessions: V2SessionStore,
    daemon_handle: Arc<LiveDaemonHandle>,
    thread_id: String,
    binding_name: String,
    spec: HostEnvSpec,
    real_name: String,
    arguments: Value,
    cancel: CancellationToken,
) -> Result<CallToolResult, String> {
    let session = match sessions.get(&thread_id, &binding_name) {
        Some(s) => s,
        None => {
            let opened = daemon_handle
                .open_session(thread_id.clone(), spec, ThreadContext::default())
                .await
                .map_err(open_session_error_msg)?;
            sessions.insert_or_keep(thread_id.clone(), binding_name.clone(), Arc::new(opened))
        }
    };

    let invoke = session.invoke_tool(real_name, arguments);
    let result = tokio::select! {
        r = invoke => r,
        _ = cancel.cancelled() => return Err("cancelled".to_string()),
    };

    match result {
        Ok(host_result) => Ok(translate_call_result(host_result)),
        Err(e) => Err(invoke_error_msg(e)),
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
    use whisper_agent_host_proto::EmbeddedResource;

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
}
