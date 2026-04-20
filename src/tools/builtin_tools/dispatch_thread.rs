//! `dispatch_thread` — spawn a child thread that runs a sub-task and
//! (optionally) returns its final message to the parent as this tool's
//! result.
//!
//! Unlike every other builtin, the handler can't run inside
//! [`super::dispatch`] — it needs live scheduler state to spawn the
//! child and register the result-return plumbing. The scheduler's
//! `io_dispatch::tool_call` intercepts the name before the generic
//! dispatch path and handles it directly; this module exists only to
//! own the tool's descriptor and argument shape.

use serde::Deserialize;
use serde_json::{Value, json};
use whisper_agent_protocol::{ThreadBindingsRequest, ThreadConfigOverride};

use super::DISPATCH_THREAD;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: DISPATCH_THREAD.into(),
        description: "Dispatch a sub-task as a child thread. The child is a \
                      full-fledged thread in this same pod — it has its own \
                      conversation, tool surface, and state — spawned with \
                      `prompt` as its first user message. `sync=true` parks \
                      this tool call until the child's terminal state and \
                      returns its final assistant text as the tool result. \
                      `sync=false` returns immediately with the child's id; \
                      the child runs in parallel while you continue, and \
                      when it terminates its final assistant text is \
                      delivered to you as a fresh user message on a new \
                      turn (bracketed with `[dispatched thread … \
                      completed]`). Use this to delegate bounded, well- \
                      scoped sub-work (research passes, exploratory \
                      searches, one-shot generators) so the main thread's \
                      context stays focused. `config_override` and \
                      `bindings_override` are the same shape the \
                      `create_thread` wire command accepts — omit them to \
                      inherit this thread's pod defaults."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "First user message seeded into the child thread."
                },
                "sync": {
                    "type": "boolean",
                    "description": "True: park this tool call until the child terminates \
                                    and return its final assistant text as the tool \
                                    result. False: return immediately with the child's \
                                    id; the child's final text arrives later as a fresh \
                                    user message on a new turn."
                },
                "config_override": {
                    "type": "object",
                    "description": "Optional ThreadConfigOverride. Shape mirrors \
                                    the wire type: {model?, max_tokens?, max_turns?, \
                                    system_prompt?, compaction?}. Omitted fields \
                                    inherit from pod defaults. The `system_prompt` \
                                    field is `{kind: \"file\", name: \"<pod-relative \
                                    filename>\"}` to read the prompt from a file in \
                                    the pod directory, or `{kind: \"text\", text: \
                                    \"<literal prompt>\"}` to inline it. Omitted \
                                    inherits the pod's `system_prompt.md`."
                },
                "bindings_override": {
                    "type": "object",
                    "description": "Optional ThreadBindingsRequest. Shape: \
                                    {backend?, host_env?, mcp_hosts?}. Omitted \
                                    fields inherit from pod defaults."
                }
            },
            "required": ["prompt", "sync"]
        }),
        annotations: ToolAnnotations {
            title: Some("Dispatch a sub-task".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
        },
    }
}

/// Arguments parsed from a `dispatch_thread` tool call. Used by the
/// scheduler's `io_dispatch::tool_call` intercept.
#[derive(Debug, Clone, Deserialize)]
pub struct DispatchThreadArgs {
    pub prompt: String,
    pub sync: bool,
    #[serde(default)]
    pub config_override: Option<ThreadConfigOverride>,
    #[serde(default)]
    pub bindings_override: Option<ThreadBindingsRequest>,
}

pub fn parse_args(value: Value) -> Result<DispatchThreadArgs, String> {
    serde_json::from_value::<DispatchThreadArgs>(value)
        .map_err(|e| format!("invalid dispatch_thread arguments: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_args() {
        let v = json!({ "prompt": "do the thing", "sync": true });
        let args = parse_args(v).unwrap();
        assert_eq!(args.prompt, "do the thing");
        assert!(args.sync);
        assert!(args.config_override.is_none());
        assert!(args.bindings_override.is_none());
    }

    #[test]
    fn parse_rejects_missing_fields() {
        let v = json!({ "prompt": "no sync flag" });
        assert!(parse_args(v).is_err());
    }

    #[test]
    fn parse_accepts_overrides() {
        let v = json!({
            "prompt": "explore",
            "sync": false,
            "config_override": { "max_tokens": 4096 },
            "bindings_override": { "backend": "anthropic" }
        });
        let args = parse_args(v).unwrap();
        assert!(!args.sync);
        assert_eq!(args.config_override.and_then(|c| c.max_tokens), Some(4096));
        assert_eq!(
            args.bindings_override.and_then(|b| b.backend).as_deref(),
            Some("anthropic")
        );
    }
}
