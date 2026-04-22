//! `sudo` — the model asks its interactive approver to run a named
//! tool with user approval, bypassing the thread's current scope for
//! that one invocation (within the pod's ceiling).
//!
//! Like `dispatch_thread`, `sudo` is intercepted at the scheduler
//! layer (`register_sudo_tool`) so it can register a Function, emit
//! the wire approval request, and park the model's tool call on the
//! Function's terminal.
//!
//! Visibility: the scheduler's tool-listing filters this tool out of
//! the catalog when the thread has no interactive approver. Headless
//! threads cannot run sudo.
//!
//! The approver gets a three-way decision (see
//! `whisper_agent_protocol::permission::SudoDecision`):
//! - `approve_once` — run the wrapped call this one time; scope
//!   stays as-is.
//! - `approve_remember` — run the wrapped call, and additionally
//!   widen `scope.tools` to admit `tool_name` for the remainder of
//!   this thread so subsequent direct calls skip the prompt.
//! - `reject` — the `reason` (if any) rides back to the model as
//!   the tool_result error text.
//!
//! On approve (either variant), the wrapped tool runs with
//! pod-ceiling caps — i.e., `pod_modify`, `dispatch`, and `behaviors`
//! gates are read from `pod.allow.caps` rather than `thread.scope`.
//! This is the per-call cap bypass: remembered approvals only widen
//! `scope.tools`, not caps.

use serde::Deserialize;
use serde_json::{Value, json};

use super::SUDO;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: SUDO.into(),
        description: "Run a tool with explicit user approval. Use when a tool you \
                      need is marked askable (not admitted to your current scope), \
                      OR when a tool is admitted but a specific call would exceed \
                      your current caps (e.g. pod_write_file to a path outside your \
                      pod_modify cap). The user sees the tool name, your inner \
                      args, and your reason, then picks approve-once / \
                      approve-remember / reject. Approve-remember admits the tool \
                      name for the rest of this thread so future direct calls \
                      skip the prompt; caps are bypassed only for this one call \
                      regardless of decision. Reject returns the user's reason as \
                      a tool error — course-correct from there. Autonomous \
                      threads (no interactive approver) cannot use sudo."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to run. Must be present in \
                                    the pod's admissible set (see find_tool / \
                                    describe_tool); sudo cannot call `sudo` \
                                    itself or `dispatch_thread` (both must be \
                                    called directly)."
                },
                "args": {
                    "type": "object",
                    "description": "Arguments for the wrapped tool. Shape must \
                                    match that tool's input_schema — call \
                                    describe_tool first if you haven't used the \
                                    wrapped tool before."
                },
                "reason": {
                    "type": "string",
                    "description": "Short justification shown to the user in the \
                                    approval UI. Tie it to the specific call, \
                                    not a generic explanation."
                }
            },
            "required": ["tool_name", "args", "reason"]
        }),
        annotations: ToolAnnotations {
            title: Some("Sudo (approval-gated call)".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
        },
    }
}

/// Arguments parsed from a `sudo` tool call. Consumed by the
/// scheduler's `register_sudo_tool` intercept.
#[derive(Debug, Clone, Deserialize)]
pub struct SudoArgs {
    pub tool_name: String,
    #[serde(default = "default_args")]
    pub args: Value,
    pub reason: String,
}

fn default_args() -> Value {
    Value::Object(serde_json::Map::new())
}

pub fn parse_args(value: Value) -> Result<SudoArgs, String> {
    let mut parsed: SudoArgs =
        serde_json::from_value(value).map_err(|e| format!("invalid sudo arguments: {e}"))?;
    if parsed.tool_name.trim().is_empty() {
        return Err("sudo: tool_name must not be empty".to_string());
    }
    // Some models double-encode `args`: passing a JSON-encoded string
    // containing the object rather than an inline object. Unwrap when
    // the string parses to an object so the wrapped tool's deserializer
    // sees the structured form instead of failing with "expected
    // struct, got string".
    if let Value::String(s) = &parsed.args
        && let Ok(inner) = serde_json::from_str::<Value>(s)
        && inner.is_object()
    {
        parsed.args = inner;
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ok() {
        let v = json!({
            "tool_name": "pod_write_file",
            "args": { "filename": "system_prompt.md", "content": "..." },
            "reason": "tweak the system prompt"
        });
        let parsed = parse_args(v).unwrap();
        assert_eq!(parsed.tool_name, "pod_write_file");
        assert_eq!(parsed.reason, "tweak the system prompt");
        assert!(parsed.args.is_object());
    }

    #[test]
    fn parse_missing_tool_name_rejected() {
        let v = json!({ "args": {}, "reason": "..." });
        assert!(parse_args(v).is_err());
    }

    #[test]
    fn parse_empty_tool_name_rejected() {
        let v = json!({ "tool_name": "   ", "args": {}, "reason": "..." });
        assert!(parse_args(v).is_err());
    }

    #[test]
    fn parse_missing_reason_rejected() {
        let v = json!({ "tool_name": "x", "args": {} });
        assert!(parse_args(v).is_err());
    }

    #[test]
    fn parse_args_optional_default() {
        // Some tools have no required args; sudo should tolerate `args`
        // being omitted and default to `{}`.
        let v = json!({ "tool_name": "pod_list_files", "reason": "inspect" });
        let parsed = parse_args(v).unwrap();
        assert_eq!(parsed.args, json!({}));
    }

    #[test]
    fn parse_args_double_encoded_string_unwrapped() {
        let v = json!({
            "tool_name": "pod_write_file",
            "args": "{\"filename\": \"x.md\", \"content\": \"hi\"}",
            "reason": "fix"
        });
        let parsed = parse_args(v).unwrap();
        assert_eq!(parsed.args, json!({ "filename": "x.md", "content": "hi" }));
    }

    #[test]
    fn parse_args_non_object_string_left_alone() {
        // A string that parses as a non-object (e.g. a number literal)
        // is not unwrapped — let the wrapped tool's deserializer decide.
        let v = json!({
            "tool_name": "x",
            "args": "42",
            "reason": "r"
        });
        let parsed = parse_args(v).unwrap();
        assert_eq!(parsed.args, json!("42"));
    }

    #[test]
    fn parse_args_garbage_string_left_alone() {
        let v = json!({
            "tool_name": "x",
            "args": "not json {{",
            "reason": "r"
        });
        let parsed = parse_args(v).unwrap();
        assert_eq!(parsed.args, json!("not json {{"));
    }
}
