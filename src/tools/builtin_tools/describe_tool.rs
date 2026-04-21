//! `describe_tool` — synchronous scheduler intercept that returns the
//! full JSON schema for a named tool.
//!
//! Pairs with [`crate::tools::builtin_tools::find_tool::descriptor`]:
//! `find_tool` is for "what tools exist?", `describe_tool` is for
//! "what does this one take?". The model typically learns names from
//! the system-prompt listing or a mid-conversation activation
//! message, then fetches schemas on demand — so most threads only
//! ever load schemas for tools they actually call.
//!
//! Scope semantics: we admit a tool for `describe_tool` if the thread
//! has it in-scope OR if the pod ceiling admits it. Askable tools are
//! described so the model can decide whether requesting an escalation
//! is worth it; truly out-of-reach tools return a deny error.

use serde::Deserialize;
use serde_json::{Value, json};

use super::DESCRIBE_TOOL;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: DESCRIBE_TOOL.into(),
        description: "Return the full JSON schema for a named tool. Use after \
                      seeing a tool name in the catalog listing or an activation \
                      message. The result is the input-schema the tool expects, \
                      plus its annotations — ready to call directly. Works for \
                      tools currently in your scope AND for tools available via \
                      escalation (the schema is readable either way; calling \
                      requires `request_escalation` first for the latter)."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Wire-level tool name. For host-env MCP tools this \
                                    includes the binding prefix (e.g., `rustdev_bash`)."
                }
            },
            "required": ["name"]
        }),
        annotations: ToolAnnotations {
            title: Some("Describe tool".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DescribeToolArgs {
    pub name: String,
}

pub fn parse_args(value: Value) -> Result<DescribeToolArgs, String> {
    serde_json::from_value::<DescribeToolArgs>(value)
        .map_err(|e| format!("invalid describe_tool arguments: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let v = json!({ "name": "rustdev_bash" });
        let args = parse_args(v).unwrap();
        assert_eq!(args.name, "rustdev_bash");
    }

    #[test]
    fn parse_rejects_missing_name() {
        assert!(parse_args(json!({})).is_err());
    }

    #[test]
    fn parse_rejects_wrong_type() {
        assert!(parse_args(json!({ "name": 42 })).is_err());
    }
}
