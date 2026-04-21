//! `request_escalation` — the model asks its owning interactive channel
//! for a specific widening of its scope.
//!
//! The tool is intercepted at the scheduler layer (same shape as
//! `dispatch_thread`) because it needs to register a Function and park
//! the tool call on its terminal. This module owns the descriptor and
//! argument parsing only.
//!
//! Visibility: the scheduler's `tool_descriptors` filters this tool out
//! of the catalog when the thread's `scope.escalation` is `None`, so a
//! model running on an autonomous thread cannot invoke it. See
//! `docs/design_permissions_rework.md`.

use serde::Deserialize;
use serde_json::{Value, json};
use whisper_agent_protocol::permission::EscalationRequest;

use super::REQUEST_ESCALATION;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: REQUEST_ESCALATION.into(),
        description: "Ask the user to widen this thread's scope. Use when you \
                      hit a capability you need but don't currently have — a \
                      specific tool, a higher pod-modify cap, a higher \
                      behaviors cap, or dispatch. The approval is \
                      scope-widening, not per-call: an approved `add_tool` \
                      admits the tool for the rest of this thread's life. \
                      The tool parks your turn until the user decides. On \
                      approve, the widened capability is immediately \
                      available. On reject, the user's reason (if supplied) \
                      is returned; course-correct accordingly. Editing the \
                      pod's `[allow]` to widen *future* threads is a \
                      separate `pod_write_file` on `pod.toml` — not this \
                      tool."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "request": {
                    "type": "object",
                    "description": "Typed widening request. One of: \
                        {\"variant\":\"add_tool\",\"name\":\"<tool>\"} — \
                        admit the named tool; \
                        {\"variant\":\"raise_pod_modify\",\"target\":\"memories|content|modify_allow\"} — \
                        raise PodModifyCap; \
                        {\"variant\":\"raise_behaviors\",\"target\":\"read|author_narrower|author_any\"} — \
                        raise BehaviorOpsCap; \
                        {\"variant\":\"raise_dispatch\",\"target\":\"within_scope\"} — \
                        raise DispatchCap. Target must be strictly higher \
                        than the current cap and bounded by the pod's ceiling."
                },
                "reason": {
                    "type": "string",
                    "description": "Short justification shown to the user in \
                                    the approval UI. Helps them decide."
                }
            },
            "required": ["request", "reason"]
        }),
        annotations: ToolAnnotations {
            title: Some("Request scope widening".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
        },
    }
}

/// Arguments parsed from a `request_escalation` tool call. Consumed by
/// the scheduler's `register_request_escalation_tool` intercept.
#[derive(Debug, Clone, Deserialize)]
pub struct RequestEscalationArgs {
    pub request: EscalationRequest,
    pub reason: String,
}

pub fn parse_args(value: Value) -> Result<RequestEscalationArgs, String> {
    serde_json::from_value::<RequestEscalationArgs>(value)
        .map_err(|e| format!("invalid request_escalation arguments: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::permission::{BehaviorOpsCap, DispatchCap, PodModifyCap};

    #[test]
    fn parse_add_tool() {
        let v = json!({
            "request": { "variant": "add_tool", "name": "bash" },
            "reason": "need the test suite"
        });
        let args = parse_args(v).unwrap();
        assert!(matches!(
            args.request,
            EscalationRequest::AddTool { ref name } if name == "bash"
        ));
    }

    #[test]
    fn parse_raise_pod_modify() {
        let v = json!({
            "request": { "variant": "raise_pod_modify", "target": "content" },
            "reason": "edit system_prompt.md"
        });
        let args = parse_args(v).unwrap();
        assert!(matches!(
            args.request,
            EscalationRequest::RaisePodModify {
                target: PodModifyCap::Content
            }
        ));
    }

    #[test]
    fn parse_raise_behaviors() {
        let v = json!({
            "request": { "variant": "raise_behaviors", "target": "author_any" },
            "reason": "author a cron behavior"
        });
        let args = parse_args(v).unwrap();
        assert!(matches!(
            args.request,
            EscalationRequest::RaiseBehaviors {
                target: BehaviorOpsCap::AuthorAny
            }
        ));
    }

    #[test]
    fn parse_raise_dispatch() {
        let v = json!({
            "request": { "variant": "raise_dispatch", "target": "within_scope" },
            "reason": "split this into sub-threads"
        });
        let args = parse_args(v).unwrap();
        assert!(matches!(
            args.request,
            EscalationRequest::RaiseDispatch {
                target: DispatchCap::WithinScope
            }
        ));
    }

    #[test]
    fn parse_rejects_missing_fields() {
        assert!(parse_args(json!({ "reason": "no request" })).is_err());
        assert!(parse_args(json!({ "request": { "variant": "add_tool" } })).is_err());
    }

    #[test]
    fn parse_rejects_unknown_variant() {
        let v = json!({
            "request": { "variant": "nope" },
            "reason": "unknown"
        });
        assert!(parse_args(v).is_err());
    }
}
