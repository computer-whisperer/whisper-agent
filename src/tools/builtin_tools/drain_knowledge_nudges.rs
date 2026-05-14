//! `drain_knowledge_nudges` — explicit model-facing handle for
//! opportunistic knowledge nudges.
//!
//! The scheduler may run background retrieval after a model turn and
//! queue knowledge nudges for the next boundary. Automatic injection
//! can still consume that queue, but this tool lets a model ask for
//! queued material explicitly at a natural tool boundary.

use serde::Deserialize;
use serde_json::{Value, json};

use super::DRAIN_KNOWLEDGE_NUDGES;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: DRAIN_KNOWLEDGE_NUDGES.into(),
        description: "Return and clear any server-generated knowledge nudges queued \
                      for this thread. The server may populate these opportunistically \
                      after reasoning or tool boundaries, for example with material \
                      surfaced by hot knowledge buckets. Call this after a tool call \
                      or before answering when you suspect background context may have \
                      arrived. If nothing is queued, the result says so."
            .into(),
        input_schema: json!({ "type": "object", "properties": {} }),
        annotations: ToolAnnotations {
            title: Some("Drain knowledge nudges".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct DrainKnowledgeNudgesArgs {}

pub fn parse_args(value: Value) -> Result<DrainKnowledgeNudgesArgs, String> {
    serde_json::from_value::<DrainKnowledgeNudgesArgs>(value)
        .map_err(|e| format!("invalid drain_knowledge_nudges arguments: {e}"))
}

pub fn render(nudges: &[String], in_flight: bool) -> String {
    if nudges.is_empty() {
        if in_flight {
            return "No knowledge nudges are queued yet; a background knowledge search is still running."
                .into();
        }
        return "No knowledge nudges are queued.".into();
    }

    let mut out = format!(
        "Drained {} queued knowledge nudge(s). Use any relevant material before answering.\n\n",
        nudges.len()
    );
    for (idx, nudge) in nudges.iter().enumerate() {
        out.push_str(&format!(
            "## Knowledge nudge {}\n{}\n\n",
            idx + 1,
            nudge.trim()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_ok() {
        assert!(parse_args(json!({})).is_ok());
    }

    #[test]
    fn render_empty_mentions_in_flight() {
        let text = render(&[], true);
        assert!(text.contains("still running"));
    }

    #[test]
    fn render_drained_nudges() {
        let text = render(&["alpha".into(), "beta".into()], false);
        assert!(text.contains("2 queued"));
        assert!(text.contains("alpha"));
        assert!(text.contains("beta"));
    }
}
