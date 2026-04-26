//! `knowledge_query` — agent-facing surface for the bucket retrieval
//! stack.
//!
//! Scope-gated: a thread can only query buckets named in its pod's
//! `[allow.knowledge_buckets]` (default empty / default-deny, matching
//! `[allow.backends]`). The `buckets` argument optionally narrows the
//! query to a subset; omitting it queries every bucket the pod is
//! allowed to read from. Out-of-scope explicit names produce a clear
//! error listing what *is* in scope so the model can correct without
//! a second discovery round-trip.
//!
//! The handler itself lives in
//! [`crate::runtime::scheduler::Scheduler::complete_knowledge_query_call`]
//! — same scheduler-intercept pattern as `list_llm_providers` (it
//! needs scheduler state for the registry, embedder catalog, and
//! reranker catalog, none of which the in-process tool dispatcher has
//! access to).

use serde::Deserialize;
use serde_json::{Value, json};

use super::KNOWLEDGE_QUERY;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

/// Default `top_k` when the caller doesn't supply one. Smaller than
/// the WebUI's default (10) because every result lands in the model's
/// context — agents should ask explicitly for more if they care.
pub const DEFAULT_TOP_K: u32 = 5;

/// Hard cap on `top_k`. The per-call response holds full chunk text
/// for each hit (~500 tokens / ~2000 chars per chunk by default
/// chunker config), so top_k=20 lands ~10k tokens in the tool result
/// — large but tractable. Above that and the result starts crowding
/// the context window.
pub const MAX_TOP_K: u32 = 20;

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: KNOWLEDGE_QUERY.into(),
        description: "Hybrid (dense + sparse + reranker) search over the pod's \
                      in-scope knowledge buckets. Returns ranked hits, each with \
                      its source title (article name / file path), retrieval-path \
                      tag (`dense` / `sparse`), reranker score, and a snippet of \
                      the chunk text. Use this when the user asks a factual \
                      question that might be answered by stored material — code \
                      docs, wikipedia articles, your own notes — instead of \
                      guessing or web-searching. Pass `buckets` to narrow to a \
                      subset of in-scope buckets; omit for all of them. \
                      `top_k` defaults to 5 (max 20). The pod's \
                      `[allow.knowledge_buckets]` list defines what's in scope \
                      — empty means no buckets are reachable from this pod and \
                      the call errors with that explanation."
            .into(),
        input_schema: json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text query. Same wording the user would \
                                    type into a search box; both retrieval paths \
                                    consume it directly."
                },
                "buckets": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional. Restrict the query to this subset of \
                                    bucket ids. Each must be in the pod's \
                                    `[allow.knowledge_buckets]`. Omit to query \
                                    everything in scope."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of hits to return after reranker fusion. \
                                    Default 5, max 20."
                }
            }
        }),
        annotations: ToolAnnotations {
            title: Some("Query knowledge buckets".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            // Bucket contents are static within a slot but the slot
            // itself can be rebuilt mid-conversation — same hit ids
            // may appear with different text after a rebuild. Treat
            // as closed-world for our process boundary.
            open_world_hint: Some(false),
        },
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct KnowledgeQueryArgs {
    pub query: String,
    #[serde(default)]
    pub buckets: Vec<String>,
    #[serde(default)]
    pub top_k: Option<u32>,
}

pub fn parse_args(value: Value) -> Result<KnowledgeQueryArgs, String> {
    let args: KnowledgeQueryArgs = serde_json::from_value(value)
        .map_err(|e| format!("invalid knowledge_query arguments: {e}"))?;
    if args.query.trim().is_empty() {
        return Err("knowledge_query: `query` must not be empty".into());
    }
    if let Some(k) = args.top_k
        && (k == 0 || k > MAX_TOP_K)
    {
        return Err(format!(
            "knowledge_query: `top_k` must be in 1..={MAX_TOP_K}"
        ));
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_query_only() {
        let args = parse_args(json!({ "query": "octopus intelligence" })).unwrap();
        assert_eq!(args.query, "octopus intelligence");
        assert!(args.buckets.is_empty());
        assert!(args.top_k.is_none());
    }

    #[test]
    fn parse_with_buckets_and_top_k() {
        let args =
            parse_args(json!({ "query": "x", "buckets": ["wiki", "notes"], "top_k": 10 })).unwrap();
        assert_eq!(args.buckets, vec!["wiki".to_string(), "notes".to_string()]);
        assert_eq!(args.top_k, Some(10));
    }

    #[test]
    fn parse_rejects_empty_query() {
        assert!(parse_args(json!({ "query": "   " })).is_err());
    }

    #[test]
    fn parse_rejects_top_k_zero() {
        assert!(parse_args(json!({ "query": "x", "top_k": 0 })).is_err());
    }

    #[test]
    fn parse_rejects_top_k_over_max() {
        assert!(parse_args(json!({ "query": "x", "top_k": MAX_TOP_K + 1 })).is_err());
    }

    #[test]
    fn parse_rejects_missing_query() {
        assert!(parse_args(json!({})).is_err());
    }
}
