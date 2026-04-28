//! `knowledge_modify` — agent-facing surface for the bucket mutation
//! triad (`Bucket::insert` and `Bucket::tombstone`).
//!
//! Scope-gated by the same `[allow.knowledge_buckets]` list as
//! [`knowledge_query`](super::knowledge_query); the per-bucket
//! source-kind check inside `Bucket::insert` / `Bucket::tombstone`
//! is the actual safety net — only `kind = "managed"` buckets
//! accept LLM-driven mutations. Wikipedia / tracked / stored buckets
//! reject the call at the bucket layer with a clear error.
//!
//! Two actions:
//! - `insert` — write a `(source_id, content)` pair into the bucket.
//!   The system chunks the content with the bucket's chunker, embeds
//!   each chunk via the bucket's embedder, and calls `Bucket::insert`.
//!   If a prior version of the same `source_id` exists, the caller
//!   must tombstone it first — `insert` doesn't auto-replace
//!   (matches the underlying `Bucket::insert` semantics).
//! - `tombstone` — remove every chunk for a given `source_id`. The
//!   tool resolves `source_id → chunk_ids` via the slot's
//!   `source_index`, then calls `Bucket::tombstone`.
//!
//! The handler lives in
//! [`crate::runtime::scheduler::Scheduler::complete_knowledge_modify_call`]
//! — same scheduler-intercept pattern as `knowledge_query`.

use serde::Deserialize;
use serde_json::{Value, json};

use super::KNOWLEDGE_MODIFY;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

/// Maximum content length for a single `insert` call. The
/// LLM-supplied content gets chunked into multiple records, embedded,
/// and indexed — runaway inputs can blow up an agent's tool-call
/// latency and the bucket's per-tick footprint. 1 MB is generous
/// (~250k tokens at typical English ratios) without enabling
/// pathological cases.
pub const MAX_CONTENT_BYTES: usize = 1_024 * 1_024;

/// Maximum length of a `source_id`. Filesystem-style ids are short
/// in practice; the cap is a sanity check, not a budget.
pub const MAX_SOURCE_ID_LEN: usize = 256;

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: KNOWLEDGE_MODIFY.into(),
        description: "Insert or tombstone records in a managed knowledge bucket — pod-scoped \
                      memory the agent owns. Requires the bucket id to be in the pod's \
                      `[allow.knowledge_buckets]` list AND for the bucket to be configured \
                      with `kind = \"managed\"`. Tracked / stored / linked buckets (e.g. \
                      Wikipedia) reject mutations regardless of the allowlist. \
                      \n\n\
                      Actions:\n\
                      - `insert`: store `content` under `source_id`. The text is chunked + \
                        embedded by the bucket's pipeline. If `source_id` already exists, \
                        the call errors — tombstone the old version first.\n\
                      - `tombstone`: remove every chunk for `source_id`. Idempotent: \
                        tombstoning a non-existent source_id is a no-op success."
            .into(),
        input_schema: json!({
            "type": "object",
            "required": ["action", "bucket_id", "source_id"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["insert", "tombstone"],
                    "description": "What to do. `insert` adds new content; `tombstone` \
                                    removes existing content by source id."
                },
                "bucket_id": {
                    "type": "string",
                    "description": "Target bucket. Must be in the pod's \
                                    `[allow.knowledge_buckets]` and configured as \
                                    `kind = \"managed\"`."
                },
                "source_id": {
                    "type": "string",
                    "description": "Caller-chosen identifier for the record. For `insert`, \
                                    this is the new record's id (must not already exist); \
                                    for `tombstone`, this names the record to remove. \
                                    Common shapes: `memo-2026-04-28`, `note-octopus-facts`, \
                                    `decision-2026-q2-roadmap`. Limited to 256 bytes."
                },
                "content": {
                    "type": "string",
                    "description": "Required for `insert`, ignored for `tombstone`. The \
                                    raw text to store. Limited to 1 MiB. The bucket's \
                                    chunker splits it into ~500-token windows; each \
                                    chunk is independently embeddable + searchable."
                }
            }
        }),
        annotations: ToolAnnotations {
            title: Some("Modify a managed knowledge bucket".into()),
            // Mutates state — not read-only, not idempotent for `insert`
            // (a duplicate `source_id` errors on retry), idempotent for
            // `tombstone`. Conservative tag: not idempotent.
            read_only_hint: Some(false),
            destructive_hint: Some(true),
            idempotent_hint: Some(false),
            // Bucket contents are confined to the server's filesystem;
            // no external resources at play.
            open_world_hint: Some(false),
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeModifyAction {
    Insert,
    Tombstone,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KnowledgeModifyArgs {
    pub action: KnowledgeModifyAction,
    pub bucket_id: String,
    pub source_id: String,
    #[serde(default)]
    pub content: Option<String>,
}

pub fn parse_args(value: Value) -> Result<KnowledgeModifyArgs, String> {
    let args: KnowledgeModifyArgs = serde_json::from_value(value)
        .map_err(|e| format!("invalid knowledge_modify arguments: {e}"))?;
    if args.bucket_id.trim().is_empty() {
        return Err("knowledge_modify: `bucket_id` must not be empty".into());
    }
    if args.source_id.trim().is_empty() {
        return Err("knowledge_modify: `source_id` must not be empty".into());
    }
    if args.source_id.len() > MAX_SOURCE_ID_LEN {
        return Err(format!(
            "knowledge_modify: `source_id` must be ≤{MAX_SOURCE_ID_LEN} bytes (got {})",
            args.source_id.len()
        ));
    }
    match args.action {
        KnowledgeModifyAction::Insert => {
            let content = args.content.as_deref().unwrap_or("");
            if content.trim().is_empty() {
                return Err(
                    "knowledge_modify: `content` is required for `insert` and must not be empty"
                        .into(),
                );
            }
            if content.len() > MAX_CONTENT_BYTES {
                return Err(format!(
                    "knowledge_modify: `content` must be ≤{MAX_CONTENT_BYTES} bytes (got {})",
                    content.len()
                ));
            }
        }
        KnowledgeModifyAction::Tombstone => {
            // `content` is ignored for tombstone but we don't error on
            // its presence — the model may include it defensively
            // (e.g. echoing what it's removing). Quietly accept.
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_insert_minimal() {
        let args = parse_args(json!({
            "action": "insert",
            "bucket_id": "pod_memory",
            "source_id": "note-1",
            "content": "hello world",
        }))
        .unwrap();
        assert!(matches!(args.action, KnowledgeModifyAction::Insert));
        assert_eq!(args.bucket_id, "pod_memory");
        assert_eq!(args.source_id, "note-1");
        assert_eq!(args.content.as_deref(), Some("hello world"));
    }

    #[test]
    fn parse_tombstone_minimal() {
        let args = parse_args(json!({
            "action": "tombstone",
            "bucket_id": "pod_memory",
            "source_id": "note-1",
        }))
        .unwrap();
        assert!(matches!(args.action, KnowledgeModifyAction::Tombstone));
    }

    #[test]
    fn parse_tombstone_ignores_content() {
        // Models may pass content defensively even though it's
        // unused — accept rather than reject.
        let args = parse_args(json!({
            "action": "tombstone",
            "bucket_id": "pod_memory",
            "source_id": "note-1",
            "content": "what was here",
        }))
        .unwrap();
        assert!(matches!(args.action, KnowledgeModifyAction::Tombstone));
    }

    #[test]
    fn parse_rejects_unknown_action() {
        assert!(
            parse_args(json!({
                "action": "delete",
                "bucket_id": "pod_memory",
                "source_id": "x",
            }))
            .is_err()
        );
    }

    #[test]
    fn parse_rejects_empty_bucket_id() {
        assert!(
            parse_args(json!({
                "action": "insert",
                "bucket_id": "  ",
                "source_id": "x",
                "content": "y",
            }))
            .is_err()
        );
    }

    #[test]
    fn parse_rejects_empty_source_id() {
        assert!(
            parse_args(json!({
                "action": "insert",
                "bucket_id": "pod_memory",
                "source_id": "",
                "content": "y",
            }))
            .is_err()
        );
    }

    #[test]
    fn parse_rejects_oversized_source_id() {
        let big = "x".repeat(MAX_SOURCE_ID_LEN + 1);
        assert!(
            parse_args(json!({
                "action": "insert",
                "bucket_id": "pod_memory",
                "source_id": big,
                "content": "y",
            }))
            .is_err()
        );
    }

    #[test]
    fn parse_insert_rejects_missing_content() {
        assert!(
            parse_args(json!({
                "action": "insert",
                "bucket_id": "pod_memory",
                "source_id": "note-1",
            }))
            .is_err()
        );
    }

    #[test]
    fn parse_insert_rejects_empty_content() {
        assert!(
            parse_args(json!({
                "action": "insert",
                "bucket_id": "pod_memory",
                "source_id": "note-1",
                "content": "   ",
            }))
            .is_err()
        );
    }

    #[test]
    fn parse_insert_rejects_oversized_content() {
        let big = "x".repeat(MAX_CONTENT_BYTES + 1);
        assert!(
            parse_args(json!({
                "action": "insert",
                "bucket_id": "pod_memory",
                "source_id": "note-1",
                "content": big,
            }))
            .is_err()
        );
    }

    #[test]
    fn parse_rejects_missing_required_fields() {
        // No action.
        assert!(
            parse_args(json!({
                "bucket_id": "pod_memory",
                "source_id": "x",
            }))
            .is_err()
        );
        // No bucket_id.
        assert!(
            parse_args(json!({
                "action": "tombstone",
                "source_id": "x",
            }))
            .is_err()
        );
    }
}
