//! `find_tool` — synchronous scheduler intercept that returns tool
//! names + one-line descriptions matching a regex and optional
//! category filter. Schemas are NOT included; the model pulls those
//! on demand via `describe_tool`.
//!
//! This is the "grep the tool catalog" primitive. It complements the
//! static system-prompt listing (which shows everything admissible at
//! thread seed) by letting the model search by keyword when the
//! listing is configured off (`initial_listing = "none"`) or when
//! it's looking for something specific in a large catalog.

use serde::Deserialize;
use serde_json::{Value, json};

use super::FIND_TOOL;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

/// Default page size for `find_tool`. Keeps the result payload small
/// even when the pattern matches a large share of a 100+ tool
/// catalog — the model can page with `offset` if it needs more.
pub const DEFAULT_LIMIT: u32 = 50;

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: FIND_TOOL.into(),
        description: "Search the tool catalog by regex over name+description. \
                      Returns name, one-line description, category, and whether \
                      the tool requires `sudo` to invoke (askable rather than \
                      admitted) — schemas are NOT included (call `describe_tool` \
                      for those). Optional `category` narrows to one coarse \
                      bucket (`builtin`, `host_env`, `shared_mcp`). \
                      `include_escalation` (default true) includes askable tools \
                      — those within the pod ceiling but outside your current \
                      scope, reachable via `sudo` with user approval. \
                      `limit` defaults to 50; use `offset` to page."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex matched against `name` and `description` (any match)."
                },
                "category": {
                    "type": "string",
                    "enum": ["builtin", "host_env", "shared_mcp"],
                    "description": "Optional coarse-bucket filter. Omit to search all sources."
                },
                "include_escalation": {
                    "type": "boolean",
                    "description": "Include tools within the pod ceiling but outside your \
                                    current scope. Default true."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Max results. Default 50."
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Skip the first N matches. Default 0."
                }
            },
            "required": ["pattern"]
        }),
        annotations: ToolAnnotations {
            title: Some("Find tool".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FindToolArgs {
    pub pattern: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default = "default_include_escalation")]
    pub include_escalation: bool,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
}

fn default_include_escalation() -> bool {
    true
}

impl FindToolArgs {
    pub fn effective_limit(&self) -> u32 {
        self.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, 500)
    }
    pub fn effective_offset(&self) -> u32 {
        self.offset.unwrap_or(0)
    }
}

pub fn parse_args(value: Value) -> Result<FindToolArgs, String> {
    serde_json::from_value::<FindToolArgs>(value)
        .map_err(|e| format!("invalid find_tool arguments: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_pattern_only() {
        let v = json!({ "pattern": "write|edit" });
        let args = parse_args(v).unwrap();
        assert_eq!(args.pattern, "write|edit");
        assert_eq!(args.category, None);
        assert!(args.include_escalation);
        assert_eq!(args.effective_limit(), DEFAULT_LIMIT);
        assert_eq!(args.effective_offset(), 0);
    }

    #[test]
    fn parse_all_fields() {
        let v = json!({
            "pattern": "^pod_",
            "category": "builtin",
            "include_escalation": false,
            "limit": 10,
            "offset": 5
        });
        let args = parse_args(v).unwrap();
        assert_eq!(args.pattern, "^pod_");
        assert_eq!(args.category.as_deref(), Some("builtin"));
        assert!(!args.include_escalation);
        assert_eq!(args.effective_limit(), 10);
        assert_eq!(args.effective_offset(), 5);
    }

    #[test]
    fn effective_limit_clamps() {
        let args = FindToolArgs {
            pattern: "x".into(),
            category: None,
            include_escalation: true,
            limit: Some(99999),
            offset: None,
        };
        assert_eq!(args.effective_limit(), 500);
    }

    #[test]
    fn parse_rejects_missing_pattern() {
        assert!(parse_args(json!({})).is_err());
    }
}
