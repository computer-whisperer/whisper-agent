//! `list_llm_providers` — scheduler intercept that enumerates the
//! configured LLM backends and, optionally, the models a specific
//! backend exposes.
//!
//! Without `backend`: cheap, synchronous — reads the scheduler's
//! backend catalog and returns `name`, `kind`, `default_model`,
//! `auth_mode` for each entry plus the server default.
//!
//! With `backend`: one network round-trip to the provider's model
//! catalog endpoint (Anthropic `/v1/models`, OpenAI `/v1/models`,
//! etc.). Returned as an async completion so the scheduler loop
//! doesn't block on the HTTP call.

use serde::Deserialize;
use serde_json::{Value, json};

use super::LIST_LLM_PROVIDERS;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: LIST_LLM_PROVIDERS.into(),
        description: "List the LLM backends this server is configured with. Each entry is \
                      `(name, kind, default_model, auth_mode)` — `name` is the alias a \
                      thread config (or `dispatch_thread`'s `bindings_override.backend`) \
                      refers to; `kind` is the wire protocol (`anthropic`, `openai_chat`, \
                      `google`, …). Pass `backend` to also enumerate the models that \
                      backend exposes — this hits the provider's catalog endpoint, so \
                      only pass it when you actually need the list. Use this to pick a \
                      backend/model for a `dispatch_thread` or to answer a user asking \
                      what's available."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "backend": {
                    "type": "string",
                    "description": "Optional. When set, also list the models advertised by \
                                    the named backend. Leave unset for a cheap \
                                    metadata-only response."
                }
            }
        }),
        annotations: ToolAnnotations {
            title: Some("List LLM providers".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            // Expanding models hits an external catalog endpoint whose
            // contents we don't control (providers add/remove models
            // independently of this server's config).
            open_world_hint: Some(true),
        },
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ListLlmProvidersArgs {
    #[serde(default)]
    pub backend: Option<String>,
}

pub fn parse_args(value: Value) -> Result<ListLlmProvidersArgs, String> {
    serde_json::from_value::<ListLlmProvidersArgs>(value)
        .map_err(|e| format!("invalid list_llm_providers arguments: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_ok() {
        let args = parse_args(json!({})).unwrap();
        assert!(args.backend.is_none());
    }

    #[test]
    fn parse_with_backend() {
        let args = parse_args(json!({ "backend": "anthropic" })).unwrap();
        assert_eq!(args.backend.as_deref(), Some("anthropic"));
    }

    #[test]
    fn parse_rejects_wrong_type() {
        assert!(parse_args(json!({ "backend": 42 })).is_err());
    }

    #[test]
    fn parse_rejects_unknown_fields() {
        // serde is permissive by default; we just need this to at
        // least accept the known shape. Belt-and-suspenders test that
        // extra keys don't blow up.
        assert!(parse_args(json!({ "backend": "x", "extra": true })).is_ok());
    }
}
