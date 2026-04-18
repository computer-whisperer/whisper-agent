//! `pod_about` — serve the in-tree documentation for the pod/behavior system.

use serde::Deserialize;
use serde_json::{Value, json};

use super::{POD_ABOUT, ToolOutcome, no_update_error, no_update_text};
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: POD_ABOUT.into(),
        description: "Read documentation about the pod-agent system you run inside: \
                      schemas for pod.toml and behavior.toml, trigger variants, cron \
                      syntax, retention policies, and how self-modification works via \
                      the pod_*_file tools. Call with no arguments (or \
                      `topic: \"index\"`) for the list of topics, then call again \
                      with a specific topic name. Use this BEFORE writing a new \
                      behavior.toml or pod.toml if you haven't recently — the schemas \
                      may have fields you don't remember."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to read. Omit or pass \"index\" for the topic list."
                }
            }
        }),
        annotations: ToolAnnotations {
            title: Some("Pod/behavior reference".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize, Default)]
struct AboutArgs {
    #[serde(default)]
    topic: Option<String>,
}

pub(super) fn run(args: Value) -> ToolOutcome {
    let parsed: AboutArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    let topic = parsed.topic.as_deref().unwrap_or("index");
    match crate::tools::pod_about_docs::topic(topic) {
        Some(text) => no_update_text(text.to_string()),
        None => no_update_error(format!(
            "unknown topic `{topic}`. Valid topics: [{}]. Call pod_about with no args for the index.",
            crate::tools::pod_about_docs::TOPIC_NAMES.join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{dispatch, join_blocks, sample_config, temp_dir};
    use super::*;

    #[tokio::test]
    async fn about_returns_index_by_default() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(dir.clone(), cfg.clone(), vec![], POD_ABOUT, json!({})).await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("pod.toml"), "index missing pod.toml: {text}");
        assert!(
            text.contains("behavior.toml"),
            "index missing behavior.toml: {text}"
        );
    }

    #[tokio::test]
    async fn about_returns_requested_topic() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_ABOUT,
            json!({ "topic": "cron" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("Five-field UNIX crontab"),
            "cron topic body not returned: {text}"
        );
    }

    #[tokio::test]
    async fn about_rejects_unknown_topic() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_ABOUT,
            json!({ "topic": "nonsense" }),
        )
        .await;
        assert!(out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("unknown topic") && text.contains("Valid topics"),
            "bad error: {text}"
        );
    }
}
