//! `pod_run_behavior` + `pod_set_behavior_enabled` — orchestration tools
//! that let an agent fire or pause/resume its pod's behaviors.
//!
//! Both tools return a [`ToolOutcome`] whose `scheduler_command` field
//! carries the requested action. The scheduler dispatches the command
//! in the same tool-result step so the wire sees the behavior-state
//! change atomically with the tool call.

use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    POD_RUN_BEHAVIOR, POD_SET_BEHAVIOR_ENABLED, SchedulerCommand, ToolOutcome, no_update_error,
    text_result,
};
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn run_behavior_descriptor() -> McpTool {
    McpTool {
        name: POD_RUN_BEHAVIOR.into(),
        description: "Manually fire one of this pod's behaviors. Equivalent to the \
                      UI's Run button — bypasses cron timers and the paused gate (an \
                      explicit action always runs). The behavior's `prompt.md` is \
                      substituted with the `payload` argument as `{{payload}}` the \
                      same way a webhook body is. Use this to test a newly-written \
                      behavior without waiting for its schedule, or to kick a \
                      webhook behavior with a custom payload. Returns immediately \
                      once the run is queued — the spawned thread appears on the \
                      wire as normal."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "behavior_id": {
                    "type": "string",
                    "description": "Which behavior under this pod to fire."
                },
                "payload": {
                    "description": "Optional JSON payload. Omitted → null."
                }
            },
            "required": ["behavior_id"]
        }),
        annotations: ToolAnnotations {
            title: Some("Run a behavior".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize)]
struct RunBehaviorArgs {
    behavior_id: String,
    #[serde(default)]
    payload: Option<Value>,
}

pub(super) fn run_behavior(behavior_ids: &[String], args: Value) -> ToolOutcome {
    let parsed: RunBehaviorArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    if !behavior_ids.iter().any(|i| i == &parsed.behavior_id) {
        return no_update_error(format!(
            "unknown behavior `{}` in this pod. Known: [{}]",
            parsed.behavior_id,
            behavior_ids.join(", ")
        ));
    }
    ToolOutcome {
        result: text_result(format!(
            "queued manual run of behavior `{}`",
            parsed.behavior_id
        )),
        pod_update: None,
        scheduler_command: Some(SchedulerCommand::RunBehavior {
            behavior_id: parsed.behavior_id,
            payload: parsed.payload,
        }),
    }
}

pub(super) fn set_behavior_enabled_descriptor() -> McpTool {
    McpTool {
        name: POD_SET_BEHAVIOR_ENABLED.into(),
        description: "Pause or resume one of this pod's automatic-trigger behaviors. \
                      Paused behaviors skip cron ticks, return 503 on webhook POSTs, \
                      and do not catch up at startup. Manual `pod_run_behavior` still \
                      works while paused — it's always an explicit action. Pausing \
                      drops any `queue_one` payload the behavior had parked; resuming \
                      bumps a cron behavior's cursor to now so catch-up doesn't \
                      replay windows missed during the pause."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "behavior_id": {
                    "type": "string",
                    "description": "Which behavior to pause or resume."
                },
                "enabled": {
                    "type": "boolean",
                    "description": "`true` to resume, `false` to pause."
                }
            },
            "required": ["behavior_id", "enabled"]
        }),
        annotations: ToolAnnotations {
            title: Some("Pause/resume a behavior".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize)]
struct SetBehaviorEnabledArgs {
    behavior_id: String,
    enabled: bool,
}

pub(super) fn set_behavior_enabled(behavior_ids: &[String], args: Value) -> ToolOutcome {
    let parsed: SetBehaviorEnabledArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    if !behavior_ids.iter().any(|i| i == &parsed.behavior_id) {
        return no_update_error(format!(
            "unknown behavior `{}` in this pod. Known: [{}]",
            parsed.behavior_id,
            behavior_ids.join(", ")
        ));
    }
    ToolOutcome {
        result: text_result(format!(
            "behavior `{}` set to enabled={}",
            parsed.behavior_id, parsed.enabled
        )),
        pod_update: None,
        scheduler_command: Some(SchedulerCommand::SetBehaviorEnabled {
            behavior_id: parsed.behavior_id,
            enabled: parsed.enabled,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{dispatch, join_blocks, sample_config, temp_dir};
    use super::*;

    #[tokio::test]
    async fn run_behavior_rejects_unknown_id() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["real".to_string()],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::AuthorAny,
            POD_RUN_BEHAVIOR,
            json!({ "behavior_id": "ghost" }),
        )
        .await;
        assert!(out.result.is_error);
        assert!(out.scheduler_command.is_none());
        let text = join_blocks(&out.result.content);
        assert!(text.contains("unknown behavior"), "wrong error: {text}");
    }

    #[tokio::test]
    async fn run_behavior_emits_scheduler_command() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["daily".to_string()],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::AuthorAny,
            POD_RUN_BEHAVIOR,
            json!({ "behavior_id": "daily", "payload": {"foo": 1} }),
        )
        .await;
        assert!(!out.result.is_error);
        match out.scheduler_command {
            Some(SchedulerCommand::RunBehavior {
                behavior_id,
                payload,
            }) => {
                assert_eq!(behavior_id, "daily");
                assert_eq!(payload, Some(json!({ "foo": 1 })));
            }
            other => panic!("expected RunBehavior command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_behavior_defaults_payload_to_none() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["daily".to_string()],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::AuthorAny,
            POD_RUN_BEHAVIOR,
            json!({ "behavior_id": "daily" }),
        )
        .await;
        assert!(!out.result.is_error);
        match out.scheduler_command {
            Some(SchedulerCommand::RunBehavior { payload, .. }) => {
                assert_eq!(payload, None);
            }
            other => panic!("expected RunBehavior command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn set_behavior_enabled_emits_scheduler_command() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["daily".to_string()],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::AuthorAny,
            POD_SET_BEHAVIOR_ENABLED,
            json!({ "behavior_id": "daily", "enabled": false }),
        )
        .await;
        assert!(!out.result.is_error);
        match out.scheduler_command {
            Some(SchedulerCommand::SetBehaviorEnabled {
                behavior_id,
                enabled,
            }) => {
                assert_eq!(behavior_id, "daily");
                assert!(!enabled);
            }
            other => panic!("expected SetBehaviorEnabled command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn set_behavior_enabled_rejects_unknown_id() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::AuthorAny,
            POD_SET_BEHAVIOR_ENABLED,
            json!({ "behavior_id": "ghost", "enabled": true }),
        )
        .await;
        assert!(out.result.is_error);
        assert!(out.scheduler_command.is_none());
    }

    #[tokio::test]
    async fn run_behavior_denied_when_behaviors_cap_is_none() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["daily".to_string()],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::None,
            POD_RUN_BEHAVIOR,
            json!({ "behavior_id": "daily" }),
        )
        .await;
        assert!(out.result.is_error);
        assert!(out.scheduler_command.is_none());
        let text = join_blocks(&out.result.content);
        assert!(text.contains("behaviors capability"), "wrong error: {text}");
    }

    #[tokio::test]
    async fn set_behavior_enabled_denied_when_behaviors_cap_is_none() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["daily".to_string()],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::None,
            POD_SET_BEHAVIOR_ENABLED,
            json!({ "behavior_id": "daily", "enabled": false }),
        )
        .await;
        assert!(out.result.is_error);
        assert!(out.scheduler_command.is_none());
    }

    #[tokio::test]
    async fn run_behavior_admitted_at_read_cap() {
        // Read is the minimum cap to fire or pause-resume an existing
        // behavior — those are invocation / state-flip, not authoring.
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["daily".to_string()],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::Read,
            POD_RUN_BEHAVIOR,
            json!({ "behavior_id": "daily" }),
        )
        .await;
        assert!(!out.result.is_error);
        assert!(out.scheduler_command.is_some());
    }
}
