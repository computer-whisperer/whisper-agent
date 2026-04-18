//! Pure helpers for building a thread's config from a pod's defaults and
//! optional overrides. Separate from the scheduler's state machine so the
//! translation rules (what a pod default is, how an override layers on
//! top) are readable without the surrounding lifecycle noise.

use whisper_agent_protocol::{HostEnvSpec, ThreadBindingsRequest, ThreadConfig, ThreadConfigOverride};

use crate::pod::Pod;

/// Synthesize a `PodConfig` from the server's runtime defaults — used to
/// bootstrap the in-memory default pod when no real pod.toml exists.
///
/// The resulting `[allow]` table reflects exactly what the server has wired
/// up at startup: every configured backend, every shared MCP host, and
/// optionally a single named host-env slot. When `default_host_env` is
/// `None`, the synthesized pod has an empty `allow.host_env` and its
/// `thread_defaults.host_env` is the empty string — threads inside the
/// default pod run with no host-env MCP connection.
pub fn build_default_pod_config(
    pod_id: &str,
    config: &ThreadConfig,
    default_backend: &str,
    default_host_env: Option<(String, HostEnvSpec)>,
    backend_names: &[String],
    shared_host_names: &[String],
) -> whisper_agent_protocol::PodConfig {
    use whisper_agent_protocol::{NamedHostEnv, PodAllow, PodConfig, PodLimits, ThreadDefaults};
    const HOST_ENV_NAME: &str = "default";
    let (host_env_entries, default_host_env_name) = match default_host_env {
        Some((provider, spec)) => (
            vec![NamedHostEnv {
                name: HOST_ENV_NAME.to_string(),
                provider,
                spec,
            }],
            HOST_ENV_NAME.to_string(),
        ),
        None => (Vec::new(), String::new()),
    };
    let now = chrono::Utc::now().to_rfc3339();
    PodConfig {
        name: pod_id.to_string(),
        description: Some(
            "Auto-synthesized default pod. Mutate via UpdatePodConfig (or hand-edit \
             pod.toml) to change thread defaults or tighten the allow cap."
                .into(),
        ),
        created_at: now,
        allow: PodAllow {
            backends: backend_names.to_vec(),
            mcp_hosts: shared_host_names.to_vec(),
            host_env: host_env_entries,
        },
        thread_defaults: ThreadDefaults {
            backend: default_backend.to_string(),
            model: config.model.clone(),
            system_prompt_file: "system_prompt.md".into(),
            max_tokens: config.max_tokens,
            max_turns: config.max_turns,
            approval_policy: config.approval_policy,
            host_env: default_host_env_name,
            mcp_hosts: shared_host_names.to_vec(),
        },
        limits: PodLimits::default(),
    }
}

/// Render the pod's `thread_defaults` (model, prompt, limits, policy) into
/// a fresh `ThreadConfig`. Binding-side defaults (backend, sandbox, shared
/// hosts) are produced separately by `resolve_bindings_choice`.
pub(super) fn base_thread_config_from_pod(pod: &Pod) -> ThreadConfig {
    let defaults = &pod.config.thread_defaults;
    ThreadConfig {
        model: defaults.model.clone(),
        system_prompt: pod.system_prompt.clone(),
        max_tokens: defaults.max_tokens,
        max_turns: defaults.max_turns,
        approval_policy: defaults.approval_policy,
    }
}

pub(super) fn apply_config_override(
    base: ThreadConfig,
    ov: Option<ThreadConfigOverride>,
) -> ThreadConfig {
    let Some(ov) = ov else { return base };
    ThreadConfig {
        model: ov.model.unwrap_or(base.model),
        system_prompt: ov.system_prompt.unwrap_or(base.system_prompt),
        max_tokens: ov.max_tokens.unwrap_or(base.max_tokens),
        max_turns: ov.max_turns.unwrap_or(base.max_turns),
        approval_policy: ov.approval_policy.unwrap_or(base.approval_policy),
    }
}

/// Translate a behavior's thread-override block into the
/// (ThreadConfigOverride, ThreadBindingsRequest) pair `create_task`
/// consumes. `None` returns for either side mean "inherit everything
/// from pod defaults."
pub(super) fn behavior_override_to_requests(
    ov: &whisper_agent_protocol::BehaviorThreadOverride,
) -> (Option<ThreadConfigOverride>, Option<ThreadBindingsRequest>) {
    let config_override = if ov.model.is_some()
        || ov.max_tokens.is_some()
        || ov.max_turns.is_some()
        || ov.approval_policy.is_some()
    {
        Some(ThreadConfigOverride {
            model: ov.model.clone(),
            system_prompt: None, // behaviors inherit pod system_prompt
            max_tokens: ov.max_tokens,
            max_turns: ov.max_turns,
            approval_policy: ov.approval_policy,
        })
    } else {
        None
    };
    let b = &ov.bindings;
    let bindings_request = if b.backend.is_some() || b.host_env.is_some() || b.mcp_hosts.is_some() {
        Some(ThreadBindingsRequest {
            backend: b.backend.clone(),
            host_env: b.host_env.clone(),
            mcp_hosts: b.mcp_hosts.clone(),
        })
    } else {
        None
    };
    (config_override, bindings_request)
}

/// Render a behavior's prompt template. Minimal v1: a single
/// `{{payload}}` token expands to the pretty-printed JSON payload
/// (empty string for `Null`). Richer templating (accessor paths,
/// conditionals) is deferred until we see what behaviors actually want.
pub(super) fn render_behavior_prompt(template: &str, payload: &serde_json::Value) -> String {
    if !template.contains("{{payload}}") {
        return template.to_string();
    }
    let payload_text = if payload.is_null() {
        String::new()
    } else {
        serde_json::to_string_pretty(payload).unwrap_or_default()
    };
    template.replace("{{payload}}", &payload_text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::{BehaviorBindingsOverride, BehaviorThreadOverride};

    #[test]
    fn render_leaves_template_alone_when_no_placeholder() {
        let out = render_behavior_prompt("do the thing", &serde_json::Value::Null);
        assert_eq!(out, "do the thing");
    }

    #[test]
    fn render_expands_null_payload_to_empty() {
        let out = render_behavior_prompt("before {{payload}} after", &serde_json::Value::Null);
        assert_eq!(out, "before  after");
    }

    #[test]
    fn render_expands_object_payload_to_pretty_json() {
        let payload = serde_json::json!({ "from": "alice", "subject": "hi" });
        let out = render_behavior_prompt("email: {{payload}}", &payload);
        assert!(out.starts_with("email: {"));
        assert!(out.contains("\"from\": \"alice\""));
        assert!(out.contains("\"subject\": \"hi\""));
    }

    #[test]
    fn override_translates_every_set_field() {
        let ov = BehaviorThreadOverride {
            model: Some("sonnet-4-6".into()),
            max_tokens: Some(8192),
            max_turns: Some(20),
            approval_policy: Some(whisper_agent_protocol::ApprovalPolicy::AutoApproveAll),
            bindings: BehaviorBindingsOverride {
                backend: Some("anthropic".into()),
                host_env: Some("readonly".into()),
                mcp_hosts: Some(vec!["fetch".into()]),
            },
        };
        let (cfg, bindings) = behavior_override_to_requests(&ov);
        let cfg = cfg.expect("config_override populated");
        assert_eq!(cfg.model.as_deref(), Some("sonnet-4-6"));
        assert_eq!(cfg.max_tokens, Some(8192));
        assert_eq!(cfg.max_turns, Some(20));
        assert!(matches!(
            cfg.approval_policy,
            Some(whisper_agent_protocol::ApprovalPolicy::AutoApproveAll)
        ));
        // Behaviors never override system_prompt — pod default is authoritative.
        assert!(cfg.system_prompt.is_none());
        let b = bindings.expect("bindings_request populated");
        assert_eq!(b.backend.as_deref(), Some("anthropic"));
        assert_eq!(b.host_env.as_deref(), Some("readonly"));
        assert_eq!(b.mcp_hosts.as_deref(), Some(&["fetch".to_string()][..]));
    }

    #[test]
    fn override_returns_none_when_nothing_set() {
        let ov = BehaviorThreadOverride::default();
        let (cfg, bindings) = behavior_override_to_requests(&ov);
        assert!(cfg.is_none());
        assert!(bindings.is_none());
    }

    #[test]
    fn override_partial_sets_only_relevant_side() {
        // Only a binding override, no config override.
        let ov = BehaviorThreadOverride {
            bindings: BehaviorBindingsOverride {
                mcp_hosts: Some(vec!["fetch".into()]),
                ..Default::default()
            },
            ..Default::default()
        };
        let (cfg, bindings) = behavior_override_to_requests(&ov);
        assert!(cfg.is_none(), "no config-side fields were set");
        assert!(bindings.is_some());
    }
}
