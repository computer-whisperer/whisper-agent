//! Resolve a thread's binding-side choices (backend, host env, shared
//! MCPs) from a pod's defaults plus an optional request, and validate
//! every pick against the pod's `[allow]` cap — and, when a parent
//! scope is supplied (dispatched child threads), against that parent
//! scope as well. Pure — no scheduler state — so the cap-enforcement
//! rules live in one readable place.

use whisper_agent_protocol::{HostEnvBinding, ThreadBindingsRequest};

use crate::permission::Scope;
use crate::pod::Pod;

/// Resolved binding-side choices for a fresh thread. Validated against
/// the pod's `[allow]` cap (and the parent's scope, for dispatched
/// children) before construction.
#[cfg_attr(test, derive(Debug))]
pub(super) struct ResolvedBindings {
    /// Backend catalog name. Empty string = "use server default".
    pub(super) backend_name: String,
    /// Host envs the thread binds to, in declared order. Empty when the
    /// pod declares no host envs (threads there get shared MCPs only)
    /// OR when the request explicitly clears the list. Each entry is a
    /// reference to a pod `[[allow.host_env]]` entry by name — the
    /// allow list is authoritative; unknown names were already rejected
    /// during resolution.
    pub(super) host_env: Vec<HostEnvBinding>,
    /// Catalog names of shared MCP hosts the thread is bound to.
    pub(super) shared_host_names: Vec<String>,
}

/// Layer the request on top of pod `thread_defaults`, then verify each
/// chosen value against the pod's `[allow]` table. Request-provided
/// host-env names are validated one by one; unknown names fail the
/// whole resolution so partial binding never lands.
///
/// `parent_scope`, when `Some`, is the dispatching parent thread's
/// effective scope. Each resolved binding is also verified against it
/// — a parent can never cause a child to bind to a resource outside
/// its own scope. `None` means top-level creation (WS client, behavior
/// fire, auto-compact continuation); the pod cap alone governs.
pub(super) fn resolve_bindings_choice(
    pod: &Pod,
    request: Option<ThreadBindingsRequest>,
    parent_scope: Option<&Scope>,
) -> Result<ResolvedBindings, String> {
    let defaults = &pod.config.thread_defaults;
    let allow = &pod.config.allow;
    let request = request.unwrap_or_default();

    let backend_name = request.backend.unwrap_or_else(|| defaults.backend.clone());
    if !backend_name.is_empty() && !allow.backends.iter().any(|b| b == &backend_name) {
        return Err(format!(
            "backend `{}` not in pod `{}`'s allow.backends ({})",
            backend_name,
            pod.id,
            allow.backends.join(", ")
        ));
    }
    if let Some(parent) = parent_scope
        && !backend_name.is_empty()
        && !parent.backends.admits(&backend_name)
    {
        return Err(format!(
            "backend `{backend_name}` not in dispatching parent's scope.backends"
        ));
    }

    // `None` → inherit the pod default list; `Some(vec)` → replace
    // exactly (empty vec means "bind to nothing, thread runs bare").
    let chosen_names: Vec<String> = request
        .host_env
        .unwrap_or_else(|| defaults.host_env.clone());
    let host_env: Vec<HostEnvBinding> = chosen_names
        .into_iter()
        .map(|name| {
            if !allow.host_env.iter().any(|nh| nh.name == name) {
                return Err(format!(
                    "host env `{name}` not in pod `{}`'s allow.host_env ({})",
                    pod.id,
                    if allow.host_env.is_empty() {
                        "empty".into()
                    } else {
                        allow
                            .host_env
                            .iter()
                            .map(|nh| nh.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    },
                ));
            }
            if let Some(parent) = parent_scope
                && !parent.host_envs.admits(&name)
            {
                return Err(format!(
                    "host env `{name}` not in dispatching parent's scope.host_envs"
                ));
            }
            Ok(HostEnvBinding::Named { name })
        })
        .collect::<Result<_, _>>()?;

    let shared_host_names = request
        .mcp_hosts
        .unwrap_or_else(|| defaults.mcp_hosts.clone());
    for name in &shared_host_names {
        if !allow.mcp_hosts.iter().any(|h| h == name) {
            return Err(format!(
                "shared MCP host `{name}` not in pod `{}`'s allow.mcp_hosts ({})",
                pod.id,
                allow.mcp_hosts.join(", "),
            ));
        }
        if let Some(parent) = parent_scope
            && !parent.mcp_hosts.admits(name)
        {
            return Err(format!(
                "shared MCP host `{name}` not in dispatching parent's scope.mcp_hosts"
            ));
        }
    }

    Ok(ResolvedBindings {
        backend_name,
        host_env,
        shared_host_names,
    })
}

/// Narrow already-resolved bindings against a scope filter. Used by
/// top-level paths that supply a `base_scope_override` (behaviors,
/// auto-compaction continuations) to enforce the same "bindings ⊆
/// scope" invariant `resolve_bindings_choice` enforces for dispatched
/// children via `parent_scope`.
///
/// Filter semantics differ from `parent_scope`'s reject semantics:
/// pod defaults that exceed a behavior's narrower `[scope]` are
/// quietly dropped (the `[scope]` block is the canonical narrower —
/// authors shouldn't need to repeat themselves in `[thread.bindings]`).
/// Backend is the exception: it's singular, so an out-of-scope pick
/// can't be silently filtered and is returned as `Err`.
pub(super) fn narrow_resolved_bindings_to_scope(
    resolved: &mut ResolvedBindings,
    scope_filter: &Scope,
) -> Result<(), String> {
    resolved.host_env.retain(|b| match b {
        HostEnvBinding::Named { name } => scope_filter.host_envs.admits(name),
        // Inline bindings have no catalog name to gate on a
        // `SetOrAll<String>`; `resolve_bindings_choice` only ever
        // emits `Named` from the top-level path, so this arm is
        // defensive cover for future changes.
        HostEnvBinding::Inline { .. } => true,
    });
    resolved
        .shared_host_names
        .retain(|n| scope_filter.mcp_hosts.admits(n));
    if !resolved.backend_name.is_empty() && !scope_filter.backends.admits(&resolved.backend_name) {
        return Err(format!(
            "backend `{}` not in scope.backends — declare \
             `[thread.bindings] backend = ...` on the behavior to pick \
             a backend its scope admits",
            resolved.backend_name,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission::{BehaviorOpsCap, DispatchCap, PodModifyCap, SetOrAll};
    use whisper_agent_protocol::sandbox::{HostEnvSpec, NetworkPolicy, PathAccess};
    use whisper_agent_protocol::{NamedHostEnv, PodAllow, PodConfig, PodLimits, ThreadDefaults};

    fn pod_with_two_backends_and_two_envs() -> Pod {
        let config = PodConfig {
            name: "p".into(),
            description: None,
            created_at: "2026-04-21T00:00:00Z".into(),
            allow: PodAllow {
                backends: vec!["anthropic".into(), "openai".into()],
                mcp_hosts: vec!["fetch".into(), "search".into()],
                host_env: vec![
                    NamedHostEnv {
                        name: "wide".into(),
                        provider: "landlock".into(),
                        spec: HostEnvSpec::Landlock {
                            allowed_paths: vec![PathAccess::read_write("/tmp/wide")],
                            network: NetworkPolicy::Unrestricted,
                        },
                    },
                    NamedHostEnv {
                        name: "narrow".into(),
                        provider: "landlock".into(),
                        spec: HostEnvSpec::Landlock {
                            allowed_paths: vec![PathAccess::read_only("/tmp/narrow")],
                            network: NetworkPolicy::Isolated,
                        },
                    },
                ],
                knowledge_buckets: Vec::new(),
                tools: crate::permission::AllowMap::allow_all(),
                caps: Default::default(),
            },
            thread_defaults: ThreadDefaults {
                backend: "anthropic".into(),
                model: "claude-opus-4-7".into(),
                system_prompt_file: "system_prompt.md".into(),
                max_tokens: 1000,
                max_turns: 10,
                host_env: vec!["wide".into()],
                mcp_hosts: vec!["fetch".into()],
                compaction: Default::default(),
                caps: Default::default(),
                tool_surface: Default::default(),
            },
            limits: PodLimits::default(),
        };
        Pod::new(
            "p".into(),
            std::path::PathBuf::from("/tmp/pod"),
            config,
            String::new(),
            String::new(),
        )
    }

    fn parent_scope_only(backends: &[&str], host_envs: &[&str], mcp_hosts: &[&str]) -> Scope {
        Scope {
            backends: SetOrAll::only(backends.iter().map(|s| s.to_string())),
            host_envs: SetOrAll::only(host_envs.iter().map(|s| s.to_string())),
            mcp_hosts: SetOrAll::only(mcp_hosts.iter().map(|s| s.to_string())),
            tools: crate::permission::AllowMap::allow_all(),
            pod_modify: PodModifyCap::Memories,
            dispatch: DispatchCap::WithinScope,
            behaviors: BehaviorOpsCap::Read,
            escalation: crate::permission::Escalation::None,
        }
    }

    #[test]
    fn top_level_create_accepts_pod_defaults() {
        let pod = pod_with_two_backends_and_two_envs();
        let r = resolve_bindings_choice(&pod, None, None).unwrap();
        assert_eq!(r.backend_name, "anthropic");
        assert_eq!(r.host_env.len(), 1);
        assert_eq!(r.shared_host_names, vec!["fetch".to_string()]);
    }

    #[test]
    fn dispatched_child_rejected_for_backend_outside_parent_scope() {
        let pod = pod_with_two_backends_and_two_envs();
        // Parent's scope allows only anthropic; child asks for openai.
        let parent = parent_scope_only(&["anthropic"], &["wide", "narrow"], &["fetch", "search"]);
        let req = ThreadBindingsRequest {
            backend: Some("openai".into()),
            host_env: None,
            mcp_hosts: None,
        };
        let err = resolve_bindings_choice(&pod, Some(req), Some(&parent)).unwrap_err();
        assert!(err.contains("scope.backends"), "err was: {err}");
    }

    #[test]
    fn dispatched_child_rejected_for_host_env_outside_parent_scope() {
        let pod = pod_with_two_backends_and_two_envs();
        let parent = parent_scope_only(&["anthropic"], &["narrow"], &["fetch"]);
        let req = ThreadBindingsRequest {
            backend: None,
            host_env: Some(vec!["wide".into()]),
            mcp_hosts: None,
        };
        let err = resolve_bindings_choice(&pod, Some(req), Some(&parent)).unwrap_err();
        assert!(err.contains("scope.host_envs"), "err was: {err}");
    }

    #[test]
    fn dispatched_child_rejected_for_mcp_host_outside_parent_scope() {
        let pod = pod_with_two_backends_and_two_envs();
        let parent = parent_scope_only(&["anthropic"], &["wide"], &["fetch"]);
        let req = ThreadBindingsRequest {
            backend: None,
            host_env: None,
            mcp_hosts: Some(vec!["search".into()]),
        };
        let err = resolve_bindings_choice(&pod, Some(req), Some(&parent)).unwrap_err();
        assert!(err.contains("scope.mcp_hosts"), "err was: {err}");
    }

    #[test]
    fn dispatched_child_accepted_when_within_parent_scope() {
        let pod = pod_with_two_backends_and_two_envs();
        let parent = parent_scope_only(&["anthropic"], &["wide", "narrow"], &["fetch", "search"]);
        let req = ThreadBindingsRequest {
            backend: Some("anthropic".into()),
            host_env: Some(vec!["narrow".into()]),
            mcp_hosts: Some(vec!["search".into()]),
        };
        let r = resolve_bindings_choice(&pod, Some(req), Some(&parent)).unwrap();
        assert_eq!(r.backend_name, "anthropic");
        assert_eq!(r.shared_host_names, vec!["search".to_string()]);
    }

    #[test]
    fn behavior_scope_filter_drops_pod_default_mcp_hosts() {
        // The original bug: a behavior with `[scope] mcp_hosts = []`
        // inherits the pod's default shared-MCP list because the
        // behavior fire path doesn't narrow bindings. The helper
        // covers that gap by filtering pod-defaults that exceed the
        // scope.
        let pod = pod_with_two_backends_and_two_envs();
        let mut resolved = resolve_bindings_choice(&pod, None, None).unwrap();
        assert_eq!(resolved.shared_host_names, vec!["fetch".to_string()]);
        let scope = parent_scope_only(&["anthropic"], &["wide"], &[]);
        narrow_resolved_bindings_to_scope(&mut resolved, &scope).unwrap();
        assert!(resolved.shared_host_names.is_empty());
    }

    #[test]
    fn behavior_scope_filter_drops_out_of_scope_host_envs() {
        let pod = pod_with_two_backends_and_two_envs();
        let mut resolved = resolve_bindings_choice(
            &pod,
            Some(ThreadBindingsRequest {
                backend: None,
                host_env: Some(vec!["wide".into(), "narrow".into()]),
                mcp_hosts: None,
            }),
            None,
        )
        .unwrap();
        let scope = parent_scope_only(&["anthropic"], &["narrow"], &["fetch"]);
        narrow_resolved_bindings_to_scope(&mut resolved, &scope).unwrap();
        assert_eq!(resolved.host_env.len(), 1);
        assert!(matches!(
            &resolved.host_env[0],
            HostEnvBinding::Named { name } if name == "narrow"
        ));
    }

    #[test]
    fn behavior_scope_filter_errors_on_out_of_scope_backend() {
        // Backend is singular — silent filter would leave the thread
        // with no backend, which is worse than a clear error pointing
        // the author at the conflict.
        let pod = pod_with_two_backends_and_two_envs();
        let mut resolved = resolve_bindings_choice(&pod, None, None).unwrap();
        assert_eq!(resolved.backend_name, "anthropic");
        let scope = parent_scope_only(&["openai"], &["wide"], &["fetch"]);
        let err = narrow_resolved_bindings_to_scope(&mut resolved, &scope).unwrap_err();
        assert!(err.contains("scope.backends"), "err was: {err}");
    }

    #[test]
    fn behavior_scope_filter_passes_when_bindings_within_scope() {
        let pod = pod_with_two_backends_and_two_envs();
        let mut resolved = resolve_bindings_choice(&pod, None, None).unwrap();
        let scope = parent_scope_only(&["anthropic"], &["wide"], &["fetch"]);
        narrow_resolved_bindings_to_scope(&mut resolved, &scope).unwrap();
        assert_eq!(resolved.backend_name, "anthropic");
        assert_eq!(resolved.host_env.len(), 1);
        assert_eq!(resolved.shared_host_names, vec!["fetch".to_string()]);
    }
}
