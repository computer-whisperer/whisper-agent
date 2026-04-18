//! Resolve a thread's binding-side choices (backend, host env, shared
//! MCPs) from a pod's defaults plus an optional request, and validate
//! every pick against the pod's `[allow]` cap. Pure — no scheduler
//! state — so the cap-enforcement rules live in one readable place.

use whisper_agent_protocol::{HostEnvBinding, ThreadBindingsRequest};

use crate::pod::Pod;

/// Resolved binding-side choices for a fresh thread. Validated against
/// the pod's `[allow]` cap before construction.
pub(super) struct ResolvedBindings {
    /// Backend catalog name. Empty string = "use server default".
    pub(super) backend_name: String,
    /// Persisted on the thread. `None` when the pod declares no host
    /// envs (threads there get shared MCPs only). `Named` references
    /// a pod `[[allow.host_env]]` entry; `Inline` carries its own
    /// (provider, spec) for ad-hoc threads.
    pub(super) host_env: Option<HostEnvBinding>,
    /// Catalog names of shared MCP hosts the thread is bound to.
    pub(super) shared_host_names: Vec<String>,
}

/// Layer the request on top of pod `thread_defaults`, then verify each
/// chosen value against the pod's `[allow]` table. Inline host-env
/// requests carry both provider + spec, so no name lookup is needed.
pub(super) fn resolve_bindings_choice(
    pod: &Pod,
    request: Option<ThreadBindingsRequest>,
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

    // Every user-originated binding resolves to a named entry in the
    // pod's allow.host_env table. Unknown names are rejected; the
    // pod's allow list is authoritative.
    let chosen_name: Option<String> = match request.host_env {
        Some(name) => Some(name),
        None if defaults.host_env.is_empty() => None,
        None => Some(defaults.host_env.clone()),
    };
    let host_env: Option<HostEnvBinding> = match chosen_name {
        None => None,
        Some(name) => {
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
            Some(HostEnvBinding::Named { name })
        }
    };

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
    }

    Ok(ResolvedBindings {
        backend_name,
        host_env,
        shared_host_names,
    })
}
