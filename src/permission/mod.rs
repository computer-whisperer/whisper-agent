//! Permission model: `PermissionScope` and its sub-types.
//!
//! One type, used at three layers:
//!
//! - **Pod.allow** — the ceiling for anything running in the pod.
//! - **Thread effective scope** — derived from the pod's scope, narrowed by
//!   the thread's bindings and config.
//! - **Caller scope** — what a Function invocation runs under, supplied or
//!   derived per caller surface.
//!
//! See `docs/design_functions.md` for the full design rationale.
//!
//! `Disposition` and `AllowMap<T>` are re-exported from
//! `whisper-agent-protocol::permission` so server and webui share the
//! canonical wire shapes; everything else lives here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub use whisper_agent_protocol::permission::{AllowMap, Disposition};

// ---------------------------------------------------------------------------
// Per-pod operation enumerations
// ---------------------------------------------------------------------------

/// Thread-level operations a pod scope may admit.
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum ThreadOp {
    Create,
    Read,
    Send,
    Cancel,
    Compact,
    Rebind,
    Archive,
}

/// Behavior-level operations a pod scope may admit.
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum BehaviorOp {
    Create,
    Modify,
    Delete,
    Run,
}

/// Pod-config-level operations a pod scope may admit.
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum PodConfigOp {
    Read,
    Modify,
}

/// Operations a scope admits within a single pod.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PodOps {
    pub threads: AllowMap<ThreadOp>,
    pub behaviors: AllowMap<BehaviorOp>,
    pub config: AllowMap<PodConfigOp>,
}

impl PodOps {
    pub fn allow_all() -> Self {
        Self {
            threads: AllowMap::allow_all(),
            behaviors: AllowMap::allow_all(),
            config: AllowMap::allow_all(),
        }
    }

    pub fn deny_all() -> Self {
        Self {
            threads: AllowMap::deny_all(),
            behaviors: AllowMap::deny_all(),
            config: AllowMap::deny_all(),
        }
    }

    pub fn narrow(&self, other: &Self) -> Self {
        Self {
            threads: self.threads.narrow(&other.threads),
            behaviors: self.behaviors.narrow(&other.behaviors),
            config: self.config.narrow(&other.config),
        }
    }
}

// ---------------------------------------------------------------------------
// PodsScope: "all pods" (including future ones) or "this enumerated set"
// ---------------------------------------------------------------------------

pub type PodId = String;

/// Which pods a scope may act on, and with what operations.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PodsScope {
    /// Caller may act on any pod — including pods created after scope
    /// construction — with the given `PodOps`. Used for scheduler-internal
    /// callers and full-trust WS clients.
    All { ops: PodOps },
    /// Caller may act only on the enumerated pods, with per-pod operation
    /// dispositions. An empty map denies all pod access.
    Per { pods: BTreeMap<PodId, PodOps> },
}

impl PodsScope {
    /// Deny all pod access.
    pub fn deny_all() -> Self {
        Self::Per {
            pods: BTreeMap::new(),
        }
    }

    /// Admit every pod, present or future, with fully-allowed ops.
    pub fn allow_all() -> Self {
        Self::All {
            ops: PodOps::allow_all(),
        }
    }

    /// Return the `PodOps` this scope applies to `pod_id`, if it admits the
    /// pod at all. `None` ⇒ the scope doesn't cover this pod (caller may not
    /// act on it).
    pub fn ops_for(&self, pod_id: &str) -> Option<&PodOps> {
        match self {
            Self::All { ops } => Some(ops),
            Self::Per { pods } => pods.get(pod_id),
        }
    }

    /// Narrowing composition. `All ∩ All = All` (ops intersected). Any
    /// `Per` on either side collapses to `Per` with only the enumerated
    /// pods from whichever side was `Per` (both sides' `PodOps` intersected
    /// per pod; missing-on-one-side falls back to that side's `All.ops`).
    pub fn narrow(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::All { ops: a }, Self::All { ops: b }) => Self::All { ops: a.narrow(b) },
            (Self::All { ops: a }, Self::Per { pods: b })
            | (Self::Per { pods: b }, Self::All { ops: a }) => {
                let pods = b
                    .iter()
                    .map(|(k, v)| (k.clone(), v.narrow(a)))
                    .collect();
                Self::Per { pods }
            }
            (Self::Per { pods: a }, Self::Per { pods: b }) => {
                // Intersection of keys; values narrowed.
                let mut pods = BTreeMap::new();
                for (k, va) in a {
                    if let Some(vb) = b.get(k) {
                        pods.insert(k.clone(), va.narrow(vb));
                    }
                }
                Self::Per { pods }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PermissionScope
// ---------------------------------------------------------------------------

pub type BackendName = String;
pub type ProviderName = String;
pub type HostName = String;
pub type ToolName = String;

/// The complete permission scope — one type used at pod, thread, and caller
/// layers.
///
/// Bindings (`backends`, `host_envs`, `mcp_hosts`) are plain `Vec<String>`
/// for the interim — implicit `Allow` for listed items, implicit `Deny`
/// default. No current use case wants `AllowWithPrompt` for bindings; if
/// one arises we upgrade those fields to `AllowMap<T>`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PermissionScope {
    pub backends: Vec<BackendName>,
    pub host_envs: Vec<ProviderName>,
    pub mcp_hosts: Vec<HostName>,
    pub tools: AllowMap<ToolName>,
    pub pods: PodsScope,
}

impl PermissionScope {
    /// Fully permissive scope — everything admitted with `Allow`. Used for
    /// hardcoded-full-trust WS clients during Phase 1b, and for
    /// scheduler-internal callers.
    pub fn allow_all() -> Self {
        Self {
            backends: Vec::new(),
            host_envs: Vec::new(),
            mcp_hosts: Vec::new(),
            tools: AllowMap::allow_all(),
            pods: PodsScope::allow_all(),
        }
    }

    /// Fully restrictive scope — nothing admitted. Useful as a starting
    /// point for narrowing or as a safe default when construction fails.
    pub fn deny_all() -> Self {
        Self {
            backends: Vec::new(),
            host_envs: Vec::new(),
            mcp_hosts: Vec::new(),
            tools: AllowMap::deny_all(),
            pods: PodsScope::deny_all(),
        }
    }

    /// Narrow `self` with `other`: pick the more restrictive disposition /
    /// set-intersection on each field. Never widens.
    pub fn narrow(&self, other: &Self) -> Self {
        fn intersect_vec(a: &[String], b: &[String]) -> Vec<String> {
            // For the interim Vec<String> bindings, "Allow" means "in list".
            // Narrowing is set intersection — only items present in both
            // survive, since Deny is the implicit default.
            let bset: BTreeSet<&String> = b.iter().collect();
            a.iter().filter(|x| bset.contains(x)).cloned().collect()
        }

        Self {
            backends: intersect_vec(&self.backends, &other.backends),
            host_envs: intersect_vec(&self.host_envs, &other.host_envs),
            mcp_hosts: intersect_vec(&self.mcp_hosts, &other.mcp_hosts),
            tools: self.tools.narrow(&other.tools),
            pods: self.pods.narrow(&other.pods),
        }
    }

    /// Admission check: may the caller act on `pod_id` at all, and with
    /// what operations?
    pub fn pod_ops(&self, pod_id: &str) -> Option<&PodOps> {
        self.pods.ops_for(pod_id)
    }

    /// Admission check for a specific thread-level operation within a pod.
    pub fn thread_op(&self, pod_id: &str, op: ThreadOp) -> Disposition {
        self.pods
            .ops_for(pod_id)
            .map(|o| o.threads.disposition(&op))
            .unwrap_or(Disposition::Deny)
    }

    /// Admission check for a behavior-level operation within a pod.
    pub fn behavior_op(&self, pod_id: &str, op: BehaviorOp) -> Disposition {
        self.pods
            .ops_for(pod_id)
            .map(|o| o.behaviors.disposition(&op))
            .unwrap_or(Disposition::Deny)
    }

    /// Admission check for a pod-config-level operation within a pod.
    pub fn pod_config_op(&self, pod_id: &str, op: PodConfigOp) -> Disposition {
        self.pods
            .ops_for(pod_id)
            .map(|o| o.config.disposition(&op))
            .unwrap_or(Disposition::Deny)
    }

    /// Admission check for a tool invocation.
    pub fn tool(&self, name: &str) -> Disposition {
        self.tools.disposition(&name.to_string())
    }

    /// Admission check for binding to a backend.
    pub fn backend(&self, name: &str) -> Disposition {
        if self.backends.iter().any(|b| b == name) {
            Disposition::Allow
        } else {
            Disposition::Deny
        }
    }

    /// Admission check for using a host-env provider.
    pub fn host_env(&self, name: &str) -> Disposition {
        if self.host_envs.iter().any(|p| p == name) {
            Disposition::Allow
        } else {
            Disposition::Deny
        }
    }

    /// Admission check for reaching an MCP host.
    pub fn mcp_host(&self, name: &str) -> Disposition {
        if self.mcp_hosts.iter().any(|h| h == name) {
            Disposition::Allow
        } else {
            Disposition::Deny
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Disposition / AllowMap tests live in `whisper-agent-protocol::permission`
    // — the types are owned there. These tests cover only the main-crate
    // types (PodsScope, PermissionScope, etc.).

    #[test]
    fn pods_scope_all_admits_any_pod() {
        let s = PodsScope::allow_all();
        assert!(s.ops_for("anything").is_some());
    }

    #[test]
    fn pods_scope_per_admits_only_enumerated() {
        let mut pods = BTreeMap::new();
        pods.insert("alpha".to_string(), PodOps::allow_all());
        let s = PodsScope::Per { pods };
        assert!(s.ops_for("alpha").is_some());
        assert!(s.ops_for("beta").is_none());
    }

    #[test]
    fn pods_scope_narrow_all_per_keeps_per_keys_only() {
        let all = PodsScope::allow_all();
        let mut pods = BTreeMap::new();
        pods.insert("alpha".to_string(), PodOps::allow_all());
        let per = PodsScope::Per { pods };
        let r = all.narrow(&per);
        match r {
            PodsScope::Per { pods } => {
                assert!(pods.contains_key("alpha"));
                assert_eq!(pods.len(), 1);
            }
            _ => panic!("expected Per"),
        }
    }

    #[test]
    fn pods_scope_narrow_per_per_intersects_keys() {
        let mut a_pods = BTreeMap::new();
        a_pods.insert("alpha".to_string(), PodOps::allow_all());
        a_pods.insert("beta".to_string(), PodOps::allow_all());
        let a = PodsScope::Per { pods: a_pods };

        let mut b_pods = BTreeMap::new();
        b_pods.insert("beta".to_string(), PodOps::allow_all());
        b_pods.insert("gamma".to_string(), PodOps::allow_all());
        let b = PodsScope::Per { pods: b_pods };

        let r = a.narrow(&b);
        match r {
            PodsScope::Per { pods } => {
                assert_eq!(pods.keys().collect::<Vec<_>>(), vec!["beta"]);
            }
            _ => panic!("expected Per"),
        }
    }

    #[test]
    fn permission_scope_narrow_intersects_bindings() {
        let a = PermissionScope {
            backends: vec!["anthropic".into(), "openai".into()],
            host_envs: vec![],
            mcp_hosts: vec!["fetch".into()],
            tools: AllowMap::allow_all(),
            pods: PodsScope::allow_all(),
        };
        let b = PermissionScope {
            backends: vec!["anthropic".into(), "gemini".into()],
            host_envs: vec![],
            mcp_hosts: vec!["fetch".into(), "search".into()],
            tools: AllowMap::allow_all(),
            pods: PodsScope::allow_all(),
        };
        let r = a.narrow(&b);
        assert_eq!(r.backends, vec!["anthropic".to_string()]);
        assert_eq!(r.mcp_hosts, vec!["fetch".to_string()]);
    }

    #[test]
    fn permission_scope_allow_all_admits_everything() {
        let s = PermissionScope::allow_all();
        assert_eq!(s.tool("bash"), Disposition::Allow);
        assert_eq!(
            s.thread_op("any-pod", ThreadOp::Create),
            Disposition::Allow
        );
    }

    #[test]
    fn permission_scope_deny_all_admits_nothing() {
        let s = PermissionScope::deny_all();
        assert_eq!(s.tool("bash"), Disposition::Deny);
        assert_eq!(s.thread_op("any-pod", ThreadOp::Create), Disposition::Deny);
        assert!(s.pod_ops("any-pod").is_none());
    }
}
