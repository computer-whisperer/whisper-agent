//! Permission-model wire types shared between server and webui.
//!
//! The central type is [`Scope`] — one struct used at every layer of
//! the capability stack (pod cap, thread effective permission, caller
//! permission for a Function registration). [`AllowMap`], [`SetOrAll`],
//! and the typed capability enums are the building blocks.
//!
//! Approval is expressed as scope-widening, not per-call prompting —
//! `Disposition` is a binary `Allow | Deny`. The "widenable by request"
//! property lives on [`Scope::escalation`], not on individual tool
//! dispositions.
//!
//! See `docs/design_permissions_rework.md` for the full design.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// What the scope admits for a given item.
///
/// The derive-order `Allow < Deny` is the restrictiveness ordering
/// used by narrowing composition — the *more* restrictive disposition
/// wins when two scopes intersect.
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    #[default]
    Allow,
    Deny,
}

impl Disposition {
    /// Combine two dispositions under narrowing composition: more
    /// restrictive wins.
    pub fn narrow(self, other: Self) -> Self {
        self.max(other)
    }

    /// Does this disposition admit the operation?
    pub fn admits(self) -> bool {
        matches!(self, Self::Allow)
    }
}

/// Per-item disposition with a default for unlisted items.
///
/// Narrowing composition picks the more restrictive disposition per
/// item: for any `k`, the result's disposition is
/// `narrow(self.disposition(k), other.disposition(k))`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AllowMap<T: Ord + Clone> {
    #[serde(default)]
    pub default: Disposition,
    #[serde(default = "BTreeMap::new", skip_serializing_if = "BTreeMap::is_empty")]
    pub overrides: BTreeMap<T, Disposition>,
}

impl<T: Ord + Clone> AllowMap<T> {
    /// All items admitted with `Allow`.
    pub fn allow_all() -> Self {
        Self {
            default: Disposition::Allow,
            overrides: BTreeMap::new(),
        }
    }

    /// All items denied.
    pub fn deny_all() -> Self {
        Self {
            default: Disposition::Deny,
            overrides: BTreeMap::new(),
        }
    }

    /// Disposition for `item`: override if present, otherwise default.
    pub fn disposition(&self, item: &T) -> Disposition {
        self.overrides.get(item).copied().unwrap_or(self.default)
    }

    /// Narrow `self` with `other`: more-restrictive disposition wins
    /// per item. Post-narrow default is `narrow(self.default, other.default)`.
    /// Overrides that equal the post-narrow default are elided to keep
    /// the map small.
    pub fn narrow(&self, other: &Self) -> Self {
        let default = self.default.narrow(other.default);
        let all_keys: std::collections::BTreeSet<&T> = self
            .overrides
            .keys()
            .chain(other.overrides.keys())
            .collect();
        let mut overrides = BTreeMap::new();
        for k in all_keys {
            let merged = self.disposition(k).narrow(other.disposition(k));
            if merged != default {
                overrides.insert(k.clone(), merged);
            }
        }
        Self { default, overrides }
    }

    /// Upgrade a specific item's disposition to `Allow`, ignoring
    /// whatever the underlying default/override said. Used when an
    /// escalation grant widens a thread's scope for a specific tool.
    pub fn set_allow(&mut self, item: T) {
        if self.default == Disposition::Allow {
            self.overrides.remove(&item);
        } else {
            self.overrides.insert(item, Disposition::Allow);
        }
    }
}

impl<T: Ord + Clone> Default for AllowMap<T> {
    fn default() -> Self {
        Self::allow_all()
    }
}

// ---------------------------------------------------------------------------
// SetOrAll<T>: resource-name admission
// ---------------------------------------------------------------------------

/// Which names of a resource category a scope admits. `All` is the
/// unlisted-inclusive variant — it admits every name in the pod catalog,
/// including names added after scope construction. `Only` is an explicit
/// whitelist; names not in the set are denied.
///
/// Defaults to `Only` with an empty set (deny-by-default), matching the
/// prior `Vec<String>` semantics from the pre-rework permission type.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SetOrAll<T: Ord + Clone> {
    All,
    Only { items: BTreeSet<T> },
}

impl<T: Ord + Clone> SetOrAll<T> {
    /// Admits every name, including ones added to the pod catalog later.
    pub fn all() -> Self {
        Self::All
    }

    /// Admits exactly the given items.
    pub fn only(items: impl IntoIterator<Item = T>) -> Self {
        Self::Only {
            items: items.into_iter().collect(),
        }
    }

    /// Admits nothing. Equivalent to `Only` with an empty set.
    pub fn none() -> Self {
        Self::Only {
            items: BTreeSet::new(),
        }
    }

    /// Does this set admit `item`?
    pub fn admits(&self, item: &T) -> bool {
        match self {
            Self::All => true,
            Self::Only { items } => items.contains(item),
        }
    }

    /// Narrow `self` with `other`: more-restrictive wins.
    ///
    /// - `All ∩ All = All`
    /// - `All ∩ Only(s) = Only(s)`
    /// - `Only(a) ∩ Only(b) = Only(a ∩ b)`
    pub fn narrow(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::All, Self::All) => Self::All,
            (Self::All, Self::Only { items }) | (Self::Only { items }, Self::All) => Self::Only {
                items: items.clone(),
            },
            (Self::Only { items: a }, Self::Only { items: b }) => Self::Only {
                items: a.intersection(b).cloned().collect(),
            },
        }
    }
}

impl<T: Ord + Clone> Default for SetOrAll<T> {
    fn default() -> Self {
        Self::none()
    }
}

// ---------------------------------------------------------------------------
// Typed capability enums
// ---------------------------------------------------------------------------

/// Pod-directory modification capability. Governs *writes* — reads use
/// the broader pod filename allowlist. See
/// `docs/design_permissions_rework.md`.
///
/// Ordered `None < Memories < Content < ModifyAllow` — lower is more
/// restrictive. Narrowing picks the minimum.
///
/// Write-path rubric:
///
/// | Level         | Write-admits                                                  |
/// |---------------|---------------------------------------------------------------|
/// | `None`        | nothing                                                       |
/// | `Memories`    | `memory/**`                                                   |
/// | `Content`     | `Memories` + `system_prompt.md`, `behaviors/*/prompt.md`, `behaviors/*/behavior.toml` |
/// | `ModifyAllow` | `Content` + `pod.toml`                                        |
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum PodModifyCap {
    #[default]
    None,
    Memories,
    Content,
    ModifyAllow,
}

impl PodModifyCap {
    /// Narrow: more restrictive wins.
    pub fn narrow(self, other: Self) -> Self {
        self.min(other)
    }

    /// Does this cap admit reads or writes to `pod_rel_path`? The path
    /// is treated as pod-relative; callers must normalize out leading
    /// slashes / `./` / traversal components before invoking. Returns
    /// `false` on any path the active level doesn't cover.
    ///
    /// This is the single source of truth for "which pod files does each
    /// cap tier reach." Adding a new pod-surface file (say,
    /// `hooks/on_start.lua`) means extending this helper, not patching
    /// every pod-editing tool site.
    pub fn admits(self, pod_rel_path: &str) -> bool {
        // Strip leading slashes but not internal `..`; callers reject
        // traversal before calling. Empty path → deny.
        let p = pod_rel_path.trim_start_matches('/');
        if p.is_empty() {
            return false;
        }
        match self {
            Self::None => false,
            Self::Memories => p == "memory" || p.starts_with("memory/"),
            Self::Content => {
                if Self::Memories.admits(p) {
                    return true;
                }
                if p == "system_prompt.md" {
                    return true;
                }
                if let Some(rest) = p.strip_prefix("behaviors/") {
                    return rest.ends_with("/prompt.md") || rest.ends_with("/behavior.toml");
                }
                false
            }
            Self::ModifyAllow => {
                if Self::Content.admits(p) {
                    return true;
                }
                p == "pod.toml"
            }
        }
    }
}

/// Dispatch capability: may this scope spawn child threads via
/// `dispatch_thread`?
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum DispatchCap {
    #[default]
    None,
    /// May spawn children with `scope ≤ self.scope`. (The narrowing check
    /// runs at dispatch time; this cap just admits the operation.)
    WithinScope,
}

impl DispatchCap {
    pub fn narrow(self, other: Self) -> Self {
        self.min(other)
    }
}

/// Behavior-subsystem capability.
///
/// Ordered `None < Read < AuthorNarrower < AuthorAny`.
///
/// - `None`: no access at all.
/// - `Read`: can read behavior configs and prompts.
/// - `AuthorNarrower`: can create/modify behaviors whose declared scope is
///   strictly narrower than the authoring thread's scope. The intended use:
///   a smart/trusted thread authoring a narrower behavior for cheaper models
///   to run autonomously.
/// - `AuthorAny`: can create/modify behaviors with any scope ≤ pod.allow.
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum BehaviorOpsCap {
    #[default]
    None,
    Read,
    AuthorNarrower,
    AuthorAny,
}

impl BehaviorOpsCap {
    pub fn narrow(self, other: Self) -> Self {
        self.min(other)
    }
}

/// Approval channel attached to a scope. Only `Interactive` scopes can
/// reach the `request_escalation` tool; autonomous scopes hard-deny
/// out-of-scope operations rather than pending for approval.
///
/// Narrowing: same `via_conn` survives; any mismatch or `None` on either
/// side collapses to `None`. A child thread inherits the parent's channel
/// only if the parent admitted it explicitly in the dispatch request.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Escalation {
    Interactive {
        via_conn: u64,
    },
    #[default]
    None,
}

impl Escalation {
    pub fn narrow(self, other: Self) -> Self {
        match (self, other) {
            (Self::Interactive { via_conn: a }, Self::Interactive { via_conn: b }) if a == b => {
                Self::Interactive { via_conn: a }
            }
            _ => Self::None,
        }
    }

    pub fn is_interactive(self) -> bool {
        matches!(self, Self::Interactive { .. })
    }
}

// ---------------------------------------------------------------------------
// Escalation request / decision
// ---------------------------------------------------------------------------

/// A model-initiated ask to widen its own thread's [`Scope`]. Surfaced on
/// the wire as the `request_escalation` tool's typed-union argument shape
/// and as `ServerToClient::EscalationRequested`'s payload, so the approval
/// UI can render a specific widening rather than stringly-typed JSON.
///
/// Each variant names the smallest widening that actually buys the model
/// something: a specific tool, or a single-step cap bump. "Add this
/// backend / host-env / mcp host" widenings are captured by editing
/// `pod.toml` instead — that's a `pod_write_file` call which itself
/// requires `PodModifyCap::ModifyAllow` and its own approval path, so the
/// escalation type doesn't need variants for them.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "variant", rename_all = "snake_case")]
pub enum EscalationRequest {
    /// Widen `scope.tools` to admit `name`. Applies to the requesting
    /// thread only; does not edit `pod.toml`.
    AddTool { name: String },
    /// Raise `scope.pod_modify` to `target`. `target` must be strictly
    /// greater than the current cap and bounded by the pod ceiling.
    RaisePodModify { target: PodModifyCap },
    /// Raise `scope.behaviors` to `target`. Same bounds as
    /// `raise_pod_modify`.
    RaiseBehaviors { target: BehaviorOpsCap },
    /// Raise `scope.dispatch` to `target`. Same bounds.
    RaiseDispatch { target: DispatchCap },
}

/// User's answer to an `EscalationRequest`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EscalationDecision {
    Approve,
    Reject,
}

// ---------------------------------------------------------------------------
// Scope
// ---------------------------------------------------------------------------

/// The unified permission scope. One type used at every layer —
/// pod.allow ceiling, thread effective permission, caller permission
/// for a Function registration.
///
/// **Scope is what a thread is *permitted to do*, not what it's
/// *currently bound to*.** A thread whose `host_envs` admits `"narrow"`
/// can author a behavior using `"narrow"` even if the thread hasn't
/// bound there — it's within scope. Current bindings are runtime state;
/// scope is permission.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct Scope {
    #[serde(default)]
    pub backends: SetOrAll<String>,
    #[serde(default)]
    pub host_envs: SetOrAll<String>,
    #[serde(default)]
    pub mcp_hosts: SetOrAll<String>,
    #[serde(default = "AllowMap::allow_all")]
    pub tools: AllowMap<String>,
    #[serde(default)]
    pub pod_modify: PodModifyCap,
    #[serde(default)]
    pub dispatch: DispatchCap,
    #[serde(default)]
    pub behaviors: BehaviorOpsCap,
    #[serde(default)]
    pub escalation: Escalation,
}

impl Scope {
    /// Fully permissive scope: every resource admitted, every cap at
    /// its highest level, `escalation: None` (channels are attached
    /// explicitly, not inherited from `allow_all`).
    pub fn allow_all() -> Self {
        Self {
            backends: SetOrAll::all(),
            host_envs: SetOrAll::all(),
            mcp_hosts: SetOrAll::all(),
            tools: AllowMap::allow_all(),
            pod_modify: PodModifyCap::ModifyAllow,
            dispatch: DispatchCap::WithinScope,
            behaviors: BehaviorOpsCap::AuthorAny,
            escalation: Escalation::None,
        }
    }

    /// Fully restrictive scope: nothing admitted anywhere.
    pub fn deny_all() -> Self {
        Self {
            backends: SetOrAll::none(),
            host_envs: SetOrAll::none(),
            mcp_hosts: SetOrAll::none(),
            tools: AllowMap::deny_all(),
            pod_modify: PodModifyCap::None,
            dispatch: DispatchCap::None,
            behaviors: BehaviorOpsCap::None,
            escalation: Escalation::None,
        }
    }

    /// Narrow `self` with `other`: the more-restrictive choice wins
    /// on every field. The core permissions invariant —
    /// `child.scope = parent.scope.narrow(requested)` — relies on
    /// this never widening.
    pub fn narrow(&self, other: &Self) -> Self {
        Self {
            backends: self.backends.narrow(&other.backends),
            host_envs: self.host_envs.narrow(&other.host_envs),
            mcp_hosts: self.mcp_hosts.narrow(&other.mcp_hosts),
            tools: self.tools.narrow(&other.tools),
            pod_modify: self.pod_modify.narrow(other.pod_modify),
            dispatch: self.dispatch.narrow(other.dispatch),
            behaviors: self.behaviors.narrow(other.behaviors),
            escalation: self.escalation.narrow(other.escalation),
        }
    }

    /// Check whether applying `request` would widen `self` within the
    /// `ceiling` scope. Returns `Ok(())` when the grant is valid (target
    /// is strictly more permissive than current, and bounded by the
    /// ceiling), or a human-readable reason string otherwise.
    ///
    /// The ceiling is typically `pod.allow`-derived — the pod's hardest
    /// cap. An approver cannot grant past it because a later narrow by
    /// pod.allow would immediately revert the grant anyway; denying at
    /// grant time gives the user a clearer explanation.
    pub fn check_escalation(
        &self,
        request: &EscalationRequest,
        ceiling: &Self,
    ) -> Result<(), String> {
        match request {
            EscalationRequest::AddTool { name } => {
                if self.tools.disposition(name).admits() {
                    return Err(format!("tool `{name}` already admitted by scope"));
                }
                if !ceiling.tools.disposition(name).admits() {
                    return Err(format!("tool `{name}` not admitted by pod ceiling"));
                }
                Ok(())
            }
            EscalationRequest::RaisePodModify { target } => {
                if *target <= self.pod_modify {
                    return Err(format!(
                        "pod_modify cap is already {:?}; request to raise to {target:?} is not a widening",
                        self.pod_modify
                    ));
                }
                if *target > ceiling.pod_modify {
                    return Err(format!(
                        "pod_modify target {target:?} exceeds pod ceiling {:?}",
                        ceiling.pod_modify
                    ));
                }
                Ok(())
            }
            EscalationRequest::RaiseBehaviors { target } => {
                if *target <= self.behaviors {
                    return Err(format!(
                        "behaviors cap is already {:?}; request to raise to {target:?} is not a widening",
                        self.behaviors
                    ));
                }
                if *target > ceiling.behaviors {
                    return Err(format!(
                        "behaviors target {target:?} exceeds pod ceiling {:?}",
                        ceiling.behaviors
                    ));
                }
                Ok(())
            }
            EscalationRequest::RaiseDispatch { target } => {
                if *target <= self.dispatch {
                    return Err(format!(
                        "dispatch cap is already {:?}; request to raise to {target:?} is not a widening",
                        self.dispatch
                    ));
                }
                if *target > ceiling.dispatch {
                    return Err(format!(
                        "dispatch target {target:?} exceeds pod ceiling {:?}",
                        ceiling.dispatch
                    ));
                }
                Ok(())
            }
        }
    }

    /// Apply `request` to `self` unconditionally. Callers should have
    /// validated the widening via [`check_escalation`] first — this
    /// helper just mutates the relevant field.
    pub fn apply_escalation(&mut self, request: &EscalationRequest) {
        match request {
            EscalationRequest::AddTool { name } => self.tools.set_allow(name.clone()),
            EscalationRequest::RaisePodModify { target } => self.pod_modify = *target,
            EscalationRequest::RaiseBehaviors { target } => self.behaviors = *target,
            EscalationRequest::RaiseDispatch { target } => self.dispatch = *target,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disposition_ordering() {
        assert!(Disposition::Allow < Disposition::Deny);
    }

    #[test]
    fn allowmap_narrow_picks_more_restrictive() {
        use Disposition::*;
        let a: AllowMap<String> = AllowMap {
            default: Allow,
            overrides: [("bash".into(), Deny)].into_iter().collect(),
        };
        let b: AllowMap<String> = AllowMap {
            default: Allow,
            overrides: [("rm".into(), Deny)].into_iter().collect(),
        };
        let r = a.narrow(&b);
        assert_eq!(r.default, Allow);
        assert_eq!(r.disposition(&"bash".into()), Deny);
        assert_eq!(r.disposition(&"rm".into()), Deny);
        assert_eq!(r.disposition(&"ls".into()), Allow);
    }

    #[test]
    fn allowmap_set_allow_removes_override_when_default_is_allow() {
        use Disposition::*;
        let mut m: AllowMap<String> = AllowMap {
            default: Allow,
            overrides: [("bash".into(), Deny)].into_iter().collect(),
        };
        m.set_allow("bash".into());
        assert!(m.overrides.is_empty());
    }

    #[test]
    fn allowmap_set_allow_inserts_override_when_default_is_not_allow() {
        use Disposition::*;
        let mut m: AllowMap<String> = AllowMap {
            default: Deny,
            overrides: BTreeMap::new(),
        };
        m.set_allow("bash".into());
        assert_eq!(m.disposition(&"bash".into()), Allow);
    }

    // ---------- SetOrAll ----------

    #[test]
    fn setorall_all_admits_anything() {
        let s: SetOrAll<String> = SetOrAll::all();
        assert!(s.admits(&"foo".into()));
        assert!(s.admits(&"".into()));
    }

    #[test]
    fn setorall_only_admits_exactly_listed() {
        let s: SetOrAll<String> = SetOrAll::only(["anthropic".to_string(), "openai".to_string()]);
        assert!(s.admits(&"anthropic".into()));
        assert!(!s.admits(&"local-llama".into()));
    }

    #[test]
    fn setorall_default_is_deny() {
        let s: SetOrAll<String> = SetOrAll::default();
        assert!(!s.admits(&"anything".into()));
    }

    #[test]
    fn setorall_narrow_all_and_only_yields_only() {
        let all: SetOrAll<String> = SetOrAll::all();
        let only: SetOrAll<String> = SetOrAll::only(["a".to_string(), "b".to_string()]);
        let r = all.narrow(&only);
        match r {
            SetOrAll::Only { items } => {
                assert_eq!(items.len(), 2);
                assert!(items.contains("a"));
            }
            _ => panic!("expected Only after narrowing with Only"),
        }
    }

    #[test]
    fn setorall_narrow_only_intersects() {
        let a: SetOrAll<String> =
            SetOrAll::only(["a".to_string(), "b".to_string(), "c".to_string()]);
        let b: SetOrAll<String> =
            SetOrAll::only(["b".to_string(), "c".to_string(), "d".to_string()]);
        let r = a.narrow(&b);
        match r {
            SetOrAll::Only { items } => {
                let got: Vec<&String> = items.iter().collect();
                assert_eq!(got, vec![&"b".to_string(), &"c".to_string()]);
            }
            _ => panic!("expected Only"),
        }
    }

    #[test]
    fn setorall_narrow_is_associative() {
        let a: SetOrAll<String> = SetOrAll::all();
        let b: SetOrAll<String> =
            SetOrAll::only(["x".to_string(), "y".to_string(), "z".to_string()]);
        let c: SetOrAll<String> = SetOrAll::only(["y".to_string(), "z".to_string()]);
        let left = a.narrow(&b).narrow(&c);
        let right = a.narrow(&b.narrow(&c));
        assert_eq!(left, right);
    }

    // ---------- PodModifyCap ----------

    #[test]
    fn pod_modify_cap_ordering() {
        assert!(PodModifyCap::None < PodModifyCap::Memories);
        assert!(PodModifyCap::Memories < PodModifyCap::Content);
        assert!(PodModifyCap::Content < PodModifyCap::ModifyAllow);
    }

    #[test]
    fn pod_modify_cap_admits_memories() {
        let cap = PodModifyCap::Memories;
        assert!(cap.admits("memory/notes.md"));
        assert!(cap.admits("memory/subdir/deep.md"));
        assert!(cap.admits("memory"));
        assert!(!cap.admits("system_prompt.md"));
        assert!(!cap.admits("pod.toml"));
    }

    #[test]
    fn pod_modify_cap_admits_content_adds_prompts_and_behaviors() {
        let cap = PodModifyCap::Content;
        assert!(cap.admits("memory/notes.md"));
        assert!(cap.admits("system_prompt.md"));
        assert!(cap.admits("behaviors/cron-daily/prompt.md"));
        assert!(cap.admits("behaviors/cron-daily/behavior.toml"));
        assert!(!cap.admits("behaviors/cron-daily/other.txt"));
        assert!(!cap.admits("pod.toml"));
    }

    #[test]
    fn pod_modify_cap_admits_modify_allow_adds_pod_toml() {
        let cap = PodModifyCap::ModifyAllow;
        assert!(cap.admits("memory/notes.md"));
        assert!(cap.admits("system_prompt.md"));
        assert!(cap.admits("pod.toml"));
    }

    #[test]
    fn pod_modify_cap_none_denies_everything() {
        let cap = PodModifyCap::None;
        assert!(!cap.admits("memory/notes.md"));
        assert!(!cap.admits("pod.toml"));
    }

    #[test]
    fn pod_modify_cap_empty_path_denied() {
        assert!(!PodModifyCap::ModifyAllow.admits(""));
        assert!(!PodModifyCap::ModifyAllow.admits("/"));
    }

    #[test]
    fn pod_modify_cap_narrow_is_min() {
        assert_eq!(
            PodModifyCap::Content.narrow(PodModifyCap::Memories),
            PodModifyCap::Memories
        );
        assert_eq!(
            PodModifyCap::None.narrow(PodModifyCap::ModifyAllow),
            PodModifyCap::None
        );
    }

    // ---------- DispatchCap / BehaviorOpsCap ----------

    #[test]
    fn dispatch_cap_ordering_and_narrow() {
        assert!(DispatchCap::None < DispatchCap::WithinScope);
        assert_eq!(
            DispatchCap::WithinScope.narrow(DispatchCap::None),
            DispatchCap::None
        );
    }

    #[test]
    fn behavior_ops_cap_ordering() {
        assert!(BehaviorOpsCap::None < BehaviorOpsCap::Read);
        assert!(BehaviorOpsCap::Read < BehaviorOpsCap::AuthorNarrower);
        assert!(BehaviorOpsCap::AuthorNarrower < BehaviorOpsCap::AuthorAny);
    }

    // ---------- Escalation ----------

    #[test]
    fn escalation_narrow_keeps_matching_conn() {
        let a = Escalation::Interactive { via_conn: 7 };
        assert_eq!(a.narrow(a), a);
    }

    #[test]
    fn escalation_narrow_mismatched_conn_collapses() {
        let a = Escalation::Interactive { via_conn: 7 };
        let b = Escalation::Interactive { via_conn: 8 };
        assert_eq!(a.narrow(b), Escalation::None);
    }

    #[test]
    fn escalation_narrow_with_none_collapses() {
        let a = Escalation::Interactive { via_conn: 7 };
        assert_eq!(a.narrow(Escalation::None), Escalation::None);
        assert_eq!(Escalation::None.narrow(a), Escalation::None);
    }

    // ---------- Scope ----------

    #[test]
    fn scope_allow_all_admits_everything_except_escalation() {
        let s = Scope::allow_all();
        assert!(s.backends.admits(&"anything".to_string()));
        assert!(s.host_envs.admits(&"anything".to_string()));
        assert_eq!(s.tools.disposition(&"bash".to_string()), Disposition::Allow);
        assert_eq!(s.pod_modify, PodModifyCap::ModifyAllow);
        assert_eq!(s.dispatch, DispatchCap::WithinScope);
        assert_eq!(s.behaviors, BehaviorOpsCap::AuthorAny);
        assert_eq!(s.escalation, Escalation::None);
    }

    #[test]
    fn scope_deny_all_admits_nothing() {
        let s = Scope::deny_all();
        assert!(!s.backends.admits(&"anything".to_string()));
        assert_eq!(s.tools.disposition(&"bash".to_string()), Disposition::Deny);
        assert_eq!(s.pod_modify, PodModifyCap::None);
        assert_eq!(s.dispatch, DispatchCap::None);
        assert_eq!(s.behaviors, BehaviorOpsCap::None);
    }

    #[test]
    fn scope_narrow_never_widens() {
        // parent is pod-wide-except-no-bash; child requests bash.
        // Narrowing cannot grant bash.
        let parent = {
            let mut s = Scope::allow_all();
            s.tools = AllowMap {
                default: Disposition::Allow,
                overrides: [("bash".to_string(), Disposition::Deny)]
                    .into_iter()
                    .collect(),
            };
            s
        };
        let child_request = Scope::allow_all();
        let child = parent.narrow(&child_request);
        assert_eq!(
            child.tools.disposition(&"bash".to_string()),
            Disposition::Deny
        );
    }

    #[test]
    fn scope_narrow_is_monotonic() {
        let parent = Scope::allow_all();
        let child_request = {
            let mut s = Scope::allow_all();
            s.pod_modify = PodModifyCap::Memories;
            s.dispatch = DispatchCap::None;
            s
        };
        let child = parent.narrow(&child_request);
        assert_eq!(child.pod_modify, PodModifyCap::Memories);
        assert_eq!(child.dispatch, DispatchCap::None);
    }

    #[test]
    fn scope_narrow_is_associative() {
        let a = Scope::allow_all();
        let b = {
            let mut s = Scope::allow_all();
            s.pod_modify = PodModifyCap::Content;
            s.backends = SetOrAll::only(["anthropic".to_string(), "openai".to_string()]);
            s
        };
        let c = {
            let mut s = Scope::allow_all();
            s.pod_modify = PodModifyCap::Memories;
            s.backends = SetOrAll::only(["anthropic".to_string()]);
            s
        };
        let left = a.narrow(&b).narrow(&c);
        let right = a.narrow(&b.narrow(&c));
        assert_eq!(left, right);
    }

    #[test]
    fn scope_narrow_escalation_inherits_with_matching_conn() {
        let parent = {
            let mut s = Scope::allow_all();
            s.escalation = Escalation::Interactive { via_conn: 42 };
            s
        };
        let child_request = {
            let mut s = Scope::allow_all();
            s.escalation = Escalation::Interactive { via_conn: 42 };
            s
        };
        let child = parent.narrow(&child_request);
        assert_eq!(child.escalation, Escalation::Interactive { via_conn: 42 });
    }

    #[test]
    fn scope_narrow_escalation_drops_on_mismatch() {
        let parent = {
            let mut s = Scope::allow_all();
            s.escalation = Escalation::Interactive { via_conn: 42 };
            s
        };
        let child_request = {
            let mut s = Scope::allow_all();
            s.escalation = Escalation::Interactive { via_conn: 99 };
            s
        };
        let child = parent.narrow(&child_request);
        assert_eq!(child.escalation, Escalation::None);
    }

    #[test]
    fn scope_roundtrips_through_serde_json() {
        let s = {
            let mut s = Scope::allow_all();
            s.pod_modify = PodModifyCap::Content;
            s.backends = SetOrAll::only(["anthropic".to_string()]);
            s.escalation = Escalation::Interactive { via_conn: 7 };
            s
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Scope = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    // ---------- EscalationRequest ----------

    fn scope_with_caps(
        pod_modify: PodModifyCap,
        dispatch: DispatchCap,
        behaviors: BehaviorOpsCap,
    ) -> Scope {
        let mut s = Scope::allow_all();
        s.pod_modify = pod_modify;
        s.dispatch = dispatch;
        s.behaviors = behaviors;
        s
    }

    #[test]
    fn check_escalation_add_tool_rejects_already_admitted() {
        let mut s = Scope::allow_all();
        s.tools = AllowMap::allow_all();
        let ceiling = Scope::allow_all();
        let err = s
            .check_escalation(
                &EscalationRequest::AddTool {
                    name: "bash".into(),
                },
                &ceiling,
            )
            .unwrap_err();
        assert!(err.contains("already admitted"), "got: {err}");
    }

    #[test]
    fn check_escalation_add_tool_rejects_above_ceiling() {
        let s = Scope::deny_all();
        // Scope default-denies tools.
        let mut ceiling = Scope::deny_all();
        ceiling.tools = {
            let mut m = AllowMap::deny_all();
            m.set_allow("bash".to_string());
            m
        };
        // Pod ceiling allows bash but a different tool isn't in the ceiling.
        let err = s
            .check_escalation(
                &EscalationRequest::AddTool {
                    name: "dangerous".into(),
                },
                &ceiling,
            )
            .unwrap_err();
        assert!(err.contains("pod ceiling"), "got: {err}");
        // Apply the grant to `s` using a valid request — should succeed.
        s.check_escalation(
            &EscalationRequest::AddTool {
                name: "bash".into(),
            },
            &ceiling,
        )
        .unwrap();
    }

    #[test]
    fn check_escalation_raise_pod_modify_admits_upward_step() {
        let s = scope_with_caps(
            PodModifyCap::Memories,
            DispatchCap::None,
            BehaviorOpsCap::None,
        );
        let ceiling = scope_with_caps(
            PodModifyCap::ModifyAllow,
            DispatchCap::WithinScope,
            BehaviorOpsCap::AuthorAny,
        );
        s.check_escalation(
            &EscalationRequest::RaisePodModify {
                target: PodModifyCap::Content,
            },
            &ceiling,
        )
        .unwrap();
    }

    #[test]
    fn check_escalation_raise_pod_modify_rejects_sideways_or_down() {
        let s = scope_with_caps(
            PodModifyCap::Content,
            DispatchCap::None,
            BehaviorOpsCap::None,
        );
        let ceiling = scope_with_caps(
            PodModifyCap::ModifyAllow,
            DispatchCap::WithinScope,
            BehaviorOpsCap::AuthorAny,
        );
        // Same level — not a widening.
        let err = s
            .check_escalation(
                &EscalationRequest::RaisePodModify {
                    target: PodModifyCap::Content,
                },
                &ceiling,
            )
            .unwrap_err();
        assert!(err.contains("not a widening"), "got: {err}");
        // Step down — also not a widening.
        assert!(
            s.check_escalation(
                &EscalationRequest::RaisePodModify {
                    target: PodModifyCap::Memories,
                },
                &ceiling,
            )
            .is_err()
        );
    }

    #[test]
    fn check_escalation_raise_pod_modify_rejects_above_ceiling() {
        let s = scope_with_caps(
            PodModifyCap::Memories,
            DispatchCap::None,
            BehaviorOpsCap::None,
        );
        let ceiling = scope_with_caps(
            PodModifyCap::Content,
            DispatchCap::WithinScope,
            BehaviorOpsCap::AuthorAny,
        );
        let err = s
            .check_escalation(
                &EscalationRequest::RaisePodModify {
                    target: PodModifyCap::ModifyAllow,
                },
                &ceiling,
            )
            .unwrap_err();
        assert!(err.contains("pod ceiling"), "got: {err}");
    }

    #[test]
    fn apply_escalation_widens_caps() {
        let mut s = scope_with_caps(
            PodModifyCap::Memories,
            DispatchCap::None,
            BehaviorOpsCap::Read,
        );
        s.apply_escalation(&EscalationRequest::RaisePodModify {
            target: PodModifyCap::Content,
        });
        s.apply_escalation(&EscalationRequest::RaiseDispatch {
            target: DispatchCap::WithinScope,
        });
        s.apply_escalation(&EscalationRequest::RaiseBehaviors {
            target: BehaviorOpsCap::AuthorNarrower,
        });
        assert_eq!(s.pod_modify, PodModifyCap::Content);
        assert_eq!(s.dispatch, DispatchCap::WithinScope);
        assert_eq!(s.behaviors, BehaviorOpsCap::AuthorNarrower);
    }

    #[test]
    fn apply_escalation_add_tool_admits_tool() {
        let mut s = Scope::deny_all();
        s.apply_escalation(&EscalationRequest::AddTool {
            name: "bash".into(),
        });
        assert!(s.tools.disposition(&"bash".to_string()).admits());
    }

    #[test]
    fn escalation_request_roundtrips_through_json() {
        let reqs = [
            EscalationRequest::AddTool {
                name: "bash".into(),
            },
            EscalationRequest::RaisePodModify {
                target: PodModifyCap::Content,
            },
            EscalationRequest::RaiseBehaviors {
                target: BehaviorOpsCap::AuthorAny,
            },
            EscalationRequest::RaiseDispatch {
                target: DispatchCap::WithinScope,
            },
        ];
        for r in reqs {
            let json = serde_json::to_string(&r).unwrap();
            let back: EscalationRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(r, back);
        }
    }

    #[test]
    fn scope_default_denies_bindings_but_allows_tools() {
        // Default-derived Scope deserializes a missing field to its
        // field-level default. `backends: SetOrAll::default()` is deny
        // (safety); `tools: AllowMap::allow_all` mirrors today's pod
        // default.
        let s = Scope::default();
        assert!(!s.backends.admits(&"anything".to_string()));
        assert_eq!(s.tools.disposition(&"bash".to_string()), Disposition::Allow);
        assert_eq!(s.pod_modify, PodModifyCap::None);
    }
}
