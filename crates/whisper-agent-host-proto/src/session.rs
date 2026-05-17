//! Session lifecycle types — identifiers, end reasons, the per-session
//! [`ThreadContext`] and its mutation delta.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use whisper_agent_protocol::ConfigurableValue;

/// Session identifier. Scheduler-assigned (typically a UUID rendered
/// as a string) — daemons never invent one. The scheduler is the
/// source of truth for session identity, which makes reconnect
/// semantics tractable: a returning daemon cannot accidentally collide
/// with a session the scheduler still considers live.
///
/// Wrapped over `String` rather than a typed UUID to avoid pulling
/// `uuid` into the protocol crate's dependency surface; consumers that
/// want UUID semantics convert at the boundary.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(pub String);

impl SessionId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Phase a [`crate::Frame::SessionFailed`] was reported in. Helps the
/// scheduler surface the right diagnostic — landlock failures vs
/// worker spawn failures vs handshake timeouts have different
/// operator-facing remedies.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProvisionPhase {
    /// `pre_exec` setup failed (cgroup attach, ulimit application, etc.).
    PreExec,
    /// Landlock ruleset construction or `restrict_self` failed.
    Landlock,
    /// Worker process spawn failed (`execve` returned an error).
    WorkerSpawn,
    /// Worker spawned but didn't complete its handshake within the
    /// daemon's startup timeout.
    WorkerHandshake,
}

/// Why a session ended. Distinguishes scheduler-initiated, worker-
/// initiated, and daemon-shutdown teardowns so the UI can phrase the
/// outcome correctly.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionEndReason {
    /// Scheduler sent [`crate::Frame::CloseSession`].
    RequestedByScheduler,
    /// Worker process exited without being asked to. `code` is the
    /// OS exit code (`None` if killed by signal — the daemon could
    /// surface signal info via [`crate::Frame::HookEvent`] if it
    /// matters).
    WorkerExited {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<i32>,
    },
    /// Daemon is shutting down.
    DaemonShutdown,
}

/// Per-session runtime configuration. Carried in
/// [`crate::Frame::OpenSession`]; updatable mid-flight via
/// [`crate::Frame::UpdateSession`].
///
/// What's deliberately **not** here:
/// - **Landlock paths / network policy.** Those are
///   [`crate::HostEnvSpec`]; changing them requires re-landlocking
///   the worker, which means tearing down and reprovisioning. That's
///   what `OpenSession` already is. `UpdateSession` is for things
///   changeable without re-landlocking.
/// - **Tool *allowlist*.** The scheduler enforces this before the
///   call ever leaves; the daemon never sees disallowed calls.
///   [`Self::tool_denylist`] exists specifically as a daemon-side
///   defense-in-depth backstop.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct ThreadContext {
    /// Override the daemon's default workspace root for this session.
    /// `None` ⇒ use the spec's writable allowed_paths root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_root: Option<PathBuf>,

    /// Environment variables for processes the worker spawns (bash
    /// and the like).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,

    /// Provider-specific session options advertised by the daemon via
    /// `DaemonCapabilities.session_configurables`. The daemon interprets
    /// only keys it advertised; the scheduler stores and forwards the map
    /// generically so new provider knobs do not need new protocol fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub options: BTreeMap<String, ConfigurableValue>,

    /// Drop to this user before exec in bash. `None` ⇒ daemon's uid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runas: Option<String>,

    /// Tools the daemon refuses to invoke for this session. Hard
    /// refusal — the daemon returns an error result without dispatching
    /// to the tool.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub tool_denylist: BTreeSet<String>,

    /// Bash hard timeout. `None` ⇒ daemon default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bash_timeout_secs: Option<u32>,

    /// Cap on streamed [`crate::Frame::ToolChunk`] bytes per call
    /// (head + tail truncation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_byte_cap: Option<usize>,
}

/// Deserialize an `Option<Option<T>>` so the three states "absent",
/// "present and null", and "present and `x`" survive round-trip.
///
/// serde's default deserializer collapses null → outer None, which
/// loses the "explicit clear" signal we want on a delta. Combined
/// with `#[serde(default, deserialize_with = ...)]`, this preserves:
///   - field absent on the wire ⇒ `None`              (no change)
///   - field present with null  ⇒ `Some(None)`        (clear)
///   - field present with `x`   ⇒ `Some(Some(x))`     (set)
fn de_double_option<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

impl ThreadContext {
    /// Compute the [`ThreadContextDelta`] that turns `self` into
    /// `new`. Fields that match are absent from the delta. For
    /// `env`, additions/overrides go into `env_set` and removed
    /// names into `env_unset`.
    ///
    /// Used by the scheduler when its stored context for a session
    /// changes — the delta is what travels in `Frame::UpdateSession`.
    pub fn diff_to(&self, new: &ThreadContext) -> ThreadContextDelta {
        let mut delta = ThreadContextDelta::default();
        if self.workspace_root != new.workspace_root {
            delta.workspace_root = Some(new.workspace_root.clone());
        }
        // Env: per-key set/unset rather than whole-map replace.
        for (k, v) in &new.env {
            if self.env.get(k) != Some(v) {
                delta.env_set.insert(k.clone(), v.clone());
            }
        }
        for k in self.env.keys() {
            if !new.env.contains_key(k) {
                delta.env_unset.insert(k.clone());
            }
        }
        for (k, v) in &new.options {
            if self.options.get(k) != Some(v) {
                delta.option_set.insert(k.clone(), v.clone());
            }
        }
        for k in self.options.keys() {
            if !new.options.contains_key(k) {
                delta.option_unset.insert(k.clone());
            }
        }
        if self.runas != new.runas {
            delta.runas = Some(new.runas.clone());
        }
        if self.tool_denylist != new.tool_denylist {
            delta.tool_denylist = Some(new.tool_denylist.clone());
        }
        if self.bash_timeout_secs != new.bash_timeout_secs {
            delta.bash_timeout_secs = Some(new.bash_timeout_secs);
        }
        if self.output_byte_cap != new.output_byte_cap {
            delta.output_byte_cap = Some(new.output_byte_cap);
        }
        delta
    }

    /// Apply a [`ThreadContextDelta`] in-place. Daemon-side: the
    /// session's stored context is mutated, and subsequent tool
    /// calls run under the new values. Workspace-root changes are
    /// recorded but cannot move the worker — the daemon should
    /// surface a warning if that field is set in a delta.
    pub fn apply_delta(&mut self, delta: &ThreadContextDelta) {
        if let Some(wr) = &delta.workspace_root {
            self.workspace_root = wr.clone();
        }
        for k in &delta.env_unset {
            self.env.remove(k);
        }
        for (k, v) in &delta.env_set {
            self.env.insert(k.clone(), v.clone());
        }
        for k in &delta.option_unset {
            self.options.remove(k);
        }
        for (k, v) in &delta.option_set {
            self.options.insert(k.clone(), v.clone());
        }
        if let Some(r) = &delta.runas {
            self.runas = r.clone();
        }
        if let Some(d) = &delta.tool_denylist {
            self.tool_denylist = d.clone();
        }
        if let Some(t) = &delta.bash_timeout_secs {
            self.bash_timeout_secs = *t;
        }
        if let Some(c) = &delta.output_byte_cap {
            self.output_byte_cap = *c;
        }
    }

    /// True when the delta would mutate `self`. Lets the scheduler
    /// avoid sending no-op `Frame::UpdateSession` frames.
    pub fn delta_is_empty(delta: &ThreadContextDelta) -> bool {
        delta.workspace_root.is_none()
            && delta.env_set.is_empty()
            && delta.env_unset.is_empty()
            && delta.option_set.is_empty()
            && delta.option_unset.is_empty()
            && delta.runas.is_none()
            && delta.tool_denylist.is_none()
            && delta.bash_timeout_secs.is_none()
            && delta.output_byte_cap.is_none()
    }
}

/// Partial update to a [`ThreadContext`].
///
/// Field semantics follow the standard `Option<Option<T>>` pattern:
/// - **Field absent on the wire** ⇒ leave alone.
/// - **Field present, value `Some(x)`** ⇒ set to `x`.
/// - **Field present, value `None`** ⇒ clear (revert to the
///   `ThreadContext` default for that field).
///
/// Environment is a special case: rather than "replace whole map",
/// the wire carries [`Self::env_set`] (entries to add or override)
/// and [`Self::env_unset`] (names to remove). This avoids forcing
/// callers to ship the full env on every tweak.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct ThreadContextDelta {
    #[serde(
        default,
        deserialize_with = "de_double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub workspace_root: Option<Option<PathBuf>>,

    /// Environment additions / overrides. Merged on top of existing
    /// `env`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env_set: BTreeMap<String, String>,
    /// Environment names to remove.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub env_unset: BTreeSet<String>,

    /// Provider option additions / overrides. Merged on top of existing
    /// `options`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub option_set: BTreeMap<String, ConfigurableValue>,
    /// Provider option names to remove.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub option_unset: BTreeSet<String>,

    #[serde(
        default,
        deserialize_with = "de_double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub runas: Option<Option<String>>,

    /// Replace the entire denylist. `None` ⇒ leave alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_denylist: Option<BTreeSet<String>>,

    #[serde(
        default,
        deserialize_with = "de_double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub bash_timeout_secs: Option<Option<u32>>,

    #[serde(
        default,
        deserialize_with = "de_double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_byte_cap: Option<Option<usize>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<T>(value: T)
    where
        T: Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        let mut bytes = Vec::new();
        ciborium::into_writer(&value, &mut bytes).unwrap();
        let back: T = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(value, back);
    }

    #[test]
    fn session_id_round_trips() {
        round_trip(SessionId::new("01HXYZ-test"));
    }

    #[test]
    fn provision_phase_serializes_snake_case() {
        let mut bytes = Vec::new();
        ciborium::into_writer(&ProvisionPhase::WorkerHandshake, &mut bytes).unwrap();
        let v: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(v.as_text(), Some("worker_handshake"));
    }

    #[test]
    fn session_end_reason_round_trips_each_variant() {
        round_trip(SessionEndReason::RequestedByScheduler);
        round_trip(SessionEndReason::WorkerExited { code: Some(0) });
        round_trip(SessionEndReason::WorkerExited { code: None });
        round_trip(SessionEndReason::DaemonShutdown);
    }

    #[test]
    fn empty_thread_context_round_trips() {
        round_trip(ThreadContext::default());
    }

    #[test]
    fn populated_thread_context_round_trips() {
        round_trip(ThreadContext {
            workspace_root: Some(PathBuf::from("/home/me/ws")),
            env: BTreeMap::from([
                ("PATH".into(), "/usr/bin".into()),
                ("HOME".into(), "/home/me".into()),
            ]),
            options: BTreeMap::from([(
                "user_env".into(),
                ConfigurableValue::Enum("runas_basic".into()),
            )]),
            runas: Some("worker".into()),
            tool_denylist: BTreeSet::from(["view_pdf".into(), "view_image".into()]),
            bash_timeout_secs: Some(120),
            output_byte_cap: Some(1 << 20),
        });
    }

    #[test]
    fn empty_thread_context_serializes_as_empty_map() {
        // Defaults must elide cleanly so an empty context isn't a
        // wall of nulls on the wire. Round-trip through CBOR Value
        // and assert the map is empty.
        let mut bytes = Vec::new();
        ciborium::into_writer(&ThreadContext::default(), &mut bytes).unwrap();
        let v: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(v.as_map().expect("map").len(), 0);
    }

    #[test]
    fn thread_context_delta_distinguishes_unset_clear_set() {
        // Field absent ⇒ no change. Field present and Some(None) ⇒
        // explicit clear. Field present and Some(Some(x)) ⇒ set.
        // Verify all three round-trip distinctly.
        let unset = ThreadContextDelta::default();
        assert!(unset.runas.is_none());

        let cleared = ThreadContextDelta {
            runas: Some(None),
            ..ThreadContextDelta::default()
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&cleared, &mut bytes).unwrap();
        let back: ThreadContextDelta = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(back.runas, Some(None));

        let set = ThreadContextDelta {
            runas: Some(Some("worker".into())),
            ..ThreadContextDelta::default()
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&set, &mut bytes).unwrap();
        let back: ThreadContextDelta = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(back.runas, Some(Some("worker".into())));
    }

    #[test]
    fn empty_thread_context_delta_serializes_as_empty_map() {
        let mut bytes = Vec::new();
        ciborium::into_writer(&ThreadContextDelta::default(), &mut bytes).unwrap();
        let v: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(v.as_map().expect("map").len(), 0);
    }

    #[test]
    fn diff_to_returns_empty_delta_for_identical_contexts() {
        let a = ThreadContext {
            runas: Some("worker".into()),
            env: BTreeMap::from([("PATH".into(), "/usr/bin".into())]),
            ..ThreadContext::default()
        };
        let b = a.clone();
        let delta = a.diff_to(&b);
        assert!(ThreadContext::delta_is_empty(&delta));
    }

    #[test]
    fn diff_to_sets_changed_scalar_fields() {
        let a = ThreadContext::default();
        let b = ThreadContext {
            runas: Some("worker".into()),
            bash_timeout_secs: Some(60),
            ..ThreadContext::default()
        };
        let delta = a.diff_to(&b);
        assert_eq!(delta.runas, Some(Some("worker".into())));
        assert_eq!(delta.bash_timeout_secs, Some(Some(60)));
        assert!(delta.workspace_root.is_none());
    }

    #[test]
    fn diff_to_clears_scalar_fields_via_double_some_none() {
        // a has runas set; b has it None → delta carries Some(None) so
        // the daemon can distinguish "clear" from "no change".
        let a = ThreadContext {
            runas: Some("worker".into()),
            ..ThreadContext::default()
        };
        let b = ThreadContext::default();
        let delta = a.diff_to(&b);
        assert_eq!(delta.runas, Some(None));
    }

    #[test]
    fn diff_to_emits_env_set_and_env_unset() {
        let a = ThreadContext {
            env: BTreeMap::from([
                ("PATH".into(), "/usr/bin".into()),
                ("OLD".into(), "x".into()),
            ]),
            ..ThreadContext::default()
        };
        let b = ThreadContext {
            env: BTreeMap::from([
                ("PATH".into(), "/usr/local/bin".into()), // changed
                ("NEW".into(), "y".into()),               // added
                                                          // OLD removed
            ]),
            ..ThreadContext::default()
        };
        let delta = a.diff_to(&b);
        assert_eq!(
            delta.env_set,
            BTreeMap::from([
                ("PATH".into(), "/usr/local/bin".into()),
                ("NEW".into(), "y".into()),
            ])
        );
        assert_eq!(delta.env_unset, BTreeSet::from(["OLD".into()]));
    }

    #[test]
    fn diff_to_replaces_whole_denylist() {
        let a = ThreadContext {
            tool_denylist: BTreeSet::from(["bash".into()]),
            ..ThreadContext::default()
        };
        let b = ThreadContext {
            tool_denylist: BTreeSet::from(["bash".into(), "edit_file".into()]),
            ..ThreadContext::default()
        };
        let delta = a.diff_to(&b);
        assert_eq!(
            delta.tool_denylist,
            Some(BTreeSet::from(["bash".into(), "edit_file".into()]))
        );
    }

    #[test]
    fn apply_delta_round_trips_through_diff() {
        // For any (a, b), apply_delta(a, diff(a → b)) should equal b.
        let a = ThreadContext {
            runas: Some("worker".into()),
            env: BTreeMap::from([("PATH".into(), "/usr/bin".into())]),
            tool_denylist: BTreeSet::from(["bash".into()]),
            bash_timeout_secs: Some(30),
            ..ThreadContext::default()
        };
        let b = ThreadContext {
            runas: None,
            env: BTreeMap::from([
                ("PATH".into(), "/usr/local/bin".into()),
                ("HOME".into(), "/home/me".into()),
            ]),
            tool_denylist: BTreeSet::new(),
            bash_timeout_secs: Some(120),
            workspace_root: Some(PathBuf::from("/work")),
            output_byte_cap: Some(1024),
            ..ThreadContext::default()
        };
        let delta = a.diff_to(&b);
        let mut applied = a.clone();
        applied.apply_delta(&delta);
        assert_eq!(applied, b);
    }

    #[test]
    fn apply_empty_delta_is_no_op() {
        let original = ThreadContext {
            runas: Some("worker".into()),
            env: BTreeMap::from([("PATH".into(), "/usr/bin".into())]),
            ..ThreadContext::default()
        };
        let mut applied = original.clone();
        applied.apply_delta(&ThreadContextDelta::default());
        assert_eq!(applied, original);
    }

    #[test]
    fn apply_delta_env_unset_takes_precedence_over_env_set_on_same_key() {
        // Pathological-but-possible: a delta lists the same key in
        // both env_set and env_unset. apply_delta processes unset
        // first then set, so the final state has the set value —
        // documented here to pin the contract.
        let mut ctx = ThreadContext {
            env: BTreeMap::from([("X".into(), "old".into())]),
            ..ThreadContext::default()
        };
        let mut delta = ThreadContextDelta::default();
        delta.env_unset.insert("X".into());
        delta.env_set.insert("X".into(), "new".into());
        ctx.apply_delta(&delta);
        assert_eq!(ctx.env.get("X").map(String::as_str), Some("new"));
    }
}
