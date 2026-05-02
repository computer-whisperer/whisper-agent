//! Session lifecycle types — identifiers, end reasons, the per-session
//! [`ThreadContext`] and its mutation delta.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
}
