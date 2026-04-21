//! Behavior wire types — captured prompt + config + trigger bundles for
//! autonomous pod work.  See `docs/design_behaviors.md`.
//!
//! Owned here (not in `whisper-agent::pod::behaviors`) because the webui
//! needs the same shape for rendering; TOML parsing, validation, and
//! on-disk I/O stay server-side.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::permission::{AllowMap, BehaviorOpsCap, DispatchCap, PodModifyCap};

/// Parsed `behavior.toml`. Round-trips through TOML; every field is
/// covered by a serde default so hand-written files can omit
/// everything but `name`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BehaviorConfig {
    /// Display name. Freely editable (unlike the behavior id, which is
    /// the immutable directory name).
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub trigger: TriggerSpec,
    #[serde(default)]
    pub thread: BehaviorThreadOverride,
    #[serde(default)]
    pub on_completion: RetentionPolicy,
    /// Per-behavior runtime scope. Composed with the pod's `[allow]`
    /// ceiling at fire time: `child_scope = pod.allow.narrow(scope)`.
    /// Every field is an `Option` so a behavior can declare "inherit
    /// the pod ceiling for this resource" without writing it out — an
    /// entirely-absent `[scope]` block means the behavior runs at the
    /// pod's full ceiling.
    #[serde(default)]
    pub scope: BehaviorScope,
}

/// Behavior-declared runtime scope. Shape mirrors the pod's `[allow]`
/// block: named resource lists, a per-tool allow map, and the three
/// typed caps. On disk every field is optional (`None` = inherit
/// pod ceiling for that field); at fire time the scheduler converts
/// this into a full [`crate::permission::Scope`] via
/// `Scope::narrow(pod_allow, behavior.resolved_scope_narrower())`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct BehaviorScope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backends: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_envs: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_hosts: Option<Vec<String>>,
    /// Tool-level allow map. `None` = inherit the pod's `allow.tools`
    /// (every admitted tool). `Some(map)` narrows per-tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<AllowMap<String>>,
    #[serde(default)]
    pub caps: BehaviorScopeCaps,
}

/// Per-cap narrowing for a behavior. Each field is `None` by default
/// (inherit the pod ceiling); when set, narrows the pod-allow cap to
/// the declared value. Since scopes narrow monotonically, a value
/// above the pod ceiling collapses to the ceiling — i.e., behaviors
/// can only reduce, never widen.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BehaviorScopeCaps {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pod_modify: Option<PodModifyCap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch: Option<DispatchCap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behaviors: Option<BehaviorOpsCap>,
}

impl BehaviorScope {
    /// Render this behavior scope as a full [`crate::permission::Scope`]
    /// suitable for use as the right-hand side of
    /// `pod_allow_scope.narrow(behavior.as_scope_narrower())`.
    ///
    /// Every `None` field becomes the most-permissive value (`All`,
    /// `AllowMap::allow_all`, highest cap) so it's an identity under
    /// narrow — the pod ceiling controls. Every `Some` field becomes
    /// the declared value; narrowing then produces `min(ceiling,
    /// declared)` per the usual rules.
    ///
    /// `escalation` is always `None` — behaviors run autonomously and
    /// cannot escalate. Narrowing with `Escalation::None` collapses
    /// any parent `Interactive` to `None`, which is exactly the
    /// desired behavior.
    pub fn as_scope_narrower(&self) -> crate::permission::Scope {
        use crate::permission::{Escalation, Scope, SetOrAll};

        fn list_to_setorall(list: &Option<Vec<String>>) -> SetOrAll<String> {
            match list {
                None => SetOrAll::all(),
                Some(names) => SetOrAll::only(names.iter().cloned()),
            }
        }

        Scope {
            backends: list_to_setorall(&self.backends),
            host_envs: list_to_setorall(&self.host_envs),
            mcp_hosts: list_to_setorall(&self.mcp_hosts),
            tools: self.tools.clone().unwrap_or_else(AllowMap::allow_all),
            pod_modify: self.caps.pod_modify.unwrap_or(PodModifyCap::ModifyAllow),
            dispatch: self.caps.dispatch.unwrap_or(DispatchCap::WithinScope),
            behaviors: self.caps.behaviors.unwrap_or(BehaviorOpsCap::AuthorAny),
            escalation: Escalation::None,
        }
    }
}

/// Trigger discriminator. Internally tagged on `kind` so TOML looks like
///
/// ```toml
/// [trigger]
/// kind = "cron"
/// schedule = "0 9 * * *"
/// ```
///
/// matching the `#[serde(tag = "...")]` pattern used elsewhere in the crate
/// (see `HostEnvSpec`). v1 variants are `manual`, `cron`, and `webhook`;
/// richer event-source triggers are deferred.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TriggerSpec {
    /// Fires only on `ClientToServer::RunBehavior`. Default for new
    /// behaviors — lets the user iterate before committing to a schedule.
    #[default]
    Manual,
    /// Fires on a five-field cron schedule.
    Cron {
        schedule: String,
        #[serde(default = "default_timezone")]
        timezone: String,
        #[serde(default)]
        overlap: Overlap,
        #[serde(default)]
        catch_up: CatchUp,
    },
    /// Fires on HTTP POST to `/triggers/<pod>/<behavior>`; the request body
    /// becomes the trigger payload. v1 has no auth / no path customization;
    /// the path is derived from pod_id + behavior_id.
    Webhook {
        /// Overlap behavior when a POST arrives while the previous run
        /// is still in flight. Matches Cron's semantics: Skip drops,
        /// QueueOne parks at most one (overwriting any existing queued
        /// payload), Allow spawns concurrent runs.
        #[serde(default)]
        overlap: Overlap,
    },
}

fn default_timezone() -> String {
    "UTC".to_string()
}

/// What to do when a cron fire arrives while a previous run is still in
/// flight. Applies only to `Cron` triggers for now; manual and webhook
/// fires are explicit user actions.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Overlap {
    /// Drop the fire if the previous run is still in flight.
    #[default]
    Skip,
    /// Keep at most one pending fire in reserve (in `BehaviorState`);
    /// drop any beyond that.
    QueueOne,
    /// Fire anyway, spawning concurrent threads.
    Allow,
}

/// What to do about cron fires missed while the server was down. Applied
/// once per behavior on scheduler startup.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CatchUp {
    /// Skip missed fires silently.
    None,
    /// Fire at most one catch-up per behavior, log the count. Default.
    #[default]
    One,
    /// Fire every missed run. Rarely what's wanted — long downtime plus a
    /// per-minute cron would hammer the scheduler with stale firings.
    All,
}

/// Per-behavior thread-config override, layered on top of the pod's
/// `thread_defaults`. Shape mirrors `ThreadConfigOverride` plus a
/// bindings subset so a behavior can narrow the thread's capability
/// surface below the pod default.
///
/// Note: `prompt.md` is the behavior's **user-message body** (the
/// first turn the thread sees), while `system_prompt` here is the
/// agent-personality preamble the model runs under. A behavior that
/// wants to run its prompt.md under a different system prompt than
/// the pod default sets `system_prompt` here — useful for e.g. a
/// summarization behavior that should run as a summarizer, not the
/// pod's default agent persona.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct BehaviorThreadOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    /// Per-thread system prompt for behavior-spawned threads. `None`
    /// inherits the pod default; see [`SystemPromptChoice`] for the
    /// File / Text variants.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<crate::SystemPromptChoice>,
    #[serde(default)]
    pub bindings: BehaviorBindingsOverride,
}

/// Bindings subset for a behavior's spawned threads. Each `Some` replaces
/// the pod-default binding; each `None` inherits. All values must resolve
/// against the pod's `[allow]` table at trigger time — the cap is still
/// authoritative.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct BehaviorBindingsOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Override the pod-default host-env list. `None` inherits;
    /// `Some(vec)` replaces exactly — empty vec drops to bare, non-
    /// empty binds to each named entry. Accepts a bare string (legacy
    /// singular shape) or a list on disk.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "opt_string_or_vec"
    )]
    pub host_env: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_hosts: Option<Vec<String>>,
}

/// Accept both `None`, a bare string (legacy), and a `Vec<String>` (new)
/// for an optional plural-of-strings field. Mirrors
/// [`crate::pod::deserialize_string_or_vec`] but for the `Option`-wrapped
/// case — TOML serializers produce the same on-the-wire shape for both.
mod opt_string_or_vec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Shape {
        Many(Vec<String>),
        One(String),
    }

    pub fn serialize<S>(value: &Option<Vec<String>>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value.serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Option<Vec<String>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Option::<Shape>::deserialize(d)?.map(|shape| match shape {
            Shape::Many(v) => v,
            Shape::One(s) if s.is_empty() => Vec::new(),
            Shape::One(s) => vec![s],
        }))
    }
}

/// What to do with a behavior-spawned thread once it terminates.
/// Interactive threads default to `Keep`; behaviors typically want one of
/// the timed variants to avoid accumulating thousands of JSONs from a
/// high-cadence cron.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(tag = "retention", rename_all = "snake_case")]
pub enum RetentionPolicy {
    /// Never sweep. Default for manual-only behaviors.
    #[default]
    Keep,
    /// Move to `<pod>/.archived/threads/` after N days past last_active.
    ArchiveAfterDays { days: u32 },
    /// Delete after N days past last_active.
    DeleteAfterDays { days: u32 },
}

/// Persistent trigger-firing state. Written to `<pod>/behaviors/<id>/state.json`.
/// Maintained by the scheduler as triggers fire; load-only in phase 1
/// (nothing fires yet, nothing writes it).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BehaviorState {
    /// Automatic-trigger gate: when false, cron ticks skip this
    /// behavior, webhook POSTs return 503, and startup catch-up
    /// ignores it. Manual `RunBehavior` (the UI Run button) always
    /// works regardless — it's an explicit user action. Default `true`
    /// so legacy state.json files without the field behave as always.
    #[serde(default = "enabled_default")]
    pub enabled: bool,
    #[serde(default)]
    pub run_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_fired_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_outcome: Option<BehaviorOutcome>,
    /// `QueueOne` overlap policy parks a payload here when a fire arrives
    /// while the previous run is in flight. The next on-completion hook
    /// consumes it. Dropped when the behavior is paused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queued_payload: Option<Value>,
}

fn enabled_default() -> bool {
    true
}

impl Default for BehaviorState {
    fn default() -> Self {
        Self {
            enabled: true,
            run_count: 0,
            last_fired_at: None,
            last_thread_id: None,
            last_outcome: None,
            queued_payload: None,
        }
    }
}

/// Terminal outcome of the last behavior-spawned thread. Feeds into UI
/// status badges ("last run: failed 20 min ago") and future retry logic.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BehaviorOutcome {
    Completed,
    Failed { message: String },
    Cancelled,
}

/// Provenance stamp carried by every thread a behavior spawned. Lets the
/// UI group "all runs of daily_ci_check" without a side table and keeps
/// the trigger context (what fired this run, when, with what payload)
/// available for debugging. Absent on interactive threads.
///
/// Persisted as part of the thread's JSON. Also echoed onto
/// `ThreadSummary` / `ThreadSnapshot` so every client observing the
/// thread sees the same origin without a second round trip.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BehaviorOrigin {
    pub behavior_id: String,
    /// RFC-3339 timestamp of when the trigger fired. Plain string so the
    /// protocol crate stays chrono-free.
    pub fired_at: String,
    /// Trigger-defined payload. `Null` for manual / cron fires that
    /// didn't supply data; arbitrary JSON for webhook fires (request
    /// body) and future event-source variants.
    #[serde(default)]
    pub trigger_payload: Value,
}

/// Lightweight list entry used in `PodSnapshot.behaviors` and in
/// `BehaviorList` responses. Carries enough for the webui to render a row
/// without asking for the full config.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BehaviorSummary {
    pub behavior_id: String,
    pub pod_id: String,
    /// Display name — `config.name` when the behavior loaded cleanly, else
    /// the behavior_id as a fallback.
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Discriminator of `TriggerSpec`: `"manual" | "cron" | "webhook"`.
    /// `None` when the behavior failed to load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_kind: Option<String>,
    /// Mirrors `BehaviorState.enabled`. Carried on the summary so list
    /// views can badge paused rows without a separate round-trip.
    #[serde(default = "enabled_default")]
    pub enabled: bool,
    #[serde(default)]
    pub run_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_fired_at: Option<String>,
    /// `None` when the behavior loaded cleanly; `Some` carries the parse
    /// / validation error message. UI surfaces this with an error badge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_error: Option<String>,
}

/// Full snapshot delivered in response to `GetBehavior`. Carries both the
/// parsed config (for form-based editing) and the raw TOML text (for a raw
/// editor), mirroring `PodSnapshot`. `config` is `None` when the on-disk
/// TOML failed to parse.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BehaviorSnapshot {
    pub behavior_id: String,
    pub pod_id: String,
    /// `None` when parsing / validation failed — `load_error` carries the
    /// reason, and `toml_text` still holds whatever is on disk so the
    /// user can fix it in a raw editor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<BehaviorConfig>,
    pub toml_text: String,
    /// Contents of the sibling `prompt.md`. Empty string when missing.
    pub prompt: String,
    pub state: BehaviorState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigger_manual_is_default() {
        let cfg = BehaviorConfig {
            name: "x".into(),
            description: None,
            trigger: TriggerSpec::default(),
            thread: BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
            scope: Default::default(),
        };
        assert!(matches!(cfg.trigger, TriggerSpec::Manual));
        assert!(matches!(cfg.on_completion, RetentionPolicy::Keep));
    }

    #[test]
    fn retention_serializes_with_kind_tag() {
        let p = RetentionPolicy::ArchiveAfterDays { days: 30 };
        let json = serde_json::to_string(&p).unwrap();
        // tag = "retention" per the serde attribute; keeps TOML readable.
        assert!(json.contains("\"retention\":\"archive_after_days\""));
        assert!(json.contains("\"days\":30"));
    }

    #[test]
    fn default_scope_is_identity_under_narrow() {
        // A behavior with no `[scope]` block should narrow to exactly
        // the pod ceiling — no fields restricted.
        use crate::permission::Scope;
        let ceiling = Scope::allow_all();
        let behavior_scope = BehaviorScope::default();
        let result = ceiling.narrow(&behavior_scope.as_scope_narrower());
        assert_eq!(result, ceiling);
    }

    #[test]
    fn as_scope_narrower_restricts_declared_fields() {
        use crate::permission::{BehaviorOpsCap, DispatchCap, PodModifyCap, Scope, SetOrAll};
        let behavior_scope = BehaviorScope {
            backends: Some(vec!["anthropic".into()]),
            host_envs: Some(vec!["narrow".into()]),
            mcp_hosts: None,
            tools: None,
            caps: BehaviorScopeCaps {
                pod_modify: Some(PodModifyCap::None),
                dispatch: Some(DispatchCap::None),
                behaviors: Some(BehaviorOpsCap::None),
            },
        };
        let narrower = behavior_scope.as_scope_narrower();
        assert_eq!(narrower.backends, SetOrAll::only(["anthropic".to_string()]));
        assert_eq!(narrower.host_envs, SetOrAll::only(["narrow".to_string()]));
        assert!(matches!(narrower.mcp_hosts, SetOrAll::All));
        assert_eq!(narrower.pod_modify, PodModifyCap::None);
        assert_eq!(narrower.dispatch, DispatchCap::None);
        assert_eq!(narrower.behaviors, BehaviorOpsCap::None);
        // Applying against a permissive ceiling: the narrower values win.
        let ceiling = Scope::allow_all();
        let composed = ceiling.narrow(&narrower);
        assert_eq!(composed.pod_modify, PodModifyCap::None);
        assert_eq!(composed.dispatch, DispatchCap::None);
    }

    #[test]
    fn behavior_scope_round_trips_through_json() {
        // TOML round-trip coverage for BehaviorConfig lives in
        // `src/pod/behaviors.rs` (the crate with the toml dep); this
        // serde test uses JSON so we stay dep-free in the protocol
        // crate. Same serde contract applies to both.
        let cfg = BehaviorConfig {
            name: "narrowed".into(),
            description: None,
            trigger: TriggerSpec::default(),
            thread: BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
            scope: BehaviorScope {
                backends: Some(vec!["anthropic".into()]),
                host_envs: None,
                mcp_hosts: Some(vec!["fetch".into()]),
                tools: None,
                caps: BehaviorScopeCaps {
                    pod_modify: Some(crate::permission::PodModifyCap::None),
                    dispatch: Some(crate::permission::DispatchCap::None),
                    behaviors: None,
                },
            },
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: BehaviorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);
        assert!(
            json.contains("\"backends\":[\"anthropic\"]"),
            "backends should serialize as bare array; got: {json}"
        );
        // `None` fields are skipped; behaviors with an inherit-everything
        // scope don't clutter the wire.
        assert!(!json.contains("\"host_envs\""));
    }
}
