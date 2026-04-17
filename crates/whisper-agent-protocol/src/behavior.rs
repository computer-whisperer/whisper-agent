//! Behavior wire types — captured prompt + config + trigger bundles for
//! autonomous pod work.  See `docs/design_behaviors.md`.
//!
//! Owned here (not in `whisper-agent::pod::behaviors`) because the webui
//! needs the same shape for rendering; TOML parsing, validation, and
//! on-disk I/O stay server-side.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ApprovalPolicy;

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
    Webhook,
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
/// `thread_defaults`. Shape mirrors `ThreadConfigOverride` minus
/// `system_prompt` (the behavior carries its own prompt in `prompt.md`)
/// and gains a bindings subset so a behavior can narrow the thread's
/// capability surface below the pod default.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct BehaviorThreadOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<ApprovalPolicy>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_hosts: Option<Vec<String>>,
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
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct BehaviorState {
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
    /// consumes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queued_payload: Option<Value>,
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
}
