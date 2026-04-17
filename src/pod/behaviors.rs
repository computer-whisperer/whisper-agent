//! Behavior data model — in-memory `Behavior` type, TOML parsing +
//! validation, on-disk loader + writer.  See `docs/design_behaviors.md`.
//!
//! Each pod owns a `behaviors/` subdirectory; each entry there is a
//! self-contained `<behavior_id>/` with:
//!   - `behavior.toml` — the `BehaviorConfig`.
//!   - `prompt.md` — the initial user-message template spawned threads
//!     run against. Substituted at fire time (`{{payload}}` → payload
//!     as pretty JSON); the pod's system_prompt_file remains the
//!     thread's system prompt.
//!   - `state.json` — `BehaviorState` (run_count, last_fired_at, ...).
//!
//! Phases 1-2 cover: types + parser + loader + write helpers. Triggers
//! (cron, webhook) and retention sweeping arrive in later phases.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use thiserror::Error;
use tokio::fs;
use tracing::warn;
use whisper_agent_protocol::{
    BehaviorConfig, BehaviorState, BehaviorSummary, RetentionPolicy, TriggerSpec,
};

use crate::pod::PodId;

/// Subdirectory under a pod that holds per-behavior dirs.
pub const BEHAVIORS_DIR: &str = "behaviors";
/// Filename for the parsed behavior config.
pub const BEHAVIOR_TOML: &str = "behavior.toml";
/// Filename for the behavior's prompt template.
pub const BEHAVIOR_PROMPT: &str = "prompt.md";
/// Filename for the persisted trigger state.
pub const BEHAVIOR_STATE: &str = "state.json";

pub type BehaviorId = String;

/// Pre-parsed cron schedule + timezone, cached on a `Behavior` whose
/// trigger is `TriggerSpec::Cron`. Absent for other trigger kinds and
/// for crons whose schedule/timezone failed to parse (those surface in
/// `Behavior::load_error`).
///
/// Parsing once at load time keeps the per-tick evaluation cheap — 30 s
/// ticks over N behaviors would otherwise re-parse every cron every
/// tick. The `Schedule` type is a parsed NFA; the `Tz` is a table
/// lookup. Neither is cheap enough to re-do on the hot path.
#[derive(Debug, Clone)]
pub struct CachedCron {
    pub schedule: cron::Schedule,
    pub timezone: chrono_tz::Tz,
}

/// In-memory representation of a behavior. One per on-disk
/// `<pod>/behaviors/<id>/` directory. `config` is `None` when
/// `behavior.toml` failed to parse — `load_error` carries the reason and
/// the UI can still render the entry so the user can fix it.
#[derive(Debug, Clone)]
pub struct Behavior {
    pub id: BehaviorId,
    pub pod_id: PodId,
    pub dir: PathBuf,
    pub config: Option<BehaviorConfig>,
    /// Verbatim contents of the on-disk `behavior.toml`. Kept so the
    /// wire can echo raw text for a raw-editor view.
    pub raw_toml: String,
    /// Contents of the sibling `prompt.md`. Empty when absent.
    pub prompt: String,
    pub state: BehaviorState,
    /// Parsed cron Schedule + timezone. `Some` only when `config.trigger`
    /// is `Cron` AND both parsed cleanly. Rebuilt by `refresh_cron` on
    /// create / update / reload.
    pub cron: Option<CachedCron>,
    pub load_error: Option<String>,
}

impl Behavior {
    pub fn summary(&self) -> BehaviorSummary {
        let (name, description, trigger_kind) = match &self.config {
            Some(cfg) => (
                cfg.name.clone(),
                cfg.description.clone(),
                Some(trigger_kind(&cfg.trigger).to_string()),
            ),
            None => (self.id.clone(), None, None),
        };
        BehaviorSummary {
            behavior_id: self.id.clone(),
            pod_id: self.pod_id.clone(),
            name,
            description,
            trigger_kind,
            run_count: self.state.run_count,
            last_fired_at: self.state.last_fired_at.clone(),
            load_error: self.load_error.clone(),
        }
    }

    pub fn snapshot(&self) -> whisper_agent_protocol::BehaviorSnapshot {
        whisper_agent_protocol::BehaviorSnapshot {
            behavior_id: self.id.clone(),
            pod_id: self.pod_id.clone(),
            config: self.config.clone(),
            toml_text: self.raw_toml.clone(),
            prompt: self.prompt.clone(),
            state: self.state.clone(),
            load_error: self.load_error.clone(),
        }
    }
}

/// Stringly-typed discriminator of `TriggerSpec`. Tracked here (not in the
/// protocol crate) because it's a server-local helper, not part of the
/// wire format.
pub fn trigger_kind(trigger: &TriggerSpec) -> &'static str {
    match trigger {
        TriggerSpec::Manual => "manual",
        TriggerSpec::Cron { .. } => "cron",
        TriggerSpec::Webhook => "webhook",
    }
}

#[derive(Debug, Error)]
pub enum BehaviorConfigError {
    #[error("toml parse: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("toml encode: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("trigger.schedule is empty for a cron trigger")]
    EmptyCronSchedule,
    #[error("trigger.schedule `{0}` is not a valid 5-field cron expression: {1}")]
    BadCronSchedule(String, String),
    #[error("trigger.timezone `{0}` is not a recognized IANA zone")]
    BadTimezone(String),
    #[error("on_completion.days must be > 0 for {policy}")]
    ZeroRetentionDays { policy: &'static str },
}

/// Parse `behavior.toml` text, then run structural validation. Does not
/// cross-reference the pod's `[allow]` table — that lives in the scheduler,
/// which has the pod config handy.
pub fn parse_toml(text: &str) -> Result<BehaviorConfig, BehaviorConfigError> {
    let cfg: BehaviorConfig = toml::from_str(text)?;
    validate(&cfg)?;
    Ok(cfg)
}

pub fn to_toml(cfg: &BehaviorConfig) -> Result<String, BehaviorConfigError> {
    Ok(toml::to_string_pretty(cfg)?)
}

/// Structural validation: trigger-variant-specific rules + retention sanity.
/// Pod-allow cross-checks (e.g. `bindings.host_env` must be in the pod's
/// `[allow.host_env]`) happen at trigger / spawn time, not here — the
/// behavior config is loaded before its pod's cap is necessarily available.
///
/// Cron schedules are parsed here (via [`parse_cron_schedule`]) so a
/// typo surfaces at write-time rather than at tick-time; the parsed
/// `Schedule` is thrown away — callers re-parse into `CachedCron`
/// after `validate` succeeds.
pub fn validate(cfg: &BehaviorConfig) -> Result<(), BehaviorConfigError> {
    match &cfg.trigger {
        TriggerSpec::Manual | TriggerSpec::Webhook => {}
        TriggerSpec::Cron {
            schedule, timezone, ..
        } => {
            if schedule.trim().is_empty() {
                return Err(BehaviorConfigError::EmptyCronSchedule);
            }
            parse_cron_schedule(schedule)?;
            parse_timezone(timezone)?;
        }
    }
    match &cfg.on_completion {
        RetentionPolicy::Keep => {}
        RetentionPolicy::ArchiveAfterDays { days } => {
            if *days == 0 {
                return Err(BehaviorConfigError::ZeroRetentionDays {
                    policy: "archive_after_days",
                });
            }
        }
        RetentionPolicy::DeleteAfterDays { days } => {
            if *days == 0 {
                return Err(BehaviorConfigError::ZeroRetentionDays {
                    policy: "delete_after_days",
                });
            }
        }
    }
    Ok(())
}

/// Parse a 5-field UNIX crontab expression (`"min hour dom mon dow"`)
/// into a `cron::Schedule`. The underlying crate expects 6+ fields with
/// seconds; we prepend `"0"` so fires land on the zeroth second of each
/// minute. Empty / malformed inputs surface as `BadCronSchedule`.
pub fn parse_cron_schedule(schedule: &str) -> Result<cron::Schedule, BehaviorConfigError> {
    let trimmed = schedule.trim();
    if trimmed.is_empty() {
        return Err(BehaviorConfigError::EmptyCronSchedule);
    }
    let six_field = format!("0 {trimmed}");
    cron::Schedule::from_str(&six_field)
        .map_err(|e| BehaviorConfigError::BadCronSchedule(trimmed.to_string(), e.to_string()))
}

/// Parse an IANA timezone name into a `chrono_tz::Tz`.
pub fn parse_timezone(tz: &str) -> Result<chrono_tz::Tz, BehaviorConfigError> {
    chrono_tz::Tz::from_str(tz).map_err(|_| BehaviorConfigError::BadTimezone(tz.to_string()))
}

/// Build `CachedCron` from a config — `None` unless the trigger is a
/// successfully-parseable Cron. Used by the loader and by the write-side
/// handlers after a successful `validate`.
pub fn cache_cron(cfg: &BehaviorConfig) -> Option<CachedCron> {
    if let TriggerSpec::Cron {
        schedule, timezone, ..
    } = &cfg.trigger
    {
        let schedule = parse_cron_schedule(schedule).ok()?;
        let timezone = parse_timezone(timezone).ok()?;
        Some(CachedCron { schedule, timezone })
    } else {
        None
    }
}

/// Same rules as pod ids — becomes a directory name, so no path separators
/// and no leading dot.
pub fn validate_behavior_id(id: &str) -> anyhow::Result<()> {
    if id.is_empty() {
        return Err(anyhow::anyhow!("behavior_id is empty"));
    }
    if id.starts_with('.') {
        return Err(anyhow::anyhow!("behavior_id may not start with '.'"));
    }
    if id.contains(['/', '\\', '\0']) || id == ".." {
        return Err(anyhow::anyhow!("behavior_id contains illegal characters"));
    }
    Ok(())
}

/// Write a fresh behavior's three files to disk. Creates the
/// `behaviors/<id>/` directory, fails if it already exists. `state.json`
/// is seeded with a default `BehaviorState` so subsequent state updates
/// can write-in-place.
pub async fn create_on_disk(
    pod_dir: &Path,
    behavior_id: &str,
    config: &BehaviorConfig,
    prompt: &str,
) -> anyhow::Result<()> {
    let dir = pod_dir.join(BEHAVIORS_DIR).join(behavior_id);
    if fs::try_exists(&dir).await.unwrap_or(false) {
        return Err(anyhow::anyhow!(
            "behavior `{behavior_id}` already exists at {}",
            dir.display()
        ));
    }
    fs::create_dir_all(&dir).await?;
    let toml_text = to_toml(config)?;
    fs::write(dir.join(BEHAVIOR_TOML), toml_text).await?;
    fs::write(dir.join(BEHAVIOR_PROMPT), prompt).await?;
    let state = BehaviorState::default();
    let state_json = serde_json::to_vec_pretty(&state)?;
    fs::write(dir.join(BEHAVIOR_STATE), state_json).await?;
    Ok(())
}

/// Overwrite a behavior's `behavior.toml` + `prompt.md`. Does NOT touch
/// `state.json` — the run history survives config edits.
pub async fn update_on_disk(
    pod_dir: &Path,
    behavior_id: &str,
    config: &BehaviorConfig,
    prompt: &str,
) -> anyhow::Result<()> {
    let dir = pod_dir.join(BEHAVIORS_DIR).join(behavior_id);
    if !fs::try_exists(&dir).await.unwrap_or(false) {
        return Err(anyhow::anyhow!(
            "behavior `{behavior_id}` does not exist at {}",
            dir.display()
        ));
    }
    let toml_text = to_toml(config)?;
    fs::write(dir.join(BEHAVIOR_TOML), toml_text).await?;
    fs::write(dir.join(BEHAVIOR_PROMPT), prompt).await?;
    Ok(())
}

/// Remove a behavior directory entirely. Idempotent for a missing dir
/// (returns Ok) so a racey double-delete doesn't error. Historical
/// threads that carry this behavior_id as their origin are left alone —
/// the UI surfaces them as orphaned runs.
pub async fn delete_on_disk(pod_dir: &Path, behavior_id: &str) -> anyhow::Result<()> {
    let dir = pod_dir.join(BEHAVIORS_DIR).join(behavior_id);
    match fs::remove_dir_all(&dir).await {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Persist a behavior's `state.json` in-place. Used by the on-terminal
/// hook when behavior state updates (run_count, last_outcome, etc.).
/// Returns an error rather than warn-logging so the caller can choose
/// how loudly to fail.
pub async fn write_state(
    pod_dir: &Path,
    behavior_id: &str,
    state: &BehaviorState,
) -> anyhow::Result<()> {
    let path = pod_dir
        .join(BEHAVIORS_DIR)
        .join(behavior_id)
        .join(BEHAVIOR_STATE);
    let bytes = serde_json::to_vec_pretty(state)?;
    fs::write(&path, bytes).await?;
    Ok(())
}

/// Walk `<pod_dir>/behaviors/*/` and load every behavior found there.
/// Missing `behaviors/` subdirectory → empty Vec.
///
/// Each behavior is loaded independently: a malformed `behavior.toml` does
/// NOT skip the behavior — we produce an entry with `config: None` and the
/// parse error in `load_error` so the UI can render the broken entry and
/// the user can fix it in place.  Missing `prompt.md` → empty prompt
/// (not an error; a behavior can be config-only during authoring).
/// Missing `state.json` → default state.
pub async fn load_behaviors_for_pod(pod_dir: &Path, pod_id: &str) -> Vec<Behavior> {
    let behaviors_dir = pod_dir.join(BEHAVIORS_DIR);
    let mut entries = match fs::read_dir(&behaviors_dir).await {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            warn!(dir = %behaviors_dir.display(), error = %e, "read behaviors dir failed");
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let metadata = match entry.metadata().await {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !metadata.is_dir() {
            continue;
        }
        if let Some(b) = load_one(&entry.path(), pod_id, &name).await {
            out.push(b);
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Load one behavior directory. Returns `None` only when the directory
/// contains no `behavior.toml` at all (partial/initializing-state dir) —
/// any present-but-malformed `behavior.toml` yields a Behavior with
/// `load_error` set.
async fn load_one(dir: &Path, pod_id: &str, behavior_id: &str) -> Option<Behavior> {
    let toml_path = dir.join(BEHAVIOR_TOML);
    let raw_toml = match fs::read_to_string(&toml_path).await {
        Ok(text) => text,
        Err(_) => {
            // Missing behavior.toml → not a behavior at all. Skip the entry.
            return None;
        }
    };

    let (config, load_error) = match parse_toml(&raw_toml) {
        Ok(cfg) => (Some(cfg), None),
        Err(e) => (None, Some(e.to_string())),
    };
    let cron = config.as_ref().and_then(cache_cron);

    let prompt_path = dir.join(BEHAVIOR_PROMPT);
    let prompt = fs::read_to_string(&prompt_path).await.unwrap_or_default();

    let state_path = dir.join(BEHAVIOR_STATE);
    let state = match fs::read(&state_path).await {
        Ok(bytes) => match serde_json::from_slice::<BehaviorState>(&bytes) {
            Ok(s) => s,
            Err(e) => {
                warn!(
                    path = %state_path.display(),
                    error = %e,
                    "behavior state.json failed to parse; starting from default"
                );
                BehaviorState::default()
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => BehaviorState::default(),
        Err(e) => {
            warn!(path = %state_path.display(), error = %e, "read state.json failed");
            BehaviorState::default()
        }
    };

    Some(Behavior {
        id: behavior_id.to_string(),
        pod_id: pod_id.to_string(),
        dir: dir.to_path_buf(),
        config,
        raw_toml,
        prompt,
        state,
        cron,
        load_error,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use whisper_agent_protocol::{Overlap, TriggerSpec};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Per-test scratch dir under `std::env::temp_dir()`. Matches the
    /// convention in `pod::persist::tests` — no `tempfile` dep, manual
    /// cleanup at end of each test.
    fn scratch_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("wa-behaviors-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn manual_is_minimal() {
        let cfg: BehaviorConfig = toml::from_str(r#"name = "x""#).unwrap();
        assert_eq!(cfg.name, "x");
        assert!(matches!(cfg.trigger, TriggerSpec::Manual));
    }

    #[test]
    fn parses_cron_trigger() {
        let text = r#"
name = "daily"

[trigger]
kind = "cron"
schedule = "0 9 * * *"
timezone = "America/Los_Angeles"
overlap = "queue_one"
"#;
        let cfg = parse_toml(text).unwrap();
        match cfg.trigger {
            TriggerSpec::Cron {
                schedule,
                timezone,
                overlap,
                ..
            } => {
                assert_eq!(schedule, "0 9 * * *");
                assert_eq!(timezone, "America/Los_Angeles");
                assert_eq!(overlap, Overlap::QueueOne);
            }
            _ => panic!("expected Cron"),
        }
    }

    #[test]
    fn rejects_empty_cron_schedule() {
        let text = r#"
name = "daily"

[trigger]
kind = "cron"
schedule = ""
"#;
        let err = parse_toml(text).unwrap_err();
        assert!(matches!(err, BehaviorConfigError::EmptyCronSchedule));
    }

    #[test]
    fn rejects_unparseable_cron() {
        let text = r#"
name = "bad"

[trigger]
kind = "cron"
schedule = "not a cron"
"#;
        let err = parse_toml(text).unwrap_err();
        assert!(matches!(err, BehaviorConfigError::BadCronSchedule(_, _)));
    }

    #[test]
    fn rejects_unknown_timezone() {
        let text = r#"
name = "bad"

[trigger]
kind = "cron"
schedule = "0 9 * * *"
timezone = "Not/A_Zone"
"#;
        let err = parse_toml(text).unwrap_err();
        assert!(matches!(err, BehaviorConfigError::BadTimezone(_)));
    }

    #[test]
    fn caches_cron_for_cron_triggers() {
        let cfg: BehaviorConfig = toml::from_str(
            r#"
name = "daily"

[trigger]
kind = "cron"
schedule = "0 9 * * *"
timezone = "America/Los_Angeles"
"#,
        )
        .unwrap();
        let cached = cache_cron(&cfg).expect("cron should parse");
        assert_eq!(cached.timezone, chrono_tz::America::Los_Angeles);
    }

    #[test]
    fn cache_cron_returns_none_for_non_cron_trigger() {
        let cfg: BehaviorConfig = toml::from_str(r#"name = "x""#).unwrap();
        assert!(cache_cron(&cfg).is_none());
    }

    #[test]
    fn rejects_zero_retention_days() {
        let text = r#"
name = "x"

[on_completion]
retention = "archive_after_days"
days = 0
"#;
        let err = parse_toml(text).unwrap_err();
        assert!(matches!(err, BehaviorConfigError::ZeroRetentionDays { .. }));
    }

    #[test]
    fn round_trips_through_toml() {
        let cfg = BehaviorConfig {
            name: "r".into(),
            description: Some("desc".into()),
            trigger: TriggerSpec::Cron {
                schedule: "*/5 * * * *".into(),
                timezone: "UTC".into(),
                overlap: Overlap::Skip,
                catch_up: whisper_agent_protocol::CatchUp::One,
            },
            thread: whisper_agent_protocol::BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::DeleteAfterDays { days: 7 },
        };
        let text = to_toml(&cfg).unwrap();
        let back = parse_toml(&text).unwrap();
        assert_eq!(back, cfg);
    }

    #[tokio::test]
    async fn loads_multiple_behaviors_from_disk() {
        let pod_dir = scratch_dir();
        let alpha = pod_dir.join(BEHAVIORS_DIR).join("alpha");
        write(
            &alpha.join(BEHAVIOR_TOML),
            r#"name = "Alpha"
description = "first"
"#,
        );
        write(&alpha.join(BEHAVIOR_PROMPT), "Do the alpha thing.");

        let beta = pod_dir.join(BEHAVIORS_DIR).join("beta");
        write(
            &beta.join(BEHAVIOR_TOML),
            r#"name = "Beta"

[trigger]
kind = "webhook"
"#,
        );
        // beta has no prompt and no state.json — both missing should be fine.

        let behaviors = load_behaviors_for_pod(&pod_dir, "testpod").await;
        assert_eq!(behaviors.len(), 2);
        assert_eq!(behaviors[0].id, "alpha");
        assert_eq!(behaviors[0].prompt, "Do the alpha thing.");
        assert!(behaviors[0].load_error.is_none());
        assert_eq!(behaviors[1].id, "beta");
        assert_eq!(behaviors[1].prompt, "");
        assert!(matches!(
            behaviors[1].config.as_ref().unwrap().trigger,
            TriggerSpec::Webhook
        ));
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn malformed_toml_surfaces_as_load_error() {
        let pod_dir = scratch_dir();
        let broken = pod_dir.join(BEHAVIORS_DIR).join("broken");
        write(&broken.join(BEHAVIOR_TOML), "not valid = toml {");
        let behaviors = load_behaviors_for_pod(&pod_dir, "testpod").await;
        assert_eq!(behaviors.len(), 1);
        assert!(behaviors[0].config.is_none());
        assert!(behaviors[0].load_error.is_some());
        assert_eq!(behaviors[0].id, "broken");
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn missing_behaviors_dir_yields_empty() {
        let pod_dir = scratch_dir();
        let behaviors = load_behaviors_for_pod(&pod_dir, "testpod").await;
        assert!(behaviors.is_empty());
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn skips_entries_without_behavior_toml() {
        let pod_dir = scratch_dir();
        let no_toml = pod_dir.join(BEHAVIORS_DIR).join("no-toml");
        std::fs::create_dir_all(&no_toml).unwrap();
        std::fs::write(no_toml.join("README.md"), "just a readme").unwrap();
        let behaviors = load_behaviors_for_pod(&pod_dir, "testpod").await;
        assert!(behaviors.is_empty());
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn parses_state_json_when_present() {
        let pod_dir = scratch_dir();
        let b = pod_dir.join(BEHAVIORS_DIR).join("stateful");
        write(&b.join(BEHAVIOR_TOML), r#"name = "S""#);
        write(
            &b.join(BEHAVIOR_STATE),
            r#"{"run_count": 42, "last_fired_at": "2026-04-17T09:00:00Z"}"#,
        );
        let behaviors = load_behaviors_for_pod(&pod_dir, "testpod").await;
        assert_eq!(behaviors[0].state.run_count, 42);
        assert_eq!(
            behaviors[0].state.last_fired_at.as_deref(),
            Some("2026-04-17T09:00:00Z")
        );
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn create_writes_all_three_files_and_round_trips() {
        let pod_dir = scratch_dir();
        let cfg = BehaviorConfig {
            name: "daily".into(),
            description: None,
            trigger: TriggerSpec::default(),
            thread: whisper_agent_protocol::BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
        };
        create_on_disk(&pod_dir, "daily", &cfg, "do the thing")
            .await
            .unwrap();
        let loaded = load_behaviors_for_pod(&pod_dir, "testpod").await;
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "daily");
        assert_eq!(loaded[0].config.as_ref().unwrap().name, "daily");
        assert_eq!(loaded[0].prompt, "do the thing");
        assert_eq!(loaded[0].state, BehaviorState::default());
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn create_rejects_collision() {
        let pod_dir = scratch_dir();
        let cfg = BehaviorConfig {
            name: "x".into(),
            description: None,
            trigger: TriggerSpec::default(),
            thread: whisper_agent_protocol::BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
        };
        create_on_disk(&pod_dir, "x", &cfg, "").await.unwrap();
        let err = create_on_disk(&pod_dir, "x", &cfg, "").await.unwrap_err();
        assert!(err.to_string().contains("already exists"));
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn update_preserves_state_json() {
        let pod_dir = scratch_dir();
        let cfg = BehaviorConfig {
            name: "v1".into(),
            description: None,
            trigger: TriggerSpec::default(),
            thread: whisper_agent_protocol::BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
        };
        create_on_disk(&pod_dir, "b", &cfg, "v1-prompt")
            .await
            .unwrap();
        // Simulate some accumulated state.
        write_state(
            &pod_dir,
            "b",
            &BehaviorState {
                run_count: 17,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let cfg2 = BehaviorConfig {
            name: "v2".into(),
            ..cfg
        };
        update_on_disk(&pod_dir, "b", &cfg2, "v2-prompt")
            .await
            .unwrap();

        let loaded = load_behaviors_for_pod(&pod_dir, "testpod").await;
        assert_eq!(loaded[0].config.as_ref().unwrap().name, "v2");
        assert_eq!(loaded[0].prompt, "v2-prompt");
        // Run history survived.
        assert_eq!(loaded[0].state.run_count, 17);
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn update_fails_on_missing() {
        let pod_dir = scratch_dir();
        let cfg = BehaviorConfig {
            name: "x".into(),
            description: None,
            trigger: TriggerSpec::default(),
            thread: whisper_agent_protocol::BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
        };
        let err = update_on_disk(&pod_dir, "ghost", &cfg, "")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"));
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn delete_is_idempotent_on_missing() {
        let pod_dir = scratch_dir();
        // No such behavior — delete succeeds silently.
        delete_on_disk(&pod_dir, "ghost").await.unwrap();
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn delete_removes_behavior_dir() {
        let pod_dir = scratch_dir();
        let cfg = BehaviorConfig {
            name: "x".into(),
            description: None,
            trigger: TriggerSpec::default(),
            thread: whisper_agent_protocol::BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
        };
        create_on_disk(&pod_dir, "gone", &cfg, "").await.unwrap();
        delete_on_disk(&pod_dir, "gone").await.unwrap();
        let loaded = load_behaviors_for_pod(&pod_dir, "testpod").await;
        assert!(loaded.is_empty());
        let _ = std::fs::remove_dir_all(&pod_dir);
    }
}
