//! Pure helpers for behavior trigger evaluation — cron-due math, RFC-3339
//! cursor parsing, webhook-target validation, catch-up counting.
//!
//! Kept separate from the scheduler state machine so the arithmetic is
//! testable in isolation and so the timezone rules (evaluate a schedule
//! in its *own* tz, not UTC) live next to the function that depends on
//! them.

use super::TriggerFireError;

/// Parse a persisted RFC-3339 timestamp back to a `DateTime<Utc>`.
/// Returns `None` for `None` input or unparseable strings — callers
/// treat both identically (behavior has no usable cursor).
pub(super) fn parse_rfc3339_utc(raw: Option<&str>) -> Option<chrono::DateTime<chrono::Utc>> {
    let text = raw?;
    chrono::DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|dt| dt.with_timezone(&chrono::Utc))
}

/// Decide whether a cron behavior's next scheduled occurrence has
/// arrived. `cursor` is the persisted `last_fired_at`; `now` is the
/// evaluation time. Returns true when the next occurrence strictly
/// after `cursor` is ≤ `now` — i.e., a scheduled fire window has
/// opened since we last acted on this behavior.
///
/// The 30 s tick cadence means this can fire up to 30 s after the
/// scheduled minute; that's acceptable slop. If a fire window is
/// missed entirely (e.g., server down through the scheduled moment),
/// the next `now` will be past `next_fire` and the function still
/// returns true — on-restart catch-up is handled in phase 3c.
pub(super) fn is_cron_due(
    cron: &crate::pod::behaviors::CachedCron,
    cursor: chrono::DateTime<chrono::Utc>,
    now: chrono::DateTime<chrono::Utc>,
) -> bool {
    // Evaluate the schedule in its own timezone — "9am Pacific" vs
    // "9am UTC" matters for every cron that encodes a human-scale
    // cadence. The `after` iterator yields zoned timestamps; we
    // convert to UTC for comparison against `now`.
    let cursor_tz = cursor.with_timezone(&cron.timezone);
    cron.schedule
        .after(&cursor_tz)
        .next()
        .map(|next| next.with_timezone(&chrono::Utc) <= now)
        .unwrap_or(false)
}

/// Validate that a behavior can service a webhook trigger: must be
/// loaded cleanly AND its trigger kind must be `Webhook`. Returns
/// the `Overlap` policy on success so callers can hand it to
/// `fire_trigger` without re-matching. Pure function — easier to
/// test than the full `handle_webhook_trigger` flow, which needs a
/// live scheduler.
pub(super) fn validate_webhook_target(
    behavior: &crate::pod::behaviors::Behavior,
) -> Result<whisper_agent_protocol::Overlap, TriggerFireError> {
    if let Some(err) = &behavior.load_error {
        return Err(TriggerFireError::BehaviorLoadError(err.clone()));
    }
    let cfg = behavior
        .config
        .as_ref()
        .ok_or_else(|| TriggerFireError::BehaviorLoadError("no parsed config".into()))?;
    match &cfg.trigger {
        whisper_agent_protocol::TriggerSpec::Webhook { overlap } => Ok(*overlap),
        whisper_agent_protocol::TriggerSpec::Manual => {
            Err(TriggerFireError::NotWebhookTrigger("manual"))
        }
        whisper_agent_protocol::TriggerSpec::Cron { .. } => {
            Err(TriggerFireError::NotWebhookTrigger("cron"))
        }
    }
}

/// Count cron occurrences strictly between `cursor` (exclusive) and
/// `now` (inclusive). Used for startup catch-up logging — informational,
/// not load-bearing. Capped at `limit` so a cursor far in the past
/// (e.g., years of accumulated downtime) doesn't wedge the scheduler
/// boot; a return value equal to `limit` means "at least N."
pub(super) fn count_missed_occurrences(
    cron: &crate::pod::behaviors::CachedCron,
    cursor: chrono::DateTime<chrono::Utc>,
    now: chrono::DateTime<chrono::Utc>,
    limit: u64,
) -> u64 {
    let cursor_tz = cursor.with_timezone(&cron.timezone);
    let mut count: u64 = 0;
    for occ in cron.schedule.after(&cursor_tz) {
        if occ.with_timezone(&chrono::Utc) > now {
            break;
        }
        count += 1;
        if count >= limit {
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::Overlap;

    #[test]
    fn parse_rfc3339_utc_round_trips() {
        let dt = chrono::DateTime::parse_from_rfc3339("2026-04-17T09:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let parsed = parse_rfc3339_utc(Some(&dt.to_rfc3339())).unwrap();
        assert_eq!(parsed, dt);
    }

    #[test]
    fn parse_rfc3339_utc_handles_missing_and_malformed() {
        assert!(parse_rfc3339_utc(None).is_none());
        assert!(parse_rfc3339_utc(Some("nope")).is_none());
    }

    #[test]
    fn cron_not_due_before_next_occurrence() {
        let cron = crate::pod::behaviors::CachedCron {
            schedule: crate::pod::behaviors::parse_cron_schedule("0 9 * * *").unwrap(),
            timezone: chrono_tz::UTC,
        };
        // cursor at 08:00 UTC, now at 08:30 UTC — next fire is 09:00, not yet due.
        let cursor = chrono::DateTime::parse_from_rfc3339("2026-04-17T08:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let now = chrono::DateTime::parse_from_rfc3339("2026-04-17T08:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert!(!is_cron_due(&cron, cursor, now));
    }

    #[test]
    fn cron_due_at_and_after_next_occurrence() {
        let cron = crate::pod::behaviors::CachedCron {
            schedule: crate::pod::behaviors::parse_cron_schedule("0 9 * * *").unwrap(),
            timezone: chrono_tz::UTC,
        };
        let cursor = chrono::DateTime::parse_from_rfc3339("2026-04-17T08:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let at = chrono::DateTime::parse_from_rfc3339("2026-04-17T09:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let after = chrono::DateTime::parse_from_rfc3339("2026-04-17T09:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert!(is_cron_due(&cron, cursor, at));
        assert!(is_cron_due(&cron, cursor, after));
    }

    #[test]
    fn validate_webhook_accepts_webhook_behaviors() {
        let behavior = behavior_fixture(
            Some(whisper_agent_protocol::TriggerSpec::Webhook {
                overlap: Overlap::QueueOne,
            }),
            None,
        );
        let overlap = validate_webhook_target(&behavior).unwrap();
        assert_eq!(overlap, Overlap::QueueOne);
    }

    #[test]
    fn validate_webhook_rejects_manual_trigger() {
        let behavior = behavior_fixture(Some(whisper_agent_protocol::TriggerSpec::Manual), None);
        let err = validate_webhook_target(&behavior).unwrap_err();
        assert!(matches!(err, TriggerFireError::NotWebhookTrigger("manual")));
    }

    #[test]
    fn validate_webhook_rejects_cron_trigger() {
        let behavior = behavior_fixture(
            Some(whisper_agent_protocol::TriggerSpec::Cron {
                schedule: "0 9 * * *".into(),
                timezone: "UTC".into(),
                overlap: Overlap::Skip,
                catch_up: whisper_agent_protocol::CatchUp::One,
            }),
            None,
        );
        let err = validate_webhook_target(&behavior).unwrap_err();
        assert!(matches!(err, TriggerFireError::NotWebhookTrigger("cron")));
    }

    #[test]
    fn validate_webhook_surfaces_load_error() {
        let behavior = behavior_fixture(None, Some("toml parse: bad".into()));
        let err = validate_webhook_target(&behavior).unwrap_err();
        assert!(matches!(err, TriggerFireError::BehaviorLoadError(_)));
    }

    fn behavior_fixture(
        trigger: Option<whisper_agent_protocol::TriggerSpec>,
        load_error: Option<String>,
    ) -> crate::pod::behaviors::Behavior {
        let config = trigger.map(|t| whisper_agent_protocol::BehaviorConfig {
            name: "t".into(),
            description: None,
            trigger: t,
            thread: whisper_agent_protocol::BehaviorThreadOverride::default(),
            on_completion: whisper_agent_protocol::RetentionPolicy::default(),
        });
        crate::pod::behaviors::Behavior {
            id: "b".into(),
            pod_id: "p".into(),
            dir: std::path::PathBuf::from("/tmp"),
            config,
            raw_toml: String::new(),
            prompt: String::new(),
            state: whisper_agent_protocol::BehaviorState::default(),
            cron: None,
            load_error,
        }
    }

    #[test]
    fn count_missed_counts_inclusive_of_now() {
        let cron = crate::pod::behaviors::CachedCron {
            schedule: crate::pod::behaviors::parse_cron_schedule("0 * * * *").unwrap(), // hourly
            timezone: chrono_tz::UTC,
        };
        let cursor = chrono::DateTime::parse_from_rfc3339("2026-04-17T09:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let now = chrono::DateTime::parse_from_rfc3339("2026-04-17T13:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        // Occurrences in (09:30, 13:00]: 10:00, 11:00, 12:00, 13:00 = 4.
        assert_eq!(count_missed_occurrences(&cron, cursor, now, 1000), 4);
    }

    #[test]
    fn count_missed_respects_limit() {
        let cron = crate::pod::behaviors::CachedCron {
            schedule: crate::pod::behaviors::parse_cron_schedule("* * * * *").unwrap(), // minutely
            timezone: chrono_tz::UTC,
        };
        let cursor = chrono::DateTime::parse_from_rfc3339("2026-04-17T09:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let now = chrono::DateTime::parse_from_rfc3339("2026-04-17T11:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        // 120 minutes worth; limit=5 must cap.
        assert_eq!(count_missed_occurrences(&cron, cursor, now, 5), 5);
    }

    #[test]
    fn cron_due_honors_timezone() {
        // 9am Pacific == 17:00 UTC (during PDT). A cursor at 16:00 UTC
        // and now at 16:30 UTC is NOT due (it's 9:30am Pacific but
        // previous occurrence was 9am yesterday, yielding next == today
        // 9am, which in UTC is still an hour off). The cursor needs
        // to be before today's 9am Pacific for is_cron_due to return
        // true against now >= 17:00 UTC.
        let cron = crate::pod::behaviors::CachedCron {
            schedule: crate::pod::behaviors::parse_cron_schedule("0 9 * * *").unwrap(),
            timezone: chrono_tz::America::Los_Angeles,
        };
        // Use a date in PDT (July, UTC-7).
        let cursor = chrono::DateTime::parse_from_rfc3339("2026-07-17T15:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc); // 08:00 Pacific
        let before = chrono::DateTime::parse_from_rfc3339("2026-07-17T15:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc); // 08:30 Pacific
        let at = chrono::DateTime::parse_from_rfc3339("2026-07-17T16:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc); // 09:00 Pacific
        assert!(!is_cron_due(&cron, cursor, before));
        assert!(is_cron_due(&cron, cursor, at));
    }
}
