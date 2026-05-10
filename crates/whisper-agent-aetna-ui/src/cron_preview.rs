//! Cron-preview helpers for the behavior editor's Trigger tab.
//!
//! Parsing matches the server's behavior-trigger parser exactly: a
//! 5-field UNIX crontab (`min hour dom mon dow`) is prefixed with
//! `"0 "` for the seconds field before handing to the [`cron`] crate,
//! so the preview's "next 5 firings" table agrees with what the
//! server's scheduler will actually fire.
//!
//! The render helpers are aetna-flavored and live in
//! [`crate::app`] alongside the rest of the behavior editor's
//! widgetry — this module owns only the parsing, presets, and the
//! relative-time formatter so a future native preview surface
//! (or a different UI) can reuse them.

use std::str::FromStr;

use chrono::{DateTime, Duration, Utc};
use chrono_tz::Tz;
use cron::Schedule;

/// Number of upcoming firings the preview surfaces.
pub const PREVIEW_COUNT: usize = 5;

/// Quick-pick crontab expressions. Labels are user-facing; expressions
/// are the raw 5-field strings written into the editor on click. Kept
/// short — common shapes a human would type, not exhaustive coverage.
pub const CRON_PRESETS: &[(&str, &str)] = &[
    ("every minute", "* * * * *"),
    ("every 15 min", "*/15 * * * *"),
    ("hourly", "0 * * * *"),
    ("daily 9am", "0 9 * * *"),
    ("weekdays 9am", "0 9 * * 1-5"),
    ("weekly Mon 9am", "0 9 * * 1"),
    ("monthly 1st 9am", "0 9 1 * *"),
];

/// Common IANA zones shown as quick-picks under the timezone input.
/// `chrono_tz::TZ_VARIANTS` has ~600 entries — the full list is too
/// much to browse in a button row, and `UTC` covers "I don't care".
pub const COMMON_TIMEZONES: &[&str] = &[
    "UTC",
    "America/Los_Angeles",
    "America/Denver",
    "America/Chicago",
    "America/New_York",
    "Europe/London",
    "Europe/Berlin",
    "Asia/Tokyo",
    "Australia/Sydney",
];

/// Parse a 5-field crontab. Matches the server's semantics exactly:
/// prepend `"0 "` for the seconds field before handing to `cron`.
pub fn parse_schedule(schedule: &str) -> Result<Schedule, String> {
    let trimmed = schedule.trim();
    if trimmed.is_empty() {
        return Err("schedule is empty".into());
    }
    let six_field = format!("0 {trimmed}");
    Schedule::from_str(&six_field).map_err(|e| e.to_string())
}

/// Parse an IANA timezone name.
pub fn parse_tz(tz: &str) -> Result<Tz, String> {
    Tz::from_str(tz.trim()).map_err(|_| format!("unknown timezone `{tz}`"))
}

/// Compute the next [`PREVIEW_COUNT`] firings from now in the given
/// timezone. Returns an empty vec when the schedule has no upcoming
/// fires (e.g. an only-in-the-past expression).
pub fn next_firings(schedule: &Schedule, tz: Tz) -> Vec<DateTime<Tz>> {
    let now_utc = Utc::now();
    let now_in_tz = now_utc.with_timezone(&tz);
    schedule.after(&now_in_tz).take(PREVIEW_COUNT).collect()
}

/// Human-readable "in Xd Yh" / "in Xh Ym" / "in Xm" / "in Xs" between
/// `now` and `target`. Negative deltas (shouldn't happen since
/// [`Schedule::after`] is strictly future) render as "now" rather
/// than "-Xs".
pub fn format_relative(now: DateTime<Utc>, target: DateTime<Utc>) -> String {
    let delta = target.signed_duration_since(now);
    if delta <= Duration::zero() {
        return "now".into();
    }
    let days = delta.num_days();
    let hours = delta.num_hours() - days * 24;
    let minutes = delta.num_minutes() - delta.num_hours() * 60;
    let seconds = delta.num_seconds() - delta.num_minutes() * 60;
    if days > 0 {
        format!("in {days}d {hours}h")
    } else if delta.num_hours() > 0 {
        format!("in {hours}h {minutes}m")
    } else if delta.num_minutes() > 0 {
        format!("in {minutes}m")
    } else {
        format!("in {seconds}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_schedule_accepts_five_fields() {
        parse_schedule("0 9 * * *").unwrap();
        parse_schedule("*/15 * * * *").unwrap();
        parse_schedule("0 9 * * 1-5").unwrap();
    }

    #[test]
    fn parse_schedule_rejects_garbage_and_empty() {
        assert!(parse_schedule("").is_err());
        assert!(parse_schedule("   ").is_err());
        assert!(parse_schedule("not a cron").is_err());
        assert!(parse_schedule("0 9").is_err());
    }

    #[test]
    fn parse_tz_accepts_known_and_rejects_garbage() {
        parse_tz("UTC").unwrap();
        parse_tz("America/Los_Angeles").unwrap();
        assert!(parse_tz("Not/A_Zone").is_err());
        assert!(parse_tz("").is_err());
    }

    #[test]
    fn presets_all_parse() {
        for (label, expr) in CRON_PRESETS {
            parse_schedule(expr).unwrap_or_else(|e| panic!("preset `{label}` broken: {e}"));
        }
    }

    #[test]
    fn common_timezones_all_parse() {
        for tz in COMMON_TIMEZONES {
            parse_tz(tz).unwrap_or_else(|e| panic!("common tz `{tz}` broken: {e}"));
        }
    }

    #[test]
    fn format_relative_shapes() {
        let now = Utc::now();
        assert_eq!(format_relative(now, now - Duration::seconds(5)), "now");
        assert_eq!(format_relative(now, now + Duration::seconds(30)), "in 30s");
        assert_eq!(format_relative(now, now + Duration::minutes(15)), "in 15m");
        assert!(format_relative(now, now + Duration::hours(2)).starts_with("in 2h"));
        assert!(format_relative(now, now + Duration::days(3)).starts_with("in 3d"));
    }
}
