//! Behavior-editor helpers for the cron trigger sub-form:
//! live schedule/timezone validation, preset quick-pick rows, and a
//! "next N firings" preview.
//!
//! Parsing mirrors `src/pod/behaviors.rs`: a 5-field UNIX crontab
//! (`min hour dom mon dow`) is prefixed with `"0 "` for the seconds
//! field before handing to the `cron` crate. That keeps preview fires
//! identical to what the server will schedule — if the preview says
//! the next fire is `09:00`, the scheduler agrees.

use std::str::FromStr;

use chrono::{DateTime, Duration, Utc};
use chrono_tz::Tz;
use cron::Schedule;
use egui::{Color32, RichText};

/// Number of upcoming firings to display.
const PREVIEW_COUNT: usize = 5;

/// Quick-pick crontab expressions. Labels are user-facing; expressions
/// are the raw 5-field strings written into the editor on click. Kept
/// short — the intent is "common shapes a human would type", not
/// exhaustive coverage.
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

/// Row of preset buttons. Clicking one overwrites `schedule` with the
/// preset expression. The hover tooltip shows the raw expression so
/// the user can confirm before committing.
pub fn render_schedule_presets(ui: &mut egui::Ui, schedule: &mut String) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("presets:").weak().small());
        for (label, expr) in CRON_PRESETS {
            if ui.small_button(*label).on_hover_text(*expr).clicked() {
                *schedule = (*expr).to_string();
            }
        }
    });
}

/// Row of common-timezone quick-picks. Clicking sets `timezone` to
/// that IANA name.
pub fn render_tz_presets(ui: &mut egui::Ui, timezone: &mut String) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("common:").weak().small());
        for tz in COMMON_TIMEZONES {
            if ui.small_button(*tz).clicked() {
                *timezone = (*tz).to_string();
            }
        }
    });
}

/// Validation + next-firings preview panel. Reads the current schedule
/// and timezone as plain strings and draws either an error line or a
/// small table of the next [`PREVIEW_COUNT`] fires.
pub fn render_preview(ui: &mut egui::Ui, schedule_str: &str, tz_str: &str) {
    match (parse_schedule(schedule_str), parse_tz(tz_str)) {
        (Err(e), _) => error_line(ui, format!("schedule: {e}")),
        (_, Err(e)) => error_line(ui, e),
        (Ok(schedule), Ok(tz)) => render_firings(ui, &schedule, tz),
    }
}

fn error_line(ui: &mut egui::Ui, msg: String) {
    ui.label(
        RichText::new(msg)
            .color(Color32::from_rgb(220, 80, 80))
            .small(),
    );
}

fn render_firings(ui: &mut egui::Ui, schedule: &Schedule, tz: Tz) {
    let now_utc = Utc::now();
    let now_in_tz = now_utc.with_timezone(&tz);
    let upcoming: Vec<DateTime<Tz>> = schedule.after(&now_in_tz).take(PREVIEW_COUNT).collect();

    if upcoming.is_empty() {
        error_line(ui, "schedule has no upcoming fires".into());
        return;
    }

    ui.label(
        RichText::new(format!("next {} firings ({})", upcoming.len(), tz.name()))
            .weak()
            .small(),
    );
    egui::Grid::new("behavior_editor_cron_preview")
        .num_columns(2)
        .spacing([16.0, 2.0])
        .show(ui, |ui| {
            for fire in &upcoming {
                ui.monospace(fire.format("%Y-%m-%d %H:%M %Z").to_string());
                let rel = format_relative(now_utc, fire.with_timezone(&Utc));
                ui.label(RichText::new(rel).weak().small());
                ui.end_row();
            }
        });
}

/// Human-readable "in Xd Yh" / "in Xh Ym" / "in Xm" / "in Xs" between
/// `now` and `target`. Negative deltas (shouldn't happen since cron's
/// `.after(now)` is strictly future) render as "now" rather than "-Xs".
fn format_relative(now: DateTime<Utc>, target: DateTime<Utc>) -> String {
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
