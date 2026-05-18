//! Codex / ChatGPT-subscription usage-snapshot parsers.
//!
//! Two surfaces for the same data:
//!
//! - [`parse_codex_headers`] reads the `x-codex-*` response headers attached
//!   to every `/responses` SSE reply. Free with every model turn — populated
//!   for the live UI chip without an extra request. Carries window
//!   utilizations and reset epochs; never carries plan type.
//! - [`parse_wham_usage_response`] decodes the JSON body of
//!   `GET https://chatgpt.com/backend-api/wham/usage`. Triggered on demand
//!   when the user opens the usage dropdown. Adds `planType` and the
//!   `credits` block that headers don't carry.
//!
//! Both produce the wire-shared [`BackendUsage`] type; the only normalization
//! that happens here is filling the `captured_at` timestamp and tagging the
//! [`UsageSource`]. Header values are already in percent units (Codex's
//! convention); JSON values too. Anthropic's future analog will need
//! 0–1 → 0–100 normalization at *its* parser, not here.

use chrono::Utc;
use reqwest::header::HeaderMap;
use serde::Deserialize;
use whisper_agent_protocol::{BackendUsage, UsageCredits, UsageSource, UsageWindow};

/// Extract a Codex usage snapshot from response headers. Returns `None`
/// when the response carries no `x-codex-*` headers at all — the API-key
/// route (`api.openai.com`) doesn't emit these, and we want a clean
/// "nothing changed" signal there rather than a stub snapshot.
///
/// Headers that are present but unparseable are silently skipped (the
/// upstream sometimes ships extra fields we don't know about yet); a
/// snapshot lands as long as *any* recognized field parsed.
pub fn parse_codex_headers(headers: &HeaderMap) -> Option<BackendUsage> {
    let primary = parse_window(headers, "x-codex-primary");
    let secondary = parse_window(headers, "x-codex-secondary");
    let credits = parse_credits(headers);

    if primary.is_none() && secondary.is_none() && credits.is_none() {
        return None;
    }

    Some(BackendUsage {
        primary,
        secondary,
        credits,
        plan_type: None,
        captured_at: Utc::now().to_rfc3339(),
        source: UsageSource::Headers,
    })
}

fn parse_window(headers: &HeaderMap, prefix: &str) -> Option<UsageWindow> {
    let used_percent = header_f32(headers, &format!("{prefix}-used-percent"));
    let window_minutes = header_u32(headers, &format!("{prefix}-window-minutes"));
    let resets_at_epoch = header_i64(headers, &format!("{prefix}-reset-at"));
    // Used-percent is the load-bearing field — without it the window
    // tells the user nothing actionable. Reset/duration alone wouldn't
    // be worth surfacing.
    used_percent.map(|used_percent| UsageWindow {
        used_percent,
        window_minutes,
        resets_at_epoch,
    })
}

fn parse_credits(headers: &HeaderMap) -> Option<UsageCredits> {
    let has_credits = header_bool(headers, "x-codex-credits-has-credits");
    let unlimited = header_bool(headers, "x-codex-credits-unlimited");
    let balance = header_str(headers, "x-codex-credits-balance").map(str::to_string);
    if has_credits.is_none() && unlimited.is_none() && balance.is_none() {
        return None;
    }
    Some(UsageCredits {
        has_credits: has_credits.unwrap_or(false),
        unlimited: unlimited.unwrap_or(false),
        balance,
    })
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}
fn header_f32(headers: &HeaderMap, name: &str) -> Option<f32> {
    header_str(headers, name).and_then(|s| s.parse().ok())
}
fn header_u32(headers: &HeaderMap, name: &str) -> Option<u32> {
    header_str(headers, name).and_then(|s| s.parse().ok())
}
fn header_i64(headers: &HeaderMap, name: &str) -> Option<i64> {
    header_str(headers, name).and_then(|s| s.parse().ok())
}
fn header_bool(headers: &HeaderMap, name: &str) -> Option<bool> {
    header_str(headers, name).and_then(|s| match s {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    })
}

/// Decoded shape of the `/backend-api/wham/usage` response. The upstream
/// returns a richer envelope (`rateLimitsByLimitId` etc.); we ignore the
/// per-limit-id breakdown and consume the top-level `rateLimits` block
/// which carries the user-facing summary.
#[derive(Debug, Deserialize)]
struct WhamUsageBody {
    #[serde(rename = "rateLimits")]
    rate_limits: Option<WhamRateLimits>,
}

#[derive(Debug, Deserialize)]
struct WhamRateLimits {
    #[serde(rename = "planType")]
    plan_type: Option<String>,
    primary: Option<WhamWindow>,
    secondary: Option<WhamWindow>,
    credits: Option<WhamCredits>,
}

#[derive(Debug, Deserialize)]
struct WhamWindow {
    #[serde(rename = "usedPercent")]
    used_percent: Option<f32>,
    #[serde(rename = "windowDurationMins")]
    window_duration_mins: Option<u32>,
    #[serde(rename = "resetsAt")]
    resets_at: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct WhamCredits {
    #[serde(rename = "hasCredits")]
    has_credits: Option<bool>,
    unlimited: Option<bool>,
    balance: Option<String>,
}

/// Decode a `/backend-api/wham/usage` JSON body. Returns `None` when the
/// body parses but carries no usable fields (e.g. the upstream returned
/// an empty `rateLimits` block for an enterprise account that's opted
/// out of the metered surface).
pub fn parse_wham_usage_response(body: &str) -> Result<Option<BackendUsage>, serde_json::Error> {
    let parsed: WhamUsageBody = serde_json::from_str(body)?;
    let Some(rl) = parsed.rate_limits else {
        return Ok(None);
    };
    let primary = rl.primary.and_then(wham_to_window);
    let secondary = rl.secondary.and_then(wham_to_window);
    let credits = rl.credits.and_then(wham_to_credits);
    if primary.is_none() && secondary.is_none() && credits.is_none() && rl.plan_type.is_none() {
        return Ok(None);
    }
    Ok(Some(BackendUsage {
        primary,
        secondary,
        credits,
        plan_type: rl.plan_type,
        captured_at: Utc::now().to_rfc3339(),
        source: UsageSource::Poll,
    }))
}

/// Decoded shape of a `codex.rate_limits` WS frame. The chatgpt.com
/// codex backend ships per-window utilization on the SSE path as
/// `x-codex-*` response headers and on the WS path as one of these
/// frames at the start of the response stream. Same data, different
/// envelope — see [`parse_codex_rate_limits_frame`] for the
/// frame-side parser that emits the same `BackendUsage` shape the
/// header path does.
#[derive(Debug, Deserialize)]
struct CodexRateLimitsFrame {
    #[serde(rename = "type")]
    ty: String,
    #[serde(rename = "plan_type")]
    plan_type: Option<String>,
    rate_limits: Option<CodexRateLimitsBody>,
}

#[derive(Debug, Deserialize)]
struct CodexRateLimitsBody {
    primary: Option<CodexRateLimitsWindow>,
    secondary: Option<CodexRateLimitsWindow>,
}

#[derive(Debug, Deserialize)]
struct CodexRateLimitsWindow {
    used_percent: Option<f32>,
    window_minutes: Option<u32>,
    reset_at: Option<i64>,
}

/// Parse a WS event frame as a `codex.rate_limits` snapshot, returning
/// the same `BackendUsage` shape the SSE header parser produces.
/// Returns `None` if the frame is not a rate-limits event, doesn't
/// parse, or carries no usable window data — the caller is expected
/// to fall through to the regular event-frame parser in that case.
pub fn parse_codex_rate_limits_frame(text: &str) -> Option<BackendUsage> {
    // Cheap discriminator probe — avoid parsing every event frame's
    // body when only a small fraction are rate-limits events.
    if !text.contains("\"codex.rate_limits\"") {
        return None;
    }
    let parsed: CodexRateLimitsFrame = serde_json::from_str(text).ok()?;
    if parsed.ty != "codex.rate_limits" {
        return None;
    }
    let rl = parsed.rate_limits?;
    let primary = rl.primary.and_then(codex_window_to_usage);
    let secondary = rl.secondary.and_then(codex_window_to_usage);
    if primary.is_none() && secondary.is_none() && parsed.plan_type.is_none() {
        return None;
    }
    Some(BackendUsage {
        primary,
        secondary,
        // The frame doesn't include credits — the codex CLI surfaces
        // those only via the wham/usage poll. Leave None here; the
        // on-demand poll fills it when the user opens the dropdown.
        credits: None,
        plan_type: parsed.plan_type,
        captured_at: Utc::now().to_rfc3339(),
        source: UsageSource::Headers,
    })
}

fn codex_window_to_usage(w: CodexRateLimitsWindow) -> Option<UsageWindow> {
    w.used_percent.map(|used_percent| UsageWindow {
        used_percent,
        window_minutes: w.window_minutes,
        resets_at_epoch: w.reset_at,
    })
}

fn wham_to_window(w: WhamWindow) -> Option<UsageWindow> {
    w.used_percent.map(|used_percent| UsageWindow {
        used_percent,
        window_minutes: w.window_duration_mins,
        resets_at_epoch: w.resets_at,
    })
}

fn wham_to_credits(c: WhamCredits) -> Option<UsageCredits> {
    if c.has_credits.is_none() && c.unlimited.is_none() && c.balance.is_none() {
        return None;
    }
    Some(UsageCredits {
        has_credits: c.has_credits.unwrap_or(false),
        unlimited: c.unlimited.unwrap_or(false),
        balance: c.balance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    fn hm(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn headers_empty_returns_none() {
        assert!(parse_codex_headers(&HeaderMap::new()).is_none());
    }

    #[test]
    fn ws_rate_limits_frame_parses() {
        // Sample lifted from a live `codex.rate_limits` WS frame.
        // The fields we care about: type discriminator, plan_type at
        // the top level, primary + secondary windows under
        // `rate_limits`.
        let frame = r#"{
            "type": "codex.rate_limits",
            "plan_type": "pro",
            "rate_limits": {
                "allowed": true,
                "limit_reached": false,
                "primary": {"used_percent": 12.5, "window_minutes": 300, "reset_after_seconds": 600, "reset_at": 1764554400},
                "secondary": {"used_percent": 73, "window_minutes": 10080, "reset_after_seconds": 600, "reset_at": 1764615600}
            }
        }"#;
        let snap = parse_codex_rate_limits_frame(frame).expect("should parse");
        assert_eq!(snap.plan_type.as_deref(), Some("pro"));
        let primary = snap.primary.expect("primary window");
        assert!((primary.used_percent - 12.5).abs() < 0.001);
        assert_eq!(primary.window_minutes, Some(300));
        assert_eq!(primary.resets_at_epoch, Some(1764554400));
        let secondary = snap.secondary.expect("secondary window");
        assert!((secondary.used_percent - 73.0).abs() < 0.001);
        assert_eq!(secondary.window_minutes, Some(10080));
        // Frames don't carry credits — those only come from the
        // wham/usage poll.
        assert!(snap.credits.is_none());
    }

    #[test]
    fn ws_rate_limits_frame_rejects_unrelated_event_types() {
        let frame = r#"{"type": "response.created", "response": {}}"#;
        assert!(parse_codex_rate_limits_frame(frame).is_none());
    }

    #[test]
    fn ws_rate_limits_frame_rejects_garbage() {
        // Not JSON.
        assert!(parse_codex_rate_limits_frame("codex.rate_limits = oops").is_none());
        // JSON but missing rate_limits body — nothing actionable.
        let frame = r#"{"type":"codex.rate_limits"}"#;
        assert!(parse_codex_rate_limits_frame(frame).is_none());
    }

    #[test]
    fn headers_with_unrelated_keys_returns_none() {
        // Headers from the API-key route (api.openai.com) carry
        // `x-ratelimit-*`, not `x-codex-*`. Must not fabricate a
        // snapshot from a foreign provider's headers.
        let h = hm(&[
            ("x-ratelimit-limit-requests", "1000"),
            ("x-ratelimit-remaining-requests", "999"),
        ]);
        assert!(parse_codex_headers(&h).is_none());
    }

    #[test]
    fn headers_parses_full_codex_set() {
        let h = hm(&[
            ("x-codex-primary-used-percent", "12.5"),
            ("x-codex-primary-window-minutes", "300"),
            ("x-codex-primary-reset-at", "1764554400"),
            ("x-codex-secondary-used-percent", "73"),
            ("x-codex-secondary-window-minutes", "10080"),
            ("x-codex-secondary-reset-at", "1764615600"),
            ("x-codex-credits-has-credits", "true"),
            ("x-codex-credits-unlimited", "false"),
            ("x-codex-credits-balance", "5.42"),
        ]);
        let usage = parse_codex_headers(&h).expect("snapshot");
        let primary = usage.primary.expect("primary");
        assert!((primary.used_percent - 12.5).abs() < f32::EPSILON);
        assert_eq!(primary.window_minutes, Some(300));
        assert_eq!(primary.resets_at_epoch, Some(1764554400));
        let secondary = usage.secondary.expect("secondary");
        assert_eq!(secondary.window_minutes, Some(10080));
        let credits = usage.credits.expect("credits");
        assert!(credits.has_credits);
        assert!(!credits.unlimited);
        assert_eq!(credits.balance.as_deref(), Some("5.42"));
        assert_eq!(usage.source, UsageSource::Headers);
        // Header path never knows the plan name — that's a poll-only
        // field.
        assert!(usage.plan_type.is_none());
    }

    #[test]
    fn headers_partial_primary_only() {
        // One window present, other absent — should still emit a
        // snapshot (the UI can render the bucket it knows about and
        // leave the other column blank).
        let h = hm(&[("x-codex-primary-used-percent", "8")]);
        let usage = parse_codex_headers(&h).expect("snapshot");
        assert!(usage.primary.is_some());
        assert!(usage.secondary.is_none());
        assert!(usage.credits.is_none());
    }

    #[test]
    fn headers_unparseable_value_is_skipped_not_fatal() {
        // Unknown / garbage field shouldn't poison a snapshot that
        // also has parseable fields. Reset-at is garbage; used-percent
        // still gives us a window.
        let h = hm(&[
            ("x-codex-primary-used-percent", "42"),
            ("x-codex-primary-reset-at", "not-a-number"),
        ]);
        let usage = parse_codex_headers(&h).expect("snapshot");
        let primary = usage.primary.expect("primary");
        assert!((primary.used_percent - 42.0).abs() < f32::EPSILON);
        assert!(primary.resets_at_epoch.is_none());
    }

    #[test]
    fn wham_full_response() {
        let body = r#"{
            "rateLimits": {
                "limitId": "codex",
                "limitName": "Codex",
                "planType": "plus",
                "primary":   { "usedPercent": 12,  "windowDurationMins": 300,   "resetsAt": 1764554400 },
                "secondary": { "usedPercent": 73,  "windowDurationMins": 10080, "resetsAt": 1764615600 },
                "credits":   { "hasCredits": true, "unlimited": false,          "balance": "5.42" },
                "rateLimitReachedType": null
            },
            "rateLimitsByLimitId": {}
        }"#;
        let usage = parse_wham_usage_response(body)
            .expect("parse ok")
            .expect("snapshot present");
        assert_eq!(usage.source, UsageSource::Poll);
        assert_eq!(usage.plan_type.as_deref(), Some("plus"));
        let primary = usage.primary.expect("primary");
        assert!((primary.used_percent - 12.0).abs() < f32::EPSILON);
        assert_eq!(primary.window_minutes, Some(300));
        let credits = usage.credits.expect("credits");
        assert_eq!(credits.balance.as_deref(), Some("5.42"));
    }

    #[test]
    fn wham_empty_rate_limits_returns_none() {
        // Enterprise / opted-out accounts can return the envelope
        // with no useful fields. Don't fabricate a snapshot from
        // pure metadata.
        let body = r#"{ "rateLimits": { "limitId": "codex" } }"#;
        let got = parse_wham_usage_response(body).expect("parse ok");
        assert!(got.is_none());
    }

    #[test]
    fn wham_missing_rate_limits_returns_none() {
        let body = r#"{}"#;
        let got = parse_wham_usage_response(body).expect("parse ok");
        assert!(got.is_none());
    }

    #[test]
    fn wham_plan_type_only_still_a_snapshot() {
        // Plan name alone is worth keeping — the dropdown badge will
        // render "Plus" with no progress bars rather than nothing.
        let body = r#"{ "rateLimits": { "planType": "free" } }"#;
        let usage = parse_wham_usage_response(body)
            .expect("parse ok")
            .expect("snapshot present");
        assert_eq!(usage.plan_type.as_deref(), Some("free"));
        assert!(usage.primary.is_none());
        assert!(usage.secondary.is_none());
    }

    #[test]
    fn wham_garbage_body_errors() {
        assert!(parse_wham_usage_response("not json").is_err());
    }
}
