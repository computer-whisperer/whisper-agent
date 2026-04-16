//! The single `web_search` tool. Hits the Brave Search API and returns a
//! numbered list of results as plain text.

use std::sync::Arc;

use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::warn;
use url::Url;

use whisper_agent_mcp_proto::{CallToolResult, Tool, ToolAnnotations};

#[derive(Debug, thiserror::Error)]
pub enum ToolDispatchError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
}

/// Brave caps `count` at 20 per request. We pass through whatever the model
/// asks for, clamped to this ceiling.
const MAX_COUNT: u32 = 20;

/// Server-side config: shared HTTP client, API key, and base URL.
pub struct SearchConfig {
    pub http: Client,
    pub api_key: String,
    pub api_base: String,
    pub default_count: u32,
}

pub fn descriptors() -> Vec<Tool> {
    vec![web_search_descriptor()]
}

pub async fn call(
    cfg: &Arc<SearchConfig>,
    name: &str,
    args: Value,
) -> Result<CallToolResult, ToolDispatchError> {
    match name {
        "web_search" => Ok(web_search(cfg, args).await),
        _ => Err(ToolDispatchError::UnknownTool(name.to_string())),
    }
}

// ---------- web_search ----------

fn web_search_descriptor() -> Tool {
    Tool {
        name: "web_search".into(),
        description: "Search the web via Brave Search and return a ranked list of \
                      results (title, url, snippet, optional age). Returns SERP-style \
                      summaries, not page bodies — pair with `web_fetch` to read a \
                      specific result. Use `allowed_domains` to restrict results to a \
                      set of domains (subdomain matches included), or `blocked_domains` \
                      to filter known-noisy ones out. `count` defaults to 10 and is \
                      capped at 20 by the upstream API."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query."
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of results to return (1-20). Defaults to 10."
                },
                "allowed_domains": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "If set, only results whose host matches one of these \
                                    domains (or a subdomain) are returned."
                },
                "blocked_domains": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Drop results whose host matches one of these domains \
                                    (or a subdomain). Blocked wins over allowed."
                }
            },
            "required": ["query"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Search the web".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            // Same query can return different results over time as the index updates;
            // not strictly idempotent.
            idempotent_hint: Some(false),
            open_world_hint: Some(true),
        }),
    }
}

#[derive(Deserialize)]
struct WebSearchArgs {
    query: String,
    #[serde(default)]
    count: Option<u32>,
    #[serde(default)]
    allowed_domains: Option<Vec<String>>,
    #[serde(default)]
    blocked_domains: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct BraveResponse {
    #[serde(default)]
    web: Option<BraveWebSection>,
}

#[derive(Deserialize)]
struct BraveWebSection {
    #[serde(default)]
    results: Vec<BraveResult>,
}

#[derive(Deserialize)]
struct BraveResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    age: Option<String>,
}

async fn web_search(cfg: &Arc<SearchConfig>, args: Value) -> CallToolResult {
    let parsed: WebSearchArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };

    let query = parsed.query.trim();
    if query.is_empty() {
        return CallToolResult::error_text("query must be non-empty");
    }

    let requested = parsed.count.unwrap_or(cfg.default_count).max(1);
    let upstream_count = requested.min(MAX_COUNT);
    if requested > MAX_COUNT {
        warn!(requested, capped = MAX_COUNT, "count clamped to upstream limit");
    }

    // Domain filters happen after we get results back, so over-request a bit
    // when filters are in play to keep the post-filter list close to `count`.
    let upstream_count = if parsed.allowed_domains.is_some() || parsed.blocked_domains.is_some() {
        MAX_COUNT
    } else {
        upstream_count
    };

    let url = format!("{}/web/search", cfg.api_base);
    let resp = match cfg
        .http
        .get(&url)
        .header("X-Subscription-Token", &cfg.api_key)
        .header("Accept", "application/json")
        .query(&[("q", query), ("count", &upstream_count.to_string())])
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return CallToolResult::error_text(format!("brave request failed: {e}")),
    };

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return CallToolResult::error_text(format!(
            "brave search returned {}{}: {}",
            status.as_u16(),
            extra_hint(status),
            truncate(&body, 1000),
        ));
    }

    let parsed_resp: BraveResponse = match resp.json().await {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("brave response parse: {e}")),
    };

    let results = parsed_resp.web.map(|w| w.results).unwrap_or_default();
    let allowed = parsed
        .allowed_domains
        .as_deref()
        .map(|d| normalize_domains(d));
    let blocked = parsed
        .blocked_domains
        .as_deref()
        .map(|d| normalize_domains(d));

    let mut kept: Vec<&BraveResult> = Vec::with_capacity(results.len());
    let mut filtered = 0usize;
    for r in &results {
        if !host_matches_filters(&r.url, allowed.as_deref(), blocked.as_deref()) {
            filtered += 1;
            continue;
        }
        kept.push(r);
        if kept.len() as u32 >= requested {
            break;
        }
    }

    if kept.is_empty() {
        let mut msg = format!("no results for query: {query}");
        if filtered > 0 {
            msg.push_str(&format!(" ({filtered} filtered by domain rules)"));
        }
        return CallToolResult::text(msg);
    }

    let mut out = format!("Top {} results for: {}\n\n", kept.len(), query);
    for (i, r) in kept.iter().enumerate() {
        out.push_str(&format!("{}. {}\n", i + 1, plain_text(&r.title)));
        out.push_str(&format!("   {}\n", r.url));
        if let Some(age) = &r.age {
            if !age.is_empty() {
                out.push_str(&format!("   {}\n", age));
            }
        }
        let snippet = plain_text(&r.description);
        if !snippet.is_empty() {
            out.push_str(&format!("   {}\n", snippet));
        }
        out.push('\n');
    }
    if filtered > 0 {
        out.push_str(&format!(
            "[{filtered} additional result(s) filtered by domain rules]\n"
        ));
    }
    CallToolResult::text(out)
}

// ---------- helpers ----------

/// HTTP-status annotations that map to common Brave errors. Returns "" for
/// uninteresting statuses so the caller can format inline.
fn extra_hint(status: StatusCode) -> &'static str {
    match status.as_u16() {
        401 => " (auth: BRAVE_API_KEY rejected)",
        403 => " (auth: subscription does not permit this endpoint)",
        // Brave returns 422 for both bad params AND invalid api keys
        // (SUBSCRIPTION_TOKEN_INVALID); the body's `error.code` disambiguates.
        422 => " (client error — see body; common causes: bad query params, BRAVE_API_KEY rejected)",
        429 => " (rate limit — Brave free tier allows ~1 query/sec, 2k/month)",
        _ => "",
    }
}

/// Lower-case + strip a leading dot so users can pass "example.com" or
/// ".example.com" interchangeably.
fn normalize_domains(input: &[String]) -> Vec<String> {
    input
        .iter()
        .map(|d| d.trim().trim_start_matches('.').to_ascii_lowercase())
        .filter(|d| !d.is_empty())
        .collect()
}

/// Apply allow/block lists to a result URL's host. Blocked wins over allowed
/// (a blocked subdomain inside an allowed domain is still blocked).
fn host_matches_filters(
    url: &str,
    allowed: Option<&[String]>,
    blocked: Option<&[String]>,
) -> bool {
    let Some(host) = Url::parse(url).ok().and_then(|u| u.host_str().map(str::to_owned)) else {
        // Result without a parsable URL — drop only if an allowlist is set;
        // keep it through (and let the caller see) when no filter applies.
        return allowed.is_none();
    };
    let host = host.to_ascii_lowercase();
    if let Some(blocked) = blocked
        && blocked.iter().any(|d| host_matches_domain(&host, d))
    {
        return false;
    }
    if let Some(allowed) = allowed {
        return allowed.iter().any(|d| host_matches_domain(&host, d));
    }
    true
}

fn host_matches_domain(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

/// Brave snippets are HTML fragments — usually just `<strong>` highlights and
/// the occasional encoded entity. Strip tags and decode the common entities.
/// Not a full HTML parser; if Brave changes its markup we'll see noisy output
/// but no panics.
fn plain_text(html: &str) -> String {
    let stripped = strip_tags(html);
    decode_entities(&stripped)
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'&' {
            // Look ahead for `;` within a small window (entities are short).
            if let Some(end) = bytes[i + 1..i + 1 + 10.min(bytes.len() - i - 1)]
                .iter()
                .position(|&b| b == b';')
            {
                let raw = &s[i + 1..i + 1 + end];
                if let Some(decoded) = decode_entity(raw) {
                    out.push(decoded);
                    i += end + 2; // skip "&...;"
                    continue;
                }
            }
        }
        // Default: copy this UTF-8 codepoint through.
        let ch_len = utf8_char_len(bytes[i]);
        out.push_str(&s[i..i + ch_len]);
        i += ch_len;
    }
    out
}

fn decode_entity(name: &str) -> Option<char> {
    match name {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" | "#39" | "#039" => Some('\''),
        "nbsp" => Some(' '),
        n if n.starts_with("#x") || n.starts_with("#X") => {
            u32::from_str_radix(&n[2..], 16).ok().and_then(char::from_u32)
        }
        n if n.starts_with('#') => n[1..].parse::<u32>().ok().and_then(char::from_u32),
        _ => None,
    }
}

fn utf8_char_len(first_byte: u8) -> usize {
    match first_byte {
        b if b < 0x80 => 1,
        b if b < 0xc0 => 1, // continuation byte — shouldn't be the first; treat as one
        b if b < 0xe0 => 2,
        b if b < 0xf0 => 3,
        _ => 4,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut cut = max;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &s[..cut])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_tags_removes_simple_markup() {
        assert_eq!(strip_tags("foo <strong>bar</strong> baz"), "foo bar baz");
        assert_eq!(strip_tags("<a href=\"x\">link</a>"), "link");
        assert_eq!(strip_tags("no tags here"), "no tags here");
    }

    #[test]
    fn decode_entities_handles_common_cases() {
        assert_eq!(decode_entities("a &amp; b"), "a & b");
        assert_eq!(decode_entities("&lt;tag&gt;"), "<tag>");
        assert_eq!(decode_entities("&quot;hi&quot;"), "\"hi\"");
        assert_eq!(decode_entities("don&#39;t"), "don't");
        assert_eq!(decode_entities("don&apos;t"), "don't");
        assert_eq!(decode_entities("nbsp&nbsp;here"), "nbsp here");
    }

    #[test]
    fn decode_entities_passes_unknown_entities_through() {
        // Unknown entity name — left as-is rather than swallowing characters.
        assert_eq!(decode_entities("&unknown;"), "&unknown;");
    }

    #[test]
    fn decode_entities_handles_numeric_refs() {
        assert_eq!(decode_entities("&#65;"), "A");
        assert_eq!(decode_entities("&#x41;"), "A");
    }

    #[test]
    fn decode_entities_preserves_multibyte_codepoints() {
        // Make sure the byte-walk doesn't split a multi-byte UTF-8 sequence.
        assert_eq!(decode_entities("héllo"), "héllo");
        assert_eq!(decode_entities("&amp; héllo"), "& héllo");
    }

    #[test]
    fn host_matches_domain_matches_exact_and_subdomains() {
        assert!(host_matches_domain("docs.rs", "docs.rs"));
        assert!(host_matches_domain("foo.docs.rs", "docs.rs"));
        assert!(!host_matches_domain("notdocs.rs", "docs.rs"));
        assert!(!host_matches_domain("docs.rs.evil.example", "docs.rs"));
    }

    #[test]
    fn host_matches_filters_blocked_wins() {
        let allow = Some(vec!["example.com".into()]);
        let block = Some(vec!["spam.example.com".into()]);
        assert!(host_matches_filters(
            "https://docs.example.com/x",
            allow.as_deref(),
            block.as_deref(),
        ));
        assert!(!host_matches_filters(
            "https://spam.example.com/y",
            allow.as_deref(),
            block.as_deref(),
        ));
        assert!(!host_matches_filters(
            "https://other.org/z",
            allow.as_deref(),
            block.as_deref(),
        ));
    }

    #[test]
    fn host_matches_filters_no_filters_keeps_everything() {
        assert!(host_matches_filters("https://example.com/", None, None));
    }

    #[test]
    fn normalize_domains_lowercases_and_strips_leading_dot() {
        let out = normalize_domains(&[
            "Example.COM".into(),
            ".docs.rs".into(),
            "  ".into(),
            "rust-lang.org".into(),
        ]);
        assert_eq!(out, vec!["example.com", "docs.rs", "rust-lang.org"]);
    }

    #[test]
    fn truncate_appends_ellipsis_on_byte_boundary() {
        let s = "héllo world";
        let out = truncate(s, 5);
        assert!(out.ends_with('…'));
        // No partial UTF-8 — `é` is 2 bytes, so truncating to 5 walks back to 4.
        assert!(out.len() <= 5 + '…'.len_utf8());
    }
}
