//! The single `web_fetch` tool. Fetches one URL over http/https with SSRF,
//! size, timeout, and redirect caps; optionally extracts readable text from
//! HTML via the `html2text` crate.

use std::net::IpAddr;
use std::sync::Arc;

use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::lookup_host;
use tracing::warn;
use url::Url;

use whisper_agent_mcp_proto::{CallToolResult, Tool, ToolAnnotations};

#[derive(Debug, thiserror::Error)]
pub enum ToolDispatchError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
}

/// Server-side limits and shared HTTP client. Constructed once in main and
/// passed through the axum state to each tool call.
pub struct FetchConfig {
    pub http: Client,
    pub max_response_bytes: usize,
    pub max_redirects: usize,
    pub allow_private_addresses: bool,
}

pub fn descriptors() -> Vec<Tool> {
    vec![web_fetch_descriptor()]
}

pub async fn call(
    cfg: &Arc<FetchConfig>,
    name: &str,
    args: Value,
) -> Result<CallToolResult, ToolDispatchError> {
    match name {
        "web_fetch" => Ok(web_fetch(cfg, args).await),
        _ => Err(ToolDispatchError::UnknownTool(name.to_string())),
    }
}

// ---------- web_fetch ----------

fn web_fetch_descriptor() -> Tool {
    Tool {
        name: "web_fetch".into(),
        description: "Fetch one http/https URL and return the response body. By default, HTML \
                      responses are run through a reader-mode extractor that strips chrome and \
                      returns plain text — much cheaper to read than raw HTML. Pass `raw: true` \
                      for the unmodified body (use this for JSON APIs, text/plain manifests, \
                      etc.). Server caps response size, request time, and redirect count; URLs \
                      resolving to private / loopback / link-local addresses are blocked unless \
                      the daemon was started with --allow-private-addresses."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s) URL to fetch."
                },
                "raw": {
                    "type": "boolean",
                    "description": "If true, return the response body unchanged. If false \
                                    (default), HTML bodies are converted to readable text."
                }
            },
            "required": ["url"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Fetch a URL".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            // Same URL can return different content over time (news, dynamic pages); not idempotent.
            idempotent_hint: Some(false),
            open_world_hint: Some(true),
        }),
    }
}

#[derive(Deserialize)]
struct WebFetchArgs {
    url: String,
    #[serde(default)]
    raw: bool,
}

async fn web_fetch(cfg: &Arc<FetchConfig>, args: Value) -> CallToolResult {
    let parsed: WebFetchArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };

    let mut current_url = match Url::parse(&parsed.url) {
        Ok(u) => u,
        Err(e) => return CallToolResult::error_text(format!("invalid url: {e}")),
    };

    let mut hops_followed: usize = 0;
    loop {
        if let Err(e) = validate_url(&current_url, cfg.allow_private_addresses).await {
            return CallToolResult::error_text(format!("web_fetch blocked: {e}"));
        }

        let resp = match cfg.http.get(current_url.clone()).send().await {
            Ok(r) => r,
            Err(e) => return CallToolResult::error_text(format!("web_fetch request: {e}")),
        };
        let status = resp.status();

        if status.is_redirection() {
            let Some(loc) = resp.headers().get(reqwest::header::LOCATION) else {
                return CallToolResult::error_text(format!(
                    "web_fetch: {} redirect with no Location header",
                    status.as_u16()
                ));
            };
            let loc_str = match loc.to_str() {
                Ok(s) => s,
                Err(e) => {
                    return CallToolResult::error_text(format!(
                        "web_fetch: non-ASCII Location header: {e}"
                    ));
                }
            };
            let next = match current_url.join(loc_str) {
                Ok(u) => u,
                Err(e) => {
                    return CallToolResult::error_text(format!("web_fetch: bad redirect: {e}"));
                }
            };
            if hops_followed >= cfg.max_redirects {
                return CallToolResult::error_text(format!(
                    "web_fetch: too many redirects ({}); last hop was {} -> {}",
                    cfg.max_redirects, current_url, next
                ));
            }
            hops_followed += 1;
            current_url = next;
            continue;
        }

        // Pre-check Content-Length to bail early on declared-oversize responses.
        if let Some(declared) = resp.content_length()
            && declared as usize > cfg.max_response_bytes
        {
            return CallToolResult::error_text(format!(
                "web_fetch: declared Content-Length {} exceeds cap {}",
                declared, cfg.max_response_bytes
            ));
        }

        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => return CallToolResult::error_text(format!("web_fetch read body: {e}")),
        };
        let (body_str, truncated) = bytes_to_string_capped(&bytes, cfg.max_response_bytes);

        let rendered = if parsed.raw || !is_html_content_type(&content_type) {
            body_str
        } else {
            clean_html(&body_str, current_url.as_str())
        };

        let mut out = format!(
            "HTTP {}{}\nURL: {}\nContent-Type: {}\n\n",
            status.as_u16(),
            status
                .canonical_reason()
                .map(|r| format!(" {r}"))
                .unwrap_or_default(),
            current_url,
            if content_type.is_empty() { "(none)" } else { &content_type }
        );
        out.push_str(&rendered);
        if truncated {
            out.push_str(&format!(
                "\n\n[response truncated to {} bytes]\n",
                cfg.max_response_bytes
            ));
        }
        if hops_followed > 0 {
            out.push_str(&format!("\n[followed {} redirect(s)]\n", hops_followed));
        }
        return if status.is_success() {
            CallToolResult::text(out)
        } else {
            // Non-2xx, non-redirect (e.g. 404, 500): return body as error so the
            // caller sees `is_error: true`. Body is still useful context.
            CallToolResult::error_text(out)
        };
    }
}

// ---------- helpers ----------

/// Return (string, was_truncated). Truncation walks forward to a UTF-8
/// boundary so we never emit a half-encoded codepoint.
fn bytes_to_string_capped(bytes: &[u8], max: usize) -> (String, bool) {
    if bytes.len() <= max {
        return (String::from_utf8_lossy(bytes).into_owned(), false);
    }
    let mut end = max;
    // Walk forward to a char boundary in the lossy decoding sense — easier to
    // do by trimming on a clean prefix of valid UTF-8.
    while end > 0 && (bytes[end] & 0b1100_0000) == 0b1000_0000 {
        end -= 1;
    }
    (String::from_utf8_lossy(&bytes[..end]).into_owned(), true)
}

fn is_html_content_type(ct: &str) -> bool {
    let main = ct.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    matches!(
        main.as_str(),
        "text/html" | "application/xhtml+xml"
    )
}

fn clean_html(body: &str, _base_url: &str) -> String {
    // 100-column wrap is wide enough to keep code blocks readable but narrow enough
    // that long paragraphs don't hide content from the model in a single mega-line.
    const WRAP_WIDTH: usize = 100;
    html2text::config::plain()
        .string_from_read(body.as_bytes(), WRAP_WIDTH)
        .unwrap_or_else(|e| format!("[html2text error: {e}]\n{body}"))
}

/// SSRF guard: parse the URL, reject non-http(s) schemes, resolve the host,
/// and reject if any resolved address is private / loopback / link-local /
/// multicast / unspecified, unless the daemon was started with
/// --allow-private-addresses.
async fn validate_url(url: &Url, allow_private: bool) -> Result<(), String> {
    match url.scheme() {
        "http" | "https" => {}
        other => return Err(format!("scheme `{other}` not allowed (only http/https)")),
    }
    let host = url
        .host_str()
        .ok_or_else(|| "url has no host".to_string())?;
    // Default ports if missing — needed for lookup_host("host:port") form.
    let port = url
        .port_or_known_default()
        .ok_or_else(|| "no port and no default for scheme".to_string())?;

    // If the host is already a literal IP, lookup_host still works but skip
    // resolution overhead by checking directly.
    if let Ok(ip) = host.parse::<IpAddr>() {
        return check_address(ip, allow_private);
    }

    let hp = format!("{host}:{port}");
    let addrs = match lookup_host(&hp).await {
        Ok(it) => it,
        Err(e) => return Err(format!("dns: {e}")),
    };
    let mut any = false;
    for sa in addrs {
        any = true;
        check_address(sa.ip(), allow_private)?;
    }
    if !any {
        return Err(format!("dns returned no addresses for {host}"));
    }
    Ok(())
}

fn check_address(ip: IpAddr, allow_private: bool) -> Result<(), String> {
    if allow_private {
        return Ok(());
    }
    let bad = match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_multicast()
                || v4.is_unspecified()
                // Carrier-grade NAT (100.64.0.0/10) — reachable on some clouds.
                || (v4.octets()[0] == 100 && (64..=127).contains(&v4.octets()[1]))
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_multicast()
                || v6.is_unspecified()
                // ULA (fc00::/7)
                || (v6.segments()[0] & 0xfe00 == 0xfc00)
                // Link-local (fe80::/10)
                || (v6.segments()[0] & 0xffc0 == 0xfe80)
                // IPv4-mapped — re-check the embedded v4 (avoid bypass via
                // ::ffff:192.168.x.x).
                || v6.to_ipv4_mapped().is_some_and(|v4| {
                    v4.is_private() || v4.is_loopback() || v4.is_link_local()
                        || v4.is_broadcast() || v4.is_documentation()
                        || v4.is_multicast() || v4.is_unspecified()
                })
        }
    };
    if bad {
        warn!(%ip, "blocked private/loopback/link-local address");
        return Err(format!(
            "address {ip} is private/loopback/link-local; pass --allow-private-addresses to enable"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_address_blocks_loopback_v4() {
        assert!(check_address("127.0.0.1".parse().unwrap(), false).is_err());
        assert!(check_address("127.0.0.1".parse().unwrap(), true).is_ok());
    }

    #[test]
    fn check_address_blocks_private_v4_ranges() {
        for ip in [
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254", // AWS/GCP metadata service
            "100.64.0.1",       // CGNAT
            "0.0.0.0",
        ] {
            assert!(
                check_address(ip.parse().unwrap(), false).is_err(),
                "expected {ip} to be blocked"
            );
        }
    }

    #[test]
    fn check_address_allows_public_v4() {
        assert!(check_address("1.1.1.1".parse().unwrap(), false).is_ok());
        assert!(check_address("8.8.8.8".parse().unwrap(), false).is_ok());
        assert!(check_address("172.32.0.1".parse().unwrap(), false).is_ok()); // outside 172.16/12
    }

    #[test]
    fn check_address_blocks_v6_loopback_and_link_local() {
        assert!(check_address("::1".parse().unwrap(), false).is_err());
        assert!(check_address("fe80::1".parse().unwrap(), false).is_err());
        assert!(check_address("fc00::1".parse().unwrap(), false).is_err());
    }

    #[test]
    fn check_address_blocks_ipv4_mapped_loopback() {
        // ::ffff:127.0.0.1 — must not bypass via v4-mapped form.
        assert!(check_address("::ffff:127.0.0.1".parse().unwrap(), false).is_err());
        assert!(check_address("::ffff:192.168.1.1".parse().unwrap(), false).is_err());
    }

    #[tokio::test]
    async fn validate_url_rejects_non_http_schemes() {
        for u in ["file:///etc/passwd", "ftp://example.com", "ssh://example.com"] {
            let url = Url::parse(u).unwrap();
            assert!(validate_url(&url, false).await.is_err(), "{u}");
        }
    }

    #[tokio::test]
    async fn validate_url_rejects_literal_loopback() {
        let url = Url::parse("http://127.0.0.1:9999/").unwrap();
        assert!(validate_url(&url, false).await.is_err());
        assert!(validate_url(&url, true).await.is_ok());
    }

    #[test]
    fn is_html_recognizes_common_ct_strings() {
        assert!(is_html_content_type("text/html"));
        assert!(is_html_content_type("text/html; charset=utf-8"));
        assert!(is_html_content_type("Text/HTML; CHARSET=UTF-8"));
        assert!(is_html_content_type("application/xhtml+xml"));
        assert!(!is_html_content_type("application/json"));
        assert!(!is_html_content_type("text/plain"));
        assert!(!is_html_content_type(""));
    }

    #[test]
    fn clean_html_extracts_text_and_drops_tags() {
        let html = "<html><body><h1>Hello</h1><p>This is <b>bold</b>.</p></body></html>";
        let txt = clean_html(html, "https://example.com/");
        assert!(txt.contains("Hello"));
        assert!(txt.contains("This is"));
        assert!(txt.contains("bold"));
        assert!(!txt.contains("<h1>"));
    }

    #[test]
    fn bytes_to_string_capped_returns_full_when_under_cap() {
        let (s, t) = bytes_to_string_capped(b"hello world", 1000);
        assert_eq!(s, "hello world");
        assert!(!t);
    }

    #[test]
    fn bytes_to_string_capped_truncates_on_utf8_boundary() {
        // "éééé" = 8 bytes (each é is 2 bytes). Cap at 5 should yield "éé" (4 bytes).
        let s_in = "éééé";
        let (s, t) = bytes_to_string_capped(s_in.as_bytes(), 5);
        assert!(t);
        assert_eq!(s, "éé"); // walked back to char boundary
    }
}
