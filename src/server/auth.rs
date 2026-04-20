//! Client-token auth gate for the HTTP/WS surface.
//!
//! Loopback peers always pass through unauthenticated. Non-loopback peers
//! must present a configured token via either `Authorization: Bearer <tok>`
//! (native clients) or a `wa_session=<tok>` cookie (browser, after POSTing
//! to `/auth/login`). Tokens compare with `subtle::ConstantTimeEq`; the
//! comparison loop runs through the full configured list rather than
//! short-circuiting so timing reveals nothing about which entry matched.
//!
//! When the configured token list is empty, every non-loopback request is
//! rejected — that's the safe default for a server that's been bound to a
//! routable address without any auth set up.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    Json,
    body::Body,
    extract::{ConnectInfo, State},
    http::{HeaderMap, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use subtle::ConstantTimeEq;

/// Snapshot of the configured client tokens. Built once at server startup
/// and shared (Arc) across the gate middleware and the `/auth/login`
/// handler. Names live in the TOML for human bookkeeping; runtime only
/// needs the token bytes.
#[derive(Debug, Default)]
pub struct AuthState {
    pub tokens: Vec<String>,
}

impl AuthState {
    pub fn new(tokens: Vec<String>) -> Self {
        Self { tokens }
    }

    /// Constant-time check: presented bytes match any configured token.
    /// Iterates the full list (no early exit) so timing doesn't leak
    /// which slot matched — or that any slot did.
    fn matches(&self, presented: &[u8]) -> bool {
        let mut matched = 0u8;
        for token in &self.tokens {
            matched |= presented.ct_eq(token.as_bytes()).unwrap_u8();
        }
        matched == 1
    }
}

/// Tower middleware: gate non-loopback requests behind a configured token.
/// Loopback peers (127.0.0.0/8, ::1) bypass the check entirely. Apply
/// via `route_layer` to the routes that need it.
pub async fn require_auth(
    State(auth): State<Arc<AuthState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    if addr.ip().is_loopback() {
        return next.run(req).await;
    }
    if auth.tokens.is_empty() {
        return (
            StatusCode::UNAUTHORIZED,
            "client auth not configured; non-loopback access denied",
        )
            .into_response();
    }
    let Some(presented) = bearer_or_cookie(&headers) else {
        return (
            StatusCode::UNAUTHORIZED,
            "missing bearer token or session cookie",
        )
            .into_response();
    };
    if !auth.matches(presented.as_bytes()) {
        return (StatusCode::UNAUTHORIZED, "invalid token").into_response();
    }
    next.run(req).await
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub token: String,
}

/// `POST /auth/login` — exchange a token for a session cookie. Browser
/// `WebSocket` cannot set custom headers, so this is the only path by
/// which the webui acquires authentication. Native clients can use the
/// bearer header directly and skip this endpoint.
///
/// The cookie value is the same opaque token; we don't issue a separate
/// session ID. HttpOnly+Secure+SameSite=Lax keeps it out of JS reach
/// and off cross-site requests; `Secure` works on `localhost` because
/// browsers treat it as a secure context even over plain HTTP.
pub async fn handle_login(
    State(auth): State<Arc<AuthState>>,
    Json(body): Json<LoginRequest>,
) -> Response {
    if auth.tokens.is_empty() || !auth.matches(body.token.as_bytes()) {
        return (StatusCode::UNAUTHORIZED, "invalid token").into_response();
    }
    let cookie = format!(
        "wa_session={}; Path=/; HttpOnly; Secure; SameSite=Lax",
        body.token
    );
    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .header(header::SET_COOKIE, cookie)
        .body(Body::empty())
        .expect("static cookie header is well-formed")
}

/// `GET /auth/check` — probe used by the webui's HTML bootstrap before
/// fetching the wasm bundle. The middleware does the actual decision;
/// this handler only fires when the request is already authenticated,
/// in which case we return 204.
pub async fn handle_check() -> StatusCode {
    StatusCode::NO_CONTENT
}

fn bearer_or_cookie(headers: &HeaderMap) -> Option<String> {
    if let Some(bearer) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
    {
        return Some(bearer.to_string());
    }
    extract_cookie(headers, "wa_session").map(|s| s.to_string())
}

fn extract_cookie<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let raw = headers.get(header::COOKIE)?.to_str().ok()?;
    for kv in raw.split(';') {
        let kv = kv.trim();
        if let Some((k, v)) = kv.split_once('=')
            && k == name
        {
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers_with(name: &'static str, value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(name, HeaderValue::from_str(value).unwrap());
        h
    }

    #[test]
    fn extract_cookie_picks_named_value() {
        let h = headers_with("cookie", "foo=1; wa_session=abc; bar=2");
        assert_eq!(extract_cookie(&h, "wa_session"), Some("abc"));
    }

    #[test]
    fn extract_cookie_returns_none_when_absent() {
        let h = headers_with("cookie", "foo=1; bar=2");
        assert_eq!(extract_cookie(&h, "wa_session"), None);
    }

    #[test]
    fn bearer_takes_precedence_over_cookie() {
        let mut h = headers_with("cookie", "wa_session=cookie_tok");
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer header_tok"),
        );
        assert_eq!(bearer_or_cookie(&h).as_deref(), Some("header_tok"));
    }

    #[test]
    fn bearer_falls_back_to_cookie() {
        let h = headers_with("cookie", "wa_session=cookie_tok");
        assert_eq!(bearer_or_cookie(&h).as_deref(), Some("cookie_tok"));
    }

    #[test]
    fn matches_finds_configured_token() {
        let auth = AuthState::new(vec!["aaa".into(), "bbb".into()]);
        assert!(auth.matches(b"aaa"));
        assert!(auth.matches(b"bbb"));
        assert!(!auth.matches(b"ccc"));
        assert!(!auth.matches(b""));
    }
}
