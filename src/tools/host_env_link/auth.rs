//! Daemon-bearer admission for the `/v1/host_env_link` endpoint.
//!
//! Distinct from [`crate::server::auth`]: client / admin tokens grant
//! chat / settings access; daemon tokens grant *only* the right to
//! dial in as a host-env provider. Two token surfaces, two
//! constant-time comparison lists, no cross-grants.
//!
//! Loopback peers are *not* exempt — daemons should always present a
//! token, even when running on the same host. The point of the token
//! is to bind the connection to a specific catalog entry name; without
//! it we can't tell two daemons apart.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq;

use crate::pod::config::AuthDaemon;

/// Snapshot of the configured daemon admission tokens. Built once at
/// server startup and shared (Arc) across the WS upgrade middleware.
#[derive(Debug, Default)]
pub struct DaemonAuthState {
    entries: Vec<AuthDaemon>,
}

impl DaemonAuthState {
    pub fn new(entries: Vec<AuthDaemon>) -> Self {
        Self { entries }
    }

    /// Number of admitted daemon names. Used at startup logging.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the daemon name a presented token resolves to, or `None`
    /// if no token matches. Constant-time over the full list — both the
    /// "did any token match" and "which one matched" decisions are
    /// resolved in one timing-uniform pass.
    pub fn resolve(&self, presented: &[u8]) -> Option<&str> {
        // Single-pass scan: track the first match in a u64 index slot
        // selected by the constant-time mask. We always look at every
        // entry; the resulting `name` is the first match, or `None`.
        let mut matched: u8 = 0;
        let mut idx: u64 = u64::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            let hit = presented.ct_eq(entry.token.as_bytes()).unwrap_u8();
            // Only set idx the *first* time we see a match. We can't
            // branch on `matched` without leaking timing, so build an
            // unconditional mask: write the new index only when this
            // entry matches AND no earlier entry did.
            let take = hit & matched.wrapping_sub(1);
            // `take == 1` iff this is the first match seen so far.
            // Build an all-ones (take=1) / all-zeros (take=0) mask so
            // the conditional update is data-flow only.
            let mask = 0u64.wrapping_sub(take as u64);
            idx = (idx & !mask) | ((i as u64) & mask);
            matched |= hit;
        }
        if matched == 1 {
            self.entries.get(idx as usize).map(|e| e.name.as_str())
        } else {
            None
        }
    }
}

/// Resolved daemon identity attached to the request after
/// [`require_daemon_auth`] approves the token. The WS handler reads
/// this back out via [`axum::extract::Extension`] to know which
/// catalog name the connection is for.
#[derive(Clone, Debug)]
pub struct AdmittedDaemon {
    pub name: String,
}

/// Tower middleware: validate the bearer token against [`DaemonAuthState`]
/// and inject [`AdmittedDaemon`] into request extensions on success.
///
/// Errors:
/// - 401 when no bearer is presented or the token doesn't match.
/// - 503 when the daemon admission table is empty (no daemons can
///   ever connect; explicit error beats a silent never-matches).
pub async fn require_daemon_auth(
    State(auth): State<Arc<DaemonAuthState>>,
    headers: HeaderMap,
    mut req: axum::extract::Request,
    next: Next,
) -> Response {
    if auth.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "no [[auth.daemons]] configured; /v1/host_env_link is closed",
        )
            .into_response();
    }
    let Some(presented) = bearer_token(&headers) else {
        return (
            StatusCode::UNAUTHORIZED,
            "missing Authorization: Bearer <token>",
        )
            .into_response();
    };
    let Some(name) = auth.resolve(presented.as_bytes()) else {
        return (StatusCode::UNAUTHORIZED, "invalid daemon token").into_response();
    };
    req.extensions_mut().insert(AdmittedDaemon {
        name: name.to_string(),
    });
    next.run(req).await
}

/// Extract the bearer token from a request's headers.
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(name: &str, token: &str) -> AuthDaemon {
        AuthDaemon {
            name: name.into(),
            token: token.into(),
        }
    }

    #[test]
    fn resolve_returns_name_on_match() {
        let s = DaemonAuthState::new(vec![d("alpha", "tok-a"), d("beta", "tok-b")]);
        assert_eq!(s.resolve(b"tok-a"), Some("alpha"));
        assert_eq!(s.resolve(b"tok-b"), Some("beta"));
    }

    #[test]
    fn resolve_returns_none_on_miss() {
        let s = DaemonAuthState::new(vec![d("alpha", "tok-a")]);
        assert_eq!(s.resolve(b"nope"), None);
        assert_eq!(s.resolve(b""), None);
    }

    #[test]
    fn empty_state_resolves_nothing() {
        let s = DaemonAuthState::default();
        assert!(s.is_empty());
        assert_eq!(s.resolve(b"anything"), None);
    }

    #[test]
    fn resolve_returns_first_match_for_duplicate_tokens() {
        // Duplicate tokens shouldn't happen in practice (config
        // validation rejects duplicate daemon names, but tokens
        // themselves aren't enforced unique). Behavior should still
        // be deterministic — first match wins.
        let s = DaemonAuthState::new(vec![d("first", "shared"), d("second", "shared")]);
        assert_eq!(s.resolve(b"shared"), Some("first"));
    }
}
