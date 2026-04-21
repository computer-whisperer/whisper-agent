//! OAuth 2.1 + PKCE + DCR primitives for the shared-MCP-host catalog.
//!
//! Third-party MCP servers advertise their authorization requirements
//! via 401 + `WWW-Authenticate: Bearer resource_metadata="..."` (or a
//! well-known file at the base of the MCP URL). This module walks the
//! discovery chain (resource metadata → authorization-server metadata),
//! runs RFC 7591 dynamic client registration when the AS supports it,
//! and performs the authorization-code + PKCE flow against the AS.
//!
//! The module is deliberately a pure-functional toolkit: it has no
//! knowledge of the catalog, the scheduler, or in-flight flow state.
//! Callers (the scheduler's OAuth handler) stitch these primitives
//! together and own the mutable state.

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use rand::RngCore;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, WWW_AUTHENTICATE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Timeout for every HTTP call in this module. Discovery + DCR + token
/// exchange are operator-initiated; a third-party AS that takes more
/// than 10s to answer is effectively broken for our use case.
const HTTP_TIMEOUT_SECS: u64 = 10;

/// Response shape of RFC 9728 Protected Resource Metadata.
/// Hosted at `<mcp_url_base>/.well-known/oauth-protected-resource`
/// (or whatever `resource_metadata=` in the 401 challenge points at).
#[derive(Debug, Clone, Deserialize)]
pub struct ResourceMetadata {
    /// The resource identifier the AS will issue tokens for. For MCP,
    /// this is the MCP server's canonical URL.
    pub resource: String,
    /// Authorization servers that can issue tokens for this resource.
    /// We pick the first; servers listing multiple ASes are a future
    /// concern.
    pub authorization_servers: Vec<String>,
    /// Optional scopes the resource advertises. When present, we use
    /// the space-joined list on the authz request unless the caller
    /// overrode it.
    #[serde(default)]
    pub scopes_supported: Vec<String>,
}

/// Response shape of RFC 8414 Authorization Server Metadata. Hosted
/// at `<issuer>/.well-known/oauth-authorization-server`.
#[derive(Debug, Clone, Deserialize)]
pub struct AuthServerMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    /// Client registration endpoint (RFC 7591). Optional — if absent,
    /// the operator must pre-register and supply a client_id out of
    /// band. Step 2a treats its absence as a hard error (we don't
    /// have a pre-registration configuration surface yet).
    #[serde(default)]
    pub registration_endpoint: Option<String>,
    #[serde(default)]
    pub code_challenge_methods_supported: Vec<String>,
    #[serde(default)]
    pub scopes_supported: Vec<String>,
}

/// Full set of endpoints + metadata needed to run an authorization-
/// code flow against a specific (MCP server, AS) pair. Produced by
/// `discover`.
#[derive(Debug, Clone)]
pub struct DiscoveredAuth {
    /// Resource indicator (RFC 8707) — echoed on every authorization
    /// request and token exchange. The MCP server URL, normalized.
    pub resource: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub registration_endpoint: Option<String>,
    pub issuer: String,
    /// Server-preferred scopes, if any. Callers pass these through on
    /// the authz URL unless they pass their own `scope` override.
    pub scopes_supported: Vec<String>,
}

/// Result of one DCR round-trip. `client_secret` is present when the
/// AS issued a confidential-client secret; public clients get `None`.
#[derive(Debug, Clone)]
pub struct RegisteredClient {
    pub client_id: String,
    pub client_secret: Option<String>,
}

/// Successful token-endpoint response. Covers both authorization_code
/// and refresh_token grants.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Seconds until `access_token` expires. Not all ASes return it;
    /// when absent, treat the token as valid until a 401 proves
    /// otherwise.
    #[serde(default)]
    pub expires_in: Option<i64>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub token_type: Option<String>,
}

/// Generated per flow. `verifier` stays server-side; `challenge` goes
/// on the authz URL; the matching `verifier` is presented again at
/// the token endpoint on code exchange.
#[derive(Debug, Clone)]
pub struct PkcePair {
    pub verifier: String,
    pub challenge: String,
}

/// Mint a fresh PKCE pair using S256 (RFC 7636 §4.2). Verifier is
/// 64 URL-safe characters (exceeds the 43-char minimum); challenge
/// is `BASE64URL(SHA256(verifier))` with no padding.
pub fn pkce_pair() -> PkcePair {
    let verifier = random_url_safe(64);
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let digest = hasher.finalize();
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    PkcePair {
        verifier,
        challenge,
    }
}

/// Mint a random URL-safe token suitable for the OAuth `state`
/// parameter (CSRF protection) and other opaque identifiers. Entropy
/// is 32 bytes, which is well over the practical collision bound we
/// care about (simultaneous in-flight flows per server).
pub fn random_state() -> String {
    random_url_safe(32)
}

/// Core RNG helper: `bytes` random bytes, base64url-encoded (no
/// padding). Callers pick the length; 32 bytes → 43 chars, 64 bytes →
/// 86 chars. Uses `rand::rng()` (thread-local CSPRNG seeded from OS
/// entropy).
fn random_url_safe(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    rand::rng().fill_bytes(&mut buf);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&buf)
}

// ---------- Discovery ----------

/// Probe the MCP server at `mcp_url` to figure out where its
/// authorization server lives. Two paths, in order:
///
///   1. Send an unauthenticated POST to the MCP URL. If the server
///      responds with 401 + `WWW-Authenticate: Bearer
///      resource_metadata="URL"`, use that URL.
///   2. Fall back to the conventional well-known location:
///      `<origin>/.well-known/oauth-protected-resource`.
///
/// Returns the resource-metadata URL to fetch next. An MCP server
/// that doesn't advertise either is unusable for OAuth; the error
/// carries the 401 body / response status so the operator can
/// diagnose a mis-deployed server.
pub async fn discover_resource_metadata_url(
    http: &reqwest::Client,
    mcp_url: &str,
) -> Result<String> {
    // Empty JSON-RPC `initialize`-shaped body — enough to trip the
    // auth middleware on a real MCP server without running a full
    // handshake. Content-type is JSON because MCP speaks JSON-RPC.
    let probe_body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
    });
    let resp = http
        .post(mcp_url)
        .header(CONTENT_TYPE, "application/json")
        .json(&probe_body)
        .send()
        .await
        .with_context(|| format!("probe POST {mcp_url}"))?;

    if resp.status() == reqwest::StatusCode::UNAUTHORIZED
        && let Some(url) = parse_www_authenticate_resource_metadata(
            resp.headers()
                .get(WWW_AUTHENTICATE)
                .and_then(|v| v.to_str().ok()),
        )
    {
        return Ok(url);
    }

    // Fall back to well-known location on the MCP URL's origin. Use
    // the URL crate to get a correct origin (scheme + host + port).
    let parsed =
        reqwest::Url::parse(mcp_url).with_context(|| format!("parse mcp_url {mcp_url}"))?;
    let origin = format!(
        "{}://{}",
        parsed.scheme(),
        parsed
            .host_str()
            .ok_or_else(|| anyhow!("mcp_url has no host: {mcp_url}"))?,
    );
    // Include port if present and non-default.
    let origin_with_port = match parsed.port() {
        Some(port) => format!("{origin}:{port}"),
        None => origin,
    };
    Ok(format!(
        "{origin_with_port}/.well-known/oauth-protected-resource"
    ))
}

/// Parse `WWW-Authenticate: Bearer resource_metadata="URL"` and
/// return the URL. Header syntax is RFC 7235 — a list of quoted
/// parameters after the scheme. We do a scoped parse here because
/// the httparse / hyperx crates are overkill for one field.
fn parse_www_authenticate_resource_metadata(header: Option<&str>) -> Option<String> {
    let h = header?.trim();
    // Strip the scheme token (`Bearer`).
    let rest = h.strip_prefix("Bearer ")?;
    // Iterate comma-separated k="v" pairs. MCP servers conforming to
    // the auth spec draft put exactly one param (resource_metadata).
    for part in rest.split(',') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("resource_metadata=") {
            let trimmed = rest.trim_matches('"');
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Fetch RFC 9728 Protected Resource Metadata from `url`. Errors on
/// non-2xx or malformed JSON.
pub async fn fetch_resource_metadata(
    http: &reqwest::Client,
    url: &str,
) -> Result<ResourceMetadata> {
    let resp = http
        .get(url)
        .header(ACCEPT, "application/json")
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("resource metadata {url} returned {status}: {body}");
    }
    resp.json::<ResourceMetadata>()
        .await
        .with_context(|| format!("parse resource metadata {url}"))
}

/// Fetch RFC 8414 Authorization Server Metadata from
/// `<issuer>/.well-known/oauth-authorization-server`. `issuer` is the
/// AS URL from `authorization_servers[0]` of the resource metadata.
pub async fn fetch_auth_server_metadata(
    http: &reqwest::Client,
    issuer: &str,
) -> Result<AuthServerMetadata> {
    let url = format!(
        "{}/.well-known/oauth-authorization-server",
        issuer.trim_end_matches('/')
    );
    let resp = http
        .get(&url)
        .header(ACCEPT, "application/json")
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("AS metadata {url} returned {status}: {body}");
    }
    let meta: AuthServerMetadata = resp
        .json()
        .await
        .with_context(|| format!("parse AS metadata {url}"))?;
    // Minimal sanity: S256 must be supported (per MCP auth spec +
    // OAuth 2.1 requirement). Some ASes omit the field entirely and
    // silently support S256; we accept an empty list but reject an
    // explicit list that excludes S256.
    if !meta.code_challenge_methods_supported.is_empty()
        && !meta
            .code_challenge_methods_supported
            .iter()
            .any(|m| m == "S256")
    {
        bail!(
            "AS {issuer} does not advertise S256 PKCE support (got {:?}); cannot proceed",
            meta.code_challenge_methods_supported
        );
    }
    Ok(meta)
}

/// One-shot discovery: probe → resource metadata → AS metadata.
/// Returns everything needed to build an authorization URL.
pub async fn discover(http: &reqwest::Client, mcp_url: &str) -> Result<DiscoveredAuth> {
    let rm_url = discover_resource_metadata_url(http, mcp_url).await?;
    let resource = fetch_resource_metadata(http, &rm_url).await?;
    let issuer = resource
        .authorization_servers
        .first()
        .ok_or_else(|| anyhow!("resource metadata at {rm_url} lists no authorization_servers"))?
        .clone();
    let as_meta = fetch_auth_server_metadata(http, &issuer).await?;
    // Union the scopes advertised by the resource + the AS; the
    // resource's scopes win as the canonical list (it's the thing
    // that'll actually enforce them).
    let scopes_supported = if !resource.scopes_supported.is_empty() {
        resource.scopes_supported.clone()
    } else {
        as_meta.scopes_supported.clone()
    };
    Ok(DiscoveredAuth {
        resource: resource.resource,
        authorization_endpoint: as_meta.authorization_endpoint,
        token_endpoint: as_meta.token_endpoint,
        registration_endpoint: as_meta.registration_endpoint,
        issuer: as_meta.issuer,
        scopes_supported,
    })
}

// ---------- Dynamic Client Registration (RFC 7591) ----------

#[derive(Debug, Clone, Serialize)]
struct DcrRequest<'a> {
    client_name: &'a str,
    redirect_uris: Vec<&'a str>,
    /// `native` — we're a server-side OAuth client in the RFC-8252
    /// sense (no UI ourselves). `web` would imply a confidential web
    /// app with a back-channel, which we're not.
    application_type: &'static str,
    /// `authorization_code` + `refresh_token` — the flows we'll use.
    grant_types: Vec<&'static str>,
    response_types: Vec<&'static str>,
    /// Optional scope string; space-separated per the spec.
    #[serde(skip_serializing_if = "Option::is_none")]
    scope: Option<&'a str>,
    /// `none` for public clients (PKCE-only). We default to `none`
    /// because we're a confidential server but PKCE is already strong;
    /// if the AS upgrades us to confidential it'll return a
    /// `client_secret` on its own accord.
    token_endpoint_auth_method: &'static str,
}

#[derive(Debug, Clone, Deserialize)]
struct DcrResponse {
    client_id: String,
    #[serde(default)]
    client_secret: Option<String>,
}

/// Perform RFC 7591 DCR against `registration_endpoint`. Returns the
/// assigned client_id (+ optional secret). `redirect_uri` is the URL
/// we'll register as the sole callback; pass the full path that
/// `/oauth/callback` resolves to on our server. `client_name` shows
/// up in the AS's consent screen.
pub async fn dynamic_client_register(
    http: &reqwest::Client,
    registration_endpoint: &str,
    redirect_uri: &str,
    client_name: &str,
    scope: Option<&str>,
) -> Result<RegisteredClient> {
    let body = DcrRequest {
        client_name,
        redirect_uris: vec![redirect_uri],
        application_type: "native",
        grant_types: vec!["authorization_code", "refresh_token"],
        response_types: vec!["code"],
        scope,
        token_endpoint_auth_method: "none",
    };
    let resp = http
        .post(registration_endpoint)
        .header(CONTENT_TYPE, "application/json")
        .header(ACCEPT, "application/json")
        .json(&body)
        .send()
        .await
        .with_context(|| format!("DCR POST {registration_endpoint}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("DCR at {registration_endpoint} returned {status}: {body}");
    }
    let reg: DcrResponse = resp
        .json()
        .await
        .with_context(|| format!("parse DCR response from {registration_endpoint}"))?;
    Ok(RegisteredClient {
        client_id: reg.client_id,
        client_secret: reg.client_secret,
    })
}

// ---------- Authorization URL builder ----------

/// Arguments for `build_authorization_url`. Grouped into a struct
/// because the positional form was eight parameters deep and shimmed
/// into call sites that had to remember the order.
pub struct AuthorizationUrlArgs<'a> {
    pub authorization_endpoint: &'a str,
    pub client_id: &'a str,
    pub redirect_uri: &'a str,
    pub state: &'a str,
    pub code_challenge: &'a str,
    pub scope: Option<&'a str>,
    /// RFC 8707 resource indicator — the MCP server URL.
    pub resource: &'a str,
}

/// Build the authorization URL the webui will open in a new window.
/// Spec-compliant parameter ordering isn't required (ASes parse by
/// name), but we alphabetize for stable URLs that are easier to
/// eyeball in logs.
pub fn build_authorization_url(args: &AuthorizationUrlArgs<'_>) -> Result<String> {
    let mut url = reqwest::Url::parse(args.authorization_endpoint).with_context(|| {
        format!(
            "parse authorization_endpoint {}",
            args.authorization_endpoint
        )
    })?;
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("client_id", args.client_id);
        q.append_pair("code_challenge", args.code_challenge);
        q.append_pair("code_challenge_method", "S256");
        q.append_pair("redirect_uri", args.redirect_uri);
        q.append_pair("resource", args.resource);
        q.append_pair("response_type", "code");
        if let Some(scope) = args.scope {
            q.append_pair("scope", scope);
        }
        q.append_pair("state", args.state);
    }
    Ok(url.into())
}

// ---------- Token endpoint exchanges ----------

/// Exchange an authorization code for tokens. `client_secret` is
/// sent only when DCR returned one (confidential client); public
/// clients pass `None` and the AS authenticates via PKCE alone.
// All arguments are required + distinct-typed strings; a struct
// adds a named type per call site without shortening anything.
#[allow(clippy::too_many_arguments)]
pub async fn exchange_code(
    http: &reqwest::Client,
    token_endpoint: &str,
    code: &str,
    redirect_uri: &str,
    client_id: &str,
    client_secret: Option<&str>,
    code_verifier: &str,
    resource: &str,
) -> Result<TokenResponse> {
    let mut form: Vec<(&str, &str)> = vec![
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("client_id", client_id),
        ("code_verifier", code_verifier),
        ("resource", resource),
    ];
    if let Some(secret) = client_secret {
        form.push(("client_secret", secret));
    }
    post_token_form(
        http,
        token_endpoint,
        &form,
        client_secret.map(|s| (client_id, s)),
    )
    .await
}

/// Refresh a (possibly expired) access token using a refresh token.
/// Kept in this module so the catalog-side refresh path in Step 2b is
/// a one-liner call into here.
pub async fn refresh_access_token(
    http: &reqwest::Client,
    token_endpoint: &str,
    refresh_token: &str,
    client_id: &str,
    client_secret: Option<&str>,
    resource: &str,
    scope: Option<&str>,
) -> Result<TokenResponse> {
    let mut form: Vec<(&str, &str)> = vec![
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", client_id),
        ("resource", resource),
    ];
    if let Some(s) = scope {
        form.push(("scope", s));
    }
    if let Some(secret) = client_secret {
        form.push(("client_secret", secret));
    }
    post_token_form(
        http,
        token_endpoint,
        &form,
        client_secret.map(|s| (client_id, s)),
    )
    .await
}

/// Shared token-endpoint POST. Uses
/// `application/x-www-form-urlencoded`. Confidential clients get an
/// additional Basic auth header (the AS accepts both body-supplied
/// and header-supplied client_secret; using Basic keeps the secret
/// off the body string in case it ends up in logs).
async fn post_token_form(
    http: &reqwest::Client,
    endpoint: &str,
    form: &[(&str, &str)],
    basic_auth: Option<(&str, &str)>,
) -> Result<TokenResponse> {
    let mut req = http
        .post(endpoint)
        .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
        .header(ACCEPT, "application/json")
        .form(form);
    if let Some((id, secret)) = basic_auth {
        let creds = format!("{id}:{secret}");
        let encoded = base64::engine::general_purpose::STANDARD.encode(creds);
        req = req.header(AUTHORIZATION, format!("Basic {encoded}"));
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("token POST {endpoint}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("token endpoint {endpoint} returned {status}: {body}");
    }
    resp.json::<TokenResponse>()
        .await
        .with_context(|| format!("parse token response from {endpoint}"))
}

/// Shared `reqwest::Client` for OAuth calls. Default client plus a
/// bounded timeout so a misbehaving AS can't hang the scheduler.
/// Kept as a free function rather than a state field on Scheduler so
/// test paths can build their own.
pub fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(HTTP_TIMEOUT_SECS))
        .build()
        .context("build oauth http client")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_pair_shape_is_correct() {
        let p = pkce_pair();
        // Verifier: 64 random bytes base64url-no-pad → 86 chars.
        assert_eq!(p.verifier.len(), 86);
        // Challenge: 32-byte SHA256 → 43 chars.
        assert_eq!(p.challenge.len(), 43);
        // Verifying the challenge is actually BASE64URL(SHA256(verifier)).
        let mut h = Sha256::new();
        h.update(p.verifier.as_bytes());
        let expected = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(h.finalize());
        assert_eq!(p.challenge, expected);
    }

    #[test]
    fn pkce_pairs_are_unique() {
        // Not a security test — just a smoke test that we seeded the
        // RNG rather than returning a constant.
        let a = pkce_pair();
        let b = pkce_pair();
        assert_ne!(a.verifier, b.verifier);
        assert_ne!(a.challenge, b.challenge);
    }

    #[test]
    fn random_state_has_entropy() {
        let a = random_state();
        let b = random_state();
        assert_ne!(a, b);
        assert_eq!(a.len(), 43);
    }

    #[test]
    fn parse_www_authenticate_extracts_resource_metadata() {
        let url = parse_www_authenticate_resource_metadata(Some(
            r#"Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource""#,
        ))
        .unwrap();
        assert_eq!(
            url,
            "https://mcp.example.com/.well-known/oauth-protected-resource"
        );
    }

    #[test]
    fn parse_www_authenticate_handles_multi_params() {
        let url = parse_www_authenticate_resource_metadata(Some(
            r#"Bearer realm="mcp", resource_metadata="https://rm.example/md", error="invalid_token""#,
        ))
        .unwrap();
        assert_eq!(url, "https://rm.example/md");
    }

    #[test]
    fn parse_www_authenticate_rejects_non_bearer() {
        assert!(parse_www_authenticate_resource_metadata(Some("Basic realm=\"x\"")).is_none());
    }

    #[test]
    fn parse_www_authenticate_none_when_missing() {
        assert!(parse_www_authenticate_resource_metadata(None).is_none());
        // No resource_metadata= param at all.
        assert!(parse_www_authenticate_resource_metadata(Some(r#"Bearer realm="x""#)).is_none());
    }

    #[test]
    fn build_authorization_url_roundtrip() {
        let url = build_authorization_url(&AuthorizationUrlArgs {
            authorization_endpoint: "https://as.example.com/authorize",
            client_id: "clid",
            redirect_uri: "http://127.0.0.1:8080/oauth/callback",
            state: "state-xyz",
            code_challenge: "CHAL",
            scope: Some("read write"),
            resource: "https://mcp.example.com/mcp",
        })
        .unwrap();
        let parsed = reqwest::Url::parse(&url).unwrap();
        let q: std::collections::HashMap<String, String> = parsed
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        assert_eq!(q.get("client_id").unwrap(), "clid");
        assert_eq!(q.get("code_challenge").unwrap(), "CHAL");
        assert_eq!(q.get("code_challenge_method").unwrap(), "S256");
        assert_eq!(
            q.get("redirect_uri").unwrap(),
            "http://127.0.0.1:8080/oauth/callback"
        );
        assert_eq!(q.get("resource").unwrap(), "https://mcp.example.com/mcp");
        assert_eq!(q.get("response_type").unwrap(), "code");
        assert_eq!(q.get("scope").unwrap(), "read write");
        assert_eq!(q.get("state").unwrap(), "state-xyz");
    }

    #[test]
    fn build_authorization_url_omits_scope_when_none() {
        let url = build_authorization_url(&AuthorizationUrlArgs {
            authorization_endpoint: "https://as.example.com/authorize",
            client_id: "clid",
            redirect_uri: "http://127.0.0.1:8080/oauth/callback",
            state: "s",
            code_challenge: "C",
            scope: None,
            resource: "https://mcp.example.com/mcp",
        })
        .unwrap();
        assert!(!url.contains("scope="));
    }
}
