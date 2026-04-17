//! Reader / refresher for Codex CLI's `auth.json`.
//!
//! When a user authenticates the OpenAI Codex CLI (`codex login`), it drops a
//! credentials file at `~/.codex/auth.json` containing the ChatGPT OAuth
//! `id_token` (a JWT), `access_token`, `refresh_token`, and the
//! `chatgpt_account_id` claim carried in the JWT. We piggyback on that file so
//! users with a ChatGPT Plus/Pro subscription can hit the Responses API
//! without us implementing our own OAuth flow.
//!
//! Access tokens are short-lived (typically 1h). Before each request the
//! provider calls [`CodexAuth::ensure_fresh`], which refreshes via
//! `POST https://auth.openai.com/oauth/token` (grant_type=refresh_token) when
//! the token's `exp` claim is within a small safety margin. Refreshed tokens
//! are written back to disk so the next process start reuses them.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// OAuth refresh endpoint. Matches Codex's own constant so a future Codex-side
/// move of the endpoint only requires updating one value here.
const REFRESH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// Codex's public OAuth client_id. Using the same id the Codex CLI uses means
/// refresh tokens minted by `codex login` are directly usable by us.
const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// Refresh an access token this many seconds before its `exp` claim so the
/// server doesn't reject in-flight requests on a thin margin.
const REFRESH_SAFETY_MARGIN: Duration = Duration::seconds(60);

/// In-memory view of `auth.json`. Holds the original file path so refreshes
/// can rewrite the same location.
pub struct CodexAuth {
    path: PathBuf,
    data: AuthDotJson,
    exp: Option<DateTime<Utc>>,
    account_id: Option<String>,
}

impl CodexAuth {
    /// Load from `~/.codex/auth.json` by default, or the user-specified path.
    pub fn load(path: Option<PathBuf>) -> Result<Self> {
        let path = match path {
            Some(p) => p,
            None => default_path()?,
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read codex auth file {}", path.display()))?;
        let data: AuthDotJson = serde_json::from_str(&text)
            .with_context(|| format!("parse codex auth file {}", path.display()))?;
        let tokens = data
            .tokens
            .as_ref()
            .ok_or_else(|| anyhow!("codex auth file has no `tokens` block — run `codex login`"))?;
        let exp = parse_jwt_exp(&tokens.access_token).unwrap_or(None);
        let account_id = tokens
            .account_id
            .clone()
            .or_else(|| parse_account_id(&tokens.id_token).ok().flatten());
        Ok(Self {
            path,
            data,
            exp,
            account_id,
        })
    }

    pub fn access_token(&self) -> &str {
        // Unwrap is safe: load() rejects a file without tokens.
        self.data
            .tokens
            .as_ref()
            .expect("tokens present")
            .access_token
            .as_str()
    }

    pub fn chatgpt_account_id(&self) -> Option<&str> {
        self.account_id.as_deref()
    }

    /// Refresh if the access token is expired or within the safety margin.
    pub async fn ensure_fresh(&mut self, http: &reqwest::Client) -> Result<()> {
        if !self.is_expired() {
            return Ok(());
        }
        self.refresh(http).await
    }

    fn is_expired(&self) -> bool {
        match self.exp {
            Some(exp) => Utc::now() + REFRESH_SAFETY_MARGIN >= exp,
            // Unknown expiry: treat as expired so we refresh before every call.
            // Conservative but safe.
            None => true,
        }
    }

    async fn refresh(&mut self, http: &reqwest::Client) -> Result<()> {
        let tokens = self
            .data
            .tokens
            .as_ref()
            .ok_or_else(|| anyhow!("no tokens to refresh"))?;
        let body = RefreshRequest {
            client_id: CODEX_CLIENT_ID,
            grant_type: "refresh_token",
            refresh_token: tokens.refresh_token.clone(),
        };
        let resp = http
            .post(REFRESH_TOKEN_URL)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("codex refresh: transport error")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "codex refresh returned {status}: {body} — run `codex login` again"
            ));
        }
        let fresh: RefreshResponse = resp.json().await.context("codex refresh: parse response")?;

        // Responses may omit id_token or refresh_token; keep the prior values in
        // those slots rather than clobbering them with None.
        let tokens_mut = self
            .data
            .tokens
            .as_mut()
            .expect("tokens present (checked above)");
        if let Some(id) = fresh.id_token {
            tokens_mut.id_token = id;
        }
        if let Some(access) = fresh.access_token {
            tokens_mut.access_token = access;
            self.exp = parse_jwt_exp(&tokens_mut.access_token).unwrap_or(None);
        }
        if let Some(refresh) = fresh.refresh_token {
            tokens_mut.refresh_token = refresh;
        }
        self.data.last_refresh = Some(Utc::now());

        // Re-derive account_id — a fresh id_token may carry an updated claim.
        if let Ok(Some(id)) = parse_account_id(&tokens_mut.id_token) {
            self.account_id = Some(id);
        }

        // Persist. Failure here is non-fatal for the current request (the
        // in-memory tokens work), but warn because a stale file means the next
        // process start will try to reuse the old expired access token.
        if let Err(err) = self.save() {
            tracing::warn!(
                path = %self.path.display(),
                error = %err,
                "failed to persist refreshed codex auth tokens"
            );
        }
        Ok(())
    }

    fn save(&self) -> Result<()> {
        let text = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(&self.path, text)
            .with_context(|| format!("write codex auth file {}", self.path.display()))?;
        Ok(())
    }
}

fn default_path() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME not set; cannot locate ~/.codex/auth.json")?;
    Ok(PathBuf::from(home).join(".codex/auth.json"))
}

/// Decode the JWT payload (no signature verification — we trust the file that
/// holds the JWT) and pull the `exp` claim.
fn parse_jwt_exp(jwt: &str) -> Result<Option<DateTime<Utc>>> {
    let payload: JwtStdClaims = decode_jwt_payload(jwt)?;
    Ok(payload
        .exp
        .and_then(|s| DateTime::<Utc>::from_timestamp(s, 0)))
}

/// Pull `chatgpt_account_id` from the `id_token` JWT. The claim lives at the
/// Auth0-flavoured namespaced path `https://api.openai.com/auth.chatgpt_account_id`.
fn parse_account_id(jwt: &str) -> Result<Option<String>> {
    let payload: JwtIdClaims = decode_jwt_payload(jwt)?;
    Ok(payload.auth.and_then(|a| a.chatgpt_account_id))
}

fn decode_jwt_payload<T: for<'a> Deserialize<'a>>(jwt: &str) -> Result<T> {
    let mut parts = jwt.split('.');
    let _header = parts.next();
    let payload_b64 = parts
        .next()
        .ok_or_else(|| anyhow!("invalid JWT: missing payload segment"))?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload_b64)
        .context("base64-decode JWT payload")?;
    serde_json::from_slice(&bytes).context("deserialize JWT payload")
}

// ---------- Wire / file types ----------

#[derive(Deserialize, Serialize, Debug, Clone)]
struct AuthDotJson {
    #[serde(
        rename = "OPENAI_API_KEY",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    openai_api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tokens: Option<TokenData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_refresh: Option<DateTime<Utc>>,
    /// Preserve any extra fields Codex may write (auth_mode, agent_identity, …)
    /// so we don't strip them on save.
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct TokenData {
    id_token: String,
    access_token: String,
    refresh_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    account_id: Option<String>,
}

#[derive(Serialize)]
struct RefreshRequest {
    client_id: &'static str,
    grant_type: &'static str,
    refresh_token: String,
}

#[derive(Deserialize)]
struct RefreshResponse {
    id_token: Option<String>,
    access_token: Option<String>,
    refresh_token: Option<String>,
}

#[derive(Deserialize)]
struct JwtStdClaims {
    #[serde(default)]
    exp: Option<i64>,
}

#[derive(Deserialize)]
struct JwtIdClaims {
    #[serde(rename = "https://api.openai.com/auth", default)]
    auth: Option<JwtAuthClaims>,
}

#[derive(Deserialize)]
struct JwtAuthClaims {
    #[serde(default)]
    chatgpt_account_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fake unsigned JWT with the given payload JSON.
    fn encode_jwt(payload: &serde_json::Value) -> String {
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"{}");
        let payload_b = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_string(payload).unwrap().as_bytes());
        let sig = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"sig");
        format!("{header}.{payload_b}.{sig}")
    }

    #[test]
    fn extracts_exp_and_account_id() {
        let jwt_access = encode_jwt(&serde_json::json!({ "exp": 1_700_000_000i64 }));
        let jwt_id = encode_jwt(&serde_json::json!({
            "https://api.openai.com/auth": { "chatgpt_account_id": "acc-123" }
        }));
        let exp = parse_jwt_exp(&jwt_access).unwrap().unwrap();
        assert_eq!(exp.timestamp(), 1_700_000_000);
        let acc = parse_account_id(&jwt_id).unwrap();
        assert_eq!(acc.as_deref(), Some("acc-123"));
    }

    #[test]
    fn preserves_unknown_fields_on_roundtrip() {
        let raw = r#"{
            "OPENAI_API_KEY": "sk-exchanged",
            "tokens": {
                "id_token": "id",
                "access_token": "at",
                "refresh_token": "rt",
                "account_id": "acc-1"
            },
            "last_refresh": "2026-04-17T10:00:00Z",
            "auth_mode": "ChatGPT"
        }"#;
        let parsed: AuthDotJson = serde_json::from_str(raw).unwrap();
        let round = serde_json::to_value(&parsed).unwrap();
        // The unknown `auth_mode` field should be preserved via flatten.
        assert_eq!(
            round.get("auth_mode").and_then(|v| v.as_str()),
            Some("ChatGPT")
        );
        assert_eq!(
            round.get("OPENAI_API_KEY").and_then(|v| v.as_str()),
            Some("sk-exchanged")
        );
    }
}
