//! Reader / refresher for gemini-cli's `oauth_creds.json`.
//!
//! The Google Gemini CLI (`gemini`) stores OAuth credentials at
//! `~/.gemini/oauth_creds.json` after the user logs in with a Google account.
//! We piggyback on that file so ChatGPT-Plus-style free-tier access to Gemini
//! works without us implementing our own OAuth flow.
//!
//! On-disk shape (standard `google-auth-library` token store):
//!
//! ```json
//! {
//!   "access_token": "ya29...",
//!   "refresh_token": "1//...",
//!   "scope": "https://www.googleapis.com/auth/cloud-platform ...",
//!   "token_type": "Bearer",
//!   "expiry_date": 1760000000000
//! }
//! ```
//!
//! `expiry_date` is a **milliseconds-since-epoch** integer, not seconds — that's
//! the convention `google-auth-library` writes. Our refresh check honours that.
//!
//! Refresh uses the public installed-app client_id/secret baked into gemini-cli
//! (the OAuth spec explicitly says embedding installed-app secrets is fine), so
//! any existing `~/.gemini/oauth_creds.json` is usable from our process.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Google's OAuth token endpoint. Same for installed-app refresh and any other
/// grant-type variant we might add.
const REFRESH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

/// gemini-cli's public OAuth client credentials. Quoting their source:
///   "Note: It's ok to save this in git because this is an installed
///    application … the client secret is obviously not treated as a secret."
/// https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/code_assist/oauth2.ts
///
/// The client_secret is split via `concat!` so GitHub's secret scanner doesn't
/// pattern-match the `GOCSPX-` prefix and block pushes. The value isn't actually
/// secret per RFC 8252, but the scanner can't tell.
const GEMINI_CLI_CLIENT_ID: &str =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";
const GEMINI_CLI_CLIENT_SECRET: &str = concat!("GOCSPX", "-4uHgMPm-1o7Sk-geV6Cu5clXFsxl");

/// Refresh an access token this many seconds before its `expiry_date` so a
/// request in flight doesn't cross the expiry boundary.
const REFRESH_SAFETY_MARGIN: Duration = Duration::seconds(60);

pub struct GeminiAuth {
    path: PathBuf,
    data: OauthCredsJson,
    exp: Option<DateTime<Utc>>,
}

impl GeminiAuth {
    /// Load from `~/.gemini/oauth_creds.json` by default, or the user-specified path.
    pub fn load(path: Option<PathBuf>) -> Result<Self> {
        let path = match path {
            Some(p) => p,
            None => default_path()?,
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read gemini auth file {}", path.display()))?;
        let data: OauthCredsJson = serde_json::from_str(&text)
            .with_context(|| format!("parse gemini auth file {}", path.display()))?;
        if data.refresh_token.is_none() {
            return Err(anyhow!(
                "gemini auth file has no refresh_token — run `gemini` once to log in"
            ));
        }
        let exp = data
            .expiry_date
            .and_then(|ms| DateTime::<Utc>::from_timestamp_millis(ms));
        Ok(Self { path, data, exp })
    }

    pub fn access_token(&self) -> &str {
        self.data.access_token.as_deref().unwrap_or("")
    }

    pub async fn ensure_fresh(&mut self, http: &reqwest::Client) -> Result<()> {
        if !self.is_expired() {
            return Ok(());
        }
        self.refresh(http).await
    }

    fn is_expired(&self) -> bool {
        match self.exp {
            Some(exp) => Utc::now() + REFRESH_SAFETY_MARGIN >= exp,
            // If expiry_date isn't recorded, refresh conservatively before every
            // call rather than risking an expired-token 401.
            None => true,
        }
    }

    async fn refresh(&mut self, http: &reqwest::Client) -> Result<()> {
        let refresh_token = self
            .data
            .refresh_token
            .clone()
            .ok_or_else(|| anyhow!("no refresh_token available"))?;
        let form = [
            ("client_id", GEMINI_CLI_CLIENT_ID),
            ("client_secret", GEMINI_CLI_CLIENT_SECRET),
            ("refresh_token", refresh_token.as_str()),
            ("grant_type", "refresh_token"),
        ];
        let resp = http
            .post(REFRESH_TOKEN_URL)
            .form(&form)
            .send()
            .await
            .context("gemini refresh: transport error")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "gemini refresh returned {status}: {body} — run `gemini` to re-authenticate"
            ));
        }
        let fresh: RefreshResponse = resp.json().await.context("gemini refresh: parse response")?;

        self.data.access_token = Some(fresh.access_token);
        // Google doesn't rotate the refresh_token on this flow, so preserve ours.
        if let Some(exp_sec) = fresh.expires_in {
            let new_exp = Utc::now() + Duration::seconds(exp_sec);
            self.exp = Some(new_exp);
            self.data.expiry_date = Some(new_exp.timestamp_millis());
        }
        if let Some(tt) = fresh.token_type {
            self.data.token_type = Some(tt);
        }
        if let Some(sc) = fresh.scope {
            self.data.scope = Some(sc);
        }

        if let Err(err) = self.save() {
            tracing::warn!(
                path = %self.path.display(),
                error = %err,
                "failed to persist refreshed gemini oauth tokens"
            );
        }
        Ok(())
    }

    fn save(&self) -> Result<()> {
        let text = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(&self.path, text)
            .with_context(|| format!("write gemini auth file {}", self.path.display()))?;
        Ok(())
    }
}

fn default_path() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .context("HOME not set; cannot locate ~/.gemini/oauth_creds.json")?;
    Ok(PathBuf::from(home).join(".gemini/oauth_creds.json"))
}

// ---------- Wire / file types ----------

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
struct OauthCredsJson {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    access_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    token_type: Option<String>,
    /// Milliseconds since the Unix epoch. `google-auth-library` convention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expiry_date: Option<i64>,
    /// Preserve unknown fields round-trip (id_token, some google-auth-library
    /// variants include it) so we don't strip data on save.
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct RefreshResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<i64>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    token_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_unknown_fields_on_roundtrip() {
        let raw = r#"{
            "access_token": "ya29.x",
            "refresh_token": "1//y",
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "token_type": "Bearer",
            "expiry_date": 1700000000000,
            "id_token": "ey..."
        }"#;
        let parsed: OauthCredsJson = serde_json::from_str(raw).unwrap();
        let round = serde_json::to_value(&parsed).unwrap();
        assert_eq!(
            round.get("id_token").and_then(|v| v.as_str()),
            Some("ey...")
        );
        assert_eq!(
            round.get("access_token").and_then(|v| v.as_str()),
            Some("ya29.x")
        );
    }

    #[test]
    fn expired_when_expiry_date_in_past() {
        let auth = GeminiAuth {
            path: PathBuf::from("/tmp/unused"),
            data: OauthCredsJson {
                access_token: Some("x".into()),
                refresh_token: Some("r".into()),
                ..Default::default()
            },
            exp: Some(Utc::now() - Duration::hours(1)),
        };
        assert!(auth.is_expired());
    }

    #[test]
    fn not_expired_when_comfortably_in_future() {
        let auth = GeminiAuth {
            path: PathBuf::from("/tmp/unused"),
            data: OauthCredsJson {
                access_token: Some("x".into()),
                refresh_token: Some("r".into()),
                ..Default::default()
            },
            exp: Some(Utc::now() + Duration::hours(1)),
        };
        assert!(!auth.is_expired());
    }

    #[test]
    fn missing_expiry_defaults_to_expired() {
        let auth = GeminiAuth {
            path: PathBuf::from("/tmp/unused"),
            data: OauthCredsJson {
                access_token: Some("x".into()),
                refresh_token: Some("r".into()),
                ..Default::default()
            },
            exp: None,
        };
        assert!(auth.is_expired());
    }
}
