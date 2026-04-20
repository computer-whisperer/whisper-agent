//! Persisted login credentials for the desktop client.
//!
//! The login screen writes the user's last server URL and bearer
//! token here so the next launch can skip the form. File lives at
//! `$XDG_CONFIG_HOME/whisper-agent/desktop.toml` (falling back to
//! `$HOME/.config/whisper-agent/desktop.toml`).
//!
//! The token is stored in plaintext — the config dir is already
//! user-scoped and there's no platform keyring dep we're willing to
//! take on. Users who want the token off disk can pass `--token-file`
//! and leave "Remember" unchecked in the form.

use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub server: Option<String>,
    pub token: Option<String>,
}

pub fn path() -> Result<PathBuf> {
    let base = if let Some(p) = std::env::var_os("XDG_CONFIG_HOME").filter(|s| !s.is_empty()) {
        PathBuf::from(p)
    } else {
        let home = std::env::var_os("HOME").context("HOME env var not set")?;
        PathBuf::from(home).join(".config")
    };
    Ok(base.join("whisper-agent").join("desktop.toml"))
}

pub fn load() -> Result<Config> {
    let p = path()?;
    if !p.exists() {
        return Ok(Config::default());
    }
    let raw = fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", p.display()))
}

pub fn save(cfg: &Config) -> Result<()> {
    let p = path()?;
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    let raw = toml::to_string_pretty(cfg).context("serialize config")?;
    fs::write(&p, raw).with_context(|| format!("write {}", p.display()))?;
    Ok(())
}
