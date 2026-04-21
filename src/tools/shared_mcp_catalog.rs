//! Durable shared-MCP-host catalog.
//!
//! Sibling of the pods directory (`<pods_root>/../shared_mcp_hosts.toml`),
//! same trust boundary as `host_env_providers.toml`. Unlike host-env
//! providers — which are our own daemons we provision and own —
//! shared MCP hosts are endpoints the operator points us at, often
//! third-party MCP servers with their own auth story. Step 1 supports
//! anonymous + static bearer tokens; Step 2 extends [`SharedMcpAuth`]
//! with an OAuth 2.1 variant for servers that require a browser-driven
//! authorization flow.
//!
//! `whisper-agent.toml`'s `[shared_mcp_hosts]` table (simple name → url
//! map with no auth field) still works — it's imported as a one-time
//! seed on load with `SharedMcpAuth::None`, matching the pre-catalog
//! behaviour. Entries added through the runtime surface (WebUI / RPC)
//! win on name collision.
//!
//! File is written 0600; load refuses broader perms for the same
//! reason as the host-env catalog — tokens live inline.

use std::fs;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::tools::host_env_catalog::CatalogOrigin;

/// Default filename, sibling of the pods directory.
pub const CATALOG_FILENAME: &str = "shared_mcp_hosts.toml";

/// Resolve the catalog path from a pods root. Same layout as the
/// host-env catalog — sibling of `pods/`, keeps tokens out of any
/// tarball of the pods subtree.
pub fn default_path_for_pods_root(pods_root: &Path) -> PathBuf {
    match pods_root.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(CATALOG_FILENAME),
        _ => PathBuf::from(CATALOG_FILENAME),
    }
}

/// Auth configuration for a catalog entry. Tagged union keyed by
/// `kind` so Step 2 can add an `oauth2` variant without rewriting
/// existing rows. `None` is the default and emits nothing on the
/// wire — an entry with no `[host.auth]` table parses as anonymous.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SharedMcpAuth {
    #[default]
    None,
    /// Static bearer presented as `Authorization: Bearer <token>`.
    /// Suitable for Slack-style bot tokens and long-lived PATs.
    Bearer { token: String },
}

impl SharedMcpAuth {
    pub fn is_none(&self) -> bool {
        matches!(self, SharedMcpAuth::None)
    }

    /// Bearer token to attach on MCP requests, if any.
    pub fn bearer(&self) -> Option<&str> {
        match self {
            SharedMcpAuth::Bearer { token } => Some(token.as_str()),
            SharedMcpAuth::None => None,
        }
    }

    /// Public auth classification — what clients are allowed to see.
    pub fn public(&self) -> whisper_agent_protocol::SharedMcpAuthPublic {
        use whisper_agent_protocol::SharedMcpAuthPublic;
        match self {
            SharedMcpAuth::None => SharedMcpAuthPublic::None,
            SharedMcpAuth::Bearer { .. } => SharedMcpAuthPublic::Bearer,
        }
    }
}

/// One persisted shared-MCP-host entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "SharedMcpAuth::is_none")]
    pub auth: SharedMcpAuth,
    pub origin: CatalogOrigin,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// On-disk wrapper. Uses `[[host]]` rather than `[[provider]]` so the
/// file reads naturally (`host_env_providers.toml` has providers,
/// `shared_mcp_hosts.toml` has hosts).
#[derive(Debug, Default, Serialize, Deserialize)]
struct CatalogFile {
    #[serde(default, rename = "host")]
    hosts: Vec<CatalogEntry>,
}

/// In-memory catalog + backing path. Same lifecycle as the host-env
/// `CatalogStore`: scheduler owns one, mutations go through here and
/// save synchronously after each change.
#[derive(Debug)]
pub struct CatalogStore {
    path: PathBuf,
    entries: Vec<CatalogEntry>,
}

impl CatalogStore {
    pub fn load(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            info!(path = %path.display(), "shared-MCP catalog not found; starting empty");
            return Ok(Self {
                path,
                entries: Vec::new(),
            });
        }
        let meta =
            fs::metadata(&path).with_context(|| format!("stat catalog {}", path.display()))?;
        let mode = meta.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            bail!(
                "catalog {} has mode {:04o}; refusing to read tokens out of a file readable by group/other (chmod 600)",
                path.display(),
                mode
            );
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read catalog {}", path.display()))?;
        let parsed: CatalogFile =
            toml::from_str(&text).with_context(|| format!("parse catalog {}", path.display()))?;
        let mut seen = std::collections::HashSet::new();
        for entry in &parsed.hosts {
            if !seen.insert(entry.name.as_str()) {
                bail!(
                    "catalog {}: duplicate host `{}`",
                    path.display(),
                    entry.name
                );
            }
        }
        Ok(Self {
            path,
            entries: parsed.hosts,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    pub fn get(&self, name: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn insert(&mut self, entry: CatalogEntry) -> Result<()> {
        if self.contains(&entry.name) {
            bail!("host `{}` already exists in catalog", entry.name);
        }
        self.entries.push(entry);
        self.save()
    }

    /// Insert when the name is absent; otherwise no-op. Used by the
    /// TOML-seed import path.
    pub fn insert_if_missing(&mut self, entry: CatalogEntry) -> Result<bool> {
        if self.contains(&entry.name) {
            return Ok(false);
        }
        self.entries.push(entry);
        self.save()?;
        Ok(true)
    }

    /// Replace `url` and `auth` on an existing entry. `origin` and
    /// `created_at` are preserved; `updated_at` advances to `now`.
    /// Passing `auth = None` leaves the existing auth untouched —
    /// callers that want to clear auth should pass
    /// `Some(SharedMcpAuth::None)`.
    pub fn update(
        &mut self,
        name: &str,
        url: String,
        auth: Option<SharedMcpAuth>,
        now: DateTime<Utc>,
    ) -> Result<()> {
        let entry = self
            .entries
            .iter_mut()
            .find(|e| e.name == name)
            .ok_or_else(|| anyhow!("host `{name}` not in catalog"))?;
        entry.url = url;
        if let Some(new_auth) = auth {
            entry.auth = new_auth;
        }
        entry.updated_at = now;
        self.save()
    }

    pub fn remove(&mut self, name: &str) -> Result<Option<CatalogEntry>> {
        let Some(idx) = self.entries.iter().position(|e| e.name == name) else {
            return Ok(None);
        };
        let removed = self.entries.remove(idx);
        self.save()?;
        Ok(Some(removed))
    }

    fn save(&self) -> Result<()> {
        let body = CatalogFile {
            hosts: self.entries.clone(),
        };
        let text = toml::to_string_pretty(&body).context("serialize catalog")?;
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).with_context(|| format!("mkdir -p {}", parent.display()))?;
        }
        let tmp = tmp_path(&self.path);
        {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&tmp)
                .with_context(|| format!("open {}", tmp.display()))?;
            f.write_all(text.as_bytes())
                .with_context(|| format!("write {}", tmp.display()))?;
            f.sync_all().ok();
        }
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 600 {}", tmp.display()))?;
        fs::rename(&tmp, &self.path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), self.path.display()))?;
        Ok(())
    }
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(name),
        _ => PathBuf::from(name),
    }
}

pub fn new_manual_entry(
    name: String,
    url: String,
    auth: SharedMcpAuth,
    now: DateTime<Utc>,
) -> CatalogEntry {
    CatalogEntry {
        name,
        url,
        auth,
        origin: CatalogOrigin::Manual,
        created_at: now,
        updated_at: now,
    }
}

pub fn new_seeded_entry(
    name: String,
    url: String,
    auth: SharedMcpAuth,
    now: DateTime<Utc>,
) -> CatalogEntry {
    CatalogEntry {
        name,
        url,
        auth,
        origin: CatalogOrigin::Seeded,
        created_at: now,
        updated_at: now,
    }
}

pub fn log_seed_result(name: &str, inserted: bool) {
    if inserted {
        info!(host = %name, "imported [shared_mcp_hosts] entry into catalog");
    } else {
        info!(
            host = %name,
            "TOML [shared_mcp_hosts] entry ignored; catalog already has `{name}`"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("wa-mcp-catalog-test-{}-{n}", std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-04-20T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn load_missing_yields_empty() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let store = CatalogStore::load(path.clone()).unwrap();
        assert!(store.entries().is_empty());
        assert_eq!(store.path(), &path);
    }

    #[test]
    fn roundtrip_preserves_entries() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path.clone()).unwrap();
        store
            .insert(new_manual_entry(
                "slack".into(),
                "https://mcp.example.com".into(),
                SharedMcpAuth::Bearer {
                    token: "xoxb-secret".into(),
                },
                now(),
            ))
            .unwrap();
        store
            .insert(new_seeded_entry(
                "fetch".into(),
                "http://127.0.0.1:9830/mcp".into(),
                SharedMcpAuth::None,
                now(),
            ))
            .unwrap();

        let reloaded = CatalogStore::load(path).unwrap();
        assert_eq!(reloaded.entries(), store.entries());
        assert_eq!(reloaded.entries().len(), 2);
        assert_eq!(reloaded.entries()[0].name, "slack");
        assert_eq!(reloaded.entries()[0].origin, CatalogOrigin::Manual);
        assert_eq!(reloaded.entries()[1].auth, SharedMcpAuth::None);
    }

    #[test]
    fn save_is_0600() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path.clone()).unwrap();
        store
            .insert(new_manual_entry(
                "a".into(),
                "http://a".into(),
                SharedMcpAuth::Bearer { token: "t".into() },
                now(),
            ))
            .unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "catalog file mode must be 0600, got {mode:o}");
    }

    #[test]
    fn refuses_to_load_world_readable_catalog() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        fs::write(&path, "").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        let err = CatalogStore::load(path).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("0644"), "expected mode complaint: {msg}");
    }

    #[test]
    fn insert_if_missing_is_idempotent() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path).unwrap();
        let inserted = store
            .insert_if_missing(new_seeded_entry(
                "a".into(),
                "http://a".into(),
                SharedMcpAuth::None,
                now(),
            ))
            .unwrap();
        assert!(inserted);
        let inserted = store
            .insert_if_missing(new_seeded_entry(
                "a".into(),
                "http://changed".into(),
                SharedMcpAuth::None,
                now(),
            ))
            .unwrap();
        assert!(!inserted);
        assert_eq!(store.get("a").unwrap().url, "http://a");
    }

    #[test]
    fn update_preserves_origin_and_auth_when_none() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path).unwrap();
        let original = now();
        store
            .insert(new_seeded_entry(
                "a".into(),
                "http://a".into(),
                SharedMcpAuth::Bearer {
                    token: "keepme".into(),
                },
                original,
            ))
            .unwrap();
        let later = original + chrono::Duration::hours(1);
        // url-only edit — auth argument is None, so the token survives.
        store.update("a", "http://b".into(), None, later).unwrap();
        let e = store.get("a").unwrap();
        assert_eq!(e.url, "http://b");
        assert_eq!(
            e.auth,
            SharedMcpAuth::Bearer {
                token: "keepme".into(),
            }
        );
        assert_eq!(e.origin, CatalogOrigin::Seeded);
        assert_eq!(e.created_at, original);
        assert_eq!(e.updated_at, later);
    }

    #[test]
    fn update_clears_auth_when_explicit_none() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path).unwrap();
        store
            .insert(new_manual_entry(
                "a".into(),
                "http://a".into(),
                SharedMcpAuth::Bearer { token: "t".into() },
                now(),
            ))
            .unwrap();
        store
            .update("a", "http://a".into(), Some(SharedMcpAuth::None), now())
            .unwrap();
        assert_eq!(store.get("a").unwrap().auth, SharedMcpAuth::None);
    }

    #[test]
    fn remove_returns_entry_then_absent() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path).unwrap();
        store
            .insert(new_manual_entry(
                "a".into(),
                "http://a".into(),
                SharedMcpAuth::None,
                now(),
            ))
            .unwrap();
        let removed = store.remove("a").unwrap();
        assert!(removed.is_some());
        assert!(!store.contains("a"));
        let removed = store.remove("a").unwrap();
        assert!(removed.is_none());
    }

    #[test]
    fn default_path_for_pods_root_is_sibling() {
        let pods_root = PathBuf::from("/var/lib/whisper-agent/pods");
        assert_eq!(
            default_path_for_pods_root(&pods_root),
            PathBuf::from("/var/lib/whisper-agent/shared_mcp_hosts.toml")
        );
    }

    #[test]
    fn anonymous_entry_omits_auth_on_disk() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path.clone()).unwrap();
        store
            .insert(new_manual_entry(
                "anon".into(),
                "http://a".into(),
                SharedMcpAuth::None,
                now(),
            ))
            .unwrap();
        let disk = fs::read_to_string(&path).unwrap();
        assert!(
            !disk.contains("kind"),
            "anonymous entry should not serialize auth kind: {disk}"
        );
        assert!(disk.contains("anon"));
    }
}
