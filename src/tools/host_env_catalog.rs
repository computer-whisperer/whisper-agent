//! Durable host-env provider catalog.
//!
//! The server persists its registered host-env providers to a TOML
//! file sibling to the pods directory (`<pods_root>/../host_env_providers.toml`).
//! Pods carry `[[allow.host_env]]` bindings that reference providers
//! by name; if the catalog vanished on every restart, those bindings
//! would dangle. So the catalog is first-class durable state, loaded
//! at startup and mutated through scheduler commands.
//!
//! `whisper-agent.toml`'s `[[host_env_providers]]` entries still work —
//! they act as a one-time seed: insert-if-missing by name on load, so
//! TOML edits can seed a fresh install but can't clobber entries the
//! operator added or edited through the runtime (WebUI / RPC) surface.
//!
//! Tokens live inline here. The file is written with 0600 perms and
//! the server refuses to load it if perms are broader.

use std::fs;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::info;

/// Default filename. Lives next to the pods directory so both are
/// under the same durable-state root.
pub const CATALOG_FILENAME: &str = "host_env_providers.toml";

/// Resolve the catalog path from a pods root. The catalog sits as a
/// sibling of the pods directory itself (`<pods_root>/../<CATALOG_FILENAME>`)
/// so the pods subtree stays self-contained — an archived pods dir
/// doesn't drag a credentials file along with it.
pub fn default_path_for_pods_root(pods_root: &Path) -> PathBuf {
    match pods_root.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(CATALOG_FILENAME),
        // `pods_root` is a bare relative path like "pods" — sibling
        // means the current working directory.
        _ => PathBuf::from(CATALOG_FILENAME),
    }
}

/// Where a catalog entry came from. `Seeded` entries were imported
/// from `[[host_env_providers]]` in `whisper-agent.toml`; `Manual`
/// entries were added at runtime (CLI can't create `Manual` entries —
/// CLI flags are a runtime overlay that isn't persisted).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogOrigin {
    Seeded,
    Manual,
}

/// One persisted provider entry. `token` is stored inline — tokens in
/// the catalog file are acceptable because the file itself is 0600 and
/// lives in the server's state root, same trust boundary as the pods
/// directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    pub origin: CatalogOrigin,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// On-disk wrapper. TOML does best with a named top-level array.
#[derive(Debug, Default, Serialize, Deserialize)]
struct CatalogFile {
    #[serde(default, rename = "provider")]
    providers: Vec<CatalogEntry>,
}

/// In-memory view plus the path it's bound to. The scheduler owns one
/// of these and calls `save` after every mutation. Uses `std::fs`
/// throughout — the catalog is small (typically <10 entries, <10 KB)
/// and synchronous I/O through the scheduler's command path is simpler
/// than threading async writes through the loop.
#[derive(Debug)]
pub struct CatalogStore {
    path: PathBuf,
    entries: Vec<CatalogEntry>,
}

impl CatalogStore {
    /// Load from `path`. A missing file yields an empty catalog (not an
    /// error — a fresh server should boot without operator setup). A
    /// file with perms broader than 0600 is a hard error: we refuse to
    /// read secrets out of a world-readable file.
    pub fn load(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            info!(path = %path.display(), "host-env catalog not found; starting empty");
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
        for entry in &parsed.providers {
            if !seen.insert(entry.name.as_str()) {
                bail!(
                    "catalog {}: duplicate provider `{}`",
                    path.display(),
                    entry.name
                );
            }
        }
        Ok(Self {
            path,
            entries: parsed.providers,
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

    /// Insert a new entry, erroring if the name is already present.
    /// Persists on success.
    pub fn insert(&mut self, entry: CatalogEntry) -> Result<()> {
        if self.contains(&entry.name) {
            bail!("provider `{}` already exists in catalog", entry.name);
        }
        self.entries.push(entry);
        self.save()
    }

    /// Insert if the name isn't present; otherwise no-op. Returns
    /// whether the entry was newly inserted. Used for TOML seed-import.
    pub fn insert_if_missing(&mut self, entry: CatalogEntry) -> Result<bool> {
        if self.contains(&entry.name) {
            return Ok(false);
        }
        self.entries.push(entry);
        self.save()?;
        Ok(true)
    }

    /// Replace `url` and `token` for an existing entry. `origin` and
    /// `created_at` are preserved; `updated_at` advances to `now`.
    pub fn update(
        &mut self,
        name: &str,
        url: String,
        token: Option<String>,
        now: DateTime<Utc>,
    ) -> Result<()> {
        let entry = self
            .entries
            .iter_mut()
            .find(|e| e.name == name)
            .ok_or_else(|| anyhow!("provider `{name}` not in catalog"))?;
        entry.url = url;
        entry.token = token;
        entry.updated_at = now;
        self.save()
    }

    /// Remove by name. Returns the removed entry for loggability;
    /// returns `None` if the name wasn't present (caller decides
    /// whether to treat that as an error).
    pub fn remove(&mut self, name: &str) -> Result<Option<CatalogEntry>> {
        let Some(idx) = self.entries.iter().position(|e| e.name == name) else {
            return Ok(None);
        };
        let removed = self.entries.remove(idx);
        self.save()?;
        Ok(Some(removed))
    }

    /// Write the catalog atomically: serialize to a tmp sibling with
    /// 0600, then rename over the target. Rename is atomic on POSIX so
    /// a crash mid-write can't leave the catalog truncated.
    fn save(&self) -> Result<()> {
        let body = CatalogFile {
            providers: self.entries.clone(),
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
        // Enforce perms even if the tmp file pre-existed with broader
        // mode (OpenOptions::mode only applies on create).
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

/// Helper used by startup seeding. Produces a fresh `Manual` entry.
pub fn new_manual_entry(
    name: String,
    url: String,
    token: Option<String>,
    now: DateTime<Utc>,
) -> CatalogEntry {
    CatalogEntry {
        name,
        url,
        token,
        origin: CatalogOrigin::Manual,
        created_at: now,
        updated_at: now,
    }
}

/// Helper used by startup seeding.
pub fn new_seeded_entry(
    name: String,
    url: String,
    token: Option<String>,
    now: DateTime<Utc>,
) -> CatalogEntry {
    CatalogEntry {
        name,
        url,
        token,
        origin: CatalogOrigin::Seeded,
        created_at: now,
        updated_at: now,
    }
}

/// Drop-on-load warning for seeded-then-removed names: callers iterate
/// TOML-seed entries and call `insert_if_missing`. If a TOML entry's
/// name doesn't land because an existing entry won the dedup, we
/// emit a single info-level log so operators can correlate TOML edits
/// that had no effect.
pub fn log_seed_result(name: &str, inserted: bool) {
    if inserted {
        info!(provider = %name, "imported [[host_env_providers]] entry into catalog");
    } else {
        // Deliberately info, not warn — the catalog entry winning is
        // the designed semantics, not a misconfiguration.
        info!(
            provider = %name,
            "TOML [[host_env_providers]] entry ignored; catalog already has `{name}`"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Per-test scratch dir under `std::env::temp_dir()`. Matches the
    /// convention in `pod::persist::tests` — no `tempfile` dep.
    fn scratch_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("wa-catalog-test-{}-{n}", std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn now() -> DateTime<Utc> {
        // Fixed timestamp keeps TOML golden-compares stable.
        DateTime::parse_from_rfc3339("2026-04-20T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn load_missing_yields_empty() {
        let dir = scratch_dir();
        let path = dir.join("host_env_providers.toml");
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
                "landlock-laptop".into(),
                "http://127.0.0.1:8080".into(),
                Some("secret".into()),
                now(),
            ))
            .unwrap();
        store
            .insert(new_seeded_entry(
                "bubblewrap-ci".into(),
                "http://ci:9090".into(),
                None,
                now(),
            ))
            .unwrap();

        let reloaded = CatalogStore::load(path).unwrap();
        assert_eq!(reloaded.entries(), store.entries());
        assert_eq!(reloaded.entries().len(), 2);
        assert_eq!(reloaded.entries()[0].name, "landlock-laptop");
        assert_eq!(reloaded.entries()[0].origin, CatalogOrigin::Manual);
        assert_eq!(reloaded.entries()[1].origin, CatalogOrigin::Seeded);
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
                Some("t".into()),
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
            .insert_if_missing(new_seeded_entry("a".into(), "http://a".into(), None, now()))
            .unwrap();
        assert!(inserted);
        // Re-insert under same name with a different url — dedup wins.
        let inserted = store
            .insert_if_missing(new_seeded_entry(
                "a".into(),
                "http://changed".into(),
                None,
                now(),
            ))
            .unwrap();
        assert!(!inserted);
        assert_eq!(store.get("a").unwrap().url, "http://a");
    }

    #[test]
    fn update_preserves_origin_and_created_at() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path).unwrap();
        let original = now();
        store
            .insert(new_seeded_entry(
                "a".into(),
                "http://a".into(),
                None,
                original,
            ))
            .unwrap();
        let later = original + chrono::Duration::hours(1);
        store
            .update("a", "http://b".into(), Some("tok".into()), later)
            .unwrap();
        let e = store.get("a").unwrap();
        assert_eq!(e.url, "http://b");
        assert_eq!(e.token.as_deref(), Some("tok"));
        assert_eq!(e.origin, CatalogOrigin::Seeded);
        assert_eq!(e.created_at, original);
        assert_eq!(e.updated_at, later);
    }

    #[test]
    fn update_missing_errors() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path).unwrap();
        let err = store
            .update("nope", "http://x".into(), None, now())
            .unwrap_err();
        assert!(format!("{err}").contains("not in catalog"));
    }

    #[test]
    fn remove_returns_entry_then_absent() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path).unwrap();
        store
            .insert(new_manual_entry("a".into(), "http://a".into(), None, now()))
            .unwrap();
        let removed = store.remove("a").unwrap();
        assert!(removed.is_some());
        assert!(!store.contains("a"));
        // Second remove yields None.
        let removed = store.remove("a").unwrap();
        assert!(removed.is_none());
    }

    #[test]
    fn insert_duplicate_errors() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path).unwrap();
        store
            .insert(new_manual_entry("a".into(), "http://a".into(), None, now()))
            .unwrap();
        let err = store
            .insert(new_manual_entry(
                "a".into(),
                "http://a2".into(),
                None,
                now(),
            ))
            .unwrap_err();
        assert!(format!("{err}").contains("already exists"));
    }

    #[test]
    fn default_path_for_pods_root_is_sibling() {
        let pods_root = PathBuf::from("/var/lib/whisper-agent/pods");
        assert_eq!(
            default_path_for_pods_root(&pods_root),
            PathBuf::from("/var/lib/whisper-agent/host_env_providers.toml")
        );
    }

    #[test]
    fn default_path_for_bare_pods_root_is_cwd() {
        let pods_root = PathBuf::from("pods");
        assert_eq!(
            default_path_for_pods_root(&pods_root),
            PathBuf::from(CATALOG_FILENAME)
        );
    }

    #[test]
    fn atomic_write_leaves_no_tmp_file_on_success() {
        let dir = scratch_dir();
        let path = dir.join(CATALOG_FILENAME);
        let mut store = CatalogStore::load(path.clone()).unwrap();
        store
            .insert(new_manual_entry("a".into(), "http://a".into(), None, now()))
            .unwrap();
        let tmp = tmp_path(&path);
        assert!(!tmp.exists(), "tmp file should be renamed away");
        assert!(path.exists());
    }
}
