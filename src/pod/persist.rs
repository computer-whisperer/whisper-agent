//! Pod-directory persistence: each task lives at
//! `<pods_root>/<pod_id>/threads/<thread_id>.json` alongside a synthesized
//! `<pods_root>/<pod_id>/pod.toml`.
//!
//! Phase 2b lays the on-disk shape down without renaming the in-memory `Thread`
//! type or splitting `ThreadConfig`. The `pod.toml` is synthesized from the
//! task's config on first flush and *not* clobbered on subsequent flushes —
//! hand-edits survive even though the scheduler doesn't yet read them
//! (Phase 3 makes pod.toml authoritative).
//!
//! Layout:
//!
//! ```text
//! <pods_root>/
//!   <pod_id>/                         pod_id == task.id == thread_id today
//!     pod.toml                        synthesized on first flush
//!     system_prompt.md                from the thread's Role::System message
//!     threads/
//!       <thread_id>.json
//!   <other-pod>/...
//!   .pre-pod-refactor-<ts>/           stash of legacy `<pods_root>/<thread_id>.json`
//!                                     files swept aside on startup
//! ```
//!
//! In-flight task states (`AwaitingModel` / `AwaitingTools` / etc.) get
//! flipped to `Failed { at_phase: "resume" }` on `load_all` — same recovery
//! story as the previous flat-file layout.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use tokio::fs;
use tracing::{info, warn};

use crate::pod::behaviors::{self as pod_behaviors};
use crate::pod::fs::is_readonly_path;
use crate::pod::{self, POD_STATE_JSON, POD_TOML, Pod, PodId, THREADS_DIR};
use crate::runtime::thread::{Thread, ThreadInternalState};
use whisper_agent_protocol::{
    FsEntry, PodAllow, PodConfig, PodLimits, PodSnapshot, PodState, PodSummary, ThreadDefaults,
    ThreadSummary,
};

/// Result of `Persister::load_all`. Pods and threads are kept side-by-side so
/// the scheduler can register threads against their owning pod without
/// re-walking the directory tree.
pub struct LoadedState {
    pub pods: Vec<Pod>,
    pub threads: Vec<Thread>,
}

/// Upper bound on files the webui's generic viewer will read. 1 MiB is
/// generous for any markdown / config / text asset a user would realistically
/// want to edit inline and keeps the wire payload bounded.
const MAX_POD_FILE_READ_BYTES: u64 = 1_048_576;

/// Prefix scanned for null bytes before declaring a file binary.
/// 8 KiB matches git's heuristic for the same purpose.
const BINARY_SNIFF_BYTES: usize = 8192;

#[derive(Clone)]
pub struct Persister {
    pods_root: PathBuf,
}

impl Persister {
    /// Create the persister rooted at `pods_root`. Sweeps any pre-pod-refactor
    /// task files (`<pods_root>/*.json` left over from the flat-file layout)
    /// into a timestamped `.pre-pod-refactor-<ts>/` subdirectory so a fresh
    /// pod tree can start cleanly. Idempotent — no-op when nothing to sweep.
    pub async fn new(pods_root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&pods_root)
            .await
            .with_context(|| format!("create pods_root {}", pods_root.display()))?;
        if let Some(stash) = sweep_legacy(&pods_root).await? {
            warn!(
                pods_root = %pods_root.display(),
                stash = %stash.display(),
                "pre-pod-refactor task files moved aside; the pod tree starts fresh"
            );
        }
        Ok(Self { pods_root })
    }

    pub fn dir(&self) -> &Path {
        &self.pods_root
    }

    fn pod_dir(&self, pod_id: &str) -> PathBuf {
        self.pods_root.join(pod_id)
    }

    fn thread_path(&self, pod_id: &str, thread_id: &str) -> PathBuf {
        self.pod_dir(pod_id)
            .join(THREADS_DIR)
            .join(format!("{thread_id}.json"))
    }

    /// Write the task's thread JSON. The thread lands at
    /// `<pods_root>/<thread.pod_id>/threads/<thread.id>.json`. If the owning
    /// pod's `pod.toml` is missing (legacy pre-2d.iii threads still using
    /// pod_id == thread_id), one is synthesized so the directory is
    /// self-describing — the scheduler is the source of truth for live pods,
    /// but a hand-edited file system shouldn't end up with orphan thread
    /// JSONs.
    pub async fn flush(&self, task: &Thread) -> Result<()> {
        // Backstop for threads that were loaded before pod_id existed in
        // the JSON (serde defaulted to ""); fall back to id so the thread
        // still lands somewhere coherent.
        let pod_id = if task.pod_id.is_empty() {
            &task.id
        } else {
            &task.pod_id
        };
        let thread_id = &task.id;

        let pod_dir = self.pod_dir(pod_id);
        let threads_dir = pod_dir.join(THREADS_DIR);
        fs::create_dir_all(&threads_dir)
            .await
            .with_context(|| format!("mkdir {}", threads_dir.display()))?;

        let pod_toml = pod_dir.join(POD_TOML);
        if !fs::try_exists(&pod_toml).await.unwrap_or(false) {
            let synthesized = synthesize_pod_config(task);
            let toml_text = pod::to_toml(&synthesized).context("encode synthesized pod.toml")?;
            fs::write(&pod_toml, toml_text)
                .await
                .with_context(|| format!("write {}", pod_toml.display()))?;
            // Drop the system prompt next to pod.toml so it's hand-editable.
            // Empty prompts skip the file entirely. Source is the thread's
            // conversation[0] `Role::System` message — the config no
            // longer stores the prompt separately.
            let prompt_text = task.conversation.system_prompt_text();
            if !prompt_text.is_empty() {
                let prompt_path = pod_dir.join(&synthesized.thread_defaults.system_prompt_file);
                fs::write(&prompt_path, prompt_text)
                    .await
                    .with_context(|| format!("write {}", prompt_path.display()))?;
            }
        }

        let json = serde_json::to_vec_pretty(task)?;
        let path = self.thread_path(pod_id, thread_id);
        fs::write(&path, json)
            .await
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// Write a fresh pod's `pod.toml` to disk. Used by the scheduler when it
    /// synthesizes the default pod at startup. Idempotent — does nothing if
    /// the file already exists.
    pub async fn ensure_pod_toml(&self, pod_id: &str, config: &PodConfig) -> Result<()> {
        validate_pod_id(pod_id)?;
        let pod_dir = self.pod_dir(pod_id);
        fs::create_dir_all(pod_dir.join(THREADS_DIR))
            .await
            .with_context(|| format!("mkdir {}", pod_dir.display()))?;
        let pod_toml = pod_dir.join(POD_TOML);
        if fs::try_exists(&pod_toml).await.unwrap_or(false) {
            return Ok(());
        }
        let toml_text = pod::to_toml(config).context("encode pod.toml")?;
        fs::write(&pod_toml, toml_text)
            .await
            .with_context(|| format!("write {}", pod_toml.display()))
    }

    /// Walk `<pods_root>/*` and load every pod's `pod.toml` + every thread
    /// JSON underneath. Hidden directories (those whose name starts with `.`)
    /// are skipped — the `.pre-pod-refactor-*/` stash and `.archived/` future
    /// archive dir are both ignored. Pods whose `pod.toml` fails to parse
    /// are skipped with a warn (their threads are dropped too — the cap they
    /// declare is the only thing that makes the threads safe to run).
    pub async fn load_all(&self) -> Result<LoadedState> {
        let mut state = LoadedState {
            pods: Vec::new(),
            threads: Vec::new(),
        };
        let mut entries = match fs::read_dir(&self.pods_root).await {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(state),
            Err(e) => {
                return Err(e).with_context(|| format!("read_dir {}", self.pods_root.display()));
            }
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            let metadata = match entry.metadata().await {
                Ok(m) => m,
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "skip pod entry: stat failed");
                    continue;
                }
            };
            if !metadata.is_dir() {
                continue;
            }
            match load_pod(&path, name).await {
                Ok((pod, threads)) => {
                    state.pods.push(pod);
                    state.threads.extend(threads);
                }
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "skip pod: failed to load");
                }
            }
        }
        info!(
            pods = state.pods.len(),
            threads = state.threads.len(),
            "loaded persisted state",
        );
        Ok(state)
    }

    // ---------- Pod CRUD (Phase 2d.i wire surface) ----------

    /// Walk `<pods_root>/` and return one `PodSummary` per non-archived pod.
    /// Pods whose `pod.toml` fails to parse are still listed (so the UI can
    /// show them with an error state) — only the name/description fields
    /// fall back to defaults.
    pub async fn list_pods(&self) -> Result<Vec<PodSummary>> {
        let mut out = Vec::new();
        let mut entries = match fs::read_dir(&self.pods_root).await {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => {
                return Err(e).with_context(|| format!("read_dir {}", self.pods_root.display()));
            }
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.starts_with('.') || !entry.metadata().await?.is_dir() {
                continue;
            }
            out.push(read_pod_summary(name, &path).await);
        }
        out.sort_by(|a, b| a.pod_id.cmp(&b.pod_id));
        Ok(out)
    }

    /// Read one pod's parsed config, raw TOML, and thread summaries.
    pub async fn get_pod(&self, pod_id: &str) -> Result<Option<PodSnapshot>> {
        validate_pod_id(pod_id)?;
        let pod_dir = self.pod_dir(pod_id);
        if !fs::try_exists(&pod_dir).await.unwrap_or(false) {
            return Ok(None);
        }
        let toml_path = pod_dir.join(POD_TOML);
        let toml_text = fs::read_to_string(&toml_path)
            .await
            .with_context(|| format!("read {}", toml_path.display()))?;
        let config =
            pod::parse_toml(&toml_text).with_context(|| format!("parse {pod_id}/pod.toml"))?;
        let threads = read_thread_summaries(&pod_dir, pod_id).await;
        let behaviors: Vec<_> = pod_behaviors::load_behaviors_for_pod(&pod_dir, pod_id)
            .await
            .into_iter()
            .map(|b| b.summary())
            .collect();
        Ok(Some(PodSnapshot {
            pod_id: pod_id.to_string(),
            config,
            toml_text,
            threads,
            archived: false,
            behaviors,
        }))
    }

    /// Shallow directory listing under `<pods_root>/<pod_id>/<rel_path>`.
    /// Empty `rel_path` lists the pod root. Returns entries with dirs
    /// first (name-sorted), then files (name-sorted); dotfiles are
    /// filtered. `readonly` is computed from [`is_readonly_path`] on the
    /// full pod-relative path of each entry.
    ///
    /// Safety: `rel_path` must be a plain relative path — any `..` or
    /// absolute-path component returns an error rather than traversing
    /// outside the pod. Used by the webui's file-tree panel; the MCP
    /// `pod_list_files` tool has its own (allowlist-gated) listing
    /// in `tools::builtin_tools::filesystem`.
    pub async fn list_pod_dir(&self, pod_id: &str, rel_path: &str) -> Result<Vec<FsEntry>> {
        validate_pod_id(pod_id)?;
        let pod_dir = self.pod_dir(pod_id);
        if !fs::try_exists(&pod_dir).await.unwrap_or(false) {
            return Err(anyhow::anyhow!("pod `{pod_id}` does not exist"));
        }
        let rel = normalize_pod_relative_path(rel_path)?;
        let dir = if rel.is_empty() {
            pod_dir.clone()
        } else {
            pod_dir.join(&rel)
        };
        let meta = fs::metadata(&dir)
            .await
            .with_context(|| format!("stat {}", dir.display()))?;
        if !meta.is_dir() {
            return Err(anyhow::anyhow!("`{rel_path}` is not a directory"));
        }

        let mut read = fs::read_dir(&dir)
            .await
            .with_context(|| format!("read_dir {}", dir.display()))?;
        let mut out: Vec<FsEntry> = Vec::new();
        while let Some(entry) = read
            .next_entry()
            .await
            .with_context(|| format!("iter {}", dir.display()))?
        {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let ft = match entry.file_type().await {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            let is_dir = ft.is_dir();
            let size = if is_dir {
                0
            } else {
                entry.metadata().await.map(|m| m.len()).unwrap_or(0)
            };
            let full_rel = if rel.is_empty() {
                name.clone()
            } else {
                format!("{rel}/{name}")
            };
            let readonly = !is_dir && is_readonly_path(&full_rel);
            out.push(FsEntry {
                name,
                is_dir,
                size,
                readonly,
            });
        }
        out.sort_by(|a, b| match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        });
        Ok(out)
    }

    /// Read one file under `<pods_root>/<pod_id>/<rel_path>` for the
    /// webui's generic text viewer. Returns the file contents plus
    /// the `is_readonly_path` verdict so the viewer can hide the Save
    /// button. Rejects files larger than [`MAX_POD_FILE_READ_BYTES`]
    /// and files whose first [`BINARY_SNIFF_BYTES`] contain a null byte
    /// (the viewer is a plain-text surface and both cases are
    /// non-renderable). `rel_path` must be a plain relative path —
    /// same normalization rules as [`Persister::list_pod_dir`].
    pub async fn read_pod_file(&self, pod_id: &str, rel_path: &str) -> Result<(String, bool)> {
        validate_pod_id(pod_id)?;
        let rel = normalize_pod_relative_path(rel_path)?;
        if rel.is_empty() {
            return Err(anyhow::anyhow!("path is empty"));
        }
        let pod_dir = self.pod_dir(pod_id);
        if !fs::try_exists(&pod_dir).await.unwrap_or(false) {
            return Err(anyhow::anyhow!("pod `{pod_id}` does not exist"));
        }
        let path = pod_dir.join(&rel);
        let meta = fs::metadata(&path)
            .await
            .with_context(|| format!("stat {}", path.display()))?;
        if !meta.is_file() {
            return Err(anyhow::anyhow!("`{rel}` is not a regular file"));
        }
        if meta.len() > MAX_POD_FILE_READ_BYTES {
            return Err(anyhow::anyhow!(
                "file `{rel}` is {} bytes; viewer limit is {} bytes",
                meta.len(),
                MAX_POD_FILE_READ_BYTES
            ));
        }
        let bytes = fs::read(&path)
            .await
            .with_context(|| format!("read {}", path.display()))?;
        let sniff_end = bytes.len().min(BINARY_SNIFF_BYTES);
        if bytes[..sniff_end].contains(&0u8) {
            return Err(anyhow::anyhow!(
                "file `{rel}` appears to be binary (null byte in first {BINARY_SNIFF_BYTES} bytes)"
            ));
        }
        let content = String::from_utf8(bytes)
            .with_context(|| format!("decode {} as UTF-8", path.display()))?;
        Ok((content, is_readonly_path(&rel)))
    }

    /// Overwrite one text file under `<pods_root>/<pod_id>/<rel_path>`
    /// for the webui's generic editor. Rejects read-only paths
    /// (thread JSONs, pod_state.json, behaviors/*/state.json) and
    /// paths that escape the pod root. Creates parent directories
    /// implicitly if the file's immediate parent is missing, but
    /// will NOT create arbitrary nested directory trees — the parent
    /// has to already exist or be one level deep, matching the
    /// granularity of files the user can reach from the tree viewer.
    pub async fn write_pod_file(&self, pod_id: &str, rel_path: &str, content: &str) -> Result<()> {
        validate_pod_id(pod_id)?;
        let rel = normalize_pod_relative_path(rel_path)?;
        if rel.is_empty() {
            return Err(anyhow::anyhow!("path is empty"));
        }
        if is_readonly_path(&rel) {
            return Err(anyhow::anyhow!(
                "path `{rel}` is read-only (runtime state owned by the scheduler)"
            ));
        }
        let pod_dir = self.pod_dir(pod_id);
        if !fs::try_exists(&pod_dir).await.unwrap_or(false) {
            return Err(anyhow::anyhow!("pod `{pod_id}` does not exist"));
        }
        let path = pod_dir.join(&rel);
        if let Some(parent) = path.parent()
            && !fs::try_exists(parent).await.unwrap_or(false)
        {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
        fs::write(&path, content.as_bytes())
            .await
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// Create a fresh pod directory and write `pod.toml` from the supplied
    /// config. Fails if a pod with the same id already exists or if the
    /// config doesn't validate.
    /// Create a fresh pod directory, write `pod.toml` from the supplied
    /// config, and seed the sibling `system_prompt_file` with
    /// `system_prompt`. An empty `system_prompt` skips the prompt-file
    /// write (caller is responsible for ensuring the prompt exists
    /// before any thread fires — otherwise the pod runs with no system
    /// prompt, which providers handle heterogeneously: Anthropic
    /// rejects cache_control over empty system blocks, OpenAI accepts
    /// empty, etc.). An empty `system_prompt_file` field skips it too.
    pub async fn create_pod(
        &self,
        pod_id: &str,
        mut config: PodConfig,
        system_prompt: &str,
    ) -> Result<PodSummary> {
        validate_pod_id(pod_id)?;
        // Stamp created_at server-side when the client left it empty —
        // wasm clients don't always have a convenient ISO-8601 source
        // and we know the server's clock is the authoritative one.
        if config.created_at.is_empty() {
            config.created_at = Utc::now().to_rfc3339();
        }
        pod::validate(&config)?;
        let pod_dir = self.pod_dir(pod_id);
        if fs::try_exists(&pod_dir).await.unwrap_or(false) {
            return Err(anyhow::anyhow!("pod `{pod_id}` already exists"));
        }
        fs::create_dir_all(pod_dir.join(THREADS_DIR))
            .await
            .with_context(|| format!("mkdir {}", pod_dir.display()))?;
        let toml_text = pod::to_toml(&config).context("encode pod.toml")?;
        fs::write(pod_dir.join(POD_TOML), &toml_text)
            .await
            .with_context(|| format!("write {pod_id}/pod.toml"))?;
        let prompt_file = config.thread_defaults.system_prompt_file.as_str();
        if !prompt_file.is_empty() && !system_prompt.is_empty() {
            fs::write(pod_dir.join(prompt_file), system_prompt)
                .await
                .with_context(|| format!("write {pod_id}/{prompt_file}"))?;
        }
        Ok(PodSummary {
            pod_id: pod_id.to_string(),
            name: config.name,
            description: config.description,
            created_at: config.created_at,
            thread_count: 0,
            archived: false,
            behaviors_enabled: true,
        })
    }

    /// Replace the pod's `pod.toml` with the given text. Parses + validates
    /// before writing — on failure the on-disk file is untouched. Returns
    /// the parsed config so the caller can broadcast the new shape.
    pub async fn update_pod_config(&self, pod_id: &str, toml_text: &str) -> Result<PodConfig> {
        validate_pod_id(pod_id)?;
        let parsed = pod::parse_toml(toml_text).context("parse new pod.toml")?;
        let pod_dir = self.pod_dir(pod_id);
        if !fs::try_exists(&pod_dir).await.unwrap_or(false) {
            return Err(anyhow::anyhow!("pod `{pod_id}` does not exist"));
        }
        fs::write(pod_dir.join(POD_TOML), toml_text)
            .await
            .with_context(|| format!("write {pod_id}/pod.toml"))?;
        Ok(parsed)
    }

    /// Move the pod to `<pods_root>/.archived/<pod_id>-<ts>/`. Idempotent on
    /// already-archived pods (no-op if the source dir is missing).
    pub async fn archive_pod(&self, pod_id: &str) -> Result<()> {
        validate_pod_id(pod_id)?;
        let src = self.pod_dir(pod_id);
        if !fs::try_exists(&src).await.unwrap_or(false) {
            return Ok(());
        }
        let archive_root = self.pods_root.join(".archived");
        fs::create_dir_all(&archive_root)
            .await
            .with_context(|| format!("mkdir {}", archive_root.display()))?;
        let stamp = Utc::now().format("%Y%m%dT%H%M%SZ");
        let dst = archive_root.join(format!("{pod_id}-{stamp}"));
        fs::rename(&src, &dst)
            .await
            .with_context(|| format!("mv {} -> {}", src.display(), dst.display()))?;
        Ok(())
    }
}

/// Load `pod_state.json` from a pod directory. Missing file or parse
/// errors both fall back to `PodState::default()` — the flag is
/// operational, not authoritative, so a corrupt file shouldn't wedge
/// the pod on load.
pub async fn load_pod_state(pod_dir: &Path) -> PodState {
    let path = pod_dir.join(POD_STATE_JSON);
    match fs::read(&path).await {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            warn!(path = %path.display(), error = %e, "pod_state.json parse failed; defaulting");
            PodState::default()
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => PodState::default(),
        Err(e) => {
            warn!(path = %path.display(), error = %e, "pod_state.json read failed; defaulting");
            PodState::default()
        }
    }
}

/// Write `pod_state.json`. Overwrites unconditionally.
pub async fn write_pod_state(pod_dir: &Path, state: &PodState) -> Result<()> {
    let path = pod_dir.join(POD_STATE_JSON);
    let json = serde_json::to_vec_pretty(state)?;
    fs::write(&path, json)
        .await
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Normalize a pod-relative path supplied by the wire. Accepts empty
/// string (the pod root). Rejects absolute prefixes, `..` components,
/// embedded nulls, and backslashes; collapses redundant separators and
/// strips leading/trailing slashes so the returned string is always of
/// the form `a/b/c` or empty. The result is safe to `join` onto the
/// pod dir — it cannot escape the pod.
fn normalize_pod_relative_path(rel: &str) -> Result<String> {
    if rel.is_empty() {
        return Ok(String::new());
    }
    if rel.contains('\0') || rel.contains('\\') {
        return Err(anyhow::anyhow!("path contains illegal characters"));
    }
    let mut out: Vec<&str> = Vec::new();
    for part in rel.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return Err(anyhow::anyhow!("path may not contain `..` components"));
        }
        out.push(part);
    }
    Ok(out.join("/"))
}

/// Reject pod ids containing path separators or other shell-hostile bits.
/// Pod ids become directory names; we don't want `..` traversal or `/`-rooted
/// absolute paths sneaking in via the wire.
pub fn validate_pod_id(pod_id: &str) -> Result<()> {
    if pod_id.is_empty() {
        return Err(anyhow::anyhow!("pod_id is empty"));
    }
    if pod_id.starts_with('.') {
        return Err(anyhow::anyhow!("pod_id may not start with '.'"));
    }
    if pod_id.contains(['/', '\\', '\0']) || pod_id == ".." {
        return Err(anyhow::anyhow!("pod_id contains illegal characters"));
    }
    Ok(())
}

/// Build a `PodSummary` from the on-disk pod directory. If `pod.toml` is
/// missing or unparseable, fall back to the directory name as the display
/// name and an epoch timestamp — the UI can render the entry but the user
/// will need to fix the TOML to do anything with the pod.
async fn read_pod_summary(pod_id: &str, pod_dir: &Path) -> PodSummary {
    let toml_path = pod_dir.join(POD_TOML);
    let (name, description, created_at) = match fs::read_to_string(&toml_path).await {
        Ok(text) => match pod::parse_toml(&text) {
            Ok(cfg) => (cfg.name, cfg.description, cfg.created_at),
            Err(_) => (pod_id.to_string(), None, "1970-01-01T00:00:00Z".to_string()),
        },
        Err(_) => (pod_id.to_string(), None, "1970-01-01T00:00:00Z".to_string()),
    };
    let thread_count = count_threads(pod_dir).await;
    let state = load_pod_state(pod_dir).await;
    PodSummary {
        pod_id: pod_id.to_string(),
        name,
        description,
        created_at,
        thread_count,
        archived: false,
        behaviors_enabled: state.behaviors_enabled,
    }
}

async fn count_threads(pod_dir: &Path) -> u32 {
    let threads_dir = pod_dir.join(THREADS_DIR);
    let mut entries = match fs::read_dir(&threads_dir).await {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let mut n = 0u32;
    while let Ok(Some(entry)) = entries.next_entry().await {
        if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
            n += 1;
        }
    }
    n
}

async fn read_thread_summaries(pod_dir: &Path, pod_id: &str) -> Vec<ThreadSummary> {
    let threads_dir = pod_dir.join(THREADS_DIR);
    let mut entries = match fs::read_dir(&threads_dir).await {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match load_one(&path, pod_id).await {
            Ok(thread) => out.push(thread.summary()),
            Err(e) => warn!(path = %path.display(), error = %e, "skip thread summary"),
        }
    }
    out.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    out
}

/// Load one pod directory: parse `pod.toml`, then walk `threads/` and read
/// every JSON underneath. Stamps `pod_id` onto each loaded thread so the
/// scheduler can register it against the right pod.
async fn load_pod(pod_dir: &Path, pod_id: &str) -> Result<(Pod, Vec<Thread>)> {
    let toml_path = pod_dir.join(POD_TOML);
    let raw_toml = fs::read_to_string(&toml_path)
        .await
        .with_context(|| format!("read {}", toml_path.display()))?;
    let config = pod::parse_toml(&raw_toml).with_context(|| format!("parse {pod_id}/pod.toml"))?;

    let system_prompt = if config.thread_defaults.system_prompt_file.is_empty() {
        String::new()
    } else {
        let path = pod_dir.join(&config.thread_defaults.system_prompt_file);
        fs::read_to_string(&path).await.unwrap_or_default()
    };

    let mut threads = Vec::new();
    let threads_dir = pod_dir.join(THREADS_DIR);
    if let Ok(mut entries) = fs::read_dir(&threads_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match load_one(&path, pod_id).await {
                Ok(task) => threads.push(task),
                Err(e) => warn!(path = %path.display(), error = %e, "skip unreadable thread file"),
            }
        }
    }

    let mut pod = Pod::new(
        PodId::from(pod_id),
        pod_dir.to_path_buf(),
        config,
        raw_toml,
        system_prompt,
    );
    pod.state = load_pod_state(pod_dir).await;
    for t in &threads {
        pod.threads.insert(t.id.clone());
    }
    for b in pod_behaviors::load_behaviors_for_pod(pod_dir, pod_id).await {
        pod.behaviors.insert(b.id.clone(), b);
    }
    Ok((pod, threads))
}

async fn load_one(path: &Path, pod_id: &str) -> Result<Thread> {
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    // Pre-deserialize migration: older snapshots persisted
    // `bindings.host_env` as a bare `he-...` id string. The new
    // protocol expects an object (HostEnvBinding::Named|Inline) or
    // null, so we strip the legacy string here and let
    // `Scheduler::load_state` re-bind to the pod's current default.
    let mut value: serde_json::Value =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    normalize_legacy_host_env_binding(&mut value);
    let mut task: Thread =
        serde_json::from_value(value).with_context(|| format!("decode {}", path.display()))?;
    // Stamp pod_id from the directory we found it in. This wins over any
    // value baked into the JSON — a pod that was renamed on disk should
    // have its threads follow.
    task.pod_id = pod_id.to_string();
    // Pre-Role::ToolResult snapshots carried tool results inside user-
    // role messages. Split them out into their own role so downstream
    // code (provider adapters, webui renderer, event emitters) can
    // treat "user message" and "tool finished" as distinct concepts.
    // Idempotent — a no-op for already-migrated threads.
    task.conversation.normalize_legacy_tool_result_role();
    if is_in_flight(&task.internal) {
        task.fail("resume", "task was in-flight at last shutdown");
    }
    Ok(task)
}

fn normalize_legacy_host_env_binding(value: &mut serde_json::Value) {
    let Some(bindings) = value.get_mut("bindings") else {
        return;
    };
    // Pre-refactor `bindings.host_env` was a bare `he-...` id string.
    // The new shape is an object (or null); rewrite strings to null so
    // the scheduler rebinds to the pod default on load.
    if let Some(host_env) = bindings.get_mut("host_env")
        && host_env.is_string()
    {
        *host_env = serde_json::Value::Null;
    }
    // Pre-refactor `bindings.mcp_hosts` contained full resource-ids:
    // `mcp-primary-<tid>` at index 0 plus `mcp-shared-<name>` entries.
    // New shape stores bare shared-host names; host-env MCP isn't
    // represented here at all (it's derived from bindings.host_env).
    // Strip the primary entry and unwrap the shared prefix.
    if let Some(mcp_hosts) = bindings.get_mut("mcp_hosts")
        && let Some(arr) = mcp_hosts.as_array_mut()
    {
        let rewritten: Vec<serde_json::Value> = arr
            .iter()
            .filter_map(|v| v.as_str())
            .filter_map(|s| s.strip_prefix("mcp-shared-").map(String::from))
            .map(serde_json::Value::String)
            .collect();
        // Only rewrite if anything looked legacy — avoid pointless
        // churn on already-normalized files.
        let already_clean = arr.iter().all(|v| {
            v.as_str()
                .map(|s| !s.starts_with("mcp-") && !s.is_empty())
                .unwrap_or(false)
        });
        if !already_clean {
            *arr = rewritten;
        }
    }
}

fn is_in_flight(state: &ThreadInternalState) -> bool {
    matches!(
        state,
        ThreadInternalState::WaitingOnResources { .. }
            | ThreadInternalState::NeedsModelCall
            | ThreadInternalState::AwaitingModel { .. }
            | ThreadInternalState::AwaitingTools { .. }
    )
}

/// Move any pre-pod-refactor `<pods_root>/*.json` files into
/// `<pods_root>/.pre-pod-refactor-<ts>/`. Returns `Some(stash_dir)` when it
/// actually moved at least one file.
async fn sweep_legacy(pods_root: &Path) -> Result<Option<PathBuf>> {
    let mut entries = match fs::read_dir(pods_root).await {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("read_dir {}", pods_root.display())),
    };
    let mut to_move: Vec<PathBuf> = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            to_move.push(path);
        }
    }
    if to_move.is_empty() {
        return Ok(None);
    }
    let stamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    let stash = pods_root.join(format!(".pre-pod-refactor-{stamp}"));
    fs::create_dir_all(&stash)
        .await
        .with_context(|| format!("create stash {}", stash.display()))?;
    for src in to_move {
        let name = src.file_name().expect("dir entry has a name").to_owned();
        let dst = stash.join(&name);
        fs::rename(&src, &dst)
            .await
            .with_context(|| format!("mv {} -> {}", src.display(), dst.display()))?;
    }
    Ok(Some(stash))
}

/// Defensive stub used by `flush` when a thread lands in a pod whose
/// `pod.toml` is missing on disk. Phase 3d.i moved the binding-side
/// fields off `ThreadConfig`, so we no longer have the original sandbox
/// spec at this point — the bindings only carry the deterministic id.
/// The stub keeps the thread persistable; the pod itself becomes
/// effectively read-only until someone restores or rewrites pod.toml.
fn synthesize_pod_config(task: &Thread) -> PodConfig {
    // `validate` requires `thread_defaults.backend` to appear in
    // `allow.backends`, so derive the backend name from bindings (or fall
    // back to the literal "default") and seed `[allow]` with just that.
    let backend = if task.bindings.backend.is_empty() {
        "default".to_string()
    } else {
        task.bindings.backend.clone()
    };
    PodConfig {
        name: task.title.clone().unwrap_or_else(|| task.id.clone()),
        description: Some("Stub config — pod.toml was missing on disk".into()),
        created_at: task.created_at.to_rfc3339(),
        allow: PodAllow {
            backends: vec![backend.clone()],
            mcp_hosts: Vec::new(),
            host_env: Vec::new(),
            tools: whisper_agent_protocol::AllowMap::allow_all(),
        },
        thread_defaults: ThreadDefaults {
            backend,
            model: task.config.model.clone(),
            system_prompt_file: "system_prompt.md".into(),
            max_tokens: task.config.max_tokens,
            max_turns: task.config.max_turns,
            host_env: Vec::new(),
            mcp_hosts: Vec::new(),
            compaction: task.config.compaction.clone(),
        },
        limits: PodLimits::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use whisper_agent_protocol::{AllowMap, ThreadBindings, ThreadConfig};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("wa-persist-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn sample_pod_config() -> PodConfig {
        PodConfig {
            name: "sample".into(),
            description: None,
            created_at: "2026-04-17T10:00:00Z".into(),
            allow: PodAllow {
                backends: vec!["anthropic".into()],
                mcp_hosts: Vec::new(),
                host_env: Vec::new(),
                tools: AllowMap::allow_all(),
            },
            thread_defaults: ThreadDefaults {
                backend: "anthropic".into(),
                model: "claude-sonnet-4-6".into(),
                system_prompt_file: "system_prompt.md".into(),
                max_tokens: 8000,
                max_turns: 30,
                host_env: Vec::new(),
                mcp_hosts: Vec::new(),
                compaction: Default::default(),
            },
            limits: PodLimits::default(),
        }
    }

    fn sample_task(id: &str) -> Thread {
        let cfg = ThreadConfig {
            model: "claude-opus-4-7".into(),
            max_tokens: 8000,
            max_turns: 50,
            compaction: Default::default(),
        };
        let bindings = ThreadBindings {
            backend: "anthropic".into(),
            host_env: Vec::new(),
            mcp_hosts: vec![format!("mcp-primary-{id}"), "mcp-shared-fetch".into()],
            tool_filter: None,
        };
        let mut task = Thread::new(id.into(), id.into(), cfg, bindings, AllowMap::allow_all());
        task.title = Some("Sample task".into());
        task.conversation
            .push(whisper_agent_protocol::Message::system_text("Hello."));
        task.conversation
            .push(whisper_agent_protocol::Message::tools_manifest(Vec::new()));
        task
    }

    #[tokio::test]
    async fn flush_then_load_round_trips() {
        let dir = temp_dir();
        let p = Persister::new(dir.clone()).await.unwrap();
        let task = sample_task("t-rt");
        p.flush(&task).await.unwrap();
        // Verify the directory shape we promised on disk.
        assert!(dir.join("t-rt").join(POD_TOML).is_file());
        assert!(dir.join("t-rt").join("system_prompt.md").is_file());
        assert!(
            dir.join("t-rt")
                .join(THREADS_DIR)
                .join("t-rt.json")
                .is_file()
        );
        let loaded = p.load_all().await.unwrap();
        assert_eq!(loaded.threads.len(), 1);
        assert_eq!(loaded.threads[0].id, "t-rt");
        assert_eq!(loaded.threads[0].pod_id, "t-rt");
        assert_eq!(loaded.threads[0].title.as_deref(), Some("Sample task"));
        assert_eq!(loaded.pods.len(), 1);
        assert_eq!(loaded.pods[0].id, "t-rt");
        assert!(loaded.pods[0].threads.contains("t-rt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn pod_toml_not_clobbered_on_subsequent_flush() {
        let dir = temp_dir();
        let p = Persister::new(dir.clone()).await.unwrap();
        let task = sample_task("t-edit");
        p.flush(&task).await.unwrap();
        let pod_toml = dir.join("t-edit").join(POD_TOML);
        // Simulate a hand-edit: append a comment.
        let original = std::fs::read_to_string(&pod_toml).unwrap();
        let edited = format!("{original}\n# hand-edited\n");
        std::fs::write(&pod_toml, &edited).unwrap();
        // Re-flush — pod.toml must keep the edit.
        p.flush(&task).await.unwrap();
        let after = std::fs::read_to_string(&pod_toml).unwrap();
        assert_eq!(after, edited);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn create_pod_writes_system_prompt_when_non_empty() {
        let dir = temp_dir();
        let p = Persister::new(dir.clone()).await.unwrap();
        let cfg = sample_pod_config();
        let prompt_name = cfg.thread_defaults.system_prompt_file.clone();
        p.create_pod("fresh", cfg, "you are helpful").await.unwrap();
        let pod_dir = dir.join("fresh");
        assert!(pod_dir.join(POD_TOML).is_file());
        assert!(pod_dir.join(&prompt_name).is_file());
        let body = std::fs::read_to_string(pod_dir.join(&prompt_name)).unwrap();
        assert_eq!(body, "you are helpful");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn create_pod_skips_prompt_file_when_empty() {
        let dir = temp_dir();
        let p = Persister::new(dir.clone()).await.unwrap();
        let cfg = sample_pod_config();
        let prompt_name = cfg.thread_defaults.system_prompt_file.clone();
        p.create_pod("noprompt", cfg, "").await.unwrap();
        let pod_dir = dir.join("noprompt");
        assert!(pod_dir.join(POD_TOML).is_file());
        assert!(
            !pod_dir.join(&prompt_name).exists(),
            "empty prompt should skip the file"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn legacy_json_files_are_swept() {
        let dir = temp_dir();
        // Write a fake pre-pod-refactor task file at the root.
        std::fs::write(dir.join("legacy-task.json"), b"{\"id\":\"legacy\"}").unwrap();
        let _ = Persister::new(dir.clone()).await.unwrap();
        // The original file is gone from the root.
        assert!(!dir.join("legacy-task.json").exists());
        // It now lives in a .pre-pod-refactor-<ts> directory.
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with(".pre-pod-refactor-")
            })
            .collect();
        assert_eq!(entries.len(), 1, "exactly one stash dir");
        assert!(entries[0].path().join("legacy-task.json").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn legacy_string_host_env_binding_normalized_to_none() {
        // Pre-refactor threads persisted `bindings.host_env` as a bare
        // `he-...` id string. The new shape is an object (or null);
        // load_one should normalize the legacy string to null so the
        // scheduler's load_state can re-bind to the pod's default.
        let dir = temp_dir();
        let p = Persister::new(dir.clone()).await.unwrap();
        // Bootstrap a pod by flushing a normal task, then overwrite
        // the persisted thread JSON with the legacy shape.
        let task = sample_task("legacy-thread");
        p.flush(&task).await.unwrap();
        let json_path = dir
            .join("legacy-thread")
            .join(THREADS_DIR)
            .join("legacy-thread.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&json_path).unwrap()).unwrap();
        value["bindings"]["host_env"] = serde_json::Value::String("he-deadbeefcafef00d".into());
        std::fs::write(&json_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        let loaded = p.load_all().await.unwrap();
        assert_eq!(loaded.threads.len(), 1);
        assert!(
            loaded.threads[0].bindings.host_env.is_empty(),
            "legacy string id should be normalized to empty on load"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn normalize_pod_relative_path_accepts_clean() {
        assert_eq!(normalize_pod_relative_path("").unwrap(), "");
        assert_eq!(normalize_pod_relative_path("a").unwrap(), "a");
        assert_eq!(normalize_pod_relative_path("a/b").unwrap(), "a/b");
        // Strips leading / and redundant separators.
        assert_eq!(normalize_pod_relative_path("/a//b/").unwrap(), "a/b");
        // Explicit `.` components are collapsed.
        assert_eq!(normalize_pod_relative_path("./a/./b").unwrap(), "a/b");
    }

    #[test]
    fn normalize_pod_relative_path_rejects_escapes() {
        assert!(normalize_pod_relative_path("..").is_err());
        assert!(normalize_pod_relative_path("a/../b").is_err());
        assert!(normalize_pod_relative_path("a\\b").is_err());
        assert!(normalize_pod_relative_path("a\0b").is_err());
    }

    #[tokio::test]
    async fn read_pod_file_accepts_text_rejects_binary_and_escape() {
        let dir = temp_dir();
        let p = Persister::new(dir.clone()).await.unwrap();
        let pod_id = "rw-pod";
        let pod_dir = dir.join(pod_id);
        std::fs::create_dir_all(&pod_dir).unwrap();
        std::fs::write(pod_dir.join("pod.toml"), "name = \"x\"\n").unwrap();
        std::fs::write(pod_dir.join("system_prompt.md"), "hello").unwrap();
        std::fs::write(pod_dir.join("blob.bin"), b"\0\0binary\0").unwrap();

        let (text, ro) = p.read_pod_file(pod_id, "system_prompt.md").await.unwrap();
        assert_eq!(text, "hello");
        assert!(!ro);

        let err = p.read_pod_file(pod_id, "blob.bin").await.unwrap_err();
        assert!(err.to_string().contains("binary"));

        // Path traversal is rejected by normalization.
        assert!(p.read_pod_file(pod_id, "../other").await.is_err());
        // Empty path is rejected (the viewer targets individual files,
        // not the pod root).
        assert!(p.read_pod_file(pod_id, "").await.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn write_pod_file_enforces_readonly() {
        let dir = temp_dir();
        let p = Persister::new(dir.clone()).await.unwrap();
        let pod_id = "write-pod";
        let pod_dir = dir.join(pod_id);
        std::fs::create_dir_all(pod_dir.join("threads")).unwrap();
        std::fs::write(pod_dir.join("pod.toml"), "stub = 1\n").unwrap();

        // Creating a brand-new file is fine.
        p.write_pod_file(pod_id, "notes.md", "new content\n")
            .await
            .unwrap();
        let roundtrip = std::fs::read_to_string(pod_dir.join("notes.md")).unwrap();
        assert_eq!(roundtrip, "new content\n");

        // Overwriting an existing editable file works.
        p.write_pod_file(pod_id, "pod.toml", "name = \"edited\"\n")
            .await
            .unwrap();

        // Read-only paths are rejected on write.
        assert!(
            p.write_pod_file(pod_id, "pod_state.json", "{}")
                .await
                .is_err()
        );
        assert!(
            p.write_pod_file(pod_id, "threads/abc.json", "{}")
                .await
                .is_err()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn list_pod_dir_flags_readonly_and_rejects_escape() {
        let dir = temp_dir();
        let p = Persister::new(dir.clone()).await.unwrap();
        let task = sample_task("fs-test");
        p.flush(&task).await.unwrap();

        let root = p.list_pod_dir("fs-test", "").await.unwrap();
        // Dirs come first; `threads` exists after flush.
        assert!(root.iter().any(|e| e.name == "threads" && e.is_dir));
        assert!(root.iter().any(|e| e.name == "pod.toml" && !e.is_dir));

        let threads = p.list_pod_dir("fs-test", "threads").await.unwrap();
        let entry = threads
            .iter()
            .find(|e| e.name == "fs-test.json")
            .expect("thread file");
        assert!(
            entry.readonly,
            "threads/<id>.json should be flagged readonly"
        );

        // `..` escape is rejected.
        assert!(p.list_pod_dir("fs-test", "../other").await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn hidden_dirs_skipped_on_load() {
        let dir = temp_dir();
        let p = Persister::new(dir.clone()).await.unwrap();
        let task = sample_task("real-pod");
        p.flush(&task).await.unwrap();
        // Drop a fake pod-shaped directory under .archived/ — must not load.
        let archived = dir.join(".archived").join("ghost").join(THREADS_DIR);
        std::fs::create_dir_all(&archived).unwrap();
        std::fs::write(
            archived.join("ghost.json"),
            serde_json::to_vec(&task).unwrap(),
        )
        .unwrap();
        let loaded = p.load_all().await.unwrap();
        assert_eq!(loaded.threads.len(), 1);
        assert_eq!(loaded.threads[0].id, "real-pod");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
