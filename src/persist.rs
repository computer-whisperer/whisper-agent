//! Pod-directory persistence: each task lives at
//! `<pods_root>/<pod_id>/threads/<thread_id>.json` alongside a synthesized
//! `<pods_root>/<pod_id>/pod.toml`.
//!
//! Phase 2b lays the on-disk shape down without renaming the in-memory `Thread`
//! type or splitting `TaskConfig`. The `pod.toml` is synthesized from the
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
//!     system_prompt.md                from task.config.system_prompt
//!     threads/
//!       <thread_id>.json
//!   <other-pod>/...
//!   .pre-pod-refactor-<ts>/           stash of legacy `<pods_root>/<task_id>.json`
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

use crate::pod::{self, POD_TOML, THREADS_DIR};
use crate::thread::{Thread, ThreadInternalState};
use whisper_agent_protocol::{
    NamedSandboxSpec, PodAllow, PodConfig, PodLimits, PodSnapshot, PodSummary, TaskSummary,
    ThreadDefaults,
};

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

    /// Write the task's thread JSON. Synthesizes `pod.toml` and
    /// `system_prompt.md` on first flush only — subsequent flushes touch
    /// only the thread file.
    pub async fn flush(&self, task: &Thread) -> Result<()> {
        // Phase 2b shim: one task = one pod = one thread, all sharing the
        // same id. The pod_id/thread_id distinction lives in the directory
        // shape so the rest of the migration only needs to rename, not
        // restructure.
        let pod_id = &task.id;
        let thread_id = &task.id;

        let pod_dir = self.pod_dir(pod_id);
        let threads_dir = pod_dir.join(THREADS_DIR);
        fs::create_dir_all(&threads_dir)
            .await
            .with_context(|| format!("mkdir {}", threads_dir.display()))?;

        let pod_toml = pod_dir.join(POD_TOML);
        if !fs::try_exists(&pod_toml).await.unwrap_or(false) {
            let synthesized = synthesize_pod_config(task);
            let toml_text =
                pod::to_toml(&synthesized).context("encode synthesized pod.toml")?;
            fs::write(&pod_toml, toml_text)
                .await
                .with_context(|| format!("write {}", pod_toml.display()))?;
            // Drop the system prompt next to pod.toml so it's hand-editable.
            // Empty prompts skip the file entirely.
            if !task.config.system_prompt.is_empty() {
                let prompt_path = pod_dir.join(&synthesized.thread_defaults.system_prompt_file);
                fs::write(&prompt_path, &task.config.system_prompt)
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

    /// Walk `<pods_root>/*/threads/*.json` and load every task. Hidden
    /// directories (those whose name starts with `.`) are skipped — the
    /// `.pre-pod-refactor-*/` stash and `.archived/` future archive dir
    /// are both ignored.
    pub async fn load_all(&self) -> Result<Vec<Thread>> {
        let mut tasks = Vec::new();
        let mut entries = match fs::read_dir(&self.pods_root).await {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(tasks),
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
            load_pod_threads(&path, &mut tasks).await;
        }
        info!(count = tasks.len(), "loaded persisted tasks");
        Ok(tasks)
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
        let config = pod::parse_toml(&toml_text).with_context(|| format!("parse {pod_id}/pod.toml"))?;
        let threads = read_thread_summaries(&pod_dir).await;
        Ok(Some(PodSnapshot {
            pod_id: pod_id.to_string(),
            config,
            toml_text,
            threads,
            archived: false,
        }))
    }

    /// Create a fresh pod directory and write `pod.toml` from the supplied
    /// config. Fails if a pod with the same id already exists or if the
    /// config doesn't validate.
    pub async fn create_pod(&self, pod_id: &str, config: PodConfig) -> Result<PodSummary> {
        validate_pod_id(pod_id)?;
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
        Ok(PodSummary {
            pod_id: pod_id.to_string(),
            name: config.name,
            description: config.description,
            created_at: config.created_at,
            thread_count: 0,
            archived: false,
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

/// Reject pod ids containing path separators or other shell-hostile bits.
/// Pod ids become directory names; we don't want `..` traversal or `/`-rooted
/// absolute paths sneaking in via the wire.
fn validate_pod_id(pod_id: &str) -> Result<()> {
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
            Err(_) => (
                pod_id.to_string(),
                None,
                "1970-01-01T00:00:00Z".to_string(),
            ),
        },
        Err(_) => (
            pod_id.to_string(),
            None,
            "1970-01-01T00:00:00Z".to_string(),
        ),
    };
    let thread_count = count_threads(pod_dir).await;
    PodSummary {
        pod_id: pod_id.to_string(),
        name,
        description,
        created_at,
        thread_count,
        archived: false,
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

async fn read_thread_summaries(pod_dir: &Path) -> Vec<TaskSummary> {
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
        match load_one(&path).await {
            Ok(thread) => out.push(thread.summary()),
            Err(e) => warn!(path = %path.display(), error = %e, "skip thread summary"),
        }
    }
    out.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    out
}

async fn load_pod_threads(pod_dir: &Path, out: &mut Vec<Thread>) {
    let threads_dir = pod_dir.join(THREADS_DIR);
    let mut entries = match fs::read_dir(&threads_dir).await {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            warn!(path = %threads_dir.display(), error = %e, "read threads/ failed");
            return;
        }
    };
    while let Some(entry) = match entries.next_entry().await {
        Ok(opt) => opt,
        Err(e) => {
            warn!(path = %threads_dir.display(), error = %e, "iter threads/ failed");
            return;
        }
    } {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match load_one(&path).await {
            Ok(task) => out.push(task),
            Err(e) => warn!(path = %path.display(), error = %e, "skip unreadable thread file"),
        }
    }
}

async fn load_one(path: &Path) -> Result<Thread> {
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    let mut task: Thread =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    if is_in_flight(&task.internal) {
        task.fail("resume", "task was in-flight at last shutdown");
    }
    Ok(task)
}

fn is_in_flight(state: &ThreadInternalState) -> bool {
    matches!(
        state,
        ThreadInternalState::NeedsMcpConnect
            | ThreadInternalState::AwaitingMcpConnect { .. }
            | ThreadInternalState::NeedsListTools
            | ThreadInternalState::AwaitingListTools { .. }
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

fn synthesize_pod_config(task: &Thread) -> PodConfig {
    // The synthesized config describes the task's bindings as if pods were
    // already authoritative. Backend may be empty (the existing TaskConfig
    // accepts that as "use server default"); we mirror that — validation
    // accepts a self-consistent config even with empty strings.
    let backend = task.config.backend.clone();
    let sandbox_name = "primary".to_string();
    PodConfig {
        name: task.title.clone().unwrap_or_else(|| task.id.clone()),
        description: None,
        created_at: task.created_at.to_rfc3339(),
        allow: PodAllow {
            backends: vec![backend.clone()],
            mcp_hosts: task.config.shared_mcp_hosts.clone(),
            sandbox: vec![NamedSandboxSpec {
                name: sandbox_name.clone(),
                spec: task.config.sandbox.clone(),
            }],
        },
        thread_defaults: ThreadDefaults {
            backend,
            model: task.config.model.clone(),
            system_prompt_file: "system_prompt.md".into(),
            max_tokens: task.config.max_tokens,
            max_turns: task.config.max_turns,
            approval_policy: task.config.approval_policy,
            sandbox: sandbox_name,
            mcp_hosts: task.config.shared_mcp_hosts.clone(),
        },
        limits: PodLimits::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use whisper_agent_protocol::{ApprovalPolicy, SandboxSpec, TaskConfig};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "wa-persist-test-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn sample_task(id: &str) -> Thread {
        let cfg = TaskConfig {
            backend: "anthropic".into(),
            model: "claude-opus-4-7".into(),
            system_prompt: "Hello.".into(),
            mcp_host_url: "http://test".into(),
            max_tokens: 8000,
            max_turns: 50,
            approval_policy: ApprovalPolicy::PromptDestructive,
            sandbox: SandboxSpec::None,
            shared_mcp_hosts: vec!["fetch".into()],
        };
        let mut task = Thread::new(id.into(), cfg);
        task.title = Some("Sample task".into());
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
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "t-rt");
        assert_eq!(loaded[0].title.as_deref(), Some("Sample task"));
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
            .filter(|e| e.file_name().to_string_lossy().starts_with(".pre-pod-refactor-"))
            .collect();
        assert_eq!(entries.len(), 1, "exactly one stash dir");
        assert!(
            entries[0]
                .path()
                .join("legacy-task.json")
                .is_file()
        );
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
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "real-pod");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
