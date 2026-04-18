//! In-process tools that let an agent edit its own pod's configuration.
//!
//! Mirrors mcp-host's `read_file` / `write_file` / `edit_file` shape but
//! scoped to a whitelist of filenames derived from the pod's config:
//!
//!   * `pod.toml` — always allowed. Writes are parsed + validated by
//!     [`crate::pod::parse_toml`] before landing on disk; on validation
//!     failure the file is untouched and the tool returns an error.
//!   * The file named by `thread_defaults.system_prompt_file` — allowed
//!     whenever non-empty. No structural validation (plain text).
//!   * `behaviors/<id>/behavior.toml` — parsed + validated by
//!     [`crate::pod::behaviors::parse_toml`]. Writing to a new `<id>`
//!     creates the behavior directory and seeds `state.json` with a
//!     default `BehaviorState`.
//!   * `behaviors/<id>/prompt.md` — plain text; the initial user-message
//!     template fired threads run against. Only writable for ids that
//!     already have a `behavior.toml` (the agent must create the
//!     behavior first).
//!
//! Successful writes emit a [`PodUpdate`] so the scheduler can refresh
//! in-memory pod state and broadcast the change to subscribers in the
//! same step as the disk write. See
//! [`crate::runtime::scheduler::Scheduler::apply_pod_config_update`] and
//! [`crate::runtime::scheduler::Scheduler::apply_behavior_config_update`].

mod about;
mod behavior_control;
mod grep;
mod list_threads;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};
use whisper_agent_protocol::{BehaviorConfig, BehaviorState, PodConfig};

use crate::pod;
use crate::pod::behaviors::{BEHAVIOR_PROMPT, BEHAVIOR_STATE, BEHAVIOR_TOML, BEHAVIORS_DIR};
use crate::tools::mcp::{
    CallToolResult, McpContentBlock, ToolAnnotations, ToolDescriptor as McpTool,
};

fn text_result(text: String) -> CallToolResult {
    CallToolResult {
        content: vec![McpContentBlock::Text { text }],
        is_error: false,
    }
}

fn error_result(text: String) -> CallToolResult {
    CallToolResult {
        content: vec![McpContentBlock::Text { text }],
        is_error: true,
    }
}

// Test fixtures live at file level (gated on `cfg(test)`) so every
// submodule's `mod tests` can reach them via `super::temp_dir()` etc.
// without duplication or a pub(super) test_support module.
#[cfg(test)]
fn temp_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("wa-builtin-tools-test-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[cfg(test)]
fn sample_config() -> PodConfig {
    let text = r#"
name = "t"
created_at = "2026-04-17T00:00:00Z"
[allow]
backends = ["anthropic"]
mcp_hosts = []
host_env = []
[thread_defaults]
backend = "anthropic"
model = "claude-opus-4-7"
system_prompt_file = "system_prompt.md"
max_tokens = 8000
max_turns = 50
"#;
    pod::parse_toml(text).unwrap()
}

#[cfg(test)]
fn join_blocks(blocks: &[McpContentBlock]) -> String {
    let mut s = String::new();
    for b in blocks {
        let McpContentBlock::Text { text } = b;
        s.push_str(text);
    }
    s
}

pub const POD_READ_FILE: &str = "pod_read_file";
pub const POD_WRITE_FILE: &str = "pod_write_file";
pub const POD_EDIT_FILE: &str = "pod_edit_file";
pub const POD_LIST_FILES: &str = "pod_list_files";
pub const POD_GREP: &str = "pod_grep";
pub const POD_LIST_THREADS: &str = "pod_list_threads";
pub const POD_ABOUT: &str = "pod_about";
pub const POD_RUN_BEHAVIOR: &str = "pod_run_behavior";
pub const POD_SET_BEHAVIOR_ENABLED: &str = "pod_set_behavior_enabled";

/// True if `name` is a builtin pod tool. Used by the scheduler's router
/// to branch the tool-call dispatch path.
pub fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        POD_READ_FILE
            | POD_WRITE_FILE
            | POD_EDIT_FILE
            | POD_LIST_FILES
            | POD_GREP
            | POD_LIST_THREADS
            | POD_ABOUT
            | POD_RUN_BEHAVIOR
            | POD_SET_BEHAVIOR_ENABLED
    )
}

/// True if `name` is a builtin tool that modifies the pod. Used by
/// `evaluate_policy` so `ApprovalPolicy::PromptPodModify` can gate these
/// specifically without prompting on every MCP tool.
pub fn is_pod_modify(name: &str) -> bool {
    matches!(name, POD_WRITE_FILE | POD_EDIT_FILE)
}

/// Tool descriptors for the builtin set, shaped for direct inclusion in
/// the model's advertised catalog.
pub fn descriptors() -> Vec<McpTool> {
    vec![
        list_descriptor(),
        read_descriptor(),
        write_descriptor(),
        edit_descriptor(),
        grep::descriptor(),
        list_threads::descriptor(),
        about::descriptor(),
        behavior_control::run_behavior_descriptor(),
        behavior_control::set_behavior_enabled_descriptor(),
    ]
}

/// Server-declared annotations for each builtin tool. Keyed by tool name
/// so `annotations_for` can merge them into the thread's overall
/// annotation map alongside MCP-sourced annotations.
pub fn annotations() -> HashMap<String, ToolAnnotations> {
    descriptors()
        .into_iter()
        .map(|t| (t.name, t.annotations))
        .collect()
}

/// Side effect to apply to in-memory pod state when a write succeeds.
/// Emitted by the tool handler and consumed by the scheduler.
#[derive(Debug, Clone)]
pub enum PodUpdate {
    /// A new `pod.toml` that has already been parsed and validated.
    Config {
        toml_text: String,
        parsed: Box<PodConfig>,
    },
    /// New system-prompt text.
    SystemPrompt { text: String },
    /// A new or updated `behaviors/<id>/behavior.toml`. Parsed and
    /// validated before emission. The scheduler decides whether this is
    /// a create or an update by looking at its in-memory pod state.
    BehaviorConfig {
        behavior_id: String,
        toml_text: String,
        parsed: Box<BehaviorConfig>,
    },
    /// New text for a behavior's `prompt.md`. Only valid when the
    /// behavior already exists in-memory (see module docs); the tool
    /// handler rejects prompt writes for unknown ids.
    BehaviorPrompt { behavior_id: String, text: String },
}

/// Runtime action a builtin tool asks the scheduler to perform AFTER
/// returning the tool result to the model. These bypass disk writes —
/// they're for flipping runtime state (pause/resume) or kicking a
/// behavior run. The scheduler applies one of these in the same loop
/// iteration that consumes the tool result, keeping broadcasts and the
/// ToolCallEnd in the same well-defined order.
#[derive(Debug, Clone)]
pub enum SchedulerCommand {
    /// Manually fire a behavior. Equivalent to the UI's Run button:
    /// bypasses the cron/paused gates (it's an explicit action). The
    /// optional payload is substituted into `prompt.md`'s `{{payload}}`
    /// the same way a webhook trigger's body is.
    RunBehavior {
        behavior_id: String,
        payload: Option<serde_json::Value>,
    },
    /// Pause or resume a single behavior. Wraps the scheduler's
    /// `SetBehaviorEnabled` handler — on pause drops any queued
    /// payload, on resume bumps the cron cursor to now so catch-up
    /// doesn't replay missed windows.
    SetBehaviorEnabled { behavior_id: String, enabled: bool },
}

/// What a builtin tool handler produces: the tool result the model sees,
/// plus optional state/command side effects the scheduler applies
/// atomically in the same completion step.
pub struct ToolOutcome {
    pub result: CallToolResult,
    pub pod_update: Option<PodUpdate>,
    pub scheduler_command: Option<SchedulerCommand>,
}

/// Dispatch a builtin tool call. `pod_dir` is the pod's on-disk directory;
/// `pod_config` is the current in-memory config; `behavior_ids` lists
/// the behaviors currently registered (used to derive the dynamic slice
/// of the allowlist and to distinguish create vs update on write).
pub async fn dispatch(
    pod_dir: PathBuf,
    pod_config: PodConfig,
    behavior_ids: Vec<String>,
    tool_name: &str,
    args: Value,
) -> ToolOutcome {
    let allowed = allowed_filenames(&pod_config, &behavior_ids);
    match tool_name {
        POD_LIST_FILES => list_files(&pod_dir, &allowed, args).await,
        POD_READ_FILE => read_file(&pod_dir, &allowed, &behavior_ids, args).await,
        POD_WRITE_FILE => write_file(&pod_dir, &allowed, &behavior_ids, args).await,
        POD_EDIT_FILE => edit_file(&pod_dir, &allowed, &behavior_ids, args).await,
        POD_GREP => grep::run(&pod_dir, args).await,
        POD_LIST_THREADS => list_threads::run(&pod_dir, args).await,
        POD_ABOUT => about::run(args),
        POD_RUN_BEHAVIOR => behavior_control::run_behavior(&behavior_ids, args),
        POD_SET_BEHAVIOR_ENABLED => behavior_control::set_behavior_enabled(&behavior_ids, args),
        other => no_update_error(format!("unknown builtin tool: {other}")),
    }
}

/// Every filename reachable by `pod_read_file` / `pod_write_file` /
/// `pod_edit_file`. Built from the current pod config plus every
/// behavior id in memory. Writes to `behaviors/<new-id>/behavior.toml`
/// are additionally allowed (for create-new) — see [`validate_filename`].
fn allowed_filenames(cfg: &PodConfig, behavior_ids: &[String]) -> Vec<String> {
    let mut out = vec![pod::POD_TOML.to_string()];
    let prompt = &cfg.thread_defaults.system_prompt_file;
    if !prompt.is_empty() && prompt != pod::POD_TOML {
        out.push(prompt.clone());
    }
    for id in behavior_ids {
        out.push(format!("{BEHAVIORS_DIR}/{id}/{BEHAVIOR_TOML}"));
        out.push(format!("{BEHAVIORS_DIR}/{id}/{BEHAVIOR_PROMPT}"));
    }
    out
}

/// Split `behaviors/<id>/<suffix>` into `(id, suffix)` when `filename`
/// matches that shape. The id and suffix must each be a single path
/// component (no nested subdirs).
fn parse_behavior_path(filename: &str) -> Option<(&str, &str)> {
    let rest = filename.strip_prefix(&format!("{BEHAVIORS_DIR}/"))?;
    let (id, suffix) = rest.split_once('/')?;
    if id.is_empty() || suffix.is_empty() || suffix.contains('/') {
        return None;
    }
    Some((id, suffix))
}

/// Return the thread id for a path like `threads/<id>.json`, or None.
fn parse_thread_path(filename: &str) -> Option<&str> {
    let rest = filename.strip_prefix(&format!("{}/", pod::THREADS_DIR))?;
    if rest.contains('/') || rest.starts_with('.') {
        return None;
    }
    let id = rest.strip_suffix(".json")?;
    if id.is_empty() {
        return None;
    }
    Some(id)
}

/// Pattern-based check: a file that is readable but NOT writable via
/// the pod_*_file tools. Runtime / observability data — thread JSONs,
/// pod_state.json, per-behavior state.json. These aren't in the rw
/// allowlist because they encode runtime state the scheduler owns;
/// letting the agent edit them would bypass that ownership.
fn is_readonly_path(filename: &str) -> bool {
    if filename == pod::POD_STATE_JSON {
        return true;
    }
    if parse_thread_path(filename).is_some() {
        return true;
    }
    if let Some((id, suffix)) = parse_behavior_path(filename)
        && suffix == crate::pod::behaviors::BEHAVIOR_STATE
        && pod::behaviors::validate_behavior_id(id).is_ok()
    {
        return true;
    }
    false
}

/// Access level a filename resolves to under the current pod snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    /// Reachable via pod_read_file / pod_write_file / pod_edit_file.
    Rw,
    /// Reachable via pod_read_file only.
    Ro,
    /// Not reachable.
    None,
}

/// Classify `filename` for list output. Purely pattern-based on the
/// existing rw allowlist plus `is_readonly_path` — does NOT touch disk.
fn access_for(filename: &str, rw_allowed: &[String]) -> Access {
    if rw_allowed.iter().any(|n| n == filename) {
        Access::Rw
    } else if is_readonly_path(filename) {
        Access::Ro
    } else {
        Access::None
    }
}

/// Kind of write a behavior path resolves to — drives whether the tool
/// handler creates the directory + seeds state.json, and whether the
/// scheduler emits `BehaviorCreated` or `BehaviorUpdated`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BehaviorWrite {
    /// `behavior.toml` for an id not yet in memory — triggers mkdir and
    /// state.json seeding.
    CreateConfig,
    /// `behavior.toml` for an existing id — content replace only.
    UpdateConfig,
    /// `prompt.md` for an existing id — content replace only.
    UpdatePrompt,
}

/// Classify a `behaviors/<id>/<suffix>` write.  Returns `Err` with a
/// helpful message when the write would produce an incoherent state
/// (prompt-before-toml on a new id, or an unknown suffix).
fn classify_behavior_write(
    id: &str,
    suffix: &str,
    behavior_ids: &[String],
) -> Result<BehaviorWrite, String> {
    let exists = behavior_ids.iter().any(|i| i == id);
    match (suffix, exists) {
        (BEHAVIOR_TOML, false) => Ok(BehaviorWrite::CreateConfig),
        (BEHAVIOR_TOML, true) => Ok(BehaviorWrite::UpdateConfig),
        (BEHAVIOR_PROMPT, true) => Ok(BehaviorWrite::UpdatePrompt),
        (BEHAVIOR_PROMPT, false) => Err(format!(
            "behavior `{id}` does not exist yet — create it first by writing \
             `{BEHAVIORS_DIR}/{id}/{BEHAVIOR_TOML}` before its `{BEHAVIOR_PROMPT}`"
        )),
        (other, _) => Err(format!(
            "unknown behavior filename suffix `{other}` for behavior `{id}`"
        )),
    }
}

fn validate_filename(
    filename: &str,
    allowed: &[String],
    behavior_ids: &[String],
    for_write: bool,
) -> Result<(), String> {
    if allowed.iter().any(|n| n == filename) {
        return Ok(());
    }
    // Create-new: writing `behaviors/<new-id>/behavior.toml` is
    // legitimate even though the id is absent from `allowed`. Reads /
    // edits against an unknown id fall through to the error below.
    if for_write && let Some((id, suffix)) = parse_behavior_path(filename) {
        if let Err(e) = pod::behaviors::validate_behavior_id(id) {
            return Err(format!("behavior id `{id}`: {e}"));
        }
        return classify_behavior_write(id, suffix, behavior_ids).map(|_| ());
    }
    // Read-only: thread JSONs, pod_state.json, per-behavior state.json.
    // Writes to these paths fall through to the error message — the
    // handler's caller won't misdirect a user.
    if !for_write && is_readonly_path(filename) {
        return Ok(());
    }
    if for_write && is_readonly_path(filename) {
        return Err(format!(
            "filename `{filename}` is read-only (runtime state owned by \
             the scheduler). Use `pod_set_behavior_enabled` / \
             `pod_run_behavior` for runtime transitions."
        ));
    }
    Err(format!(
        "filename `{filename}` is not in this pod's allowlist. Allowed rw: [{}]. \
         Read-only patterns: `pod_state.json`, `{}/<id>.json`, `{BEHAVIORS_DIR}/<id>/{}`.",
        allowed.join(", "),
        pod::THREADS_DIR,
        crate::pod::behaviors::BEHAVIOR_STATE,
    ))
}

fn no_update_error(message: String) -> ToolOutcome {
    ToolOutcome {
        result: error_result(message),
        pod_update: None,
        scheduler_command: None,
    }
}

fn no_update_text(message: String) -> ToolOutcome {
    ToolOutcome {
        result: text_result(message),
        pod_update: None,
        scheduler_command: None,
    }
}

// ---------- list ----------

fn list_descriptor() -> McpTool {
    McpTool {
        name: POD_LIST_FILES.into(),
        description: "List the contents of this pod's directory, including its \
                      `behaviors/<id>/` subdirectories and a summary of the \
                      `threads/` dir. Entries are tagged by access: `[rw]` = \
                      readable and writable (the rw config files), `[r-]` = \
                      read-only (runtime state: `pod_state.json`, per-behavior \
                      `state.json`, `threads/<id>.json`), `[--]` = not reachable. \
                      Pod files live OUTSIDE your workspace and are reachable \
                      ONLY through the `pod_*_file` tools — not `list_dir` or \
                      `bash`. Call this before editing to see which config files \
                      and behaviors exist."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {}
        }),
        annotations: ToolAnnotations {
            title: Some("List pod directory".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

/// Maximum number of thread entries listed individually before
/// `list_files` falls back to a count-only summary. Pods with a long
/// run history would otherwise blow the tool output size.
const THREADS_LIST_CAP: usize = 30;

async fn list_files(pod_dir: &Path, allowed: &[String], _args: Value) -> ToolOutcome {
    let mut out = String::new();
    if let Err(e) = list_one_dir(pod_dir, "", allowed, &mut out).await {
        return no_update_error(e);
    }

    // Each known behavior gets its own nested section — the agent sees
    // `behavior.toml`, `prompt.md`, and the read-only `state.json` together.
    let behaviors_dir = pod_dir.join(BEHAVIORS_DIR);
    if tokio::fs::try_exists(&behaviors_dir).await.unwrap_or(false) {
        let mut read = match tokio::fs::read_dir(&behaviors_dir).await {
            Ok(r) => r,
            Err(e) => return no_update_error(format!("list {BEHAVIORS_DIR}/: {e}")),
        };
        let mut ids: Vec<String> = Vec::new();
        while let Ok(Some(entry)) = read.next_entry().await {
            if let Ok(ft) = entry.file_type().await
                && ft.is_dir()
                && let Some(name) = entry.file_name().to_str()
                && !name.starts_with('.')
            {
                ids.push(name.to_string());
            }
        }
        ids.sort();
        for id in ids {
            let prefix = format!("{BEHAVIORS_DIR}/{id}/");
            out.push_str(&format!("\n-- {prefix} --\n"));
            if let Err(e) = list_one_dir(&behaviors_dir.join(&id), &prefix, allowed, &mut out).await
            {
                return no_update_error(e);
            }
        }
    }

    // `threads/` section: show the N most-recently-modified entries.
    // For full filtering (by behavior, state, turns, since, ...) the
    // agent should reach for `pod_list_threads` — this is just a
    // presence-check so the listing is useful in its own right.
    let threads_dir = pod_dir.join(pod::THREADS_DIR);
    if tokio::fs::try_exists(&threads_dir).await.unwrap_or(false) {
        let mut read = match tokio::fs::read_dir(&threads_dir).await {
            Ok(r) => r,
            Err(e) => return no_update_error(format!("list {}/: {e}", pod::THREADS_DIR)),
        };
        let mut entries: Vec<(String, u64, std::time::SystemTime)> = Vec::new();
        while let Ok(Some(entry)) = read.next_entry().await {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || !name.ends_with(".json") {
                continue;
            }
            let meta = match entry.metadata().await {
                Ok(m) => m,
                Err(_) => continue,
            };
            let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            entries.push((name, meta.len(), mtime));
        }
        entries.sort_by_key(|e| std::cmp::Reverse(e.2)); // newest first
        let total = entries.len();
        out.push_str(&format!(
            "\n-- {}/ (read-only; {} thread file{}; newest first) --\n",
            pod::THREADS_DIR,
            total,
            if total == 1 { "" } else { "s" }
        ));
        let show = total.min(THREADS_LIST_CAP);
        for (name, size, _) in entries.iter().take(show) {
            out.push_str(&format!("[r-]  f  {size:>10}  {name}\n"));
        }
        if total > show {
            out.push_str(&format!(
                "...and {} more; use `pod_list_threads` to filter by behavior / state / recency\n",
                total - show,
            ));
        }
    }

    // Mention whitelisted-but-absent top-level filenames so the agent
    // knows a write will create them. Behavior files are covered by the
    // per-id section above, so skip any entry with a path separator.
    let absent: Vec<&String> = allowed
        .iter()
        .filter(|a| !a.contains('/'))
        .filter(|a| {
            let rel_path = pod_dir.join(a);
            !rel_path.exists()
        })
        .collect();
    if !absent.is_empty() {
        out.push_str(
            "\nTop-level allowlist entries not yet present (pod_write_file will create):\n",
        );
        for name in absent {
            out.push_str(&format!("  {name}\n"));
        }
    }
    out.push_str(&format!(
        "\nCreate a new behavior with:\n  pod_write_file(\"{BEHAVIORS_DIR}/<new-id>/{BEHAVIOR_TOML}\", ...)\n  pod_write_file(\"{BEHAVIORS_DIR}/<new-id>/{BEHAVIOR_PROMPT}\", ...)\n"
    ));

    no_update_text(out)
}

/// List the immediate contents of `dir`, each entry prefixed with
/// `path_prefix` when checking `allowed` and when printing. Entries are
/// sorted: allowed files first, then directories, then other files —
/// each group name-sorted. Skips hidden (dot) entries.
async fn list_one_dir(
    dir: &Path,
    path_prefix: &str,
    allowed: &[String],
    out: &mut String,
) -> Result<(), String> {
    let mut read = tokio::fs::read_dir(dir)
        .await
        .map_err(|e| format!("list {}: {e}", dir.display()))?;
    let mut entries: Vec<(String, bool, u64)> = Vec::new();
    loop {
        match read.next_entry().await {
            Ok(None) => break,
            Ok(Some(entry)) => {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with('.') {
                    continue;
                }
                let (is_dir, size) = match entry.file_type().await {
                    Ok(ft) if ft.is_dir() => (true, 0),
                    Ok(_) => {
                        let sz = entry.metadata().await.map(|m| m.len()).unwrap_or(0);
                        (false, sz)
                    }
                    Err(_) => (false, 0),
                };
                entries.push((name, is_dir, size));
            }
            Err(e) => return Err(format!("list {} iter: {e}", dir.display())),
        }
    }
    // Sort: rw files first, then ro files, then dirs, then unreachable.
    // Within each group, name-sorted.
    entries.sort_by(|a, b| {
        let a_full = format!("{path_prefix}{}", a.0);
        let b_full = format!("{path_prefix}{}", b.0);
        let a_rank = access_rank(access_for(&a_full, allowed), a.1);
        let b_rank = access_rank(access_for(&b_full, allowed), b.1);
        a_rank.cmp(&b_rank).then_with(|| a.0.cmp(&b.0))
    });
    for (name, is_dir, size) in &entries {
        let full = format!("{path_prefix}{name}");
        let tag = match access_for(&full, allowed) {
            Access::Rw => "[rw]",
            Access::Ro => "[r-]",
            Access::None => "[--]",
        };
        let kind = if *is_dir { 'd' } else { 'f' };
        let display = if *is_dir {
            format!("{name}/")
        } else {
            name.clone()
        };
        if *is_dir {
            out.push_str(&format!("{tag}  {kind}  {:>10}  {}\n", "", display));
        } else {
            out.push_str(&format!("{tag}  {kind}  {size:>10}  {}\n", display));
        }
    }
    Ok(())
}

/// Small integer used to sort list entries. Lower ranks print first.
fn access_rank(access: Access, is_dir: bool) -> u8 {
    match (access, is_dir) {
        (Access::Rw, _) => 0,
        (Access::Ro, _) => 1,
        (Access::None, true) => 2,
        (Access::None, false) => 3,
    }
}

// ---------- read ----------

fn read_descriptor() -> McpTool {
    McpTool {
        name: POD_READ_FILE.into(),
        description: "Read one of this pod's files: any rw config (`pod.toml`, \
                      system-prompt file, `behaviors/<id>/behavior.toml` or \
                      `prompt.md`) or any read-only runtime file (`pod_state.json`, \
                      `behaviors/<id>/state.json`, `threads/<id>.json`). Thread \
                      JSONs carry the full message history of a run that spawned \
                      from this pod. These files live OUTSIDE your workspace — \
                      reach them ONLY through the `pod_*_file` tools, not \
                      `read_file`/`list_dir`/`bash`. Without `offset`/`limit`, \
                      returns the whole file if it fits within 500 lines; longer \
                      files (typical for thread JSONs) are auto-truncated to the \
                      first 500 lines with a note. Use `offset`/`limit` to page \
                      or target a specific range. For searching across many \
                      files, `pod_grep` is usually a better starting point."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "File to read. Must be one of this pod's \
                                    allowed filenames (the error message lists them)."
                },
                "offset": {
                    "type": "integer",
                    "description": "1-indexed line to start from. Default 1."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return. Default: no limit."
                }
            },
            "required": ["filename"]
        }),
        annotations: ToolAnnotations {
            title: Some("Read pod config file".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize)]
struct ReadArgs {
    filename: String,
    #[serde(default)]
    offset: Option<u32>,
    #[serde(default)]
    limit: Option<u32>,
}

/// Default line cap when the caller supplies neither `offset` nor
/// `limit` — keeps `pod_read_file("threads/abc.json")` on a multi-
/// thousand-line thread from filling the agent's context before it
/// knew to ask for a range.
const READ_DEFAULT_LINE_CAP: usize = 500;

async fn read_file(
    pod_dir: &Path,
    allowed: &[String],
    behavior_ids: &[String],
    args: Value,
) -> ToolOutcome {
    let parsed: ReadArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    if let Err(e) = validate_filename(&parsed.filename, allowed, behavior_ids, false) {
        return no_update_error(e);
    }
    let path = pod_dir.join(&parsed.filename);
    let content = match tokio::fs::read_to_string(&path).await {
        Ok(s) => s,
        Err(e) => return no_update_error(format!("read {}: {e}", parsed.filename)),
    };
    let offset_supplied = parsed.offset.is_some();
    let limit_supplied = parsed.limit.is_some();
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    // Small file and no explicit slicing → whole file, preserve original
    // text (trailing newline is lost by `.lines()`).
    if !offset_supplied && !limit_supplied && total <= READ_DEFAULT_LINE_CAP {
        return no_update_text(content);
    }

    let offset = parsed.offset.unwrap_or(1).max(1) as usize;
    // When neither offset nor limit is supplied on a large file, cap
    // at READ_DEFAULT_LINE_CAP. When the caller supplied `limit`,
    // honor it. When the caller supplied only `offset`, cap from
    // there — avoids a surprise when paging past the default.
    let limit = parsed
        .limit
        .map(|n| n as usize)
        .unwrap_or(READ_DEFAULT_LINE_CAP);
    let start = (offset - 1).min(total);
    let end = start.saturating_add(limit).min(total);
    let mut out = lines[start..end].join("\n");
    if end == total && content.ends_with('\n') {
        out.push('\n');
    }
    if end < total {
        let note = if !offset_supplied && !limit_supplied {
            format!(
                "\n[showing first {READ_DEFAULT_LINE_CAP} of {total} lines; pass offset/limit to see more]\n"
            )
        } else {
            format!("\n[showing lines {offset}-{end} of {total}; pass offset/limit to see more]\n")
        };
        out.push_str(&note);
    }
    no_update_text(out)
}

// ---------- write ----------

fn write_descriptor() -> McpTool {
    McpTool {
        name: POD_WRITE_FILE.into(),
        description: "Fully overwrite (or create) one of this pod's configuration \
                      files: `pod.toml`, the pod-level system-prompt file, or any \
                      `behaviors/<id>/behavior.toml` / `behaviors/<id>/prompt.md`. \
                      Writing `behaviors/<new-id>/behavior.toml` creates a new \
                      behavior — the directory is mkdir'd and `state.json` is seeded \
                      with defaults. A behavior's `prompt.md` may only be written \
                      after its `behavior.toml` exists. These files live OUTSIDE \
                      your workspace and are reachable ONLY through the `pod_*_file` \
                      tools — not `write_file` or `bash`. Structured files \
                      (`pod.toml`, `behavior.toml`) are parsed and validated before \
                      landing on disk; invalid TOML or schema violations return a \
                      tool error and the on-disk file is untouched. Prefer \
                      `pod_edit_file` for targeted changes — this is for full \
                      rewrites and creates."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "File to rewrite. Must be in this pod's allowlist."
                },
                "content": {
                    "type": "string",
                    "description": "New contents."
                }
            },
            "required": ["filename", "content"]
        }),
        annotations: ToolAnnotations {
            title: Some("Write pod config file".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(true),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize)]
struct WriteArgs {
    filename: String,
    content: String,
}

async fn write_file(
    pod_dir: &Path,
    allowed: &[String],
    behavior_ids: &[String],
    args: Value,
) -> ToolOutcome {
    let parsed: WriteArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    if let Err(e) = validate_filename(&parsed.filename, allowed, behavior_ids, true) {
        return no_update_error(e);
    }
    let update = match prepare_update(&parsed.filename, &parsed.content) {
        Ok(u) => u,
        Err(e) => return no_update_error(e),
    };
    if let Err(e) = prepare_behavior_dir(pod_dir, &parsed.filename, behavior_ids).await {
        return no_update_error(e);
    }
    let path = pod_dir.join(&parsed.filename);
    if let Err(e) = tokio::fs::write(&path, parsed.content.as_bytes()).await {
        return no_update_error(format!("write {}: {e}", parsed.filename));
    }
    if let Err(e) = seed_behavior_state_if_needed(pod_dir, &update).await {
        return no_update_error(e);
    }
    ToolOutcome {
        result: text_result(format!(
            "wrote {} bytes to {}",
            parsed.content.len(),
            parsed.filename
        )),
        pod_update: Some(update),
        scheduler_command: None,
    }
}

// ---------- edit ----------

fn edit_descriptor() -> McpTool {
    McpTool {
        name: POD_EDIT_FILE.into(),
        description: "Replace a literal substring in one of this pod's config files \
                      (`pod.toml`, system-prompt file, or any existing \
                      `behaviors/<id>/behavior.toml` / `behaviors/<id>/prompt.md`). \
                      Use `pod_write_file` to create a new behavior — this tool \
                      requires the target file to already exist. These files live \
                      OUTSIDE your workspace and are reachable ONLY through the \
                      `pod_*_file` tools — not `edit_file` or `bash`. `old_string` \
                      must match literally (no regex) and by default must occur \
                      exactly once; pass `replace_all: true` to change every \
                      occurrence. If old_string isn't found, the error shows the \
                      closest matching region; if it matches multiple times, the \
                      error lists each match site with line numbers. When extending \
                      `old_string` with surrounding context to disambiguate a \
                      multi-match, extend `new_string` by the same neighboring text \
                      — extending only `old_string` will delete the intervening \
                      lines from the output. For structured files (`pod.toml`, \
                      `behavior.toml`) the resulting text is parsed and validated \
                      before landing on disk."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "File to edit. Must be in this pod's allowlist."
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact substring to find. Must match literally, \
                                    including whitespace and indentation."
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement. May be empty to delete the matched span."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace every occurrence. If false (default), \
                                    require exactly one match."
                }
            },
            "required": ["filename", "old_string", "new_string"]
        }),
        annotations: ToolAnnotations {
            title: Some("Edit pod config file".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(true),
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize)]
struct EditArgs {
    filename: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

async fn edit_file(
    pod_dir: &Path,
    allowed: &[String],
    behavior_ids: &[String],
    args: Value,
) -> ToolOutcome {
    let parsed: EditArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    // Edits target an existing file — always treat as non-create for
    // filename validation so a path like `behaviors/<new-id>/prompt.md`
    // is rejected with the right error.
    if let Err(e) = validate_filename(&parsed.filename, allowed, behavior_ids, false) {
        return no_update_error(e);
    }
    if parsed.old_string.is_empty() {
        return no_update_error(
            "pod_edit_file: old_string must be non-empty (use pod_write_file to create a new file)"
                .into(),
        );
    }
    if parsed.old_string == parsed.new_string {
        return no_update_error(
            "pod_edit_file: old_string and new_string are identical — no-op".into(),
        );
    }
    let path = pod_dir.join(&parsed.filename);
    let content = match tokio::fs::read_to_string(&path).await {
        Ok(s) => s,
        Err(e) => return no_update_error(format!("read {}: {e}", parsed.filename)),
    };
    let found = content.matches(&parsed.old_string).count();
    if found == 0 {
        let hint = nearest_match_hint(&content, &parsed.old_string);
        return no_update_error(format!(
            "pod_edit_file({}): old_string not found.{hint}",
            parsed.filename
        ));
    }
    if found > 1 && !parsed.replace_all {
        let hint = multi_match_hint(&content, &parsed.old_string);
        return no_update_error(format!(
            "pod_edit_file({}): old_string matches {found} times. Either extend old_string \
             AND new_string with surrounding context to make the target unique (extend both \
             by the same text — extending only old_string will delete the intervening lines), \
             or pass replace_all:true to replace every occurrence.{hint}",
            parsed.filename
        ));
    }
    let new_content = content.replace(&parsed.old_string, &parsed.new_string);
    let update = match prepare_update(&parsed.filename, &new_content) {
        Ok(u) => u,
        Err(e) => return no_update_error(e),
    };
    if let Err(e) = tokio::fs::write(&path, new_content.as_bytes()).await {
        return no_update_error(format!("write {}: {e}", parsed.filename));
    }
    ToolOutcome {
        result: text_result(format!(
            "pod_edit_file({}): replaced {found} occurrence{}",
            parsed.filename,
            if found == 1 { "" } else { "s" }
        )),
        pod_update: Some(update),
        scheduler_command: None,
    }
}

/// Decide what kind of [`PodUpdate`] a successful write produces.
/// For structured files (pod.toml, behavior.toml) we parse + validate
/// here — failure becomes a tool error BEFORE any disk write, so partial
/// state is never reached.
fn prepare_update(filename: &str, content: &str) -> Result<PodUpdate, String> {
    if filename == pod::POD_TOML {
        let parsed = pod::parse_toml(content).map_err(|e| format!("pod.toml: {e}"))?;
        return Ok(PodUpdate::Config {
            toml_text: content.to_string(),
            parsed: Box::new(parsed),
        });
    }
    if let Some((id, suffix)) = parse_behavior_path(filename) {
        return match suffix {
            BEHAVIOR_TOML => {
                let parsed = pod::behaviors::parse_toml(content)
                    .map_err(|e| format!("{BEHAVIORS_DIR}/{id}/{BEHAVIOR_TOML}: {e}"))?;
                Ok(PodUpdate::BehaviorConfig {
                    behavior_id: id.to_string(),
                    toml_text: content.to_string(),
                    parsed: Box::new(parsed),
                })
            }
            BEHAVIOR_PROMPT => Ok(PodUpdate::BehaviorPrompt {
                behavior_id: id.to_string(),
                text: content.to_string(),
            }),
            other => Err(format!(
                "unknown behavior filename suffix `{other}` for behavior `{id}`"
            )),
        };
    }
    Ok(PodUpdate::SystemPrompt {
        text: content.to_string(),
    })
}

/// Create `behaviors/<id>/` on disk when the write targets a not-yet-
/// existent behavior. No-op for all other writes. Runs after content
/// validation so a bad behavior.toml doesn't leave an empty dir behind.
async fn prepare_behavior_dir(
    pod_dir: &Path,
    filename: &str,
    behavior_ids: &[String],
) -> Result<(), String> {
    let Some((id, _)) = parse_behavior_path(filename) else {
        return Ok(());
    };
    if behavior_ids.iter().any(|i| i == id) {
        return Ok(());
    }
    let dir = pod_dir.join(BEHAVIORS_DIR).join(id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("mkdir {}: {e}", dir.display()))
}

/// Write a default `state.json` next to a newly created `behavior.toml`
/// when one isn't already on disk. Idempotent — a later update to an
/// existing behavior never overwrites an accumulated state.
async fn seed_behavior_state_if_needed(pod_dir: &Path, update: &PodUpdate) -> Result<(), String> {
    let behavior_id = match update {
        PodUpdate::BehaviorConfig { behavior_id, .. } => behavior_id,
        _ => return Ok(()),
    };
    let state_path = pod_dir
        .join(BEHAVIORS_DIR)
        .join(behavior_id)
        .join(BEHAVIOR_STATE);
    match tokio::fs::try_exists(&state_path).await {
        Ok(true) => return Ok(()),
        Ok(false) => {}
        Err(e) => return Err(format!("stat {}: {e}", state_path.display())),
    }
    let bytes = serde_json::to_vec_pretty(&BehaviorState::default())
        .map_err(|e| format!("serialize default BehaviorState: {e}"))?;
    tokio::fs::write(&state_path, bytes)
        .await
        .map_err(|e| format!("seed {}: {e}", state_path.display()))
}

// ---------- nearest_match_hint ----------
//
// Adapted from mcp-host's edit_file: when old_string isn't in the file,
// scan for the window of lines whose whitespace-trimmed contents best
// match, so the model can see where it was close.

fn nearest_match_hint(content: &str, old_string: &str) -> String {
    const CONTEXT: usize = 2;
    let file_lines: Vec<&str> = content.lines().collect();
    let old_lines: Vec<&str> = old_string.lines().collect();
    if old_lines.is_empty() || file_lines.is_empty() {
        return String::new();
    }
    let window_size = old_lines.len();
    if window_size > file_lines.len() {
        return String::new();
    }

    let mut best_score = 0usize;
    let mut best_start = 0usize;
    for start in 0..=(file_lines.len() - window_size) {
        let score: usize = (0..window_size)
            .filter(|&i| file_lines[start + i].trim() == old_lines[i].trim())
            .count();
        if score > best_score {
            best_score = score;
            best_start = start;
        }
    }

    if best_score == 0 {
        return " No closely-matching region found in the file.".into();
    }

    let ctx_start = best_start.saturating_sub(CONTEXT);
    let ctx_end = (best_start + window_size + CONTEXT).min(file_lines.len());
    let mut out = format!(
        "\nClosest matching region (lines {}-{}; {}/{} lines match after whitespace trim):\n",
        best_start + 1,
        best_start + window_size,
        best_score,
        window_size,
    );
    for (offset, line) in file_lines[ctx_start..ctx_end].iter().enumerate() {
        let i = ctx_start + offset;
        let in_window = i >= best_start && i < best_start + window_size;
        let marker = if in_window { ">" } else { " " };
        out.push_str(&format!("{marker} {:5} │ {}\n", i + 1, line));
    }
    out
}

/// When `old_string` appears more than once, show the first few match
/// sites with line numbers and ±2 lines of surrounding context so the
/// model can extend `old_string` AND `new_string` with unique
/// neighboring text. Caps at 3 matches — the model only needs enough
/// information to tell the targets apart.
fn multi_match_hint(content: &str, old_string: &str) -> String {
    const CONTEXT: usize = 2;
    const MAX_SHOWN: usize = 3;

    let file_lines: Vec<&str> = content.lines().collect();
    let old_lines: Vec<&str> = old_string.lines().collect();
    if old_lines.is_empty() || file_lines.is_empty() {
        return String::new();
    }
    let window_size = old_lines.len();
    if window_size > file_lines.len() {
        return String::new();
    }

    // Find every starting line whose window equals old_string exactly.
    // Literal compare — match `content.matches(old_string)`'s semantics.
    let mut hits: Vec<usize> = Vec::new();
    for start in 0..=(file_lines.len() - window_size) {
        if (0..window_size).all(|i| file_lines[start + i] == old_lines[i]) {
            hits.push(start);
        }
    }
    if hits.len() < 2 {
        // `content.matches(old_string).count()` can exceed line-based
        // hits when the pattern starts mid-line. Fall through with no
        // hint rather than mislead the model.
        return String::new();
    }

    let total = hits.len();
    let shown = hits.len().min(MAX_SHOWN);
    let mut out = format!("\nMatch sites ({} of {} shown):\n", shown, total);
    for (idx, start) in hits.iter().take(MAX_SHOWN).enumerate() {
        let ctx_start = start.saturating_sub(CONTEXT);
        let ctx_end = (start + window_size + CONTEXT).min(file_lines.len());
        out.push_str(&format!(
            "  match {} (lines {}-{}):\n",
            idx + 1,
            start + 1,
            start + window_size,
        ));
        for (offset, line) in file_lines[ctx_start..ctx_end].iter().enumerate() {
            let i = ctx_start + offset;
            let in_window = i >= *start && i < *start + window_size;
            let marker = if in_window { ">" } else { " " };
            out.push_str(&format!("    {marker} {:5} │ {}\n", i + 1, line));
        }
    }
    if total > MAX_SHOWN {
        out.push_str(&format!(
            "  ... ({} further match{} omitted)\n",
            total - MAX_SHOWN,
            if total - MAX_SHOWN == 1 { "" } else { "es" }
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn read_file_whitelists_filename() {
        let dir = temp_dir();
        tokio::fs::write(dir.join("pod.toml"), "hello")
            .await
            .unwrap();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg.clone(),
            vec![],
            POD_READ_FILE,
            json!({ "filename": "secrets.env" }),
        )
        .await;
        assert!(out.result.is_error);
        assert!(out.pod_update.is_none());
        let ok = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_READ_FILE,
            json!({ "filename": "pod.toml" }),
        )
        .await;
        assert!(!ok.result.is_error);
    }

    #[tokio::test]
    async fn write_pod_toml_validates_before_disk() {
        let dir = temp_dir();
        let cfg = sample_config();
        // Seed the pod.toml so a later read on failure would still work.
        tokio::fs::write(dir.join("pod.toml"), pod::to_toml(&cfg).unwrap())
            .await
            .unwrap();
        let before = tokio::fs::read_to_string(dir.join("pod.toml"))
            .await
            .unwrap();
        let out = dispatch(
            dir.clone(),
            cfg.clone(),
            vec![],
            POD_WRITE_FILE,
            json!({ "filename": "pod.toml", "content": "name = \"broken\" # no thread_defaults" }),
        )
        .await;
        assert!(out.result.is_error, "invalid pod.toml should error");
        assert!(out.pod_update.is_none());
        let after = tokio::fs::read_to_string(dir.join("pod.toml"))
            .await
            .unwrap();
        assert_eq!(before, after, "on-disk file must be untouched on failure");
    }

    #[tokio::test]
    async fn write_system_prompt_accepts_any_text() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_WRITE_FILE,
            json!({ "filename": "system_prompt.md", "content": "You are helpful." }),
        )
        .await;
        assert!(!out.result.is_error);
        match out.pod_update {
            Some(PodUpdate::SystemPrompt { text }) => assert_eq!(text, "You are helpful."),
            other => panic!("expected SystemPrompt update, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn edit_requires_unique_match() {
        let dir = temp_dir();
        tokio::fs::write(dir.join("system_prompt.md"), "hi\nhi\n")
            .await
            .unwrap();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_EDIT_FILE,
            json!({
                "filename": "system_prompt.md",
                "old_string": "hi",
                "new_string": "bye"
            }),
        )
        .await;
        assert!(
            out.result.is_error,
            "two matches without replace_all should error"
        );
        let text = join_blocks(&out.result.content);
        // Multi-match error must surface both match sites with line
        // numbers so the model can disambiguate by surrounding text.
        assert!(
            text.contains("Match sites"),
            "missing match-sites hint: {text}"
        );
        assert!(text.contains("match 1"), "missing 'match 1' label: {text}");
        assert!(text.contains("match 2"), "missing 'match 2' label: {text}");
    }

    #[test]
    fn multi_match_hint_caps_at_three() {
        let content =
            "name = \"test\"\nfoo\nname = \"test\"\nbar\nname = \"test\"\nbaz\nname = \"test\"\n";
        let hint = multi_match_hint(content, "name = \"test\"");
        assert!(hint.contains("3 of 4 shown"));
        assert!(hint.contains("1 further match omitted"));
    }

    #[tokio::test]
    async fn edit_replace_all_succeeds() {
        let dir = temp_dir();
        tokio::fs::write(dir.join("system_prompt.md"), "hi\nhi\n")
            .await
            .unwrap();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_EDIT_FILE,
            json!({
                "filename": "system_prompt.md",
                "old_string": "hi",
                "new_string": "bye",
                "replace_all": true
            }),
        )
        .await;
        assert!(!out.result.is_error);
        let disk = tokio::fs::read_to_string(dir.join("system_prompt.md"))
            .await
            .unwrap();
        assert_eq!(disk, "bye\nbye\n");
    }

    #[test]
    fn is_pod_modify_classification() {
        assert!(!is_pod_modify(POD_READ_FILE));
        assert!(!is_pod_modify(POD_LIST_FILES));
        assert!(is_pod_modify(POD_WRITE_FILE));
        assert!(is_pod_modify(POD_EDIT_FILE));
        assert!(!is_pod_modify("some_mcp_tool"));
    }

    #[tokio::test]
    async fn list_files_marks_allowed_and_absent() {
        let dir = temp_dir();
        let cfg = sample_config();
        // One allowed file present, one absent. Plus a non-allowed subdir.
        tokio::fs::write(dir.join("pod.toml"), pod::to_toml(&cfg).unwrap())
            .await
            .unwrap();
        tokio::fs::create_dir(dir.join("threads")).await.unwrap();

        let out = dispatch(dir.clone(), cfg, vec![], POD_LIST_FILES, json!({})).await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("[rw]"), "missing rw marker: {text}");
        assert!(text.contains("pod.toml"), "missing pod.toml: {text}");
        assert!(text.contains("[--]"), "missing non-allowed marker: {text}");
        assert!(text.contains("threads/"), "missing threads subdir: {text}");
        assert!(
            text.contains("not yet present") && text.contains("system_prompt.md"),
            "missing absent-file note: {text}"
        );
    }

    // ---------- behavior-write coverage ----------

    #[test]
    fn parse_behavior_path_recognizes_valid_shapes() {
        assert_eq!(
            parse_behavior_path("behaviors/nightly/behavior.toml"),
            Some(("nightly", "behavior.toml"))
        );
        assert_eq!(
            parse_behavior_path("behaviors/x/prompt.md"),
            Some(("x", "prompt.md"))
        );
        assert_eq!(parse_behavior_path("pod.toml"), None);
        assert_eq!(parse_behavior_path("behaviors/x"), None);
        // Nested suffix isn't a valid behavior file.
        assert_eq!(parse_behavior_path("behaviors/x/nested/file"), None);
    }

    #[tokio::test]
    async fn create_new_behavior_via_write_file() {
        let dir = temp_dir();
        let cfg = sample_config();
        let body = r#"name = "Nightly"

[trigger]
kind = "cron"
schedule = "0 9 * * *"
"#;
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_WRITE_FILE,
            json!({
                "filename": "behaviors/nightly/behavior.toml",
                "content": body,
            }),
        )
        .await;
        assert!(!out.result.is_error, "write failed: {:?}", out.result);

        // Emits a BehaviorConfig PodUpdate with the parsed config.
        match out.pod_update {
            Some(PodUpdate::BehaviorConfig {
                behavior_id,
                parsed,
                ..
            }) => {
                assert_eq!(behavior_id, "nightly");
                assert_eq!(parsed.name, "Nightly");
            }
            other => panic!("expected BehaviorConfig update, got {other:?}"),
        }

        // Directory + behavior.toml + default state.json on disk.
        let behavior_dir = dir.join("behaviors").join("nightly");
        assert!(behavior_dir.join("behavior.toml").exists());
        let state_json = tokio::fs::read_to_string(behavior_dir.join("state.json"))
            .await
            .unwrap();
        let state: whisper_agent_protocol::BehaviorState =
            serde_json::from_str(&state_json).unwrap();
        assert_eq!(state, whisper_agent_protocol::BehaviorState::default());
    }

    #[tokio::test]
    async fn prompt_before_behavior_toml_is_rejected() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_WRITE_FILE,
            json!({
                "filename": "behaviors/ghost/prompt.md",
                "content": "hello",
            }),
        )
        .await;
        assert!(
            out.result.is_error,
            "prompt.md without behavior.toml must error"
        );
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("does not exist yet"),
            "missing explanatory error: {text}"
        );
        // Nothing hit disk.
        assert!(!dir.join("behaviors").join("ghost").exists());
    }

    #[tokio::test]
    async fn invalid_behavior_toml_leaves_disk_untouched_and_no_state_seeded() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_WRITE_FILE,
            json!({
                "filename": "behaviors/bad/behavior.toml",
                // missing required `name` field
                "content": "description = \"oops\"\n",
            }),
        )
        .await;
        assert!(out.result.is_error, "invalid TOML must error");
        assert!(out.pod_update.is_none());
        // Validation errors must not leave an orphan directory behind.
        assert!(!dir.join("behaviors").join("bad").exists());
    }

    #[tokio::test]
    async fn update_existing_behavior_keeps_state_json() {
        let dir = temp_dir();
        let cfg = sample_config();

        // Seed a behavior on disk.
        let behavior_dir = dir.join("behaviors").join("daily");
        tokio::fs::create_dir_all(&behavior_dir).await.unwrap();
        tokio::fs::write(behavior_dir.join("behavior.toml"), r#"name = "v1""#)
            .await
            .unwrap();
        let state = whisper_agent_protocol::BehaviorState {
            run_count: 42,
            ..Default::default()
        };
        tokio::fs::write(
            behavior_dir.join("state.json"),
            serde_json::to_vec_pretty(&state).unwrap(),
        )
        .await
        .unwrap();

        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["daily".to_string()],
            POD_WRITE_FILE,
            json!({
                "filename": "behaviors/daily/behavior.toml",
                "content": "name = \"v2\"\n",
            }),
        )
        .await;
        assert!(!out.result.is_error);

        // state.json must still carry its run_count.
        let state_json = tokio::fs::read_to_string(behavior_dir.join("state.json"))
            .await
            .unwrap();
        let state: whisper_agent_protocol::BehaviorState =
            serde_json::from_str(&state_json).unwrap();
        assert_eq!(state.run_count, 42);
    }

    #[tokio::test]
    async fn prompt_md_update_emits_behavior_prompt_update() {
        let dir = temp_dir();
        let cfg = sample_config();

        // Seed a behavior on disk so `behaviors/daily/prompt.md` is in the allowlist.
        let behavior_dir = dir.join("behaviors").join("daily");
        tokio::fs::create_dir_all(&behavior_dir).await.unwrap();
        tokio::fs::write(behavior_dir.join("behavior.toml"), r#"name = "d""#)
            .await
            .unwrap();

        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["daily".to_string()],
            POD_WRITE_FILE,
            json!({
                "filename": "behaviors/daily/prompt.md",
                "content": "do the thing",
            }),
        )
        .await;
        assert!(!out.result.is_error);
        match out.pod_update {
            Some(PodUpdate::BehaviorPrompt { behavior_id, text }) => {
                assert_eq!(behavior_id, "daily");
                assert_eq!(text, "do the thing");
            }
            other => panic!("expected BehaviorPrompt update, got {other:?}"),
        }
    }

    // ---------- read-only access coverage ----------

    #[tokio::test]
    async fn read_file_accepts_thread_json() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::create_dir_all(dir.join("threads"))
            .await
            .unwrap();
        tokio::fs::write(dir.join("threads").join("abc.json"), "{\"id\": \"abc\"}\n")
            .await
            .unwrap();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_READ_FILE,
            json!({ "filename": "threads/abc.json" }),
        )
        .await;
        assert!(!out.result.is_error, "unexpected error: {:?}", out.result);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("\"id\""), "unexpected body: {text}");
    }

    #[tokio::test]
    async fn write_to_readonly_path_rejected() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::create_dir_all(dir.join("threads"))
            .await
            .unwrap();
        tokio::fs::write(dir.join("threads").join("abc.json"), "{}")
            .await
            .unwrap();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_WRITE_FILE,
            json!({ "filename": "threads/abc.json", "content": "{}" }),
        )
        .await;
        assert!(out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("read-only"),
            "expected read-only error: {text}"
        );
    }

    #[tokio::test]
    async fn read_file_applies_default_line_cap_on_large_files() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::create_dir_all(dir.join("threads"))
            .await
            .unwrap();
        let mut body = String::new();
        for i in 0..READ_DEFAULT_LINE_CAP + 100 {
            body.push_str(&format!("line {i}\n"));
        }
        tokio::fs::write(dir.join("threads").join("big.json"), &body)
            .await
            .unwrap();
        // No offset/limit — should truncate and mention it.
        let out = dispatch(
            dir.clone(),
            cfg.clone(),
            vec![],
            POD_READ_FILE,
            json!({ "filename": "threads/big.json" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        let line_count = text.lines().filter(|l| l.starts_with("line ")).count();
        assert_eq!(line_count, READ_DEFAULT_LINE_CAP);
        assert!(
            text.contains("showing first") && text.contains("of "),
            "missing truncation note: {text}"
        );
        // Explicit limit overrides the default.
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_READ_FILE,
            json!({ "filename": "threads/big.json", "limit": 5 }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        let line_count = text.lines().filter(|l| l.starts_with("line ")).count();
        assert_eq!(line_count, 5);
    }

    #[tokio::test]
    async fn list_files_surfaces_threads_dir() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::write(dir.join("pod.toml"), pod::to_toml(&cfg).unwrap())
            .await
            .unwrap();
        tokio::fs::create_dir_all(dir.join("threads"))
            .await
            .unwrap();
        for id in ["a", "b", "c"] {
            tokio::fs::write(dir.join("threads").join(format!("{id}.json")), "{}")
                .await
                .unwrap();
        }
        let out = dispatch(dir.clone(), cfg, vec![], POD_LIST_FILES, json!({})).await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("threads/ (read-only"),
            "missing threads section: {text}"
        );
        assert!(text.contains("3 thread file"), "missing count: {text}");
        for id in ["a", "b", "c"] {
            assert!(
                text.contains(&format!("{id}.json")),
                "missing thread {id}.json in listing: {text}"
            );
        }
    }

    #[tokio::test]
    async fn list_files_shows_behavior_sections() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::write(dir.join("pod.toml"), pod::to_toml(&cfg).unwrap())
            .await
            .unwrap();

        // Two behaviors on disk, one with prompt.md + state.json.
        let a = dir.join("behaviors").join("alpha");
        tokio::fs::create_dir_all(&a).await.unwrap();
        tokio::fs::write(a.join("behavior.toml"), r#"name = "A""#)
            .await
            .unwrap();
        tokio::fs::write(a.join("prompt.md"), "do alpha")
            .await
            .unwrap();
        tokio::fs::write(a.join("state.json"), r#"{"run_count": 3}"#)
            .await
            .unwrap();
        let b = dir.join("behaviors").join("beta");
        tokio::fs::create_dir_all(&b).await.unwrap();
        tokio::fs::write(b.join("behavior.toml"), r#"name = "B""#)
            .await
            .unwrap();

        let out = dispatch(
            dir.clone(),
            cfg,
            vec!["alpha".to_string(), "beta".to_string()],
            POD_LIST_FILES,
            json!({}),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("behaviors/alpha/"),
            "missing alpha section: {text}"
        );
        assert!(
            text.contains("behaviors/beta/"),
            "missing beta section: {text}"
        );
        assert!(
            text.contains("behavior.toml") && text.contains("prompt.md"),
            "missing behavior files: {text}"
        );
        // state.json listed as read-only (runtime state).
        let state_line = text
            .lines()
            .find(|l| l.contains("state.json"))
            .expect("state.json missing from listing");
        assert!(
            state_line.starts_with("[r-]"),
            "state.json should be read-only, got: {state_line}"
        );
    }
}
