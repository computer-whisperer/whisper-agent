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
//!   * Future: Lua files (add their names here when support lands).
//!
//! Successful writes emit a [`PodUpdate`] so the scheduler can refresh
//! in-memory pod state and broadcast the change to subscribers in the
//! same step as the disk write. See [`crate::runtime::scheduler::Scheduler::apply_pod_config_update`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};
use whisper_agent_protocol::PodConfig;

use crate::pod;
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

pub const POD_READ_FILE: &str = "pod_read_file";
pub const POD_WRITE_FILE: &str = "pod_write_file";
pub const POD_EDIT_FILE: &str = "pod_edit_file";
pub const POD_LIST_FILES: &str = "pod_list_files";

/// True if `name` is a builtin pod tool. Used by the scheduler's router
/// to branch the tool-call dispatch path.
pub fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        POD_READ_FILE | POD_WRITE_FILE | POD_EDIT_FILE | POD_LIST_FILES
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
}

/// What a builtin tool handler produces: the tool result the model sees,
/// plus an optional state update for the scheduler to apply atomically.
pub struct ToolOutcome {
    pub result: CallToolResult,
    pub pod_update: Option<PodUpdate>,
}

/// Dispatch a builtin tool call. `pod_dir` is the pod's on-disk directory;
/// `pod_config` is the current in-memory config (used to derive the
/// filename whitelist and to validate writes).
pub async fn dispatch(
    pod_dir: PathBuf,
    pod_config: PodConfig,
    tool_name: &str,
    args: Value,
) -> ToolOutcome {
    let allowed = allowed_filenames(&pod_config);
    match tool_name {
        POD_LIST_FILES => list_files(&pod_dir, &allowed, args).await,
        POD_READ_FILE => read_file(&pod_dir, &allowed, args).await,
        POD_WRITE_FILE => write_file(&pod_dir, &allowed, args).await,
        POD_EDIT_FILE => edit_file(&pod_dir, &allowed, args).await,
        other => no_update_error(format!("unknown builtin tool: {other}")),
    }
}

fn allowed_filenames(cfg: &PodConfig) -> Vec<String> {
    let mut out = vec![pod::POD_TOML.to_string()];
    let prompt = &cfg.thread_defaults.system_prompt_file;
    if !prompt.is_empty() && prompt != pod::POD_TOML {
        out.push(prompt.clone());
    }
    out
}

fn validate_filename(filename: &str, allowed: &[String]) -> Result<(), String> {
    if allowed.iter().any(|n| n == filename) {
        Ok(())
    } else {
        Err(format!(
            "filename `{filename}` is not in this pod's allowlist. Allowed: [{}]",
            allowed.join(", ")
        ))
    }
}

fn no_update_error(message: String) -> ToolOutcome {
    ToolOutcome {
        result: error_result(message),
        pod_update: None,
    }
}

fn no_update_text(message: String) -> ToolOutcome {
    ToolOutcome {
        result: text_result(message),
        pod_update: None,
    }
}

// ---------- list ----------

fn list_descriptor() -> McpTool {
    McpTool {
        name: POD_LIST_FILES.into(),
        description: "List the contents of this pod's directory. Each entry is \
                      tagged `[rw]` if it is reachable via `pod_read_file` / \
                      `pod_write_file` / `pod_edit_file`, or `[--]` otherwise \
                      (e.g. the internal `threads/` subdirectory). Pod files live \
                      OUTSIDE your workspace and are reachable ONLY through the \
                      `pod_*_file` tools — not `list_dir` or `bash`. Use this \
                      before editing to see which config files exist and how \
                      large they are."
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

async fn list_files(pod_dir: &Path, allowed: &[String], _args: Value) -> ToolOutcome {
    let mut read = match tokio::fs::read_dir(pod_dir).await {
        Ok(r) => r,
        Err(e) => return no_update_error(format!("list pod dir: {e}")),
    };
    // Collect (name, is_dir, size) then sort: allowed files first,
    // then directories, then other files — each group name-sorted.
    let mut entries: Vec<(String, bool, u64)> = Vec::new();
    loop {
        match read.next_entry().await {
            Ok(None) => break,
            Ok(Some(entry)) => {
                let name = entry.file_name().to_string_lossy().into_owned();
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
            Err(e) => return no_update_error(format!("list pod dir iter: {e}")),
        }
    }
    entries.sort_by(|a, b| {
        let a_allowed = allowed.iter().any(|n| n == &a.0);
        let b_allowed = allowed.iter().any(|n| n == &b.0);
        b_allowed
            .cmp(&a_allowed)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut out = String::new();
    for (name, is_dir, size) in &entries {
        let is_allowed = allowed.iter().any(|n| n == name);
        let access = if is_allowed { "[rw]" } else { "[--]" };
        let kind = if *is_dir { 'd' } else { 'f' };
        let display_name = if *is_dir {
            format!("{name}/")
        } else {
            name.clone()
        };
        if *is_dir {
            out.push_str(&format!("{access}  {kind}  {:>10}  {}\n", "", display_name));
        } else {
            out.push_str(&format!("{access}  {kind}  {size:>10}  {}\n", display_name));
        }
    }
    // Mention whitelisted-but-absent files so the agent knows a write
    // will create them rather than targeting something that already
    // exists elsewhere.
    let absent: Vec<&String> = allowed
        .iter()
        .filter(|a| !entries.iter().any(|(n, _, _)| n == *a))
        .collect();
    if !absent.is_empty() {
        out.push_str("\nAllowed filenames not yet present (pod_write_file will create):\n");
        for name in absent {
            out.push_str(&format!("  {name}\n"));
        }
    }
    if out.is_empty() {
        out.push_str("(pod directory is empty)\n");
    }
    no_update_text(out)
}

// ---------- read ----------

fn read_descriptor() -> McpTool {
    McpTool {
        name: POD_READ_FILE.into(),
        description: "Read one of this pod's configuration files (pod.toml or the \
                      system-prompt file). These files live OUTSIDE your workspace — \
                      they are the pod's definition, held by the agent host, and are \
                      reachable ONLY through the `pod_*_file` tools. Do not look for \
                      them via `read_file`, `list_dir`, or `bash` (those operate on \
                      the workspace and will not find pod.toml). With no line options, \
                      returns the whole file. Use `offset`/`limit` to target a line \
                      range for large files."
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

async fn read_file(pod_dir: &Path, allowed: &[String], args: Value) -> ToolOutcome {
    let parsed: ReadArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    if let Err(e) = validate_filename(&parsed.filename, allowed) {
        return no_update_error(e);
    }
    let path = pod_dir.join(&parsed.filename);
    let content = match tokio::fs::read_to_string(&path).await {
        Ok(s) => s,
        Err(e) => return no_update_error(format!("read {}: {e}", parsed.filename)),
    };
    if parsed.offset.is_none() && parsed.limit.is_none() {
        return no_update_text(content);
    }
    let offset = parsed.offset.unwrap_or(1).max(1) as usize;
    let limit = parsed.limit.map(|n| n as usize).unwrap_or(usize::MAX);
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = (offset - 1).min(total);
    let end = start.saturating_add(limit).min(total);
    let mut out = lines[start..end].join("\n");
    if end == total && content.ends_with('\n') {
        out.push('\n');
    }
    if end < total {
        out.push_str(&format!(
            "\n[showing lines {offset}-{end} of {total}; pass offset/limit to see more]\n"
        ));
    }
    no_update_text(out)
}

// ---------- write ----------

fn write_descriptor() -> McpTool {
    McpTool {
        name: POD_WRITE_FILE.into(),
        description: "Fully overwrite one of this pod's configuration files \
                      (pod.toml or the system-prompt file). These files live OUTSIDE \
                      your workspace and are reachable ONLY through the `pod_*_file` \
                      tools — not `write_file` or `bash`. For `pod.toml` the new text \
                      is parsed and validated before landing on disk; invalid TOML or \
                      schema violations are returned as tool errors and the on-disk \
                      file is untouched. Prefer `pod_edit_file` for targeted changes \
                      — this is for full rewrites only."
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

async fn write_file(pod_dir: &Path, allowed: &[String], args: Value) -> ToolOutcome {
    let parsed: WriteArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    if let Err(e) = validate_filename(&parsed.filename, allowed) {
        return no_update_error(e);
    }
    let update = match prepare_update(&parsed.filename, &parsed.content) {
        Ok(u) => u,
        Err(e) => return no_update_error(e),
    };
    let path = pod_dir.join(&parsed.filename);
    if let Err(e) = tokio::fs::write(&path, parsed.content.as_bytes()).await {
        return no_update_error(format!("write {}: {e}", parsed.filename));
    }
    ToolOutcome {
        result: text_result(format!(
            "wrote {} bytes to {}",
            parsed.content.len(),
            parsed.filename
        )),
        pod_update: Some(update),
    }
}

// ---------- edit ----------

fn edit_descriptor() -> McpTool {
    McpTool {
        name: POD_EDIT_FILE.into(),
        description: "Replace a literal substring in one of this pod's config files. \
                      These files live OUTSIDE your workspace and are reachable ONLY \
                      through the `pod_*_file` tools — not `edit_file` or `bash`. \
                      `old_string` must match literally (no regex) and by default \
                      must occur exactly once; pass `replace_all: true` to change \
                      every occurrence. If old_string isn't found, the error shows \
                      the closest matching region; if it matches multiple times, the \
                      error lists each match site with line numbers. When extending \
                      `old_string` with surrounding context to disambiguate a \
                      multi-match, extend `new_string` by the same neighboring text \
                      — extending only `old_string` will delete the intervening \
                      lines from the output. For `pod.toml` the resulting text is \
                      parsed and validated before landing on disk."
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

async fn edit_file(pod_dir: &Path, allowed: &[String], args: Value) -> ToolOutcome {
    let parsed: EditArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    if let Err(e) = validate_filename(&parsed.filename, allowed) {
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
    }
}

/// Decide what kind of [`PodUpdate`] a successful write produces.
/// For `pod.toml` we parse + validate here — failure becomes a tool error
/// BEFORE any disk write, so partial state is never reached.
fn prepare_update(filename: &str, content: &str) -> Result<PodUpdate, String> {
    if filename == pod::POD_TOML {
        let parsed = pod::parse_toml(content).map_err(|e| format!("pod.toml: {e}"))?;
        Ok(PodUpdate::Config {
            toml_text: content.to_string(),
            parsed: Box::new(parsed),
        })
    } else {
        Ok(PodUpdate::SystemPrompt {
            text: content.to_string(),
        })
    }
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
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("wa-builtin-tools-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

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
            POD_READ_FILE,
            json!({ "filename": "secrets.env" }),
        )
        .await;
        assert!(out.result.is_error);
        assert!(out.pod_update.is_none());
        let ok = dispatch(
            dir.clone(),
            cfg,
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

    fn join_blocks(blocks: &[crate::tools::mcp::McpContentBlock]) -> String {
        let mut s = String::new();
        for b in blocks {
            let crate::tools::mcp::McpContentBlock::Text { text } = b;
            s.push_str(text);
        }
        s
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

        let out = dispatch(dir.clone(), cfg, POD_LIST_FILES, json!({})).await;
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
}
