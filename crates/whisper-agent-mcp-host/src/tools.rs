//! The three MVP tools: read_file, write_file, bash.
//!
//! Each tool returns a [`CallToolResult`]. Tool-level errors (file not found, command failed)
//! are returned as `is_error: true` results — not JSON-RPC errors. JSON-RPC errors are reserved
//! for protocol-level problems (unknown tool, bad arguments).

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::timeout;

use crate::protocol::{CallToolResult, Tool, ToolAnnotations};
use crate::workspace::Workspace;

pub fn descriptors() -> Vec<Tool> {
    vec![
        read_file_descriptor(),
        write_file_descriptor(),
        edit_file_descriptor(),
        bash_descriptor(),
        list_dir_descriptor(),
        glob_descriptor(),
        grep_descriptor(),
    ]
}

pub async fn call(workspace: &Arc<Workspace>, name: &str, args: Value) -> Result<CallToolResult, ToolDispatchError> {
    match name {
        "read_file" => Ok(read_file(workspace, args).await),
        "write_file" => Ok(write_file(workspace, args).await),
        "edit_file" => Ok(edit_file(workspace, args).await),
        "bash" => Ok(bash(workspace, args).await),
        "list_dir" => Ok(list_dir(workspace, args).await),
        "glob" => Ok(glob(workspace, args).await),
        "grep" => Ok(grep(workspace, args).await),
        _ => Err(ToolDispatchError::UnknownTool(name.to_string())),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolDispatchError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
}

// ---------- read_file ----------

fn read_file_descriptor() -> Tool {
    Tool {
        name: "read_file".into(),
        description: "Read the contents of a UTF-8 text file within the workspace. With no line \
                      options, returns the whole file. Use `offset`/`limit` to target a line \
                      range when working with large files — cheaper than loading everything just \
                      to edit a few lines."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path — absolute, or relative to the workspace root."
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
            "required": ["path"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Read file".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {
    path: PathBuf,
    #[serde(default)]
    offset: Option<u32>,
    #[serde(default)]
    limit: Option<u32>,
}

async fn read_file(workspace: &Arc<Workspace>, args: Value) -> CallToolResult {
    let parsed: ReadFileArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };
    let path = match workspace.resolve(&parsed.path) {
        Ok(p) => p,
        Err(e) => return CallToolResult::error_text(e.to_string()),
    };
    // Whole-file fast path when no line slicing is requested.
    if parsed.offset.is_none() && parsed.limit.is_none() {
        return match tokio::fs::read_to_string(&path).await {
            Ok(content) => CallToolResult::text(content),
            Err(e) => CallToolResult::error_text(format!(
                "read_file({}): {e}",
                parsed.path.display()
            )),
        };
    }
    // Line-sliced read — stream so we don't load a huge file just to keep a few lines.
    let content = match tokio::fs::read_to_string(&path).await {
        Ok(s) => s,
        Err(e) => {
            return CallToolResult::error_text(format!(
                "read_file({}): {e}",
                parsed.path.display()
            ));
        }
    };
    let offset = parsed.offset.unwrap_or(1).max(1) as usize;
    let limit = parsed.limit.map(|n| n as usize).unwrap_or(usize::MAX);
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = (offset - 1).min(total);
    let end = start.saturating_add(limit).min(total);
    let slice = lines[start..end].join("\n");
    // Preserve the trailing newline if the whole file has one — avoids surprising
    // models with asymmetric read/write behavior.
    let mut out = slice;
    if end == total && content.ends_with('\n') {
        out.push('\n');
    }
    if end < total {
        let last = end;
        out.push_str(&format!(
            "\n[showing lines {offset}-{last} of {total}; pass offset/limit to see more]\n"
        ));
    }
    CallToolResult::text(out)
}

// ---------- write_file ----------

fn write_file_descriptor() -> Tool {
    Tool {
        name: "write_file".into(),
        description: "Write UTF-8 text to a file within the workspace, creating parent \
                      directories as needed. Overwrites if the file exists. Prefer \
                      `edit_file` for changes to an existing file — it's much cheaper than \
                      rewriting. Use `write_file` only to create a new file or fully rewrite \
                      one. Do NOT create Markdown (*.md) or README files unless the user \
                      explicitly asks."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path — absolute, or relative to the workspace root." },
                "content": { "type": "string", "description": "File contents to write." }
            },
            "required": ["path", "content"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Write file".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(true),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct WriteFileArgs {
    path: PathBuf,
    content: String,
}

async fn write_file(workspace: &Arc<Workspace>, args: Value) -> CallToolResult {
    let parsed: WriteFileArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };
    let path = match workspace.resolve(&parsed.path) {
        Ok(p) => p,
        Err(e) => return CallToolResult::error_text(e.to_string()),
    };
    if let Some(parent) = path.parent()
        && let Err(e) = tokio::fs::create_dir_all(parent).await {
            return CallToolResult::error_text(format!("create_dir_all({}): {e}", parent.display()));
        }
    match tokio::fs::write(&path, parsed.content.as_bytes()).await {
        Ok(()) => CallToolResult::text(format!("wrote {} bytes to {}", parsed.content.len(), parsed.path.display())),
        Err(e) => CallToolResult::error_text(format!("write_file({}): {e}", parsed.path.display())),
    }
}

// ---------- edit_file ----------

fn edit_file_descriptor() -> Tool {
    Tool {
        name: "edit_file".into(),
        description: "Replace text in a file. `old_string` must match literally — no regex. \
                      By default requires exactly one match; pass `replace_all: true` to \
                      change every occurrence. If old_string isn't found, the error shows the \
                      closest matching region of the file so you can correct the next call. \
                      Prefer this over write_file for targeted changes."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path — absolute, or relative to the workspace root."
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact substring to find. Must match literally, including \
                                    whitespace and indentation. Be specific enough that it \
                                    occurs only once, or set replace_all."
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement. May be empty to delete the matched span."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace every occurrence of old_string. If false \
                                    (default), require exactly one match."
                }
            },
            "required": ["path", "old_string", "new_string"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Edit file".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(true),
            // Not idempotent — rerunning after success would find zero matches and error.
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct EditFileArgs {
    path: PathBuf,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

async fn edit_file(workspace: &Arc<Workspace>, args: Value) -> CallToolResult {
    let parsed: EditFileArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };
    if parsed.old_string.is_empty() {
        return CallToolResult::error_text(
            "edit_file: old_string must be non-empty (use write_file to create new files)",
        );
    }
    if parsed.old_string == parsed.new_string {
        return CallToolResult::error_text(
            "edit_file: old_string and new_string are identical — no-op",
        );
    }
    let path = match workspace.resolve(&parsed.path) {
        Ok(p) => p,
        Err(e) => return CallToolResult::error_text(e.to_string()),
    };
    let content = match tokio::fs::read_to_string(&path).await {
        Ok(s) => s,
        Err(e) => {
            return CallToolResult::error_text(format!(
                "edit_file({}): {e}",
                parsed.path.display()
            ));
        }
    };
    let found = content.matches(&parsed.old_string).count();
    if found == 0 {
        let hint = nearest_match_hint(&content, &parsed.old_string);
        return CallToolResult::error_text(format!(
            "edit_file({}): old_string not found.{hint}",
            parsed.path.display()
        ));
    }
    if found > 1 && !parsed.replace_all {
        return CallToolResult::error_text(format!(
            "edit_file({}): old_string matches {found} times. Either extend old_string with \
             surrounding context to make it unique, or pass replace_all:true to replace every \
             occurrence.",
            parsed.path.display()
        ));
    }
    let new_content = content.replace(&parsed.old_string, &parsed.new_string);
    match tokio::fs::write(&path, new_content.as_bytes()).await {
        Ok(()) => CallToolResult::text(format!(
            "edit_file({}): replaced {found} occurrence{}",
            parsed.path.display(),
            if found == 1 { "" } else { "s" }
        )),
        Err(e) => {
            CallToolResult::error_text(format!("edit_file({}): {e}", parsed.path.display()))
        }
    }
}

/// When `old_string` isn't in the file, scan for the sliding window of lines
/// whose whitespace-trimmed contents best match. Returns a formatted hint
/// with line numbers so the model can see the real content and correct its
/// next call. Returns an empty string (caller's error message stands alone)
/// if nothing matches at all.
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
        "\nClosest matching region (lines {}-{}; {}/{} lines match after whitespace trim). \
         Lines marked '>' are where old_string would have replaced; check for indentation, \
         trailing whitespace, or stale content:\n",
        best_start + 1,
        best_start + window_size,
        best_score,
        window_size,
    );
    for i in ctx_start..ctx_end {
        let in_window = i >= best_start && i < best_start + window_size;
        let marker = if in_window { ">" } else { " " };
        out.push_str(&format!("{marker} {:5} │ {}\n", i + 1, file_lines[i]));
    }
    out
}

// ---------- bash ----------

fn bash_descriptor() -> Tool {
    Tool {
        name: "bash".into(),
        description: "Run a bash command within the workspace. Returns stdout and stderr \
                      merged in shell-emission order; on non-zero exit the first line is \
                      `Exit code <N>`. Large outputs are head-truncated with a marker. \
                      Do NOT use bash for tasks that have a dedicated tool: `read_file` (not \
                      cat/head/tail), `edit_file` (not sed/awk), `write_file` (not echo >/\
                      heredoc), `grep` (not grep/rg), `glob` (not find), `list_dir` (not ls). \
                      Reserve bash for builds, tests, git, and other shell-only operations."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Command to run via `bash -c`." },
                "cwd": { "type": "string", "description": "Optional cwd — absolute, or relative to the workspace root. Defaults to the workspace root." },
                "timeout_seconds": { "type": "integer", "description": "Kill the command after this many seconds. Default 120, max 600." },
                "strip_ansi": { "type": "boolean", "description": "Strip ANSI escape sequences (colors, cursor codes) from stdout and stderr. Default true — only turn off if you specifically need the raw escape bytes." }
            },
            "required": ["command"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Run bash command".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(true),
            idempotent_hint: Some(false),
            open_world_hint: Some(true),
        }),
    }
}

fn default_strip_ansi() -> bool {
    true
}

#[derive(Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    timeout_seconds: Option<u64>,
    #[serde(default = "default_strip_ansi")]
    strip_ansi: bool,
}

static ANSI_ESCAPE: LazyLock<Regex> = LazyLock::new(|| {
    // Matches CSI sequences (ESC `[` … final byte 0x40–0x7E) and every simple
    // two-char escape in the Fp/Fe/Fs ranges (ESC + 0x30–0x7E, excluding `[`
    // which starts CSI). Covers SGR colors, cursor moves, erase codes, keypad
    // mode switches, save/restore cursor — the full set that cargo/gcc/ls/git
    // actually emit.
    Regex::new(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[0-Z\\-~])").expect("ANSI regex is valid")
});

fn strip_ansi(s: &str) -> String {
    ANSI_ESCAPE.replace_all(s, "").into_owned()
}

/// Cap on bash tool_result size. Matches claude-code's BASH_MAX_OUTPUT_LENGTH.
/// Oversized output is head-truncated — cargo-style compile errors land at the
/// tail, which is the part the model almost always needs.
const BASH_MAX_OUTPUT_BYTES: usize = 30_000;

async fn bash(workspace: &Arc<Workspace>, args: Value) -> CallToolResult {
    let parsed: BashArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };

    let cwd = match parsed.cwd.as_deref() {
        None => workspace.root().to_path_buf(),
        Some(rel) => match workspace.resolve(rel) {
            Ok(p) => p,
            Err(e) => return CallToolResult::error_text(e.to_string()),
        },
    };

    let timeout_secs = parsed.timeout_seconds.unwrap_or(120).min(600);

    // Merge stdout+stderr at shell level so the model sees one interleaved
    // stream in shell-emission order (matching claude-code's bash envelope).
    // The curly-brace group preserves the inner command's exit code; the
    // newline before `}` stops a trailing `#` comment from swallowing the
    // close.
    let wrapped = format!("{{ {cmd}\n}} 2>&1", cmd = parsed.command);

    let mut cmd = Command::new("bash");
    cmd.arg("-c")
        .arg(&wrapped)
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        // bash's own stderr is only written to on wrapper-parse errors; keep
        // it piped so we don't silently swallow those.
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return CallToolResult::error_text(format!("spawn bash: {e}")),
    };

    let output_future = child.wait_with_output();
    match timeout(Duration::from_secs(timeout_secs), output_future).await {
        Ok(Ok(output)) => {
            let mut merged = String::new();
            merged.push_str(&String::from_utf8_lossy(&output.stdout));
            merged.push_str(&String::from_utf8_lossy(&output.stderr));
            if parsed.strip_ansi {
                merged = strip_ansi(&merged);
            }
            let merged = head_truncate(merged, BASH_MAX_OUTPUT_BYTES);
            let exit_code = output.status.code().unwrap_or(-1);
            let body = if output.status.success() {
                merged
            } else {
                format!("Exit code {exit_code}\n{merged}")
            };
            if output.status.success() {
                CallToolResult::text(body)
            } else {
                CallToolResult::error_text(body)
            }
        }
        Ok(Err(e)) => CallToolResult::error_text(format!("bash wait failed: {e}")),
        Err(_) => CallToolResult::error_text(format!("bash timed out after {timeout_secs}s")),
    }
}

/// Keep the last `max` bytes of `s`, prefixed with a marker noting how much
/// was dropped. Walks forward to a UTF-8 char boundary so we never split a
/// multi-byte sequence. Returns `s` unchanged if it already fits.
fn head_truncate(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut start = s.len() - max;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    let dropped = start;
    format!("... {dropped} bytes truncated ...\n{}", &s[start..])
}

// ---------- list_dir ----------

fn list_dir_descriptor() -> Tool {
    Tool {
        name: "list_dir".into(),
        description: "List the immediate children of a directory within the workspace. Returns \
                      one line per entry with a type marker (d/f/l), byte size (0 for dirs), and \
                      name. Use this for quick directory exploration — for recursive search use \
                      `glob`."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path — absolute, or relative to the workspace root. Defaults to \".\"."
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include dotfiles in the listing. Defaults to false."
                }
            }
        }),
        annotations: Some(ToolAnnotations {
            title: Some("List directory".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct ListDirArgs {
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    include_hidden: bool,
}

async fn list_dir(workspace: &Arc<Workspace>, args: Value) -> CallToolResult {
    let parsed: ListDirArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };
    let rel = parsed.path.unwrap_or_else(|| PathBuf::from("."));
    let abs = match workspace.resolve(&rel) {
        Ok(p) => p,
        Err(e) => return CallToolResult::error_text(e.to_string()),
    };
    let mut read = match tokio::fs::read_dir(&abs).await {
        Ok(r) => r,
        Err(e) => return CallToolResult::error_text(format!("list_dir({}): {e}", rel.display())),
    };

    let mut entries: Vec<(String, char, u64)> = Vec::new();
    loop {
        match read.next_entry().await {
            Ok(None) => break,
            Ok(Some(entry)) => {
                let name = entry.file_name().to_string_lossy().into_owned();
                if !parsed.include_hidden && name.starts_with('.') {
                    continue;
                }
                let (kind, size) = match entry.file_type().await {
                    Ok(ft) if ft.is_dir() => ('d', 0u64),
                    Ok(ft) if ft.is_symlink() => {
                        let size = entry.metadata().await.map(|m| m.len()).unwrap_or(0);
                        ('l', size)
                    }
                    Ok(_) => {
                        let size = entry.metadata().await.map(|m| m.len()).unwrap_or(0);
                        ('f', size)
                    }
                    Err(_) => ('?', 0),
                };
                entries.push((name, kind, size));
            }
            Err(e) => return CallToolResult::error_text(format!("list_dir iter: {e}")),
        }
    }
    entries.sort_by(|a, b| {
        // dirs first, then by name
        let ad = a.1 == 'd';
        let bd = b.1 == 'd';
        bd.cmp(&ad).then_with(|| a.0.cmp(&b.0))
    });

    let mut out = String::new();
    for (name, kind, size) in &entries {
        let display = if *kind == 'd' {
            format!("{kind}  {:>10}  {name}/\n", "")
        } else {
            format!("{kind}  {size:>10}  {name}\n")
        };
        out.push_str(&display);
    }
    if out.is_empty() {
        out = format!("(empty directory: {})\n", rel.display());
    }
    CallToolResult::text(out)
}

// ---------- glob ----------

fn glob_descriptor() -> Tool {
    Tool {
        name: "glob".into(),
        description: "Find files in the workspace whose paths match a glob pattern. Respects \
                      `.gitignore`, `.git/info/exclude`, and hidden files (opt-in). Use standard \
                      glob syntax: `**/*.rs`, `src/**/test_*.py`, `docs/*.md`. Returns one path \
                      per line, relative to the workspace root."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern matched against paths relative to the workspace root."
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Traverse into dotfiles / dotdirs. Defaults to false."
                },
                "include_ignored": {
                    "type": "boolean",
                    "description": "Include paths that would be ignored by .gitignore. Defaults to false."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Cap the number of results. Defaults to 1000."
                }
            },
            "required": ["pattern"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Glob-find files".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct GlobArgs {
    pattern: String,
    #[serde(default)]
    include_hidden: bool,
    #[serde(default)]
    include_ignored: bool,
    #[serde(default)]
    max_results: Option<u32>,
}

async fn glob(workspace: &Arc<Workspace>, args: Value) -> CallToolResult {
    let parsed: GlobArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };
    let matcher = match globset::Glob::new(&parsed.pattern) {
        Ok(g) => g.compile_matcher(),
        Err(e) => return CallToolResult::error_text(format!("invalid glob pattern: {e}")),
    };
    let max = parsed.max_results.unwrap_or(1000).max(1) as usize;
    let root = workspace.root().to_path_buf();
    let include_hidden = parsed.include_hidden;
    let include_ignored = parsed.include_ignored;

    // Walk on a blocking thread — ignore::Walk is synchronous and can touch thousands of
    // dirs; don't tie up the async runtime.
    let result = tokio::task::spawn_blocking(move || {
        let mut walker = ignore::WalkBuilder::new(&root);
        walker
            .hidden(!include_hidden)
            .git_ignore(!include_ignored)
            .git_exclude(!include_ignored)
            .git_global(!include_ignored)
            .parents(!include_ignored);
        let mut hits: Vec<String> = Vec::new();
        let mut truncated = false;
        for entry in walker.build() {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            // Only match files (not directories) — mirrors what users expect from glob.
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let Ok(rel) = entry.path().strip_prefix(&root) else { continue };
            if matcher.is_match(rel) {
                if hits.len() >= max {
                    truncated = true;
                    break;
                }
                hits.push(rel.to_string_lossy().into_owned());
            }
        }
        hits.sort();
        (hits, truncated)
    })
    .await;
    let (hits, truncated) = match result {
        Ok(t) => t,
        Err(e) => return CallToolResult::error_text(format!("glob walk task panicked: {e}")),
    };

    if hits.is_empty() {
        return CallToolResult::text(format!("(no matches for {})", parsed.pattern));
    }
    let mut out = hits.join("\n");
    out.push('\n');
    if truncated {
        out.push_str(&format!(
            "(results truncated at {max}; narrow the pattern or raise max_results)\n"
        ));
    }
    CallToolResult::text(out)
}

// ---------- grep ----------

fn grep_descriptor() -> Tool {
    Tool {
        name: "grep".into(),
        description: "Search file contents with a regex. Walks the workspace respecting \
                      `.gitignore`, scans each text file line-by-line, and returns matches as \
                      `path:line:text` (ripgrep-style). Use `path_glob` to narrow the file set. \
                      Binary files are skipped automatically."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Rust-regex-crate syntax. See https://docs.rs/regex/latest/regex/#syntax."
                },
                "path_glob": {
                    "type": "string",
                    "description": "Only search files whose path matches this glob (e.g. `**/*.rs`). Optional."
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive match. Default false."
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Traverse dotfiles / dotdirs. Default false."
                },
                "include_ignored": {
                    "type": "boolean",
                    "description": "Include paths that would be excluded by .gitignore. Default false."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Cap the total number of hits. Default 200."
                },
                "max_per_file": {
                    "type": "integer",
                    "description": "Cap per-file hits. Default 20."
                }
            },
            "required": ["pattern"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Grep file contents".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct GrepArgs {
    pattern: String,
    #[serde(default)]
    path_glob: Option<String>,
    #[serde(default)]
    ignore_case: bool,
    #[serde(default)]
    include_hidden: bool,
    #[serde(default)]
    include_ignored: bool,
    #[serde(default)]
    max_results: Option<u32>,
    #[serde(default)]
    max_per_file: Option<u32>,
}

async fn grep(workspace: &Arc<Workspace>, args: Value) -> CallToolResult {
    let parsed: GrepArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };

    let mut rb = regex::RegexBuilder::new(&parsed.pattern);
    rb.case_insensitive(parsed.ignore_case);
    let re = match rb.build() {
        Ok(r) => r,
        Err(e) => return CallToolResult::error_text(format!("invalid regex: {e}")),
    };

    let path_matcher = match parsed.path_glob.as_deref() {
        None => None,
        Some(g) => match globset::Glob::new(g) {
            Ok(g) => Some(g.compile_matcher()),
            Err(e) => return CallToolResult::error_text(format!("invalid path_glob: {e}")),
        },
    };

    let max_total = parsed.max_results.unwrap_or(200).max(1) as usize;
    let max_per_file = parsed.max_per_file.unwrap_or(20).max(1) as usize;
    let root = workspace.root().to_path_buf();
    let include_hidden = parsed.include_hidden;
    let include_ignored = parsed.include_ignored;

    // Blocking pool: sync IO + scanning on many files.
    let result = tokio::task::spawn_blocking(move || {
        let mut walker = ignore::WalkBuilder::new(&root);
        walker
            .hidden(!include_hidden)
            .git_ignore(!include_ignored)
            .git_exclude(!include_ignored)
            .git_global(!include_ignored)
            .parents(!include_ignored);

        let mut hits: Vec<String> = Vec::new();
        let mut truncated = false;
        'walk: for entry in walker.build() {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let Ok(rel) = entry.path().strip_prefix(&root) else { continue };
            if let Some(m) = &path_matcher
                && !m.is_match(rel)
            {
                continue;
            }
            match scan_file(entry.path(), &re, max_per_file, max_total - hits.len()) {
                Ok(lines) => {
                    for (line_no, line) in lines {
                        hits.push(format!("{}:{}:{}", rel.display(), line_no, line));
                        if hits.len() >= max_total {
                            truncated = true;
                            break 'walk;
                        }
                    }
                }
                // Silently skip files we couldn't open or decide are binary.
                Err(_) => continue,
            }
        }
        (hits, truncated)
    })
    .await;
    let (hits, truncated) = match result {
        Ok(t) => t,
        Err(e) => return CallToolResult::error_text(format!("grep task panicked: {e}")),
    };

    if hits.is_empty() {
        return CallToolResult::text(format!("(no matches for {})", parsed.pattern));
    }
    let mut out = hits.join("\n");
    out.push('\n');
    if truncated {
        out.push_str(&format!(
            "(truncated at {max_total} total hits; narrow the pattern, use path_glob, or raise max_results)\n"
        ));
    }
    CallToolResult::text(out)
}

/// Scan one file's lines against `re`. Returns (line_no, line_text) tuples.
///
/// - Skips files that look binary (null byte in the first 8 KiB).
/// - Caps per-file hits at `per_file_cap` and never returns more than `remaining_total`.
/// - Truncates very long lines at 4 KiB so a pathological file can't wedge the search.
fn scan_file(
    path: &std::path::Path,
    re: &regex::Regex,
    per_file_cap: usize,
    remaining_total: usize,
) -> std::io::Result<Vec<(usize, String)>> {
    use std::io::{BufRead, BufReader, Read};
    let mut f = std::fs::File::open(path)?;
    let mut head = [0u8; 8192];
    let n = f.read(&mut head)?;
    if head[..n].contains(&0u8) {
        return Ok(Vec::new()); // binary; skip
    }
    // Reopen so we can stream-read from the start.
    let f = std::fs::File::open(path)?;
    let reader = BufReader::new(f);
    let cap = per_file_cap.min(remaining_total);
    let mut out: Vec<(usize, String)> = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(_) => return Ok(out), // non-UTF8 line — stop scanning this file
        };
        if re.is_match(&line) {
            const MAX_LINE: usize = 4096;
            let trimmed = if line.len() > MAX_LINE {
                let mut s = line[..MAX_LINE].to_string();
                s.push('…');
                s
            } else {
                line
            };
            out.push((idx + 1, trimmed));
            if out.len() >= cap {
                break;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_ansi_removes_color_codes() {
        // Representative cargo-style output with SGR color/reset and bold.
        let input = "\x1b[1m\x1b[31merror\x1b[0m: \x1b[1mcannot find value `foo`\x1b[0m";
        assert_eq!(strip_ansi(input), "error: cannot find value `foo`");
    }

    #[test]
    fn strip_ansi_removes_cursor_and_erase_codes() {
        // Progress bars: cursor-up, erase-line, carriage return left intact.
        let input = "\r\x1b[2K\x1b[1A Downloading crates …";
        assert_eq!(strip_ansi(input), "\r Downloading crates …");
    }

    #[test]
    fn strip_ansi_passes_through_plain_text() {
        let input = "Compiling foo v0.1.0\nFinished dev profile\n";
        assert_eq!(strip_ansi(input), input);
    }

    #[test]
    fn strip_ansi_handles_simple_two_char_escapes() {
        // ESC + letter (e.g. alternate charset, keypad modes).
        let input = "hello\x1bMworld\x1b=done";
        assert_eq!(strip_ansi(input), "helloworlddone");
    }

    #[test]
    fn head_truncate_no_op_when_under_cap() {
        let s = "short output".to_string();
        assert_eq!(head_truncate(s.clone(), 1000), s);
    }

    #[test]
    fn head_truncate_keeps_tail_with_marker() {
        let s: String = "abcdefghij".into(); // 10 bytes
        let got = head_truncate(s, 4);
        // Keeps last 4 bytes, notes 6 dropped.
        assert_eq!(got, "... 6 bytes truncated ...\nghij");
    }

    #[test]
    fn head_truncate_walks_forward_to_char_boundary() {
        // 3-byte é repeated — if we cut mid-codepoint, walk forward.
        let s: String = "éééé".into(); // 4 × 2 bytes = 8 bytes
        let got = head_truncate(s, 5);
        // Cutting to keep 5 trailing bytes would start mid-é; we walk forward
        // to the next boundary (byte 4) and report 4 dropped.
        assert_eq!(got, "... 4 bytes truncated ...\néé");
    }

    #[test]
    fn nearest_match_hint_finds_best_window_with_indent_drift() {
        let content = "\
fn main() {
    let x = 10;
    let y = 20;
    println!(\"hi\");
}
";
        // Model guessed 4-space indent but the file happens to have 4-space
        // too. The whitespace-trim match catches it even when we throw in
        // slightly different leading whitespace.
        let old = "    let x = 99;\n    let y = 20;\n";
        let hint = nearest_match_hint(content, old);
        assert!(hint.contains("Closest matching region"));
        assert!(hint.contains("lines 2-3"));
        // The marker '>' should precede the lines inside the window.
        assert!(hint.contains(">     2"));
        assert!(hint.contains(">     3"));
    }

    #[test]
    fn nearest_match_hint_empty_when_window_bigger_than_file() {
        let content = "just one line\n";
        let old = "five\nline\nold\nstring\nhere\n";
        let hint = nearest_match_hint(content, old);
        assert_eq!(hint, "");
    }

    #[test]
    fn nearest_match_hint_reports_no_match() {
        let content = "totally unrelated content\nwith three lines\nof nothing\n";
        let old = "fn example() {\n    unused();\n}\n";
        let hint = nearest_match_hint(content, old);
        assert!(hint.contains("No closely-matching region"));
    }
}
