//! The three MVP tools: read_file, write_file, bash.
//!
//! Each tool returns a [`CallToolResult`]. Tool-level errors (file not found, command failed)
//! are returned as `is_error: true` results — not JSON-RPC errors. JSON-RPC errors are reserved
//! for protocol-level problems (unknown tool, bad arguments).

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

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
        bash_descriptor(),
        list_dir_descriptor(),
        glob_descriptor(),
    ]
}

pub async fn call(workspace: &Arc<Workspace>, name: &str, args: Value) -> Result<CallToolResult, ToolDispatchError> {
    match name {
        "read_file" => Ok(read_file(workspace, args).await),
        "write_file" => Ok(write_file(workspace, args).await),
        "bash" => Ok(bash(workspace, args).await),
        "list_dir" => Ok(list_dir(workspace, args).await),
        "glob" => Ok(glob(workspace, args).await),
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
        description: "Read the contents of a UTF-8 text file within the workspace.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the workspace root."
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
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => CallToolResult::text(content),
        Err(e) => CallToolResult::error_text(format!("read_file({}): {e}", parsed.path.display())),
    }
}

// ---------- write_file ----------

fn write_file_descriptor() -> Tool {
    Tool {
        name: "write_file".into(),
        description: "Write UTF-8 text to a file within the workspace, creating parent directories as needed. Overwrites if the file exists.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path relative to the workspace root." },
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

// ---------- bash ----------

fn bash_descriptor() -> Tool {
    Tool {
        name: "bash".into(),
        description: "Run a bash command within the workspace. Returns stdout, stderr, and exit code on completion.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Command to run via `bash -c`." },
                "cwd": { "type": "string", "description": "Optional cwd relative to the workspace root. Defaults to the workspace root." },
                "timeout_seconds": { "type": "integer", "description": "Kill the command after this many seconds. Default 120, max 600." }
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

#[derive(Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    timeout_seconds: Option<u64>,
}

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

    let mut cmd = Command::new("bash");
    cmd.arg("-c")
        .arg(&parsed.command)
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return CallToolResult::error_text(format!("spawn bash: {e}")),
    };

    let output_future = child.wait_with_output();
    match timeout(Duration::from_secs(timeout_secs), output_future).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);
            let body = format!(
                "exit_code: {exit_code}\n--- stdout ---\n{stdout}--- stderr ---\n{stderr}"
            );
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
                    "description": "Directory path relative to the workspace root. Defaults to \".\"."
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
