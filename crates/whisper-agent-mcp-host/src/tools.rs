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
    vec![read_file_descriptor(), write_file_descriptor(), bash_descriptor()]
}

pub async fn call(workspace: &Arc<Workspace>, name: &str, args: Value) -> Result<CallToolResult, ToolDispatchError> {
    match name {
        "read_file" => Ok(read_file(workspace, args).await),
        "write_file" => Ok(write_file(workspace, args).await),
        "bash" => Ok(bash(workspace, args).await),
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
