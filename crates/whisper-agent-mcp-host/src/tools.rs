//! The three MVP tools: read_file, write_file, bash.
//!
//! Each tool returns a [`CallToolResult`]. Tool-level errors (file not found, command failed)
//! are returned as `is_error: true` results — not JSON-RPC errors. JSON-RPC errors are reserved
//! for protocol-level problems (unknown tool, bad arguments).

use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use async_stream::stream;
use futures::Stream;
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::{Instant, timeout_at};

use whisper_agent_mcp_proto::{CallToolResult, ContentBlock, Tool, ToolAnnotations};

use crate::workspace::Workspace;

/// One item from a streaming tool call.
///
/// A tool that has no incremental output yields exactly one `Final`. A tool
/// that streams (today: `bash`) yields zero or more `Chunk` items followed by
/// exactly one `Final`. The `Final` carries the complete, definitive result
/// the model will see in its next turn — `Chunk`s are purely for live UI.
pub enum ToolStreamItem {
    Chunk(ContentBlock),
    Final(CallToolResult),
}

pub fn descriptors() -> Vec<Tool> {
    vec![
        read_file_descriptor(),
        view_image_descriptor(),
        view_pdf_descriptor(),
        write_file_descriptor(),
        edit_file_descriptor(),
        bash_descriptor(),
        list_dir_descriptor(),
        glob_descriptor(),
        grep_descriptor(),
        crate_source_descriptor(),
    ]
}

/// Dispatch a tool call as a stream of [`ToolStreamItem`]s. All tools
/// terminate with exactly one `Final`. Streaming tools (bash) yield
/// `Chunk`s before the `Final`; non-streaming tools yield only the
/// `Final`.
pub fn call_stream(
    workspace: &Arc<Workspace>,
    name: &str,
    args: Value,
) -> Result<Pin<Box<dyn Stream<Item = ToolStreamItem> + Send>>, ToolDispatchError> {
    let ws = workspace.clone();
    match name {
        "bash" => Ok(bash_stream(ws, args)),
        "read_file" => Ok(single(async move { read_file(&ws, args).await })),
        "view_image" => Ok(single(async move { view_image(&ws, args).await })),
        "view_pdf" => Ok(single(async move { view_pdf(&ws, args).await })),
        "write_file" => Ok(single(async move { write_file(&ws, args).await })),
        "edit_file" => Ok(single(async move { edit_file(&ws, args).await })),
        "list_dir" => Ok(single(async move { list_dir(&ws, args).await })),
        "glob" => Ok(single(async move { glob(&ws, args).await })),
        "grep" => Ok(single(async move { grep(&ws, args).await })),
        "crate_source" => Ok(single(crate_source(args))),
        _ => Err(ToolDispatchError::UnknownTool(name.to_string())),
    }
}

/// Wrap a one-shot tool's future as a single-item stream yielding `Final`.
fn single<F>(fut: F) -> Pin<Box<dyn Stream<Item = ToolStreamItem> + Send>>
where
    F: std::future::Future<Output = CallToolResult> + Send + 'static,
{
    Box::pin(stream! {
        let result = fut.await;
        yield ToolStreamItem::Final(result);
    })
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
    // Refuse before we send the model a stream of garbage. A binary file
    // that happens to mostly decode as UTF-8 (or a giant minified blob)
    // can wedge a model into a repetitive loop trying to make sense of
    // it. The null-byte sniff is the same check `grep` uses to skip
    // binaries — see `scan_file` below.
    if let Some(kind) = sniff_binary(&path).await {
        return CallToolResult::error_text(refuse_binary_message(&parsed.path, kind));
    }
    // Whole-file fast path when no line slicing is requested.
    if parsed.offset.is_none() && parsed.limit.is_none() {
        return match tokio::fs::read_to_string(&path).await {
            Ok(content) => CallToolResult::text(content),
            Err(e) => {
                CallToolResult::error_text(format!("read_file({}): {e}", parsed.path.display()))
            }
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

/// What kind of binary content we sniffed at the head of a file.
/// Drives the steering message read_file emits when refusing.
#[derive(Debug, Clone, Copy)]
enum BinaryKind {
    Image(image::ImageFormat),
    Pdf,
    Other,
}

/// First four bytes of a PDF file — `%PDF`. Standard PDFs start with
/// the version line `%PDF-1.x`; older "linearized" PDFs may have
/// leading binary garbage but the magic is in the first ~1 KiB. We
/// only sniff the very first bytes for simplicity.
const PDF_MAGIC: &[u8] = b"%PDF-";

/// Read the first 8 KiB of `path` and decide whether the file is a
/// binary blob that read_file should refuse. Returns `None` when the
/// head looks like text (no NUL byte and no PDF magic) or when the
/// file can't be opened (we let the existing read_to_string error
/// path produce that message). When the head *is* binary, tries to
/// name a known format so the steering message can point at the
/// right tool.
async fn sniff_binary(path: &std::path::Path) -> Option<BinaryKind> {
    use tokio::io::AsyncReadExt;
    let mut f = tokio::fs::File::open(path).await.ok()?;
    let mut head = [0u8; 8192];
    let n = f.read(&mut head).await.ok()?;
    let head = &head[..n];
    // PDFs are technically all-printable-ASCII at the head (no NUL),
    // so we check the magic bytes explicitly.
    if head.starts_with(PDF_MAGIC) {
        return Some(BinaryKind::Pdf);
    }
    if !head.contains(&0u8) {
        return None;
    }
    let kind = match image::guess_format(head) {
        Ok(fmt) => BinaryKind::Image(fmt),
        Err(_) => BinaryKind::Other,
    };
    Some(kind)
}

/// Steering message returned when `read_file` refuses a binary file.
/// Names `view_image` / `view_pdf` explicitly so the model knows
/// where to go instead of retrying read_file with new offsets.
fn refuse_binary_message(display: &std::path::Path, kind: BinaryKind) -> String {
    match kind {
        BinaryKind::Image(fmt) => format!(
            "read_file({}): file looks like a binary image (format: {:?}) — use `view_image` to load it as an image attachment instead",
            display.display(),
            fmt,
        ),
        BinaryKind::Pdf => format!(
            "read_file({}): file looks like a PDF — use `view_pdf` to attach it as a document the model can analyze",
            display.display(),
        ),
        BinaryKind::Other => format!(
            "read_file({}): file looks binary (NUL bytes in first 8 KiB); read_file is for UTF-8 text only",
            display.display(),
        ),
    }
}

// ---------- view_image ----------

/// Maximum image dimension on either axis. Anthropic accepts up to
/// 8000px and OpenAI up to 4096px on Chat Completions, but token cost
/// scales with pixel area — 2048 keeps the input cheap while staying
/// well above readable resolution. Matches Codex's
/// `codex-utils-image::MAX_DIMENSION`.
const VIEW_IMAGE_MAX_DIMENSION: u32 = 2048;

/// Refuse images bigger than this on disk. Generous enough for any
/// legitimate screenshot or photo; small enough to prevent the agent
/// from accidentally decoding a multi-gigabyte file (which would still
/// resize down to 2048px afterwards but blows up RAM during decode).
const VIEW_IMAGE_MAX_BYTES: u64 = 50 * 1024 * 1024;

fn view_image_descriptor() -> Tool {
    Tool {
        name: "view_image".into(),
        description: "Load an image file from the workspace and attach it to the conversation \
                      so the model can see it. Use this whenever you need to look at a PNG / \
                      JPEG / WebP / GIF — `read_file` refuses binary files. Large images are \
                      resized to 2048px on the longest side to keep token cost bounded; \
                      smaller images pass through untouched."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the image — absolute, or relative to the workspace root."
                }
            },
            "required": ["path"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("View image".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct ViewImageArgs {
    path: PathBuf,
}

async fn view_image(workspace: &Arc<Workspace>, args: Value) -> CallToolResult {
    let parsed: ViewImageArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };
    let path = match workspace.resolve(&parsed.path) {
        Ok(p) => p,
        Err(e) => return CallToolResult::error_text(e.to_string()),
    };
    let metadata = match tokio::fs::metadata(&path).await {
        Ok(m) => m,
        Err(e) => {
            return CallToolResult::error_text(format!(
                "view_image({}): {e}",
                parsed.path.display()
            ));
        }
    };
    if !metadata.is_file() {
        return CallToolResult::error_text(format!(
            "view_image({}): not a regular file",
            parsed.path.display()
        ));
    }
    if metadata.len() > VIEW_IMAGE_MAX_BYTES {
        return CallToolResult::error_text(format!(
            "view_image({}): file is {} bytes; cap is {} bytes",
            parsed.path.display(),
            metadata.len(),
            VIEW_IMAGE_MAX_BYTES,
        ));
    }
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) => {
            return CallToolResult::error_text(format!(
                "view_image({}): {e}",
                parsed.path.display()
            ));
        }
    };
    // Decode + maybe-resize on a blocking pool — the `image` crate is
    // CPU-bound and we don't want to stall the host's tokio executor on
    // a multi-megabyte decode.
    let display = parsed.path.clone();
    let prepared = tokio::task::spawn_blocking(move || prepare_image(&bytes)).await;
    let prepared = match prepared {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            return CallToolResult::error_text(format!("view_image({}): {e}", display.display()));
        }
        Err(e) => {
            return CallToolResult::error_text(format!(
                "view_image({}): decode task panicked: {e}",
                display.display()
            ));
        }
    };
    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&prepared.bytes);
    CallToolResult::image(encoded, prepared.mime_type)
}

/// Output of [`prepare_image`]: bytes ready for base64 + the IANA MIME
/// the agent runtime should advertise to the model.
#[derive(Debug)]
struct PreparedImage {
    bytes: Vec<u8>,
    mime_type: &'static str,
}

/// Decode `bytes`, downscale to [`VIEW_IMAGE_MAX_DIMENSION`] if either
/// dimension exceeds it, otherwise pass the original bytes through
/// unchanged. Returns an error string when the format isn't one we
/// have features for or when decode fails.
///
/// Mirrors Codex's `load_for_prompt_bytes` shape — keep originals when
/// the image is already in a supported format and small enough,
/// re-encode as PNG (lossless) on the resize path so we don't double-
/// quantize JPEG.
fn prepare_image(bytes: &[u8]) -> Result<PreparedImage, String> {
    use image::ImageFormat;
    let format =
        image::guess_format(bytes).map_err(|_| "could not detect image format".to_string())?;
    let mime = match format {
        ImageFormat::Png => "image/png",
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::WebP => "image/webp",
        ImageFormat::Gif => "image/gif",
        other => {
            return Err(format!(
                "unsupported image format {other:?} (accepted: PNG, JPEG, WebP, GIF)"
            ));
        }
    };
    let decoded = image::load_from_memory_with_format(bytes, format)
        .map_err(|e| format!("decode failed: {e}"))?;
    let (w, h) = (decoded.width(), decoded.height());
    if w.max(h) <= VIEW_IMAGE_MAX_DIMENSION {
        return Ok(PreparedImage {
            bytes: bytes.to_vec(),
            mime_type: mime,
        });
    }
    let resized = decoded.resize(
        VIEW_IMAGE_MAX_DIMENSION,
        VIEW_IMAGE_MAX_DIMENSION,
        image::imageops::FilterType::Triangle,
    );
    let mut out = Vec::with_capacity(bytes.len());
    resized
        .write_to(&mut std::io::Cursor::new(&mut out), ImageFormat::Png)
        .map_err(|e| format!("re-encode after resize failed: {e}"))?;
    Ok(PreparedImage {
        bytes: out,
        mime_type: "image/png",
    })
}

// ---------- view_pdf ----------

/// Refuse PDFs bigger than this on disk. Anthropic's API caps a single
/// request at 32 MB, OpenAI at 50 MB, Gemini takes much larger files
/// but the others are the binding constraint. We pick a value safely
/// under the smallest cap to leave headroom for the rest of the
/// request payload.
const VIEW_PDF_MAX_BYTES: u64 = 25 * 1024 * 1024;

fn view_pdf_descriptor() -> Tool {
    Tool {
        name: "view_pdf".into(),
        description: "Load a PDF file from the workspace and attach it to the conversation so \
                      the model can analyze its text + visual layout (charts, tables, diagrams). \
                      `read_file` refuses PDFs because they're binary; this is the right tool \
                      for them. Each provider does its own page rasterization + text \
                      extraction server-side, so the model sees both modalities."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF — absolute, or relative to the workspace root."
                }
            },
            "required": ["path"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("View PDF".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct ViewPdfArgs {
    path: PathBuf,
}

async fn view_pdf(workspace: &Arc<Workspace>, args: Value) -> CallToolResult {
    let parsed: ViewPdfArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };
    let path = match workspace.resolve(&parsed.path) {
        Ok(p) => p,
        Err(e) => return CallToolResult::error_text(e.to_string()),
    };
    let metadata = match tokio::fs::metadata(&path).await {
        Ok(m) => m,
        Err(e) => {
            return CallToolResult::error_text(format!("view_pdf({}): {e}", parsed.path.display()));
        }
    };
    if !metadata.is_file() {
        return CallToolResult::error_text(format!(
            "view_pdf({}): not a regular file",
            parsed.path.display()
        ));
    }
    if metadata.len() > VIEW_PDF_MAX_BYTES {
        return CallToolResult::error_text(format!(
            "view_pdf({}): file is {} bytes; cap is {} bytes",
            parsed.path.display(),
            metadata.len(),
            VIEW_PDF_MAX_BYTES,
        ));
    }
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) => {
            return CallToolResult::error_text(format!("view_pdf({}): {e}", parsed.path.display()));
        }
    };
    if !bytes.starts_with(PDF_MAGIC) {
        return CallToolResult::error_text(format!(
            "view_pdf({}): file does not look like a PDF (missing %PDF- header)",
            parsed.path.display(),
        ));
    }
    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let uri = format!("file://{}", path.display());
    CallToolResult::resource_blob(uri, "application/pdf", encoded)
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
                "content": { "type": "string", "description": "File contents to write." },
                "run_as": { "type": "string", "description": "Optional Unix username to write the file as. Parent directories are still created with the host's identity if missing; the leaf file itself is written by a child process running as the target user (so the file's owner is that user). Fails the call if the host lacks the privilege to assume that user." }
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
    #[serde(default)]
    run_as: Option<String>,
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
        && let Err(e) = tokio::fs::create_dir_all(parent).await
    {
        return CallToolResult::error_text(format!("create_dir_all({}): {e}", parent.display()));
    }
    #[cfg(unix)]
    if let Some(name) = parsed.run_as.as_deref() {
        return write_file_as_user(name, &path, &parsed.path, parsed.content.as_bytes()).await;
    }
    match tokio::fs::write(&path, parsed.content.as_bytes()).await {
        Ok(()) => CallToolResult::text(format!(
            "wrote {} bytes to {}",
            parsed.content.len(),
            parsed.path.display()
        )),
        Err(e) => CallToolResult::error_text(format!("write_file({}): {e}", parsed.path.display())),
    }
}

/// Write `content` to `abs_path` as `user`. Spawns `tee <abs_path>` with
/// `pre_exec` setuid so the leaf file's owner is the target user; the
/// host-resolved `display_path` is only used for the success/error
/// message so the model sees the same path it asked for.
#[cfg(unix)]
async fn write_file_as_user(
    user: &str,
    abs_path: &std::path::Path,
    display_path: &std::path::Path,
    content: &[u8],
) -> CallToolResult {
    use tokio::io::AsyncWriteExt;

    let id = match crate::runas::lookup(user) {
        Ok(id) => id,
        Err(e) => return CallToolResult::error_text(format!("run_as `{user}`: {e}")),
    };

    let mut cmd = Command::new("tee");
    cmd.arg(abs_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .env("HOME", &id.home)
        .env("USER", &id.name)
        .env("LOGNAME", &id.name)
        .kill_on_drop(true);
    let id_for_child = id.clone();
    // Safety: closure runs in the forked child between fork and exec,
    // so it only changes the child's identity. Calls only async-signal-
    // safe libc routines (setgroups/setgid/setuid).
    unsafe {
        cmd.pre_exec(move || crate::runas::apply_in_child(&id_for_child));
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return CallToolResult::error_text(format!("spawn tee as {}: {e}", id.name));
        }
    };

    let mut stdin = child.stdin.take().expect("stdin piped");
    if let Err(e) = stdin.write_all(content).await {
        return CallToolResult::error_text(format!("write to tee stdin: {e}"));
    }
    drop(stdin); // close to signal EOF so tee exits

    let output = match child.wait_with_output().await {
        Ok(o) => o,
        Err(e) => return CallToolResult::error_text(format!("wait tee: {e}")),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);
        return CallToolResult::error_text(format!(
            "tee as {} exited {code}: {}",
            id.name,
            stderr.trim()
        ));
    }
    CallToolResult::text(format!(
        "wrote {} bytes to {} as {}",
        content.len(),
        display_path.display(),
        id.name
    ))
}

// ---------- edit_file ----------

fn edit_file_descriptor() -> Tool {
    Tool {
        name: "edit_file".into(),
        description: "Replace text in a file. `old_string` must match literally — no regex. \
                      By default requires exactly one match; pass `replace_all: true` to \
                      change every occurrence. If old_string isn't found, the error shows the \
                      closest matching region of the file; if it matches multiple times, the \
                      error lists each match site with line numbers so you can extend \
                      old_string by unique surrounding text. When you extend old_string to \
                      disambiguate, extend new_string by the same neighboring text — \
                      extending only old_string will delete the intervening lines from the \
                      output. Prefer this over write_file for targeted changes."
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
        let hint = multi_match_hint(&content, &parsed.old_string);
        return CallToolResult::error_text(format!(
            "edit_file({}): old_string matches {found} times. Either extend old_string AND \
             new_string with the same surrounding context to make the target unique (extend \
             both by identical text — extending only old_string will delete the intervening \
             lines), or pass replace_all:true to replace every occurrence.{hint}",
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
        Err(e) => CallToolResult::error_text(format!("edit_file({}): {e}", parsed.path.display())),
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
/// model can extend `old_string` AND `new_string` by unique neighboring
/// text. Caps at 3 matches — the model only needs enough to tell the
/// targets apart.
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

    // Literal compare — match `content.matches(old_string)`'s semantics.
    let mut hits: Vec<usize> = Vec::new();
    for start in 0..=(file_lines.len() - window_size) {
        if (0..window_size).all(|i| file_lines[start + i] == old_lines[i]) {
            hits.push(start);
        }
    }
    if hits.len() < 2 {
        // When old_string starts mid-line, `content.matches` may count
        // more hits than the line-oriented search finds. Skip the hint
        // rather than mislead the model.
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
                "strip_ansi": { "type": "boolean", "description": "Strip ANSI escape sequences (colors, cursor codes) from stdout and stderr. Default true — only turn off if you specifically need the raw escape bytes." },
                "run_as": { "type": "string", "description": "Optional Unix username to run the command as. The host drops to that user's uid/gid/groups before exec, sets HOME/USER/LOGNAME accordingly, and fails the call if it lacks the privilege to assume that user. Omit to run with the host's default identity." }
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
    #[serde(default)]
    run_as: Option<String>,
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

/// Per-line cap when reading child stdout. Bounds worst-case memory for a
/// command that emits a single enormous line with no `\n` (pathological
/// — `python -c "print('x'*1_000_000_000)"`), where `read_line` would
/// otherwise buffer the whole thing. Lines longer than this get a
/// `[line truncated, cap=N bytes]` marker appended.
const BASH_MAX_LINE_BYTES: usize = 8_192;

/// Total bytes of streaming `Chunk` events forwarded to the subscriber
/// before we stop emitting and silently drain the rest. Past this point
/// the subscriber has seen enough progress; the final tool_result still
/// carries the 30KB tail the model needs. Keeps client memory and the
/// over-the-wire byte count bounded for a runaway `ls -R /` etc.
const BASH_STREAM_TOTAL_BYTES: usize = 262_144;

/// Spawn bash and stream its merged stdout/stderr line-by-line. Each line
/// becomes a `Chunk(Text)` event the server forwards to the client as an
/// SSE notification; the full accumulated body is returned in the `Final`
/// event so the model (and retention) see the complete output.
///
/// Read one line from `reader` into `out`, stopping at `\n`, EOF, or
/// `max_bytes` of non-newline content — whichever comes first. Returns
/// `Ok(ReadOutcome)` describing the outcome.
///
/// On `Truncated`, `out` was filled to `max_bytes` and the reader was
/// then advanced past the rest of the line (up to and including the
/// next `\n` or EOF) so the following call sees the start of the next
/// line. Bytes beyond the cap are dropped on the floor, not counted —
/// the only guarantee is that the caller's memory doesn't grow with
/// the runaway line.
///
/// Used in place of [`AsyncBufReadExt::read_line`] so a pathological
/// child that prints gigabytes with no `\n` can't OOM the mcp-host
/// process.
async fn read_capped_line<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
    out: &mut Vec<u8>,
    max_bytes: usize,
) -> std::io::Result<ReadOutcome> {
    let start = out.len();
    let mut truncated = false;
    loop {
        let buf = reader.fill_buf().await?;
        if buf.is_empty() {
            if out.len() == start {
                return Ok(ReadOutcome::Eof);
            }
            return Ok(if truncated {
                ReadOutcome::Truncated
            } else {
                ReadOutcome::Line
            });
        }
        let nl_pos = buf.iter().position(|&b| b == b'\n');
        let end_in_buf = nl_pos.map(|p| p + 1).unwrap_or(buf.len());

        if truncated {
            // Already at cap; discard bytes until the line's newline.
            reader.consume(end_in_buf);
            if nl_pos.is_some() {
                return Ok(ReadOutcome::Truncated);
            }
            continue;
        }

        let room = max_bytes.saturating_sub(out.len() - start);
        if end_in_buf <= room {
            out.extend_from_slice(&buf[..end_in_buf]);
            reader.consume(end_in_buf);
            if nl_pos.is_some() {
                return Ok(ReadOutcome::Line);
            }
            continue;
        }
        // The next line chunk would exceed the cap. Fill to cap, then
        // drain the rest of the line below.
        out.extend_from_slice(&buf[..room]);
        reader.consume(end_in_buf);
        truncated = true;
        if nl_pos.is_some() {
            return Ok(ReadOutcome::Truncated);
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ReadOutcome {
    /// `out` was extended by a full line (up to and including `\n`, or
    /// the final line's content before EOF).
    Line,
    /// `out` was filled to `max_bytes` and the remainder of the line
    /// was discarded.
    Truncated,
    /// EOF at start of line — nothing appended.
    Eof,
}

/// Dropping the returned stream drops the child `Command`, which (via
/// `kill_on_drop`) kills bash — so a cancelled HTTP connection aborts
/// the command rather than leaving it running.
fn bash_stream(
    workspace: Arc<Workspace>,
    args: Value,
) -> Pin<Box<dyn Stream<Item = ToolStreamItem> + Send>> {
    Box::pin(stream! {
        let parsed: BashArgs = match serde_json::from_value(args) {
            Ok(v) => v,
            Err(e) => {
                yield ToolStreamItem::Final(CallToolResult::error_text(
                    format!("invalid arguments: {e}"),
                ));
                return;
            }
        };

        let cwd = match parsed.cwd.as_deref() {
            None => workspace.root().to_path_buf(),
            Some(rel) => match workspace.resolve(rel) {
                Ok(p) => p,
                Err(e) => {
                    yield ToolStreamItem::Final(CallToolResult::error_text(e.to_string()));
                    return;
                }
            },
        };

        let timeout_secs = parsed.timeout_seconds.unwrap_or(120).min(600);
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);

        // Resolve `run_as` in the parent: NSS lookups aren't async-signal-safe
        // and so can't run inside `pre_exec`. The actual setuid happens in the
        // child between fork and exec; if the host can't assume that user, the
        // libc error surfaces as a clean tool-result error from `cmd.spawn()`.
        #[cfg(unix)]
        let run_as_id = match parsed.run_as.as_deref() {
            None => None,
            Some(name) => match crate::runas::lookup(name) {
                Ok(id) => Some(id),
                Err(e) => {
                    yield ToolStreamItem::Final(CallToolResult::error_text(
                        format!("run_as `{name}`: {e}"),
                    ));
                    return;
                }
            },
        };

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

        #[cfg(unix)]
        if let Some(id) = &run_as_id {
            // Override the env that bash and most CLI tools read to identify
            // "who am I". Without these, e.g. `~` expansion would still point
            // at the host user's home.
            cmd.env("HOME", &id.home)
                .env("USER", &id.name)
                .env("LOGNAME", &id.name);
            let id_for_child = id.clone();
            // Safety: the closure runs in the forked child between fork and
            // exec, so it only changes the child's identity. It calls only
            // async-signal-safe libc routines (setgroups/setgid/setuid).
            unsafe {
                cmd.pre_exec(move || crate::runas::apply_in_child(&id_for_child));
            }
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                yield ToolStreamItem::Final(CallToolResult::error_text(
                    format!("spawn bash: {e}"),
                ));
                return;
            }
        };

        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
        let mut stdout_reader = BufReader::new(stdout);
        let mut stderr_reader = BufReader::new(stderr);

        let mut accumulated = String::new();
        let mut streamed_bytes: usize = 0;
        let mut stream_cap_announced = false;
        let mut timed_out = false;

        // Stream stdout (= merged child output via shell redirection) line-by-line
        // until EOF or timeout. stderr from *bash itself* is drained afterwards
        // so we never block on it while the child is running.
        //
        // Three caps keep a runaway command from blowing up memory or flooding
        // the subscriber:
        //   * `read_capped_line` bounds per-line buffering at
        //     `BASH_MAX_LINE_BYTES` — a child emitting gigabytes without a `\n`
        //     can't OOM us.
        //   * `accumulated` is re-head-truncated to `BASH_MAX_OUTPUT_BYTES` once
        //     it grows past 2x that — the final `head_truncate` below wants the
        //     tail, so trimming early is lossless.
        //   * Past `BASH_STREAM_TOTAL_BYTES` of emitted chunks we stop yielding
        //     chunks (after one trailing marker) but keep draining the child so
        //     its stdout pipe doesn't block.
        loop {
            let mut line_bytes: Vec<u8> = Vec::new();
            tokio::select! {
                res = read_capped_line(&mut stdout_reader, &mut line_bytes, BASH_MAX_LINE_BYTES) => {
                    let outcome = match res {
                        Ok(o) => o,
                        Err(e) => {
                            yield ToolStreamItem::Final(CallToolResult::error_text(
                                format!("bash read failed: {e}"),
                            ));
                            return;
                        }
                    };
                    if outcome == ReadOutcome::Eof {
                        break;
                    }
                    // Decode — child output isn't guaranteed UTF-8 (tools that
                    // emit binary, locale mismatches). `from_utf8_lossy`
                    // substitutes the replacement char for invalid sequences
                    // rather than failing the whole tool call.
                    let mut emitted = String::from_utf8_lossy(&line_bytes).into_owned();
                    if outcome == ReadOutcome::Truncated {
                        if !emitted.ends_with('\n') {
                            emitted.push('\n');
                        }
                        emitted.insert_str(
                            emitted.len() - 1,
                            &format!(" [line truncated at {BASH_MAX_LINE_BYTES} bytes]"),
                        );
                    }
                    if parsed.strip_ansi {
                        emitted = strip_ansi(&emitted);
                    }
                    accumulated.push_str(&emitted);
                    // Keep `accumulated` bounded so a long-running command
                    // can't grow us past ~2x the final cap. Trim leaves the
                    // last `BASH_MAX_OUTPUT_BYTES` on a char boundary; final
                    // `head_truncate` handles the rest.
                    if accumulated.len() > 2 * BASH_MAX_OUTPUT_BYTES {
                        let mut start = accumulated.len() - BASH_MAX_OUTPUT_BYTES;
                        while start < accumulated.len() && !accumulated.is_char_boundary(start) {
                            start += 1;
                        }
                        accumulated.drain(..start);
                    }
                    if streamed_bytes + emitted.len() <= BASH_STREAM_TOTAL_BYTES {
                        streamed_bytes += emitted.len();
                        yield ToolStreamItem::Chunk(ContentBlock::Text { text: emitted });
                    } else if !stream_cap_announced {
                        stream_cap_announced = true;
                        streamed_bytes = BASH_STREAM_TOTAL_BYTES; // clamp for the log
                        yield ToolStreamItem::Chunk(ContentBlock::Text {
                            text: format!(
                                "\n... [streaming cap reached at {BASH_STREAM_TOTAL_BYTES} bytes; \
                                 command still running, final tool_result will show the tail] ...\n"
                            ),
                        });
                    }
                    // Past the cap: silently drain `accumulated` (it stays
                    // bounded above) so bash's stdout pipe doesn't fill.
                }
                _ = tokio::time::sleep_until(deadline) => {
                    timed_out = true;
                    break;
                }
            }
        }

        if timed_out {
            // kill_on_drop will reap bash when `child` drops at scope end; start
            // the kill explicitly so we don't wait on a hung child.
            let _ = child.start_kill();
            yield ToolStreamItem::Final(CallToolResult::error_text(
                format!("bash timed out after {timeout_secs}s"),
            ));
            return;
        }

        // Drain bash's own stderr (wrapper-parse errors, etc.) before waiting.
        // Same per-line and total caps as stdout: this channel is practically
        // always tiny (bash complaints about the wrapper script), but a
        // misbehaving shell replacement has no reason to bring down the host.
        let mut bash_stderr = String::new();
        loop {
            let mut chunk_bytes: Vec<u8> = Vec::new();
            match timeout_at(
                deadline,
                read_capped_line(&mut stderr_reader, &mut chunk_bytes, BASH_MAX_LINE_BYTES),
            )
            .await
            {
                Ok(Ok(ReadOutcome::Eof)) => break,
                Ok(Ok(_)) => {
                    bash_stderr.push_str(&String::from_utf8_lossy(&chunk_bytes));
                    if bash_stderr.len() >= BASH_MAX_OUTPUT_BYTES {
                        break;
                    }
                }
                Ok(Err(_)) | Err(_) => break,
            }
        }

        let status = match timeout_at(deadline, child.wait()).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                yield ToolStreamItem::Final(CallToolResult::error_text(
                    format!("bash wait failed: {e}"),
                ));
                return;
            }
            Err(_) => {
                let _ = child.start_kill();
                yield ToolStreamItem::Final(CallToolResult::error_text(
                    format!("bash timed out after {timeout_secs}s"),
                ));
                return;
            }
        };

        let mut merged = accumulated;
        if !bash_stderr.is_empty() {
            if parsed.strip_ansi {
                merged.push_str(&strip_ansi(&bash_stderr));
            } else {
                merged.push_str(&bash_stderr);
            }
        }
        let merged = head_truncate(merged, BASH_MAX_OUTPUT_BYTES);
        let exit_code = status.code().unwrap_or(-1);
        let body = if status.success() {
            merged
        } else {
            format!("Exit code {exit_code}\n{merged}")
        };
        let result = if status.success() {
            CallToolResult::text(body)
        } else {
            CallToolResult::error_text(body)
        };
        yield ToolStreamItem::Final(result);
    })
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
            let Ok(rel) = entry.path().strip_prefix(&root) else {
                continue;
            };
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
            let Ok(rel) = entry.path().strip_prefix(&root) else {
                continue;
            };
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

// ---------- crate_source ----------

fn crate_source_descriptor() -> Tool {
    Tool {
        name: "crate_source".into(),
        description: "Resolve a Rust crate name (and optional version) to its vendored source \
                      directory in the local Cargo registry. Returns the absolute path so you \
                      can `read_file` / `grep` / `list_dir` / `glob` against it to study type \
                      definitions, doc comments, and examples. Only resolves crates already \
                      downloaded under $CARGO_HOME/registry/src — which is every dep of any \
                      workspace you've built, plus anything touched by `cargo fetch`. No \
                      network; if the crate isn't vendored, build a workspace that depends on \
                      it or run `cargo fetch -p <name>` first."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Crate name as it appears on crates.io (e.g. \"tokio\", \
                                    \"serde_json\"). Underscores vs hyphens matter."
                },
                "version": {
                    "type": "string",
                    "description": "Exact version, e.g. \"1.48.0\". Omit to pick the highest \
                                    vendored version; other available versions are listed in \
                                    the response."
                }
            },
            "required": ["crate_name"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Resolve Rust crate source".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct CrateSourceArgs {
    crate_name: String,
    #[serde(default)]
    version: Option<String>,
}

async fn crate_source(args: Value) -> CallToolResult {
    let parsed: CrateSourceArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };
    let name = parsed.crate_name.trim().to_string();
    if name.is_empty() {
        return CallToolResult::error_text("crate_source: crate_name must be non-empty");
    }

    let cargo_home = match resolve_cargo_home() {
        Some(p) => p,
        None => {
            return CallToolResult::error_text(
                "crate_source: neither CARGO_HOME nor HOME is set; cannot locate cargo registry",
            );
        }
    };
    let registry_src = cargo_home.join("registry").join("src");
    let requested_version = parsed.version.clone();
    let scan_name = name.clone();

    // Directory walking is sync; shunt to the blocking pool to keep the runtime clean.
    let scan = tokio::task::spawn_blocking(move || {
        scan_registry_for_crate(&registry_src, &scan_name).map(|hits| (registry_src, hits))
    })
    .await;
    let (registry_src, candidates) = match scan {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => return CallToolResult::error_text(format!("crate_source scan: {e}")),
        Err(e) => return CallToolResult::error_text(format!("crate_source task panicked: {e}")),
    };

    if !registry_src.exists() {
        return CallToolResult::error_text(format!(
            "crate_source: {} does not exist — no crates have been vendored",
            registry_src.display()
        ));
    }
    if candidates.is_empty() {
        return CallToolResult::error_text(format!(
            "crate_source: no vendored copy of `{name}` under {}. \
             Run `cargo fetch -p {name}` or build a workspace that depends on it.",
            registry_src.display()
        ));
    }

    let chosen = match &requested_version {
        Some(v) => match candidates.iter().find(|(ver, _)| ver == v) {
            Some(c) => c.clone(),
            None => {
                let mut versions: Vec<&str> = candidates.iter().map(|(v, _)| v.as_str()).collect();
                versions.sort_by(|a, b| compare_versions(b, a));
                return CallToolResult::error_text(format!(
                    "crate_source: `{name}@{v}` not vendored. Available: {}",
                    versions.join(", ")
                ));
            }
        },
        None => {
            let mut sorted = candidates.clone();
            sorted.sort_by(|a, b| compare_versions(&b.0, &a.0));
            sorted[0].clone()
        }
    };

    let (version, abs_path) = chosen;
    let listing = match brief_listing(&abs_path) {
        Ok(s) => s,
        Err(e) => format!("(could not list directory: {e})\n"),
    };

    let mut out = format!("{name} v{version}\n{}\n", abs_path.display());
    if requested_version.is_none() && candidates.len() > 1 {
        let mut others: Vec<&str> = candidates
            .iter()
            .map(|(v, _)| v.as_str())
            .filter(|v| *v != version)
            .collect();
        others.sort_by(|a, b| compare_versions(b, a));
        out.push_str(&format!("Other vendored versions: {}\n", others.join(", ")));
    }
    out.push_str("\nTop-level entries:\n");
    out.push_str(&listing);
    CallToolResult::text(out)
}

fn resolve_cargo_home() -> Option<PathBuf> {
    if let Ok(h) = std::env::var("CARGO_HOME")
        && !h.is_empty()
    {
        return Some(PathBuf::from(h));
    }
    std::env::var("HOME")
        .ok()
        .filter(|h| !h.is_empty())
        .map(|h| PathBuf::from(h).join(".cargo"))
}

/// Walk `registry_src/*` (one dir per configured registry) and collect every
/// `<name>-<version>` entry. Version must begin with a digit so that e.g. a
/// query for `serde` doesn't accidentally pick up `serde_json-1.0.0`.
fn scan_registry_for_crate(
    registry_src: &std::path::Path,
    name: &str,
) -> std::io::Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    let prefix = format!("{name}-");
    let outer = match std::fs::read_dir(registry_src) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e),
    };
    for entry in outer {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let reg_path = entry.path();
        let inner = match std::fs::read_dir(&reg_path) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for child in inner {
            let child = match child {
                Ok(c) => c,
                Err(_) => continue,
            };
            if !child.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let fname = child.file_name();
            let fname = fname.to_string_lossy();
            if let Some(ver) = fname.strip_prefix(&prefix)
                && ver.chars().next().is_some_and(|c| c.is_ascii_digit())
            {
                out.push((ver.to_string(), child.path()));
            }
        }
    }
    Ok(out)
}

/// Split version into (numeric parts, suffix). "1.48.0-alpha.2+build" →
/// ([1, 48, 0], "-alpha.2+build"). Non-parseable numeric parts become 0.
fn parse_version(v: &str) -> (Vec<u64>, String) {
    let cut = v.find(['-', '+']).unwrap_or(v.len());
    let nums_part = &v[..cut];
    let suffix = v[cut..].to_string();
    let nums: Vec<u64> = nums_part
        .split('.')
        .map(|p| p.parse().unwrap_or(0))
        .collect();
    (nums, suffix)
}

fn compare_versions(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let (na, sa) = parse_version(a);
    let (nb, sb) = parse_version(b);
    for i in 0..na.len().max(nb.len()) {
        let x = na.get(i).copied().unwrap_or(0);
        let y = nb.get(i).copied().unwrap_or(0);
        match x.cmp(&y) {
            Ordering::Equal => continue,
            o => return o,
        }
    }
    // Per semver: empty pre-release suffix > non-empty. "1.0.0" > "1.0.0-rc1".
    match (sa.is_empty(), sb.is_empty()) {
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => sa.cmp(&sb),
    }
}

/// One-line-per-entry summary of a crate's top-level directory. Dirs first,
/// then files; trailing `/` marks dirs; file sizes are shown so the agent can
/// judge whether a `read_file` is cheap or needs an `offset`/`limit`.
fn brief_listing(dir: &std::path::Path) -> std::io::Result<String> {
    let mut entries: Vec<(String, bool, u64)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        let (is_dir, size) = match entry.file_type() {
            Ok(ft) if ft.is_dir() => (true, 0),
            Ok(_) => (false, entry.metadata().map(|m| m.len()).unwrap_or(0)),
            Err(_) => (false, 0),
        };
        entries.push((name, is_dir, size));
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut out = String::new();
    for (name, is_dir, size) in entries {
        if is_dir {
            out.push_str(&format!("d  {:>10}  {name}/\n", ""));
        } else {
            out.push_str(&format!("f  {size:>10}  {name}\n"));
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

    #[test]
    fn multi_match_hint_lists_each_site_with_context() {
        let content = "name = \"test\"\nfoo\n\n[[section]]\nname = \"test\"\nother\n";
        let hint = multi_match_hint(content, "name = \"test\"");
        assert!(hint.contains("Match sites (2 of 2 shown)"));
        assert!(hint.contains("match 1 (lines 1-1)"));
        assert!(hint.contains("match 2 (lines 5-5)"));
        assert!(hint.contains("[[section]]"));
    }

    #[test]
    fn multi_match_hint_caps_at_three() {
        let content = "x\nx\nx\nx\nx\n";
        let hint = multi_match_hint(content, "x");
        assert!(hint.contains("3 of 5 shown"));
        assert!(hint.contains("2 further matches omitted"));
    }

    #[test]
    fn compare_versions_orders_numeric_parts() {
        use std::cmp::Ordering;
        assert_eq!(compare_versions("1.48.0", "1.49.0"), Ordering::Less);
        assert_eq!(compare_versions("1.49.0", "1.48.0"), Ordering::Greater);
        assert_eq!(compare_versions("1.48.0", "1.48.0"), Ordering::Equal);
        // 1.10.0 > 1.9.0 (numeric, not lexicographic)
        assert_eq!(compare_versions("1.10.0", "1.9.0"), Ordering::Greater);
        // Short version padded with zeros
        assert_eq!(compare_versions("2.0", "1.999.999"), Ordering::Greater);
    }

    #[test]
    fn compare_versions_prerelease_semver_ordering() {
        use std::cmp::Ordering;
        // Empty suffix beats non-empty: "1.0.0" > "1.0.0-rc1"
        assert_eq!(compare_versions("1.0.0", "1.0.0-rc1"), Ordering::Greater);
        assert_eq!(compare_versions("1.0.0-rc1", "1.0.0"), Ordering::Less);
        assert_eq!(
            compare_versions("1.0.0-alpha", "1.0.0-beta"),
            Ordering::Less
        );
    }

    #[test]
    fn parse_version_handles_plus_build_metadata() {
        let (nums, suffix) = parse_version("1.2.3+build.5");
        assert_eq!(nums, vec![1, 2, 3]);
        assert_eq!(suffix, "+build.5");
    }

    #[test]
    fn scan_registry_filters_by_name_prefix_and_digit() {
        // Build a fake registry-src tree and verify we pick only `<name>-<digit>*` dirs.
        let tmp =
            std::env::temp_dir().join(format!("wamh-crate-source-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let reg = tmp.join("reg-abc");
        std::fs::create_dir_all(reg.join("tokio-1.48.0/src")).unwrap();
        std::fs::create_dir_all(reg.join("tokio-1.49.0/src")).unwrap();
        // Same-prefix trap: `tokio_util` should not match a query for `tokio`.
        std::fs::create_dir_all(reg.join("tokio_util-0.7.0")).unwrap();
        // Non-version-shaped suffix: should be skipped.
        std::fs::create_dir_all(reg.join("tokio-macros")).unwrap();

        let hits = scan_registry_for_crate(&tmp, "tokio").unwrap();
        let mut versions: Vec<String> = hits.iter().map(|(v, _)| v.clone()).collect();
        versions.sort();
        assert_eq!(versions, vec!["1.48.0".to_string(), "1.49.0".to_string()]);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn scan_registry_returns_empty_when_dir_missing() {
        let nope = std::env::temp_dir().join("wamh-does-not-exist-xyz");
        let _ = std::fs::remove_dir_all(&nope);
        let hits = scan_registry_for_crate(&nope, "tokio").unwrap();
        assert!(hits.is_empty());
    }

    // ---------- read_capped_line ----------

    async fn read_capped_line_from_slice(input: &[u8], cap: usize) -> Vec<(ReadOutcome, Vec<u8>)> {
        let mut reader = BufReader::new(input);
        let mut out = Vec::new();
        loop {
            let mut buf = Vec::new();
            let outcome = read_capped_line(&mut reader, &mut buf, cap).await.unwrap();
            if outcome == ReadOutcome::Eof {
                break;
            }
            out.push((outcome, buf));
        }
        out
    }

    #[tokio::test]
    async fn read_capped_line_splits_on_newlines() {
        let lines = read_capped_line_from_slice(b"alpha\nbeta\ngamma\n", 1024).await;
        assert_eq!(
            lines,
            vec![
                (ReadOutcome::Line, b"alpha\n".to_vec()),
                (ReadOutcome::Line, b"beta\n".to_vec()),
                (ReadOutcome::Line, b"gamma\n".to_vec()),
            ]
        );
    }

    #[tokio::test]
    async fn read_capped_line_handles_trailing_no_newline() {
        let lines = read_capped_line_from_slice(b"alpha\nbeta", 1024).await;
        assert_eq!(
            lines,
            vec![
                (ReadOutcome::Line, b"alpha\n".to_vec()),
                (ReadOutcome::Line, b"beta".to_vec()),
            ]
        );
    }

    #[tokio::test]
    async fn read_capped_line_truncates_single_overlong_line() {
        // A 10_000-byte line with no `\n`, followed by a normal line —
        // mimics a child that emits a huge JSON blob on one line then
        // continues. The capped reader returns the first `cap` bytes
        // marked Truncated, drains the rest of the overlong line, then
        // surfaces the next line cleanly.
        let mut input = Vec::from(vec![b'x'; 10_000].as_slice());
        input.extend_from_slice(b"\nnext-line\n");
        let lines = read_capped_line_from_slice(&input, 128).await;
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].0, ReadOutcome::Truncated);
        assert_eq!(lines[0].1.len(), 128);
        assert_eq!(lines[0].1, vec![b'x'; 128]);
        assert_eq!(lines[1], (ReadOutcome::Line, b"next-line\n".to_vec()));
    }

    #[tokio::test]
    async fn read_capped_line_reports_truncated_on_cap_hit_at_eof() {
        // Overlong line with no trailing newline and no following data —
        // still reports Truncated (not Eof or Line) so callers can
        // append a marker.
        let input = vec![b'a'; 1024];
        let lines = read_capped_line_from_slice(&input, 64).await;
        assert_eq!(lines, vec![(ReadOutcome::Truncated, vec![b'a'; 64])]);
    }

    // ---------- bash_stream end-to-end (caps) ----------

    fn test_workspace() -> Arc<Workspace> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut root = std::env::temp_dir();
        root.push(format!("wamh-bash-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        Arc::new(Workspace::new(&root).unwrap())
    }

    async fn run_bash_to_final(ws: Arc<Workspace>, args: Value) -> (Vec<String>, CallToolResult) {
        use futures::StreamExt;
        let mut stream = bash_stream(ws, args);
        let mut chunks: Vec<String> = Vec::new();
        let mut final_result: Option<CallToolResult> = None;
        while let Some(item) = stream.next().await {
            match item {
                ToolStreamItem::Chunk(ContentBlock::Text { text }) => chunks.push(text),
                // bash_stream only emits text chunks; image / resource
                // chunks are tool-result shapes used by view_image /
                // view_pdf, not bash.
                ToolStreamItem::Chunk(ContentBlock::Image { .. }) => {
                    panic!("bash_stream emitted unexpected image chunk")
                }
                ToolStreamItem::Chunk(ContentBlock::Resource { .. }) => {
                    panic!("bash_stream emitted unexpected resource chunk")
                }
                ToolStreamItem::Final(r) => {
                    final_result = Some(r);
                    break;
                }
            }
        }
        (chunks, final_result.expect("bash_stream produced no Final"))
    }

    #[tokio::test]
    async fn bash_streaming_chunks_cap_at_total_budget() {
        // Emit ~4 MB of output (40k lines × ~100 bytes) — well past the
        // streaming cap. The subscriber sees a bounded chunk volume
        // plus one cap-reached marker, and the final tool_result shows
        // the tail.
        let ws = test_workspace();
        let (chunks, final_result) = run_bash_to_final(
            ws,
            json!({
                "command": "for i in $(seq 1 40000); do printf 'line %06d %s\\n' $i xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx; done",
                "timeout_seconds": 60,
            }),
        )
        .await;
        let streamed_bytes: usize = chunks.iter().map(|c| c.len()).sum();
        assert!(
            streamed_bytes <= BASH_STREAM_TOTAL_BYTES + 512,
            "streamed {streamed_bytes} exceeds cap"
        );
        let saw_cap_marker = chunks.iter().any(|c| c.contains("streaming cap reached"));
        assert!(saw_cap_marker, "expected streaming-cap marker chunk");
        let body = final_tool_text(&final_result);
        assert!(
            body.len() <= BASH_MAX_OUTPUT_BYTES + 256,
            "final body {}B exceeds 30KB cap",
            body.len()
        );
        // Final tail should include the latest lines — head-truncate keeps tail.
        assert!(
            body.contains("line 040000"),
            "expected final tail to contain last line; body ends with: {}",
            &body[body.len().saturating_sub(200)..]
        );
    }

    #[tokio::test]
    async fn bash_truncates_overlong_single_line() {
        // A single 200_000-byte line would OOM `read_line`. With our
        // capped reader, the tool survives and the output shows a
        // truncation marker.
        let ws = test_workspace();
        let (chunks, final_result) = run_bash_to_final(
            ws,
            json!({
                "command": "printf 'x%.0s' $(seq 1 200000); echo",
                "timeout_seconds": 30,
            }),
        )
        .await;
        let streamed: String = chunks.concat();
        assert!(
            streamed.contains("line truncated"),
            "expected line-truncation marker; got {} bytes",
            streamed.len()
        );
        let body = final_tool_text(&final_result);
        assert!(body.len() <= BASH_MAX_OUTPUT_BYTES + 256);
    }

    /// Collapse a `CallToolResult`'s `Text` blocks into a single `&str` for
    /// test assertions. Every bash_stream Final has exactly one Text block;
    /// image and resource blocks are produced by view_image / view_pdf,
    /// not by bash.
    fn final_tool_text(r: &CallToolResult) -> &str {
        match r.content.first() {
            Some(ContentBlock::Text { text }) => text.as_str(),
            Some(ContentBlock::Image { .. }) => panic!("expected text result, got image"),
            Some(ContentBlock::Resource { .. }) => panic!("expected text result, got resource"),
            None => "",
        }
    }

    // ---------- read_file binary refusal ----------

    /// PNG magic bytes followed by a NUL — sufficient for both the
    /// binary sniff (NUL in first 8 KiB) and `image::guess_format`
    /// (PNG header).
    fn fake_png_bytes() -> Vec<u8> {
        let mut v = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        v.extend_from_slice(&[0u8; 64]);
        v
    }

    #[tokio::test]
    async fn read_file_refuses_binary_image_with_view_image_hint() {
        let ws = test_workspace();
        let path = ws.root().join("dummy.png");
        std::fs::write(&path, fake_png_bytes()).unwrap();
        let result = read_file(&ws, json!({ "path": "dummy.png" })).await;
        assert_eq!(result.is_error, Some(true));
        let body = final_tool_text(&result);
        assert!(
            body.contains("view_image"),
            "expected steering to view_image; got {body:?}"
        );
        assert!(
            body.contains("Png") || body.contains("PNG") || body.contains("image"),
            "expected format hint; got {body:?}"
        );
    }

    #[tokio::test]
    async fn read_file_refuses_non_image_binary_generically() {
        // Random bytes containing a NUL but no recognizable image
        // signature.
        let ws = test_workspace();
        let path = ws.root().join("blob.bin");
        let mut bytes = vec![1u8, 2, 3, 0, 4, 5];
        bytes.extend_from_slice(&[0u8; 64]);
        std::fs::write(&path, bytes).unwrap();
        let result = read_file(&ws, json!({ "path": "blob.bin" })).await;
        assert_eq!(result.is_error, Some(true));
        let body = final_tool_text(&result);
        assert!(
            body.contains("binary"),
            "expected generic binary message; got {body:?}"
        );
        assert!(
            !body.contains("view_image"),
            "non-image binary shouldn't suggest view_image; got {body:?}"
        );
    }

    #[tokio::test]
    async fn read_file_still_reads_text_with_high_bit_chars() {
        // Non-ASCII UTF-8 (smart quotes, emoji) has no NUL byte, so the
        // sniff must let it through.
        let ws = test_workspace();
        let path = ws.root().join("utf8.txt");
        std::fs::write(&path, "héllo “world” 🎉").unwrap();
        let result = read_file(&ws, json!({ "path": "utf8.txt" })).await;
        assert!(result.is_error.is_none());
        let body = final_tool_text(&result);
        assert!(body.contains("héllo"));
    }

    #[tokio::test]
    async fn read_file_refuses_pdf_with_view_pdf_hint() {
        // PDFs don't usually have NUL in the first 8 KiB — the sniff
        // recognizes the `%PDF-` magic explicitly.
        let ws = test_workspace();
        let path = ws.root().join("doc.pdf");
        let mut bytes = b"%PDF-1.4\n".to_vec();
        bytes.extend_from_slice(b"%fake-pdf-tail-no-nul-bytes\n");
        std::fs::write(&path, bytes).unwrap();
        let result = read_file(&ws, json!({ "path": "doc.pdf" })).await;
        assert_eq!(result.is_error, Some(true));
        let body = final_tool_text(&result);
        assert!(
            body.contains("view_pdf"),
            "expected view_pdf hint; got {body:?}"
        );
        assert!(!body.contains("view_image"));
    }

    // ---------- view_image: prepare_image ----------

    /// Build a real PNG of the requested dimensions. Uses a single-color
    /// fill so the encoded bytes are deterministic enough for size
    /// assertions across runs.
    fn make_png(w: u32, h: u32) -> Vec<u8> {
        use image::{ImageBuffer, ImageFormat, Rgb};
        let img: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_fn(w, h, |_, _| Rgb([10, 20, 30]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn prepare_image_passes_through_small_png() {
        let bytes = make_png(64, 64);
        let prepared = prepare_image(&bytes).unwrap();
        assert_eq!(prepared.mime_type, "image/png");
        assert_eq!(
            prepared.bytes, bytes,
            "small images should not be re-encoded"
        );
    }

    #[test]
    fn prepare_image_resizes_oversize_image_to_png() {
        let bytes = make_png(VIEW_IMAGE_MAX_DIMENSION + 200, 32);
        let prepared = prepare_image(&bytes).unwrap();
        assert_eq!(prepared.mime_type, "image/png");
        // Decode the result and confirm the long axis was clamped.
        let decoded = image::load_from_memory(&prepared.bytes).unwrap();
        assert!(
            decoded.width().max(decoded.height()) <= VIEW_IMAGE_MAX_DIMENSION,
            "max dim {} > cap {}",
            decoded.width().max(decoded.height()),
            VIEW_IMAGE_MAX_DIMENSION
        );
    }

    #[test]
    fn prepare_image_rejects_non_image_bytes() {
        let bytes = b"this is just some text, not an image".to_vec();
        let err = prepare_image(&bytes).unwrap_err();
        assert!(err.contains("could not detect"), "unexpected error: {err}");
    }

    // ---------- view_pdf ----------

    /// Smallest possible byte sequence that starts with a valid PDF
    /// header. Real PDFs have an xref table and a `%%EOF` trailer; the
    /// view_pdf tool only sniffs the header, so this is enough to
    /// exercise the tool's golden path without pulling in a PDF
    /// generator.
    fn fake_pdf_bytes() -> Vec<u8> {
        let mut v = b"%PDF-1.4\n".to_vec();
        v.extend_from_slice(b"%minimal-stub-payload\n");
        v.extend_from_slice(b"%%EOF\n");
        v
    }

    /// Pull the single Resource block out of a Final result for
    /// view_pdf assertions. Panics on any other shape.
    fn final_resource(r: &CallToolResult) -> &whisper_agent_mcp_proto::EmbeddedResource {
        match r.content.first() {
            Some(ContentBlock::Resource { resource }) => resource,
            other => panic!("expected resource result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn view_pdf_returns_resource_blob_with_pdf_mime() {
        let ws = test_workspace();
        let bytes = fake_pdf_bytes();
        let path = ws.root().join("doc.pdf");
        std::fs::write(&path, &bytes).unwrap();
        let result = view_pdf(&ws, json!({ "path": "doc.pdf" })).await;
        assert!(result.is_error.is_none(), "unexpected error result");
        let resource = final_resource(&result);
        assert_eq!(resource.mime_type.as_deref(), Some("application/pdf"));
        assert!(resource.uri.starts_with("file://"));
        assert!(resource.text.is_none());
        let blob = resource.blob.as_deref().expect("expected blob");
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(blob)
            .unwrap();
        assert_eq!(decoded, bytes, "round-trip should preserve bytes");
    }

    #[tokio::test]
    async fn view_pdf_rejects_non_pdf_bytes() {
        let ws = test_workspace();
        let path = ws.root().join("not-a-pdf.bin");
        std::fs::write(&path, b"GIF89a... not a pdf").unwrap();
        let result = view_pdf(&ws, json!({ "path": "not-a-pdf.bin" })).await;
        assert_eq!(result.is_error, Some(true));
        let body = match result.content.first() {
            Some(ContentBlock::Text { text }) => text.as_str(),
            other => panic!("expected text error result, got {other:?}"),
        };
        assert!(body.contains("not look like a PDF"));
    }

    #[tokio::test]
    async fn view_pdf_rejects_directory_target() {
        // Pre-flight rejection: target must be a regular file. Pointing
        // at a directory is the cheap reproduction; the byte-cap branch
        // is symmetric (separate metadata check) and not worth a 25 MB
        // tempfile to exercise.
        let ws = test_workspace();
        let path = ws.root().join("not-a-file");
        std::fs::create_dir(&path).unwrap();
        let result = view_pdf(&ws, json!({ "path": "not-a-file" })).await;
        assert_eq!(result.is_error, Some(true));
        let body = match result.content.first() {
            Some(ContentBlock::Text { text }) => text.as_str(),
            other => panic!("expected text error result, got {other:?}"),
        };
        assert!(body.contains("not a regular file"));
    }
}
