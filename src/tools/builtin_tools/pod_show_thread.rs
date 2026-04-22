//! `pod_show_thread` — render a thread's conversation as a compact
//! text transcript.
//!
//! The raw thread JSON at `threads/<id>.json` is a serialized
//! state-machine checkpoint: system prompt, tool schemas, turn log,
//! usage counters, and the conversation's content blocks all inlined.
//! A file with ~40 meaningful messages is routinely 1500+ lines on
//! disk. An agent reading it via `pod_read_file` pays a heavy token
//! tax to find what the thread actually did.
//!
//! This tool does the extraction server-side: skip Role::System and
//! Role::Tools (pure boilerplate), skip Thinking blocks by default,
//! render tool_use inline with its arguments preview, pair tool_use
//! ids to tool_result blocks so the downstream result shows which
//! tool it came from, and truncate oversize tool outputs to a
//! configurable cap.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use serde_json::{Value, json};

use super::{POD_SHOW_THREAD, ToolOutcome, no_update_error, no_update_text};
use crate::pod;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

/// Per-tool-result preview cap when the caller doesn't override.
/// Chosen to fit a couple of screens of output while leaving budget
/// for a conversation with a dozen tool calls to all fit in a single
/// response.
const DEFAULT_MAX_TOOL_CHARS: usize = 500;

/// Per-Thinking-block preview cap when `include_thinking=true`. The
/// raw blocks are often >2000 tokens of reasoning; even when the
/// caller opts in they usually want a gist, not the whole thing.
const DEFAULT_MAX_THINKING_CHARS: usize = 800;

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: POD_SHOW_THREAD.into(),
        description: "Render a thread's conversation as a compact text transcript. \
                      Much cheaper than `pod_read_file(\"threads/<id>.json\")` for \
                      inspection: the raw file carries tool schemas, state-machine \
                      bookkeeping, and thinking-blocks that together dwarf the \
                      actual conversation. This tool strips that boilerplate and \
                      pairs each `tool_use` with its `tool_result` for inline \
                      reading. Use `pod_list_threads` first to find the `thread_id`. \
                      Defaults are tuned for triage (`include_thinking=false`, \
                      `max_tool_chars=500`); pass explicit values to expand."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "Thread id as surfaced by `pod_list_threads` \
                                    (e.g. `task-18a885f3a2aa599d`). Resolved to \
                                    `threads/<thread_id>.json` under this pod's \
                                    directory."
                },
                "tail_messages": {
                    "type": "integer",
                    "description": "Render only the last N visible messages \
                                    (after role filtering). Default: all."
                },
                "include_system": {
                    "type": "boolean",
                    "description": "Include `Role::System` and `Role::Tools` \
                                    messages. Default: false — both are \
                                    repeated across every thread in the pod \
                                    and usually noise."
                },
                "include_thinking": {
                    "type": "boolean",
                    "description": "Include `Thinking` content blocks, capped \
                                    at 800 chars each. Default: false."
                },
                "max_tool_chars": {
                    "type": "integer",
                    "description": "Max chars of each tool_result preview. \
                                    Default 500. Set to 0 to suppress \
                                    tool_result bodies entirely (only the \
                                    length + ok/ERROR tag shown)."
                }
            },
            "required": ["thread_id"]
        }),
        annotations: ToolAnnotations {
            title: Some("Show thread transcript".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize)]
struct ShowThreadArgs {
    thread_id: String,
    #[serde(default)]
    tail_messages: Option<usize>,
    #[serde(default)]
    include_system: Option<bool>,
    #[serde(default)]
    include_thinking: Option<bool>,
    #[serde(default)]
    max_tool_chars: Option<usize>,
}

pub(super) async fn run(pod_dir: &Path, args: Value) -> ToolOutcome {
    let parsed: ShowThreadArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    // Guard against path traversal: the thread_id becomes a filename
    // segment, so reject anything containing `/`, `\`, or `..`.
    if parsed.thread_id.contains(['/', '\\'])
        || parsed.thread_id.contains("..")
        || parsed.thread_id.is_empty()
    {
        return no_update_error(format!(
            "invalid thread_id `{}` — expected a bare id like `task-...`",
            parsed.thread_id
        ));
    }
    let path = pod_dir
        .join(pod::THREADS_DIR)
        .join(format!("{}.json", parsed.thread_id));
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(_) => {
            return no_update_error(format!(
                "thread `{}` not found (looked for `{}/{}.json`)",
                parsed.thread_id,
                pod::THREADS_DIR,
                parsed.thread_id
            ));
        }
    };
    let v: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return no_update_error(format!(
                "failed to parse thread `{}`: {e}",
                parsed.thread_id
            ));
        }
    };
    no_update_text(render(&v, &parsed))
}

fn render(v: &Value, args: &ShowThreadArgs) -> String {
    let include_system = args.include_system.unwrap_or(false);
    let include_thinking = args.include_thinking.unwrap_or(false);
    let max_tool_chars = args.max_tool_chars.unwrap_or(DEFAULT_MAX_TOOL_CHARS);

    let mut out = String::new();
    render_header(&mut out, v);

    let empty: Vec<Value> = Vec::new();
    let messages: &[Value] = v
        .get("conversation")
        .and_then(|c| c.get("messages"))
        .and_then(|m| m.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(empty.as_slice());

    let tool_names = build_tool_name_map(messages);

    // Count role occurrences up front so the header can report what
    // was hidden without needing to walk the messages twice.
    let mut hidden = RoleHidden::default();
    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("unknown");
        match role {
            "system" if !include_system => hidden.system += 1,
            "tools" if !include_system => hidden.tools += 1,
            _ => {}
        }
    }
    write_hidden_notice(&mut out, &hidden, messages.len());

    // Select visible messages, preserving the original indices so the
    // transcript carries `[i=N]` markers that match the raw JSON.
    let visible: Vec<(usize, &Value)> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
            match role {
                "system" | "tools" => include_system,
                _ => true,
            }
        })
        .collect();
    let visible = apply_tail(visible, args.tail_messages);

    if visible.is_empty() {
        out.push_str("(no messages to show after filtering)\n");
        return out;
    }
    for (idx, msg) in visible {
        render_message(
            &mut out,
            idx,
            msg,
            &tool_names,
            include_thinking,
            max_tool_chars,
        );
    }
    out
}

fn render_header(out: &mut String, v: &Value) {
    let id = v.get("id").and_then(|s| s.as_str()).unwrap_or("<unknown>");
    let title = v
        .get("title")
        .and_then(|s| s.as_str())
        .filter(|s| !s.is_empty());
    let internal_kind = v
        .get("internal")
        .and_then(|i| i.get("kind"))
        .and_then(|k| k.as_str())
        .unwrap_or("unknown");
    let state = match internal_kind {
        "idle" => "idle",
        "completed" => "completed",
        "waiting_on_resources" | "needs_model_call" | "awaiting_model" | "awaiting_tools" => {
            "working"
        }
        "failed" => "failed",
        "cancelled" => "cancelled",
        _ => internal_kind,
    };
    let msg_count = v
        .get("conversation")
        .and_then(|c| c.get("messages"))
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let created = v.get("created_at").and_then(|s| s.as_str()).unwrap_or("-");
    let last_active = v.get("last_active").and_then(|s| s.as_str()).unwrap_or("-");
    let backend = v
        .get("bindings")
        .and_then(|b| b.get("backend"))
        .and_then(|s| s.as_str())
        .unwrap_or("?");
    let model = v
        .get("config")
        .and_then(|c| c.get("model"))
        .and_then(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("?");
    let input_tokens = v
        .get("total_usage")
        .and_then(|u| u.get("input_tokens"))
        .and_then(|n| n.as_u64())
        .unwrap_or(0);
    let output_tokens = v
        .get("total_usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(|n| n.as_u64())
        .unwrap_or(0);

    out.push_str(&format!("thread {id} (state={state}, msgs={msg_count})\n"));
    if let Some(t) = title {
        out.push_str(&format!("title: {t}\n"));
    }
    out.push_str(&format!("created: {created}  last_active: {last_active}\n"));
    out.push_str(&format!(
        "backend: {backend}  model: {model}  usage: {input_tokens} in / {output_tokens} out\n"
    ));
    // Origin / dispatch lineage only when non-interactive — suppresses
    // noise for the common bare-thread case.
    if let Some(origin) = v.get("origin").and_then(|o| {
        if o.is_null() {
            None
        } else {
            o.get("behavior_id").and_then(|b| b.as_str())
        }
    }) {
        out.push_str(&format!("origin: behavior={origin}\n"));
    }
    if let Some(parent) = v
        .get("dispatched_by")
        .and_then(|s| s.as_str())
        .filter(|s| !s.is_empty())
    {
        out.push_str(&format!("dispatched_by: {parent}\n"));
    }
    if let Some(continued) = v
        .get("continued_from")
        .and_then(|s| s.as_str())
        .filter(|s| !s.is_empty())
    {
        out.push_str(&format!("continued_from: {continued}\n"));
    }
    out.push('\n');
}

#[derive(Default)]
struct RoleHidden {
    system: usize,
    tools: usize,
}

fn write_hidden_notice(out: &mut String, h: &RoleHidden, total: usize) {
    if h.system == 0 && h.tools == 0 {
        return;
    }
    let mut parts: Vec<String> = Vec::new();
    if h.system > 0 {
        parts.push(format!("{} system", h.system));
    }
    if h.tools > 0 {
        parts.push(format!("{} tool-schema", h.tools));
    }
    out.push_str(&format!(
        "(hidden: {} of {total} messages; pass include_system=true to show)\n\n",
        parts.join(", ")
    ));
}

fn apply_tail(msgs: Vec<(usize, &Value)>, tail: Option<usize>) -> Vec<(usize, &Value)> {
    match tail {
        Some(n) if msgs.len() > n => msgs[msgs.len() - n..].to_vec(),
        _ => msgs,
    }
}

/// Walk the whole conversation once collecting `tool_use_id -> name`
/// so later `tool_result` blocks can say WHICH tool returned the
/// result without a back-reference search per-render.
fn build_tool_name_map(messages: &[Value]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for m in messages {
        let Some(content) = m.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for block in content {
            let Some(ty) = block.get("type").and_then(|t| t.as_str()) else {
                continue;
            };
            if ty != "tool_use" {
                continue;
            }
            let id = block
                .get("id")
                .and_then(|s| s.as_str())
                .unwrap_or_default()
                .to_string();
            let name = block
                .get("name")
                .and_then(|s| s.as_str())
                .unwrap_or("<unknown>")
                .to_string();
            if !id.is_empty() {
                map.insert(id, name);
            }
        }
    }
    map
}

fn render_message(
    out: &mut String,
    idx: usize,
    msg: &Value,
    tool_names: &HashMap<String, String>,
    include_thinking: bool,
    max_tool_chars: usize,
) {
    let role = msg
        .get("role")
        .and_then(|r| r.as_str())
        .unwrap_or("unknown");
    let empty: Vec<Value> = Vec::new();
    let content: &[Value] = msg
        .get("content")
        .and_then(|c| c.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(empty.as_slice());

    match role {
        "tools" => {
            // include_system=true path — still compact; tool schemas
            // are too verbose to print even when the caller asked.
            out.push_str(&format!(
                "[i={idx} tools]  ({} tool schemas)\n\n",
                content.len()
            ));
            return;
        }
        _ => {
            out.push_str(&format!("[i={idx} {role}]\n"));
        }
    }

    for block in content {
        let ty = block
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("unknown");
        match ty {
            "text" => {
                if let Some(t) = block.get("text").and_then(|s| s.as_str()) {
                    out.push_str(t);
                    if !t.ends_with('\n') {
                        out.push('\n');
                    }
                }
            }
            "thinking" => {
                if !include_thinking {
                    continue;
                }
                let thinking = block.get("thinking").and_then(|s| s.as_str()).unwrap_or("");
                let (preview, over) = truncate(thinking, DEFAULT_MAX_THINKING_CHARS);
                out.push_str("<thinking>\n");
                out.push_str(&preview);
                if !preview.ends_with('\n') {
                    out.push('\n');
                }
                if over > 0 {
                    out.push_str(&format!("[+{over} chars truncated]\n"));
                }
                out.push_str("</thinking>\n");
            }
            "tool_use" => {
                let name = block
                    .get("name")
                    .and_then(|s| s.as_str())
                    .unwrap_or("<unknown>");
                let id = block.get("id").and_then(|s| s.as_str()).unwrap_or("");
                let args_one_line = compact_args_preview(block.get("input"));
                out.push_str(&format!("→ {name}({args_one_line})"));
                if !id.is_empty() {
                    out.push_str(&format!("  #{id}"));
                }
                out.push('\n');
            }
            "tool_result" => {
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                let is_error = block
                    .get("is_error")
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);
                let body = tool_result_text(block.get("content"));
                let tool_name = tool_names
                    .get(tool_use_id)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                let status = if is_error { "ERROR" } else { "ok" };
                let body_len = body.chars().count();
                out.push_str(&format!("← {tool_name} [{status}, {body_len} chars]"));
                if !tool_use_id.is_empty() {
                    out.push_str(&format!("  #{tool_use_id}"));
                }
                out.push('\n');
                if max_tool_chars > 0 && !body.is_empty() {
                    let (preview, over) = truncate(&body, max_tool_chars);
                    out.push_str(&preview);
                    if !preview.ends_with('\n') {
                        out.push('\n');
                    }
                    if over > 0 {
                        out.push_str(&format!(
                            "[+{over} chars truncated; pass max_tool_chars to expand]\n"
                        ));
                    }
                }
            }
            // tool_schema only appears in Role::Tools messages, which
            // we've already short-circuited above. Unknown types: skip
            // silently — the transcript is best-effort.
            _ => {}
        }
    }
    out.push('\n');
}

/// Single-line JSON-ish rendering of a tool_use's `input`, capped so
/// a huge bash script or file contents don't dominate the transcript.
/// Strings retain their human readability (not re-escaped as JSON);
/// numbers/bools/nulls render as JSON literals.
fn compact_args_preview(input: Option<&Value>) -> String {
    const MAX: usize = 200;
    let Some(v) = input else {
        return String::new();
    };
    let s = match v {
        Value::Object(m) if m.is_empty() => String::new(),
        _ => serde_json::to_string(v).unwrap_or_default(),
    };
    let (truncated, over) = truncate(&s, MAX);
    if over > 0 {
        format!("{truncated}…[+{over} chars]")
    } else {
        truncated
    }
}

/// Extract a displayable string from a `tool_result`'s `content`
/// field. The field is untagged: either a bare string (`Text`
/// variant) or an array of content blocks (`Blocks` variant). We
/// concatenate all nested text blocks when the latter.
fn tool_result_text(content: Option<&Value>) -> String {
    let Some(c) = content else {
        return String::new();
    };
    if let Some(s) = c.as_str() {
        return s.to_string();
    }
    if let Some(arr) = c.as_array() {
        let mut out = String::new();
        for block in arr {
            if block.get("type").and_then(|t| t.as_str()) == Some("text")
                && let Some(t) = block.get("text").and_then(|s| s.as_str())
            {
                out.push_str(t);
            }
        }
        return out;
    }
    String::new()
}

/// Character-bounded truncation. Returns `(preview, over)` where
/// `over` is the number of chars dropped. Works on codepoints so it
/// never cuts in the middle of a multi-byte UTF-8 sequence.
fn truncate(s: &str, max_chars: usize) -> (String, usize) {
    let total = s.chars().count();
    if total <= max_chars {
        return (s.to_string(), 0);
    }
    let preview: String = s.chars().take(max_chars).collect();
    (preview, total - max_chars)
}

#[cfg(test)]
mod tests {
    use super::super::{dispatch, join_blocks, sample_config, temp_dir};
    use super::*;

    fn seed_thread_dir(dir: &Path) {
        std::fs::create_dir_all(dir.join(pod::THREADS_DIR)).unwrap();
    }

    /// Write a minimal thread JSON with the caller-supplied
    /// conversation body. Only the fields the renderer reads are
    /// populated; unknown fields are ignored by serde anyway.
    fn write_thread(dir: &Path, id: &str, messages: Value, extra: Value) {
        let mut body = json!({
            "id": id,
            "pod_id": "p",
            "created_at": "2026-04-22T10:00:00Z",
            "last_active": "2026-04-22T10:30:00Z",
            "title": "test thread",
            "config": { "model": "claude-sonnet-4-6" },
            "bindings": { "backend": "anthropic" },
            "conversation": { "messages": messages },
            "total_usage": { "input_tokens": 1234, "output_tokens": 567 },
            "internal": { "kind": "completed" }
        });
        for (k, v) in extra.as_object().cloned().unwrap_or_default() {
            body[k] = v;
        }
        std::fs::write(
            dir.join(pod::THREADS_DIR).join(format!("{id}.json")),
            serde_json::to_vec_pretty(&body).unwrap(),
        )
        .unwrap();
    }

    async fn show(dir: &Path, args: Value) -> String {
        let out = dispatch(
            dir.to_path_buf(),
            sample_config(),
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::AuthorAny,
            POD_SHOW_THREAD,
            args,
        )
        .await;
        assert!(!out.result.is_error, "expected ok: {:?}", out.result);
        join_blocks(&out.result.content)
    }

    #[tokio::test]
    async fn renders_conversation_skipping_system_and_tools() {
        let dir = temp_dir();
        seed_thread_dir(&dir);
        let messages = json!([
            { "role": "system", "content": [{"type": "text", "text": "you are helpful"}] },
            { "role": "tools", "content": [{"type": "tool_schema", "name": "bash", "description": "run", "input_schema": {}}] },
            { "role": "user", "content": [{"type": "text", "text": "list /tmp"}] },
            { "role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should call bash"},
                {"type": "text", "text": "I'll run ls."},
                {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"command": "ls /tmp"}}
            ]},
            { "role": "tool_result", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "file_a\nfile_b\n", "is_error": false}
            ]}
        ]);
        write_thread(&dir, "task-abc", messages, json!({}));

        let text = show(&dir, json!({ "thread_id": "task-abc" })).await;
        // Header present.
        assert!(text.contains("thread task-abc"), "header: {text}");
        assert!(text.contains("state=completed"), "state: {text}");
        assert!(text.contains("usage: 1234 in / 567 out"), "usage: {text}");
        // Hidden-notice announces the two boilerplate messages we skipped.
        assert!(
            text.contains("hidden: 1 system, 1 tool-schema"),
            "hidden notice: {text}"
        );
        // User and assistant text survived.
        assert!(text.contains("list /tmp"));
        assert!(text.contains("I'll run ls."));
        // Thinking suppressed by default.
        assert!(
            !text.contains("I should call bash"),
            "thinking leaked: {text}"
        );
        // Tool call inline.
        assert!(text.contains("→ bash"), "tool_use inline: {text}");
        assert!(text.contains("ls /tmp"), "args preview: {text}");
        // Tool result paired with name + ok status.
        assert!(text.contains("← bash [ok,"), "tool_result header: {text}");
        assert!(text.contains("file_a"), "tool_result body: {text}");
    }

    #[tokio::test]
    async fn include_thinking_shows_blocks_truncated() {
        let dir = temp_dir();
        seed_thread_dir(&dir);
        let long_thinking = "X".repeat(2000);
        let messages = json!([
            { "role": "user", "content": [{"type": "text", "text": "hi"}] },
            { "role": "assistant", "content": [
                {"type": "thinking", "thinking": long_thinking},
                {"type": "text", "text": "reply"}
            ]}
        ]);
        write_thread(&dir, "task-think", messages, json!({}));

        let text = show(
            &dir,
            json!({ "thread_id": "task-think", "include_thinking": true }),
        )
        .await;
        assert!(text.contains("<thinking>"));
        // Capped at DEFAULT_MAX_THINKING_CHARS (800); 2000 input → 1200 over.
        assert!(
            text.contains("[+1200 chars truncated]"),
            "truncation marker: {text}"
        );
    }

    #[tokio::test]
    async fn tool_result_truncates_and_reports_length() {
        let dir = temp_dir();
        seed_thread_dir(&dir);
        let big = "A".repeat(2000);
        let messages = json!([
            { "role": "user", "content": [{"type": "text", "text": "run it"}] },
            { "role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_big", "name": "bash", "input": {"command": "echo"}}
            ]},
            { "role": "tool_result", "content": [
                {"type": "tool_result", "tool_use_id": "tu_big", "content": big, "is_error": false}
            ]}
        ]);
        write_thread(&dir, "task-big", messages, json!({}));

        // Default 500-char cap.
        let text = show(&dir, json!({ "thread_id": "task-big" })).await;
        assert!(text.contains("← bash [ok, 2000 chars]"));
        assert!(text.contains("[+1500 chars truncated"));

        // max_tool_chars=0 suppresses the body entirely.
        let text = show(
            &dir,
            json!({ "thread_id": "task-big", "max_tool_chars": 0 }),
        )
        .await;
        assert!(text.contains("← bash [ok, 2000 chars]"));
        assert!(!text.contains("AAAAA"), "body should be suppressed: {text}");
    }

    #[tokio::test]
    async fn is_error_tool_results_flagged() {
        let dir = temp_dir();
        seed_thread_dir(&dir);
        let messages = json!([
            { "role": "user", "content": [{"type": "text", "text": "fail"}] },
            { "role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_err", "name": "bash", "input": {}}
            ]},
            { "role": "tool_result", "content": [
                {"type": "tool_result", "tool_use_id": "tu_err", "content": "permission denied", "is_error": true}
            ]}
        ]);
        write_thread(&dir, "task-err", messages, json!({}));

        let text = show(&dir, json!({ "thread_id": "task-err" })).await;
        assert!(text.contains("← bash [ERROR,"), "error tag: {text}");
        assert!(text.contains("permission denied"));
    }

    #[tokio::test]
    async fn tail_messages_trims_to_last_n_visible() {
        let dir = temp_dir();
        seed_thread_dir(&dir);
        // Five user/assistant pairs (10 visible messages) plus a
        // hidden system. tail=3 should keep the last three visible.
        let mut messages = vec![json!({
            "role": "system",
            "content": [{"type": "text", "text": "sys"}]
        })];
        for i in 0..5 {
            messages.push(json!({
                "role": "user",
                "content": [{"type": "text", "text": format!("u{i}")}]
            }));
            messages.push(json!({
                "role": "assistant",
                "content": [{"type": "text", "text": format!("a{i}")}]
            }));
        }
        write_thread(&dir, "task-tail", Value::Array(messages), json!({}));

        let text = show(
            &dir,
            json!({ "thread_id": "task-tail", "tail_messages": 3 }),
        )
        .await;
        // Last three visible are a3, u4, a4.
        assert!(text.contains("a3"));
        assert!(text.contains("u4"));
        assert!(text.contains("a4"));
        // Earlier content is gone.
        assert!(!text.contains("u0"));
        assert!(!text.contains("a0"));
        assert!(!text.contains("u3"));
    }

    #[tokio::test]
    async fn missing_thread_reports_error() {
        let dir = temp_dir();
        seed_thread_dir(&dir);
        let out = dispatch(
            dir.clone(),
            sample_config(),
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            crate::permission::BehaviorOpsCap::AuthorAny,
            POD_SHOW_THREAD,
            json!({ "thread_id": "task-nope" }),
        )
        .await;
        assert!(out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("not found"));
    }

    #[tokio::test]
    async fn invalid_thread_id_rejected() {
        let dir = temp_dir();
        seed_thread_dir(&dir);
        for bad in &["../secret", "a/b", "a\\b", "../../etc/passwd", ""] {
            let out = dispatch(
                dir.clone(),
                sample_config(),
                vec![],
                crate::permission::PodModifyCap::ModifyAllow,
                crate::permission::BehaviorOpsCap::AuthorAny,
                POD_SHOW_THREAD,
                json!({ "thread_id": bad }),
            )
            .await;
            assert!(out.result.is_error, "{bad} should be rejected");
            let text = join_blocks(&out.result.content);
            assert!(text.contains("invalid thread_id"), "bad id {bad}: {text}");
        }
    }

    #[tokio::test]
    async fn origin_and_lineage_surface_when_present() {
        let dir = temp_dir();
        seed_thread_dir(&dir);
        let messages = json!([
            { "role": "user", "content": [{"type": "text", "text": "triggered"}] }
        ]);
        write_thread(
            &dir,
            "task-beh",
            messages,
            json!({
                "origin": { "behavior_id": "daily_summary", "fired_at": "2026-04-22T10:00:00Z", "trigger_payload": null },
                "dispatched_by": "task-parent-1",
                "continued_from": "task-ancestor-0"
            }),
        );

        let text = show(&dir, json!({ "thread_id": "task-beh" })).await;
        assert!(text.contains("origin: behavior=daily_summary"));
        assert!(text.contains("dispatched_by: task-parent-1"));
        assert!(text.contains("continued_from: task-ancestor-0"));
    }
}
