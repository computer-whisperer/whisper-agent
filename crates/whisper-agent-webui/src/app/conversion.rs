//! Pure wire→display transforms.
//!
//! Two entry points: `snapshot_summary` distills a `ThreadSnapshot`
//! into the lightweight `ThreadSummary` the task list keeps, and
//! `conversation_to_items` walks a `Conversation` (plus `TurnLog` for
//! per-turn usage stats) into the `Vec<DisplayItem>` the chat renderer
//! consumes. The remaining functions are pieces of the
//! `add_message_items` walk plus the small XML/JSON helpers used to
//! parse dispatch notifications and fuse sync tool results onto their
//! originating `ToolCall` rows.
//!
//! Free functions, no `ChatApp` access: every input is a borrow of a
//! wire type and every output is a `DisplayItem` (or scalar). The
//! corresponding tests live in this module too — they're regression
//! fixtures over `conversation_to_items` and don't share state with
//! anything else in `app.rs`.

use whisper_agent_protocol::{
    Attachment, ContentBlock, Conversation, ImageSource, Message, Role, ThreadSummary,
    ToolResultContent, TurnLog,
};

use super::{DiffPayload, DisplayItem, FusedToolResult};

pub(super) fn snapshot_summary(s: &whisper_agent_protocol::ThreadSnapshot) -> ThreadSummary {
    ThreadSummary {
        thread_id: s.thread_id.clone(),
        pod_id: s.pod_id.clone(),
        title: s.title.clone(),
        state: s.state,
        created_at: s.created_at.clone(),
        last_active: s.last_active.clone(),
        origin: s.origin.clone(),
        continued_from: s.continued_from.clone(),
        dispatched_by: s.dispatched_by.clone(),
    }
}

pub(super) fn conversation_to_items(conv: &Conversation, turn_log: &TurnLog) -> Vec<DisplayItem> {
    // Interleave a `DisplayItem::TurnStats` after each Assistant-role
    // message, pulled in order from `turn_log.entries`. The runtime
    // pushes exactly one entry per `integrate_model_response`, so entry
    // N corresponds to the Nth Assistant message. Extra entries (never
    // expected) are dropped; a short log (older threads, or
    // mid-migration) leaves trailing turns without a stats row rather
    // than crashing.
    let mut items = Vec::new();
    let mut entry_iter = turn_log.entries.iter();
    for (msg_index, msg) in conv.messages().iter().enumerate() {
        add_message_items(msg, msg_index, &mut items);
        if msg.role == Role::Assistant
            && let Some(entry) = entry_iter.next()
        {
            items.push(DisplayItem::TurnStats { usage: entry.usage });
        }
    }
    items
}

pub(super) fn add_message_items(msg: &Message, msg_index: usize, out: &mut Vec<DisplayItem>) {
    match msg.role {
        Role::System => {
            // System prompt lives at the head of the conversation as a
            // single `ContentBlock::Text`. Empty prompts produce no row
            // so the chat log doesn't start with a meaningless
            // "(empty)" entry.
            let text = msg
                .content
                .iter()
                .find_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .unwrap_or("");
            if !text.is_empty() {
                out.push(DisplayItem::SetupPrompt {
                    text: text.to_string(),
                });
            }
        }
        Role::Tools => {
            // Tool manifest: one `ContentBlock::ToolSchema` per tool.
            // The collapsed row shows the count; expanded shows
            // name + description + input-schema for each entry.
            let mut rendered = String::new();
            let mut count = 0usize;
            for block in &msg.content {
                if let ContentBlock::ToolSchema {
                    name,
                    description,
                    input_schema,
                } = block
                {
                    if count > 0 {
                        rendered.push_str("\n\n");
                    }
                    rendered.push_str(name);
                    if !description.is_empty() {
                        rendered.push_str(" — ");
                        rendered.push_str(description);
                    }
                    rendered.push('\n');
                    rendered
                        .push_str(&serde_json::to_string_pretty(input_schema).unwrap_or_default());
                    count += 1;
                }
            }
            if count > 0 {
                out.push(DisplayItem::SetupTools {
                    count,
                    text: rendered,
                });
            }
        }
        Role::User => {
            // Collapse a single User message's text + image content
            // into one `DisplayItem::User` entry so they render as
            // one speech bubble. A user turn with multiple text
            // blocks (rare, but possible via fork-edit) fuses them
            // with a blank line so rendered Markdown sees a stable
            // shape.
            let mut text_buf = String::new();
            let mut attachments: Vec<Attachment> = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        if !text_buf.is_empty() {
                            text_buf.push_str("\n\n");
                        }
                        text_buf.push_str(text);
                    }
                    ContentBlock::Image { source } => {
                        attachments.push(Attachment::Image {
                            source: source.clone(),
                        });
                    }
                    _ => {}
                }
            }
            if !text_buf.is_empty() || !attachments.is_empty() {
                out.push(DisplayItem::User {
                    text: text_buf,
                    msg_index,
                    attachments,
                });
            }
        }
        Role::ToolResult => {
            // Role::ToolResult carries either structured ToolResult
            // content blocks (synchronous tool-call results) or plain
            // text (async `dispatch_thread` callbacks). Both get
            // rendered as their own chat row at their chronological
            // position — no fusion into earlier tool-call items —
            // so the operator can see each tool response where it
            // actually landed in the conversation. Proximity to the
            // matching tool-call item drives the default-collapsed
            // behavior.
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        push_tool_result_from_text(out, text);
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let text = tool_result_text(content);
                        let attachments = content.image_sources().into_iter().cloned().collect();
                        push_tool_result(out, tool_use_id.clone(), text, *is_error, attachments);
                    }
                    _ => {}
                }
            }
        }
        Role::Assistant => {
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        out.push(DisplayItem::AssistantText { text: text.clone() });
                    }
                    ContentBlock::ToolUse {
                        id, name, input, ..
                    } => {
                        let preview =
                            truncate(serde_json::to_string(input).unwrap_or_default(), 200);
                        out.push(build_tool_call_item(
                            id.clone(),
                            name.clone(),
                            Some(input),
                            preview,
                        ));
                    }
                    ContentBlock::Thinking { thinking, .. } => {
                        out.push(DisplayItem::Reasoning {
                            text: thinking.clone(),
                        });
                    }
                    ContentBlock::Image { source } => {
                        out.push(DisplayItem::AssistantImage {
                            source: source.clone(),
                        });
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Push a `DisplayItem::ToolResult` row built from a text payload.
/// For `dispatch_thread` async callbacks the payload is a
/// `<dispatched-thread-notification>` XML envelope — we extract the
/// originating `tool-use-id`, the inner `<result>`, and the `<status>`
/// to drive `is_error`. Any other shape falls back to a `SystemNote`
/// row so unknown text stays visible.
pub(super) fn push_tool_result_from_text(items: &mut Vec<DisplayItem>, text: &str) {
    if let Some((tool_use_id, result_body, is_error)) = parse_dispatch_notification(text) {
        push_tool_result(items, tool_use_id, result_body, is_error, Vec::new());
        return;
    }
    items.push(DisplayItem::SystemNote {
        text: text.to_string(),
        is_error: false,
    });
}

/// Route a tool_result payload onto the items stream.
///
/// If the matching `DisplayItem::ToolCall` can be found walking
/// backward without crossing a `User` / `AssistantText` boundary, the
/// result is fused onto that call (set the call's `result` slot). The
/// common case — sync tool calls and the initial ack of an async
/// `dispatch_thread` — renders as a single combined row instead of
/// two stacked rows, cutting chat-log noise.
///
/// Otherwise (boundary crossed, or no matching call in view) push a
/// standalone `DisplayItem::ToolResult` at the tail. This covers the
/// distant async-callback case where the originating call lives
/// several turns earlier.
/// Append a streaming `ContentBlock` to the matching in-flight tool
/// call's `streaming_output` buffer. Only text blocks have a natural
/// inline rendering today; future non-text block kinds will show as
/// placeholders.
pub(super) fn append_streaming_output(
    items: &mut [DisplayItem],
    tool_use_id: &str,
    block: &ContentBlock,
) {
    for item in items.iter_mut().rev() {
        match item {
            DisplayItem::ToolCall {
                tool_use_id: id,
                streaming_output,
                result: None,
                ..
            } if id == tool_use_id => {
                match block {
                    ContentBlock::Text { text } => streaming_output.push_str(text),
                    _ => streaming_output.push_str("[non-text content]"),
                }
                return;
            }
            DisplayItem::User { .. } | DisplayItem::AssistantText { .. } => return,
            _ => continue,
        }
    }
}

pub(super) fn push_tool_result(
    items: &mut Vec<DisplayItem>,
    tool_use_id: String,
    text: String,
    is_error: bool,
    attachments: Vec<ImageSource>,
) {
    for item in items.iter_mut().rev() {
        match item {
            DisplayItem::ToolCall {
                tool_use_id: id,
                result,
                ..
            } if id == &tool_use_id => {
                // Fuse only if the call doesn't already have a
                // result. A second arriving result for the same
                // tool_use_id is unusual but can happen (error
                // retries, protocol quirks); fall through to the
                // standalone-row path so the newer data isn't lost
                // silently.
                if result.is_none() {
                    *result = Some(FusedToolResult {
                        text,
                        is_error,
                        attachments,
                    });
                    return;
                }
                break;
            }
            // Crossing an assistant/user turn means the result is
            // chronologically separated from its call — push as a
            // standalone row at its arrival position.
            DisplayItem::User { .. } | DisplayItem::AssistantText { .. } => break,
            _ => continue,
        }
    }
    let name = items
        .iter()
        .rev()
        .find_map(|item| match item {
            DisplayItem::ToolCall {
                tool_use_id: id,
                name,
                ..
            } if id == &tool_use_id => Some(name.clone()),
            _ => None,
        })
        .unwrap_or_default();
    items.push(DisplayItem::ToolResult {
        tool_use_id,
        name,
        text,
        is_error,
        attachments,
    });
}

/// Parse a `<dispatched-thread-notification>` XML envelope emitted by
/// the server's async dispatch flush. Returns `(tool_use_id, result,
/// is_error)` on a match. Deliberately string-based rather than a
/// full XML parser — our envelope is a fixed shape, tiny, and the
/// server XML-escapes its fields, so a tag-bounded scan is enough.
/// Returns `None` on any structural mismatch so the caller falls
/// back to a generic display.
pub(super) fn parse_dispatch_notification(text: &str) -> Option<(String, String, bool)> {
    if !text
        .trim_start()
        .starts_with("<dispatched-thread-notification>")
    {
        return None;
    }
    let tool_use_id = extract_tagged_value(text, "tool-use-id")?;
    let result = extract_tagged_value(text, "result").unwrap_or_default();
    let status = extract_tagged_value(text, "status").unwrap_or_default();
    let is_error = matches!(status.as_str(), "failed" | "cancelled");
    // Unescape the XML-escaped <result> body so diffs / code show
    // through correctly rather than as `&lt;T&gt;`.
    Some((tool_use_id, unescape_xml(&result), is_error))
}

pub(super) fn extract_tagged_value(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end_rel = text[start..].find(&close)?;
    Some(text[start..start + end_rel].to_string())
}

pub(super) fn unescape_xml(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

/// Build a `DisplayItem::ToolCall` from the wire shape (full input
/// `Value` available). `args_preview` is the server-truncated string;
/// we only fall back to it for the summary when the structured args
/// don't have a specialized renderer.
///
/// Special cases:
///   * `edit_file` / `write_file` — pull `path` for the header, and
///     compute a `DiffPayload` so the renderer can show old vs new.
///   * `bash` — pull `command` for the header.
///   * everything else — use the truncated JSON preview as the summary.
pub(super) fn build_tool_call_item(
    tool_use_id: String,
    name: String,
    args: Option<&serde_json::Value>,
    args_preview: String,
) -> DisplayItem {
    let summary = tool_summary(&name, args, &args_preview);
    let args_pretty = args.and_then(|v| serde_json::to_string_pretty(v).ok());
    let diff = args.and_then(|v| extract_diff(&name, v));
    DisplayItem::ToolCall {
        tool_use_id,
        name,
        summary,
        args_pretty,
        diff,
        streaming_output: String::new(),
        result: None,
    }
}

pub(super) fn tool_summary(name: &str, args: Option<&serde_json::Value>, fallback: &str) -> String {
    let Some(v) = args else {
        return fallback.to_string();
    };
    let pick = |key: &str| v.get(key).and_then(|s| s.as_str()).map(str::to_owned);
    match name {
        "edit_file" | "write_file" | "read_file" => pick("path").unwrap_or_else(|| fallback.into()),
        "bash" => pick("command")
            .map(|c| truncate(c, 120))
            .unwrap_or_else(|| fallback.into()),
        "grep" => pick("pattern").unwrap_or_else(|| fallback.into()),
        "glob" => pick("pattern").unwrap_or_else(|| fallback.into()),
        "list_dir" => pick("path").unwrap_or_else(|| ".".into()),
        _ => fallback.to_string(),
    }
}

pub(super) fn extract_diff(name: &str, args: &serde_json::Value) -> Option<DiffPayload> {
    let s = |key: &str| args.get(key).and_then(|v| v.as_str()).map(str::to_owned);
    match name {
        "edit_file" => Some(DiffPayload {
            path: s("path")?,
            old_text: s("old_string")?,
            new_text: s("new_string")?,
            is_creation: false,
        }),
        "write_file" => Some(DiffPayload {
            path: s("path")?,
            old_text: String::new(),
            new_text: s("content")?,
            is_creation: true,
        }),
        _ => None,
    }
}

pub(super) fn tool_result_text(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(t) => t.clone(),
        ToolResultContent::Blocks(blocks) => {
            let mut out = String::new();
            for b in blocks {
                if let ContentBlock::Text { text } = b {
                    out.push_str(text);
                }
            }
            out
        }
    }
}

pub(super) fn truncate(mut s: String, max: usize) -> String {
    if s.len() > max {
        let mut cut = max;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push('…');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::{ContentBlock, Conversation, Message, ToolResultContent};

    fn conv_with_tool_call() -> Conversation {
        let mut conv = Conversation::new();
        conv.push(Message::user_text("hi"));
        conv.push(Message::assistant_blocks(vec![
            ContentBlock::Text {
                text: "running a tool".into(),
            },
            ContentBlock::ToolUse {
                id: "tu-1".into(),
                name: "bash".into(),
                input: serde_json::json!({ "command": "ls" }),
                replay: None,
            },
        ]));
        conv.push(Message::tool_result_blocks(vec![
            ContentBlock::ToolResult {
                tool_use_id: "tu-1".into(),
                content: ToolResultContent::Text("file1\nfile2\n".into()),
                is_error: false,
            },
        ]));
        conv
    }

    #[test]
    fn snapshot_rebuild_fuses_sync_result_into_tool_call() {
        let items = conversation_to_items(&conv_with_tool_call(), &TurnLog::default());
        // The matching sync tool_result should fuse into the ToolCall
        // (populate its `result` slot) rather than producing a
        // standalone ToolResult row.
        let fused = items
            .iter()
            .find_map(|i| match i {
                DisplayItem::ToolCall {
                    tool_use_id,
                    result,
                    ..
                } if tool_use_id == "tu-1" => result.as_ref().map(|r| (r.text.clone(), r.is_error)),
                _ => None,
            })
            .expect("ToolCall with fused result should be present");
        assert_eq!(fused.0, "file1\nfile2\n");
        assert!(!fused.1);
        let standalone = items.iter().any(
            |i| matches!(i, DisplayItem::ToolResult { tool_use_id, .. } if tool_use_id == "tu-1"),
        );
        assert!(
            !standalone,
            "sync result should not also appear as a standalone ToolResult row"
        );
    }

    #[test]
    fn snapshot_rebuild_of_real_file_shows_tool_calls() {
        // Regression-style fixture: parse the real persisted thread
        // the user reported as missing tool calls in the webui and
        // assert that add_message_items produces ToolCall items for
        // every assistant tool_use in the file.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../sandbox/pods/workspace/threads/task-18a79d4b2206aa08.json");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipping: fixture {:?} not available in this env", path);
            return;
        };
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let conv: Conversation =
            serde_json::from_value(val.get("conversation").unwrap().clone()).unwrap();
        let items = conversation_to_items(&conv, &TurnLog::default());
        let tool_call_count = items
            .iter()
            .filter(|i| matches!(i, DisplayItem::ToolCall { .. }))
            .count();
        let roles_debug: Vec<&'static str> = items
            .iter()
            .map(|i| match i {
                DisplayItem::User { .. } => "user",
                DisplayItem::AssistantText { .. } => "assistant_text",
                DisplayItem::AssistantImage { .. } => "assistant_image",
                DisplayItem::Reasoning { .. } => "reasoning",
                DisplayItem::ToolCall { .. } => "tool_call",
                DisplayItem::ToolCallStreaming { .. } => "tool_call_streaming",
                DisplayItem::ToolResult { .. } => "tool_result",
                DisplayItem::SystemNote { .. } => "system_note",
                DisplayItem::SetupPrompt { .. } => "setup_prompt",
                DisplayItem::SetupTools { .. } => "setup_tools",
                DisplayItem::TurnStats { .. } => "turn_stats",
            })
            .collect();
        assert!(
            tool_call_count > 0,
            "expected at least one ToolCall item, got items={roles_debug:?}"
        );
        // Every ToolCall should have a fused result (the 4 sync
        // tool_result messages land on their originating calls
        // directly via the fusion walk).
        for item in &items {
            if let DisplayItem::ToolCall {
                tool_use_id,
                result,
                ..
            } = item
            {
                assert!(
                    result.is_some(),
                    "expected fused result on ToolCall {tool_use_id}"
                );
            }
        }
        // The 2 async XML callbacks (msg 10/12) land after an
        // intervening assistant turn, so they push standalone
        // ToolResult rows.
        let standalone_count = items
            .iter()
            .filter(|i| matches!(i, DisplayItem::ToolResult { .. }))
            .count();
        assert_eq!(
            standalone_count, 2,
            "expected 2 standalone ToolResult rows for the async callbacks"
        );
    }

    #[test]
    fn snapshot_rebuild_emits_two_rows_for_async_dispatch() {
        // Async `dispatch_thread` produces an initial sync ack plus
        // a later XML-envelope callback — two separate tool_result
        // rows bound to the same tool_use_id. Both should appear
        // distinctly in the snapshot replay.
        let mut conv = Conversation::new();
        conv.push(Message::user_text("dispatch async"));
        conv.push(Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: "tu-async".into(),
            name: "dispatch_thread".into(),
            input: serde_json::json!({ "prompt": "go", "sync": false }),
            replay: None,
        }]));
        conv.push(Message::tool_result_blocks(vec![
            ContentBlock::ToolResult {
                tool_use_id: "tu-async".into(),
                content: ToolResultContent::Text("Dispatched task-X".into()),
                is_error: false,
            },
        ]));
        conv.push(Message::assistant_blocks(vec![ContentBlock::Text {
            text: "waiting for the dispatched thread".into(),
        }]));
        conv.push(Message::tool_result_text(
            "<dispatched-thread-notification>\n  <thread-id>task-X</thread-id>\n  \
             <tool-use-id>tu-async</tool-use-id>\n  <status>completed</status>\n  \
             <summary>done</summary>\n  <result>the real final answer with &lt;code&gt; in it</result>\n  \
             <usage><total_tokens>10</total_tokens><tool_uses>0</tool_uses><duration_ms>5</duration_ms></usage>\n\
             </dispatched-thread-notification>",
        ));
        let items = conversation_to_items(&conv, &TurnLog::default());
        // The initial sync ack fuses into the ToolCall (no intervening
        // turn at the time it lands); the async XML callback arrives
        // after an assistant turn and pushes a standalone row.
        let fused = items
            .iter()
            .find_map(|i| match i {
                DisplayItem::ToolCall {
                    tool_use_id,
                    result,
                    ..
                } if tool_use_id == "tu-async" => result.as_ref().map(|r| r.text.clone()),
                _ => None,
            })
            .expect("ToolCall should have the sync ack fused onto it");
        assert_eq!(fused, "Dispatched task-X");
        let standalone: Vec<String> = items
            .iter()
            .filter_map(|i| match i {
                DisplayItem::ToolResult {
                    tool_use_id, text, ..
                } if tool_use_id == "tu-async" => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            standalone.len(),
            1,
            "expected only the async callback as a standalone row"
        );
        assert_eq!(standalone[0], "the real final answer with <code> in it");
    }
}
