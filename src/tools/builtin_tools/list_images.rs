//! `list_images` — synchronous scheduler intercept that returns the
//! conversation's image clipboard with content-addressed handles.
//!
//! Pairs with `save_image` (mcp-host worker tool) and `recall_image`
//! (server-side scheduler intercept). The model reads this catalog
//! to discover handles, then passes them to the other two by name.
//! Handles are stable under later pushes — see
//! `crate::runtime::v2_dispatch::image_handle` — so a handle the
//! model sees here remains valid for the rest of the conversation
//! even as new images arrive.

use serde_json::{Value, json};
use whisper_agent_protocol::{
    ContentBlock, Conversation, ImageMime, ImageSource, Role, ToolResultContent,
};

use super::LIST_IMAGES;
use crate::runtime::v2_dispatch::image_handle;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: LIST_IMAGES.into(),
        description: "List images currently visible in this conversation, most recent first. \
                      Each entry has a stable, content-addressed `handle` (8 hex chars) you \
                      can pass to `save_image` (write the image to a file) or `recall_image` \
                      (re-introduce the image content into the conversation, useful when an \
                      older image is no longer in your view). Handles are content-addressed: \
                      they don't shift when new images arrive, and identical bytes produce \
                      identical handles. URL-source images aren't reachable and are omitted."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {},
        }),
        annotations: ToolAnnotations {
            title: Some("List images".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

/// One image entry in the listing. Position-from-end gives recency
/// in messages (`0` = current message, `1` = one back, …); the
/// renderer turns that into prose.
#[derive(Debug, PartialEq)]
pub struct ImageEntry {
    pub handle: String,
    pub mime: ImageMime,
    pub source: ImageOrigin,
    pub messages_back: usize,
}

/// Where an image lives in the conversation. `ToolResult` covers
/// both built-in and host-env tool results — we don't try to surface
/// the originating tool name today (would need to follow the
/// `tool_use_id` back to its `ToolUse` block, which is more work
/// than v1 needs).
#[derive(Debug, PartialEq, Eq)]
pub enum ImageOrigin {
    User,
    Assistant,
    ToolResult,
}

impl ImageOrigin {
    fn label(&self) -> &'static str {
        match self {
            ImageOrigin::User => "user upload",
            ImageOrigin::Assistant => "assistant",
            ImageOrigin::ToolResult => "tool result",
        }
    }
}

/// Walk `conversation` in most-recent-first order, yielding one
/// [`ImageEntry`] per inline-bytes image. Images embedded in tool
/// results are surfaced too. URL-source images are skipped (they
/// can't ride as inline-bytes attachments and so aren't reachable
/// by `save_image` / `recall_image` anyway).
pub fn collect_entries(conversation: &Conversation) -> Vec<ImageEntry> {
    let total = conversation.messages().len();
    let mut out = Vec::new();
    for (idx, msg) in conversation.messages().iter().enumerate().rev() {
        let messages_back = total.saturating_sub(idx + 1);
        let role_origin = match msg.role {
            Role::User => ImageOrigin::User,
            _ => ImageOrigin::Assistant,
        };
        for block in msg.content.iter().rev() {
            match block {
                ContentBlock::Image {
                    source: ImageSource::Bytes { data, media_type },
                    ..
                } => {
                    out.push(ImageEntry {
                        handle: image_handle(data),
                        mime: *media_type,
                        // Inline image blocks sit in the message
                        // directly; their origin is the message's
                        // role (not a tool result).
                        source: match role_origin {
                            ImageOrigin::User => ImageOrigin::User,
                            ImageOrigin::Assistant => ImageOrigin::Assistant,
                            ImageOrigin::ToolResult => ImageOrigin::Assistant,
                        },
                        messages_back,
                    });
                }
                ContentBlock::ToolResult {
                    content: ToolResultContent::Blocks(blocks),
                    ..
                } => {
                    for inner in blocks.iter().rev() {
                        if let ContentBlock::Image {
                            source: ImageSource::Bytes { data, media_type },
                            ..
                        } = inner
                        {
                            out.push(ImageEntry {
                                handle: image_handle(data),
                                mime: *media_type,
                                source: ImageOrigin::ToolResult,
                                messages_back,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// Format the listing as a human-readable text block. Caller wraps
/// in a `CallToolResult::text` for the tool result. Empty histories
/// produce a one-line "no images" notice rather than an empty body
/// so the model can tell the call succeeded but found nothing.
pub fn render(entries: &[ImageEntry]) -> String {
    if entries.is_empty() {
        return "No images in current conversation.".into();
    }
    let mut out = format!(
        "{} image{} in current conversation, most recent first:\n\n",
        entries.len(),
        if entries.len() == 1 { "" } else { "s" }
    );
    for e in entries {
        let when = match e.messages_back {
            0 => "current message".to_string(),
            1 => "1 message ago".to_string(),
            n => format!("{n} messages ago"),
        };
        out.push_str(&format!(
            "  {handle}  {mime:<11}  {when:<20}  {source}\n",
            handle = e.handle,
            mime = e.mime.as_mime_str(),
            when = when,
            source = e.source.label(),
        ));
    }
    out.push('\n');
    out.push_str(
        "Pass a `handle` to `save_image` to write to a file, or to `recall_image` \
         to re-introduce the image into the conversation.\n",
    );
    out
}

/// Trivial parser — the tool takes no arguments today. Kept as a
/// function so the intercept site has a single error surface for
/// "model passed garbage."
pub fn parse_args(value: Value) -> Result<(), String> {
    if value.is_null() || matches!(value.as_object(), Some(o) if o.is_empty()) {
        Ok(())
    } else {
        Err(format!("list_images takes no arguments; got: {}", value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::{Message, Role};

    fn user_image_msg(bytes: &[u8], mime: ImageMime) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Image {
                source: ImageSource::Bytes {
                    data: bytes.to_vec(),
                    media_type: mime,
                },
                replay: None,
            }],
        }
    }

    #[test]
    fn empty_conversation_renders_no_images_notice() {
        let conv = Conversation::new();
        let entries = collect_entries(&conv);
        assert!(entries.is_empty());
        let body = render(&entries);
        assert!(body.contains("No images"));
    }

    #[test]
    fn most_recent_first_ordering() {
        let mut conv = Conversation::new();
        conv.push(user_image_msg(b"oldest", ImageMime::Jpeg));
        conv.push(user_image_msg(b"middle", ImageMime::Png));
        conv.push(user_image_msg(b"newest", ImageMime::Webp));
        let entries = collect_entries(&conv);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].handle, image_handle(b"newest"));
        assert_eq!(entries[1].handle, image_handle(b"middle"));
        assert_eq!(entries[2].handle, image_handle(b"oldest"));
        assert_eq!(entries[0].messages_back, 0);
        assert_eq!(entries[2].messages_back, 2);
    }

    /// Identical bytes (e.g. a recall echoing an earlier image) get
    /// the same handle. The catalog will show both rows, both with
    /// the same handle — that's accurate and the model can address
    /// either with the same `save_image` call.
    #[test]
    fn identical_bytes_share_a_handle() {
        let mut conv = Conversation::new();
        conv.push(user_image_msg(b"same-bytes", ImageMime::Png));
        conv.push(user_image_msg(b"different", ImageMime::Png));
        conv.push(user_image_msg(b"same-bytes", ImageMime::Png));
        let entries = collect_entries(&conv);
        assert_eq!(entries[0].handle, entries[2].handle);
        assert_ne!(entries[0].handle, entries[1].handle);
    }

    #[test]
    fn url_source_images_skipped() {
        let mut conv = Conversation::new();
        conv.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Image {
                source: ImageSource::Url {
                    url: "https://example/cat.png".into(),
                },
                replay: None,
            }],
        });
        assert!(collect_entries(&conv).is_empty());
    }

    #[test]
    fn tool_result_images_classified_as_tool_result() {
        let mut conv = Conversation::new();
        conv.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: ToolResultContent::Blocks(vec![ContentBlock::Image {
                    source: ImageSource::Bytes {
                        data: b"screenshot".to_vec(),
                        media_type: ImageMime::Png,
                    },
                    replay: None,
                }]),
                is_error: false,
            }],
        });
        let entries = collect_entries(&conv);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].source, ImageOrigin::ToolResult);
    }

    #[test]
    fn rejects_arguments() {
        assert!(parse_args(json!({})).is_ok());
        assert!(parse_args(Value::Null).is_ok());
        assert!(parse_args(json!({"x": 1})).is_err());
    }

    /// Pin the rendered shape so future formatter edits don't
    /// silently change what the model reads.
    #[test]
    fn render_includes_handle_mime_recency_source() {
        let entries = vec![ImageEntry {
            handle: "abc12345".into(),
            mime: ImageMime::Png,
            source: ImageOrigin::ToolResult,
            messages_back: 3,
        }];
        let body = render(&entries);
        assert!(body.contains("abc12345"));
        assert!(body.contains("image/png"));
        assert!(body.contains("3 messages ago"));
        assert!(body.contains("tool result"));
    }
}
