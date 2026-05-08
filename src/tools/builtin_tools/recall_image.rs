//! `recall_image` — synchronous scheduler intercept that
//! re-introduces an image from earlier in the conversation as a
//! tool-result content block.
//!
//! Useful when an older image has fallen out of the model's view
//! (long conversation, compaction in some future world, or just
//! "I want a fresh look at it"). Pairs with `list_images` for
//! handle discovery and `save_image` for persistence.
//!
//! Intentional design choice: handles are content-addressed, so a
//! recall doesn't shift handles for any other image. The recalled
//! echo has the *same* handle as the original — both rows show up
//! in subsequent `list_images` calls but addressing either with
//! `save_image` / `recall_image` is unambiguous.

use serde::Deserialize;
use serde_json::{Value, json};
use whisper_agent_protocol::{
    ContentBlock, Conversation, ImageMime, ImageSource, ToolResultContent,
};

use super::RECALL_IMAGE;
use crate::runtime::v2_dispatch::image_handle;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: RECALL_IMAGE.into(),
        description: "Re-introduce an image from this conversation's history into your view. \
                      Pass a `handle` from `list_images`. The result includes the image \
                      content plus a short metadata header. Useful when an older image is \
                      no longer visible to you (long conversation, etc.) and you want a \
                      fresh look. Handles are content-addressed and stable: a recall \
                      doesn't shift any other image's handle, and re-recalling resolves \
                      to the same content."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Content-addressed image handle (8 hex chars) from `list_images`."
                }
            },
            "required": ["handle"]
        }),
        annotations: ToolAnnotations {
            title: Some("Recall image".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize)]
pub struct RecallImageArgs {
    pub handle: String,
}

pub fn parse_args(value: Value) -> Result<RecallImageArgs, String> {
    serde_json::from_value::<RecallImageArgs>(value)
        .map_err(|e| format!("invalid recall_image arguments: {e}"))
}

/// Resolved image data plus the bytes-position metadata the
/// renderer attaches as a header. `messages_back = 0` would mean
/// the image lives in the current message — common when the model
/// is recalling something it just saw, but normal otherwise.
pub struct RecalledImage {
    pub bytes: Vec<u8>,
    pub mime: ImageMime,
    pub messages_back: usize,
}

/// Walk the conversation looking for an image whose
/// [`image_handle`] matches `handle`. Returns the first match
/// (most-recent-first traversal) — duplicates are by definition
/// identical bytes, so any pick is fine. `handle` is matched
/// case-insensitively to mirror the resolver in
/// [`crate::runtime::v2_dispatch::resolve_content_refs`].
pub fn find_by_handle(conversation: &Conversation, handle: &str) -> Option<RecalledImage> {
    let target = handle.trim().to_ascii_lowercase();
    let total = conversation.messages().len();
    for (idx, msg) in conversation.messages().iter().enumerate().rev() {
        let messages_back = total.saturating_sub(idx + 1);
        for block in msg.content.iter().rev() {
            match block {
                ContentBlock::Image {
                    source: ImageSource::Bytes { data, media_type },
                    ..
                } if image_handle(data) == target => {
                    return Some(RecalledImage {
                        bytes: data.clone(),
                        mime: *media_type,
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
                            && image_handle(data) == target
                        {
                            return Some(RecalledImage {
                                bytes: data.clone(),
                                mime: *media_type,
                                messages_back,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Build the metadata text header that goes alongside the image
/// content block. Short and machine-readable so the model can pair
/// the visual with provenance without parsing prose.
pub fn render_header(handle: &str, image: &RecalledImage) -> String {
    let when = match image.messages_back {
        0 => "current message".to_string(),
        1 => "1 message ago".to_string(),
        n => format!("{n} messages ago"),
    };
    format!(
        "Recalled image {handle} ({mime}, originally from {when}):",
        mime = image.mime.as_mime_str(),
    )
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
    fn finds_image_by_handle() {
        let mut conv = Conversation::new();
        conv.push(user_image_msg(b"first", ImageMime::Png));
        conv.push(user_image_msg(b"second", ImageMime::Jpeg));
        let target = image_handle(b"first");
        let recalled = find_by_handle(&conv, &target).expect("found");
        assert_eq!(recalled.bytes, b"first");
        assert_eq!(recalled.mime, ImageMime::Png);
        // first message was at index 0, total=2, so messages_back = 1.
        assert_eq!(recalled.messages_back, 1);
    }

    #[test]
    fn case_insensitive_handle_match() {
        let mut conv = Conversation::new();
        conv.push(user_image_msg(b"x", ImageMime::Png));
        let target = image_handle(b"x").to_ascii_uppercase();
        assert!(find_by_handle(&conv, &target).is_some());
    }

    #[test]
    fn unknown_handle_returns_none() {
        let mut conv = Conversation::new();
        conv.push(user_image_msg(b"x", ImageMime::Png));
        assert!(find_by_handle(&conv, "00000000").is_none());
    }

    /// When the same bytes appear twice, the most-recent occurrence
    /// wins. Both rows share a handle by content-addressing, so
    /// either pick is correct — but we should at least be
    /// deterministic about it.
    #[test]
    fn duplicate_bytes_returns_most_recent() {
        let mut conv = Conversation::new();
        conv.push(user_image_msg(b"same", ImageMime::Png));
        conv.push(user_image_msg(b"intermediate", ImageMime::Webp));
        conv.push(user_image_msg(b"same", ImageMime::Png));
        let target = image_handle(b"same");
        let recalled = find_by_handle(&conv, &target).expect("found");
        // Most recent of the two duplicates is at messages_back = 0.
        assert_eq!(recalled.messages_back, 0);
    }

    #[test]
    fn header_includes_handle_mime_and_recency() {
        let img = RecalledImage {
            bytes: vec![],
            mime: ImageMime::Png,
            messages_back: 5,
        };
        let h = render_header("abc12345", &img);
        assert!(h.contains("abc12345"));
        assert!(h.contains("image/png"));
        assert!(h.contains("5 messages ago"));
    }

    #[test]
    fn parse_args_requires_handle() {
        assert!(parse_args(json!({})).is_err());
        assert!(parse_args(json!({ "handle": "abc12345" })).is_ok());
    }
}
