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

/// One image entry in the listing. `turns_back` counts user-role
/// messages between the image and the end of the conversation —
/// `0` means the image was emitted in the current assistant turn;
/// `1` means a user message has happened since (so it's "1 turn
/// ago" relative to the current assistant turn). `dimensions` is
/// `None` when the format is one we don't know how to parse
/// (HEIC/HEIF) or the bytes are malformed.
#[derive(Debug, PartialEq)]
pub struct ImageEntry {
    pub handle: String,
    pub mime: ImageMime,
    pub source: ImageOrigin,
    pub turns_back: usize,
    pub dimensions: Option<(u32, u32)>,
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
///
/// `turns_back` is computed by tracking how many `Role::User`
/// messages we've passed during the reverse walk: the count seen
/// *before* processing a message's images is exactly the number of
/// user messages that came after it. So an image in the current
/// assistant turn (no user message after it yet) gets
/// `turns_back = 0`.
pub fn collect_entries(conversation: &Conversation) -> Vec<ImageEntry> {
    let mut out = Vec::new();
    let mut user_msgs_after: usize = 0;
    for msg in conversation.messages().iter().rev() {
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
                        turns_back: user_msgs_after,
                        dimensions: image_dimensions(data),
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
                                turns_back: user_msgs_after,
                                dimensions: image_dimensions(data),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        if msg.role == Role::User {
            user_msgs_after += 1;
        }
    }
    out
}

/// Best-effort header-only dimension parser for the formats in
/// `ImageMime`. Avoids pulling the full `image` crate just to read
/// width/height. Returns `None` for HEIC/HEIF (structurally
/// complex) and for any malformed input.
fn image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    // PNG: signature 89 50 4E 47 0D 0A 1A 0A, then 4-byte chunk
    // length, then "IHDR", then 4-byte width + 4-byte height
    // (big-endian). Width starts at offset 16, height at 20.
    if bytes.len() >= 24 && bytes.starts_with(b"\x89PNG\r\n\x1a\n") && &bytes[12..16] == b"IHDR" {
        let w = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
        let h = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
        return Some((w, h));
    }
    // GIF: "GIF87a" or "GIF89a", then 2-byte width + 2-byte height
    // (little-endian) at offset 6.
    if bytes.len() >= 10 && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        let w = u16::from_le_bytes(bytes[6..8].try_into().ok()?) as u32;
        let h = u16::from_le_bytes(bytes[8..10].try_into().ok()?) as u32;
        return Some((w, h));
    }
    // WebP: "RIFF" .... "WEBP" then a chunk; VP8 / VP8L / VP8X
    // each encode dimensions slightly differently.
    if bytes.len() >= 30 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        match &bytes[12..16] {
            // VP8 (lossy): start-code at offset 23-25 (9d 01 2a),
            // dims as 14-bit little-endian at 26-29.
            b"VP8 " if bytes.len() >= 30 && &bytes[23..26] == b"\x9d\x01\x2a" => {
                let w = u16::from_le_bytes(bytes[26..28].try_into().ok()?) & 0x3fff;
                let h = u16::from_le_bytes(bytes[28..30].try_into().ok()?) & 0x3fff;
                return Some((w as u32, h as u32));
            }
            // VP8L (lossless): signature 0x2f at offset 20, then
            // 14-bit width and 14-bit height packed across bytes
            // 21-24.
            b"VP8L" if bytes.len() >= 25 && bytes[20] == 0x2f => {
                let b0 = bytes[21] as u32;
                let b1 = bytes[22] as u32;
                let b2 = bytes[23] as u32;
                let b3 = bytes[24] as u32;
                let w = (b0 | ((b1 & 0x3f) << 8)) + 1;
                let h = ((b1 >> 6) | (b2 << 2) | ((b3 & 0x0f) << 10)) + 1;
                return Some((w, h));
            }
            // VP8X (extended): canvas width-1 / height-1 stored
            // as 24-bit little-endian at offsets 24-26 / 27-29.
            b"VP8X" if bytes.len() >= 30 => {
                let w = (bytes[24] as u32) | ((bytes[25] as u32) << 8) | ((bytes[26] as u32) << 16);
                let h = (bytes[27] as u32) | ((bytes[28] as u32) << 8) | ((bytes[29] as u32) << 16);
                return Some((w + 1, h + 1));
            }
            _ => {}
        }
    }
    // JPEG: scan for the first SOFn marker (FF C0..CF, excluding
    // FF C4 / FF C8 / FF CC which are not start-of-frame). After
    // FF Cn comes a 2-byte segment length, 1 byte precision, then
    // 2 bytes height, 2 bytes width — both big-endian.
    if bytes.len() >= 4 && bytes[0] == 0xff && bytes[1] == 0xd8 {
        let mut i = 2;
        while i + 9 < bytes.len() {
            if bytes[i] != 0xff {
                return None;
            }
            // Skip fill bytes (0xff 0xff ...).
            while i < bytes.len() && bytes[i] == 0xff {
                i += 1;
            }
            if i >= bytes.len() {
                return None;
            }
            let marker = bytes[i];
            i += 1;
            // Standalone markers (no length): RST0..7 (D0..D7),
            // SOI (D8), EOI (D9), TEM (01).
            if (0xd0..=0xd9).contains(&marker) || marker == 0x01 {
                continue;
            }
            // SOFn except DHT (C4), JPG (C8), DAC (CC).
            if (0xc0..=0xcf).contains(&marker)
                && marker != 0xc4
                && marker != 0xc8
                && marker != 0xcc
                && i + 7 < bytes.len()
            {
                // Skip 2-byte length + 1-byte precision.
                let h = u16::from_be_bytes(bytes[i + 3..i + 5].try_into().ok()?) as u32;
                let w = u16::from_be_bytes(bytes[i + 5..i + 7].try_into().ok()?) as u32;
                return Some((w, h));
            }
            // Any other marker carries a 2-byte length.
            if i + 1 >= bytes.len() {
                return None;
            }
            let seg_len = u16::from_be_bytes(bytes[i..i + 2].try_into().ok()?) as usize;
            i += seg_len;
        }
    }
    None
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
        let when = match e.turns_back {
            0 => "this turn".to_string(),
            1 => "1 turn ago".to_string(),
            n => format!("{n} turns ago"),
        };
        let dims = match e.dimensions {
            Some((w, h)) => format!("{w}x{h}"),
            None => "?x?".into(),
        };
        out.push_str(&format!(
            "  {handle}  {mime:<4}  {dims:<11}  {when:<13}  {source}\n",
            handle = e.handle,
            mime = mime_short(e.mime),
            dims = dims,
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

/// Compact mime label for the listing: just the subtype (`png`,
/// `jpeg`, …) without the `image/` prefix. Saves a column of width
/// in the rendered listing without losing information — every
/// entry in this catalog is an image by definition.
fn mime_short(mime: ImageMime) -> &'static str {
    match mime {
        ImageMime::Png => "png",
        ImageMime::Jpeg => "jpg",
        ImageMime::Gif => "gif",
        ImageMime::Webp => "webp",
        ImageMime::Heic => "heic",
        ImageMime::Heif => "heif",
    }
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
    }

    /// Each user message bumps `turns_back`. The current assistant
    /// turn (no user message after) reads as `0`; the previous
    /// turn's images are at `1`, etc.
    #[test]
    fn turns_back_counts_user_messages_after_image() {
        let mut conv = Conversation::new();
        // User turn 0:
        conv.push(user_image_msg(b"u0-img", ImageMime::Png));
        // User turn 1 starts:
        conv.push(user_image_msg(b"u1-img", ImageMime::Png));
        // Assistant emits an image during this turn (no further user msg).
        conv.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Image {
                source: ImageSource::Bytes {
                    data: b"a-img".to_vec(),
                    media_type: ImageMime::Png,
                },
                replay: None,
            }],
        });
        let entries = collect_entries(&conv);
        // newest-first: a-img (this turn), u1-img (this turn — the
        // user msg at idx 1 IS the start of this turn, no later
        // user msg yet), u0-img (1 turn ago).
        assert_eq!(entries[0].handle, image_handle(b"a-img"));
        assert_eq!(entries[0].turns_back, 0);
        assert_eq!(entries[1].handle, image_handle(b"u1-img"));
        assert_eq!(entries[1].turns_back, 0);
        assert_eq!(entries[2].handle, image_handle(b"u0-img"));
        assert_eq!(entries[2].turns_back, 1);
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
    fn render_includes_handle_mime_dimensions_recency_source() {
        let entries = vec![ImageEntry {
            handle: "abc12345".into(),
            mime: ImageMime::Png,
            source: ImageOrigin::ToolResult,
            turns_back: 3,
            dimensions: Some((1024, 768)),
        }];
        let body = render(&entries);
        assert!(body.contains("abc12345"));
        assert!(body.contains("png"));
        assert!(body.contains("1024x768"));
        assert!(body.contains("3 turns ago"));
        assert!(body.contains("tool result"));
    }

    /// Unknown dimensions render as `?x?` rather than being
    /// silently omitted — the model should be able to tell the
    /// difference between "we couldn't parse" and a real value.
    #[test]
    fn render_unknown_dimensions_as_placeholder() {
        let entries = vec![ImageEntry {
            handle: "abc12345".into(),
            mime: ImageMime::Heic,
            source: ImageOrigin::User,
            turns_back: 0,
            dimensions: None,
        }];
        let body = render(&entries);
        assert!(body.contains("?x?"));
    }

    /// Minimal PNG file (8-byte signature + IHDR chunk) — header
    /// alone, no real image data — should still parse to its
    /// declared dimensions. Pinning the parser against the spec
    /// rather than a particular generator's output.
    #[test]
    fn parses_png_dimensions_from_header() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        bytes.extend_from_slice(&[0, 0, 0, 13]); // IHDR length
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&1024u32.to_be_bytes()); // width
        bytes.extend_from_slice(&768u32.to_be_bytes()); // height
        bytes.extend_from_slice(&[8, 2, 0, 0, 0]); // bit depth, color type, compression, filter, interlace
        bytes.extend_from_slice(&[0, 0, 0, 0]); // CRC (not validated)
        assert_eq!(image_dimensions(&bytes), Some((1024, 768)));
    }

    #[test]
    fn parses_gif_dimensions_from_header() {
        let mut bytes = b"GIF89a".to_vec();
        bytes.extend_from_slice(&640u16.to_le_bytes()); // width
        bytes.extend_from_slice(&480u16.to_le_bytes()); // height
        bytes.extend_from_slice(&[0; 4]); // packed fields + bg + aspect
        assert_eq!(image_dimensions(&bytes), Some((640, 480)));
    }

    /// Synthetic JPEG: SOI + SOF0 marker carrying width/height,
    /// no image data. Just enough to exercise the marker scan.
    #[test]
    fn parses_jpeg_dimensions_from_sof0_marker() {
        let mut bytes = vec![0xff, 0xd8]; // SOI
        bytes.extend_from_slice(&[0xff, 0xc0]); // SOF0
        bytes.extend_from_slice(&[0, 17]); // segment length (incl. these 2 bytes)
        bytes.push(8); // sample precision
        bytes.extend_from_slice(&480u16.to_be_bytes()); // height
        bytes.extend_from_slice(&640u16.to_be_bytes()); // width
        bytes.extend_from_slice(&[0; 10]); // pad to satisfy seg_len
        assert_eq!(image_dimensions(&bytes), Some((640, 480)));
    }

    #[test]
    fn unparseable_bytes_return_none() {
        assert_eq!(image_dimensions(b"not an image"), None);
        assert_eq!(image_dimensions(&[]), None);
    }
}
