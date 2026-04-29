//! Conversation renderer — event-log layout.
//!
//! Each item draws as a full-width row with a 3px role-colored gutter on
//! the left edge. User messages get a faint background tint to mark
//! them as section breaks (they're rare relative to model output).
//! Agent text and reasoning have transparent backgrounds — the gutter
//! alone carries the role signal so the bulk of the conversation reads
//! as a quiet stream of model output, not a chat log.
//!
//! Tool calls render as a collapsible row that fuses the call and
//! its immediate result into one entry — `name summary preview
//! [status]` in the collapsed header, args/diff + full result in
//! the expanded body. Results that arrive "distant" from their
//! call (separated by an assistant or user turn — typically an
//! async `dispatch_thread` callback) render as their own standalone
//! row instead, so chronology stays intact. Both kinds default to
//! collapsed.

use egui::{Color32, RichText};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use whisper_agent_protocol::{Attachment, ImageSource, Usage};

use super::{DiffPayload, DisplayItem, FusedToolResult};

/// Side-channel actions a rendered chat row can emit in the frame
/// it's drawn. The caller pairs events with the current `thread_id`
/// (which `render_item` doesn't see) and turns them into wire
/// messages or modal-state changes. Kept as an enum so future
/// hover-revealed row controls (retry, copy, drop) slot in without
/// widening `render_item`'s signature again.
#[derive(Debug)]
pub(super) enum ChatItemEvent {
    /// The "fork from here" button on a `User` row was clicked.
    /// `msg_index` is the absolute index into
    /// `Conversation.messages()` that the new thread should truncate
    /// at (exclusive). `seed_text` is the original user-message body
    /// — the caller pre-fills it as the new thread's draft so the
    /// user lands on the compose box ready to edit rather than
    /// retyping from scratch.
    ForkRequested { msg_index: usize, seed_text: String },
}

const GUTTER_WIDTH: f32 = 3.0;

const COLOR_USER: Color32 = Color32::from_rgb(120, 180, 240);
const COLOR_AGENT: Color32 = Color32::from_rgb(120, 200, 140);
const COLOR_REASONING: Color32 = Color32::from_rgb(170, 150, 200);
const COLOR_TOOL: Color32 = Color32::from_rgb(220, 180, 100);
const COLOR_ERROR: Color32 = Color32::from_rgb(220, 120, 120);
const COLOR_NEUTRAL: Color32 = Color32::from_gray(140);
const COLOR_SETUP: Color32 = Color32::from_gray(110);

fn item_palette(item: &DisplayItem) -> (Color32, Color32) {
    // (gutter color, frame fill)
    match item {
        DisplayItem::User { .. } => (
            COLOR_USER,
            Color32::from_rgba_unmultiplied(120, 180, 240, 18),
        ),
        DisplayItem::AssistantText { .. } => (COLOR_AGENT, Color32::TRANSPARENT),
        DisplayItem::AssistantImage { .. } => (COLOR_AGENT, Color32::TRANSPARENT),
        DisplayItem::Reasoning { .. } => (COLOR_REASONING, Color32::TRANSPARENT),
        DisplayItem::ToolCall {
            result: Some(FusedToolResult { is_error: true, .. }),
            ..
        } => (COLOR_ERROR, Color32::TRANSPARENT),
        DisplayItem::ToolCall { .. } => (COLOR_TOOL, Color32::TRANSPARENT),
        DisplayItem::ToolCallStreaming { .. } => (COLOR_TOOL, Color32::TRANSPARENT),
        DisplayItem::ToolResult { is_error: true, .. } => (COLOR_ERROR, Color32::TRANSPARENT),
        DisplayItem::ToolResult { .. } => (COLOR_TOOL, Color32::TRANSPARENT),
        DisplayItem::SystemNote { is_error: true, .. } => (
            COLOR_ERROR,
            Color32::from_rgba_unmultiplied(220, 120, 120, 18),
        ),
        DisplayItem::SystemNote { .. } => (COLOR_NEUTRAL, Color32::TRANSPARENT),
        // Setup-prefix rows are neutral-gray — they belong at the head
        // of the log as quiet "what the model saw" metadata, not
        // loud like user or assistant content.
        DisplayItem::SetupPrompt { .. } | DisplayItem::SetupTools { .. } => {
            (COLOR_SETUP, Color32::TRANSPARENT)
        }
        // No gutter for stats — it's a dim diagnostic footer, not a
        // semantic role. Rendered inline at low emphasis so it doesn't
        // compete with conversation content.
        DisplayItem::TurnStats { .. } => (Color32::TRANSPARENT, Color32::TRANSPARENT),
    }
}

pub(super) fn render_item(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    idx: usize,
    item: &DisplayItem,
    is_tail: bool,
    thread_streaming: bool,
) -> Option<ChatItemEvent> {
    let (gutter_color, fill) = item_palette(item);
    let frame = egui::Frame::default()
        .fill(fill)
        .inner_margin(egui::Margin {
            left: 12,
            right: 8,
            top: 4,
            bottom: 4,
        });
    let mut event: Option<ChatItemEvent> = None;
    // Hover state lags by one frame: we need to know "is this row
    // hovered" before drawing the button, but the row's final rect
    // isn't known until after. egui memory stashes last frame's
    // answer; the one-frame lag is imperceptible.
    let row_hover_id = match item {
        DisplayItem::User { msg_index, .. } => {
            Some(ui.make_persistent_id(("chat-row-hover", "user", *msg_index)))
        }
        DisplayItem::AssistantText { .. } => {
            Some(ui.make_persistent_id(("chat-row-hover", "assistant", idx)))
        }
        _ => None,
    };
    let hovered_prev_frame = row_hover_id
        .map(|id| {
            ui.ctx()
                .memory(|m| m.data.get_temp::<bool>(id).unwrap_or(false))
        })
        .unwrap_or(false);
    let resp = frame.show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        match item {
            DisplayItem::User {
                text,
                msg_index,
                attachments,
            } => {
                if render_user(ui, cache, text, attachments, *msg_index, hovered_prev_frame) {
                    event = Some(ChatItemEvent::ForkRequested {
                        msg_index: *msg_index,
                        seed_text: text.clone(),
                    });
                }
            }
            DisplayItem::AssistantText { text } => render_assistant_text(ui, cache, text),
            DisplayItem::AssistantImage { source } => render_assistant_image(ui, idx, source),
            DisplayItem::Reasoning { text } => {
                // Reasoning blocks are default-collapsed, so streaming
                // deltas are invisible past the one-line preview header
                // — the spinner is the operator's only signal that the
                // model is still emitting CoT. Only the tail Reasoning
                // gets it; an earlier Reasoning the turn moved past
                // isn't streaming anymore.
                render_reasoning(ui, cache, text, is_tail && thread_streaming)
            }
            DisplayItem::ToolCall {
                tool_use_id,
                name,
                summary,
                args_pretty,
                diff,
                streaming_output,
                result,
            } => render_tool_call(
                ui,
                tool_use_id,
                name,
                summary,
                args_pretty.as_deref(),
                diff.as_ref(),
                streaming_output,
                result.as_ref(),
                // A ToolCall with no fused result while the thread is
                // Working is still in flight — applies to every running
                // call in a parallel-tool batch, not just the tail one.
                result.is_none() && thread_streaming,
            ),
            DisplayItem::ToolCallStreaming {
                tool_use_id,
                name,
                args_chars,
            } => render_tool_call_streaming(ui, tool_use_id, name, *args_chars),
            DisplayItem::ToolResult {
                tool_use_id,
                name,
                text,
                is_error,
                attachments,
            } => render_tool_result(ui, tool_use_id, name, text, *is_error, attachments),
            DisplayItem::SystemNote { text, is_error } => render_system_note(ui, text, *is_error),
            DisplayItem::SetupPrompt { text } => render_setup_prompt(ui, text),
            DisplayItem::SetupTools { count, text } => render_setup_tools(ui, *count, text),
            DisplayItem::TurnStats { usage } => render_turn_stats(ui, usage),
        }
    });
    // Paint the gutter into the reserved 3px on the left side of the
    // frame's inner margin. Done after .show() returns so the painted
    // rect lands on top of the (subtle) frame fill — fine because the
    // gutter sits inside the inner margin where there's no content.
    let r = resp.response.rect;
    let gutter_rect = egui::Rect::from_min_max(r.min, egui::pos2(r.min.x + GUTTER_WIDTH, r.max.y));
    ui.painter().rect_filled(gutter_rect, 0.0, gutter_color);

    // AssistantText has no header to hang controls off of, so the
    // hover-revealed copy button is overlaid at top-right via `ui.put`.
    // This avoids a layout shift when the button appears (vs. reserving
    // a header row that's empty when not hovered).
    if hovered_prev_frame && let DisplayItem::AssistantText { text } = item {
        overlay_copy_button(ui, r, text, ("assistant-copy", idx));
    }

    // Persist hover for next frame; repaint on the edge so the
    // button transitions don't wait for an unrelated event.
    if let Some(id) = row_hover_id {
        let hovered_now = resp.response.contains_pointer();
        if hovered_now != hovered_prev_frame {
            ui.ctx().memory_mut(|m| m.data.insert_temp(id, hovered_now));
            ui.ctx().request_repaint();
        }
    }

    event
}

/// Draw a small "copy" button anchored to the top-right of `frame_rect`
/// (clamped to the scroll viewport's clip rect so a too-wide row can't
/// push it off-screen) that copies `text` to the clipboard on click.
/// `id_source` scopes the inner widget id so multiple overlay buttons
/// (one per row) don't collide on a shared call-site id.
fn overlay_copy_button(
    ui: &mut egui::Ui,
    frame_rect: egui::Rect,
    text: &str,
    id_source: (&'static str, usize),
) {
    let clip_right = ui.clip_rect().right();
    let right = clip_right.min(frame_rect.right()) - 8.0;
    let top = frame_rect.top() + 4.0;
    let size = egui::vec2(40.0, ui.spacing().interact_size.y);
    let btn_rect = egui::Rect::from_min_size(egui::pos2(right - size.x, top), size);
    let btn = egui::Button::new(RichText::new("copy").color(Color32::from_gray(160)).small())
        .frame(false);
    let mut clicked = false;
    ui.push_id(id_source, |ui| {
        if ui
            .put(btn_rect, btn)
            .on_hover_text("Copy this turn's text to clipboard")
            .clicked()
        {
            clicked = true;
        }
    });
    if clicked {
        ui.ctx().copy_text(text.to_string());
    }
}

/// Returns `true` when the user clicked the hover-reveal "fork" button on
/// this row. `row_hovered` controls whether the button is drawn — when
/// false we don't reserve space for it, so non-hovered rows stay visually
/// quiet.
fn render_user(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    text: &str,
    attachments: &[Attachment],
    msg_index: usize,
    row_hovered: bool,
) -> bool {
    // Role label sits above the body rather than inline so a multi-line
    // markdown body (code block, list, blockquote) doesn't wrap awkwardly
    // around the "USER" chip the way horizontal_wrapped would force.
    let mut fork_clicked = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new("USER").color(COLOR_USER).strong().small());
        if row_hovered {
            // Anchor the chips to the scroll viewport's right edge, not
            // the frame's. A non-wrapping markdown child (wide code
            // block, long URL) pushes the frame wider than the viewport
            // so a plain right-to-left layout lands the buttons in the
            // clipped region off-screen.
            let clip_right = ui.clip_rect().right();
            let cursor_x = ui.cursor().min.x;
            let budget = (clip_right - cursor_x).max(0.0);
            if budget > 0.0 {
                let height = ui.spacing().interact_size.y;
                ui.allocate_ui_with_layout(
                    egui::vec2(budget, height),
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        let fork_btn = egui::Button::new(
                            RichText::new("⑂ fork")
                                .color(Color32::from_gray(160))
                                .small(),
                        )
                        .frame(false);
                        if ui
                            .add(fork_btn)
                            .on_hover_text("Fork a new thread starting from this message")
                            .clicked()
                        {
                            fork_clicked = true;
                        }
                        let copy_btn = egui::Button::new(
                            RichText::new("copy").color(Color32::from_gray(160)).small(),
                        )
                        .frame(false);
                        if ui
                            .add(copy_btn)
                            .on_hover_text("Copy this message's text to clipboard")
                            .clicked()
                        {
                            ui.ctx().copy_text(text.to_string());
                        }
                    },
                );
            }
        }
    });
    if !text.is_empty() {
        render_markdown(ui, cache, ("user", text), text);
    }
    render_user_attachments(ui, attachments, msg_index);
    fork_clicked
}

/// Inline thumbnail strip for the image attachments that travelled
/// with a user message. Delegates to the shared
/// [`render_image_strip`] with a `history-{msg}` URI namespace so
/// per-message images stay cache-stable.
fn render_user_attachments(ui: &mut egui::Ui, attachments: &[Attachment], msg_index: usize) {
    if attachments.is_empty() {
        return;
    }
    let sources: Vec<&ImageSource> = attachments
        .iter()
        .map(|Attachment::Image { source }| source)
        .collect();
    render_image_strip(ui, &sources, &format!("history-{msg_index}"));
}

/// Inline thumbnail strip for a slice of image sources. Bytes sources
/// render through egui's loader via a content-addressed
/// `bytes://image-{hash}` URI so distinct images never share a cache
/// slot and identical images dedupe naturally. URL sources render as
/// a labelled placeholder (we don't fetch client-side).
///
/// Why content-addressed: egui's `DefaultBytesLoader::insert` uses
/// `entry().or_insert_with_key(...)` — first-write-wins. Reusing a
/// URI with new bytes is silently a no-op, so any position-based key
/// (e.g. `assistant-{display_item_idx}`) breaks as soon as two
/// different images land on the same key across thread switches,
/// snapshot rebuilds, or re-renders. Hashing the bytes guarantees
/// the URI tracks the content, so `include_bytes` is always either
/// a fresh insert or a stable identity.
///
/// `uri_prefix` is kept on the signature for backward-compatible
/// debug logging only — the URI's identity comes from the hash. The
/// trailing index `i` likewise just disambiguates within a strip if
/// callers ever pass two byte-identical images side by side.
///
/// Clicking a Bytes thumbnail stashes its URI into [`ENLARGED_IMAGE_KEY`]
/// in egui memory; the app's top-level `update` reads that slot and
/// shows the full image in a modal.
fn render_image_strip(ui: &mut egui::Ui, sources: &[&ImageSource], _uri_prefix: &str) {
    if sources.is_empty() {
        return;
    }
    ui.horizontal_wrapped(|ui| {
        for src in sources.iter() {
            match src {
                ImageSource::Bytes { data, .. } => {
                    let uri = bytes_image_uri(data);
                    ui.ctx().include_bytes(uri.clone(), data.clone());
                    let resp = ui.add(
                        egui::Image::new(&uri)
                            .max_size(egui::vec2(160.0, 160.0))
                            .fit_to_exact_size(egui::vec2(160.0, 160.0))
                            .sense(egui::Sense::click()),
                    );
                    if resp.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }
                    if resp.clicked() {
                        ui.ctx().memory_mut(|mem| {
                            mem.data
                                .insert_temp(egui::Id::new(ENLARGED_IMAGE_KEY), uri.clone());
                        });
                    }
                }
                ImageSource::Url { url } => {
                    ui.add(
                        egui::Label::new(
                            RichText::new(format!("🌐 {url}"))
                                .color(Color32::from_gray(160))
                                .small(),
                        )
                        .truncate(),
                    );
                }
            }
        }
    });
}

/// Memory key under which the chat-thumbnail strip stashes the URI of
/// the image the user just clicked. Read at the top of `ChatApp::ui`
/// to drive the lightbox modal. Lives in `egui::Memory::data` as a
/// temp value (cleared on app close) — the modal only needs it for
/// the duration of the dismiss.
pub(crate) const ENLARGED_IMAGE_KEY: &str = "whisper.enlarged_image_uri";

fn render_assistant_text(ui: &mut egui::Ui, cache: &mut CommonMarkCache, text: &str) {
    // No role label — the gutter color carries it. This is the bulk
    // of the conversation; we want it visually quiet.
    render_markdown(ui, cache, ("assistant", text), text);
}

/// Render a model-emitted image as a single-cell strip. Reuses
/// [`render_image_strip`] so the click-to-enlarge lightbox plumbing
/// works the same as compose attachments and tool-result images. The
/// `idx` is passed through as a debug-only namespace prefix —
/// `render_image_strip` derives the actual cache identity from a
/// content hash of the bytes (egui's bytes-loader is first-write-wins
/// per URI, so position-based keys break on multi-image conversations).
fn render_assistant_image(ui: &mut egui::Ui, idx: usize, source: &ImageSource) {
    render_image_strip(ui, &[source], &format!("assistant-{idx}"));
}

/// Hash an image's raw bytes into a stable `bytes://image-{hash}` URI.
/// Used as the cache key fed to `egui::Context::include_bytes` so
/// distinct images never share a slot (egui's default bytes loader
/// silently drops repeat inserts on the same URI). SipHash via
/// `DefaultHasher` is plenty here — we only need uniqueness across
/// the few hundred images a chat session might display, not
/// collision resistance against an attacker.
fn bytes_image_uri(data: &[u8]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut h);
    format!("bytes://image-{:016x}", h.finish())
}

fn render_reasoning(ui: &mut egui::Ui, cache: &mut CommonMarkCache, text: &str, streaming: bool) {
    let id = ui.make_persistent_id(("reasoning", text.as_ptr() as usize));
    let preview = text.lines().next().unwrap_or("").trim();
    let header = if preview.is_empty() {
        "(reasoning)".to_string()
    } else if preview.chars().count() > 80 {
        let mut h: String = preview.chars().take(80).collect();
        h.push('…');
        h
    } else {
        preview.to_string()
    };
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
        .show_header(ui, |ui| {
            if streaming {
                ui.add(egui::Spinner::new().size(12.0));
                ui.add_space(4.0);
            }
            ui.label(
                RichText::new("REASONING")
                    .color(COLOR_REASONING)
                    .strong()
                    .small(),
            );
            ui.add_space(8.0);
            ui.label(
                RichText::new(header)
                    .color(Color32::from_gray(140))
                    .italics(),
            );
            // Right-aligned copy button. Anchored to the scroll
            // viewport's right edge for the same reason as the
            // user-row controls — a wide preview line could otherwise
            // push the button off-screen.
            let clip_right = ui.clip_rect().right();
            let cursor_x = ui.cursor().min.x;
            let budget = (clip_right - cursor_x).max(0.0);
            if budget > 0.0 {
                let height = ui.spacing().interact_size.y;
                ui.allocate_ui_with_layout(
                    egui::vec2(budget, height),
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        let copy_btn = egui::Button::new(
                            RichText::new("copy").color(Color32::from_gray(160)).small(),
                        )
                        .frame(false);
                        if ui
                            .add(copy_btn)
                            .on_hover_text("Copy reasoning text to clipboard")
                            .clicked()
                        {
                            ui.ctx().copy_text(text.to_string());
                        }
                    },
                );
            }
        })
        .body(|ui| {
            render_markdown(ui, cache, ("reasoning", text), text);
        });
}

/// Render a chat-log body as CommonMark. `id_source` scopes the viewer's
/// internal widget ids so two items with identical text (e.g. two "ok"
/// assistant replies) don't collide on persistent state like collapsing
/// headers inside the rendered markdown.
fn render_markdown(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    id_source: (&'static str, &str),
    text: &str,
) {
    // Cap the render width to the scroll viewport so a non-wrapping
    // markdown child (fenced code block, long URL) can't push the
    // frame past the viewport's right edge — overflow there hides
    // the trailing text and kicks the row's hover controls (the
    // "⑂ fork" chip) off-screen. egui_commonmark derives code-block
    // wrap_width from `ui.available_width()`, so constraining it on
    // the scoped Ui propagates through `TextEdit::multiline` layout.
    let clip_right = ui.clip_rect().right();
    let cursor_x = ui.cursor().min.x;
    let budget = (clip_right - cursor_x).max(0.0);
    ui.scope(|ui| {
        if budget > 0.0 {
            ui.set_max_width(budget);
        }
        ui.push_id(id_source, |ui| {
            // Inline code (`foo`) uses `Visuals::code_bg_color` as its
            // rect fill via RichText::code(). egui's dark-mode default
            // (from_gray(64)) reads as a blocky highlight; swap in a
            // softer translucent tint that sits closer to GitHub's
            // dark-mode inline-code style. push_id creates a scoped
            // child Ui, so the override doesn't leak out of this block.
            ui.visuals_mut().code_bg_color = INLINE_CODE_BG;
            CommonMarkViewer::new().show(ui, cache, text);
        });
    });
}

/// Tint used behind `inline code`. Slightly-lighter-than-background
/// neutral at low alpha — reads as "subtly different" rather than the
/// blocky "highlighted span" effect of egui's dark-mode default (#404040).
/// Premultiplied: the unmultiplied equivalent is roughly rgba(150, 160, 180, 80).
const INLINE_CODE_BG: Color32 = Color32::from_rgba_premultiplied(47, 50, 56, 80);

fn render_system_note(ui: &mut egui::Ui, text: &str, is_error: bool) {
    let color = if is_error { COLOR_ERROR } else { COLOR_NEUTRAL };
    ui.label(RichText::new(text).color(color).italics());
}

/// Render the thread-prefix system-prompt entry. Default-collapsed
/// with a one-line preview, expands to show the full prompt in a
/// code-styled read-only block. Mirrors `Reasoning`'s treatment so
/// the head of the log reads quietly.
fn render_setup_prompt(ui: &mut egui::Ui, text: &str) {
    let id = ui.make_persistent_id(("setup-prompt", text.as_ptr() as usize));
    let preview = text.lines().next().unwrap_or("").trim();
    let header = if preview.is_empty() {
        "(empty)".to_string()
    } else if preview.chars().count() > 80 {
        let mut h: String = preview.chars().take(80).collect();
        h.push('…');
        h
    } else {
        preview.to_string()
    };
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
        .show_header(ui, |ui| {
            ui.label(RichText::new("SYSTEM").color(COLOR_SETUP).strong().small());
            ui.add_space(8.0);
            ui.label(
                RichText::new(header)
                    .color(Color32::from_gray(140))
                    .italics(),
            );
        })
        .body(|ui| {
            ui.label(
                RichText::new(text)
                    .color(Color32::from_gray(180))
                    .monospace()
                    .small(),
            );
        });
}

/// Render the thread-prefix tool-manifest entry. Collapsed header
/// shows the advertised count; expanded body lists each tool's
/// name, description, and input schema. Default-collapsed so a
/// thread with dozens of tools doesn't bury the conversation.
fn render_setup_tools(ui: &mut egui::Ui, count: usize, text: &str) {
    let id = ui.make_persistent_id(("setup-tools", text.as_ptr() as usize));
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
        .show_header(ui, |ui| {
            ui.label(RichText::new("TOOLS").color(COLOR_SETUP).strong().small());
            ui.add_space(8.0);
            ui.label(
                RichText::new(format!("{count} tool{}", if count == 1 { "" } else { "s" }))
                    .color(Color32::from_gray(140))
                    .italics(),
            );
        })
        .body(|ui| {
            ui.label(
                RichText::new(text)
                    .color(Color32::from_gray(180))
                    .monospace()
                    .small(),
            );
        });
}

/// Per-turn diagnostic footer. Format: `tokens in 12,345 · cached
/// 11,204 · created 0 · out 287`. "cached" counts are shown only
/// when non-zero so turns that bypass the cache (first turn, cache
/// miss) don't carry visually-empty columns. Small, dim, no gutter —
/// this is auxiliary information, not conversation.
fn render_turn_stats(ui: &mut egui::Ui, usage: &Usage) {
    let mut parts: Vec<String> = Vec::with_capacity(4);
    parts.push(format!("in {}", fmt_count(usage.input_tokens)));
    if usage.cache_read_input_tokens > 0 {
        parts.push(format!(
            "cached {}",
            fmt_count(usage.cache_read_input_tokens)
        ));
    }
    if usage.cache_creation_input_tokens > 0 {
        parts.push(format!(
            "created {}",
            fmt_count(usage.cache_creation_input_tokens)
        ));
    }
    parts.push(format!("out {}", fmt_count(usage.output_tokens)));
    let text = parts.join(" · ");
    ui.label(
        RichText::new(text)
            .color(Color32::from_gray(110))
            .monospace()
            .small(),
    );
}

/// Render `12345` as `12,345`. Cheap enough per-frame that a dedicated
/// helper is fine; keeps the format consistent across all four columns.
fn fmt_count(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let bytes = s.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

#[allow(clippy::too_many_arguments)]
/// Placeholder row while the model is streaming a tool-call's args
/// JSON. Shows name + spinner + char count. Replaced by
/// [`render_tool_call`] the moment the scheduler dispatches the call
/// (via `ThreadToolCallBegin`) and the reducer swaps the list entry
/// for a full [`DisplayItem::ToolCall`].
fn render_tool_call_streaming(ui: &mut egui::Ui, _tool_use_id: &str, name: &str, args_chars: u32) {
    ui.horizontal(|ui| {
        // Custom spinner already animates; no extra request_repaint
        // needed — egui repaints while any Spinner is on screen.
        ui.add(egui::Spinner::new().size(12.0));
        ui.add_space(4.0);
        ui.label(RichText::new(name).color(COLOR_TOOL).strong().monospace());
        ui.add_space(6.0);
        let suffix = if args_chars == 0 {
            "writing args…".to_string()
        } else {
            format!("writing args · {args_chars} chars")
        };
        ui.label(
            RichText::new(suffix)
                .color(Color32::from_gray(160))
                .monospace()
                .italics(),
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn render_tool_call(
    ui: &mut egui::Ui,
    tool_use_id: &str,
    name: &str,
    summary: &str,
    args_pretty: Option<&str>,
    diff: Option<&DiffPayload>,
    streaming_output: &str,
    result: Option<&FusedToolResult>,
    streaming: bool,
) {
    let id = ui.make_persistent_id(("tool", tool_use_id));
    // Default-collapsed. Chat stream reads as a sequence of one-line
    // `name summary [status]` headers; expand to see args / diff /
    // result. A fused result — the common case for sync calls and
    // async dispatch acks — is rendered in the body alongside the
    // args/diff, avoiding a second chat row.
    // Auto-expand while content is streaming so the user can see bash-
    // style output scroll without clicking through. Collapses back to
    // default-closed once the final result lands.
    let default_open = result.is_none() && !streaming_output.is_empty();
    let (chip_text, chip_color) = match result {
        None => ("running", Color32::from_rgb(200, 180, 60)),
        Some(FusedToolResult { is_error: true, .. }) => ("error", COLOR_ERROR),
        Some(_) => ("ok", Color32::from_rgb(140, 200, 140)),
    };
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, default_open)
        .show_header(ui, |ui| {
            ui.horizontal(|ui| {
                if streaming {
                    ui.add(egui::Spinner::new().size(12.0));
                    ui.add_space(4.0);
                }
                ui.label(RichText::new(name).color(COLOR_TOOL).strong().monospace());
                ui.add_space(6.0);
                ui.label(
                    RichText::new(summary)
                        .color(Color32::from_gray(180))
                        .monospace(),
                );
                // One-line result preview in the collapsed header so
                // the outcome is visible without expanding.
                if let Some(FusedToolResult { text, is_error, .. }) = result {
                    let preview = first_line_preview(text, 80);
                    if !preview.is_empty() {
                        ui.add_space(8.0);
                        let preview_color = if *is_error {
                            Color32::from_rgb(220, 140, 140)
                        } else {
                            Color32::from_gray(150)
                        };
                        ui.label(
                            RichText::new(preview)
                                .color(preview_color)
                                .monospace()
                                .small(),
                        );
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(chip_text).color(chip_color).small().strong());
                });
            });
        })
        .body(|ui| {
            if let Some(diff) = diff {
                render_diff(ui, diff);
            } else if let Some(args) = args_pretty {
                ui.label(
                    RichText::new(args)
                        .color(Color32::from_gray(170))
                        .monospace()
                        .small(),
                );
            }
            if let Some(FusedToolResult {
                text,
                is_error,
                attachments,
            }) = result
            {
                ui.add_space(6.0);
                ui.label(
                    RichText::new("result")
                        .color(Color32::from_gray(140))
                        .small()
                        .strong(),
                );
                let color = if *is_error {
                    Color32::from_rgb(220, 140, 140)
                } else {
                    Color32::from_gray(180)
                };
                if !text.is_empty() {
                    ui.label(RichText::new(text).color(color).monospace().small());
                }
                if !attachments.is_empty() {
                    ui.add_space(4.0);
                    let refs: Vec<&ImageSource> = attachments.iter().collect();
                    render_image_strip(ui, &refs, &format!("toolcall-{tool_use_id}"));
                }
            } else if !streaming_output.is_empty() {
                ui.add_space(6.0);
                ui.label(
                    RichText::new("output")
                        .color(Color32::from_gray(140))
                        .small()
                        .strong(),
                );
                ui.label(
                    RichText::new(streaming_output)
                        .color(Color32::from_gray(180))
                        .monospace()
                        .small(),
                );
            }
        });
}

/// Render a tool-result row. Always default-collapsed — the row
/// renders as a one-line `name preview [status]` header; clicking
/// expands to show the full result text and any image attachments.
/// Matches the treatment of tool calls and reasoning rows so the
/// chat stream stays quiet by default.
fn render_tool_result(
    ui: &mut egui::Ui,
    tool_use_id: &str,
    name: &str,
    text: &str,
    is_error: bool,
    attachments: &[ImageSource],
) {
    // Hash the result text into the persistent id so multiple
    // ToolResult rows that share a tool_use_id (sync ack + later
    // async callback for the same `dispatch_thread` call) get
    // distinct collapsing states — otherwise egui treats them as
    // the same widget and clicking one toggles the other.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    tool_use_id.hash(&mut hasher);
    text.hash(&mut hasher);
    let id = ui.make_persistent_id(("tool_result", hasher.finish()));
    let (chip_text, chip_color) = if is_error {
        ("error", COLOR_ERROR)
    } else {
        ("ok", Color32::from_rgb(140, 200, 140))
    };
    let label = if name.is_empty() { "tool_result" } else { name };
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
        .show_header(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(label).color(COLOR_TOOL).strong().monospace());
                ui.add_space(6.0);
                let preview = first_line_preview(text, 120);
                if !preview.is_empty() {
                    let preview_color = if is_error {
                        Color32::from_rgb(220, 140, 140)
                    } else {
                        Color32::from_gray(150)
                    };
                    ui.label(
                        RichText::new(preview)
                            .color(preview_color)
                            .monospace()
                            .small(),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(chip_text).color(chip_color).small().strong());
                });
            });
        })
        .body(|ui| {
            let color = if is_error {
                Color32::from_rgb(220, 140, 140)
            } else {
                Color32::from_gray(180)
            };
            if !text.is_empty() {
                ui.label(RichText::new(text).color(color).monospace().small());
            }
            if !attachments.is_empty() {
                ui.add_space(4.0);
                let refs: Vec<&ImageSource> = attachments.iter().collect();
                render_image_strip(ui, &refs, &format!("toolresult-{tool_use_id}"));
            }
        });
}

/// Pull a short preview of a tool-call result for the collapsed
/// header row. Grabs the first non-empty line and truncates at
/// `max_chars` code points (not bytes) so multibyte content (code
/// with non-ASCII identifiers, emoji in test output) doesn't break
/// the boundary. Returns an empty string if `body` has no visible
/// content.
fn first_line_preview(body: &str, max_chars: usize) -> String {
    let line = body.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let line = line.trim_end();
    if line.chars().count() > max_chars {
        let mut out: String = line.chars().take(max_chars).collect();
        out.push('…');
        out
    } else {
        line.to_string()
    }
}

fn render_diff(ui: &mut egui::Ui, diff: &DiffPayload) {
    use similar::{ChangeTag, TextDiff};
    let header = if diff.is_creation {
        format!("(new) {}", diff.path)
    } else {
        diff.path.clone()
    };
    ui.label(
        RichText::new(header)
            .color(Color32::from_gray(180))
            .monospace()
            .small()
            .strong(),
    );
    let text_diff = TextDiff::from_lines(&diff.old_text, &diff.new_text);
    let fg_eq = Color32::from_gray(150);
    let fg_del = Color32::from_rgb(230, 130, 130);
    let fg_add = Color32::from_rgb(150, 220, 170);
    egui::Frame::default()
        .fill(Color32::from_gray(22))
        .inner_margin(egui::Margin {
            left: 6,
            right: 6,
            top: 4,
            bottom: 4,
        })
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.style_mut().spacing.item_spacing.y = 0.0;
            for change in text_diff.iter_all_changes() {
                let (prefix, color) = match change.tag() {
                    ChangeTag::Equal => (' ', fg_eq),
                    ChangeTag::Delete => ('-', fg_del),
                    ChangeTag::Insert => ('+', fg_add),
                };
                let raw = change.value();
                let trimmed = raw.strip_suffix('\n').unwrap_or(raw);
                ui.label(
                    RichText::new(format!("{prefix}{trimmed}"))
                        .color(color)
                        .monospace()
                        .small(),
                );
            }
        });
}
