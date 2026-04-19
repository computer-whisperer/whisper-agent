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
use whisper_agent_protocol::Usage;

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

fn item_palette(item: &DisplayItem) -> (Color32, Color32) {
    // (gutter color, frame fill)
    match item {
        DisplayItem::User { .. } => (
            COLOR_USER,
            Color32::from_rgba_unmultiplied(120, 180, 240, 18),
        ),
        DisplayItem::AssistantText { .. } => (COLOR_AGENT, Color32::TRANSPARENT),
        DisplayItem::Reasoning { .. } => (COLOR_REASONING, Color32::TRANSPARENT),
        DisplayItem::ToolCall {
            result: Some(FusedToolResult { is_error: true, .. }),
            ..
        } => (COLOR_ERROR, Color32::TRANSPARENT),
        DisplayItem::ToolCall { .. } => (COLOR_TOOL, Color32::TRANSPARENT),
        DisplayItem::ToolResult { is_error: true, .. } => (COLOR_ERROR, Color32::TRANSPARENT),
        DisplayItem::ToolResult { .. } => (COLOR_TOOL, Color32::TRANSPARENT),
        DisplayItem::SystemNote { is_error: true, .. } => (
            COLOR_ERROR,
            Color32::from_rgba_unmultiplied(220, 120, 120, 18),
        ),
        DisplayItem::SystemNote { .. } => (COLOR_NEUTRAL, Color32::TRANSPARENT),
        // No gutter for stats — it's a dim diagnostic footer, not a
        // semantic role. Rendered inline at low emphasis so it doesn't
        // compete with conversation content.
        DisplayItem::TurnStats { .. } => (Color32::TRANSPARENT, Color32::TRANSPARENT),
    }
}

pub(super) fn render_item(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    item: &DisplayItem,
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
    let row_hover_id = if let DisplayItem::User { msg_index, .. } = item {
        Some(ui.make_persistent_id(("chat-row-hover", "user", *msg_index)))
    } else {
        None
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
            DisplayItem::User { text, msg_index } => {
                if render_user(ui, cache, text, hovered_prev_frame) {
                    event = Some(ChatItemEvent::ForkRequested {
                        msg_index: *msg_index,
                        seed_text: text.clone(),
                    });
                }
            }
            DisplayItem::AssistantText { text } => render_assistant_text(ui, cache, text),
            DisplayItem::Reasoning { text } => render_reasoning(ui, cache, text),
            DisplayItem::ToolCall {
                tool_use_id,
                name,
                summary,
                args_pretty,
                diff,
                result,
            } => render_tool_call(
                ui,
                tool_use_id,
                name,
                summary,
                args_pretty.as_deref(),
                diff.as_ref(),
                result.as_ref(),
            ),
            DisplayItem::ToolResult {
                tool_use_id,
                name,
                text,
                is_error,
            } => render_tool_result(ui, tool_use_id, name, text, *is_error),
            DisplayItem::SystemNote { text, is_error } => render_system_note(ui, text, *is_error),
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

/// Returns `true` when the user clicked the hover-reveal "fork" button on
/// this row. `row_hovered` controls whether the button is drawn — when
/// false we don't reserve space for it, so non-hovered rows stay visually
/// quiet.
fn render_user(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    text: &str,
    row_hovered: bool,
) -> bool {
    // Role label sits above the body rather than inline so a multi-line
    // markdown body (code block, list, blockquote) doesn't wrap awkwardly
    // around the "USER" chip the way horizontal_wrapped would force.
    let mut fork_clicked = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new("USER").color(COLOR_USER).strong().small());
        if row_hovered {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let btn = egui::Button::new(
                    RichText::new("⑂ fork")
                        .color(Color32::from_gray(160))
                        .small(),
                )
                .frame(false);
                if ui
                    .add(btn)
                    .on_hover_text("Fork a new thread starting from this message")
                    .clicked()
                {
                    fork_clicked = true;
                }
            });
        }
    });
    render_markdown(ui, cache, ("user", text), text);
    fork_clicked
}

fn render_assistant_text(ui: &mut egui::Ui, cache: &mut CommonMarkCache, text: &str) {
    // No role label — the gutter color carries it. This is the bulk
    // of the conversation; we want it visually quiet.
    render_markdown(ui, cache, ("assistant", text), text);
}

fn render_reasoning(ui: &mut egui::Ui, cache: &mut CommonMarkCache, text: &str) {
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
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn render_tool_call(
    ui: &mut egui::Ui,
    tool_use_id: &str,
    name: &str,
    summary: &str,
    args_pretty: Option<&str>,
    diff: Option<&DiffPayload>,
    result: Option<&FusedToolResult>,
) {
    let id = ui.make_persistent_id(("tool", tool_use_id));
    // Default-collapsed. Chat stream reads as a sequence of one-line
    // `name summary [status]` headers; expand to see args / diff /
    // result. A fused result — the common case for sync calls and
    // async dispatch acks — is rendered in the body alongside the
    // args/diff, avoiding a second chat row.
    let default_open = false;
    let (chip_text, chip_color) = match result {
        None => ("running", Color32::from_rgb(200, 180, 60)),
        Some(FusedToolResult { is_error: true, .. }) => ("error", COLOR_ERROR),
        Some(_) => ("ok", Color32::from_rgb(140, 200, 140)),
    };
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, default_open)
        .show_header(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(name).color(COLOR_TOOL).strong().monospace());
                ui.add_space(6.0);
                ui.label(
                    RichText::new(summary)
                        .color(Color32::from_gray(180))
                        .monospace(),
                );
                // One-line result preview in the collapsed header so
                // the outcome is visible without expanding.
                if let Some(FusedToolResult { text, is_error }) = result {
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
            if let Some(FusedToolResult { text, is_error }) = result {
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
                ui.label(RichText::new(text).color(color).monospace().small());
            }
        });
}

/// Render a tool-result row. Always default-collapsed — the row
/// renders as a one-line `name preview [status]` header; clicking
/// expands to show the full result text. Matches the treatment of
/// tool calls and reasoning rows so the chat stream stays quiet
/// by default.
fn render_tool_result(
    ui: &mut egui::Ui,
    tool_use_id: &str,
    name: &str,
    text: &str,
    is_error: bool,
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
            ui.label(RichText::new(text).color(color).monospace().small());
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
