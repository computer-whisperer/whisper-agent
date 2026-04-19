//! Conversation renderer — event-log layout.
//!
//! Each item draws as a full-width row with a 3px role-colored gutter on
//! the left edge. User messages get a faint background tint to mark
//! them as section breaks (they're rare relative to model output).
//! Agent text and reasoning have transparent backgrounds — the gutter
//! alone carries the role signal so the bulk of the conversation reads
//! as a quiet stream of model output, not a chat log.
//!
//! Tool calls and tool results each render as their own collapsible
//! row at the position they landed in the conversation. Tool-call
//! headers read `name summary`; tool-result headers read `name
//! preview [status]`. All tool calls collapse by default. Tool
//! results default to collapsed when they immediately follow their
//! call (dense, expected); they default to expanded when separated
//! from their call by an assistant or user turn (typically an async
//! `dispatch_thread` callback) so the operator sees the fresh
//! content without clicking.

use egui::{Color32, RichText};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};

use super::{DiffPayload, DisplayItem};

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
        DisplayItem::ToolCall { .. } => (COLOR_TOOL, Color32::TRANSPARENT),
        DisplayItem::ToolResult { is_error: true, .. } => (COLOR_ERROR, Color32::TRANSPARENT),
        DisplayItem::ToolResult { .. } => (COLOR_TOOL, Color32::TRANSPARENT),
        DisplayItem::SystemNote { is_error: true, .. } => (
            COLOR_ERROR,
            Color32::from_rgba_unmultiplied(220, 120, 120, 18),
        ),
        DisplayItem::SystemNote { .. } => (COLOR_NEUTRAL, Color32::TRANSPARENT),
    }
}

pub(super) fn render_item(ui: &mut egui::Ui, cache: &mut CommonMarkCache, item: &DisplayItem) {
    let (gutter_color, fill) = item_palette(item);
    let frame = egui::Frame::default()
        .fill(fill)
        .inner_margin(egui::Margin {
            left: 12,
            right: 8,
            top: 4,
            bottom: 4,
        });
    let resp = frame.show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        match item {
            DisplayItem::User { text } => render_user(ui, cache, text),
            DisplayItem::AssistantText { text } => render_assistant_text(ui, cache, text),
            DisplayItem::Reasoning { text } => render_reasoning(ui, cache, text),
            DisplayItem::ToolCall {
                tool_use_id,
                name,
                summary,
                args_pretty,
                diff,
            } => render_tool_call(
                ui,
                tool_use_id,
                name,
                summary,
                args_pretty.as_deref(),
                diff.as_ref(),
            ),
            DisplayItem::ToolResult {
                tool_use_id,
                name,
                text,
                is_error,
                default_open,
            } => render_tool_result(ui, tool_use_id, name, text, *is_error, *default_open),
            DisplayItem::SystemNote { text, is_error } => render_system_note(ui, text, *is_error),
        }
    });
    // Paint the gutter into the reserved 3px on the left side of the
    // frame's inner margin. Done after .show() returns so the painted
    // rect lands on top of the (subtle) frame fill — fine because the
    // gutter sits inside the inner margin where there's no content.
    let r = resp.response.rect;
    let gutter_rect = egui::Rect::from_min_max(r.min, egui::pos2(r.min.x + GUTTER_WIDTH, r.max.y));
    ui.painter().rect_filled(gutter_rect, 0.0, gutter_color);
}

fn render_user(ui: &mut egui::Ui, cache: &mut CommonMarkCache, text: &str) {
    // Role label sits above the body rather than inline so a multi-line
    // markdown body (code block, list, blockquote) doesn't wrap awkwardly
    // around the "USER" chip the way horizontal_wrapped would force.
    ui.label(RichText::new("USER").color(COLOR_USER).strong().small());
    render_markdown(ui, cache, ("user", text), text);
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

fn render_tool_call(
    ui: &mut egui::Ui,
    tool_use_id: &str,
    name: &str,
    summary: &str,
    args_pretty: Option<&str>,
    diff: Option<&DiffPayload>,
) {
    let id = ui.make_persistent_id(("tool", tool_use_id));
    // Default-collapsed: chat stream reads as a sequence of
    // `name summary` headers, expandable for args / diff. The
    // matching result lands as its own `DisplayItem::ToolResult`
    // row, not inside this body.
    let default_open = false;
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
        });
}

/// Render a tool-result row. Proximity to the originating tool call
/// drives the default-open state: results immediately following
/// their call default-collapsed (dense, expected, quiet); results
/// arriving after an intervening assistant/user turn default-
/// expanded so the operator sees the new content. Header shows
/// `name summary [status]`; expanded body shows the full result
/// text.
fn render_tool_result(
    ui: &mut egui::Ui,
    tool_use_id: &str,
    name: &str,
    text: &str,
    is_error: bool,
    default_open: bool,
) {
    let id = ui.make_persistent_id(("tool_result", tool_use_id));
    let (chip_text, chip_color) = if is_error {
        ("error", COLOR_ERROR)
    } else {
        ("ok", Color32::from_rgb(140, 200, 140))
    };
    let label = if name.is_empty() { "tool_result" } else { name };
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, default_open)
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
