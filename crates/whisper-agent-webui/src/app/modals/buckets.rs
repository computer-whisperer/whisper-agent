//! Knowledge-buckets management modal.
//!
//! Slice 8b — read-only list. Renders one row per bucket with id /
//! name / source kind / embedder provider / active-slot status. The
//! Create / Build / Delete actions land in slice 8c; the modal already
//! exists so the new buttons can grow into it without restructuring
//! the navigation again.

use egui::{Color32, RichText, ScrollArea};
use whisper_agent_protocol::{BucketSummary, SlotStateLabel};

use super::super::BucketsModalState;

/// Side-channel actions a `render_buckets_modal` call can emit. Empty
/// today; the slice 8c create/build/delete events grow here.
pub(crate) enum BucketsEvent {}

pub(crate) fn render_buckets_modal(
    ctx: &egui::Context,
    slot: &mut Option<BucketsModalState>,
    buckets: &[BucketSummary],
) -> Vec<BucketsEvent> {
    let events: Vec<BucketsEvent> = Vec::new();
    let Some(modal) = slot.take() else {
        return events;
    };
    let mut open = true;

    egui::Window::new("Knowledge buckets")
        .collapsible(false)
        .resizable(true)
        .default_width(640.0)
        .default_height(420.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            if buckets.is_empty() {
                ui.add_space(8.0);
                ui.label(
                    RichText::new(
                        "No buckets configured yet. Create one by adding a \
                         <buckets_root>/<id>/bucket.toml on the server.",
                    )
                    .color(Color32::from_gray(160)),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new("Inline create / build actions arrive in the next slice.")
                        .small()
                        .color(Color32::from_gray(140)),
                );
                return;
            }

            ScrollArea::vertical().show(ui, |ui| {
                for (i, b) in buckets.iter().enumerate() {
                    if i > 0 {
                        ui.separator();
                    }
                    render_bucket_row(ui, b);
                }
            });
        });

    if open {
        *slot = Some(modal);
    }
    events
}

fn render_bucket_row(ui: &mut egui::Ui, b: &BucketSummary) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(&b.name).strong());
        ui.label(
            RichText::new(format!("({})", b.id))
                .small()
                .color(Color32::from_gray(150)),
        );
    });

    if let Some(desc) = b.description.as_deref()
        && !desc.is_empty()
    {
        ui.label(RichText::new(desc).small().color(Color32::from_gray(180)));
    }

    ui.horizontal_wrapped(|ui| {
        meta_chip(ui, "scope", &b.scope);
        meta_chip(ui, "source", &b.source_kind);
        if let Some(detail) = b.source_detail.as_deref() {
            ui.label(RichText::new(detail).small().color(Color32::from_gray(160)));
        }
    });

    ui.horizontal_wrapped(|ui| {
        meta_chip(ui, "embedder", &b.embedder_provider);
        let dense = if b.dense_enabled {
            "dense ✓"
        } else {
            "dense ✗"
        };
        let sparse = if b.sparse_enabled {
            "sparse ✓"
        } else {
            "sparse ✗"
        };
        ui.label(RichText::new(dense).small().color(Color32::from_gray(180)));
        ui.label(RichText::new(sparse).small().color(Color32::from_gray(180)));
    });

    match &b.active_slot {
        None => {
            ui.label(
                RichText::new("no active slot — bucket has not been built yet")
                    .small()
                    .italics()
                    .color(Color32::from_gray(160)),
            );
        }
        Some(slot) => {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("slot").small().color(Color32::from_gray(150)));
                ui.label(RichText::new(short_slot(&slot.slot_id)).small().monospace());
                ui.label(state_chip(slot.state));
                ui.label(
                    RichText::new(format!(
                        "{}-d · {} chunks · {}",
                        slot.dimension,
                        format_count(slot.chunk_count),
                        format_bytes(slot.disk_size_bytes),
                    ))
                    .small()
                    .color(Color32::from_gray(180)),
                );
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new(format!("model: {}", slot.embedder_model))
                        .small()
                        .color(Color32::from_gray(170)),
                );
                if let Some(built) = slot.built_at.as_deref() {
                    ui.label(
                        RichText::new(format!("built: {built}"))
                            .small()
                            .color(Color32::from_gray(170)),
                    );
                }
            });
        }
    }
    ui.add_space(2.0);
}

fn meta_chip(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(
        RichText::new(format!("{label}: "))
            .small()
            .color(Color32::from_gray(150)),
    );
    ui.label(RichText::new(value).small());
}

fn state_chip(state: SlotStateLabel) -> RichText {
    let (text, color) = match state {
        SlotStateLabel::Planning => ("planning", Color32::from_rgb(180, 180, 100)),
        SlotStateLabel::Building => ("building", Color32::from_rgb(180, 180, 100)),
        SlotStateLabel::Ready => ("ready", Color32::from_rgb(120, 200, 120)),
        SlotStateLabel::Failed => ("failed", Color32::from_rgb(220, 120, 120)),
        SlotStateLabel::Archived => ("archived", Color32::from_gray(150)),
    };
    RichText::new(text).small().color(color)
}

fn short_slot(slot_id: &str) -> String {
    // 30-char slot id is unwieldy in a row; show the leading 8 chars
    // (the timestamp portion sorts and is enough to disambiguate).
    if slot_id.len() > 8 {
        format!("{}…", &slot_id[..8])
    } else {
        slot_id.to_string()
    }
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}
