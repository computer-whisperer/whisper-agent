//! Knowledge-buckets management modal.
//!
//! Hosts two surfaces today: a read-only list of buckets in the
//! registry (slice 8b) and a query form against the selected bucket
//! (slice 9). The Create / Build / Delete actions for slice 8c will
//! grow into the same modal — keeping everything bucket-related in
//! one place avoids fanning out the navigation surface.

use egui::{Color32, ComboBox, RichText, ScrollArea, TextEdit};
use whisper_agent_protocol::{BucketSummary, QueryHit, SlotStateLabel};

use super::super::{BucketsModalState, QueryStatus};

/// Side-channel actions a `render_buckets_modal` call can emit.
/// `RunQuery` is the slice-9 addition; the slice-8c create/build/
/// delete events grow alongside it.
pub(crate) enum BucketsEvent {
    /// User clicked Search. Caller mints a correlation, stamps it on
    /// the modal, and dispatches `ClientToServer::QueryBuckets`.
    RunQuery {
        bucket_id: String,
        query: String,
        top_k: u32,
    },
}

pub(crate) fn render_buckets_modal(
    ctx: &egui::Context,
    slot: &mut Option<BucketsModalState>,
    buckets: &[BucketSummary],
) -> Vec<BucketsEvent> {
    let mut events: Vec<BucketsEvent> = Vec::new();
    let Some(mut modal) = slot.take() else {
        return events;
    };
    let mut open = true;

    egui::Window::new("Knowledge buckets")
        .collapsible(false)
        .resizable(true)
        .default_width(720.0)
        .default_height(560.0)
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

            // Auto-select the first ready bucket on first open so the
            // user can type and search immediately. Skipped if the
            // user has manually picked already.
            if modal.selected_bucket.is_none()
                && let Some(first_ready) = buckets.iter().find(|b| {
                    b.active_slot
                        .as_ref()
                        .is_some_and(|s| s.state == SlotStateLabel::Ready)
                })
            {
                modal.selected_bucket = Some(first_ready.id.clone());
            }

            render_query_section(ui, &mut modal, buckets, &mut events);
            ui.separator();
            ui.add_space(4.0);
            ui.label(
                RichText::new("Buckets")
                    .small()
                    .color(Color32::from_gray(150)),
            );
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

fn render_query_section(
    ui: &mut egui::Ui,
    modal: &mut BucketsModalState,
    buckets: &[BucketSummary],
    events: &mut Vec<BucketsEvent>,
) {
    ui.label(
        RichText::new("Search")
            .small()
            .color(Color32::from_gray(150)),
    );

    let ready: Vec<&BucketSummary> = buckets
        .iter()
        .filter(|b| {
            b.active_slot
                .as_ref()
                .is_some_and(|s| s.state == SlotStateLabel::Ready)
        })
        .collect();

    if ready.is_empty() {
        ui.label(
            RichText::new("No ready buckets to query — build one first.")
                .small()
                .color(Color32::from_gray(160)),
        );
        return;
    }

    ui.horizontal(|ui| {
        let selected_id = modal
            .selected_bucket
            .clone()
            .unwrap_or_else(|| ready[0].id.clone());
        ComboBox::from_id_salt("bucket-picker")
            .selected_text(&selected_id)
            .show_ui(ui, |ui| {
                for b in &ready {
                    if ui
                        .selectable_label(modal.selected_bucket.as_deref() == Some(&b.id), &b.id)
                        .clicked()
                    {
                        modal.selected_bucket = Some(b.id.clone());
                    }
                }
            });

        let busy = matches!(modal.query_status, QueryStatus::InFlight { .. });
        let response = ui.add(
            TextEdit::singleline(&mut modal.query_input)
                .hint_text("query — Enter to run")
                .desired_width(360.0),
        );
        let submit = !busy
            && (ui.add_enabled(!busy, egui::Button::new("Search")).clicked()
                || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))));
        if busy {
            ui.spinner();
        }
        if submit && let Some(bucket_id) = modal.selected_bucket.clone() {
            let q = modal.query_input.trim().to_string();
            if !q.is_empty() {
                events.push(BucketsEvent::RunQuery {
                    bucket_id,
                    query: q.clone(),
                    top_k: modal.top_k,
                });
                modal.query_status = QueryStatus::InFlight { query: q };
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label(
            RichText::new("top_k")
                .small()
                .color(Color32::from_gray(150)),
        );
        ui.add(
            egui::DragValue::new(&mut modal.top_k)
                .range(1..=50)
                .speed(1.0),
        );
    });

    match &modal.query_status {
        QueryStatus::Idle => {}
        QueryStatus::InFlight { query } => {
            ui.label(
                RichText::new(format!("running query: {query:?}"))
                    .small()
                    .italics()
                    .color(Color32::from_gray(160)),
            );
        }
        QueryStatus::Error { message } => {
            ui.label(
                RichText::new(format!("query error: {message}"))
                    .small()
                    .color(Color32::from_rgb(220, 120, 120)),
            );
        }
        QueryStatus::Results { query, hits } => {
            ui.label(
                RichText::new(format!("results for {query:?} — {} hit(s)", hits.len()))
                    .small()
                    .color(Color32::from_gray(170)),
            );
            ui.add_space(2.0);
            ScrollArea::vertical()
                .id_salt("query-results")
                .max_height(220.0)
                .show(ui, |ui| {
                    for (i, h) in hits.iter().enumerate() {
                        render_hit(ui, i + 1, h);
                    }
                });
        }
    }
}

fn render_hit(ui: &mut egui::Ui, rank: usize, h: &QueryHit) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{rank:>2}.")).small().monospace());
        ui.label(
            RichText::new(format!("[{}]", h.bucket_id))
                .small()
                .color(Color32::from_gray(150)),
        );
        ui.label(
            RichText::new(short_chunk_id(&h.chunk_id))
                .small()
                .monospace()
                .color(Color32::from_gray(150)),
        );
        ui.label(
            RichText::new(format!("via {}", h.source_path))
                .small()
                .color(Color32::from_gray(170)),
        );
        ui.label(
            RichText::new(format!("rerank={:.3}", h.rerank_score))
                .small()
                .color(Color32::from_rgb(120, 200, 120)),
        );
        ui.label(
            RichText::new(format!("src={:.3}", h.source_score))
                .small()
                .color(Color32::from_gray(170)),
        );
    });
    let snippet = snippet_120(&h.chunk_text);
    ui.label(
        RichText::new(snippet)
            .small()
            .color(Color32::from_gray(200)),
    );
    ui.add_space(4.0);
}

fn snippet_120(s: &str) -> String {
    let one_line: String = s.replace('\n', " ");
    one_line.chars().take(180).collect::<String>()
}

fn short_chunk_id(s: &str) -> String {
    if s.len() > 12 {
        format!("{}…", &s[..12])
    } else {
        s.to_string()
    }
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
