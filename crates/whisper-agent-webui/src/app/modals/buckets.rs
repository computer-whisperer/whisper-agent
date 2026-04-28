//! Knowledge-buckets management modal.
//!
//! Now hosts the full lifecycle (slice 8c): a create form, search
//! against ready buckets, per-row Build / Cancel / Delete (two-click
//! confirm) controls, and live build-progress display. Keeps everything
//! bucket-related in one modal so the navigation surface stays narrow.

use egui::{Color32, ComboBox, RichText, ScrollArea, TextEdit};
use whisper_agent_protocol::{
    BucketBuildPhase, BucketCreateInput, BucketSourceInput, BucketSummary, QueryHit, SlotStateLabel,
};

use super::super::{BucketsModalState, CreateBucketForm, QueryStatus, SourceKindChoice};

/// Side-channel actions a `render_buckets_modal` call can emit. The
/// caller mints correlation ids and dispatches the matching wire op.
pub(crate) enum BucketsEvent {
    RunQuery {
        bucket_id: String,
        query: String,
        top_k: u32,
    },
    /// Submit-create from the form. The caller validates / mints
    /// correlation; the form keeps `pending_correlation` so it shows
    /// a saving state until the wire response lands.
    CreateBucket {
        id: String,
        config: BucketCreateInput,
    },
    /// User confirmed delete (second click on the armed Delete button).
    DeleteBucket { id: String },
    /// "Build" pressed on a bucket row.
    StartBuild { id: String },
    /// "Cancel" pressed on a building row.
    CancelBuild { id: String },
    /// "Poll now" pressed on a tracked-bucket row — manually wake
    /// the feed worker rather than waiting for the next cadence
    /// tick. Only emitted for `source_kind = "tracked"` buckets.
    PollFeedNow { id: String },
    /// "Resync now" pressed on a tracked-bucket row — rebuild the
    /// bucket off the driver's current `latest_base()`. Server
    /// short-circuits if already at latest. Only emitted for
    /// `source_kind = "tracked"` buckets with no in-flight build.
    ResyncBucket { id: String },
}

/// Per-bucket progress snapshot the modal carries while a build is in
/// flight. Keyed by bucket id in the parent's
/// `BucketsModalState::build_progress`.
#[derive(Clone)]
pub(crate) struct BuildProgressView {
    pub(crate) phase: BucketBuildPhase,
    pub(crate) source_records: u64,
    pub(crate) chunks: u64,
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
        .default_height(620.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ctx, |ui| {
            // Top bar: + New bucket toggle.
            ui.horizontal(|ui| {
                let label = if modal.creating.is_some() {
                    "− Cancel new"
                } else {
                    "+ New bucket"
                };
                if ui.button(label).clicked() {
                    modal.creating = match modal.creating.take() {
                        Some(_) => None,
                        None => Some(CreateBucketForm::default()),
                    };
                }
            });

            if modal.creating.is_some() {
                ui.separator();
                render_create_section(ui, &mut modal, &mut events);
            }

            ui.separator();
            ui.add_space(4.0);

            // Auto-select the first ready bucket on first open so the
            // user can type and search immediately, but only if any
            // ready bucket exists yet.
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

            if buckets.is_empty() {
                ui.label(
                    RichText::new("No buckets yet — use \"+ New bucket\" above to create one.")
                        .small()
                        .color(Color32::from_gray(160)),
                );
            } else {
                ScrollArea::vertical().show(ui, |ui| {
                    for (i, b) in buckets.iter().enumerate() {
                        if i > 0 {
                            ui.separator();
                        }
                        render_bucket_row(ui, &mut modal, b, &mut events);
                    }
                });
            }
        });

    if open {
        *slot = Some(modal);
    }
    events
}

// ---------- create form ----------

fn render_create_section(
    ui: &mut egui::Ui,
    modal: &mut BucketsModalState,
    events: &mut Vec<BucketsEvent>,
) {
    let Some(form) = modal.creating.as_mut() else {
        return;
    };
    let saving = form.pending_correlation.is_some();

    ui.label(
        RichText::new("New bucket")
            .small()
            .color(Color32::from_gray(150)),
    );

    egui::Grid::new("create-bucket-grid")
        .num_columns(2)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            ui.label(RichText::new("id").small());
            ui.add_enabled(
                !saving,
                TextEdit::singleline(&mut form.id)
                    .desired_width(280.0)
                    .hint_text("filesystem-safe id, e.g. notes_2026"),
            );
            ui.end_row();

            ui.label(RichText::new("name").small());
            ui.add_enabled(
                !saving,
                TextEdit::singleline(&mut form.name)
                    .desired_width(360.0)
                    .hint_text("display name"),
            );
            ui.end_row();

            ui.label(RichText::new("description").small());
            ui.add_enabled(
                !saving,
                TextEdit::singleline(&mut form.description).desired_width(360.0),
            );
            ui.end_row();

            ui.label(RichText::new("embedder").small());
            ui.add_enabled(
                !saving,
                TextEdit::singleline(&mut form.embedder)
                    .desired_width(280.0)
                    .hint_text("provider name from [embedding_providers.X]"),
            );
            ui.end_row();

            ui.label(RichText::new("source kind").small());
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!saving, |ui| {
                    ui.selectable_value(&mut form.source_kind, SourceKindChoice::Stored, "stored");
                    ui.selectable_value(&mut form.source_kind, SourceKindChoice::Linked, "linked");
                    ui.selectable_value(
                        &mut form.source_kind,
                        SourceKindChoice::Managed,
                        "managed",
                    );
                });
            });
            ui.end_row();

            match form.source_kind {
                SourceKindChoice::Stored => {
                    ui.label(RichText::new("source.adapter").small());
                    ui.label(
                        RichText::new("mediawiki_xml")
                            .small()
                            .color(Color32::from_gray(160)),
                    );
                    ui.end_row();
                    ui.label(RichText::new("source.archive_path").small());
                    ui.add_enabled(
                        !saving,
                        TextEdit::singleline(&mut form.source_detail)
                            .desired_width(420.0)
                            .hint_text("/path/to/dump.xml.bz2 (must exist on the server)"),
                    );
                    ui.end_row();
                }
                SourceKindChoice::Linked => {
                    ui.label(RichText::new("source.adapter").small());
                    ui.label(
                        RichText::new("markdown_dir")
                            .small()
                            .color(Color32::from_gray(160)),
                    );
                    ui.end_row();
                    ui.label(RichText::new("source.path").small());
                    ui.add_enabled(
                        !saving,
                        TextEdit::singleline(&mut form.source_detail)
                            .desired_width(420.0)
                            .hint_text("/path/to/notes (server-side directory)"),
                    );
                    ui.end_row();
                }
                SourceKindChoice::Managed => {
                    ui.label(RichText::new("").small());
                    ui.label(
                        RichText::new("no external source — content authored via the API")
                            .small()
                            .color(Color32::from_gray(160)),
                    );
                    ui.end_row();
                }
            }

            ui.label(RichText::new("chunk tokens").small());
            ui.add_enabled(
                !saving,
                egui::DragValue::new(&mut form.chunk_tokens)
                    .range(50..=4096)
                    .speed(10.0),
            );
            ui.end_row();

            ui.label(RichText::new("overlap tokens").small());
            ui.add_enabled(
                !saving,
                egui::DragValue::new(&mut form.overlap_tokens)
                    .range(0..=512)
                    .speed(5.0),
            );
            ui.end_row();

            ui.label(RichText::new("paths").small());
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!saving, |ui| {
                    ui.checkbox(&mut form.dense_enabled, "dense");
                    ui.checkbox(&mut form.sparse_enabled, "sparse");
                });
            });
            ui.end_row();
        });

    if let Some(err) = form.error.as_deref() {
        ui.label(
            RichText::new(format!("error: {err}"))
                .small()
                .color(Color32::from_rgb(220, 120, 120)),
        );
    }

    ui.horizontal(|ui| {
        let create_clicked = ui
            .add_enabled(!saving, egui::Button::new("Create"))
            .clicked();
        if saving {
            ui.spinner();
            ui.label(
                RichText::new("creating…")
                    .small()
                    .color(Color32::from_gray(160)),
            );
        }

        if create_clicked && let Some(input) = build_create_input(form) {
            // Mark as pending; the parent stamps the correlation id
            // after dispatching the wire op.
            form.pending_correlation = Some(String::new());
            form.error = None;
            events.push(BucketsEvent::CreateBucket {
                id: form.id.trim().to_string(),
                config: input,
            });
        }
    });
}

/// Sanity-check the form locally; the server does authoritative
/// validation. Errors here are inline to keep the round-trip count
/// down for obviously-wrong forms.
fn build_create_input(form: &mut CreateBucketForm) -> Option<BucketCreateInput> {
    if form.id.trim().is_empty() {
        form.error = Some("id is required".into());
        return None;
    }
    if form.name.trim().is_empty() {
        form.error = Some("name is required".into());
        return None;
    }
    if form.embedder.trim().is_empty() {
        form.error = Some("embedder is required".into());
        return None;
    }
    let source = match form.source_kind {
        SourceKindChoice::Stored => {
            if form.source_detail.trim().is_empty() {
                form.error = Some("archive_path is required for kind=stored".into());
                return None;
            }
            BucketSourceInput::Stored {
                adapter: "mediawiki_xml".into(),
                archive_path: form.source_detail.trim().to_string(),
            }
        }
        SourceKindChoice::Linked => {
            if form.source_detail.trim().is_empty() {
                form.error = Some("path is required for kind=linked".into());
                return None;
            }
            BucketSourceInput::Linked {
                adapter: "markdown_dir".into(),
                path: form.source_detail.trim().to_string(),
            }
        }
        SourceKindChoice::Managed => BucketSourceInput::Managed {},
    };
    let description = if form.description.trim().is_empty() {
        None
    } else {
        Some(form.description.trim().to_string())
    };
    Some(BucketCreateInput {
        name: form.name.trim().to_string(),
        description,
        source,
        embedder: form.embedder.trim().to_string(),
        chunk_tokens: form.chunk_tokens,
        overlap_tokens: form.overlap_tokens,
        dense_enabled: form.dense_enabled,
        sparse_enabled: form.sparse_enabled,
    })
}

// ---------- query section (unchanged from slice 9, lifted as-is) ----------

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

    match &mut modal.query_status {
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
        QueryStatus::Results {
            query,
            hits,
            expanded,
        } => {
            ui.label(
                RichText::new(format!("results for {query:?} — {} hit(s)", hits.len()))
                    .small()
                    .color(Color32::from_gray(170)),
            );
            ui.add_space(2.0);
            ScrollArea::vertical()
                .id_salt("query-results")
                .max_height(320.0)
                .show(ui, |ui| {
                    for (i, h) in hits.iter().enumerate() {
                        render_hit(ui, i + 1, h, expanded);
                    }
                });
        }
    }
}

/// One result. Header row carries rank / via / scores; clicking it
/// toggles an expanded panel below with the full chunk text. Snippet
/// is shown collapsed; full text replaces it when expanded.
fn render_hit(
    ui: &mut egui::Ui,
    rank: usize,
    h: &QueryHit,
    expanded: &mut std::collections::HashSet<String>,
) {
    let is_open = expanded.contains(&h.chunk_id);

    // Two-line header: source title on top (the thing the eye scans
    // for "is this hit relevant"), metadata row below (chevron + via
    // + scores + chunk-id tail). Clicking the source title toggles
    // expand — large click target, matches what the eye reads first.
    let title_label = if h.source_id.is_empty() {
        format!("(no source id) — chunk {}", short_chunk_id(&h.chunk_id))
    } else {
        h.source_id.clone()
    };
    let title_color = if h.source_id.is_empty() {
        Color32::from_gray(150)
    } else {
        Color32::from_rgb(180, 200, 230)
    };
    let title_resp = ui
        .add(
            egui::Label::new(
                RichText::new(format!(
                    "{chev} {rank}. {title}",
                    chev = if is_open { "▾" } else { "▸" },
                    title = title_label,
                ))
                .strong()
                .color(title_color),
            )
            .sense(egui::Sense::click()),
        )
        .on_hover_cursor(egui::CursorIcon::PointingHand);

    ui.horizontal(|ui| {
        ui.add_space(16.0); // indent under the rank
        ui.label(via_chip(&h.source_path));
        ui.label(
            RichText::new(format!("rerank {:.3}", h.rerank_score))
                .small()
                .color(Color32::from_rgb(120, 200, 120)),
        );
        ui.label(
            RichText::new(format!("src {:.3}", h.source_score))
                .small()
                .color(Color32::from_gray(150)),
        );
        if let Some(loc) = h.source_locator.as_deref()
            && !loc.is_empty()
        {
            ui.label(
                RichText::new(format!("· {loc}"))
                    .small()
                    .color(Color32::from_gray(150)),
            );
        }
        ui.label(
            RichText::new(format!("· {}", short_chunk_id(&h.chunk_id)))
                .small()
                .monospace()
                .color(Color32::from_gray(130)),
        );
    });

    if title_resp.clicked() {
        if is_open {
            expanded.remove(&h.chunk_id);
        } else {
            expanded.insert(h.chunk_id.clone());
        }
    }

    if is_open {
        // Full chunk text in a faintly-bordered, scrollable monospace
        // panel. Word-wrapped via egui's default Label wrap (no
        // explicit wrap mode needed when label width fits the column).
        egui::Frame::group(ui.style())
            .stroke(egui::Stroke::new(1.0, Color32::from_gray(60)))
            .inner_margin(8.0)
            .show(ui, |ui| {
                ScrollArea::vertical()
                    .id_salt(format!("hit-body-{}", h.chunk_id))
                    .max_height(220.0)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(&h.chunk_text)
                                .monospace()
                                .color(Color32::from_gray(220)),
                        );
                    });
            });
    } else {
        let snippet = snippet_180(&h.chunk_text);
        ui.label(
            RichText::new(snippet)
                .small()
                .color(Color32::from_gray(200)),
        );
    }
    ui.add_space(6.0);
}

/// Color the via= label by source path so dense / sparse contributions
/// are visually distinguishable at a glance. Path values are lowercased
/// from the server-side `SearchPath` enum (`"dense"` / `"sparse"`);
/// anything else falls back to the neutral grey.
fn via_chip(path: &str) -> RichText {
    let (label, color) = match path {
        "dense" => ("via dense", Color32::from_rgb(140, 180, 220)),
        "sparse" => ("via sparse", Color32::from_rgb(220, 180, 140)),
        other => {
            return RichText::new(format!("via {other}"))
                .small()
                .color(Color32::from_gray(170));
        }
    };
    RichText::new(label).small().color(color)
}

fn snippet_180(s: &str) -> String {
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

// ---------- per-row render with Build / Cancel / Delete ----------

fn render_bucket_row(
    ui: &mut egui::Ui,
    modal: &mut BucketsModalState,
    b: &BucketSummary,
    events: &mut Vec<BucketsEvent>,
) {
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

    let in_flight_build = modal.build_progress.contains_key(&b.id);
    match &b.active_slot {
        None if !in_flight_build => {
            ui.label(
                RichText::new("no active slot — bucket has not been built yet")
                    .small()
                    .italics()
                    .color(Color32::from_gray(160)),
            );
        }
        None => {} // building, render below
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

    if let Some(progress) = modal.build_progress.get(&b.id) {
        render_build_progress(ui, progress);
    }

    if let Some(err) = modal.build_errors.get(&b.id) {
        ui.label(
            RichText::new(format!("last build error: {err}"))
                .small()
                .color(Color32::from_rgb(220, 120, 120)),
        );
    }

    render_row_actions(ui, modal, b, in_flight_build, events);
    ui.add_space(2.0);
}

fn render_build_progress(ui: &mut egui::Ui, p: &BuildProgressView) {
    let phase = match p.phase {
        BucketBuildPhase::Downloading => "downloading",
        BucketBuildPhase::Planning => "planning",
        BucketBuildPhase::Indexing => "indexing",
        BucketBuildPhase::BuildingDense => "building HNSW",
        BucketBuildPhase::Finalizing => "finalizing",
    };
    ui.horizontal_wrapped(|ui| {
        ui.spinner();
        ui.label(
            RichText::new(format!(
                "{phase} · {} pages · {} chunks",
                format_count(p.source_records),
                format_count(p.chunks),
            ))
            .small()
            .color(Color32::from_rgb(180, 180, 100)),
        );
    });
}

fn render_row_actions(
    ui: &mut egui::Ui,
    modal: &mut BucketsModalState,
    b: &BucketSummary,
    in_flight_build: bool,
    events: &mut Vec<BucketsEvent>,
) {
    ui.horizontal(|ui| {
        if in_flight_build {
            // "Pause" rather than "Cancel" — the server preserves
            // partial slot state on disk (build.state log + chunks +
            // vectors), and a subsequent Build click picks up where
            // we left off rather than starting from scratch.
            if ui.button("Pause build").clicked() {
                events.push(BucketsEvent::CancelBuild { id: b.id.clone() });
            }
        } else if ui
            .add_enabled(
                !matches!(b.source_kind.as_str(), "managed"),
                egui::Button::new("Build"),
            )
            .clicked()
        {
            // The server detects an in-progress slot from a previous
            // pause/crash and resumes it; otherwise this starts a
            // fresh slot.
            events.push(BucketsEvent::StartBuild { id: b.id.clone() });
        }

        // "Poll now" — tracked buckets only, no in-flight build.
        // Wakes the per-bucket FeedWorker rather than waiting for
        // the daily cadence tick. Server's trigger channel is
        // bounded at 1 so multiple rapid clicks coalesce server-
        // side; the UI just fires-and-forgets.
        if !in_flight_build
            && b.source_kind.as_str() == "tracked"
            && ui
                .button("Poll now")
                .on_hover_text(
                    "Wake the feed worker to poll for new deltas immediately, \
                     instead of waiting for the next cadence tick.",
                )
                .clicked()
        {
            events.push(BucketsEvent::PollFeedNow { id: b.id.clone() });
        }

        // "Resync now" — tracked buckets only, no in-flight build.
        // Rebuilds the bucket off the driver's current `latest_base()`
        // (a months-fresh snapshot for Wikipedia). Server short-
        // circuits if the recorded base is already at latest, so
        // a stray click on an already-current bucket is cheap.
        if !in_flight_build
            && b.source_kind.as_str() == "tracked"
            && ui
                .button("Resync now")
                .on_hover_text(
                    "Rebuild this bucket off the driver's latest base snapshot. \
                     Multi-hour to multi-day for Wikipedia-scale buckets — see the \
                     bucket's expected build time before clicking.",
                )
                .clicked()
        {
            events.push(BucketsEvent::ResyncBucket { id: b.id.clone() });
        }

        // Two-click delete: first click arms; second click confirms.
        // Click on any other button (or anywhere outside the row)
        // wouldn't disarm — the simplest gating is "armed only sticks
        // for this same render pass + the next click on the row".
        let armed = modal.delete_armed.as_deref() == Some(b.id.as_str());
        let label = if armed { "Confirm delete" } else { "Delete" };
        let button = egui::Button::new(if armed {
            RichText::new(label).color(Color32::from_rgb(220, 120, 120))
        } else {
            RichText::new(label)
        });
        if ui.add(button).clicked() {
            if armed {
                modal.delete_armed = None;
                events.push(BucketsEvent::DeleteBucket { id: b.id.clone() });
            } else {
                modal.delete_armed = Some(b.id.clone());
            }
        }
        if armed && ui.small_button(RichText::new("(cancel)").small()).clicked() {
            modal.delete_armed = None;
        }
    });
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
