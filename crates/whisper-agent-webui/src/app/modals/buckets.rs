//! Knowledge-buckets management modal.
//!
//! Now hosts the full lifecycle (slice 8c): a create form, search
//! against ready buckets, per-row Build / Cancel / Delete (two-click
//! confirm) controls, and live build-progress display. Keeps everything
//! bucket-related in one modal so the navigation surface stays narrow.

use egui::{Color32, ComboBox, RichText, ScrollArea, TextEdit};
use whisper_agent_protocol::{
    BucketBuildPhase, BucketCreateInput, BucketSourceInput, BucketSummary, EmbeddingProviderInfo,
    PodSummary, QuantizationInput, QueryHit, SlotStateLabel, TrackedCadenceInput,
    TrackedDriverInput,
};

use super::super::{
    BucketsModalState, CreateBucketForm, QuantizationChoice, QueryStatus, SourceKindChoice,
    TrackedCadenceChoice, TrackedDriverChoice,
};

/// Side-channel actions a `render_buckets_modal` call can emit. The
/// caller mints correlation ids and dispatches the matching wire op.
///
/// Every variant carries `pod_id: Option<String>` so the dispatcher
/// can route to the correct sub-registry: `None` ⇒ server-scope,
/// `Some(pod)` ⇒ pod-scope. Row-action events read it from the
/// originating `BucketSummary.pod_id`; the create form reads it from
/// the form's pod-scope ComboBox.
pub(crate) enum BucketsEvent {
    RunQuery {
        bucket_id: String,
        pod_id: Option<String>,
        query: String,
        top_k: u32,
    },
    /// Submit-create from the form. The caller validates / mints
    /// correlation; the form keeps `pending_correlation` so it shows
    /// a saving state until the wire response lands.
    CreateBucket {
        id: String,
        pod_id: Option<String>,
        config: BucketCreateInput,
    },
    /// User confirmed delete (second click on the armed Delete button).
    DeleteBucket { id: String, pod_id: Option<String> },
    /// "Build" pressed on a bucket row.
    StartBuild { id: String, pod_id: Option<String> },
    /// "Cancel" pressed on a building row.
    CancelBuild { id: String, pod_id: Option<String> },
    /// "Poll now" pressed on a tracked-bucket row — manually wake
    /// the feed worker rather than waiting for the next cadence
    /// tick. Only emitted for `source_kind = "tracked"` buckets.
    PollFeedNow { id: String, pod_id: Option<String> },
    /// "Resync now" pressed on a tracked-bucket row — rebuild the
    /// bucket off the driver's current `latest_base()`. Server
    /// short-circuits if already at latest. Only emitted for
    /// `source_kind = "tracked"` buckets with no in-flight build.
    ResyncBucket { id: String, pod_id: Option<String> },
}

/// All cadence variants in dropdown order. Daily / Weekly / Monthly /
/// Quarterly / Manual matches the wire enum's declaration order; the
/// form's two cadence ComboBoxes (delta + resync) iterate this slice.
const TRACKED_CADENCE_CHOICES: [TrackedCadenceChoice; 5] = [
    TrackedCadenceChoice::Daily,
    TrackedCadenceChoice::Weekly,
    TrackedCadenceChoice::Monthly,
    TrackedCadenceChoice::Quarterly,
    TrackedCadenceChoice::Manual,
];

/// Driver-picker label. Pulled out so the driver list grows in one
/// place when a second tracked driver lands.
fn tracked_driver_label(driver: TrackedDriverChoice) -> &'static str {
    match driver {
        TrackedDriverChoice::Wikipedia => "wikipedia",
    }
}

/// Translate the form's UI cadence enum to the wire type. One-to-one
/// mapping; lives here rather than as a `From` impl because both
/// types are `Copy` enums and a small free fn keeps the form-side
/// dispatch readable.
fn cadence_to_wire(c: TrackedCadenceChoice) -> TrackedCadenceInput {
    match c {
        TrackedCadenceChoice::Daily => TrackedCadenceInput::Daily,
        TrackedCadenceChoice::Weekly => TrackedCadenceInput::Weekly,
        TrackedCadenceChoice::Monthly => TrackedCadenceInput::Monthly,
        TrackedCadenceChoice::Quarterly => TrackedCadenceInput::Quarterly,
        TrackedCadenceChoice::Manual => TrackedCadenceInput::Manual,
    }
}

/// All quantization variants in dropdown order. F32 first so it stays
/// the visible default when the user opens the form.
const QUANTIZATION_CHOICES: [QuantizationChoice; 3] = [
    QuantizationChoice::F32,
    QuantizationChoice::F16,
    QuantizationChoice::Int8,
];

fn quantization_label(q: QuantizationChoice) -> &'static str {
    match q {
        QuantizationChoice::F32 => "f32",
        QuantizationChoice::F16 => "f16",
        QuantizationChoice::Int8 => "int8",
    }
}

fn quantization_to_wire(q: QuantizationChoice) -> QuantizationInput {
    match q {
        QuantizationChoice::F32 => QuantizationInput::F32,
        QuantizationChoice::F16 => QuantizationInput::F16,
        QuantizationChoice::Int8 => QuantizationInput::Int8,
    }
}

/// Per-bucket progress snapshot the modal carries while a build is in
/// flight. Keyed by bucket id in the parent's
/// `BucketsModalState::build_progress`.
#[derive(Clone)]
pub(crate) struct BuildProgressView {
    pub(crate) phase: BucketBuildPhase,
    pub(crate) source_records: u64,
    pub(crate) chunks: u64,
    /// RFC3339 wall-clock dispatch time, as forwarded by the server's
    /// `BucketBuildStarted` / `BucketBuildProgress`. `None` for very
    /// old servers that pre-date the field; UI renders no elapsed
    /// stopwatch when missing.
    pub(crate) started_at: Option<String>,
}

pub(crate) fn render_buckets_modal(
    ctx: &egui::Context,
    slot: &mut Option<BucketsModalState>,
    buckets: &[BucketSummary],
    pods: &[PodSummary],
    embedding_providers: &[EmbeddingProviderInfo],
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
                render_create_section(ui, &mut modal, pods, embedding_providers, &mut events);
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
    pods: &[PodSummary],
    embedding_providers: &[EmbeddingProviderInfo],
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

            // Scope: server (`None`) vs. pod-scope. Pod-scope buckets
            // live under `<pods_root>/<pod_id>/buckets/<id>/` and are
            // private to that pod's threads (auto-allowed without a
            // `[allow.knowledge_buckets]` grant).
            ui.label(RichText::new("scope").small());
            ui.add_enabled_ui(!saving, |ui| {
                let selected_label = match form.pod_id.as_deref() {
                    None => "(server)".to_string(),
                    Some(pid) => pid.to_string(),
                };
                ComboBox::from_id_salt("create-bucket-scope")
                    .selected_text(selected_label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut form.pod_id, None, "(server)");
                        for p in pods {
                            ui.selectable_value(
                                &mut form.pod_id,
                                Some(p.pod_id.clone()),
                                &p.pod_id,
                            );
                        }
                    });
            });
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

            // Embedder — populated from `ListEmbeddingProviders`. The
            // combobox label shows `name (kind)` so users with multiple
            // TEI endpoints (e.g. small + large model) can distinguish
            // them. If the catalog hasn't arrived yet (or is empty),
            // fall back to a TextEdit so a freshly-spun-up server with
            // no providers configured still surfaces a typeable field
            // rather than an empty disabled control.
            ui.label(RichText::new("embedder").small());
            ui.add_enabled_ui(!saving, |ui| {
                if embedding_providers.is_empty() {
                    ui.add(
                        TextEdit::singleline(&mut form.embedder)
                            .desired_width(280.0)
                            .hint_text("no [embedding_providers.*] configured"),
                    );
                } else {
                    let selected_label = if form.embedder.is_empty() {
                        "(select a provider)".to_string()
                    } else {
                        match embedding_providers.iter().find(|p| p.name == form.embedder) {
                            Some(p) => format!("{} ({})", p.name, p.kind),
                            None => format!("{} (unknown)", form.embedder),
                        }
                    };
                    ComboBox::from_id_salt("create-bucket-embedder")
                        .selected_text(selected_label)
                        .show_ui(ui, |ui| {
                            for p in embedding_providers {
                                ui.selectable_value(
                                    &mut form.embedder,
                                    p.name.clone(),
                                    format!("{} ({})", p.name, p.kind),
                                );
                            }
                        });
                }
            });
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
                    ui.selectable_value(
                        &mut form.source_kind,
                        SourceKindChoice::Tracked,
                        "tracked",
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
                SourceKindChoice::Tracked => {
                    // Driver picker — one entry today (Wikipedia); when
                    // a second driver lands, add another arm to
                    // `TrackedDriverChoice` and the wire type and the
                    // dropdown picks it up automatically.
                    ui.label(RichText::new("source.driver").small());
                    ui.add_enabled_ui(!saving, |ui| {
                        ComboBox::from_id_salt("create-bucket-driver")
                            .selected_text(tracked_driver_label(form.tracked_driver))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut form.tracked_driver,
                                    TrackedDriverChoice::Wikipedia,
                                    "wikipedia",
                                );
                            });
                    });
                    ui.end_row();

                    match form.tracked_driver {
                        TrackedDriverChoice::Wikipedia => {
                            ui.label(RichText::new("source.language").small());
                            ui.add_enabled(
                                !saving,
                                TextEdit::singleline(&mut form.tracked_language)
                                    .desired_width(120.0)
                                    .hint_text("en, simple, de, …"),
                            );
                            ui.end_row();

                            ui.label(RichText::new("source.mirror").small());
                            ui.add_enabled(
                                !saving,
                                TextEdit::singleline(&mut form.tracked_mirror)
                                    .desired_width(360.0)
                                    .hint_text("(optional) https://dumps.wikimedia.org"),
                            );
                            ui.end_row();
                        }
                    }

                    ui.label(RichText::new("source.delta_cadence").small());
                    ui.add_enabled_ui(!saving, |ui| {
                        ComboBox::from_id_salt("create-bucket-delta-cadence")
                            .selected_text(form.tracked_delta_cadence.label())
                            .show_ui(ui, |ui| {
                                for c in TRACKED_CADENCE_CHOICES {
                                    ui.selectable_value(
                                        &mut form.tracked_delta_cadence,
                                        c,
                                        c.label(),
                                    );
                                }
                            });
                    });
                    ui.end_row();

                    ui.label(RichText::new("source.resync_cadence").small());
                    ui.add_enabled_ui(!saving, |ui| {
                        ComboBox::from_id_salt("create-bucket-resync-cadence")
                            .selected_text(form.tracked_resync_cadence.label())
                            .show_ui(ui, |ui| {
                                for c in TRACKED_CADENCE_CHOICES {
                                    ui.selectable_value(
                                        &mut form.tracked_resync_cadence,
                                        c,
                                        c.label(),
                                    );
                                }
                            });
                    });
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

            // Vectors-bin quantization. f32 = full precision (default,
            // largest disk + RAM), f16 = half precision (2× smaller, ~no
            // recall loss), int8 = quantized (4× smaller, slight recall
            // loss). Frozen into the slot at build time.
            ui.label(RichText::new("quantization").small());
            ui.add_enabled_ui(!saving, |ui| {
                ComboBox::from_id_salt("create-bucket-quantization")
                    .selected_text(quantization_label(form.quantization))
                    .show_ui(ui, |ui| {
                        for q in QUANTIZATION_CHOICES {
                            ui.selectable_value(&mut form.quantization, q, quantization_label(q));
                        }
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
                pod_id: form.pod_id.clone(),
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
        SourceKindChoice::Tracked => {
            let driver = match form.tracked_driver {
                TrackedDriverChoice::Wikipedia => {
                    if form.tracked_language.trim().is_empty() {
                        form.error =
                            Some("language is required for kind=tracked driver=wikipedia".into());
                        return None;
                    }
                    let mirror = form.tracked_mirror.trim();
                    let mirror = if mirror.is_empty() {
                        None
                    } else if !(mirror.starts_with("http://") || mirror.starts_with("https://")) {
                        form.error = Some("mirror must be an http(s):// URL when set".into());
                        return None;
                    } else {
                        Some(mirror.to_string())
                    };
                    TrackedDriverInput::Wikipedia {
                        language: form.tracked_language.trim().to_string(),
                        mirror,
                    }
                }
            };
            BucketSourceInput::Tracked {
                driver,
                delta_cadence: cadence_to_wire(form.tracked_delta_cadence),
                resync_cadence: cadence_to_wire(form.tracked_resync_cadence),
            }
        }
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
        quantization: Some(quantization_to_wire(form.quantization)),
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
                // Resolve the selected bucket back to its `BucketSummary`
                // so the wire op carries the correct scope. Pod-scope
                // buckets named the same as a server-scope bucket would
                // route to the wrong sub-registry without this — the
                // ready-list iterator above is the authoritative source
                // of which (id, pod_id) pair the user actually picked.
                let pod_id = ready
                    .iter()
                    .find(|b| b.id == bucket_id)
                    .and_then(|b| b.pod_id.clone());
                events.push(BucketsEvent::RunQuery {
                    bucket_id,
                    pod_id,
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

    let row_key = (b.pod_id.clone(), b.id.clone());
    let in_flight_build = modal.build_progress.contains_key(&row_key);
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

    if let Some(progress) = modal.build_progress.get(&row_key) {
        render_build_progress(ui, progress);
    }

    if let Some(err) = modal.build_errors.get(&row_key) {
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
    let elapsed_str = p
        .started_at
        .as_deref()
        .map(format_build_elapsed)
        .filter(|s| !s.is_empty());
    let body = match elapsed_str {
        Some(elapsed) => format!(
            "{phase} · {} pages · {} chunks · {elapsed}",
            format_count(p.source_records),
            format_count(p.chunks),
        ),
        None => format!(
            "{phase} · {} pages · {} chunks",
            format_count(p.source_records),
            format_count(p.chunks),
        ),
    };
    ui.horizontal_wrapped(|ui| {
        ui.spinner();
        ui.label(
            RichText::new(body)
                .small()
                .color(Color32::from_rgb(180, 180, 100)),
        );
    });
}

/// Format build elapsed time at a granularity that scales from
/// seconds (smoke runs) through days (enwiki). Multi-day builds want
/// `2d 14h`, not `60840:21`.
fn format_build_elapsed(started_at_rfc3339: &str) -> String {
    use chrono::{DateTime, Utc};
    let Ok(parsed) = DateTime::parse_from_rfc3339(started_at_rfc3339) else {
        return String::new();
    };
    let secs = (Utc::now() - parsed.with_timezone(&Utc)).num_seconds();
    if secs < 0 {
        // Clock skew between server and client — surface as 0s.
        return "0s elapsed".to_string();
    }
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let d = secs / 86400;
    let body = if d > 0 {
        format!("{d}d {h}h")
    } else if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    };
    format!("{body} elapsed")
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
                events.push(BucketsEvent::CancelBuild {
                    id: b.id.clone(),
                    pod_id: b.pod_id.clone(),
                });
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
            events.push(BucketsEvent::StartBuild {
                id: b.id.clone(),
                pod_id: b.pod_id.clone(),
            });
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
            events.push(BucketsEvent::PollFeedNow {
                id: b.id.clone(),
                pod_id: b.pod_id.clone(),
            });
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
            events.push(BucketsEvent::ResyncBucket {
                id: b.id.clone(),
                pod_id: b.pod_id.clone(),
            });
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
                events.push(BucketsEvent::DeleteBucket {
                    id: b.id.clone(),
                    pod_id: b.pod_id.clone(),
                });
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    #[test]
    fn elapsed_formatter_seconds() {
        let started = (Utc::now() - Duration::seconds(7)).to_rfc3339();
        let s = format_build_elapsed(&started);
        // Tolerance for the s/s+1 rounding boundary; we just want
        // "single-digit seconds, ends with `s elapsed`".
        assert!(s.ends_with("s elapsed"), "got `{s}`");
        let prefix: &str = s.split_whitespace().next().unwrap();
        let n: i64 = prefix.trim_end_matches('s').parse().unwrap();
        assert!((6..=9).contains(&n), "expected ~7s, got `{s}`");
    }

    #[test]
    fn elapsed_formatter_minutes_seconds() {
        let started = (Utc::now() - Duration::seconds(125)).to_rfc3339();
        let s = format_build_elapsed(&started);
        // 2m 5s ± 1s rounding. Just check shape.
        assert!(s.starts_with("2m") && s.ends_with("s elapsed"), "got `{s}`");
    }

    #[test]
    fn elapsed_formatter_hours() {
        let started = (Utc::now() - Duration::seconds(3 * 3600 + 240)).to_rfc3339();
        let s = format_build_elapsed(&started);
        assert!(s.starts_with("3h") && s.ends_with("m elapsed"), "got `{s}`");
    }

    #[test]
    fn elapsed_formatter_days() {
        let started = (Utc::now() - Duration::seconds(2 * 86400 + 5 * 3600)).to_rfc3339();
        let s = format_build_elapsed(&started);
        assert!(s.starts_with("2d") && s.ends_with("h elapsed"), "got `{s}`");
    }

    #[test]
    fn elapsed_formatter_unparseable_input_is_empty() {
        // Empty result lets the caller's `.filter(|s| !s.is_empty())`
        // suppress the elapsed segment for malformed input.
        assert_eq!(format_build_elapsed("not a date"), "");
    }
}
