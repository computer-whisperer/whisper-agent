//! Dump aetna bundle artifacts (SVG, tree dump, draw_ops, lint, manifest)
//! for each known scene of [`ChatApp`]. CPU-only — no window, no GPU.
//!
//! Usage: `cargo run -p whisper-agent-aetna-ui --example dump_bundles`
//!
//! Output: `crates/whisper-agent-aetna-ui/out/<scene>.{svg,tree.txt,
//! draw_ops.txt,shader_manifest.txt,lint.txt}` — gitignored.
//!
//! The SVG is a layout-accurate fallback; the tree dump shows computed
//! rects + roles + source locations; lint findings call out raw colors,
//! duplicate ids, and overflowing text. Together they make layout
//! regressions obvious without spinning up a session, and give an LLM
//! author a textual handle on the rendered tree.
//!
//! Scenes are seeded by pushing synthetic [`InboundEvent`]s into the
//! same queue the live host shell uses, then driving [`UiEvent`]s
//! through `on_event` for click-driven state. Stage 2 deliberately
//! does not synthesize a `ThreadSnapshot` (would require fabricating
//! a full `ThreadConfig` + `Conversation` mock); the live dev-script
//! path validates the snapshot pane.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::rc::Rc;

use aetna_core::prelude::{Rect, render_bundle_themed, write_bundle};
use aetna_core::{App, BuildCx, UiEvent};
use whisper_agent_aetna_ui::{
    ChatApp, Inbound, InboundEvent, LoginApp, LoginInput, SendFn, SubmitFn,
};
use whisper_agent_protocol::{
    BackendSummary, CompactionConfig, ContentBlock, ContentCapabilities, Conversation, ImageMime,
    ImageSource, Message, ModelSummary, PodSummary, Role, ServerToClient, ThreadBindings,
    ThreadConfig, ThreadSnapshot, ThreadStateLabel, ThreadSummary, ToolResultContent, TurnLog,
    Usage, permission::Scope,
};

fn main() -> std::io::Result<()> {
    // Native window viewport — the bug, if any, shouldn't depend on
    // viewport size. Keeping it identical to the desktop binary's
    // window dimensions makes the SVG read like the live UI at idle.
    let viewport = Rect::new(0.0, 0.0, 1200.0, 800.0);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out");

    for scene in Scene::ALL {
        // Both `ChatApp` and `LoginApp` implement `App`. The dump
        // loop is generic over `Box<dyn App>` so a new App variant
        // (e.g. a future modal-host App) can be added without
        // teaching the renderer about it.
        let mut app: Box<dyn App> = build_app(scene);
        app.before_build();
        for click in scene.clicks() {
            app.on_event(UiEvent::synthetic_click(click));
        }
        let theme = app.theme();
        let cx = BuildCx::new(&theme);
        let mut tree = app.build(&cx);
        let bundle =
            render_bundle_themed(&mut tree, viewport, Some(env!("CARGO_PKG_NAME")), &theme);

        let name = scene.slug();
        let written = write_bundle(&bundle, &out_dir, name)?;
        for p in &written {
            println!("wrote {}", p.display());
        }
        if !bundle.lint.findings.is_empty() {
            eprintln!("\n[{name}] lint findings ({}):", bundle.lint.findings.len());
            eprint!("{}", bundle.lint.text());
        }
    }
    Ok(())
}

/// Each scene parameterizes [`ChatApp`] into a distinct visible state.
/// Add a variant + the matching `seed` / `slug` arms when a new state
/// becomes worth dumping.
#[derive(Clone, Copy)]
enum Scene {
    Connecting,
    Connected,
    Closed,
    Error,
    /// Connected, server has replied with pod + thread lists, no
    /// thread selected.
    PopulatedNoSelection,
    /// Same as `PopulatedNoSelection`, but with a thread clicked —
    /// snapshot hasn't arrived, so the right pane shows "loading…".
    LoadingThread,
    /// Thread clicked + snapshot pushed. Exercises the event-log row
    /// shape (gutter + content) plus the markdown-rendered assistant
    /// turn and the reasoning accordion (collapsed by default).
    ThreadWithMessages,
    /// A thread whose conversation includes a tool call fused with
    /// its result, plus a one-error tool call. Verifies the
    /// args-block + result-block layout in the tool-call accordion
    /// and the destructive gutter on errored calls.
    ThreadWithToolCall,
    /// New-thread compose form, backends + pods seeded but no
    /// pickers touched yet. Verifies the card / form / select_trigger
    /// composition in its idle state.
    NewThreadFormReady,
    /// New-thread compose form, backend dropdown open. Verifies the
    /// `select_menu` popover anchors below the trigger and shows the
    /// inherit row + per-backend rows.
    NewThreadFormBackendOpen,
    /// New-thread compose form with backend, model, and pod all
    /// picked. Validates the trigger label fallbacks and that
    /// picking a backend triggers a model-list fetch.
    NewThreadFormFilled,
    /// Subscribed thread mid-prefill — `ThreadPrefillProgress` has
    /// arrived but no text/reasoning delta has yet. Verifies the
    /// thin progress bar + token-count caption above the chat log.
    ThreadPrefilling,
    /// Subscribed thread with a hydrated draft. Verifies the
    /// per-thread `text_area` binding picks the draft buffer up
    /// from the snapshot's `draft` field.
    ThreadWithDraft,
    /// Subscribed thread whose snapshot includes a user-attached
    /// PNG image plus an assistant image-output turn. Exercises the
    /// decode path (PNG bytes → `aetna_core::Image`), the height
    /// cap, and both gutters (info for user-supplied, success for
    /// assistant-emitted).
    ThreadWithImages,
    /// Empty login form — `LoginApp` straight out of `new(None,
    /// None, ...)`. Verifies the card / form / text_input
    /// composition before any user input.
    LoginFormEmpty,
    /// Login form pre-filled with a server URL + token (the path
    /// the desktop binary takes when CLI args or saved config
    /// supplied creds but `--login` was passed).
    LoginFormPrefilled,
    /// Login form with a connection error — `set_error` was called
    /// after a failed `derive_ws_url`. Verifies the destructive
    /// alert renders below the form.
    LoginFormWithError,
}

impl Scene {
    const ALL: [Scene; 17] = [
        Scene::Connecting,
        Scene::Connected,
        Scene::Closed,
        Scene::Error,
        Scene::PopulatedNoSelection,
        Scene::LoadingThread,
        Scene::ThreadWithMessages,
        Scene::ThreadWithToolCall,
        Scene::NewThreadFormReady,
        Scene::NewThreadFormBackendOpen,
        Scene::NewThreadFormFilled,
        Scene::ThreadPrefilling,
        Scene::ThreadWithDraft,
        Scene::ThreadWithImages,
        Scene::LoginFormEmpty,
        Scene::LoginFormPrefilled,
        Scene::LoginFormWithError,
    ];

    fn slug(self) -> &'static str {
        match self {
            Scene::Connecting => "connecting",
            Scene::Connected => "connected",
            Scene::Closed => "closed",
            Scene::Error => "error",
            Scene::PopulatedNoSelection => "populated_no_selection",
            Scene::LoadingThread => "loading_thread",
            Scene::ThreadWithMessages => "thread_with_messages",
            Scene::ThreadWithToolCall => "thread_with_tool_call",
            Scene::NewThreadFormReady => "new_thread_form_ready",
            Scene::NewThreadFormBackendOpen => "new_thread_form_backend_open",
            Scene::NewThreadFormFilled => "new_thread_form_filled",
            Scene::ThreadPrefilling => "thread_prefilling",
            Scene::ThreadWithDraft => "thread_with_draft",
            Scene::ThreadWithImages => "thread_with_images",
            Scene::LoginFormEmpty => "login_form_empty",
            Scene::LoginFormPrefilled => "login_form_prefilled",
            Scene::LoginFormWithError => "login_form_with_error",
        }
    }

    /// Synthetic clicks dispatched after the wire seed. Threaded
    /// through `App::on_event`, so they exercise the same routing the
    /// live UI does.
    fn clicks(self) -> Vec<&'static str> {
        match self {
            Scene::LoadingThread
            | Scene::ThreadWithMessages
            | Scene::ThreadWithToolCall
            | Scene::ThreadPrefilling
            | Scene::ThreadWithDraft
            | Scene::ThreadWithImages => vec!["thread:t-1"],
            // One toggle click on the backend trigger leaves the menu
            // open — render captures the popover.
            Scene::NewThreadFormBackendOpen => vec!["picker:backend"],
            // Pick a backend (which fires a no-op `ListModels`),
            // then a model id from the pre-seeded `ModelsList`,
            // then a pod. Each `option:` click goes through
            // `select::classify_event` -> `SelectAction::Pick`.
            Scene::NewThreadFormFilled => vec![
                "picker:backend",
                "picker:backend:option:anthropic-prod",
                "picker:model",
                "picker:model:option:claude-opus-4-7",
                "picker:pod",
                "picker:pod:option:scratch",
            ],
            _ => Vec::new(),
        }
    }
}

fn build_app(scene: Scene) -> Box<dyn App> {
    // Login scenes don't need an Inbound queue; build a fresh
    // `LoginApp` straight from the `LoginInput` signature.
    if let Some(login_app) = build_login_app(scene) {
        return Box::new(login_app);
    }

    let inbound: Inbound = Rc::new(std::cell::RefCell::new(VecDeque::new()));
    // No-op outbound: bundle dumping never sends to a server. Kept on
    // the same `SendFn` shape so [`ChatApp::new`] is the same call
    // shape the live binary uses.
    let send_fn: SendFn = Box::new(|_msg| {});
    let app = ChatApp::new(inbound.clone(), send_fn);

    let mut q = inbound.borrow_mut();
    match scene {
        Scene::Connecting => {} // default state
        Scene::Connected => {
            q.push_back(InboundEvent::ConnectionOpened);
        }
        Scene::Closed => {
            q.push_back(InboundEvent::ConnectionClosed {
                detail: "code 1000 normal".into(),
            });
        }
        Scene::Error => {
            q.push_back(InboundEvent::ConnectionError {
                detail: "DNS resolution failed".into(),
            });
        }
        Scene::PopulatedNoSelection
        | Scene::LoadingThread
        | Scene::ThreadWithMessages
        | Scene::ThreadWithToolCall
        | Scene::ThreadPrefilling
        | Scene::ThreadWithDraft
        | Scene::ThreadWithImages => {
            q.push_back(InboundEvent::ConnectionOpened);
            q.push_back(InboundEvent::Wire(ServerToClient::PodList {
                correlation_id: None,
                pods: mock_pods(),
                default_pod_id: "default".into(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::ThreadList {
                correlation_id: None,
                tasks: mock_threads(),
            }));
            if matches!(scene, Scene::ThreadWithMessages) {
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_snapshot(),
                }));
            }
            if matches!(scene, Scene::ThreadWithToolCall) {
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_tool_snapshot(),
                }));
            }
            if matches!(scene, Scene::ThreadPrefilling) {
                // Fresh snapshot (no assistant content yet) plus an
                // active prefill event — the indicator should render
                // until a delta arrives, which it won't here.
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_prefill_snapshot(),
                }));
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadPrefillProgress {
                    thread_id: "t-1".into(),
                    tokens_processed: 4231,
                    tokens_total: 8192,
                }));
            }
            if matches!(scene, Scene::ThreadWithDraft) {
                // Snapshot carries `draft` text — the per-thread
                // `text_area` should pick it up via the snapshot
                // hydration path (no separate ThreadDraftUpdated
                // needed).
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_draft_snapshot(),
                }));
            }
            if matches!(scene, Scene::ThreadWithImages) {
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_image_snapshot(),
                }));
            }
        }
        Scene::NewThreadFormReady
        | Scene::NewThreadFormBackendOpen
        | Scene::NewThreadFormFilled => {
            // Same baseline as `PopulatedNoSelection` — connection,
            // pods, and an empty thread list — plus a `BackendsList`
            // so the backend picker has options. The `Filled` scene
            // also pre-seeds a `ModelsList` for the backend the
            // synthetic clicks pick.
            q.push_back(InboundEvent::ConnectionOpened);
            q.push_back(InboundEvent::Wire(ServerToClient::PodList {
                correlation_id: None,
                pods: mock_pods(),
                default_pod_id: "default".into(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::ThreadList {
                correlation_id: None,
                tasks: Vec::new(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::BackendsList {
                correlation_id: None,
                backends: mock_backends(),
            }));
            if matches!(scene, Scene::NewThreadFormFilled) {
                q.push_back(InboundEvent::Wire(ServerToClient::ModelsList {
                    correlation_id: None,
                    backend: "anthropic-prod".into(),
                    models: mock_models(),
                }));
            }
        }
        // Login scenes route through `build_login_app` above and
        // never reach here.
        Scene::LoginFormEmpty | Scene::LoginFormPrefilled | Scene::LoginFormWithError => {
            unreachable!("login scenes handled by build_login_app");
        }
    }
    drop(q);
    Box::new(app)
}

/// Construct a `LoginApp` for the login-form scenes. Returns
/// `None` for non-login scenes so `build_app` can fall through to
/// the `ChatApp` builder.
fn build_login_app(scene: Scene) -> Option<LoginApp> {
    let submit: SubmitFn = Box::new(|_input: LoginInput| {
        // Bundle dumping never actually submits — the form just
        // needs a valid callback to live in its struct.
    });
    match scene {
        Scene::LoginFormEmpty => Some(LoginApp::new(None, None, submit)),
        Scene::LoginFormPrefilled => Some(LoginApp::new(
            Some("https://chat.example.internal:8443".into()),
            Some("ya29.example-token-redacted".into()),
            submit,
        )),
        Scene::LoginFormWithError => {
            let mut app = LoginApp::new(Some("not-a-url".into()), None, submit);
            app.set_error("invalid server URL: relative URL without a base");
            Some(app)
        }
        _ => None,
    }
}

fn mock_backends() -> Vec<BackendSummary> {
    vec![
        BackendSummary {
            name: "anthropic-prod".into(),
            kind: "anthropic".into(),
            default_model: Some("claude-opus-4-7".into()),
            auth_mode: Some("api_key".into()),
        },
        BackendSummary {
            name: "openai-team".into(),
            kind: "openai_chat".into(),
            default_model: Some("gpt-5".into()),
            auth_mode: Some("api_key".into()),
        },
        BackendSummary {
            name: "local-llama".into(),
            kind: "openai_chat".into(),
            default_model: None,
            auth_mode: None,
        },
    ]
}

fn mock_models() -> Vec<ModelSummary> {
    vec![
        ModelSummary {
            id: "claude-opus-4-7".into(),
            display_name: Some("Claude Opus 4.7".into()),
            context_window: Some(1_000_000),
            max_output_tokens: Some(8192),
            capabilities: ContentCapabilities::default(),
        },
        ModelSummary {
            id: "claude-sonnet-4-6".into(),
            display_name: Some("Claude Sonnet 4.6".into()),
            context_window: Some(200_000),
            max_output_tokens: Some(8192),
            capabilities: ContentCapabilities::default(),
        },
        ModelSummary {
            id: "claude-haiku-4-5".into(),
            display_name: Some("Claude Haiku 4.5".into()),
            context_window: Some(200_000),
            max_output_tokens: Some(8192),
            capabilities: ContentCapabilities::default(),
        },
    ]
}

fn mock_pods() -> Vec<PodSummary> {
    vec![
        PodSummary {
            pod_id: "default".into(),
            name: "default".into(),
            description: Some("the synthesized server-default pod".into()),
            created_at: "2026-05-01T00:00:00Z".into(),
            thread_count: 2,
            archived: false,
            behaviors_enabled: true,
        },
        PodSummary {
            pod_id: "scratch".into(),
            name: "scratch".into(),
            description: None,
            created_at: "2026-05-02T00:00:00Z".into(),
            thread_count: 1,
            archived: false,
            behaviors_enabled: true,
        },
    ]
}

fn mock_threads() -> Vec<ThreadSummary> {
    vec![
        ThreadSummary {
            thread_id: "t-1".into(),
            pod_id: "default".into(),
            title: Some("first conversation".into()),
            state: ThreadStateLabel::Idle,
            created_at: "2026-05-08T10:00:00Z".into(),
            last_active: "2026-05-08T11:00:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: None,
        },
        ThreadSummary {
            thread_id: "t-2".into(),
            pod_id: "default".into(),
            title: None,
            state: ThreadStateLabel::Working,
            created_at: "2026-05-08T09:00:00Z".into(),
            last_active: "2026-05-08T09:30:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: None,
        },
        ThreadSummary {
            thread_id: "t-3".into(),
            pod_id: "scratch".into(),
            title: Some("scratch experiment".into()),
            state: ThreadStateLabel::Idle,
            created_at: "2026-05-07T14:00:00Z".into(),
            last_active: "2026-05-07T14:15:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: None,
        },
    ]
}

/// A few-message snapshot for the `ThreadWithMessages` scene. Touches
/// every DisplayItem variant we render today so the bundle dump
/// regression-tests the event-log row + accordion + markdown paths in
/// one go.
fn mock_snapshot() -> ThreadSnapshot {
    let mut conv = Conversation::new();
    conv.push(Message::system_text(""));
    // Deliberately a long user prompt so a single message exceeds the
    // pane's width and we can confirm wrap is reaching the body. The
    // upstream row-intrinsic bug shows up only when content is long
    // enough that NoWrap measurement disagrees with Wrap measurement.
    conv.push(Message::user_text(
        "Can you summarize the design of the knowledge-bucket layer? I'm interested in \
         the on-disk shape, the build pipeline, the embedder rotation story, and how \
         pod-scoped buckets differ from server-scoped ones — keep each piece short \
         enough that I can scan it.",
    ));
    conv.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentBlock::Thinking {
                thinking: "User wants a high-level summary covering on-disk shape, \
                    build pipeline, embedder rotation, and pod- vs. server-scope. \
                    The design doc is at `docs/design_knowledge_db.md`. Hit the \
                    entry-point types, the slot directory layout, and one sentence \
                    on each of their other questions."
                    .into(),
                replay: None,
            },
            ContentBlock::Text {
                text: "The knowledge-bucket layer adds **dense + sparse retrieval** to \
                    whisper-agent. Each bucket is a directory under `<buckets_root>/` \
                    (server scope) or `<pods_root>/<pod>/buckets/` (pod scope) — same \
                    shape on disk, different visibility:\n\
                    \n\
                    - `bucket.toml` — config: embedder name, chunker settings, source \
                      adapter (stored / linked / managed).\n\
                    - `slots/` — append-only segments. Each build writes a fresh slot \
                      with its own HNSW + tantivy indexes, so older slots stay \
                      readable while a new build runs.\n\
                    - `source-cache/` — raw artifacts the chunker reads. Stored buckets \
                      copy bytes in here; linked buckets just point at an external path.\n\
                    \n\
                    Embedder rotation is driven by the slot model: a new embedder ⇒ \
                    a new slot, and the old slot stays serving until you delete it. \
                    Pod-scoped buckets are reachable only from their owning pod's \
                    threads; server-scoped buckets are visible to every thread.\n\
                    \n\
                    See [`docs/design_knowledge_db.md`](docs/design_knowledge_db.md) \
                    for the full design — the on-disk format, slot manifest layout, \
                    and the resolver logic all live there."
                    .into(),
            },
        ],
    });
    conv.push(Message::user_text("Thanks — that's exactly what I needed."));

    ThreadSnapshot {
        thread_id: "t-1".into(),
        pod_id: "default".into(),
        title: Some("first conversation".into()),
        config: ThreadConfig {
            model: "claude-opus-4-7".into(),
            max_tokens: 4096,
            max_turns: 8,
            compaction: CompactionConfig::default(),
        },
        bindings: ThreadBindings::default(),
        state: ThreadStateLabel::Idle,
        conversation: conv,
        total_usage: Usage::default(),
        turn_log: TurnLog::default(),
        draft: String::new(),
        created_at: "2026-05-08T10:00:00Z".into(),
        last_active: "2026-05-08T11:00:00Z".into(),
        failure: None,
        origin: None,
        continued_from: None,
        dispatched_by: None,
        scope: Scope::default(),
    }
}

/// Snapshot for the `ThreadPrefilling` scene — one user turn, no
/// assistant content yet, and the thread state is Working so the
/// pane reflects the in-flight turn. Pairs with a follow-on
/// `ThreadPrefillProgress` event seeded by the scene builder.
fn mock_prefill_snapshot() -> ThreadSnapshot {
    let mut conv = Conversation::new();
    conv.push(Message::system_text(""));
    conv.push(Message::user_text(
        "Summarize the entire knowledge-db design doc, then highlight the parts \
         most relevant to a fresh embedder rotation. Lots of context to chew on.",
    ));
    base_snapshot(conv, ThreadStateLabel::Working, String::new())
}

/// Snapshot for the `ThreadWithDraft` scene — exercises the
/// snapshot's `draft` hydration path. The chat log itself is the
/// same single user turn as the prefill mock; the only thing
/// distinguishing the bundle is the populated `text_area` body.
fn mock_draft_snapshot() -> ThreadSnapshot {
    let mut conv = Conversation::new();
    conv.push(Message::system_text(""));
    conv.push(Message::user_text(
        "Sketch the Stage 7 plan for image attachments before I start coding.",
    ));
    base_snapshot(
        conv,
        ThreadStateLabel::Idle,
        "I should also clarify whether the staging pipeline decodes \
         server-side or client-side before I send the request."
            .into(),
    )
}

/// Build a `ThreadSnapshot` from a conversation, state, and draft
/// — matches the shape of `mock_snapshot` / `mock_tool_snapshot`
/// without duplicating the boilerplate.
fn base_snapshot(
    conversation: Conversation,
    state: ThreadStateLabel,
    draft: String,
) -> ThreadSnapshot {
    ThreadSnapshot {
        thread_id: "t-1".into(),
        pod_id: "default".into(),
        title: Some("first conversation".into()),
        config: ThreadConfig {
            model: "claude-opus-4-7".into(),
            max_tokens: 4096,
            max_turns: 8,
            compaction: CompactionConfig::default(),
        },
        bindings: ThreadBindings::default(),
        state,
        conversation,
        total_usage: Usage::default(),
        turn_log: TurnLog::default(),
        draft,
        created_at: "2026-05-08T10:00:00Z".into(),
        last_active: "2026-05-08T11:00:00Z".into(),
        failure: None,
        origin: None,
        continued_from: None,
        dispatched_by: None,
        scope: Scope::default(),
    }
}

/// Snapshot for the `ThreadWithImages` scene. Two image rows: a
/// user-supplied screenshot (encoded as PNG bytes by
/// [`encode_checker_png`]) and a small URL-only image so the
/// fallback rendering path is exercised in the same dump. The
/// assistant turn then emits text + a model-generated image so
/// the "is_user=false" gutter case is also covered.
fn mock_image_snapshot() -> ThreadSnapshot {
    let mut conv = Conversation::new();
    conv.push(Message::system_text(""));
    conv.push(Message {
        role: Role::User,
        content: vec![
            ContentBlock::Text {
                text: "Here's a screenshot of the bug — also, here's a URL of the original \
                       on the wiki for reference:"
                    .into(),
            },
            ContentBlock::Image {
                source: ImageSource::Bytes {
                    media_type: ImageMime::Png,
                    data: encode_checker_png(160, 100),
                },
                replay: None,
            },
            ContentBlock::Image {
                source: ImageSource::Url {
                    url: "https://example.com/wiki/screenshots/bug-1234.png".into(),
                },
                replay: None,
            },
        ],
    });
    conv.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentBlock::Text {
                text: "I see the issue — the second column overflows. \
                       Generated a fix preview:"
                    .into(),
            },
            ContentBlock::Image {
                source: ImageSource::Bytes {
                    media_type: ImageMime::Png,
                    // Different dims than the user's image so the cap
                    // logic and per-row aspect handling are exercised
                    // independently.
                    data: encode_checker_png(240, 80),
                },
                replay: None,
            },
        ],
    });
    base_snapshot(conv, ThreadStateLabel::Idle, String::new())
}

/// Encode a small RGB checker pattern as PNG bytes. Used by
/// [`mock_image_snapshot`] to produce a real PNG payload that the
/// app's `image::load_from_memory` decoder can chew on, without
/// shipping a binary fixture. 8-pixel tile so the pattern is
/// visible at the cap'd display height.
fn encode_checker_png(width: u32, height: u32) -> Vec<u8> {
    use image::{ImageBuffer, ImageFormat, Rgba};
    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_fn(width, height, |x, y| {
        let cell = ((x / 8) + (y / 8)) % 2;
        if cell == 0 {
            Rgba([0xff, 0x00, 0xaa, 0xff])
        } else {
            Rgba([0x00, 0xaa, 0xff, 0xff])
        }
    });
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, ImageFormat::Png)
        .expect("encode checker PNG");
    buf.into_inner()
}

/// Snapshot exercising the tool-call accordion path. One assistant
/// turn issues a tool call, a tool-result message fuses with it on
/// snapshot walk; a second tool call ends in an error so the
/// destructive gutter + "✗" header treatment shows.
fn mock_tool_snapshot() -> ThreadSnapshot {
    let mut conv = Conversation::new();
    conv.push(Message::system_text(""));
    conv.push(Message::user_text(
        "List the files in /sandbox/buckets and grep them for 'TODO'.",
    ));
    conv.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentBlock::Text {
                text: "I'll list the bucket dir first, then grep each file.".into(),
            },
            ContentBlock::ToolUse {
                id: "tool-call-001".into(),
                name: "list_files".into(),
                input: serde_json::json!({
                    "path": "/sandbox/buckets",
                    "include_hidden": false
                }),
                replay: None,
            },
        ],
    });
    conv.push(Message {
        role: Role::ToolResult,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: "tool-call-001".into(),
            content: ToolResultContent::Text(
                "drwxr-xr-x rust-stdlib/\ndrwxr-xr-x cargo-docs/\n-rw-r--r-- README.md".into(),
            ),
            is_error: false,
        }],
    });
    conv.push(Message {
        role: Role::Assistant,
        content: vec![ContentBlock::ToolUse {
            id: "tool-call-002".into(),
            name: "grep".into(),
            input: serde_json::json!({
                "pattern": "TODO",
                "path": "/sandbox/buckets/missing-file"
            }),
            replay: None,
        }],
    });
    conv.push(Message {
        role: Role::ToolResult,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: "tool-call-002".into(),
            content: ToolResultContent::Text(
                "grep: /sandbox/buckets/missing-file: No such file or directory".into(),
            ),
            is_error: true,
        }],
    });

    ThreadSnapshot {
        thread_id: "t-1".into(),
        pod_id: "default".into(),
        title: Some("first conversation".into()),
        config: ThreadConfig {
            model: "claude-opus-4-7".into(),
            max_tokens: 4096,
            max_turns: 8,
            compaction: CompactionConfig::default(),
        },
        bindings: ThreadBindings::default(),
        state: ThreadStateLabel::Idle,
        conversation: conv,
        total_usage: Usage::default(),
        turn_log: TurnLog::default(),
        draft: String::new(),
        created_at: "2026-05-08T10:00:00Z".into(),
        last_active: "2026-05-08T11:00:00Z".into(),
        failure: None,
        origin: None,
        continued_from: None,
        dispatched_by: None,
        scope: Scope::default(),
    }
}
