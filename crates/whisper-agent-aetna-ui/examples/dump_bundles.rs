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
use whisper_agent_aetna_ui::{ChatApp, Inbound, InboundEvent, SendFn};
use whisper_agent_protocol::{
    CompactionConfig, ContentBlock, Conversation, Message, PodSummary, Role, ServerToClient,
    ThreadBindings, ThreadConfig, ThreadSnapshot, ThreadStateLabel, ThreadSummary,
    ToolResultContent, TurnLog, Usage, permission::Scope,
};

fn main() -> std::io::Result<()> {
    // Native window viewport — the bug, if any, shouldn't depend on
    // viewport size. Keeping it identical to the desktop binary's
    // window dimensions makes the SVG read like the live UI at idle.
    let viewport = Rect::new(0.0, 0.0, 1200.0, 800.0);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out");

    for scene in Scene::ALL {
        let mut app = build_app(scene);
        // before_build drains seeded wire frames into the App's
        // state machine — same path the live host calls. Done before
        // any synthetic clicks so a click that depends on the wire
        // state (selecting a known thread) sees that state.
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
}

impl Scene {
    const ALL: [Scene; 8] = [
        Scene::Connecting,
        Scene::Connected,
        Scene::Closed,
        Scene::Error,
        Scene::PopulatedNoSelection,
        Scene::LoadingThread,
        Scene::ThreadWithMessages,
        Scene::ThreadWithToolCall,
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
        }
    }

    /// Synthetic clicks dispatched after the wire seed. Threaded
    /// through `App::on_event`, so they exercise the same routing the
    /// live UI does.
    fn clicks(self) -> Vec<&'static str> {
        match self {
            Scene::LoadingThread | Scene::ThreadWithMessages | Scene::ThreadWithToolCall => {
                vec!["thread:t-1"]
            }
            _ => Vec::new(),
        }
    }
}

fn build_app(scene: Scene) -> ChatApp {
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
        | Scene::ThreadWithToolCall => {
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
        }
    }
    drop(q);
    app
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
