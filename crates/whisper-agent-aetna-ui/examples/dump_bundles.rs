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
use whisper_agent_protocol::{PodSummary, ServerToClient, ThreadStateLabel, ThreadSummary};

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
}

impl Scene {
    const ALL: [Scene; 6] = [
        Scene::Connecting,
        Scene::Connected,
        Scene::Closed,
        Scene::Error,
        Scene::PopulatedNoSelection,
        Scene::LoadingThread,
    ];

    fn slug(self) -> &'static str {
        match self {
            Scene::Connecting => "connecting",
            Scene::Connected => "connected",
            Scene::Closed => "closed",
            Scene::Error => "error",
            Scene::PopulatedNoSelection => "populated_no_selection",
            Scene::LoadingThread => "loading_thread",
        }
    }

    /// Synthetic clicks dispatched after the wire seed. Threaded
    /// through `App::on_event`, so they exercise the same routing the
    /// live UI does.
    fn clicks(self) -> Vec<&'static str> {
        match self {
            Scene::LoadingThread => vec!["thread:t-1"],
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
        Scene::PopulatedNoSelection | Scene::LoadingThread => {
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
