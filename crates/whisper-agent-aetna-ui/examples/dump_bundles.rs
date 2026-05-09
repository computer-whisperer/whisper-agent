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
    BackendSummary, BehaviorOrigin, BehaviorSummary, CompactionConfig, ContentBlock,
    ContentCapabilities, Conversation, ImageMime, ImageSource, Message, ModelSummary, PodSummary,
    Role, ServerToClient, ThreadBindings, ThreadConfig, ThreadSnapshot, ThreadStateLabel,
    ThreadSummary, ToolKind, ToolResultContent, TurnEntry, TurnLog, Usage, permission::Scope,
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
    /// Subscribed thread whose conversation includes a real system
    /// prompt (`Role::System` with non-empty text) and a tools
    /// manifest (`Role::Tools` with several `ToolSchema` blocks)
    /// at the head, plus one user/assistant exchange. Verifies that
    /// the new `SetupPrompt` and `SetupTools` accordion rows render
    /// in their default-collapsed shape — head of chat reads as two
    /// muted one-line headers rather than dumping the full prompt.
    ThreadWithSetup,
    /// Subscribed thread with an `edit_file` tool call (in-place
    /// substitution) and a `write_file` tool call (creation) so the
    /// inline diff renderer is exercised — `+`/`-` colored line
    /// rows, file-path header, `(new) {path}` for the creation
    /// case. Both calls are pre-expanded so the diff bodies render
    /// without a synthetic click.
    ThreadWithDiff,
    /// Subscribed thread whose summary carries every provenance
    /// field (origin = behavior, continued_from = a previous
    /// thread, dispatched_by = a parent thread). Verifies the
    /// pane header's three chip slots (`via`, `forked from`,
    /// `dispatched from`) all render. A real thread won't usually
    /// carry all three at once — origin and dispatched_by
    /// typically don't co-occur — but this is the visual
    /// regression scene.
    ThreadWithProvenance,
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
    /// Sidebar with a dispatch chain: one parent thread fans out to
    /// three children. Verifies the depth-indent path in
    /// `order_threads_by_dispatch` renders the children below the
    /// parent.
    SidebarDispatchChain,
    /// Sidebar with more threads than `SIDEBAR_THREAD_PREVIEW`.
    /// Verifies the "Show N more" toggle button appears at the
    /// bottom of the truncated list.
    SidebarManyThreads,
    /// Sidebar with a behaviors-rich pod (mirrors the production
    /// `mavis` shape: 4 cron behaviors, one paused, one errored)
    /// plus interactive threads above and behavior-spawned runs
    /// nested under their parent behavior. Exercises the per-pod
    /// `BehaviorList` ingest path and the `behaviors_section`
    /// rendering — including the muted second line's status
    /// priority (errored > paused > last-fired-time).
    SidebarBehaviors,
    /// Same as `SidebarBehaviors` but with one behavior expanded so
    /// the nested-thread render path is exercised. Click sequence
    /// targets `behavior-row:default:architect` to toggle the
    /// architect row open.
    SidebarBehaviorsExpanded,
    /// Same as `SidebarBehaviorsExpanded` but with the Delete
    /// button armed (clicked once). Verifies the label flip
    /// ("Delete" → "Confirm delete?") and the destructive solid
    /// fill that's only meant to land on the second click. Same
    /// arm pattern slice γ uses for any future danger affordances.
    SidebarBehaviorsDeleteArmed,
    /// "+ New pod" modal opened straight from the sidebar header
    /// with no fields filled. Verifies the dialog scaffolding —
    /// scrim + content panel + form + cancel/create footer — paints
    /// over the populated chat surface and that Create renders
    /// disabled with empty inputs.
    NewPodModalEmpty,
    /// Same modal, but with `BackendsList { backends: [] }` seeded
    /// so the "no backends configured" warning alert renders
    /// inside the dialog body. Create stays disabled. (Filled-form
    /// state isn't dumped — synthesizing `UiEventKind::TextInput`
    /// requires fabricating the `#[non_exhaustive]` `UiTarget`,
    /// which isn't worth a public test seam.)
    NewPodModalNoBackends,
    /// "+ New behavior" modal opened from the per-pod sidebar
    /// `behaviors_section` header. Click sequence routes
    /// `sidebar:new-behavior:default` (the active tab) to scope
    /// the dialog title. Behaviors registry seeded with the
    /// production-shaped `mavis` set so the dialog overlays a real
    /// list, not an empty section.
    NewBehaviorModalEmpty,
    /// Sidebar with a pod that has zero behaviors yet — verifies
    /// the empty-state copy renders and the "+" affordance still
    /// sits in the section header (an empty pod is precisely where
    /// the user *needs* "+ New behavior" the most). No modal is
    /// opened; this just exercises the section's empty branch.
    SidebarBehaviorsEmpty,
}

impl Scene {
    const ALL: [Scene; 29] = [
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
        Scene::ThreadWithSetup,
        Scene::ThreadWithDiff,
        Scene::ThreadWithProvenance,
        Scene::LoginFormEmpty,
        Scene::LoginFormPrefilled,
        Scene::LoginFormWithError,
        Scene::SidebarDispatchChain,
        Scene::SidebarManyThreads,
        Scene::SidebarBehaviors,
        Scene::SidebarBehaviorsExpanded,
        Scene::SidebarBehaviorsDeleteArmed,
        Scene::SidebarBehaviorsEmpty,
        Scene::NewPodModalEmpty,
        Scene::NewPodModalNoBackends,
        Scene::NewBehaviorModalEmpty,
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
            Scene::ThreadWithSetup => "thread_with_setup",
            Scene::ThreadWithDiff => "thread_with_diff",
            Scene::ThreadWithProvenance => "thread_with_provenance",
            Scene::LoginFormEmpty => "login_form_empty",
            Scene::LoginFormPrefilled => "login_form_prefilled",
            Scene::LoginFormWithError => "login_form_with_error",
            Scene::SidebarDispatchChain => "sidebar_dispatch_chain",
            Scene::SidebarManyThreads => "sidebar_many_threads",
            Scene::SidebarBehaviors => "sidebar_behaviors",
            Scene::SidebarBehaviorsExpanded => "sidebar_behaviors_expanded",
            Scene::SidebarBehaviorsDeleteArmed => "sidebar_behaviors_delete_armed",
            Scene::SidebarBehaviorsEmpty => "sidebar_behaviors_empty",
            Scene::NewPodModalEmpty => "new_pod_modal_empty",
            Scene::NewPodModalNoBackends => "new_pod_modal_no_backends",
            Scene::NewBehaviorModalEmpty => "new_behavior_modal_empty",
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
            | Scene::ThreadWithImages
            | Scene::ThreadWithSetup
            | Scene::ThreadWithProvenance => vec!["thread:t-1"],
            // Open the thread, then click each diff tool's
            // accordion so the bodies render expanded. Indices
            // come from the conversation's display-item order:
            // user (0), assistant text (1), edit_file (2),
            // write_file (3).
            Scene::ThreadWithDiff => vec!["thread:t-1", "tool:accordion:2", "tool:accordion:3"],
            // Toggle the architect behavior row open so the nested
            // spawned-thread row renders inside `behaviors_section`.
            Scene::SidebarBehaviorsExpanded => vec!["behavior-row:default:architect"],
            // Open the architect row, then click its Delete button
            // once to arm. Auto-disarm pre-handler at the top of
            // `on_event` will leave the arm intact since this click
            // is on the matching `behavior-delete:` key.
            Scene::SidebarBehaviorsDeleteArmed => vec![
                "behavior-row:default:architect",
                "behavior-delete:default:architect",
            ],
            // One toggle click on the backend trigger leaves the menu
            // open — render captures the popover.
            Scene::NewThreadFormBackendOpen => vec!["picker:backend"],
            // Open the "+ New pod" dialog by clicking the sidebar
            // header's plus button. Same click path the live UI
            // takes — the modal then renders as an overlay layer.
            Scene::NewPodModalEmpty | Scene::NewPodModalNoBackends => vec!["sidebar:new-pod"],
            // Open the "+ New behavior" dialog scoped to the
            // active `default` pod. Click route carries the pod
            // id as suffix.
            Scene::NewBehaviorModalEmpty => vec!["sidebar:new-behavior:default"],
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
        Scene::SidebarDispatchChain => {
            q.push_back(InboundEvent::ConnectionOpened);
            q.push_back(InboundEvent::Wire(ServerToClient::PodList {
                correlation_id: None,
                pods: mock_pods(),
                default_pod_id: "default".into(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::ThreadList {
                correlation_id: None,
                tasks: mock_dispatch_chain_threads(),
            }));
        }
        Scene::SidebarManyThreads => {
            q.push_back(InboundEvent::ConnectionOpened);
            q.push_back(InboundEvent::Wire(ServerToClient::PodList {
                correlation_id: None,
                pods: mock_pods(),
                default_pod_id: "default".into(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::ThreadList {
                correlation_id: None,
                tasks: mock_many_threads(15),
            }));
        }
        Scene::SidebarBehaviors
        | Scene::SidebarBehaviorsExpanded
        | Scene::SidebarBehaviorsDeleteArmed => {
            q.push_back(InboundEvent::ConnectionOpened);
            q.push_back(InboundEvent::Wire(ServerToClient::PodList {
                correlation_id: None,
                pods: mock_pods(),
                default_pod_id: "default".into(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::ThreadList {
                correlation_id: None,
                tasks: mock_mavis_threads(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::BehaviorList {
                correlation_id: None,
                pod_id: "default".into(),
                behaviors: mock_mavis_behaviors(),
            }));
        }
        Scene::PopulatedNoSelection
        | Scene::LoadingThread
        | Scene::ThreadWithMessages
        | Scene::ThreadWithToolCall
        | Scene::ThreadPrefilling
        | Scene::ThreadWithDraft
        | Scene::ThreadWithImages
        | Scene::ThreadWithSetup
        | Scene::ThreadWithDiff
        | Scene::ThreadWithProvenance => {
            q.push_back(InboundEvent::ConnectionOpened);
            q.push_back(InboundEvent::Wire(ServerToClient::PodList {
                correlation_id: None,
                pods: mock_pods(),
                default_pod_id: "default".into(),
            }));
            // Provenance scene swaps in a thread list whose t-1
            // carries origin / continued_from / dispatched_by so
            // the pane header's chip cluster has all three to
            // render. Other scenes keep `mock_threads()` for
            // visual stability.
            let threads = if matches!(scene, Scene::ThreadWithProvenance) {
                mock_provenance_threads()
            } else {
                mock_threads()
            };
            q.push_back(InboundEvent::Wire(ServerToClient::ThreadList {
                correlation_id: None,
                tasks: threads,
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
            if matches!(scene, Scene::ThreadWithSetup) {
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_setup_snapshot(),
                }));
            }
            if matches!(scene, Scene::ThreadWithDiff) {
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_diff_snapshot(),
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
        Scene::NewPodModalEmpty | Scene::NewPodModalNoBackends => {
            // Populated baseline so the dialog overlays a non-empty
            // surface (catches any clipping / blend issues that an
            // empty viewport would hide). The synthetic click on
            // `sidebar:new-pod` (in `clicks()`) opens the modal.
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
            // Empty backends list → the dialog renders the warning
            // alert; otherwise seed a real list so Create can flip
            // to enabled in normal use (still disabled here only
            // because the inputs are blank).
            let backends = if matches!(scene, Scene::NewPodModalNoBackends) {
                Vec::new()
            } else {
                mock_backends()
            };
            q.push_back(InboundEvent::Wire(ServerToClient::BackendsList {
                correlation_id: None,
                backends,
            }));
        }
        Scene::NewBehaviorModalEmpty => {
            // Same baseline as `SidebarBehaviors` — the dialog
            // overlays a real behaviors registry so the scrim
            // layering shows over a populated section header.
            q.push_back(InboundEvent::ConnectionOpened);
            q.push_back(InboundEvent::Wire(ServerToClient::PodList {
                correlation_id: None,
                pods: mock_pods(),
                default_pod_id: "default".into(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::ThreadList {
                correlation_id: None,
                tasks: mock_mavis_threads(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::BehaviorList {
                correlation_id: None,
                pod_id: "default".into(),
                behaviors: mock_mavis_behaviors(),
            }));
        }
        Scene::SidebarBehaviorsEmpty => {
            // Pod-but-no-behaviors scene: the empty state copy
            // ("no behaviors in this pod yet") + the "+" affordance
            // in the section header are the focus. No `BehaviorList`
            // wire event so the section sees an empty list.
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
            q.push_back(InboundEvent::Wire(ServerToClient::BehaviorList {
                correlation_id: None,
                pod_id: "default".into(),
                behaviors: Vec::new(),
            }));
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

/// Same shape as [`mock_threads`] but t-1 carries every
/// provenance field — origin (behavior-spawned), continued_from
/// (forked), dispatched_by (parent thread). Used by the
/// `ThreadWithProvenance` scene to validate the pane header's
/// three chip slots.
fn mock_provenance_threads() -> Vec<ThreadSummary> {
    let mut t = mock_threads();
    t[0].origin = Some(BehaviorOrigin {
        behavior_id: "architect".into(),
        fired_at: "2026-05-08T10:30:00Z".into(),
        trigger_payload: serde_json::Value::Null,
    });
    t[0].continued_from = Some("task-7c9c8d6a4f0b1234".into());
    t[0].dispatched_by = Some("task-18abcff7ad2a29b9".into());
    t
}

/// One parent thread plus three dispatched children, all in the
/// `default` pod. Mirrors the production `mavis` instance's
/// dispatch fan-out (`task-18abcff7ad2a29b9` → 3 children) so the
/// sidebar's depth-indent rendering exercises a realistic shape.
fn mock_dispatch_chain_threads() -> Vec<ThreadSummary> {
    let parent_id = "task-parent".to_string();
    vec![
        ThreadSummary {
            thread_id: parent_id.clone(),
            pod_id: "default".into(),
            title: Some("Investigation: behavior compaction".into()),
            state: ThreadStateLabel::Completed,
            created_at: "2026-05-08T08:00:00Z".into(),
            last_active: "2026-05-08T08:30:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: None,
        },
        ThreadSummary {
            thread_id: "task-child-a".into(),
            pod_id: "default".into(),
            title: Some("subthread A — survey current pipeline".into()),
            state: ThreadStateLabel::Completed,
            created_at: "2026-05-08T08:10:00Z".into(),
            last_active: "2026-05-08T08:20:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: Some(parent_id.clone()),
        },
        ThreadSummary {
            thread_id: "task-child-b".into(),
            pod_id: "default".into(),
            title: Some("subthread B — measure compaction overhead".into()),
            state: ThreadStateLabel::Completed,
            created_at: "2026-05-08T08:11:00Z".into(),
            last_active: "2026-05-08T08:22:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: Some(parent_id.clone()),
        },
        ThreadSummary {
            thread_id: "task-child-c".into(),
            pod_id: "default".into(),
            title: Some("subthread C — identify failure modes".into()),
            state: ThreadStateLabel::Failed,
            created_at: "2026-05-08T08:12:00Z".into(),
            last_active: "2026-05-08T08:24:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: Some(parent_id),
        },
    ]
}

/// Threads mirroring the production `mavis` pod's mix: interactive
/// user-initiated conversations + a few behavior-spawned runs. The
/// behavior-spawned threads carry a populated `origin` so they're
/// filtered out of the interactive list and nest under their
/// behavior in `behaviors_section`.
fn mock_mavis_threads() -> Vec<ThreadSummary> {
    let mk_origin = |behavior: &str, fired: &str| BehaviorOrigin {
        behavior_id: behavior.into(),
        fired_at: fired.into(),
        trigger_payload: serde_json::Value::Null,
    };
    vec![
        // --- interactive threads ---
        ThreadSummary {
            thread_id: "task-int-001".into(),
            pod_id: "default".into(),
            title: Some("Mavis svg redesign".into()),
            state: ThreadStateLabel::Idle,
            created_at: "2026-05-09T07:00:00Z".into(),
            last_active: "2026-05-09T07:30:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: None,
        },
        ThreadSummary {
            thread_id: "task-int-002".into(),
            pod_id: "default".into(),
            title: Some("Description blurbs for github profile".into()),
            state: ThreadStateLabel::Completed,
            created_at: "2026-05-08T22:00:00Z".into(),
            last_active: "2026-05-08T22:45:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: None,
        },
        ThreadSummary {
            thread_id: "task-int-003".into(),
            pod_id: "default".into(),
            title: Some("testing tool surface".into()),
            state: ThreadStateLabel::Failed,
            created_at: "2026-05-08T18:00:00Z".into(),
            last_active: "2026-05-08T18:10:00Z".into(),
            origin: None,
            continued_from: None,
            dispatched_by: None,
        },
        // --- behavior-spawned (architect) ---
        ThreadSummary {
            thread_id: "task-architect-001".into(),
            pod_id: "default".into(),
            title: Some("Analyze recent agent threads. Identify failure pat…".into()),
            state: ThreadStateLabel::Completed,
            created_at: "2026-05-09T02:00:00Z".into(),
            last_active: "2026-05-09T02:18:00Z".into(),
            origin: Some(mk_origin("architect", "2026-05-09T02:00:00Z")),
            continued_from: None,
            dispatched_by: None,
        },
        // --- behavior-spawned (researcher, two recent fires) ---
        ThreadSummary {
            thread_id: "task-researcher-001".into(),
            pod_id: "default".into(),
            title: Some("Periodic wiki sweep — root /home/christian".into()),
            state: ThreadStateLabel::Completed,
            created_at: "2026-05-09T07:50:00Z".into(),
            last_active: "2026-05-09T07:55:00Z".into(),
            origin: Some(mk_origin("researcher", "2026-05-09T07:50:00Z")),
            continued_from: None,
            dispatched_by: None,
        },
        ThreadSummary {
            thread_id: "task-researcher-002".into(),
            pod_id: "default".into(),
            title: Some("Periodic wiki sweep — root /home/christian".into()),
            state: ThreadStateLabel::Completed,
            created_at: "2026-05-09T06:50:00Z".into(),
            last_active: "2026-05-09T06:54:00Z".into(),
            origin: Some(mk_origin("researcher", "2026-05-09T06:50:00Z")),
            continued_from: None,
            dispatched_by: None,
        },
        // --- orphan-origin (behavior 'old-greeter' was deleted but
        //     this thread survived; it should fall through to the
        //     interactive list with the 'via old-greeter' marker).
        ThreadSummary {
            thread_id: "task-orphan-001".into(),
            pod_id: "default".into(),
            title: Some("welcome scan — pre-rename".into()),
            state: ThreadStateLabel::Completed,
            created_at: "2026-05-01T10:00:00Z".into(),
            last_active: "2026-05-01T10:05:00Z".into(),
            origin: Some(mk_origin("old-greeter", "2026-05-01T10:00:00Z")),
            continued_from: None,
            dispatched_by: None,
        },
    ]
}

/// Behaviors mirroring the production `mavis` pod (4 cron agents),
/// with two enabled-and-recently-fired entries, one paused, and one
/// errored. Exercises the four status branches in
/// `behavior_item_row`'s description line.
fn mock_mavis_behaviors() -> Vec<BehaviorSummary> {
    vec![
        BehaviorSummary {
            behavior_id: "architect".into(),
            pod_id: "default".into(),
            name: "architect".into(),
            description: Some(
                "Meta-agent. Observes agent performance and modifies behaviors \
                 to optimize the system loop."
                    .into(),
            ),
            trigger_kind: Some("cron".into()),
            enabled: true,
            run_count: 12,
            last_fired_at: Some("2026-05-09T02:00:00Z".into()),
            load_error: None,
        },
        BehaviorSummary {
            behavior_id: "researcher".into(),
            pod_id: "default".into(),
            name: "researcher".into(),
            description: Some("Periodic knowledge-gathering agent.".into()),
            trigger_kind: Some("cron".into()),
            enabled: true,
            run_count: 312,
            last_fired_at: Some("2026-05-09T07:50:00Z".into()),
            load_error: None,
        },
        BehaviorSummary {
            behavior_id: "reviewer".into(),
            pod_id: "default".into(),
            name: "reviewer".into(),
            description: Some("Cartographer agent.".into()),
            trigger_kind: Some("cron".into()),
            enabled: false,
            run_count: 48,
            last_fired_at: Some("2026-05-08T22:20:00Z".into()),
            load_error: None,
        },
        BehaviorSummary {
            behavior_id: "synthesizer".into(),
            pod_id: "default".into(),
            name: "synthesizer".into(),
            description: Some("Janitorial agent.".into()),
            trigger_kind: None,
            enabled: true,
            run_count: 0,
            last_fired_at: None,
            load_error: Some("behavior.toml: invalid cron `30 * * *` (expected 5 fields)".into()),
        },
    ]
}

/// `n` synthetic threads in the default pod, for the
/// `SidebarManyThreads` scene that exercises the "Show N more"
/// pagination toggle once the total exceeds
/// `SIDEBAR_THREAD_PREVIEW`. Titles are deterministic so the dump
/// is stable across regenerations.
fn mock_many_threads(n: usize) -> Vec<ThreadSummary> {
    (0..n)
        .map(|i| {
            // Vary the state across rows so the second-line caption
            // exercises every label the sidebar renders.
            let state = match i % 4 {
                0 => ThreadStateLabel::Idle,
                1 => ThreadStateLabel::Working,
                2 => ThreadStateLabel::Completed,
                _ => ThreadStateLabel::Failed,
            };
            // Walk `last_active` backward in 47-minute steps so the
            // relative-time captions span minutes/hours/days.
            let minutes_ago = i as i64 * 47;
            let last = chrono::DateTime::parse_from_rfc3339("2026-05-09T08:00:00Z").unwrap()
                - chrono::Duration::minutes(minutes_ago);
            ThreadSummary {
                thread_id: format!("task-many-{i:02}"),
                pod_id: "default".into(),
                title: Some(format!("synthetic thread #{i}")),
                state,
                created_at: last.to_rfc3339(),
                last_active: last.to_rfc3339(),
                origin: None,
                continued_from: None,
                dispatched_by: None,
            }
        })
        .collect()
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

/// Snapshot for the `ThreadWithSetup` scene — a real `Role::System`
/// system prompt and a `Role::Tools` manifest carrying a handful of
/// `ContentBlock::ToolSchema` entries, plus one user/assistant
/// exchange afterwards. Exercises the default-collapsed
/// `SetupPrompt` and `SetupTools` accordion rows at the head of
/// the chat — the noisy "rah-rah here is your full system prompt"
/// wall is replaced by a one-line preview the user can expand.
fn mock_setup_snapshot() -> ThreadSnapshot {
    let mut conv = Conversation::new();
    conv.push(Message::system_text(
        "You are Mavis, a coding agent embedded inside whisper-agent.\n\
         \n\
         Behaviors:\n\
         - Read the user's request literally, then verify against actual code\n\
         - Prefer minimal diffs — match existing style and conventions\n\
         - When you are uncertain, ask before guessing\n\
         - Never silently expand scope past the request\n\
         \n\
         When you finish a task, summarize the changes and the next step in\n\
         no more than two sentences.",
    ));
    conv.push(Message {
        role: Role::Tools,
        content: vec![
            ContentBlock::ToolSchema {
                name: "list_files".into(),
                description: "List the files at a directory path. Honors the pod's allow-list.".into(),
                params: Vec::new(),
                kind: ToolKind::default(),
            },
            ContentBlock::ToolSchema {
                name: "read_file".into(),
                description: "Read a file's contents as UTF-8 text.".into(),
                params: Vec::new(),
                kind: ToolKind::default(),
            },
            ContentBlock::ToolSchema {
                name: "write_file".into(),
                description: "Replace a file's contents. Returns a unified diff against the previous content.".into(),
                params: Vec::new(),
                kind: ToolKind::default(),
            },
            ContentBlock::ToolSchema {
                name: "grep".into(),
                description: "Search a path tree for a regex pattern. Returns up to 200 hits.".into(),
                params: Vec::new(),
                kind: ToolKind::default(),
            },
            ContentBlock::ToolSchema {
                name: "run_bash".into(),
                description: "Run a bash command in the pod's sandbox. Captures stdout / stderr / exit code.".into(),
                params: Vec::new(),
                kind: ToolKind::default(),
            },
        ],
    });
    conv.push(Message::user_text(
        "Quick sanity check — do you see the system prompt I just loaded?",
    ));
    conv.push(Message {
        role: Role::Assistant,
        content: vec![ContentBlock::Text {
            text: "Yes — it's the Mavis prompt that scopes responses to literal reads of code,\
                   minimal-diff edits, and asking-before-guessing on uncertainty. Ready when you are."
                .into(),
        }],
    });
    let mut snapshot = base_snapshot(conv, ThreadStateLabel::Idle, String::new());
    // One assistant turn → one TurnEntry. Stats line lands under
    // the assistant message in the rendered scene.
    snapshot.turn_log = TurnLog {
        entries: vec![TurnEntry {
            usage: Usage {
                input_tokens: 1284,
                output_tokens: 86,
                cache_read_input_tokens: 980,
                cache_creation_input_tokens: 0,
            },
        }],
    };
    snapshot
}

/// Snapshot for the `ThreadWithDiff` scene — one `edit_file`
/// call (in-place substitution) and one `write_file` call
/// (creation). Both fused with their tool results so the call
/// rows render with the diff body in their accordion. Synthetic
/// clicks open both accordions so the rendered scene shows the
/// `+`/`-` rows directly.
fn mock_diff_snapshot() -> ThreadSnapshot {
    let mut conv = Conversation::new();
    conv.push(Message::system_text(""));
    conv.push(Message::user_text(
        "Add a `total_count()` helper to the cache module, then create \
         a tiny module-level docs file alongside it.",
    ));
    conv.push(Message {
        role: Role::Assistant,
        content: vec![
            ContentBlock::Text {
                text: "Sure — substituting the helper in first, then creating the docs file."
                    .into(),
            },
            ContentBlock::ToolUse {
                id: "tool-edit-001".into(),
                name: "edit_file".into(),
                input: serde_json::json!({
                    "path": "src/cache.rs",
                    "old_string":
                        "pub struct Cache {\n    inner: HashMap<String, Entry>,\n}\n\
                         \n\
                         impl Cache {\n    pub fn new() -> Self {\n        \
                             Self { inner: HashMap::new() }\n    }\n}\n",
                    "new_string":
                        "pub struct Cache {\n    inner: HashMap<String, Entry>,\n}\n\
                         \n\
                         impl Cache {\n    pub fn new() -> Self {\n        \
                             Self { inner: HashMap::new() }\n    }\n\n    \
                             /// Total number of entries currently held.\n    \
                             pub fn total_count(&self) -> usize {\n        \
                                 self.inner.len()\n    }\n}\n",
                }),
                replay: None,
            },
        ],
    });
    conv.push(Message {
        role: Role::ToolResult,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: "tool-edit-001".into(),
            content: ToolResultContent::Text("edited src/cache.rs (1 substitution applied)".into()),
            is_error: false,
        }],
    });
    conv.push(Message {
        role: Role::Assistant,
        content: vec![ContentBlock::ToolUse {
            id: "tool-write-001".into(),
            name: "write_file".into(),
            input: serde_json::json!({
                "path": "src/cache.md",
                "content":
                    "# Cache\n\nIn-memory key/value store; entries expire when the\n\
                     pod's `cache_ttl` setting elapses.\n",
            }),
            replay: None,
        }],
    });
    conv.push(Message {
        role: Role::ToolResult,
        content: vec![ContentBlock::ToolResult {
            tool_use_id: "tool-write-001".into(),
            content: ToolResultContent::Text("wrote src/cache.md (4 lines)".into()),
            is_error: false,
        }],
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
