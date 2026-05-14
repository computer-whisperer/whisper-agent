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
    ActiveSlotSummary, BackendSummary, BehaviorOrigin, BehaviorSummary, BucketBuildPhase,
    BucketSummary, ClientToServer, CompactionConfig, ContentBlock, ContentCapabilities,
    Conversation, EmbeddingProviderInfo, FsEntry, ImageMime, ImageSource, Message, ModelSummary,
    PodSummary, Role, ServerToClient, SlotStateLabel, ThreadBindings, ThreadConfig, ThreadSnapshot,
    ThreadStateLabel, ThreadSummary, ToolKind, ToolResultContent, TurnEntry, TurnLog, Usage,
    permission::Scope,
};

fn main() -> std::io::Result<()> {
    // Native window viewport. Keep this identical to the desktop
    // binary's default so the SVG reads like the live UI at idle.
    // 1600x900 logical pixels is conservative for a 3840x2160 display
    // at scale=2 while avoiding an unrealistically cramped desktop
    // validation target.
    let viewport = Rect::new(0.0, 0.0, 1600.0, 900.0);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out");

    for scene in Scene::ALL {
        // Both `ChatApp` and `LoginApp` implement `App`. The dump
        // loop is generic over `Box<dyn App>` so a new App variant
        // (e.g. a future modal-host App) can be added without
        // teaching the renderer about it.
        let mut app: Box<dyn App> = build_app(scene);
        // Drain the wire seed once (mirrors a real frame start).
        app.before_build();
        for click in scene.clicks() {
            app.on_event(UiEvent::synthetic_click(click));
        }
        // Second drain: scenes that depend on a wire response to a
        // request the click loop just fired (e.g. behavior-editor
        // scenes that pre-queue the matching `BehaviorSnapshot` for
        // the click-side `GetBehavior`) need this to land their
        // response into a UI state that already opened. A no-op for
        // scenes whose queue is empty by now.
        app.before_build();
        let theme = app.theme();
        let cx = BuildCx::new(&theme);
        let mut tree = app.build(&cx);
        let bundle = render_bundle_themed(&mut tree, viewport, &theme);

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
    /// New-thread compose form with the separate thread-overrides
    /// modal open. This keeps root-level compose controls compact
    /// while still dumping the full per-thread override surface.
    NewThreadOverridesInheritedOpen,
    /// Same modal, with each optional override group enabled so the
    /// dump covers the detailed controls behind the activation rows.
    NewThreadOverridesOpen,
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
    /// Subscribed thread in `Failed` state with a failure detail
    /// on the snapshot. Verifies the destructive banner renders
    /// between the header and the chat log, and that the gate
    /// (state == Failed AND view.failure.is_some()) behaves
    /// correctly.
    ThreadWithFailure,
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
    /// Per-behavior editor sheet, hydrated against the architect
    /// behavior's snapshot. Click sequence: expand the behavior
    /// row, click Edit. The seed pre-pushes a `BehaviorSnapshot`
    /// reply with a deterministic correlation id (`aui-1`, the first
    /// id the editor mints — see `next_correlation_id`); the click
    /// fires the matching `GetBehavior` from `open_behavior_editor`
    /// and the inbound queue's BehaviorSnapshot then hydrates the
    /// form so the dump captures the populated body, not "loading…".
    BehaviorEditorHydrated,
    /// Same baseline as [`BehaviorEditorHydrated`] but with the
    /// trigger-kind `select_menu` toggled open via a third click.
    /// Verifies the menu paints over the sheet panel via
    /// `popover_layers` ordering.
    BehaviorEditorTriggerKindOpen,
    /// Pod editor sheet, hydrated against the active pod's
    /// `pod.toml`. Click sequence: gear icon in the sidebar header
    /// (only rendered when `pod_tab.is_some()`). The per-scene
    /// SendFn synthesizes a `PodSnapshot` reply when the matching
    /// `GetPod` lands, then the dump loop's second `before_build`
    /// drain hydrates the editor before render.
    PodEditorHydrated,
    /// Pod editor sheet, opened and switched to the Defaults tab.
    /// Same hydration shape as `PodEditorHydrated`; an extra click
    /// on the `pod-editor:tabs:tab:defaults` trigger flips the
    /// active tab without exercising any picker popovers.
    PodEditorDefaultsTab,
    /// Behavior editor sheet, opened and switched to the Thread tab.
    /// The architect mock uses `BehaviorThreadOverride::default()`,
    /// so every override row renders in the "(inherit pod default)"
    /// shape — a useful baseline for the structured layout, even if
    /// it doesn't exercise the override-on numeric inputs.
    BehaviorEditorThreadTab,
    /// Behavior editor sheet, opened and switched to the Scope
    /// tab. Architect mock has `BehaviorScope::default()` so every
    /// row renders in the inherit-only shape — useful for the
    /// structured layout, even though it doesn't exercise the
    /// override-on multi-checks / cap pickers.
    BehaviorEditorScopeTab,
    /// Behavior editor sheet, opened and switched to the Retention
    /// tab. Architect mock has `RetentionPolicy::default()` (Keep),
    /// so the days numeric input doesn't render — the tab shows
    /// only the kind picker.
    BehaviorEditorRetentionTab,
    /// Behavior editor sheet, opened and switched to the System
    /// (system_prompt) tab. Architect mock has
    /// `cfg.thread.system_prompt = None`, so the body renders the
    /// inherit-only paragraph; the override checkbox + content
    /// text_area only appear after the user toggles override on.
    BehaviorEditorSystemPromptTab,
    /// Behavior editor sheet, opened and switched to the Raw TOML
    /// tab. The `working_toml` buffer was seeded from the architect
    /// snapshot's parsed config in `hydrate`, so the textarea shows
    /// the round-tripped TOML at first visit. Switching back to a
    /// structured tab reparses; a parse error keeps the user on
    /// Raw with the message in the destructive alert.
    BehaviorEditorRawTab,
    /// File-tree → behavior editor deep-link. Opens the file tree
    /// modal, expands `behaviors/` and `behaviors/architect/`, then
    /// clicks `prompt.md`. The dispatch routes
    /// `PodFileDispatch::BehaviorPrompt(architect)` to
    /// `open_behavior_editor_on_tab(..., Prompt)`, so the resulting
    /// editor lands on the Prompt tab (mirroring the egui sibling's
    /// `open_behavior_editor_on_tab(..., BehaviorEditorTab::Prompt)`
    /// call). Same `mock_architect_snapshot` + `mock_default_pod_snapshot`
    /// replies the BehaviorEditor* scenes use; the second
    /// `before_build` drain hydrates the editor's tab body before
    /// the dump paints.
    FileTreeBehaviorPromptDeepLink,
    /// Pod editor sheet, opened on the General tab. Same hydration
    /// shape as `PodEditorHydrated`; kept as its own dump so the
    /// identity + pod-wide controls remain covered even if the
    /// default tab changes later.
    PodEditorGeneralTab,
    /// Pod editor sheet switched to the Allow tab so the allowed
    /// resources and permission ceiling columns are covered now that
    /// the hydrated editor defaults to General.
    PodEditorAllowTab,
    /// Fork-from-message dialog with a User row pre-selected. The
    /// dialog opens via a synthetic click on the per-row fork
    /// affordance (`chat:user-fork:{msg_index}`) — routing is
    /// key-based, not hit-test-based, so this works even though the
    /// dump's `BuildCx` doesn't carry a `UiState` to drive
    /// `is_hovering_within`. (The hover-revealed affordance itself
    /// is verified live in the dev binary; the dump captures the
    /// post-click modal layout.)
    ForkModalOpen,
    /// Image lightbox modal opened by clicking an inline image.
    /// Exercises the `CHAT_IMAGE_LIGHTBOX_PREFIX` route, the
    /// `LightboxState` slot, and the `render_lightbox_modal`
    /// dialog layout (image at constrained dims + caption + close
    /// button + scrim dismiss).
    LightboxOpen,
    /// Pending sudo approval banner. Seeded via a synthetic
    /// `ServerToClient::SudoRequested` event so the renderer can
    /// snapshot the warning-styled `alert` shape and the three
    /// approve / remember / reject affordances above the chat log.
    SudoPending,
    /// Read-only JSON tree viewer modal opened over a mock pod file.
    /// Seeded by calling `ChatApp::open_json_viewer` before the first
    /// drain; the per-scene SendFn synthesizes a `PodFileContent`
    /// reply carrying a mixed-shape payload (scalars + arrays +
    /// nested object) so the recursive renderer + collapsible
    /// trigger rows + child indents all paint in one scene.
    JsonViewerOpen,
    /// Edit-with-save file viewer modal opened over a mock editable
    /// pod file. Same wiring as `JsonViewerOpen` — the per-scene
    /// SendFn synthesizes a `PodFileContent` reply (readonly = false)
    /// for the `ReadPodFile` that `open_file_viewer` fires. The body
    /// shows the populated text_area and the footer carries Close /
    /// Revert (disabled — buffer == baseline) / Save (disabled).
    FileViewerEditable,
    /// Read-only variant of the file viewer modal. Same path, but
    /// the synthesized reply sets `readonly = true`, so the footer
    /// collapses to a single Close button and the description
    /// carries the runtime-owned hint.
    FileViewerReadOnly,
    /// File tree modal scoped to a pod, with one subdir expanded.
    /// Seeded by calling `ChatApp::open_file_tree_modal` (fires
    /// `ListPodDir` on the root) and a synthetic click that
    /// expands the `behaviors/` subdir. The per-scene SendFn
    /// answers both `ListPodDir` requests with mock entries.
    FileTreeOpen,
    /// Server-settings modal on the Backends tab — read-only
    /// catalog over a pre-seeded `BackendsList`. No wire
    /// interaction; just snapshots the per-backend card shape.
    SettingsBackends,
    /// Server-settings modal on the Server-config tab — the lazy
    /// `FetchServerConfig` round-trip is answered by the per-scene
    /// SendFn with a mock whisper-agent.toml, so the text_area
    /// paints populated and the Save / Revert affordances render
    /// disabled (buffer == baseline).
    SettingsServerConfig,
    /// Server-settings modal on the Host env tab over a pre-seeded
    /// daemon registry: one connected daemon advertising landlock +
    /// container support, one admitted-but-offline daemon.
    SettingsHostEnv,
    /// Thread inspector panel expanded over a populated thread.
    /// Same baseline as `ThreadWithMessages`; the click loop selects
    /// the thread, then toggles the inspector. The panel paints
    /// key/value rows for bindings, max-tokens / max-turns,
    /// cumulative usage, and any trigger origin.
    ThreadInspectorOpen,
    /// Knowledge-buckets modal over a populated catalog. The
    /// `BucketsList` wire seed carries three buckets covering the
    /// three states the row chrome cares about: ready (with active
    /// slot), in-flight build (driven by a synthetic
    /// `BucketBuildStarted` + `BucketBuildProgress` pair), and
    /// armed-delete (driven by a synthetic click that lands the
    /// row in `delete_armed`).
    BucketsModalCatalog,
    /// Buckets modal with search results painted. Same baseline
    /// catalog as `BucketsModalCatalog`; the click loop fires a
    /// query against the ready bucket and the per-scene SendFn
    /// answers with mock hits, so the results pane paints with one
    /// expanded hit and one collapsed snippet.
    BucketsModalSearchResults,
    /// Buckets modal with the +New bucket create form open at its
    /// default `linked` source kind. The `EmbeddingProvidersList`
    /// wire seed populates the embedder picker; the form is
    /// pre-seeded with reasonable typed values so every input
    /// paints content rather than a placeholder ghost.
    BucketsModalCreateForm,
    /// Same shell as `BucketsModalCreateForm`, but with the
    /// source kind switched to `tracked` so the driver / language /
    /// mirror / cadence sub-fields render.
    BucketsModalCreateFormTracked,
    /// Server-settings modal on Backends with the Codex rotate
    /// sub-form open above it. Seeded backend list includes a
    /// `chatgpt_subscription` entry whose Rotate button the click
    /// loop activates; the sub-form text_area paints the empty
    /// state so the inline hint reads.
    SettingsCodexRotate,
    /// Server-settings modal on the Shared MCP tab over a
    /// populated `SharedMcpHostsList`. Three rows: anonymous
    /// connected, bearer connected, oauth disconnected (with
    /// last_error). The third row is armed for removal so the
    /// destructive Confirm + Cancel pair renders.
    SettingsSharedMcpList,
    /// Shared MCP editor sub-form open in Add mode over the
    /// settings modal. Auth picker shows Anonymous / Bearer /
    /// OAuth (OAuth disabled on native — `OAUTH_AVAILABLE` is
    /// false), prefix picker shows the three Default / None /
    /// Custom options.
    SettingsSharedMcpEditorAdd,
    /// Shared MCP editor sub-form open in Edit mode pre-populated
    /// from a seeded bearer host. Name field is locked; bearer
    /// input shows the "leave blank to keep existing" hint.
    SettingsSharedMcpEditorEdit,
}

impl Scene {
    const ALL: [Scene; 63] = [
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
        Scene::NewThreadOverridesInheritedOpen,
        Scene::NewThreadOverridesOpen,
        Scene::NewThreadFormFilled,
        Scene::ThreadPrefilling,
        Scene::ThreadWithDraft,
        Scene::ThreadWithImages,
        Scene::ThreadWithSetup,
        Scene::ThreadWithDiff,
        Scene::ThreadWithProvenance,
        Scene::ThreadWithFailure,
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
        Scene::BehaviorEditorHydrated,
        Scene::BehaviorEditorTriggerKindOpen,
        Scene::PodEditorHydrated,
        Scene::PodEditorDefaultsTab,
        Scene::PodEditorGeneralTab,
        Scene::PodEditorAllowTab,
        Scene::BehaviorEditorThreadTab,
        Scene::BehaviorEditorScopeTab,
        Scene::BehaviorEditorRetentionTab,
        Scene::BehaviorEditorSystemPromptTab,
        Scene::BehaviorEditorRawTab,
        Scene::FileTreeBehaviorPromptDeepLink,
        Scene::ForkModalOpen,
        Scene::LightboxOpen,
        Scene::SudoPending,
        Scene::JsonViewerOpen,
        Scene::FileViewerEditable,
        Scene::FileViewerReadOnly,
        Scene::FileTreeOpen,
        Scene::SettingsBackends,
        Scene::SettingsServerConfig,
        Scene::SettingsHostEnv,
        Scene::ThreadInspectorOpen,
        Scene::BucketsModalCatalog,
        Scene::BucketsModalSearchResults,
        Scene::BucketsModalCreateForm,
        Scene::BucketsModalCreateFormTracked,
        Scene::SettingsCodexRotate,
        Scene::SettingsSharedMcpList,
        Scene::SettingsSharedMcpEditorAdd,
        Scene::SettingsSharedMcpEditorEdit,
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
            Scene::NewThreadOverridesInheritedOpen => "new_thread_overrides_inherited_open",
            Scene::NewThreadOverridesOpen => "new_thread_overrides_open",
            Scene::NewThreadFormFilled => "new_thread_form_filled",
            Scene::ThreadPrefilling => "thread_prefilling",
            Scene::ThreadWithDraft => "thread_with_draft",
            Scene::ThreadWithImages => "thread_with_images",
            Scene::ThreadWithSetup => "thread_with_setup",
            Scene::ThreadWithDiff => "thread_with_diff",
            Scene::ThreadWithProvenance => "thread_with_provenance",
            Scene::ThreadWithFailure => "thread_with_failure",
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
            Scene::BehaviorEditorHydrated => "behavior_editor_hydrated",
            Scene::BehaviorEditorTriggerKindOpen => "behavior_editor_trigger_kind_open",
            Scene::PodEditorHydrated => "pod_editor_hydrated",
            Scene::PodEditorDefaultsTab => "pod_editor_defaults_tab",
            Scene::PodEditorGeneralTab => "pod_editor_general_tab",
            Scene::PodEditorAllowTab => "pod_editor_allow_tab",
            Scene::BehaviorEditorThreadTab => "behavior_editor_thread_tab",
            Scene::BehaviorEditorScopeTab => "behavior_editor_scope_tab",
            Scene::BehaviorEditorRetentionTab => "behavior_editor_retention_tab",
            Scene::BehaviorEditorSystemPromptTab => "behavior_editor_system_prompt_tab",
            Scene::BehaviorEditorRawTab => "behavior_editor_raw_tab",
            Scene::FileTreeBehaviorPromptDeepLink => "file_tree_behavior_prompt_deep_link",
            Scene::ForkModalOpen => "fork_modal_open",
            Scene::LightboxOpen => "lightbox_open",
            Scene::SudoPending => "sudo_pending",
            Scene::JsonViewerOpen => "json_viewer_open",
            Scene::FileViewerEditable => "file_viewer_editable",
            Scene::FileViewerReadOnly => "file_viewer_readonly",
            Scene::FileTreeOpen => "file_tree_open",
            Scene::SettingsBackends => "settings_backends",
            Scene::SettingsServerConfig => "settings_server_config",
            Scene::SettingsHostEnv => "settings_host_env",
            Scene::ThreadInspectorOpen => "thread_inspector_open",
            Scene::BucketsModalCatalog => "buckets_modal_catalog",
            Scene::BucketsModalSearchResults => "buckets_modal_search_results",
            Scene::BucketsModalCreateForm => "buckets_modal_create_form",
            Scene::BucketsModalCreateFormTracked => "buckets_modal_create_form_tracked",
            Scene::SettingsCodexRotate => "settings_codex_rotate",
            Scene::SettingsSharedMcpList => "settings_shared_mcp_list",
            Scene::SettingsSharedMcpEditorAdd => "settings_shared_mcp_editor_add",
            Scene::SettingsSharedMcpEditorEdit => "settings_shared_mcp_editor_edit",
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
            | Scene::ThreadWithProvenance
            | Scene::ThreadWithFailure => vec!["thread:t-1"],
            // Inspector scene: select the thread, then click the
            // toolbar info button so the inspector panel paints.
            Scene::ThreadInspectorOpen => vec!["thread:t-1", "chat:inspector-toggle"],
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
            // Open the separate overrides modal in its inherited
            // state so inactive override affordances stay covered.
            Scene::NewThreadOverridesInheritedOpen => vec!["new-thread:overrides:open"],
            // Open the separate overrides modal and enable each
            // optional group so the dump covers the detailed widgets,
            // not just the inherited summary rows.
            Scene::NewThreadOverridesOpen => vec![
                "new-thread:overrides:open",
                "new-thread:overrides:max-tokens:override",
                "new-thread:overrides:max-turns:override",
                "new-thread:overrides:system-prompt:mode:radio:file",
                "new-thread:overrides:compaction:override",
                "new-thread:overrides:autoquery:override",
                "new-thread:overrides:caps:override",
                "new-thread:overrides:tool-surface:override",
            ],
            // Open the "+ New pod" dialog by clicking the sidebar
            // header's plus button. Same click path the live UI
            // takes — the modal then renders as an overlay layer.
            Scene::NewPodModalEmpty | Scene::NewPodModalNoBackends => vec!["sidebar:new-pod"],
            // Open the "+ New behavior" dialog scoped to the
            // active `default` pod. Click route carries the pod
            // id as suffix.
            Scene::NewBehaviorModalEmpty => vec!["sidebar:new-behavior:default"],
            // Expand the architect behavior row, then click Edit
            // to open the editor sheet. The pre-seeded
            // `BehaviorSnapshot` correlation matches the first
            // outgoing `GetBehavior` correlation (`aui-1`), so the
            // form hydrates inside this same render frame rather
            // than rendering the loading placeholder.
            Scene::BehaviorEditorHydrated => vec![
                "behavior-row:default:architect",
                "behavior-edit:default:architect",
            ],
            // Same as above plus a click on the trigger-kind
            // `select_trigger`, which classify_select_event maps
            // to `Toggle` and our handler flips
            // `editor.trigger_kind_open`.
            Scene::BehaviorEditorTriggerKindOpen => vec![
                "behavior-row:default:architect",
                "behavior-edit:default:architect",
                "behavior-editor:trigger-kind",
            ],
            // Open the editor, then click the Thread tab trigger.
            // The architect mock has a defaulted BehaviorThreadOverride
            // so every override row renders in the inherit-only state.
            Scene::BehaviorEditorThreadTab => vec![
                "behavior-row:default:architect",
                "behavior-edit:default:architect",
                "behavior-editor:tabs:tab:thread",
            ],
            // Same as above plus a click on the Scope tab.
            // Architect mock has BehaviorScope::default() so every
            // row renders in the inherit-only shape — the multi-
            // checks / cap pickers don't show.
            Scene::BehaviorEditorScopeTab => vec![
                "behavior-row:default:architect",
                "behavior-edit:default:architect",
                "behavior-editor:tabs:tab:scope",
            ],
            // Same as above plus a click on the Retention tab.
            // Architect mock has RetentionPolicy::Keep so the days
            // numeric input doesn't render — the tab shows only
            // the kind picker.
            Scene::BehaviorEditorRetentionTab => vec![
                "behavior-row:default:architect",
                "behavior-edit:default:architect",
                "behavior-editor:tabs:tab:retention",
            ],
            // Same as above plus a click on the System tab. The
            // architect mock has cfg.thread.system_prompt = None
            // so the body renders the inherit-only paragraph.
            Scene::BehaviorEditorSystemPromptTab => vec![
                "behavior-row:default:architect",
                "behavior-edit:default:architect",
                "behavior-editor:tabs:tab:system_prompt",
            ],
            // Same as above plus a click on the Raw tab. The
            // architect mock's parsed config gets serialized into
            // `working_toml` on hydrate, so the textarea shows the
            // round-tripped TOML.
            Scene::BehaviorEditorRawTab => vec![
                "behavior-row:default:architect",
                "behavior-edit:default:architect",
                "behavior-editor:tabs:tab:raw",
            ],
            // Open the file tree, drill down to
            // `behaviors/architect/prompt.md`, and click it. The
            // file-tree dispatch routes the BehaviorPrompt variant
            // to `open_behavior_editor_on_tab(..., Prompt)`, so the
            // resulting editor renders with the Prompt tab active —
            // the deep-link behavior the egui sibling shipped.
            Scene::FileTreeBehaviorPromptDeepLink => vec![
                "sidebar:pod-files",
                "file-tree:dir:default:behaviors",
                "file-tree:dir:default:behaviors/architect",
                "file-tree:file:default:behaviors/architect/prompt.md",
            ],
            // Click the gear icon — sidebar header's pod-settings
            // affordance. Renders only when `pod_tab.is_some()`,
            // which it is here (PodList seeded a default).
            Scene::PodEditorHydrated => vec!["sidebar:pod-settings"],
            // Same as above plus a click on the Defaults tab
            // trigger. The dump loop drains inbound twice (once
            // before clicks, once after) so the `PodSnapshot` reply
            // synthesized by this scene's SendFn lands before we
            // touch the tab.
            Scene::PodEditorDefaultsTab => {
                vec!["sidebar:pod-settings", "pod-editor:tabs:tab:defaults"]
            }
            Scene::PodEditorGeneralTab => {
                vec!["sidebar:pod-settings", "pod-editor:tabs:tab:general"]
            }
            Scene::PodEditorAllowTab => vec!["sidebar:pod-settings", "pod-editor:tabs:tab:allow"],
            // Select the thread, then click the per-row fork
            // affordance for the first User message. `mock_snapshot`
            // pushes an empty system_text at msg_index=0 first, so
            // the first user message lands at msg_index=1. Synthetic
            // clicks route by key alone — they don't go through
            // hit-test, so the hover-conditional render of the
            // affordance doesn't matter for the on_event dispatch.
            Scene::ForkModalOpen => vec!["thread:t-1", "chat:user-fork:1"],
            // Select the image-bearing thread, then click the user's
            // PNG screenshot to open the lightbox. `mock_image_snapshot`
            // pushes [system(empty, skipped), user{Text, Image(bytes),
            // Image(url)}, assistant{Text, Image}, TurnStats] — so the
            // user's decoded PNG lands at display-item idx 1.
            Scene::LightboxOpen => vec!["thread:t-1", "chat:image-lightbox:1"],
            // Select the thread; the sudo wire seed lands during
            // before_build, so the banner is already rendered by the
            // time clicks finish. No further interaction needed —
            // this scene just snapshots the pending-approval shape.
            Scene::SudoPending => vec!["thread:t-1"],
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
            // Expand the `behaviors/` subdir so the file-tree scene
            // shows both the root entries and one expanded child
            // level (where the per-behavior config + prompt sit).
            // The SendFn answers the resulting `ListPodDir`, and
            // the second `before_build` drains the response into
            // `pod_files`.
            Scene::FileTreeOpen => vec!["file-tree:dir:default:behaviors"],
            // Settings modal Server-config scene: switch to the
            // Server-config tab so the lazy `FetchServerConfig`
            // fires. The per-scene SendFn answers with mock TOML;
            // the second `before_build` hydrates the editor.
            Scene::SettingsServerConfig => vec!["settings:tabs:tab:server-config"],
            Scene::SettingsHostEnv => vec!["settings:tabs:tab:host-env"],
            // Buckets-modal catalog scene: arm delete on the third
            // (ready) bucket so the destructive Confirm + Cancel
            // pair renders. The pre-seeded BucketBuildProgress
            // already paints the second row's spinner inline.
            Scene::BucketsModalCatalog => vec!["buckets:delete:__server__:wiki-en"],
            // Buckets-modal search-results scene: the seed already
            // primed the picker + query buffer; click Submit to
            // fire the QueryBuckets, then expand the first hit.
            Scene::BucketsModalSearchResults => vec![
                "buckets:search:submit",
                "buckets:search:hit:0a1b2c3d4e5f6789a0b1c2d3e4f56789",
            ],
            // Settings Codex rotate: click the Rotate button on the
            // pre-seeded `chatgpt_subscription` backend so the
            // sub-form opens.
            Scene::SettingsCodexRotate => vec!["settings:codex-rotate:open:codex"],
            // Settings Shared MCP list: switch to the tab, then arm
            // the third row's Remove so the Confirm / Cancel pair
            // renders.
            Scene::SettingsSharedMcpList => vec![
                "settings:tabs:tab:shared-mcp",
                "settings:shared-mcp:remove:legacy",
            ],
            // Shared MCP editor (Add): switch to the tab, then click
            // "+ Add host" to open the sub-form fresh.
            Scene::SettingsSharedMcpEditorAdd => {
                vec!["settings:tabs:tab:shared-mcp", "settings:shared-mcp:add"]
            }
            // Shared MCP editor (Edit): switch to the tab, then
            // click Edit on the bearer-auth host so the sub-form
            // opens with that row's values pre-populated.
            Scene::SettingsSharedMcpEditorEdit => vec![
                "settings:tabs:tab:shared-mcp",
                "settings:shared-mcp:edit:slack",
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
    // Outbound: most scenes don't observe the wire side and use a
    // pure no-op SendFn. Wire-response-driven scenes — currently
    // just the behavior-editor pair — capture the inbound queue and
    // synthesize a server reply for the request the click loop
    // emits, so the second `before_build` drain in the dump loop
    // sees the matching response. Captures here are per-scene
    // because the response's correlation has to match the request's
    // correlation, and that pairing is scene-specific.
    let send_fn: SendFn = match scene {
        Scene::BehaviorEditorHydrated
        | Scene::BehaviorEditorTriggerKindOpen
        | Scene::BehaviorEditorThreadTab
        | Scene::BehaviorEditorScopeTab
        | Scene::BehaviorEditorRetentionTab
        | Scene::BehaviorEditorSystemPromptTab
        | Scene::BehaviorEditorRawTab
        | Scene::FileTreeBehaviorPromptDeepLink => {
            let queue = inbound.clone();
            Box::new(move |msg| match msg {
                // The behavior editor opens with a `GetBehavior` /
                // `GetPod` pair (parallel correlations); the dump
                // synthesizes both replies so the Thread tab's
                // host_env / mcp_hosts override pickers see a
                // populated `pod_config` slot.
                ClientToServer::GetBehavior {
                    correlation_id,
                    pod_id,
                    behavior_id,
                } if pod_id == "default" && behavior_id == "architect" => {
                    queue.borrow_mut().push_back(InboundEvent::Wire(
                        ServerToClient::BehaviorSnapshot {
                            correlation_id,
                            snapshot: mock_architect_snapshot(),
                        },
                    ));
                }
                ClientToServer::GetPod {
                    correlation_id,
                    pod_id,
                } if pod_id == "default" => {
                    queue
                        .borrow_mut()
                        .push_back(InboundEvent::Wire(ServerToClient::PodSnapshot {
                            correlation_id,
                            snapshot: mock_default_pod_snapshot(),
                        }));
                }
                // FileTreeBehaviorPromptDeepLink also drives the
                // file-tree dispatch through `ListPodDir`s — same
                // mock entries shape as `FileTreeOpen`, restricted to
                // the path that leads to `behaviors/architect/prompt.md`.
                ClientToServer::ListPodDir {
                    correlation_id,
                    pod_id,
                    path,
                } if pod_id == "default" => {
                    let p = path.unwrap_or_default();
                    let entries: Vec<FsEntry> = match p.as_str() {
                        "" => vec![
                            FsEntry {
                                name: "pod.toml".into(),
                                is_dir: false,
                                size: 320,
                                readonly: false,
                            },
                            FsEntry {
                                name: "behaviors".into(),
                                is_dir: true,
                                size: 0,
                                readonly: false,
                            },
                        ],
                        "behaviors" => vec![FsEntry {
                            name: "architect".into(),
                            is_dir: true,
                            size: 0,
                            readonly: false,
                        }],
                        "behaviors/architect" => vec![
                            FsEntry {
                                name: "behavior.toml".into(),
                                is_dir: false,
                                size: 480,
                                readonly: false,
                            },
                            FsEntry {
                                name: "prompt.md".into(),
                                is_dir: false,
                                size: 760,
                                readonly: false,
                            },
                        ],
                        _ => Vec::new(),
                    };
                    queue.borrow_mut().push_back(InboundEvent::Wire(
                        ServerToClient::PodDirListing {
                            correlation_id,
                            pod_id,
                            path: p,
                            entries,
                        },
                    ));
                }
                _ => {}
            })
        }
        Scene::PodEditorHydrated
        | Scene::PodEditorDefaultsTab
        | Scene::PodEditorGeneralTab
        | Scene::PodEditorAllowTab
        | Scene::NewThreadOverridesInheritedOpen
        | Scene::NewThreadOverridesOpen => {
            let queue = inbound.clone();
            Box::new(move |msg| {
                if let ClientToServer::GetPod {
                    correlation_id,
                    pod_id,
                } = msg
                    && pod_id == "default"
                {
                    queue
                        .borrow_mut()
                        .push_back(InboundEvent::Wire(ServerToClient::PodSnapshot {
                            correlation_id,
                            snapshot: mock_default_pod_snapshot(),
                        }));
                }
            })
        }
        Scene::SettingsServerConfig => {
            // Answer the lazy `FetchServerConfig` with a small mock
            // TOML so the text_area paints content rather than the
            // loading placeholder.
            let queue = inbound.clone();
            Box::new(move |msg| {
                if let ClientToServer::FetchServerConfig { correlation_id } = msg {
                    let toml_text = "[server]\n\
                                     host = \"0.0.0.0\"\n\
                                     port = 8080\n\
                                     \n\
                                     [backends.anthropic-prod]\n\
                                     kind = \"anthropic\"\n\
                                     auth = \"api_key\"\n"
                        .to_string();
                    queue.borrow_mut().push_back(InboundEvent::Wire(
                        ServerToClient::ServerConfigFetched {
                            toml_text,
                            correlation_id,
                        },
                    ));
                }
            })
        }
        Scene::BucketsModalSearchResults => {
            // Answer `QueryBuckets` with two mock hits — one with a
            // proper source_id (rendered with a colored title),
            // one with an empty source_id (rendered muted with a
            // chunk-id fallback).
            let queue = inbound.clone();
            Box::new(move |msg| {
                if let ClientToServer::QueryBuckets {
                    correlation_id,
                    query,
                    ..
                } = msg
                {
                    use whisper_agent_protocol::QueryHit;
                    let hits = vec![
                        QueryHit {
                            bucket_id: "wiki-en".into(),
                            chunk_id: "0a1b2c3d4e5f6789a0b1c2d3e4f56789".into(),
                            chunk_text: "Knowledge graphs encode entities and \
                                         relationships as nodes and edges. The \
                                         original graph database concept dates \
                                         from work at Bell Labs in the late \
                                         1980s; the term \"knowledge graph\" was \
                                         later popularized by Google's 2012 \
                                         deployment over the Knowledge Vault."
                                .into(),
                            source_path: "dense".into(),
                            source_score: 0.847,
                            rerank_score: 0.912,
                            source_id: "Knowledge graph".into(),
                            source_locator: Some("chars 1024-1820".into()),
                        },
                        QueryHit {
                            bucket_id: "wiki-en".into(),
                            chunk_id: "ff00aa11bb22cc33dd44ee55ff66aa77".into(),
                            chunk_text: "Vector embeddings are continuous \
                                         dense representations of categorical \
                                         data, typically learned by \
                                         contrastive objectives over large \
                                         text corpora."
                                .into(),
                            source_path: "sparse".into(),
                            source_score: 1.42,
                            rerank_score: 0.687,
                            source_id: String::new(),
                            source_locator: None,
                        },
                    ];
                    queue.borrow_mut().push_back(InboundEvent::Wire(
                        ServerToClient::QueryResults {
                            correlation_id,
                            query,
                            hits,
                        },
                    ));
                }
            })
        }
        Scene::FileTreeOpen => {
            // Answer `ListPodDir` for the pod root and the expanded
            // `behaviors/` subdir with mock entries that cover every
            // dispatch shape: `pod.toml` (PodConfig), `behaviors/`
            // (dir), `threads/t-mock.json` (JsonViewer), `notes.md`
            // (TextEditor), and the per-behavior config + prompt
            // inside `behaviors/architect/`.
            let queue = inbound.clone();
            Box::new(move |msg| {
                if let ClientToServer::ListPodDir {
                    correlation_id,
                    pod_id,
                    path,
                } = msg
                    && pod_id == "default"
                {
                    let p = path.unwrap_or_default();
                    let entries: Vec<FsEntry> = match p.as_str() {
                        "" => vec![
                            FsEntry {
                                name: "pod.toml".into(),
                                is_dir: false,
                                size: 320,
                                readonly: false,
                            },
                            FsEntry {
                                name: "behaviors".into(),
                                is_dir: true,
                                size: 0,
                                readonly: false,
                            },
                            FsEntry {
                                name: "threads".into(),
                                is_dir: true,
                                size: 0,
                                readonly: false,
                            },
                            FsEntry {
                                name: "notes.md".into(),
                                is_dir: false,
                                size: 240,
                                readonly: false,
                            },
                            FsEntry {
                                name: "pod_state.json".into(),
                                is_dir: false,
                                size: 1024,
                                readonly: true,
                            },
                        ],
                        "behaviors" => vec![FsEntry {
                            name: "architect".into(),
                            is_dir: true,
                            size: 0,
                            readonly: false,
                        }],
                        "behaviors/architect" => vec![
                            FsEntry {
                                name: "behavior.toml".into(),
                                is_dir: false,
                                size: 480,
                                readonly: false,
                            },
                            FsEntry {
                                name: "prompt.md".into(),
                                is_dir: false,
                                size: 760,
                                readonly: false,
                            },
                            FsEntry {
                                name: "state.json".into(),
                                is_dir: false,
                                size: 256,
                                readonly: true,
                            },
                        ],
                        _ => Vec::new(),
                    };
                    queue.borrow_mut().push_back(InboundEvent::Wire(
                        ServerToClient::PodDirListing {
                            correlation_id,
                            pod_id,
                            path: p,
                            entries,
                        },
                    ));
                }
            })
        }
        Scene::FileViewerEditable | Scene::FileViewerReadOnly => {
            // Synthesize a `PodFileContent` reply for the
            // `ReadPodFile` that `open_file_viewer` fires below.
            // The editable variant lets Save / Revert paint as
            // disabled (no edits yet); the read-only variant
            // collapses the footer to Close only.
            let readonly = matches!(scene, Scene::FileViewerReadOnly);
            let queue = inbound.clone();
            Box::new(move |msg| {
                if let ClientToServer::ReadPodFile {
                    correlation_id,
                    pod_id,
                    path,
                } = msg
                    && pod_id == "default"
                    && path == "notes.md"
                {
                    let content = "# Notes\n\nA short markdown body so the \
                                   text_area paints content rather than the \
                                   placeholder ghost.\n"
                        .to_string();
                    queue.borrow_mut().push_back(InboundEvent::Wire(
                        ServerToClient::PodFileContent {
                            correlation_id,
                            pod_id,
                            path,
                            content,
                            readonly,
                        },
                    ));
                }
            })
        }
        Scene::JsonViewerOpen => {
            // Synthesize a `PodFileContent` reply for the
            // `ReadPodFile` that `open_json_viewer` fires below.
            // The content is mixed-shape mock JSON (scalars,
            // nested object, array of objects) so every arm of
            // `render_json_node` paints in this one scene.
            let queue = inbound.clone();
            Box::new(move |msg| {
                if let ClientToServer::ReadPodFile {
                    correlation_id,
                    pod_id,
                    path,
                } = msg
                    && pod_id == "default"
                    && path == "threads/t-mock.json"
                {
                    let content = serde_json::json!({
                        "thread_id": "t-mock",
                        "title": "Investigate failing test",
                        "state": "Idle",
                        "summary": "Looking into the failure — the test expects a sorted result but the underlying query returns rows in insertion order. The fix is either to add an ORDER BY clause or to sort client-side before comparing. I'll go with ORDER BY so the contract is enforced at the data layer rather than relying on every call site to remember.",
                        "messages": [
                            { "role": "user", "text": "Why does test_foo fail?" },
                            { "role": "assistant", "text": "Looking…" },
                        ],
                        "bindings": {
                            "backend": "anthropic-prod",
                            "model": "claude-opus-4-7",
                        },
                        "archived": false,
                        "draft": null,
                    });
                    let content_str =
                        serde_json::to_string_pretty(&content).expect("serialize mock JSON");
                    queue.borrow_mut().push_back(InboundEvent::Wire(
                        ServerToClient::PodFileContent {
                            correlation_id,
                            pod_id,
                            path,
                            content: content_str,
                            readonly: true,
                        },
                    ));
                }
            })
        }
        _ => Box::new(|_msg| {}),
    };
    let mut app = ChatApp::new(inbound.clone(), send_fn);

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
        | Scene::ThreadWithProvenance
        | Scene::ThreadWithFailure
        | Scene::LightboxOpen
        | Scene::SudoPending
        | Scene::ThreadInspectorOpen => {
            q.push_back(InboundEvent::ConnectionOpened);
            q.push_back(InboundEvent::Wire(ServerToClient::PodList {
                correlation_id: None,
                pods: mock_pods(),
                default_pod_id: "default".into(),
            }));
            // Provenance / failure scenes swap in tweaked thread
            // lists so the relevant summary fields drive the pane
            // header (provenance chips) and the failure banner's
            // state gate (`state == Failed`). Other scenes keep
            // `mock_threads()` for visual stability.
            let threads = if matches!(scene, Scene::ThreadWithProvenance) {
                mock_provenance_threads()
            } else if matches!(scene, Scene::ThreadWithFailure) {
                mock_failure_threads()
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
            if matches!(scene, Scene::ThreadInspectorOpen) {
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_inspector_snapshot(),
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
            if matches!(scene, Scene::ThreadWithImages | Scene::LightboxOpen) {
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
            if matches!(scene, Scene::ThreadWithFailure) {
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_failure_snapshot(),
                }));
            }
            if matches!(scene, Scene::SudoPending) {
                // Bare snapshot first so the chat pane has something
                // to render under the banner; then a synthetic sudo
                // request the model would have emitted mid-turn.
                q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                    thread_id: "t-1".into(),
                    snapshot: mock_snapshot(),
                }));
                q.push_back(InboundEvent::Wire(ServerToClient::SudoRequested {
                    function_id: 7,
                    thread_id: "t-1".into(),
                    tool_name: "bash".into(),
                    args: serde_json::json!({
                        "command": "rm -rf node_modules && npm ci",
                        "cwd": "/repo",
                    }),
                    reason: "Need to wipe and reinstall to clear the corrupted lockfile state \
                         before retrying the failing test."
                        .into(),
                }));
            }
        }
        Scene::NewThreadFormReady
        | Scene::NewThreadFormBackendOpen
        | Scene::NewThreadOverridesInheritedOpen
        | Scene::NewThreadOverridesOpen
        | Scene::NewThreadFormFilled => {
            // Same baseline as `PopulatedNoSelection` — connection,
            // pods, and an empty thread list — plus `BackendsList`
            // and `BucketsList` so picker menus and override bucket
            // groups have options. The `Filled` scene also pre-seeds
            // a `ModelsList` for the backend the
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
            q.push_back(InboundEvent::Wire(ServerToClient::BucketsList {
                correlation_id: None,
                buckets: mock_buckets(),
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
        Scene::BehaviorEditorHydrated
        | Scene::BehaviorEditorTriggerKindOpen
        | Scene::BehaviorEditorThreadTab
        | Scene::BehaviorEditorScopeTab
        | Scene::BehaviorEditorRetentionTab
        | Scene::BehaviorEditorSystemPromptTab
        | Scene::BehaviorEditorRawTab => {
            // Same baseline as `SidebarBehaviorsExpanded` — pods +
            // threads + behaviors registry — so the click loop can
            // expand the architect row and click Edit. The
            // matching `BehaviorSnapshot` reply is synthesized by
            // the per-scene SendFn (see `build_app`'s match) when
            // the click loop fires `GetBehavior`; the second
            // `before_build` drain after clicks then hydrates the
            // editor. Order matters: we don't pre-queue the
            // snapshot here because the editor doesn't exist at
            // first-drain time, and the correlation id wouldn't be
            // known yet anyway.
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
        Scene::PodEditorHydrated
        | Scene::PodEditorDefaultsTab
        | Scene::PodEditorGeneralTab
        | Scene::PodEditorAllowTab => {
            // Connection + pod list (so the active-pod gear renders),
            // plus an empty thread list so the right pane is the
            // no-selection compose form (visible behind the right-
            // attached sheet). Pod settings click then fires GetPod;
            // the per-scene SendFn injects the matching PodSnapshot
            // reply, drained by the dump loop's second before_build.
            //
            // Seed BackendsList + SharedMcpHostsList + BucketsList so
            // the structured Allow tab's multi-checks have actual
            // catalog rows to render. The Defaults-tab scene also
            // benefits from BackendsList for the backend trigger's
            // post-click menu (though this scene doesn't open the
            // menu).
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
            q.push_back(InboundEvent::Wire(ServerToClient::SharedMcpHostsList {
                correlation_id: None,
                hosts: mock_shared_mcp_hosts(),
            }));
            q.push_back(InboundEvent::Wire(ServerToClient::BucketsList {
                correlation_id: None,
                buckets: mock_buckets(),
            }));
        }
        Scene::ForkModalOpen => {
            // Same baseline as `ThreadWithMessages` — pods + thread
            // list + a snapshot for t-1 with a single User
            // message at msg_index=0. The fork-affordance click
            // pulls the seed text out of that message; the dialog
            // opens in the post-click frame.
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
            q.push_back(InboundEvent::Wire(ServerToClient::ThreadSnapshot {
                thread_id: "t-1".into(),
                snapshot: mock_snapshot(),
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
        Scene::JsonViewerOpen
        | Scene::FileViewerEditable
        | Scene::FileViewerReadOnly
        | Scene::FileTreeOpen
        | Scene::FileTreeBehaviorPromptDeepLink
        | Scene::SettingsBackends
        | Scene::SettingsServerConfig
        | Scene::SettingsHostEnv
        | Scene::SettingsCodexRotate
        | Scene::SettingsSharedMcpList
        | Scene::SettingsSharedMcpEditorAdd
        | Scene::SettingsSharedMcpEditorEdit
        | Scene::BucketsModalCatalog
        | Scene::BucketsModalSearchResults
        | Scene::BucketsModalCreateForm
        | Scene::BucketsModalCreateFormTracked => {
            // Populated baseline so the dialog overlays a non-empty
            // sidebar / pane (same idea as the modal scenes above).
            // The viewer itself is opened after the queue borrow
            // ends — the per-scene SendFn synthesizes the matching
            // `PodFileContent` reply onto this same queue. The outer
            // drain in `main` then hydrates the modal before build.
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
            if matches!(
                scene,
                Scene::SettingsBackends
                    | Scene::SettingsServerConfig
                    | Scene::SettingsHostEnv
                    | Scene::SettingsCodexRotate
                    | Scene::SettingsSharedMcpList
                    | Scene::SettingsSharedMcpEditorAdd
                    | Scene::SettingsSharedMcpEditorEdit
            ) {
                // Backends tab reads from the server-known catalog;
                // seed it so the per-backend cards have content
                // even on the Server-config tab variant (the Backends
                // tab is the modal's default). The Codex rotate
                // scene needs a `chatgpt_subscription` row so the
                // Rotate button appears.
                q.push_back(InboundEvent::Wire(ServerToClient::BackendsList {
                    correlation_id: None,
                    backends: if matches!(scene, Scene::SettingsCodexRotate) {
                        mock_backends_with_codex()
                    } else {
                        mock_backends()
                    },
                }));
            }
            if matches!(scene, Scene::SettingsHostEnv) {
                q.push_back(InboundEvent::Wire(ServerToClient::HostEnvDaemonsList {
                    correlation_id: None,
                    daemons: mock_host_env_daemons(),
                }));
            }
            if matches!(
                scene,
                Scene::SettingsSharedMcpList
                    | Scene::SettingsSharedMcpEditorAdd
                    | Scene::SettingsSharedMcpEditorEdit
            ) {
                // SharedMcp tab catalog. Three rows: anonymous
                // connected, bearer connected, oauth disconnected
                // (with a last_error so the destructive caption
                // paints).
                q.push_back(InboundEvent::Wire(ServerToClient::SharedMcpHostsList {
                    correlation_id: None,
                    hosts: mock_settings_shared_mcp_hosts(),
                }));
            }
            if matches!(
                scene,
                Scene::BucketsModalCatalog | Scene::BucketsModalSearchResults
            ) {
                // Three buckets covering each row-chrome state:
                //   - notes — managed, no slot yet (Build disabled).
                //   - design — linked, in-flight build (counters
                //     populated by the BucketBuildProgress below).
                //   - wiki-en — tracked, ready with active slot;
                //     the catalog scene arms its Delete; the
                //     search scene queries against it.
                q.push_back(InboundEvent::Wire(ServerToClient::BucketsList {
                    correlation_id: None,
                    buckets: mock_buckets_modal_catalog(),
                }));
            }
            if matches!(
                scene,
                Scene::BucketsModalCreateForm | Scene::BucketsModalCreateFormTracked
            ) {
                // No catalog needed for the create form scenes —
                // an empty buckets list collapses the catalog body
                // to a muted hint without competing for attention
                // with the form. Keep an empty `BucketsList` so
                // the `list_requested` flag flips and `ListBuckets`
                // doesn't fire on connect.
                q.push_back(InboundEvent::Wire(ServerToClient::BucketsList {
                    correlation_id: None,
                    buckets: Vec::new(),
                }));
                // Embedder picker — without this seed the form
                // falls back to the empty-text-input branch and
                // the picker doesn't paint.
                q.push_back(InboundEvent::Wire(ServerToClient::EmbeddingProvidersList {
                    correlation_id: None,
                    providers: mock_embedding_providers(),
                }));
            }
            if matches!(scene, Scene::BucketsModalCatalog) {
                // Build start + first progress tick for `design`,
                // synthesized at known correlation_id::None and
                // fields the renderer cares about. Fed *into* the
                // queue so the first `before_build` drain in main
                // hydrates `build_progress` before the body paints.
                q.push_back(InboundEvent::Wire(ServerToClient::BucketBuildStarted {
                    correlation_id: None,
                    bucket_id: "design".into(),
                    pod_id: None,
                    slot_id: "20261103-0001".into(),
                    started_at: Some("2026-05-10T14:00:00Z".into()),
                }));
                q.push_back(InboundEvent::Wire(ServerToClient::BucketBuildProgress {
                    bucket_id: "design".into(),
                    pod_id: None,
                    slot_id: "20261103-0001".into(),
                    phase: BucketBuildPhase::Indexing,
                    source_records: 1_240,
                    chunks: 8_973,
                    started_at: Some("2026-05-10T14:00:00Z".into()),
                    dense_inserted: None,
                    dense_total: None,
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
    if matches!(scene, Scene::JsonViewerOpen) {
        app.open_json_viewer("default".into(), "threads/t-mock.json".into());
    }
    if matches!(scene, Scene::FileViewerEditable | Scene::FileViewerReadOnly) {
        app.open_file_viewer("default".into(), "notes.md".into());
    }
    if matches!(scene, Scene::FileTreeOpen) {
        app.open_file_tree_modal("default".into());
    }
    if matches!(
        scene,
        Scene::SettingsBackends
            | Scene::SettingsServerConfig
            | Scene::SettingsHostEnv
            | Scene::SettingsCodexRotate
            | Scene::SettingsSharedMcpList
            | Scene::SettingsSharedMcpEditorAdd
            | Scene::SettingsSharedMcpEditorEdit
    ) {
        app.open_settings_modal();
    }
    if matches!(scene, Scene::BucketsModalCatalog) {
        app.open_buckets_modal();
    }
    if matches!(scene, Scene::BucketsModalSearchResults) {
        app.dev_seed_bucket_search(None, "wiki-en".into(), "knowledge graph history".into());
    }
    if matches!(scene, Scene::BucketsModalCreateForm) {
        app.dev_seed_bucket_create("linked");
    }
    if matches!(scene, Scene::BucketsModalCreateFormTracked) {
        app.dev_seed_bucket_create("tracked");
    }
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

fn mock_backends_with_codex() -> Vec<BackendSummary> {
    let mut v = mock_backends();
    v.push(BackendSummary {
        name: "codex".into(),
        kind: "chatgpt_subscription".into(),
        default_model: Some("gpt-5-codex".into()),
        auth_mode: Some("auth.json".into()),
    });
    v
}

fn mock_settings_shared_mcp_hosts() -> Vec<whisper_agent_protocol::SharedMcpHostInfo> {
    use whisper_agent_protocol::{CatalogOrigin, SharedMcpAuthPublic, SharedMcpHostInfo};
    vec![
        SharedMcpHostInfo {
            name: "fetch".into(),
            url: "http://localhost:8123/mcp".into(),
            origin: CatalogOrigin::Seeded,
            auth: SharedMcpAuthPublic::None,
            prefix: None,
            connected: true,
            last_error: String::new(),
        },
        SharedMcpHostInfo {
            name: "slack".into(),
            url: "https://slack.example.com/mcp".into(),
            origin: CatalogOrigin::Manual,
            auth: SharedMcpAuthPublic::Bearer,
            prefix: Some("slack".into()),
            connected: true,
            last_error: String::new(),
        },
        SharedMcpHostInfo {
            name: "legacy".into(),
            url: "https://legacy.example.com/mcp".into(),
            origin: CatalogOrigin::Manual,
            auth: SharedMcpAuthPublic::Oauth2 {
                issuer: "https://idp.example.com".into(),
                scope: Some("read write".into()),
            },
            prefix: Some(String::new()),
            connected: false,
            last_error: "tcp connect refused: 503".into(),
        },
    ]
}

fn mock_host_env_daemons() -> Vec<whisper_agent_protocol::HostEnvDaemonSummary> {
    vec![
        whisper_agent_protocol::HostEnvDaemonSummary {
            name: "local-dev".into(),
            admitted: true,
            connected: true,
            daemon_version: Some("0.3.19".into()),
            protocol_version: Some(2),
            spec_kinds: vec!["landlock".into(), "container".into()],
            tools: vec![
                "bash".into(),
                "read_file".into(),
                "write_file".into(),
                "list_dir".into(),
                "apply_patch".into(),
                "git_status".into(),
                "cargo_check".into(),
            ],
            max_concurrent_sessions: Some(4),
            supports_background_tasks: true,
            last_active_ms_ago: Some(3_200),
        },
        whisper_agent_protocol::HostEnvDaemonSummary {
            name: "linux-builder".into(),
            admitted: true,
            connected: false,
            daemon_version: None,
            protocol_version: None,
            spec_kinds: Vec::new(),
            tools: Vec::new(),
            max_concurrent_sessions: None,
            supports_background_tasks: false,
            last_active_ms_ago: None,
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

fn mock_embedding_providers() -> Vec<EmbeddingProviderInfo> {
    vec![
        EmbeddingProviderInfo {
            name: "tei-bge-small".into(),
            kind: "tei".into(),
            endpoint: "http://localhost:8081".into(),
            auth_mode: None,
        },
        EmbeddingProviderInfo {
            name: "tei-jina-base".into(),
            kind: "tei".into(),
            endpoint: "http://localhost:8082".into(),
            auth_mode: None,
        },
    ]
}

fn mock_buckets_modal_catalog() -> Vec<BucketSummary> {
    vec![
        BucketSummary {
            id: "notes".into(),
            scope: "server".into(),
            pod_id: None,
            name: "Field notes".into(),
            description: Some("Managed bucket — corpus appended via the agent.".into()),
            source_kind: "managed".into(),
            source_detail: None,
            embedder_provider: "voyage-3-lite".into(),
            dense_enabled: true,
            sparse_enabled: false,
            created_at: "2026-04-12T09:30:00Z".into(),
            active_slot: None,
        },
        BucketSummary {
            id: "design".into(),
            scope: "server".into(),
            pod_id: None,
            name: "Design docs".into(),
            description: Some("Linked bucket — points at /var/whisper/docs/design.".into()),
            source_kind: "linked".into(),
            source_detail: Some("/var/whisper/docs/design".into()),
            embedder_provider: "voyage-3-lite".into(),
            dense_enabled: true,
            sparse_enabled: true,
            created_at: "2026-04-15T10:00:00Z".into(),
            active_slot: None,
        },
        BucketSummary {
            id: "wiki-en".into(),
            scope: "server".into(),
            pod_id: None,
            name: "Wikipedia (en)".into(),
            description: Some(
                "Tracked bucket — monthly rebuild from enwiki dumps + daily deltas.".into(),
            ),
            source_kind: "tracked".into(),
            source_detail: Some("wikipedia (en)".into()),
            embedder_provider: "qwen3-embedding-0.6b".into(),
            dense_enabled: true,
            sparse_enabled: true,
            created_at: "2026-02-01T00:00:00Z".into(),
            active_slot: Some(ActiveSlotSummary {
                slot_id: "20260301-0001-abcd".into(),
                state: SlotStateLabel::Ready,
                embedder_model: "Qwen/Qwen3-Embedding-0.6B".into(),
                dimension: 1024,
                chunk_count: 142_873_204,
                vector_count: 142_873_204,
                disk_size_bytes: 312_452_096_000,
                created_at: "2026-03-01T00:00:00Z".into(),
                built_at: Some("2026-03-14T18:42:00Z".into()),
            }),
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

/// Mark t-1 as failed so the pane's failure-banner gate
/// (`state == Failed && view.failure.is_some()`) trips. Body
/// arrives via [`mock_failure_snapshot`] which carries the
/// `snapshot.failure` text the banner shows.
fn mock_failure_threads() -> Vec<ThreadSummary> {
    let mut t = mock_threads();
    t[0].state = ThreadStateLabel::Failed;
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

/// Hydrated `BehaviorSnapshot` for the architect cron behavior the
/// editor scenes open. Mirrors what `GetBehavior` returns from a
/// running server: a parsed `BehaviorConfig` with a cron trigger,
/// a non-empty `prompt.md` body, and a normal `BehaviorState`. No
/// `system_prompt.md` (override is dormant), no `load_error`.
fn mock_architect_snapshot() -> whisper_agent_protocol::BehaviorSnapshot {
    use whisper_agent_protocol::{
        BehaviorConfig, BehaviorScope, BehaviorSnapshot, BehaviorState, BehaviorThreadOverride,
        CatchUp, Overlap, RetentionPolicy, TriggerSpec,
    };
    BehaviorSnapshot {
        behavior_id: "architect".into(),
        pod_id: "default".into(),
        config: Some(BehaviorConfig {
            name: "architect".into(),
            description: Some(
                "Meta-agent. Observes agent performance and modifies behaviors \
                 to optimize the system loop."
                    .into(),
            ),
            trigger: TriggerSpec::Cron {
                schedule: "0 2 * * *".into(),
                timezone: "UTC".into(),
                overlap: Overlap::Skip,
                catch_up: CatchUp::One,
            },
            thread: BehaviorThreadOverride::default(),
            on_completion: RetentionPolicy::default(),
            scope: BehaviorScope::default(),
        }),
        toml_text: String::new(),
        prompt: "Review the last 24 hours of agent threads. Look for \
                 repeated failure patterns, then propose targeted edits \
                 to the affected behaviors.\n\npayload: {{payload}}\n"
            .into(),
        system_prompt: None,
        state: BehaviorState {
            enabled: true,
            run_count: 12,
            last_fired_at: Some("2026-05-09T02:00:00Z".into()),
            last_thread_id: Some("task-architect-001".into()),
            last_outcome: None,
            queued_payload: None,
        },
        load_error: None,
    }
}

/// Hydrated `PodSnapshot` for the `default` pod that the pod-editor
/// scene opens. Mirrors what `GetPod` returns from a running server:
/// a parsed `PodConfig` plus the raw `pod.toml` text the sheet's
/// `text_area` renders. The sheet only reads `toml_text`, so the
/// `config` stub stays minimal — the on-wire surface is what the
/// scene exercises visually.
fn mock_default_pod_snapshot() -> whisper_agent_protocol::PodSnapshot {
    use whisper_agent_protocol::{
        AllowMap, CompactionConfig, NamedHostEnv, PodAllow, PodConfig, PodLimits, PodSnapshot,
        ThreadDefaults,
        sandbox::{AccessMode, HostEnvSpec, Mount, NetworkPolicy, PathAccess, ResourceLimits},
    };
    let toml_text = r#"name = "default"
description = "the synthesized server-default pod"
created_at = "2026-05-01T00:00:00Z"

[allow]
backends = ["anthropic-prod", "openai-team"]
mcp_hosts = ["filesystem", "git"]
knowledge_buckets = ["wiki"]

[[allow.host_env]]
name = "main"
provider = "shell"
kind = "landlock"
allowed_paths = ["/home/christian/workspace:rw", "/tmp:rw"]
network = "isolated"

[[allow.host_env]]
name = "browser"
provider = "containerd"
kind = "container"
image = "ghcr.io/example/browser-agent:latest"
network = "allow_list"
network_hosts = ["example.com", "docs.rs"]

[[allow.host_env.mounts]]
host = "/home/christian/Downloads"
guest = "/downloads"
mode = "ro"

[allow.tools]
default = "allow"

[thread_defaults]
backend = "anthropic-prod"
model = "claude-opus-4-7"
system_prompt_file = "system_prompt.md"
max_tokens = 16384
max_turns = 30
host_env = ["main"]
mcp_hosts = ["filesystem", "git"]
"#
    .to_string();
    let config = PodConfig {
        name: "default".into(),
        description: Some("the synthesized server-default pod".into()),
        created_at: "2026-05-01T00:00:00Z".into(),
        allow: PodAllow {
            backends: vec!["anthropic-prod".into(), "openai-team".into()],
            mcp_hosts: vec!["filesystem".into(), "git".into()],
            host_env: vec![
                NamedHostEnv {
                    name: "main".into(),
                    provider: "shell".into(),
                    spec: HostEnvSpec::Landlock {
                        allowed_paths: vec![
                            PathAccess {
                                path: "/home/christian/workspace".into(),
                                mode: AccessMode::ReadWrite,
                            },
                            PathAccess {
                                path: "/tmp".into(),
                                mode: AccessMode::ReadWrite,
                            },
                        ],
                        network: NetworkPolicy::Isolated,
                    },
                },
                NamedHostEnv {
                    name: "browser".into(),
                    provider: "containerd".into(),
                    spec: HostEnvSpec::Container {
                        image: "ghcr.io/example/browser-agent:latest".into(),
                        mounts: vec![Mount {
                            host: "/home/christian/Downloads".into(),
                            guest: "/downloads".into(),
                            mode: AccessMode::ReadOnly,
                        }],
                        network: NetworkPolicy::AllowList {
                            hosts: vec!["example.com".into(), "docs.rs".into()],
                        },
                        limits: Some(ResourceLimits {
                            cpus: Some(2),
                            memory_mb: Some(4096),
                            timeout_s: Some(900),
                        }),
                        env: Default::default(),
                    },
                },
            ],
            knowledge_buckets: vec!["wiki".into()],
            tools: AllowMap::allow_all(),
            caps: Default::default(),
        },
        thread_defaults: ThreadDefaults {
            backend: "anthropic-prod".into(),
            model: "claude-opus-4-7".into(),
            system_prompt_file: "system_prompt.md".into(),
            max_tokens: 16384,
            max_turns: 30,
            host_env: vec!["main".into()],
            mcp_hosts: vec!["filesystem".into(), "git".into()],
            compaction: CompactionConfig::default(),
            autoquery: Default::default(),
            caps: Default::default(),
            tool_surface: Default::default(),
        },
        limits: PodLimits::default(),
    };
    PodSnapshot {
        pod_id: "default".into(),
        config,
        toml_text,
        threads: Vec::new(),
        archived: false,
        behaviors: Vec::new(),
    }
}

/// Server-known shared MCP hosts for the pod-editor Allow tab dump.
/// Two entries (one connected, one with a stale auth) — enough to
/// render two adjacent toggle items and exercise the multi-check
/// layout.
fn mock_shared_mcp_hosts() -> Vec<whisper_agent_protocol::SharedMcpHostInfo> {
    use whisper_agent_protocol::{CatalogOrigin, SharedMcpAuthPublic, SharedMcpHostInfo};
    vec![
        SharedMcpHostInfo {
            name: "filesystem".into(),
            url: "http://127.0.0.1:7401/mcp".into(),
            origin: CatalogOrigin::Seeded,
            auth: SharedMcpAuthPublic::None,
            prefix: None,
            connected: true,
            last_error: String::new(),
        },
        SharedMcpHostInfo {
            name: "git".into(),
            url: "http://127.0.0.1:7402/mcp".into(),
            origin: CatalogOrigin::Seeded,
            auth: SharedMcpAuthPublic::None,
            prefix: None,
            connected: true,
            last_error: String::new(),
        },
    ]
}

/// Known knowledge buckets for config-surface bundle scenes.
fn mock_buckets() -> Vec<whisper_agent_protocol::BucketSummary> {
    use whisper_agent_protocol::BucketSummary;
    vec![
        BucketSummary {
            id: "wiki".into(),
            scope: "server".into(),
            pod_id: None,
            name: "wiki".into(),
            description: Some("English Wikipedia (2025-04 dump).".into()),
            source_kind: "stored".into(),
            source_detail: None,
            embedder_provider: "qwen3-embedding-0.6b".into(),
            dense_enabled: true,
            sparse_enabled: true,
            created_at: "2026-04-12T00:00:00Z".into(),
            active_slot: None,
        },
        BucketSummary {
            id: "rust-stdlib".into(),
            scope: "server".into(),
            pod_id: None,
            name: "rust-stdlib".into(),
            description: Some("Rust standard library docs.".into()),
            source_kind: "stored".into(),
            source_detail: None,
            embedder_provider: "qwen3-embedding-0.6b".into(),
            dense_enabled: true,
            sparse_enabled: false,
            created_at: "2026-04-20T00:00:00Z".into(),
            active_slot: None,
        },
        BucketSummary {
            id: "scratch-notes".into(),
            scope: "pod".into(),
            pod_id: Some("default".into()),
            name: "scratch-notes".into(),
            description: Some("Default pod working notes.".into()),
            source_kind: "stored".into(),
            source_detail: None,
            embedder_provider: "qwen3-embedding-0.6b".into(),
            dense_enabled: true,
            sparse_enabled: true,
            created_at: "2026-05-02T00:00:00Z".into(),
            active_slot: None,
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
            autoquery: Default::default(),
        },
        // Seed a backend so the thread-header `backend/model` chip
        // exercises the populated branch in dump scenes that ride
        // this snapshot.
        bindings: ThreadBindings {
            backend: "anthropic-prod".into(),
            ..ThreadBindings::default()
        },
        state: ThreadStateLabel::Idle,
        conversation: conv,
        // Seed cumulative usage so the header's right-aligned
        // ↑/↓/cache chip renders with realistic data instead of a
        // suppressed-zero state.
        total_usage: Usage {
            input_tokens: 1234,
            output_tokens: 287,
            cache_read_input_tokens: 800,
            cache_creation_input_tokens: 200,
        },
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

/// Snapshot for the `ThreadInspectorOpen` scene. Same conversation
/// as `mock_snapshot` but with a populated `scope` (mixed `All` /
/// `Only` shapes so every `set_or_all_label` arm renders) and a
/// `BehaviorOrigin` (cron-style fired_at plus a small JSON payload)
/// so the inspector's new Scope + Trigger origin sections paint.
fn mock_inspector_snapshot() -> ThreadSnapshot {
    use whisper_agent_protocol::permission::{
        AllowMap, BehaviorOpsCap, DispatchCap, Disposition, Escalation, PodModifyCap, SetOrAll,
    };
    use whisper_agent_protocol::{BehaviorOrigin, HostEnvBinding};

    let mut base = mock_snapshot();
    base.bindings = ThreadBindings {
        backend: "anthropic-prod".into(),
        host_env: vec![HostEnvBinding::Named {
            name: "c-dtop".into(),
            workspace_root: Some(std::path::PathBuf::from(
                "/home/user/workspace/whisper-agent",
            )),
        }],
        mcp_hosts: vec!["filesystem".into(), "memory".into()],
        ..ThreadBindings::default()
    };
    base.scope = Scope {
        backends: SetOrAll::Only {
            items: std::iter::once("anthropic-prod".to_string()).collect(),
        },
        host_envs: SetOrAll::All,
        mcp_hosts: SetOrAll::Only {
            items: ["filesystem".to_string(), "memory".to_string()]
                .into_iter()
                .collect(),
        },
        knowledge_buckets: SetOrAll::All,
        tools: {
            let mut tools = AllowMap::allow_all();
            tools
                .overrides
                .insert("shell_exec".into(), Disposition::Deny);
            tools
        },
        pod_modify: PodModifyCap::ModifyAllow,
        dispatch: DispatchCap::WithinScope,
        behaviors: BehaviorOpsCap::AuthorAny,
        escalation: Escalation::None,
    };
    base.origin = Some(BehaviorOrigin {
        behavior_id: "nightly-summary".into(),
        fired_at: "2026-05-09T03:00:00Z".into(),
        trigger_payload: serde_json::json!({
            "schedule": "0 3 * * *",
            "tz": "UTC",
            "fired_by": "scheduler",
        }),
    });
    base
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
            autoquery: Default::default(),
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

/// Snapshot for the `ThreadWithFailure` scene — one user turn,
/// the assistant's first response triggered a tool-validation
/// error mid-loop, and the integrator marked the thread as
/// `Failed` with the same message echoed onto `snapshot.failure`.
/// Mirrors the wire shape the server emits when a thread enters
/// the failed state.
fn mock_failure_snapshot() -> ThreadSnapshot {
    let mut conv = Conversation::new();
    conv.push(Message::system_text(""));
    conv.push(Message::user_text(
        "Run `validate_release` against the candidate tag and report the diff.",
    ));
    conv.push(Message {
        role: Role::Assistant,
        content: vec![ContentBlock::Text {
            text: "Pulling the candidate tag and validating now.".into(),
        }],
    });
    let mut snapshot = base_snapshot(conv, ThreadStateLabel::Failed, String::new());
    snapshot.failure = Some(
        "tool `validate_release` not in this pod's allow-list \
         (allow.tools.deny set to {validate_release}); \
         scheduler aborted the turn"
            .into(),
    );
    snapshot
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
            autoquery: Default::default(),
        },
        // Seed a backend so the thread-header `backend/model` chip
        // exercises the populated branch in dump scenes that ride
        // this snapshot.
        bindings: ThreadBindings {
            backend: "anthropic-prod".into(),
            ..ThreadBindings::default()
        },
        state: ThreadStateLabel::Idle,
        conversation: conv,
        // Seed cumulative usage so the header's right-aligned
        // ↑/↓/cache chip renders with realistic data instead of a
        // suppressed-zero state.
        total_usage: Usage {
            input_tokens: 1234,
            output_tokens: 287,
            cache_read_input_tokens: 800,
            cache_creation_input_tokens: 200,
        },
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
