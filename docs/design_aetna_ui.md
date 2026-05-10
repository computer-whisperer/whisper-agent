# Aetna UI Pivot

We're moving the chat UI off egui (`whisper-agent-webui`) onto
[aetna](https://github.com/computer-whisperer/aetna), a small declarative GPU
UI library shaped around how an LLM authors UI. This doc tracks the why, the
crate layout, the architectural choices that fall out of aetna's `App` model,
and where each stage of the port stands.

**Current entry points:**
- Native: `cargo build -p whisper-agent-desktop-aetna && ./target/.../whisper-agent-desktop-aetna --server http://127.0.0.1:8080`
- Dev all-in-one: `./scripts/dev_aetna.sh` (server + daemons + native binary)
- Bundle dump: `cargo run -p whisper-agent-aetna-ui --example dump_bundles`

The old egui stack (`whisper-agent-webui`, `whisper-agent-desktop`) stays
shipping until aetna reaches feature parity and polish. Both build cleanly
side-by-side; nothing in either crate depends on the other.

## Why aetna

The thesis is the load-bearing argument: aetna is shaped around how a *model*
authors UI, not how a human web developer does. For a project where the
loop driving the codebase is itself an LLM, that fit dominates other
trade-offs:

- **Vocabulary parity with shadcn / Radix / Tailwind.** Stock widgets
  (`sidebar`, `sidebar_menu_button`, `card_header`, `alert`, `badge`, `dialog`,
  `popover`, `tabs_list`, `text_area`, …) match the shapes the model has the
  most training-time exposure to. The same prompt that produced reasonable
  egui code produces *better* aetna code.
- **Build/on_event split is enforced.** `App::build(&self, &BuildCx) -> El`
  takes `&self`; mutations only happen in `on_event(&mut self, UiEvent)`.
  The egui webui interleaves reads and mutations freely inside
  `App::ui(&mut self, ui)` — an architectural pattern that becomes a
  refactor liability as the surface area grows. Aetna structurally prevents
  it.
- **Bundle artifacts as the agent-loop feedback channel.** A single CPU-only
  call produces SVG + tree dump + draw_op IR + lint findings + shader
  manifest per scene, with no GPU device required. The model can verify
  layout intent end-to-end without waiting for a window to come up.
- **One implementation across native + browser.** `aetna-winit-wgpu` for
  desktop, `aetna-web` (wasm + WebGPU) for the browser. Single `App` impl
  paints both, which makes the server's embedded webui bundle a
  build-time choice, not a code choice.

## Crate layout

```
crates/
  whisper-agent-aetna-ui/         ← platform-agnostic ChatApp (cdylib + rlib)
    src/
      lib.rs                        re-exports + (eventually) wasm entry
      app.rs                        ChatApp impl of aetna_core::App
    examples/
      dump_bundles.rs               scene-driven bundle export
    out/                            bundle artifacts (gitignored)

  whisper-agent-desktop-aetna/    ← native binary on aetna-winit-wgpu
    src/main.rs                     winit host + tokio + tungstenite bridge
```

Sibling to `whisper-agent-webui` / `whisper-agent-desktop`. Same workspace
membership, no new top-level deps spawned at the workspace level — both
crates pull `aetna-core`, `aetna-winit-wgpu` in via path-deps.

### Aetna as a path-dep

Aetna is under active development; we track upstream `HEAD` rather than a
crates.io release. Path-dep reaches up out of the workspace:

```toml
# crates/whisper-agent-aetna-ui/Cargo.toml
aetna-core = { path = "../../../../aetna/aetna.main/crates/aetna-core" }
```

The four `..`s reflect the user's worktree convention:
`~/workspace/{aetna,whisper-agent}/{aetna,whisper}.main/`. When aetna
stabilizes and ships to crates.io, this becomes a versioned dep + the path
gets dropped (or kept behind a `[patch]` for development).

## Architecture

### The `App` contract

```rust
impl aetna_core::App for ChatApp {
    fn before_build(&mut self) { /* drain inbound queue, mutate state */ }
    fn build(&self, _cx: &BuildCx) -> El { /* read state, return tree */ }
    fn on_event(&mut self, event: UiEvent) { /* route by key, mutate */ }
    fn selection(&self) -> Selection { self.selection.clone() }
    fn theme(&self) -> Theme { Theme::aetna_dark() }
}
```

The runtime calls these in this order per frame:

1. `before_build(&mut self)` — drain the inbound `WS event` queue, fold
   server frames into `ChatApp` state.
2. `build(&self, cx)` — pure projection of state into an `El` tree.
3. (host paints; pumps OS events)
4. `on_event(&mut self, event)` — for each routed click / activate / text
   input / hotkey, mutate state and optionally call `(self.send_fn)(...)`
   to send a wire frame back to the server.

This split is *the* load-bearing constraint of the port. Every interactive
element gets a `.key("...")` that names a route; every action is an
`event.is_click_or_activate("...")` arm in `on_event`. We never reach for
mutable state during build.

### Routing key conventions

Two-tier scheme:
- **Static keys** for singletons: `compose`, `send`. Declared as `const &str`.
- **Templated keys** for collection items: `thread:{id}`, future
  `pod:{id}`, `behavior:{pod_id}/{behavior_id}`. Parsed in `on_event` with
  `key.strip_prefix("thread:")` and friends.

Compose-target keys (`compose`, `send`) are deliberately short — there's
exactly one compose box at a time. The new-thread form (Stage 4) reuses
both keys; the same `send_compose` routes between `SendUserMessage` and
`CreateThread` based on `selected.is_some()`. The picker triggers use
`picker:{backend|model|pod}` with `select_menu` option keys carrying the
canonical `:option:{value}` suffix.

### Transport bridge

The native binary owns a tokio runtime for the WebSocket client; the UI runs
on the main thread inside winit's event loop. Two bridges connect them:

```
WS task ──► tokio mpsc ──► bouncer task ──► Mutex<Vec<InboundEvent>> staging
                                                       │
                                       drained by App::before_build
                                                       ▼
                                       Rc<RefCell<VecDeque<InboundEvent>>>
                                                       │
                                                drained per-frame
                                                       ▼
                                                  ChatApp state
```

The `Mutex<Vec>` step exists because the UI thread can't poll a
`tokio::sync::mpsc::Receiver` directly without a tokio context. The
`Rc<RefCell<VecDeque>>` is the same shape the egui webui exposes, so the
host shells (native + future wasm) hand the same `Inbound` / `SendFn`
pair to either UI crate.

Outbound is a `Box<dyn Fn(ClientToServer)>` closure — the UI calls
`(self.send_fn)(msg)`, the closure encodes CBOR + pushes onto a tokio
mpsc that the WS task drains.

### Wakeup caveat

Aetna's `aetna-winit-wgpu` host doesn't currently expose a cross-thread
"wake the event loop" primitive (no `EventLoopProxy` re-export). To keep
WS frames visible within ~one frame after they arrive, the desktop binary
runs with `HostConfig::with_redraw_interval(Duration::from_millis(33))`
(~30 Hz). Costs idle CPU; eventually we want to either:

- upstream a wakeup channel in `aetna-winit-wgpu`, or
- hold the `winit::EventLoopProxy` on the bouncer side ourselves and
  `proxy.send_event(())` on each frame.

Flagged as a stage-1 scaffold concern; not a blocker.

### Bundle export

`examples/dump_bundles.rs` produces a five-file bundle per scene:

| file | content |
|---|---|
| `<scene>.svg` | layout-accurate visual fallback |
| `<scene>.tree.txt` | semantic walk: computed rects, roles, source paths |
| `<scene>.draw_ops.txt` | the same draw-op IR the GPU runner consumes |
| `<scene>.shader_manifest.txt` | shaders the tree references + uniforms |
| `<scene>.lint.txt` | findings (raw colors, duplicate ids, overflow) |

Scenes are seeded by pushing synthetic `InboundEvent`s into the `Inbound`
queue and dispatching `UiEvent::synthetic_click` for selection — same
paths the live binary uses, so the dump reflects real state mutations
through the routing layer. Adding a new scene is a one-arm change to the
`Scene` enum + a `seed` arm in `build_app`.

The `Scene` enum is the extension point as more state becomes worth
dumping (sidebar empty / populated, chat log streaming, modal open,
behavior editor active, …).

## Visual idioms — "tailwind/shadcn for free"

Successive refactors moved the UI off hand-rolled chrome onto aetna's
purpose-built widgets. The rules of thumb:

- **Sidebar.** `sidebar([...])` for the surface, `sidebar_header([...])`
  for the title row. The active *workspace* is selected via a `tabs_list`
  pod-tab strip at the top — single active pod at a time, so the rest of
  the sidebar can spend its vertical real estate on that pod's threads
  (and eventually behaviors, dispatch nesting, filters) instead of
  competing collapsible sections. Threads render as
  `item_group([item([item_content([item_title, item_description])])])`
  rows — aetna's "object/action list row" widget gets us the right
  hover/press/focus rail + cursor + keying for free. Title sits above a
  muted second line of `state · relative_time`. Dispatch children are
  left-padded by `SPACE_3 * depth` so chains read top-down without
  inline marker glyphs on every row. Selected row uses `.current()`
  (ACCENT fill — quieter than `.primary()`).
- **Toolbar headers, not card headers, for thread identity.**
  `toolbar([toolbar_title, spacer, state_badge])` is the thread-pane
  header recipe. Cards isolate objects; the chat content frame is one
  continuous record with a header bar above it.
- **Event-log rows for the chat log, not cards per message.** Per
  upstream aetna guidance (`aetna-core/README.md` "Conversation /
  event-log row"), each message renders as a 3px role-colored gutter
  beside the message content — `row([gutter, content])`. User rows
  optionally take a faint info-tinted fill so they pop out of a long
  assistant stream. Local helper `log_row(role_color, faint_fill,
  content)` in `app.rs`.
- **Markdown for assistant text.** Assistant content goes through
  `aetna_markdown::md(text)`, which walks pulldown-cmark's event API
  into `paragraph` / `bullet_list` / `code_block` / `inlines` /
  `text(...).code()` etc. — same surface a hand-authored aetna tree
  would use. User text stays as `paragraph(text)` (we don't render
  user-typed markdown).
- **Accordion for noisy rows.** Reasoning and tool-call rows nest
  `accordion_item(group, value, label, open, [body])`. Open state
  lives on `ChatApp.open_accordions: HashSet<String>` keyed by
  `accordion_item_key(group, value)`, toggled in `on_event` by
  matching `:accordion:` in the routed key.
- **Alerts.** Connection failures surface as `alert([alert_title,
  alert_description]).warning() / .destructive()` at the top of the
  content pane. Status-tier UX, separate from the per-thread chrome.
- **Badges.** Connection chip in the sidebar header and the per-thread
  state pill (idle / working / done / failed / cancelled) are plain
  `badge(text)` with `.success() / .muted() / .warning() / .destructive()`
  variants.
- **Layout.** Always reach for `tokens::SPACE_*` for gaps and padding,
  `tokens::SIDEBAR_WIDTH` for the sidebar size, `Size::Fill(1.0)` /
  `Size::Hug` / `Size::Fixed` for sizing. No raw px literals in chrome.

When the helpers don't fit a shape (collapsible section in the sidebar,
nested sub-groups, custom row anatomy), wrap a custom composition *inside*
the helper rather than replacing it — keeps the canonical surface recipe
correct without forcing data into the helper mold. Per aetna's
`widget_kit.md`.

## Stages

Each stage is a self-contained slice with its own commit. The progression is
roughly the inverse of dependency depth: read-only shell first, then
write paths, then composition, then cross-cutting features.

### Sidebar redesign (in flight)

The egui sidebar's pod-as-collapsible-section idiom doesn't scale when
one pod dominates and others are sparse — a real-world snapshot of the
production server has 2 pods (4 threads vs. 29 threads + 4 cron
behaviors). The aetna sidebar is being redesigned around an active-pod
tab strip rather than ported piecemeal. Multi-slice rollout:

- **✅ Slice α — pod tabs + item-row threads.** `tabs_list` selector at
  the top of the sidebar; `item_group` of `item([item_content([title,
  description])])` rows for the active pod's threads. State and
  relative-activity in the muted second line. Dispatch children
  left-padded by `SPACE_3 * depth`. "Show N more" pagination at
  `SIDEBAR_THREAD_PREVIEW = 10` rows. No new wire calls.
- **✅ Slice β — behaviors first-class.** Per-pod `Behaviors (N)`
  subsection *below* the threads section, populated via
  `ListBehaviors` (deduped per pod per connection) and kept fresh
  by `BehaviorList` / `BehaviorCreated` / `BehaviorUpdated` /
  `BehaviorDeleted` / `BehaviorStateChanged`. Each row is an
  item-shaped expandable container: title is the display name
  (`⚠`-prefixed when load_error), description is `"{kind} ·
  {status}"` for healthy rows (`status` = `paused` /
  `last fired Nh` / `no runs yet`) or `"errored: {reason}"` for
  malformed configs, plus a `"{N} runs"` caption when the row has
  spawned threads in view, plus a chevron in the actions slot.
  Click toggles `expanded_behaviors` membership; when expanded,
  the behavior's spawned threads (`origin.behavior_id == this`)
  render nested below with `depth=1` indent. The threads section
  filters its rows to interactive (no origin) plus orphan-origin
  (origin set but behavior not in registry) — orphans get a
  `· via {behavior}` description suffix so deleted-behavior
  provenance stays visible.

  **β.2** added inline action toolbars in the expanded body: a
  Run-now button (`RunBehavior`, disabled on errored rows since
  the scheduler would just bounce) and a Pause/Resume toggle
  (`SetBehaviorEnabled`). Wire echoes (`ThreadCreated` for runs,
  `BehaviorStateChanged` for the toggle) update the local view
  on the next tick — no optimistic mutation, so the UI never
  diverges from server state.

  Edit / Delete still pending γ (need modals + arm-confirm). Cron
  schedule humanization also γ since `BehaviorSummary` doesn't
  carry the schedule string and needs a `GetBehavior` round-trip.
- **✅ Slice γ — danger affordances.** Two-click arm-confirm
  landed for Delete Behavior. State is a single
  `delete_armed_behavior: Option<(pod, behavior)>` slot — only one
  arm at a time so the UI never has two "confirm" buttons live.
  Pre-handler at the top of `on_event` auto-disarms when the next
  click lands anywhere other than the matching `behavior-delete:`
  key, so opening a modal / switching tabs / picking a thread all
  cancel a pending arm.

  Visual contract: idle Delete is a trash `icon_button().ghost()`;
  armed is the same trash icon under `.destructive()` (solid red
  fill — `.ghost().destructive()` doesn't compose, since
  `destructive()` writes back to `fill` regardless of ghost).
  The ghost→destructive color flip is the arm signal: a quiet
  gray icon goes loud red on the user's first click, with time
  to render before the second click of a normal-tempo double-tap
  lands. Pre-handler at the top of `on_event` clears the arm if
  any other click happens first. Layout: single row of four
  icon-buttons (Run / Pause·Resume / Edit / Delete) clustered
  side-by-side. With every action shrunk to 32 px the strip is
  ~152 px wide, well under the sidebar's 224 px, so the two-row
  split the text-labelled version needed is gone.

  Chrome leans on icons over text per `feedback_aetna_chrome_icons`
  — Run is `zap` (lightning, distinct from Pause's `play`-on-
  resume), Pause is `pause`/`play`, Edit is `square-pen`, Delete
  is `trash`. None of these ship in aetna's built-in
  [`IconName`] registry yet; the lucide-shaped SVGs are bundled
  in `crates/whisper-agent-aetna-ui/src/icons.rs` and constructed
  via `SvgIcon::parse_current_color`. When a future aetna
  upstream registers them, the bundled `LazyLock<SvgIcon>` slots
  collapse to plain `"name"` strings.

  Edit landed via the per-behavior editor sheet (see Stage 8).
  Archive-pod and other danger ops will reuse the same
  arm-confirm shape when they land. **Cron schedule humanization
  in the sidebar description still pending** — needs a server-
  side `BehaviorSummary` field (the schedule string isn't
  carried today, only the trigger kind), so it's a wire-protocol
  change, not aetna-side work.
- **✅ Slice δ — entry points.** Three "+" affordances landed,
  each scoped to where it appears:
    - **Per-pod "new thread"** — `icon_button("plus")` keyed
      `sidebar:new-thread` in the Threads section header. Click
      clears `selected` and pre-binds the active pod into
      `picker_pod` so the new-thread compose pane opens scoped
      to where the user clicked. No wire round-trip until Start.
    - **Global "new pod"** — `icon_button("plus")` keyed
      `sidebar:new-pod` in the sidebar header next to the
      connection badge. Opens the `+ New pod` dialog (Stage 8
      modal scaffold). The title `.ellipsis()`s when the badge
      grows ("connecting…") so the chrome stays one row.
    - **Per-pod "new behavior"** — `icon_button("plus")` keyed
      `sidebar:new-behavior:{pod}` in the Behaviors section
      header (suffix carries the pod id so the dialog scopes
      correctly even if the active tab changes between click
      and modal mount). Opens the `+ New behavior` dialog.

  The Behaviors section now renders an empty-state message
  ("no behaviors in this pod yet") + the "+" affordance once
  the per-pod `BehaviorList` round-trip lands — the affordance
  is most needed precisely when the pod is empty. Pre-list
  pods skip the section entirely so a "just connected"
  sidebar doesn't flash an empty section.

### ✅ Stage 1 — Scaffold (commit `a46b421`)

Two crates wired up; native binary opens; connection-status banner only.
Dev script + bundle dump infrastructure in place. No chat content.

### ✅ Stage 2 — Sidebar + read-only chat log (same commit, then `3db5938`)

Sidebar with pods + threads, click-to-select with `nav-thread:{id}`
routing. Snapshot rendering of `display_items` (text only, no markdown).
Validates the build/on_event split end-to-end.

Polish refactor swapped hand-rolled column/row chrome for `sidebar` /
`card` / `alert` / `badge` primitives.

Chat-log polish (commits `fb6d675` / `55269bb`):
- thread-prefix `Role::System` and `Role::Tools` messages render as
  default-collapsed `SetupPrompt` / `SetupTools` accordion rows
  (`SYSTEM · {first-line preview}` and `TOOLS · {N} tools`) instead
  of dumping the full prompt as a noisy warning-gutter row. The user
  expands when they want detail
- `TurnStats` row (no gutter, no fill, right-aligned muted caption
  `in 1,284 · cached 980 · out 86`) interleaved after each
  Assistant message. `conversation_to_display_items` walks
  `snapshot.turn_log` in lockstep with assistant turns; live
  streaming appends from `ThreadAssistantEnd { usage }`
- pane-header provenance chips: `via {behavior_id}` for behavior-
  spawned threads, `forked from {short_id}` for continued threads,
  `dispatched from {short_id}` for dispatched children. Muted
  badges between the title and state badge so the user can see
  the trigger context without opening the (eventual) thread
  inspector. `short_id` truncates to 12 chars + `…` so chips
  stay compact
- destructive failure banner above the chat log when the thread
  is `state == Failed && view.failure.is_some()`. `view.failure`
  is mirrored from `snapshot.failure` on subscribe; the wire-side
  `Error { thread_id, message, .. }` arm also writes it so a
  fresh failure surfaces before the next snapshot lands. The
  conjunction (state AND failure) avoids stale banners on
  recovered threads

### ✅ Stage 3 — Compose + send (commit `66d7c3e`)

Compose `text_area` + Send button at the bottom of the selected thread's
pane, controlled via `text_area::apply_event` and `App::selection`.
`SendUserMessage` fires on click, input clears.

`ThreadUserMessage`, `ThreadAssistantTextDelta`,
`ThreadAssistantReasoningDelta` wired into the per-thread `views` map so
generations paint live. Deltas coalesce onto the trailing item of their
own kind (one growing card, not one card per delta).

### ✅ Stage 4 — Compose into a fresh thread

The no-selection content pane is now a centered card with a
`form([...])` of three `select_trigger` pickers (Backend / Model /
Pod), a `text_area` compose, and a primary `Start` button. Pickers
go through `widgets::select` (`select_trigger` + conditional
`select_menu` riding the overlay layer); event routing uses
`select::classify_event` so picking a backend can side-effect into
clearing the model selection and firing a `ListModels` for the new
backend.

Still deferred:
- bindings surface (knowledge-db / behavior / shared MCP hosts) — the
  egui sibling exposes these as a collapsible "advanced" panel; we'll
  port it once Stage 8's modals exist as a precedent for the picker
  shape
- composing-new alongside an existing selection (the egui sibling's
  toggle between "send to selected thread" and "compose new"). Today
  the form is reachable only when no thread is selected.

Wire flow: `ConnectionOpened` fires `ListBackends` alongside
`ListPods` / `ListThreads`. Picking a backend triggers
`ListModels { backend }` once per connection (deduped by
`requested_models_for`). On send, `build_creation_request` resolves
`(config_override, pod_id)` from the picker state — including
falling through to the picked backend's `default_model` or first
fetched model so the server doesn't accidentally use the wrong
backend's default. `ThreadCreated` auto-selects the new thread when
no current selection (so the post-create transition lands the user
in the live conversation).

### ✅ Stage 5 — Drafts, prefill progress

Per-thread compose drafts: `ChatApp.drafts: HashMap<String, String>`
holds one buffer per thread the user has touched. The compose
`text_area` binds to `drafts[selected]` when a thread is selected
and to `compose_input` for the new-thread form (the latter remains
the only buffer for the no-selection case).

`SetThreadDraft` fires on every text-changing edit — bandwidth is
trivial vs. the value of always-persistent drafts, and a per-frame
clock-debounce doesn't ride the build/on_event split cleanly.
`ThreadDraftUpdated` broadcasts (other clients editing the same
thread) overwrite the local buffer wholesale; the server only
fans out to non-senders so we never receive our own echo.
Snapshot `draft` field hydrates `drafts[id]` on subscribe.
Selection cursor resets on thread switch (offsets index a
different string).

`ThreadPrefillProgress` renders as a thin `progress(...)` bar plus
a muted token-count caption between the toolbar and the
chat-log scroll. Cleared on first text/reasoning delta — the
protocol guarantees prefill events stop once the model starts
emitting output.

Deferred to a later slice:
- per-keystroke fanout debounce (chrono-backed wall clock or a
  frame-counter) — currently every keystroke hits the wire
- per-message hover affordances (copy, fork-from-here) — wants
  precedent from Stage 8's modals (fork dialog) first

### ✅ Stage 6 — Tool calls and tool results

Tool rows are now real `accordion_item`s with role-coded gutters
(WARNING for in-flight / success, DESTRUCTIVE for errors) and a
`code_block` body that shows args, streaming output (when in
flight), and integrated result. The header carries a status glyph
(⏳ / ✓ / ✗) plus the tool name so a closed row communicates pass /
fail at a glance.

Snapshot walk fuses `ContentBlock::ToolUse` + `ContentBlock::ToolResult`
into a single row when they're adjacent (no intervening user /
assistant text turn) — same fusion the egui sibling does. Streaming
events `ThreadToolCallBegin`, `ThreadToolCallContent`,
`ThreadToolCallEnd`, and `ThreadToolCallStreaming` (args-being-typed
placeholder) all land into the same row shape.

Header polish (commits `55269bb` / `39617e3`):
- collapsed row appends the first-line result preview after the
  glyph + name so a closed row reads `✓ list_files · drwxr-xr-x
  rust-stdlib/` rather than just `✓ list_files`
- relevant arg surfaces in `[brackets]` between name and preview
  (`✓ read_file [src/lib.rs] · pub mod foo;`), pulled by
  `tool_summary_from_args` from the tool's well-known field
  (`path`, `command`, `pattern`, …). Same heuristic as egui's
  `tool_summary`, plus aliases for whisper-agent's own tool
  catalog (`run_bash`, `list_files`)
- streaming tool calls (`result.is_none() && !streaming_output.is_empty()`)
  auto-expand so the user sees output scroll without clicking;
  collapse back when `End` lands and the buffer clears (commit
  `8a3db38`)

Diff rendering landed: `edit_file` (`old_string` / `new_string`)
and `write_file` (`content`, treated as a creation) tool calls now
resolve a `DiffPayload` at conversion time and render through
`diff_body` instead of dumping raw JSON args. Lines come through
`similar::TextDiff::from_lines` and paint as monospace rows with
`+` / `-` / ` ` prefixes, colored via two registered theme tokens
(`diff-add-foreground`, `diff-del-foreground`) — the destructive /
success tokens read poorly on the muted body fill, so we declare
GitHub-shaped hues that the bundle linter accepts.

Still deferred:
- async `dispatch_thread` callbacks routing into the originating
  call's result slot — currently they fall through to a standalone
  `ToolResult` row when the call has scrolled out of fusion range
- streaming-args placeholder text (`writing args · {N} chars`) —
  egui shows progress while args JSON is being typed; aetna pushes
  a name-only `ToolCall` placeholder which works but doesn't
  signal "still arriving"

### ✅ Stage 7 — Images / attachments

Inbound rendering: `ContentBlock::Image { source: Bytes, .. }` and
`ServerToClient::ThreadAssistantImage { source }` decode through the
`image` crate at push time (PNG / JPEG / WebP / GIF — HEIC/HEIF land
as a `Failed` placeholder until a HEIC dep is on the table). The
decoded pixels become an `aetna_core::image::Image` cached inside
`DisplayItem::Image` so rebuilds don't redecode. `event_log_row` then
renders the image through aetna's `image()` widget at a capped
display height (320px max — preserves aspect, never up-scales) with
a small dimensions caption. URL sources fall through to a muted
placeholder + the URL caption (no fetch yet); decode errors surface
as a destructive-text + reason caption row.

Input/staging landed once aetna upstream shipped
`UiEventKind::FileDropped` / `FileHovered` /
`FileHoverCancelled` and `aetna-winit-wgpu` wired the matching
`WindowEvent::DroppedFile` into the renderer's event surface.

`ChatApp` grew four fields:
- `compose_attachments: Vec<StagedAttachment>` — staged image
  rows. Each carries the canonical `Attachment` (drains into
  outbound `SendUserMessage` / `CreateThread` on submit), a
  pre-decoded `aetna_core::Image` for the thumbnail, and a
  monotonic `id` so duplicates the user staged deliberately
  remain individually addressable through
  `compose:attach:remove:{id}`.
- `next_attachment_id: u64` — counter for the id above.
- `pending_picks: Arc<Mutex<Vec<RawPick>>>` — async handoff
  queue from the rfd file picker thread. `before_build`
  drains it through the same `stage_raw_pick` pipeline that
  drag-drop uses, so success / rejection diagnostics don't
  drift between input sources.
- `compose_hint: Option<(String, Instant)>` — ephemeral
  status line under the thumbnail strip, self-expiring after
  4 seconds. Surfaces feedback on every attach attempt
  (success, MIME mismatch, read errors) so silently-rejected
  drops stop being silent.

`on_event`'s `UiEventKind::FileDropped` arm reads the path
synchronously, MIME-sniffs (`sniff_image_mime` mirrors the
egui sibling — magic-byte checks for jpeg / png / gif / webp /
heic / heif; unknown shapes get rejected with a hint), pre-
decodes a thumbnail, and pushes onto `compose_attachments`.
The compose bar grew a `paperclip` icon-button keyed
`compose:attach`; native click spawns rfd's `AsyncFileDialog`
on a background `std::thread::spawn` driven by a tokio
current-thread runtime (xdg-portal backend; no GTK3
dependency). Wasm-side click sets a "not yet" hint until
Stage 11 wires the browser path.

The compose bar's row now hosts text_area + paperclip + send
horizontally, with a thumbnail strip + hint paragraph
stacked above it via a wrapping column. Each thumbnail tile
is 96×96 with an `x` icon-button overlaid for one-click
removal and a truncated filename caption below.

Clipboard-paste still pending: aetna upstream doesn't yet
expose a `ClipboardImage` / paste event variant. The native
binary handles drag-drop + filepicker today, so screenshots
land via paste-into-temp + drag, or via the picker.

Also still deferred:
- URL fetching via the host shell (caching + a placeholder while
  in-flight)
- inline images in `ContentBlock::ToolResult` bodies (today
  collapsed via `tool_result_text_summary`'s text-only extractor)

### 🌗 Stage 8 — Modals (in flight)

`dialog` is the widget for centered form modals; `sheet` is the
fit for document-shaped surfaces (anything multi-column or
multi-tab). The egui sibling has 8 modals, each a few hundred
lines:
- settings (server config, shared MCP hosts, embedding providers)
- knowledge buckets
- ✅ behavior editor (v1: name / description / trigger kind /
  cron schedule / prompt — sheet, not dialog)
- ✅ pod editor (raw TOML — sheet, not dialog)
- ✅ new pod
- ✅ new behavior
- ✅ fork thread (paired with per-User-row hover affordance)
- file / JSON / image viewers

**Pattern landed with the `+ New pod` modal:**

- Per-modal state struct (`NewPodModalState`) wrapped in
  `Option<…>` on `ChatApp` — `Some` while open, `None` while
  closed; opening twice resets the form (matches the user's
  intent of "start over").
- Routed-key prefix (`new-pod:…`) covering scrim dismiss, the
  text inputs, and the primary / cancel buttons. The dialog's
  scrim auto-emits `{prefix}:dismiss` so we don't have to wire
  it explicitly.
- `correlation_id`-based wire round-trip: the modal stamps a
  monotonic id on `CreatePod` and stays open with a disabled
  primary button until either `PodCreated` (close + switch tab
  to the new pod) or `Error { correlation_id }` (re-enable +
  surface `message`). Match by id, not pod_id, so an unrelated
  echo from another client can't accidentally close the modal.
- Dialog body composes shadcn-shaped primitives: `dialog_header`
  (title + description) → `form` (two `form_item`s, each
  label / control / description) → optional `alert` blocks
  (server warning, validation error) → `dialog_footer`
  (cancel + create).
- Renderer hangs off `popover_layers()` so it overlays whatever
  surface was underneath without restructuring the chat content.

Validation lives client-side (`validate_pod_id_client`,
empty-fields and duplicate-id checks); the server runs the same
checks plus a few more during `CreatePod` and any rejection
echoes back through the `Error` path. `fresh_pod_config` is
deliberately the lighter of the two egui paths — it only knows
about `backends`, not the default-pod TOML template — so it
lands a working pod the user can edit afterwards. A
`GetPod`-based default-template clone is a follow-up.

The remaining modals slot into this scaffolding pattern.

**Behavior editor sheet (landed):** the first modal that uses
`sheet` (right-attached `SheetSide::Right`) instead of a
centered `dialog`. The on-disk surface is too wide for a
form-shaped dialog — full egui parity has 7 tabs (Trigger,
Thread, Scope, Retention, Prompt, System Prompt, Raw TOML).
The aetna v1 ships the 80% case as a single scrollable form:
name, description (multi-line `text_area` so production-shaped
~100-char descriptions wrap), trigger kind picker (manual /
cron / webhook), cron schedule (visible only when kind ==
Cron), and prompt `text_area`. Fields not exposed by the form
ride through `working_config` unchanged on save — the editor
holds a clone of the loaded `BehaviorConfig` and only mutates
the surfaced fields, so a v1 edit can never strip thread
overrides, scope, retention, or the system-prompt override
config.

Lifecycle:
1. Edit click on the per-behavior toolbar opens the sheet,
   mints `pending_get` correlation, fires `GetBehavior`.
2. Sheet renders a "loading…" placeholder until the matching
   `BehaviorSnapshot` arrives. Hydrate populates
   `working_config` + `working_prompt`; trigger kind +
   schedule are derived from the variant tag (preserving
   timezone / overlap / catch_up for cron in baseline so a
   no-op save never rewrites them).
3. Save mints `pending_save`, ships `UpdateBehavior` with
   `system_prompt: None` (v1 must not clobber the override
   file the form doesn't expose). `BehaviorUpdated` echo
   closes the sheet on correlation match; an `Error` surfaces
   into the same destructive `alert` slot client-side
   validation uses.

The body is wrapped in a `scroll(...)` between header and
footer so the prompt's 160 px text_area + form items + alert
never overflow vertically inside the sheet's `Size::Fill`
panel. Trigger-kind picker rides on `popover_layers()` above
the sheet — same pattern the new-thread compose pickers use,
extended in `close_other_pickers` to include this menu so the
single-active-menu invariant holds.

**Behavior editor — tabs strip + structured Trigger tab
(landed):** rewrites the editor body around a 7-tab strip
(Trigger / Thread / Scope / Retain / Prompt / System / Raw)
matching the egui sibling. Sheet width bumped to 600 px from
the stock 360 px — 7 labels at the narrow width all overflow.
"Retention" is shortened to "Retain" so the strip lints clean.

The Trigger tab body has identity (name + description), the
trigger-kind picker, and a kind-conditional sub-form:
- **Manual** — a hint paragraph; no further config.
- **Cron** — schedule + timezone text inputs, plus
  `overlap` / `catch_up` `select_trigger`s driven by a new
  `BehaviorEditorPicker` enum (single-active-at-a-time via
  `close_other_pickers`).
- **Webhook** — `overlap` picker + an endpoint hint
  showing `POST /triggers/{pod}/{behavior}`.

`BehaviorEditorSheetState` grew `timezone_buffer`,
`overlap_buffer`, `catch_up_buffer` (preserved across kind
switches so toggling Cron→Webhook→Cron doesn't lose typed
values), `tab: BehaviorEditorTab`, and `open_picker:
Option<BehaviorEditorPicker>`. Hydrate seeds all the buffers
from the snapshot's `TriggerSpec` variant; `resolved_trigger`
rebuilds the variant from the live buffers on save.

The Prompt tab body is the existing `prompt.md` text_area
relocated under its own tab (was the trailing form item in
the previous single-form layout).

**Cron preview + presets (landed):** new `cron_preview`
sub-module mirrors webui's verbatim — `parse_schedule` (5-field
with `"0 "` seconds prefix matching the server's scheduler),
`parse_tz` (chrono_tz IANA names), `CRON_PRESETS` /
`COMMON_TIMEZONES` static tables, plus `next_firings` and
`format_relative` for the preview body. Six unit tests
cover the parsers and `format_relative`'s output shapes.

In the editor, the Cron arm of the Trigger tab grew preset
chip rows (CRON_PRESETS as 4-then-3, COMMON_TIMEZONES as 3×3)
plus a preview form item. Aetna doesn't flex-wrap rows, so
chip rows are split into multiple `row(...)`s under one
`column(...)`. Click routing on a chip rewrites the
schedule/timezone buffer by index — keys carry the array
position, not the literal expression, to avoid escaping
`* / -` inside routed key strings.

The preview body parses the live schedule + timezone buffers
each frame, renders a destructive paragraph on parse failure,
or a `next 5 firings ({tz})` header followed by a column of
`[mono(timestamp), muted(relative)]` rows. `Utc::now()` makes
the preview wall-clock-dependent — fine for rendering, makes
bundle outputs non-deterministic across runs (they're not
committed, so no churn).

**Behavior editor — Thread tab scalar overrides (landed):**
ports the egui sibling's `render_behavior_editor_thread_tab`'s
"Thread overrides" section. Each scalar field on
`BehaviorThreadOverride` (`model`, `max_tokens`, `max_turns`)
becomes a `[checkbox, override-label, control]` row — the
checkbox flips `Some/None`, the control is a `select_trigger`
or `numeric_input` when on, a muted `(inherit pod default)`
paragraph when off. Default-on values match the egui pre-fill
(empty model string, 16384 tokens, 30 turns).

`BehaviorEditorSheetState` grew `thread_max_tokens_buf` /
`thread_max_turns_buf` (same buffer-then-parse-back pattern as
the pod editor's Defaults numeric inputs);
`sync_thread_buffers_from_config` seeds them at hydrate.
`BehaviorEditorPicker::ThreadModel` joins the picker family;
its menu reads from
`models_by_backend[bindings.backend.unwrap_or("")]` (the
binding sub-slice will fix the backend lookup against the
pod's effective default).

The bindings sub-struct (`backend` / `host_env` / `mcp_hosts`)
is deferred to its own sub-slice — they share the
override-checkbox shape but each carries its own catalog
lookup, and folding them into this commit doubled its size.

**Behavior editor — Thread bindings backend (landed):**
adds the `bindings.backend` override row using the global
`self.backends` catalog. `BehaviorEditorPicker::ThreadBackend`
joins the picker family. Picking a backend clears the model
override (its prior id is almost certainly invalid for the
new backend) — same shape as the pod editor's Defaults
backend pick.

`bindings.host_env` and `bindings.mcp_hosts` are still
deferred — they want the pod's `[allow.host_env]` /
`[allow.mcp_hosts]` lists as their option set, which means
the behavior editor needs to load the pod's full config.
Plumbing that fetch is a separate slice; until it lands the
two override fields ride through `working_config` unchanged
on save.

**Behavior editor — SystemPrompt tab (landed):** the
behavior's optional override for the spawned thread's
agent-personality preamble. An override checkbox flips
`cfg.thread.system_prompt` between `None` (inherit pod
default) and `Some(File { name = behaviors/<id>/system_prompt.md })`.
When on with the File variant, a 280 px text_area binds to a
new `working_system_prompt: Option<String>` buffer
(hydrated from `snapshot.system_prompt`); `submit_behavior_editor`
ships the buffer as `UpdateBehavior.system_prompt`, and the
server writes it to the conventional sibling path. The Text
variant binds the text_area directly to the inline string —
no side-file write, content rides through `config` itself.

Toggle-off drops the config reference but leaves
`working_system_prompt` alone, so toggling back on doesn't
lose the user's draft. Non-conventional file paths surface a
muted "edits land at the conventional path" warning since
`UpdateBehavior` only writes the canonical
`behaviors/<id>/system_prompt.md`.

**Behavior editor — Retention tab (landed):** the on-completion
policy editor. A `select_trigger` 3-way kind picker (Keep /
ArchiveAfterDays / DeleteAfterDays) plus a `numeric_input`
`days` row that's only rendered for the timed variants.
`BehaviorEditorPicker::RetentionKind` joins the picker family
and `retention_days_buf: String` lives on
`BehaviorEditorSheetState` (default `"30"` so a Keep → Archive
flip lands at the same default the egui sibling uses).
`sync_retention_buffer_from_config` seeds the buffer at hydrate.
Picking a timed variant re-applies the live buffer's parsed
days, so kind switches in the live editor preserve typed
values.

All structured tabs and the Raw TOML escape hatch landed.

**Pod editor sheet (landed):** mirrors the behavior editor's
sheet shape but the body is much simpler — a single
monospace `text_area` over the pod's raw `pod.toml`. The wire
surface (`UpdatePodConfig { toml_text: String }`) takes the
text directly; the server parses + validates and replies with
`Error` on parse failure (surfaced inline) or
`PodConfigUpdated` on success (closes the sheet on correlation
match). The text_area's default Hug height combined with the
outer `scroll` lets long pod.tomls scroll the whole sheet body
rather than clipping internally.

**Fork-thread dialog + per-row hover affordance (landed):**
the first slice that uses aetna's new `BuildCx::is_hovering_within(key)`
+ subtree-interaction-envelope cascade (upstream gh#9 / gh#10).
Each User row in the chat log gets a keyed wrapper
(`chat:user-row:{idx}`) and `event_log_row` reads
`is_hovering_within(row_key)` during build to decide whether to
render a `git-branch` icon-button next to the user text. Click
on the affordance routes through `chat:user-fork:{msg_index}`,
which opens the fork dialog pre-populated with the clicked
message's `from_message_index` + seed text.

The fork dialog itself is a regular `dialog`-shaped modal: an
explainer paragraph + two `switch`-shaped form items (Archive
original — default on; Reset capabilities — default off). Confirm
mints a correlation, ships `ClientToServer::ForkThread`, and
stashes `(correlation, seed_text)` in `pending_fork_seed`. The
matching `ThreadCreated` echo (correlation-matched) seeds the new
thread's `drafts[new_id]` slot with the captured prompt and auto-
selects the new thread — so the user lands on a compose box that's
ready for "edit slightly and resend." Mirrors the egui sibling's
`pending_fork_seed` flow.

`DisplayItem::User` grew a `msg_index: usize` field for this. The
snapshot path computes it from the message's position in
`Conversation.messages()`. Streaming `ThreadUserMessage` reads it
from a new `ThreadView::next_msg_index` counter that hydrates
from the snapshot's message count and bumps on commit-shaped wire
arms (`ThreadUserMessage`, `ThreadAssistantEnd`,
`ThreadToolResultMessage`).

**Hover note:** the affordance is hover-conditional in the live
binary but always rendered in the bundle dump's `ForkModalOpen`
scene — synthetic clicks route by key alone, so the click
through to the modal doesn't depend on the affordance being
visually rendered. Bundle-side hover simulation needs a
`UiState`-injecting `BuildCx` and isn't worth the surface area
for one scene.

Entry point: a `settings`-icon button in the sidebar header,
rendered only when some pod tab is active. Single key
(`sidebar:pod-settings`) — at any moment exactly one pod is
active, so the click target is implicitly scoped to whatever's
selected.

Save short-circuits the no-op case (`working_toml ==
baseline_toml`) so a re-save without edits surfaces "no
changes to save" rather than a wire round-trip. v1 explicitly
does not surface a structured form (allow lists, host_envs,
MCP hosts, sandbox, thread_defaults) — those land in
follow-up slices that mirror the egui pod editor's
multi-tabbed shape, but the raw-only path covers the 80%
case (and remains the escape hatch for any malformed config
the structured form can't represent).

**Pod editor — Allow tab + tabs strip (landed):** first
structured slice on top of the raw-TOML escape hatch. The
sheet body now starts with a 4-button tabs strip
(Allow / Defaults / Limits / Raw) that switches between
structured tabs and the legacy text_area. State grows a
`working_config: Option<PodConfig>` (the structured side's
truth) plus `working_toml`/`raw_dirty` (the raw side's
escape hatch). `switch_tab()` enforces the sync invariant:
leaving Raw with `raw_dirty == true` reparses the toml back
into `working_config` (or surfaces the parse error and bounces
the user back to Raw); entering Raw re-serializes the structured
state so the user sees a fresh round-trip. `resolved_save_toml()`
returns whichever side is authoritative for the active tab —
raw text when Raw is current and dirty, otherwise
`toml::to_string_pretty(working_config)`.

The Allow tab itself ports the egui sibling's
`render_allow_tab`: identity (id is read-only / created_by is
read-only / display_name + description text fields), allowed
backends (multi-check across `state.backends`), allowed shared
MCP hosts (multi-check across `state.shared_mcp_hosts`),
allowed knowledge buckets (multi-check across `state.buckets`),
plus the three pod-modify / dispatch / behaviors caps as
`select_triggers` driven by the new `PodEditorPicker` enum.
Catalog wiring: app state grew `shared_mcp_hosts:
Vec<SharedMcpHostInfo>` and `buckets: Vec<BucketSummary>`, both
hydrated from `ClientToServer::List{SharedMcpHosts,Buckets}` at
boot (mirrors the existing `ListPods` / `ListBackends` arms).

Multi-checks use a custom `checkbox_column` helper rather than
aetna's `toggle_group_multi`, since the latter doesn't wrap and
the 360px sheet width truncates 4+ items. The helper renders a
column of `[checkbox, label]` rows that all share a single key
prefix; click routing collects the picked id via
`apply_checkbox_list_to_vec()` (push if missing, remove if
present) into the corresponding `Vec<String>` on `working_config`.

`PodEditorPicker` originally carried
`#[allow(clippy::enum_variant_names)]` to keep the `AllowCaps*`
prefix grouping; the Defaults-tab slice added
`DefaultsBackend` / `DefaultsModel` / `DefaultsToolGate` /
`DefaultsCaps*` variants which broke the shared prefix, so the
attribute is gone and the lint stays quiet on its own. The
enum gained a `key()` method that maps each variant to its
routed `select_trigger` key so `close_other_pickers`,
`handle_pod_editor_picker`, and `pod_editor_picker_menu` no
longer carry separate per-variant key tables.

**Pod editor — Defaults tab (landed):** structured port of the
egui sibling's `render_pod_editor_defaults_tab`. Form items:
backend / model `select_trigger`s (catalog-driven; backend
options carry `(not in allow)` suffix labels for catalog rows
that aren't in `allow.backends`, matching the egui sibling's
"would error on save" warning shape), `system_prompt_file`
text input, `max_tokens` / `max_turns` aetna `numeric_input`s,
tool-gate default + per-cap defaults `select_trigger`s, and
host_env / mcp_hosts multi-checks scoped to the pod's
`allow.host_env` / `allow.mcp_hosts` lists. Per-tool overrides
defer to the Raw tab (the egui sibling does the same — they're
an unbounded `String → Disposition` map; a structured editor
would balloon the sheet).

`max_tokens_buf` / `max_turns_buf` live on `PodEditorSheetState`
because aetna's `numeric_input` owns its visible text as an
external `String` (so mid-edit states like `"1"` survive
between keystrokes). The handler parses each parseable buffer
back into the `working_config.thread_defaults.{max_tokens,
max_turns}` u32 on every event so `dirty()` and the save
round-trip stay accurate. Buffers re-sync from
`working_config` on hydrate and after Raw → structured
reparses, via a new `sync_buffers_from_config()` helper.

The Defaults backend pick clears `thread_defaults.model` (the
prior model id is almost certainly invalid for the new
backend); the model menu options come from
`models_by_backend[thread_defaults.backend]`, so changing
backends routes the user to a fresh pick on the next click.

**Pod editor — Limits tab (landed):** trivial slice — the
protocol's [`PodLimits`] struct only carries
`max_concurrent_threads` today, so the form is one
`numeric_input` (1–1000, step 1). Same buffer pattern as the
Defaults numeric inputs (`max_concurrent_threads_buf` lives on
`PodEditorSheetState`, parsed back to `working_config.limits`
on every event). New form items drop in alongside as the
schema grows.

**Pod & behavior editors — modal conversion (landed):** the
right-attached sheets were always a compromise — even at 600 px
the behavior editor's 7-tab strip and the Defaults / Thread /
Scope rows felt cramped, and the egui sibling has always used a
centered `egui::Window::new(...).anchor(CENTER_CENTER)` shape.
Both editors swapped to `dialog_content` panels (720 × 640,
matching egui's `default_width(720) / default_height(560)` plus
headroom for aetna's larger form_item gaps), centered via
`overlay([scrim, panel]).align(Align::Center).justify(Justify::Center)`.
Same `dialog_header` / `scroll(body)` / `dialog_footer` anatomy
as the create modals; only the panel sizing differs. Picker
menus (`select_menu` overlays) anchor to their `select_trigger`
keys regardless of where the parent dialog sits, so no rework
to `behavior_editor_picker_menu` /
`behavior_editor_trigger_kind_menu` / `pod_editor_picker_menu`
was needed. State struct names (`PodEditorSheetState` /
`BehaviorEditorSheetState`) and key prefixes (`pod-editor:` /
`behavior-editor:`) stay as-is — they're internal and the
churn isn't worth it. Renderer entry points are now
`render_{pod,behavior}_editor_modal`.

**Behavior editor — pod_config plumbing + Thread bindings
host_env / mcp_hosts (landed):** the Thread tab's host_env /
mcp_hosts override rows landed as a placeholder paragraph
("…land in a follow-up sub-slice once the editor loads the
pod's allow lists"). This slice hooks them up.

`BehaviorEditorSheetState` grew `pod_config: Option<PodConfig>`
and `pending_pod_get: Option<String>`. `open_behavior_editor`
mints a second correlation alongside the existing `GetBehavior`
correlation and fires `GetPod` in parallel; both round-trips
race, and whichever lands first hydrates its slice (so the
prompt / trigger fields show up the instant the behavior
snapshot lands, even if the pod snapshot is still in flight).
The `PodSnapshot` inbound arm grew a second match: the existing
pod-editor branch stays, and a new behavior-editor branch checks
the open editor's `pending_pod_get` correlation + `pod_id` and
stashes `snapshot.config` into `pod_config`.

The Thread tab now renders host_env / mcp_hosts override rows
in the same `[checkbox, override-text, value-or-hint]` shape as
the existing model / max_tokens / max_turns / backend rows. With
override OFF (the default), the value cell renders the standard
"(inherit pod default)" muted hint. With override ON, the cell
checks `pending_pod_get` first ("(loading pod allow list…)"
while in flight), then renders a `checkbox_column` over the
pod's `allow.host_env` / `allow.mcp_hosts` names, identical to
the pod editor's Defaults tab multi-checks. Empty allow lists
surface "(no host envs in pod [allow])" / "(no shared MCP hosts
in pod [allow])" so the user knows they need to widen the pod
first.

Two new override-checkbox handlers and two new
`apply_checkbox_list_to_vec` calls cover the new keys
(`behavior-editor:thread:host-env:override` /
`behavior-editor:thread:host-env:item:{name}` and the
mcp-hosts mirror). Same Some/None toggle pattern as the
backend override: ON seeds `Some(Vec::new())` (empty
override = "bind to none"), OFF drops to `None` (inherit pod
default).

Bundle scenes that open the behavior editor now answer both
`GetBehavior` and `GetPod` (the SendFn match arm grew from
single-pattern `if let` to a `match` over both wire kinds), so
the dump's second `before_build` pass sees a populated
`pod_config` and the host_env / mcp_hosts inherit hints render
the same way they will in the live binary.

**Behavior editor — Scope tab (landed):** ports the egui
sibling's `render_behavior_editor_scope_tab`. Every field on
[`BehaviorScope`] is `Option`-shaped (`None` = inherit pod
ceiling), so each row is the same `[checkbox, "override",
value-or-inherit]` shape the Thread tab settled on. Eight
form items: three resource-set multi-checks (backends /
host_envs / mcp_hosts) over the pod's allow lists; the
`tools.default` Disposition picker (`allow` / `deny`) plus an
override-count text reading "(N per-tool override(s) — edit
via Raw TOML)" since per-tool overrides are an unbounded
`String → Disposition` map; three cap pickers (pod_modify /
dispatch / behaviors); and a tool-surface row whose ON state
shows a "(structured editor coming in a follow-up; use Raw
TOML to edit fields)" hint, deferred for the same reason as
the pod editor's Defaults `tool_surface` sub-slice.

`BehaviorEditorPicker` grew four variants
(`ScopeToolsDefault` / `ScopeCapsPodModify` /
`ScopeCapsDispatch` / `ScopeCapsBehaviors`); the picker-
iteration list, picker-handler match, and `picker_menu`
options switch picked up new arms. Override toggles seed
sensible identity-under-narrow defaults: resource sets
toggle `Some(Vec::new())` ↔ `None`; tools toggles
`Some(AllowMap::allow_all())` ↔ `None`; caps toggle a
ceiling-shaped value (`PodModifyCap::ModifyAllow`,
`DispatchCap::WithinScope`, `BehaviorOpsCap::AuthorAny`) ↔
`None`; tool_surface toggles `Some(ToolSurface::default())` ↔
`None`. New `disposition_label` / `disposition_from_wire`
helpers cover the bare `allow`/`deny` shape (distinct from
the pod editor's `tool_gate_label`'s `_all`-suffixed shape —
the Scope tools default isn't a pod-wide gate).

A new `behavior_editor_scope_tab` bundle scene exercises the
tab; the architect mock has `BehaviorScope::default()` so
every row renders in the inherit-only state — useful for the
structured layout, even though the override-on multi-checks
and cap pickers don't show.

**Pod editor — Defaults tool_surface (landed):** the Defaults
tab placeholder paragraph for `thread_defaults.tool_surface`
got replaced with the structured editor that mirrors the egui
sibling's `render_tool_surface_editor`: three sections under a
single form_item, each headed by a small label.
- **`Wire tools: core set`** — a 2-option `radio_group`
  (`all` vs `named`); when `named` is selected, a multi-line
  monospace `text_area` (96 px, one tool per line) takes over.
  The named buffer lives on `PodEditorSheetState` so mid-edit
  states like a trailing newline survive between keystrokes.
- **`System-prompt listing`** — a 3-option `radio_group`
  (`none` / `all_names` / `core_only`).
- **`Mid-conversation activation`** — a 2-option `radio_group`
  (`announce` / `inject_schema`).

The radio routes use `aetna_core::widgets::radio::apply_event`
with the tab's matching `_from_wire` parser (e.g.
`initial_listing_from_wire`); the textarea routes through
`text_area::apply_event` and parses-and-writes-back to
`CoreTools::Named(parse_core_tools_named(buf))` on every
keystroke. New helpers: `default_core_tools_text` (seed for
the `All → Named` toggle), `parse_core_tools_named`, and the
three `_label` / `_from_wire` pairs for `InitialListing` and
`ActivationSurface`.

The named buffer hydrates from
`sync_buffers_from_config`: `CoreTools::Named` joins the list
back to a newline-separated string; `CoreTools::All` falls
back to `default_core_tools_text()` so toggling `All → Named`
lands the conventional default text without a separate
keystroke. The matching textarea-driven write-back also
parses on every edit, so a stray empty line trims out before
landing in `CoreTools::Named`.

The existing `pod_editor_defaults_tab` bundle scene picks up
the new editor without changes — `mock_default_pod_snapshot`
ships `tool_surface: Default::default()` (which is
`CoreTools::Named` of the conventional three names), so the
scene shows the named buffer populated and the radio at
`named`.

**Behavior editor — Raw TOML tab (landed):**
`BehaviorEditorSheetState` grew `working_toml: String` and
`raw_dirty: bool`. Hydrate seeds the buffer from
`toml::to_string_pretty(snapshot.config)` so a first visit
to Raw shows the round-tripped TOML rather than a blank
pane. Tab routing now goes through a new `switch_tab`
helper modeled on the pod editor's: leaving Raw with
`raw_dirty` reparses the buffer back into `working_config`
(re-deriving the trigger-kind / cron / numeric / retention
buffers in the process so the structured tabs reflect the
just-typed TOML); a parse error keeps the user on Raw with
the message in `error`. Entering Raw re-serializes from
`working_config` after applying `resolved_trigger()` so
pending kind / cron buffer edits survive the round-trip.

`submit_behavior_editor` grew the same Raw-current branch
the pod editor's `resolved_save_toml` carries: when the
user is on Raw with edits, the buffer parses to
`BehaviorConfig` and ships through `UpdateBehavior`
verbatim; a parse failure keeps the editor open with the
error surfaced in the destructive alert slot.

The Raw tab body is a hint paragraph above a `text_area`
keyed `behavior-editor:raw-toml`. The textarea uses Hug
height (the editor modal's outer scroll handles overflow,
matching the pod editor's Raw shape) and the wrapping
column carries `tokens::RING_WIDTH` of horizontal padding
so the textarea's focus ring isn't clipped by the scroll's
scissor (lint catches the bare-edge case).

A new `behavior_editor_raw_tab` bundle scene exercises the
tab — the architect mock's parsed config gets serialized on
hydrate, so the textarea shows the round-tripped TOML the
moment the scene paints.

### ✅ Stage 9 — Login form

`whisper_agent_aetna_ui::LoginApp` is a separate aetna [`App`] —
centered card with a server URL field, a password-masked token
field (`text_input_with(...).password()`), a "Remember on this
device" checkbox, and a primary Connect button. Submission rides a
host-supplied `SubmitFn` callback so the lib stays platform-agnostic
(both the wasm browser entry and the desktop binary will reuse it).

Desktop binary owns a `RootApp` that wraps either a `LoginApp` or
the existing `DesktopApp(ChatApp)` in a `Phase` enum. Startup
loads `$XDG_CONFIG_HOME/whisper-agent/desktop.toml` (same shape as
the egui sibling — both clients can share the file) and
auto-connects when both server URL and token are known. CLI flags
override saved config; `--login` forces the form even when both
are present.

Submission flow: `LoginApp::submit` writes to an
`Arc<Mutex<Option<LoginInput>>>` "phase signal" the host shell
reads in `before_build`. On a pending submission, the host
persists creds (gating token write on the "Remember" checkbox),
spawns the WS task, and swaps `Phase` to `Connected`. Failed
`derive_ws_url` calls bounce back to a fresh `LoginApp` with
`set_error(...)` carrying the parse error.

Bundle scenes: `LoginFormEmpty`, `LoginFormPrefilled`,
`LoginFormWithError`. The dump loop is now generic over
`Box<dyn App>` so future App variants (e.g. modal hosts) can join
without reshaping the loop.

### ✅ Stage 10 — Markdown rendering

`aetna-markdown` landed upstream and we now route assistant text
through `aetna_markdown::md(text)`. Pulldown-cmark walks the source
into headings, paragraphs, lists, code blocks, inline runs, and
emphasis — identical to a hand-authored aetna tree. Got picked up as
part of the event-log refactor since both touched the same code.

Future polish: GFM tables landed in `aetna-markdown 0.3.0` already;
math + footnotes + task lists + raw HTML are still upstream
deferreds.

### ⏳ Stage 11 — wasm browser entry + server bundle switch

The crate's `Cargo.toml` already lists wasm-only deps. Adding the
`#[wasm_bindgen(start)]` entry mirrors `aetna-web/src/lib.rs`:
- open a wgpu surface against a `<canvas>` from `aetna-web` shape
- mount the same `ChatApp` impl, swapping the WS bridge for the browser
  WebSocket via `web-sys`

Server-side: a workspace feature on `whisper-agent` (e.g.
`--features aetna-ui`) that swaps the `RustEmbed` folder from
`crates/whisper-agent-webui/pkg/` to `crates/whisper-agent-aetna-ui/pkg/`.
At that point we can ship a single binary that defaults to whichever ui
the build picks.

## Migration gaps from egui webui

Once the staged surfaces above are green, the remaining diff between
`whisper-agent-aetna-ui` and `whisper-agent-webui` is a pile of
smaller features that don't slot under any one stage. Tracked here so
they don't get lost in the long tail. Effort tags are
trivial / small / medium / large.

### Chat-log / per-message details

- ✅ **Per-turn token & cache stats.** `DisplayItem::TurnStats` (rendered
  via `turn_stats_text` in `app.rs`) emits `in 1,234 · cached 800 ·
  created 200 · out 567` after every assistant turn. Mirrors the egui
  sibling's `render_turn_stats` (chat_render.rs:865).
- ✅ **User-message inline thumbnails.** Wire arm now decodes each
  `Attachment::Image` in `ThreadUserMessage` and pushes a
  `DisplayItem::Image { is_user: true, … }` row, mirroring the
  snapshot path. (Snapshot path was already correct via
  `conversation_to_display_items`; live wire arm caught up.)

### Thread header / chrome

- ✅ **Backend / model chip.** Right-aligned caption under the title:
  `anthropic-prod/claude-opus-4-7`. Hydrated from
  `ThreadSnapshot::bindings.backend` + `config.model`.
- ✅ **Cumulative usage chip.** Right-aligned caption: `↑ in · ↓ out
  · cache r/c`. Hydrated from `ThreadSnapshot::total_usage`,
  incremented on each `ThreadAssistantEnd` arm. Mirrors egui's
  status-bar strip (app.rs:2693 in webui).

### Modals

- ✅ **Image lightbox.** Decoded inline images carry a click route
  (`chat:image-lightbox:{idx}`); the click hydrates a `LightboxState`
  slot and `render_lightbox_modal` paints a fullscreen `dialog` with
  the image at constrained max dims, a dimensions caption, and a
  Close button. Scrim auto-dismisses on outside-click.
- ⏳ **JSON tree viewer.** Read-only collapsible-tree display for
  tool-result and server-config JSON payloads. Egui's
  `render_json_viewer_modal` (modals/viewers.rs:175). *Small.*
- ⏳ **File browser / editor.** Tree of pod files; click-to-edit in a
  `text_area` with save. Egui's `render_file_viewer_modal`
  (modals/viewers.rs:260). *Medium — tree navigation + dirty-tracking.*
- ⏳ **Server settings.** Three tabs (LLM backends with Codex auth
  rotate, Shared MCP hosts CRUD with auth config, raw server
  TOML). Egui's `modals/settings.rs`. Reachable via the cog icon
  in the egui top bar; aetna's equivalent surface is the sidebar
  footer or an as-yet-unbuilt top bar. *Medium.*
- ⏳ **Knowledge buckets.** Full CRUD for the bucket lifecycle:
  create form (quantization / driver / cadence pickers for
  tracked buckets), build progress, search / query interface,
  two-click delete confirmation. Egui's `modals/buckets.rs`.
  *Large — async build state, complex form, search UX.*

### Sidebar / chrome

- ✅ **Pending-sudo banner.** `SudoRequested` populates a
  `pending_sudos: HashMap<u64, PendingSudoState>` slot keyed by
  `function_id`; the matching thread's pane renders one
  warning-styled `alert` per entry above the chat log, with
  Approve / Remember / Reject buttons + an optional reject-reason
  text input. Resolution is optimistic — the local slot is dropped
  before `ResolveSudo` ships so the banner doesn't flicker through
  the wire round-trip. `SudoResolved` echo cleans up server-side
  resolutions from any client.
- ⏳ **Thread inspector.** Egui's `ThreadInspector` struct surfaces
  per-thread metadata the chip can't fit (allowlist breakdown,
  resolved bindings, host envs in use). A popover on the thread
  header would be the natural home. *Small to surface; depends
  on which fields make the cut.*

### Misc

- ⏳ **Pod editor structured tabs.** v1 ships raw-TOML only (a
  single `text_area` sheet). Egui's pod editor has tabs for
  Allow / Defaults / Limits — aetna already has all three
  (`render_pod_editor_*_tab`), so this gap is closed; track here
  only as a note that future surfacing of MCP host CRUD lands
  inside the existing tabs.
- ⏳ **Behavior editor parity.** v1 ships the 80% form (name /
  description / trigger kind / cron schedule / prompt). All
  remaining tabs (Scope, Retention, Thread, System Prompt, Raw
  TOML) have already landed (see Stage 8). Track here only as
  an inventory note.

## Conventions

Picked in this codebase, not enforced by aetna:

- **One file per crate until split is forced.** `app.rs` will eventually
  break into `app/{sidebar, compose, chat_log, modals, wire_handler}.rs`
  the same way the egui sibling does — but only once the line count
  passes ~1500. Premature splitting just hides the architecture.
- **Wire handlers always replace state wholesale where the server is
  authoritative.** `ThreadList` replaces the entire `threads` map. We
  don't merge; the server's view is canonical.
- **Per-thread display state is lazy.** Only threads the user has
  opened get a `ThreadView` entry; long thread lists don't bloat
  memory.
- **Resubscribe on reconnect.** `ConnectionOpened` clears the
  `subscribed` set so re-selecting a thread re-asks for a fresh
  snapshot. The server drops subscriptions on disconnect.
- **Streaming deltas coalesce onto the trailing item.** Mirrors the
  egui sibling. A long generation paints as one growing card, not one
  card per delta.
- **`overlays(main, [])` is the canonical root.** Even when there are
  no current overlay layers, wrapping the root in `overlays(...)`
  reserves the shape for runtime-injected toasts / tooltips / popovers.

## Open questions

- **Bundle dump for streaming states.** The current scenes only cover
  static states. Driving a `ThreadAssistantTextDelta` sequence through
  the dump path is mechanically straightforward but increases
  per-scene seed code. Worth doing once stage 6 lands and tool-call
  rendering becomes complex enough to regression-test visually.
- **Live reload during development.** Aetna doesn't have hot-reload;
  edits to `app.rs` require a recompile + restart. The bundle-dump
  loop fills part of this gap (CPU-only, fast feedback on layout
  intent), but interaction-driven edges (focus traversal, hover
  envelopes, scroll persistence) still require the full window.
- **Picking up upstream aetna changes.** Aetna's API is moving while
  we port. The bundle dump's "no findings" output is the canary —
  when a new version's `lint` flags something we hadn't seen, that's
  the cue to reread the upstream changelog.

## Upstream gaps

Items the aetna sibling needs from upstream before we can close
the gap with the egui webui:

- **`scroll` widget stick-to-bottom mode.** The egui sibling
  uses `egui::ScrollArea::stick_to_bottom(true)` so the chat log
  hugs the most recent turn during streaming. Aetna's `scroll`
  widget exposes only an offset; the App trait can read /
  `set_scroll_offset` on `UiState`, but the App can only see
  `BuildCx { theme }` — the scroll-state setter is host-private,
  so even a "watch the offset and clamp it on each rebuild"
  workaround isn't available from this side. File against
  aetna; once a `pinned_to_end` (or `auto_follow_tail`)
  flag lands, the aetna chat-log scroll picks it up in one
  line.
- **`winit` drag/drop + clipboard-image events on
  `aetna-winit-wgpu`.** Stage 7's input side (image attachment
  via paste / drop / file picker) is gated on this. egui has
  it natively; aetna needs the host to surface
  `WindowEvent::DroppedFile` / clipboard-image readers via a
  `UiEvent::FileDropped` style variant before this side becomes
  a thin port.
- **Markdown viewer scroll-from-source.** The aetna-markdown
  body re-builds when the source string grows past a threshold;
  a streaming assistant turn occasionally scrolls flat to the
  top of the body. Probably tied to the stick-to-bottom gap
  above; revisit after that lands.

## Files of interest

- `crates/whisper-agent-aetna-ui/src/app.rs` — the App impl + wire
  dispatch + rendering
- `crates/whisper-agent-aetna-ui/examples/dump_bundles.rs` — scene
  catalogue and seeding helpers
- `crates/whisper-agent-desktop-aetna/src/main.rs` — winit host +
  tokio bridge
- `scripts/dev_aetna.sh` — server + daemons + native binary, one
  invocation
- `~/workspace/aetna/aetna.main/docs/LIBRARY_VISION.md` — aetna's
  app/widget-layer architecture (read alongside this doc)
- `~/workspace/aetna/aetna.main/crates/aetna-core/src/widget_kit.md` —
  the widget-kit invariant + author guidance
