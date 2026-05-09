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
- **🌗 Slice β — behaviors first-class.** Per-pod `Behaviors (N)`
  subsection above threads, populated via `ListBehaviors` (deduped
  per pod per connection) and kept fresh by `BehaviorList` /
  `BehaviorCreated` / `BehaviorUpdated` / `BehaviorDeleted` /
  `BehaviorStateChanged`. Each row is an `item_group([item([
  item_content([title, description])])])` — title is the display
  name (`⚠`-prefixed when load_error), description is
  `"{kind} · {status}"` for healthy rows (`status` =
  `paused` / `last fired Nh` / `no runs yet`) or
  `"errored: {reason}"` for malformed configs. Per-row actions
  (Run, Pause, Edit, Delete) and accordion expansion to a
  recent-threads list still pending — slice β.2 / γ. Cron schedule
  humanization deferred since `BehaviorSummary` doesn't carry the
  schedule string; lands in γ alongside `GetBehavior`.
- **⏳ Slice γ — danger affordances.** Inline arm-confirm pattern
  (probably a small `danger_button` helper that animates "Click again
  to confirm") for archive-pod / delete-behavior. Hooks into the
  ⋯ menu and the per-behavior toolbar.
- **⏳ Slice δ — entry points.** "+ New pod" in the pod-tab overflow,
  per-pod ➕ "new thread", ➕ "new behavior". Several stub out until
  Stage 8's modals exist.

### ✅ Stage 1 — Scaffold (commit `a46b421`)

Two crates wired up; native binary opens; connection-status banner only.
Dev script + bundle dump infrastructure in place. No chat content.

### ✅ Stage 2 — Sidebar + read-only chat log (same commit, then `3db5938`)

Sidebar with pods + threads, click-to-select with `nav-thread:{id}`
routing. Snapshot rendering of `display_items` (text only, no markdown).
Validates the build/on_event split end-to-end.

Polish refactor swapped hand-rolled column/row chrome for `sidebar` /
`card` / `alert` / `badge` primitives.

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

Still deferred:
- inline unified-diff rendering for `edit_file` / `write_file`
  (currently shown as raw JSON args)
- async `dispatch_thread` callbacks routing into the originating
  call's result slot — currently they fall through to a standalone
  `ToolResult` row when the call has scrolled out of fusion range

### 🌗 Stage 7 — Images / attachments (rendering side done)

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

Input/staging side deferred: `aetna-winit-wgpu` doesn't yet expose
winit's `WindowEvent::DroppedFile`/`HoveredFile` or platform
clipboard-image readers, so drag/drop/paste/filepicker need an
upstream feature first. Track in a follow-up issue against aetna —
the egui sibling's compose attachment strip becomes a thin port once
those events ride into `UiEvent`.

Also still deferred:
- URL fetching via the host shell (caching + a placeholder while
  in-flight)
- inline images in `ContentBlock::ToolResult` bodies (today
  collapsed via `tool_result_text_summary`'s text-only extractor)

### ⏳ Stage 8 — Modals

`dialog` is the widget. The egui sibling has 8 modals, each a few
hundred lines:
- settings (server config, shared MCP hosts, embedding providers)
- knowledge buckets
- behavior editor (with cron preview)
- new behavior
- pod editor (raw TOML)
- new pod
- fork thread
- file / JSON / image viewers

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
