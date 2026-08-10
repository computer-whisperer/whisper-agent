# Configurable Threads and Weaves

Status: **steps 1–7 landed** (scripted Lua drivers with the auto-mode
checker exercise case working end-to-end, plus the 7b presentation
vocabulary, weave wire tier, and damascene-ui weave view) —
participant foundation, execution profiles,
the compatibility driver with its durable effect journal, the weave
entity (singleton coordination, driver policy inverted out of `Thread`,
persistence at `<pod>/weaves/<weave_id>.json`), and the cross-thread
effect vocabulary: `append_entry` / `derive_thread` / `adopt_ticker` /
`release_ticker` executors with scheduler admission, journaled outcomes
(refusals included), persisted per-ref ticker flags, and dormant-thread
semantics. The executors' first emitter is the scripted driver of step 7;
their admission/journaling choreography is covered by scheduler-level
tests in `src/runtime/scheduler/testing.rs` (a real `Scheduler` over an
empty resource surface — no backends, no persister, lazy I/O futures
never polled). The architectural target was re-ratified 2026-07-31 into
the pod/weave/thread model below.

This document records the migration from the original one-user/one-model
agent loop to pods hosting materialized model contexts (threads) coordinated
by durable driver instances (weaves).

Companion docs:

- [`design_pod_thread_scheduler.md`](design_pod_thread_scheduler.md) describes
  the durable pod/thread/resource layout and scheduler single-writer model.
- [`design_functions.md`](design_functions.md) describes caller-initiated
  Functions, scope admission, and result delivery.
- [`design_headless_loop.md`](design_headless_loop.md) describes the provider
  and remote-tool boundaries.

## Core model

Three tiers, with the analogy that fixes their responsibilities:

```text
pod : weave : thread  ::  namespace : process : file
```

- **Thread — one coherent LLM context.** An append-only entry log that *is*
  what one model saw: its own KV-cache lineage (prefix-stable by
  construction), its own frozen execution profile instance, its own internal
  state machine. Threads are pod-level durable objects. A thread has no
  knowledge of its relationships; those live in weave journals.
- **Weave — a durable driver instance.** The unit the UI tracks. Owns the
  driver program reference + version, persisted driver state, the effect
  journal, refcounted *references* to threads, and a presentation structure.
  A weave holds no tokens of its own. Candidate name ratified provisionally;
  "conversation head" survives as the concept of the presentation's primary
  thread pointer (see Presentation).
- **Pod — security context, resources, topic, memory organization.**
  Unchanged. The scope ceiling, backend/MCP/host-env registry, and memory
  directory stay here.

Grouping is referential, not containative. Within a pod, threads may tangle
freely — be seeded from, read by, and appended into by any weave that
references them — with every cross-thread flow recorded as a journaled
effect carrying provenance. Thread lifetime uses the existing resource
refcount idiom: weaves are users of threads; unreferenced threads are
collectable under retention policy.

### Ratified rules

1. **Materialized contexts.** A provider request is built from the target
   thread's own log, never from a computed view over some other log. What
   the model saw is the log; forensics and provider caching need no
   recomputation and no view-function versioning.
2. **Explicit pollution.** Content moves between threads only via journaled
   append effects carrying provenance (source thread/entry, author
   participant, generation run id). The foreign-message rendering rule
   (labeled `participant-message` block, reasoning replay stripped) is the
   default renderer for such appends.
3. **Single ticker.** An active thread has exactly one weave that ticks it —
   drives its turns. Other weaves may reference and read it. A thread with
   no ticker is dormant: readable, referenceable, not running. Ticker
   transfer is an explicit journaled effect. This replaces any finer-grained
   turn-lease scheme.
4. **Curate, never conceal.** The driver's presentation structure can only
   *reference* threads and entries. Drill-down from the weave to any raw
   referenced thread is guaranteed ground truth in every client.
5. **Compatibility.** Every legacy thread loads as a singleton weave: the
   compatibility driver, one referenced thread it ticks, degenerate
   presentation (one primary-transcript block). Same idiom as legacy
   threads deserializing into the user+agent participant pair.

### Rejected: shared log with per-participant projections

The previous target (one shared thread log; each model participant's context
computed by `project_body_for`-style projection) was rejected 2026-07-31.
The corpse, for future re-litigators:

- A projection is a monotone function of the shared log, so it can only
  express contexts that grow on the conversation's own axis. A permission
  checker's context grows per-check, wants a curated seed and custom prompt,
  and (if persistent) its own KV lineage — a different growth axis entirely.
- Nontrivial view policies require a selector language in which every
  program must be proven monotone or provider prefix caching silently dies
  (a last-N window is the canonical violation). Prefix stability should be
  structural, not a property policed per selector.
- Forensics: "what did model X see at turn N" becomes a recomputation
  parameterized by projection-function version; changing the rendering rule
  silently rewrites what historical requests "were."

What survives from that machinery: the foreign-message rendering rule (now
the pollution renderer, rule 2), message authorship/run-id provenance, and
computed composition for the *UI* — the one consumer for whom views are
exactly right, since cache-safety is irrelevant and the human wants an
interleaved composite no single thread contains.

## Compatibility foundation (landed, steps 1–4)

The landed slice is substrate for the weave model; nothing in it is
discarded, but several pieces migrate up a tier when weaves land.

- `ParticipantId`, `ThreadParticipant(s)` in the shared protocol;
  `ThreadConfig.participants` defaulting to the historical user+agent pair;
  optional `Message.author` with legacy role-based inference. Authorship and
  generation-run ids are the provenance vocabulary for pollution effects.
  → registry and default input/responder routing migrate to the weave.
- Per-participant execution profiles: creation requests resolved once under
  pod ceilings into frozen `ParticipantExecutionProfile` snapshots (model,
  max_tokens, system prompt, bindings, narrowed scope, tool surface, frozen
  tool schemas, scripted opening context). → profile *definitions* migrate
  to the weave; each thread instantiates exactly one. The
  `thread_id:participant` request-cache-key suffix becomes unnecessary once
  each context is its own thread.
- `Conversation::project_body_for(participant)` — retained as the pollution
  renderer and for the landed intra-thread compatibility path.
- The compatibility driver `builtin_single_agent_chat`
  (`src/runtime/driver.rs`): pure policy functions consuming turn boundaries
  and emitting `RunAgent` / `DispatchTools` / `Continue` / `Finish`; mutable
  cycle state in `Thread.driver_state`; legacy `turns_in_cycle` migrates on
  load. → driver state and journal migrate from Thread to the weave.
- The durable effect journal: every driver request recorded with a monotonic
  id and persisted pending/completed/failed/interrupted outcome before
  external I/O; `AwaitingModel`/`AwaitingTools` carry their pending effect
  id; restart healing marks in-flight effects interrupted. The scheduler
  flushes dirty threads before polling lazy provider/tool futures — a
  write-ahead boundary without per-dispatch synchronous filesystem I/O.
  This journaling discipline extends unchanged to cross-thread effects.

## Weave contract (target)

A driver consumes durable events from the threads its weave references and
emits effect requests. It does not mutate scheduler state or touch
provider/tool handles. The scheduler remains the single writer: it admits
effects against the pod ceiling and the weave/thread scopes, journals them
before external I/O, and completes them back into driver state.

Target effect vocabulary (supersedes the earlier single-thread list):

```text
append_entry(thread, author, content, provenance)   -- pollution included
run_agent(thread, participant?)                     -- tick one turn; which
                                                    -- registered Model voice
                                                    -- speaks (default: the
                                                    -- thread's responder).
                                                    -- (Earlier drafts listed a
                                                    -- `limits` arg; never
                                                    -- implemented — per-cycle
                                                    -- caps come from max_turns
                                                    -- on the thread.)
call_tool(thread, tool, arguments)
derive_thread(definition, seed, relationship)       -- fork/compact/check/...
advance_head(thread)                                -- promote to primary (step 8)
adopt_ticker(thread) / release_ticker(thread)
emit_event(payload)
await_input(selector)
finish(outcome)
```

(An `update_presentation(structure)` effect appeared in earlier drafts of
this list; it was rejected when 7b was ratified — see Presentation. The
display is a pure function of driver state, not an imperative effect.)

`run_agent` remains a compound effect containing the tested
model → tools → same-model loop; decomposing its sub-turns is not required
before multi-thread arrangements work. Relationship metadata on
`derive_thread`/`append_entry` generalizes and subsumes the existing
compaction lineage and fork/dispatch parent links.

Drivers are scripted (Lua) creations at the pod level; the built-in
compatibility driver is the degenerate program.

## Presentation

The driver assembles a typed presentation data structure that fixed
machinery in the UI renders — the driver composes blocks, it does not
paint.

**Ratified 2026-08-01: presentation is a pure function of persisted driver
state.** A scripted program may define a second entry point
`present(state) -> blocks`, evaluated in the same sandbox after each
activation. It sees only the persisted driver state, so it is replayable
by construction; there is no journal growth from display churn, and the
display cannot drift from state. The journaled-`update_presentation`
alternative was REJECTED because: every update bloats the effect journal
with display-only records, reconnect needs the latest structure persisted
anyway (so journaling buys no replay capability the pure function lacks),
and an imperative update can be forgotten, leaving a stale display that
the pure form makes unrepresentable. A program with no `present` — and
every builtin weave — gets the degenerate presentation: one
`primary_transcript` block plus the auxiliary thread list.

The block vocabulary starts deliberately minimal and grows only as real
drivers demand (a markdown panel was considered and deferred until a
driver needs it):

- `primary_transcript(thread_ref)` — the conversation head. A compaction
  roll is the driver advancing this pointer along a journaled
  `compaction` edge; the old context stays reachable via drill-down.
  Typed input targets whatever thread this block references;
  weave-routed input (`await_input`) is deferred until a driver needs it.
- `thread_list([...])` — auxiliary referenced threads (checkers, subagents).
- `status(text)` — one-line driver state for the chrome.

A malformed or erroring presentation falls back to the drill-down thread
list; presentation can degrade, ground truth cannot. Blocks referencing
threads the weave does not reference are dropped at validation — the
structure can curate, never fabricate.

## Clients

The Kotlin Android app is deprecated (ruling 2026-07-31). whisper-agent-
damascene-ui gains a mobile-responsive layout system and is wrapped as the
Android app. Wire-protocol growth for weaves therefore does not need to
preserve the hand-mirrored Kotlin protocol layer; new weave messages are
not mirrored into the Kotlin codec.

**Wire shape ratified 2026-08-01 (7b):** the weave tier lands *additively*
beside the existing thread tier. The thread tier — list broadcasts,
per-thread subscription, snapshots, streaming turn events — is untouched
and remains the single ground-truth stream. New messages carry weave
snapshots (driver identity + program hash, thread refs with roles and
relationships, presentation blocks) and presentation updates to weave
subscribers. Composition is client-side: presentation blocks *reference*
thread ids, and the client subscribes to those threads through the
existing per-thread tier, mounting their streams into presentation slots.
A server-side multiplex of thread events into weave envelopes was
REJECTED as a translation layer with no gain — drill-down would no longer
be literally the same machinery as normal rendering. `ThreadSummary`
grows `weave_id` + `weave_role` tags so the conversation list can nest
auxiliary threads under their weave's primary row (hiding auxiliaries
entirely was rejected: ground truth stays one click, not one hop, away).
The weave-first protocol rework (list weaves, subscribe by weave id,
thread tier demoted to drill-down) is deferred to the final-naming step.

## Migration sequence

1. Participant identity and actor-aware projection. **Landed.**
2. Participant-scoped generation ids on runs, usage records, transcript
   entries, and streaming events. **Landed.**
3. Participant-specific execution profiles: backend, model, system context,
   tools, bindings, and narrowed scope. **Landed.**
4. Extract the current state machine as `builtin_single_agent_chat` behind
   the driver/effect contract with a durable journal. **Landed.**
5. Introduce the weave entity: persist driver instance (program ref,
   state, journal, thread refs) separately from threads; singleton-weave
   migration for legacy threads; move driver state and journal off
   `Thread`; invert boundary policy so the scheduler routes
   `StepOutcome::Boundary` through the ticking weave. **Landed.**
   (Presentation and thread refcounting deferred to steps 6–7 with their
   first real consumers.)
6. Cross-thread effect vocabulary: `derive_thread`, provenance-carrying
   `append_entry`, single-ticker admission with `adopt_ticker`/
   `release_ticker`. **Landed.** Semantics chosen: a derived thread is
   referenced and ticked by the deriving weave from birth (release makes
   it dormant); its base scope is the primary thread's active scope;
   `append_entry` is pure transcript pollution (no wake) and is refused
   while the target is mid-generation, preserving the materialized-log
   forensics rule; input to a dormant thread is rejected at the input
   path rather than failing the thread, while queued work (knowledge
   nudges, async dispatch follow-ups) stays queued until a weave adopts
   it, and compaction is refused outright; a derived thread's caps
   override composes by narrowing against the primary scope, never
   assignment; sweeps unreference from every weave and retire emptied
   ones, and pod archival drops weaves by pod ownership (not the ticker
   index) so dormant-only weaves don't leak. Ticker claims persist per-ref
   (`ticks`, defaulting true for step-5 JSON); conflicting claims are
   healed at load by demoting all but the first-loaded ref and flushing
   the demotion, so the winner sticks across restarts.
7. First scripted (Lua) driver plus the minimal presentation vocabulary.
   Exercise case: the auto-mode permission checker — a derived thread with
   curated seed, custom prompt, tools disabled, intercepting tool admission —
   chosen because it stresses exactly the cases the rejected projection
   model could not express.
   **Driver half landed** (7a; sequencing ratified 2026-07-31: driver
   first, presentation after, shaped by what working drivers need).
   Contract: a program at `<pod>/drivers/<name>.lua` defines
   `on_event(state, event) -> { effects, state }` — an event handler
   with explicit JSON state, deliberately not a coroutine (a parked
   coroutine can't be snapshotted or replayed after restart). Events
   are the thread-tagged boundaries plus `input_accepted` and
   `thread_derived`; effects are thread-targeted
   run/dispatch/resolve/continue/finish plus the step-6 cross-thread
   vocabulary. Each event runs in a fresh sandboxed VM (no io/os,
   memory + instruction budgets); the weave snapshots the program's
   sha256. Tool interception works by parking: the driver leaves the
   primary at its agent boundary (the step loop breaks instead of
   spinning; boundaries re-fire, so handlers are idempotent), runs the
   checker, then emits `resolve_tools` — per-tool admission whose
   denials become synthesized error tool_results in one batch.
   `examples/drivers/auto_mode_checker.lua` is the working exercise
   case, tested end-to-end (deny and allow paths) in the scheduler
   harness.
   **Presentation half (7b) ratified and landed 2026-08-01:** pure
   `present(state)` (see Presentation), additive wire tier with
   client-side composition (see Clients), `ThreadSummary` weave tags for
   list nesting, and a minimal damascene-ui weave view (coordination
   strip with driver badge / status / auxiliary jump chips, sidebar
   nesting of auxiliaries) so the vocabulary is shaped by a real
   renderer. External input is admitted only to non-auxiliary threads
   of a scripted weave — input targets the presentation head; an
   auxiliary is the driver's workspace, and input landing there would
   bulk-interrupt in-flight coordination (refused at the input path,
   compose box replaced by a hint in the drill-down view).
   Known gaps accepted at 7b close: a driver has no thread-removed
   event, so archiving a live multi-thread weave's primary leaves
   `state.primary` naming a gone thread — validation drops the head
   block and the display runs headless until the next cycle (display
   only; revisit when the event vocabulary next grows). The Kotlin
   codec would mis-decode `WeaveSnapshot` (its `snapshot` key
   collides with `ThreadSnapshot`'s discriminator heuristic) — 
   unreachable today since Android cannot subscribe to weaves, and
   the client is deprecated; do not route weave messages onto
   broadcast tiers while any Kotlin client lives.
8. Move compaction onto weave machinery: head-advance along a `compaction`
   edge; delete the compaction-specific in-flight bit, internal originator,
   state hook, lineage field, and wire lifecycle.
   **Ratified 2026-08-01 (all four forks):**
   - *Old head becomes a dormant auxiliary* — ref kept (drill-down
     guaranteed), `ticks=false`; input paths refuse it like any dormant
     thread and fork is the deliberate revive. The live-auxiliary
     alternative (preserving the legacy you-can-keep-typing behavior)
     was REJECTED: every compaction would permanently add another live
     thread to the weave, and typing into compacted-away context is a
     footgun, not a feature.
   - *Head-advance is a journaled effect* — `advance_head(thread)`
     promotes a referenced thread to primary and demotes the old
     primary per the rule above. Builtin compaction journals
     `derive_thread` + `advance_head`, the same records a scripted
     driver would produce; Lua drivers can therefore compose their own
     compaction from primitives. Builtin-internal ref-twiddling was
     REJECTED because the journal must explain why the primary changed
     (explanatory-journal invariant) and scripted drivers need the
     same primitive.
   - *Wire lifecycle deleted* — `ThreadCompacted`, its Function
     terminal payload, and `ThreadSummary`/`ThreadSnapshot.continued_from`
     are gone. The roll rides the weave tier: damascene-ui follows the
     head (when a `WeaveSnapshot` shows the selected thread demoted
     along a `compaction` edge, it selects the new primary); the
     summary text is readable as the continuation's seed message. A
     transition shim keeping `ThreadCompacted` firing was REJECTED as
     two sources of truth for one transition.
   - *Legacy `continued_from` is ignored, not migrated* — the field is
     deleted from the struct; old JSON keys drop off on next save
     (Completed threads rarely resave, so raw files keep their
     lineage for `pod_show_thread` indefinitely). Load-time synthesis
     of weave edges from historical chains was REJECTED: walking
     N-length chains and reconciling both threads' existing singleton
     weave JSONs is the fiddliest code in the slice for a display
     nicety.
   - Standing calls: the in-progress marker lives in builtin weave
     driver state (not on Thread); `CompactThread` survives as the
     manual trigger with admission "thread is its weave's current
     primary" (which also replaces the auto-compact `continued_from`
     scan); scripted weaves still refuse both triggers — they get the
     primitives, and a compaction driver event waits until a real
     driver needs one.
   **Multi-agent enablers landed post-8 (2026-08-01):** `run_agent`
   takes an optional `participant` (which registered Model member
   speaks; unknown/non-Model ids refuse and journal — profile
   resolution would otherwise silently fall back). Harness proofs:
   two threads of one weave await models concurrently with
   out-of-order completions resolving their own records, and two
   participants alternate voices in one thread (`finish_cycle` +
   `run_agent(participant)` — `begin_model_call` admits Idle/Completed,
   not a parked boundary). **Journal per-thread precision** (the gap
   accepted since step 6) is retired: cycle records (`run_agent`,
   `dispatch_tools`, `resolve_tools`, `continue`, `finish`) carry the
   thread they ran on, and interrupt/fail resolution is thread-scoped
   (`*_pending_for`); records persisted before the attribution match
   any thread, and the load path's interrupt-everything remains the
   one legitimate bulk form (every thread heals at startup).
9. First real multi-agent conversation driver.
   **Ratified 2026-08-01 (three forks):**
   - *Tangled-threads first* — each voice is its own thread (private
     context; per-voice system prompt/model/tools set by
     `derive_thread`), coordinated by the weave. The primary thread is
     a driver-maintained minutes view: user input lands there, and
     every voice reply is pollinated back as an attributed
     `append_entry`, so the existing transcript UI *is* the merged
     view and no presentation-vocabulary growth is required
     (drill-down to each voice's private thread rides the step-7b
     sidebar nesting). Shared-transcript-first was REJECTED for the
     first driver: its cast needs a new creation-time mechanism
     (evaluate the program, freeze a participant roster into
     `participant_profiles`) whereas tangled needs no new server
     surface at all; the per-participant projection layer stays
     landed, and a shared-transcript driver remains a later
     roster-slice + .lua. Both-in-one-slice REJECTED: doubles the
     dogfooding surface before either shape is polished.
   - *Cast is program-declared* — a table in the Lua program (voice
     id, system prompt, optional model) feeding `derive_thread`
     effects at runtime; picking the driver picks the cast. A UI
     roster editor (the `ThreadOverrides` participant fields already
     ride the wire unused) stays open for later; REJECTED as the
     first mechanism because real UI work would precede any running
     conversation, and a themed driver with a user-emptied cast is
     meaningless.
   - *First policy is one-pass round-robin* — per user input each
     voice takes one turn in declared order; a voice's thread receives
     the input plus earlier voices' replies as attributed entries
     before its `run_agent`, and the round ends with `finish_cycle`
     on the primary. Multi-round debate and moderator-selected
     speakers are later .lua files, not architecture.
   Required UI slice: transcript entries render their `author`
   (per-entry authorship already rides the wire on every message;
   no client displays it, and an unattributed minutes view is
   unreadable).
   **Code half landed 2026-08-01** (live dogfood pending):
   damascene-ui renders voice chips (deterministic per-author accent,
   gutter + caption; the user fill stays reserved for genuine input);
   `input_accepted` events carry the accepted text (parity with
   `agent_completed`); `examples/drivers/roundtable.lua` implements
   the ratified policy (voice-tagged `derive_thread` relationships map
   cast ids to threads; stacked input queues and rounds run
   back-to-back before one primary `finish_cycle`); the scheduler
   harness proves two full rounds end-to-end including attribution
   order in the minutes, cross-pollination into idle voice contexts,
   presentation status, and cast reuse on the second round.
   **Review-driven vocabulary growth:** `thread_failed` — a
   coordinated thread dying outside the driver's own effects
   (model/tool I/O failure on the Failed transition, external cancel)
   is reported to scripted ticking weaves; without it a coordinating
   driver waits forever on a completion that can never arrive (the
   review's one HIGH: a mid-round voice failure permanently wedged
   the weave, and a Failed thread refuses `run_agent`, so the dead
   voice could never speak again). Deliberately NOT fired for
   driver-fault failures (an erroring program would error again on
   the notification) or during load-path healing (drivers don't run
   at load). The roundtable skips the dead voice's turn, unmaps it,
   and derives a fresh-context replacement next round; the harness
   pins the failure path and the stacked-input path through the real
   io-completion layer.
   **Restart healing (ratified as the arc closer):** no new event
   kind — `load_state` collects every ticked thread of a scripted
   weave found dead after persister healing (durable facts, so a
   restart-before-first-activation re-detects them) and the weave's
   first activation replays them as deferred `thread_failed` events
   ahead of the triggering event, primary-role deaths first (an
   auxiliary's death reported while the head's is pending would
   advance stale coordination). Review hardening: "dead" includes
   `Cancelled` (refuses `run_agent` like Failed, and a cancel can
   reach the thread's JSON while the weave's post-notification driver
   state does not — threads flush before weaves); a failed first
   drain re-stashes undelivered death facts instead of dropping them;
   weave removal (sweep, pod archive) clears pending notices;
   driver-fault deaths, suppressed live, DO replay at load (the
   program may have been fixed in between — contract text updated). `thread_failed` is therefore an
   idempotent FACT ("this thread is dead"), re-reportable across
   restarts — drivers must no-op on already-handled deaths. A
   mid-round restart heals both the parked primary and the in-flight
   voice, so the roundtable voids the stale round and the healing
   input runs clean; surviving voices keep their contexts.
   Accepted gaps at step-9 close: `input_accepted` cannot
   distinguish human input from `send_tool_result_text`'s
   machine-rendered callbacks (unreachable for the roundtable —
   voices are tools-off and the primary never dispatches — wants a
   `source` field before a driver both coordinates and dispatches);
   a replacement voice does not inherit the dead voice's private
   history (fresh context; entry-copy vocabulary would be needed).
10. Driver-declared configuration knobs (ratified 2026-08-08, three forks
   via AskUserQuestion — pulled ahead of builtins-as-drivers because both
   pending consumers demand it: title generation wants a dedicated model,
   and the roundtable wants per-voice "provider x, model y" casts).
   A driver optionally defines a third pure entry point beside
   `on_event`/`present`: `describe() -> { label?, description?, knobs }`,
   evaluated in the same sandbox with no state. **Declaration source
   ratified: in-program `describe()`** — this deliberately reverses step
   9's deferral of creation-time program evaluation, now that a consumer
   demands it; being code, it keeps one source of truth (the roundtable
   generates one model knob per CAST entry) and needs no repeated-group
   schema machinery. REJECTED: sidecar schema file (`<name>.knobs.json`)
   — two sources of truth that drift, and no dynamic knob generation.
   **Schema language ratified: a typed knob vocabulary**, a flat list of
   `{ id, label, type, default, constraints }` with v1 types `model`
   (value is a `{backend, model}` pair, rendered by the client's existing
   backend/model pickers and validated against the backend catalog),
   `string` (optional `multiline` for prompts), `boolean`,
   `integer`/`number` (min/max), `select` (static options). REJECTED:
   CDDL / JSON-Schema subset — structural expressiveness a generic form
   cannot render, no domain types (`model` would degrade to a bare string,
   losing catalog validation and the picker widget), plus a parser
   dependency; validation errors also stop mapping cleanly onto form
   fields. **Lifecycle ratified: values frozen at creation** — collected
   by the new-thread form (a `DescribeDriver` request fetches the knob
   list when the picker selects a scripted driver), submitted inside
   `ThreadDriverConfig::Scripted { name, config }`, validated server-side
   against the declaration (refusing creation on mismatch), and
   snapshotted onto the `Weave` beside `driver_program_hash` per the
   "weave snapshots what explains its behavior" invariant. REJECTED for
   now: editable knobs with a `config_changed` event — deferred until
   demanded; the escape hatch is starting a new weave. Delivery: config
   rides every activation as an extra argument — `on_event(state, event,
   config)` and `present(state, config)` — never seeded into driver
   state (state stays purely driver-computed; Lua ignores extra
   arguments, so existing drivers are untouched). Creation-time
   validation is the only validation: a program edited after creation may
   see stale or missing knob values and must tolerate `nil` like any
   table access. Companion effect growth: `derive_thread` gains an
   optional `backend`, mapped onto the `ThreadBindingsRequest` parameter
   `weave_derive_thread` already accepts — without it a knob-configured
   cross-provider voice has no way to land on its provider.
   **Both halves landed 2026-08-08** (4d58d39 server, 70d3add UI).
   Protocol: `whisper-agent-protocol/src/driver.rs` (KnobKind/KnobSpec/
   DriverDescription + structural `validate_config`, flat spec structs
   by design — internal tagging fought the Lua bridge in step 7);
   `ThreadDriverConfig::Scripted` gained the `config` map (serde-default
   empty; `Eq` dropped from the enum for `serde_json::Value`); wire pair
   `DescribeDriver`/`DriverDescribed` with in-band `error` so the form
   can pin authoring failures to the picker. Server: `lua::run_describe`
   beside run_event/run_present (absent function or nil return = empty
   declaration); creation validates in `create_task` right where the
   program-load check lived — structural validation plus the pod
   `allow.backends` check on model knobs — and rewrites the submitted
   map with defaults materialized before the weave snapshots it via its
   `driver` field; both activation paths destructure the frozen map and
   pass it as the third VM argument. `load_driver_program`'s existing
   name sanitization covers the new request. The roundtable's
   `describe()` generates one optional model knob per CAST seat;
   replacements re-read the same frozen knob, so a configured seat keeps
   its provider across voice deaths (harness-pinned, including refusal
   messages naming the knob and the materialized-default assertion).
   UI: knob rows render between the runtime pickers and the message
   editor; model knobs are chained backend/model menus over the
   server-known catalog (per-backend model lists fetched on demand),
   select/boolean are single menus, string/integer/number bind keyed
   text inputs (multiline strings a text_area) re-parsed per edit;
   incomplete model pairs are withheld from submission. Accepted gaps
   at landing: no client-side required/bounds enforcement (the server's
   named refusal is the backstop); a model knob's model menu opens
   empty until its backend is picked; knob values are not counted in
   the overrides-modal count (they render in the main pane).
   **Reviewed (fixes in 840399a), verdict sound, no HIGH findings.**
   The review verified the load-bearing structure: every creation path
   that can carry `Scripted{config}` flows through create_task's
   validation (behavior startup and derive_thread cannot carry it;
   compaction is gated off scripted weaves; fork copies an
   already-frozen map), both VM call sites deliver the frozen config on
   every activation including restart-healing replays, and the
   child-within-parent-scope invariant holds on the new derive-backend
   plumbing because `weave_derive_thread` supplies the primary's scope
   as `base_scope_override`, which errors on out-of-scope backends.
   Hardening taken: `validate_declaration` at describe() evaluation
   (ids non-empty/unique/colon-free — UI route keys are string-glued
   from ids; selects must declare options); `run_describe` normalizes
   `knobs = {}` (previously a type error — empty Lua tables are
   ambiguous) and loudly refuses map-shaped knobs (previously silently
   zero knobs); integer knobs accept whole-valued floats (Lua `/`
   always yields floats — a computed default refused every creation);
   the create-time model-knob check also enforces the dispatching
   parent's `scope.backends` (was pod-allow only — a dispatched
   scripted thread's bad knob refused only at the eventual derive);
   UI same-driver re-pick refetches describe() (the retry path after
   a failed eval or program edit). Reviewer-accepted lows: forking a
   scripted primary whose program was deleted mints a weave that fails
   at first event, and fork's `reset_capabilities` silently resets the
   driver to builtin (both pre-existing fork semantics); a derive from
   a scripted weave whose primary was archived out from under it
   resolves bindings against full pod scope (pre-existing structure,
   unexercised); `DescribeDriver` is the first client-triggerable Lua
   eval — synchronous on the scheduler thread, bounded by the sandbox
   budgets, same exposure class as per-activation loads; the knob
   backend menu lists the full server catalog rather than pod allow
   (consistent with the existing backend picker; the named refusal
   covers it).
11. Title generation, autoquery, behavior startup, and dispatch callbacks as
   weave drivers, as concrete cases justify.
   **Slice 1 ratified 2026-08-08** (three forks via AskUserQuestion):
   model-based title generation via the driver contract. Ground truth
   that shaped the slice: the pre-existing "title generation" is
   `derive_title` — a pure truncation of the first user message, no
   model call anywhere — so this slice ADDS model titling rather than
   migrating machinery. New effect `set_title(thread, title)`:
   admission is reference-only (the target must be a thread the weave
   references — curate-what-you-coordinate; ticking not required),
   last-write-wins, journaled. Last-write-wins is load-bearing: the
   scheduler's truncation title fires on first input for scripted
   primaries too, so a driver's model title overwrites the placeholder
   and a failed title model gracefully leaves the truncation in place.
   **Carrying case ratified: `examples/drivers/titled_chat.lua`** — the
   degenerate single-agent chat loop in Lua (turn_start→run_agent,
   dispatch_tools, continue, finish — the "builtin compatibility driver
   is the degenerate program" made literal) plus titling: on the first
   completed reply, derive a one-turn tools-off title thread seeded
   with the opening exchange, model from an optional `title.model`
   knob (unset = pod default), and `set_title` the primary with the
   cleaned reply. Doubles as the builtin-parity proof the rest of step
   11 needs before any builtin converts. REJECTED for this slice:
   converting the builtin chat driver now (forces the pod-config model
   surface and builtin-driver-state questions before the vocabulary is
   proven); roundtable-only titling (leaves no chat-shaped driver).
   **Lifecycle ratified: the title thread lingers as a referenced
   auxiliary** (relationship `title`) — the title's provenance stays
   inspectable in drill-down, consistent with curate-never-conceal; no
   unref/archive vocabulary pulled in (front-3 item stays deferred).
   **Roundtable rider ratified**: the roundtable gains the same
   optional `title.model` knob and titles its minutes when the first
   round closes.
   **Slice 1 landed** (07939ed; review pending at time of writing).
   `ScriptedEffect::SetTitle` + `PersistedDriverEffect::SetTitle
   {thread_id, title}` (the journal explains where a name came from) +
   `weave_set_title` executor following the adopt_ticker
   admission/journal shape (trimmed; empty titles refused; refusals
   journal Failed). titled_chat.lua and the roundtable both derive the
   title thread with `max_turns = 1`, clean the reply in Lua
   (trim/unquote/collapse/strip-period/clip-60 on UTF-8 boundaries;
   an empty cleaned title skips set_title but still closes the title
   thread's cycle), never retry a dead title model, and drop the title
   thread from presentation once done (drill-down keeps it). Roundtable
   test pins updated for the always-on title thread (ref/journal
   counts); the flagship test drives the title end-to-end (placeholder
   → cleaned model title), and titled_chat tests pin the knob's
   backend+model on the title thread's bindings plus the
   placeholder-stands failure path. Incidental fix folded in: two
   clippy needless-borrow errors in the knob form (the session had
   been running clippy without `-- -D warnings`; CI would have
   refused 70d3add/840399a).
   **Reviewed (fixes in da79a4e), verdict sound, no HIGH findings.**
   Clean traces confirmed: last-write-wins ordering (truncation fires
   only on `title.is_none()`, before the driver sees input);
   `turn_start` cannot reach the title thread (fires only from
   NeedsModelCall; auxiliary input refused; hallucinated tool_use
   closed by finish_cycle's orphan synthesis); input mid-title-flight
   is per-thread isolated; no double-finish (mechanical max_turns
   triggers only on a second run_agent neither driver issues);
   restart heals a mid-flight title thread via the dead-ticked
   collection and both drivers' thread_failed checks the title branch
   first. Fixes taken: dispatch-tools coverage (new `respond_tool`
   harness helper; the titled_chat test now runs a full
   dispatch→respond→continue round-trip — previously NO scheduler
   test exercised `ScriptedEffect::DispatchTools`); clean_title
   re-trims after unquoting so the Lua emptiness guard agrees with
   the scheduler's trim-then-refuse (a whitespace-only quoted title
   would have failed the effect and flipped the Completed title
   thread to Failed); defensive pod check on set_title (parity with
   append_entry/adopt_ticker; invariant already held by
   construction); un-spliced adopt_ticker's rustdoc (the set_title
   insertion had split doc from body) and dropped its vestigial
   `#[allow(dead_code)]`; restart-test wording. Accepted, recorded:
   a failing derive at the roundtable's round close drops the
   remaining effects and fails the origin voice without a
   thread_failed (driver-fault deaths fire nothing live) — a
   PRE-EXISTING hazard class shared with replacement-voice derives in
   the same position, deserving its own consideration; codepoint
   clip can split grapheme clusters (cosmetic).
   **Slice 2 ratified 2026-08-08** (three forks via AskUserQuestion):
   behavior startup — a behavior can spawn a scripted weave. Ground
   truth that shaped the slice: `run_behavior` already flows through
   `create_task` (which validates `Scripted{config}` end-to-end since
   step 10) and already supplies the behavior's fire-time scope as
   `base_scope_override`; the only reason a behavior can't start a
   roundtable today is that `BehaviorThreadOverride` hardcodes
   `driver: None`. The slice is therefore mostly plumbing plus one
   piece of new vocabulary:
   - *Authoring surface: TOML-only.* `[thread]` gains `driver` (program
     name) and a `[thread.driver_config]` table of knob values (TOML
     values crossing into the step-10 JSON knob map verbatim; `model`
     knobs are inline `{backend, model}` tables). The structured
     behavior editor only renders a subset of `BehaviorConfig` and
     preserves non-exposed fields on save, so the new fields survive
     editor round-trips unrendered; behaviors stay fully authorable as
     files. REJECTED: shipping the editor's driver picker + knob form
     in the same slice — real sheet-form composition work would land
     before the server semantics have been dogfooded; it reuses the
     step-10 components and becomes its own follow-up slice.
     Validation stays fire-time-only (each fire re-validates against
     the program as it exists then — the behavior analogue of step
     10's "creation-time validation is the only validation"), with one
     parse-time addition: `driver_config` without `driver` refuses at
     load like any config error.
   - *Run-done is driver-declared* — new journaled effect
     `complete_run{outcome?, message?}` (outcome defaults `completed`;
     `failed` carries the message into `BehaviorOutcome::Failed`). The
     scheduler resolves the declaring weave's origin-carrying thread
     and routes the declaration into behavior bookkeeping (run_count,
     last_outcome, overlap-queue release). REJECTED: primary-terminal
     as run-done (the recommended-and-declined option): it matches the
     builtin hook's semantics for free, but a scripted primary goes
     Completed at every `finish_cycle` while the weave's actual work
     (title thread in flight, later multi-phase drivers) continues —
     only the driver knows when the triggered unit of work is done.
     Consequences ratified with it: the declaration must be an effect,
     not presentation (presentation is non-authoritative by
     invariant; the journal must explain why run_count moved);
     mechanical death of the origin thread stays as the
     Failed/Cancelled backstop (first fact wins — the existing
     idempotence guard arbitrates declaration-vs-death races); a
     Completed origin thread in a scripted weave defers to the
     declaration, so a driver that never declares holds the run
     in-flight and the overlap queue open — an authoring contract
     note, visible in the behavior state UI, bounded by weave death.
     A `complete_run` from a weave with no behavior origin journals
     with `behavior_id: None` and moves nothing — drivers declare
     done unconditionally and stay origin-agnostic.
   - *Retention treats the weave as the unit.* The sweep's candidate
     rule keys on `BehaviorOrigin`, which only the spawned primary
     carries — auxiliaries (voices, title threads) would outlive every
     sweep and a daily cron roundtable would leak a headless weave per
     day. Ratified: when the origin thread belongs to a scripted
     weave, candidacy requires every thread the weave references to be
     terminal and the window is measured from the newest `last_active`
     across members ("the weave has been idle N days"); the action
     then sweeps every member (each already tears down refs via
     `sweep_thread`, emptying and retiring the weave). A member
     referenced by another weave is left unswept (its ref keeps the
     weave alive — accepted, unreachable for behavior-spawned
     drivers). REJECTED: propagating `BehaviorOrigin` onto derived
     threads (voices sit Completed between rounds and would be swept
     out from under a live weave); skipping scripted weaves entirely
     (leaks by design).
   Rider: the create-time model-knob backend check gains fire-scope
   parity — it enforces `base_scope_override` (the behavior's
   fire-time scope) the way it enforces a dispatching parent's scope;
   bindings already narrowed against it, knobs did not. Both example
   drivers declare `complete_run` at title resolution (title completed
   or title thread dead — the points where their triggered work is
   actually finished).
   **Slice 2 landed 2026-08-08.** Protocol: `BehaviorThreadOverride`
   `driver` + `driver_config` (TOML values crossing to the JSON knob
   map verbatim, pinned by a parse test incl. an inline model-knob
   table; `Eq` dropped); `to_create_thread_requests` maps them to
   `Scripted{name, config}`; `DriverConfigWithoutDriver` parse
   refusal. Server: `ScriptedEffect::CompleteRun` (typed
   `ScriptedRunOutcome`, default `completed`) →
   `PersistedDriverEffect::CompleteRun{outcome: BehaviorOutcome,
   behavior_id}` → `weave_complete_run` executor (origin resolved by
   walking the weave's refs for the origin-carrying thread; no-origin
   declarations journal `behavior_id: None`); the terminal-hook tail
   refactored into the shared `record_behavior_run_outcome` (guard +
   state + broadcast + queued re-fire) with the hook deferring
   Completed for scripted-ticked threads; `behavior_has_inflight_run`
   holds the Skip/QueueOne gate while a scripted run is undeclared
   (`last_outcome` None + live scripted ticker); weave-unit retention
   with reference-resolved membership and gone-ref pruning; fire-scope
   model-knob check in create_task. Harness firsts: `install_behavior`
   + the first end-to-end behavior-fire tests (declaration arc incl.
   deferral + gate + journal provenance, death backstop via the
   production step-pairing, queued-payload release, fire-scope knob
   refusal, weave-unit sweep + working-member deferral + dormant
   origin + gone-ref prune).
   **Reviewed (same session), verdict sound after fixes.** The review's
   HIGH was real: the overlap gate keyed only on the last thread's
   terminal state, so a scripted primary's per-cycle Completed opened
   Skip/QueueOne while the run was undeclared — the ratified "holds
   the overlap queue open" claim existed only in this document. Fixed
   in `behavior_has_inflight_run` as above. Also taken: both drivers
   now clear `state.title_thread` in the completed-title branch
   (an external cancel of the lingering auxiliary could re-enter the
   failed-title branch and declare twice — `CancelThread` has no
   terminal-state precondition and cancel transitions Completed →
   Cancelled); retention membership by reference (ticker-keyed lookup
   silently demoted a dormant-origin weave — the `advance_head` roll
   shape — back to the leaking per-thread sweep) plus gone-ref
   pruning with stale-ticker cleanup (a ref to a vanished thread made
   the weave an unsweepable zombie once its origin thread swept);
   journal doc wording ("routed to", the recorder's guard may still
   discard). Accepted, recorded for a future pass:
   - *Single-slot idempotence guard* — `(last_thread_id,
     last_outcome)` is one slot, so with `overlap = allow` (or any
     interleaving) a late fact about run N after run N+1 fired can
     re-record N (e.g. declared-then-dies-mid-round: Completed then
     backstop Failed both count). Pre-existing weakness, amplified by
     scripted weaves outliving their recorded run; the fix is a
     per-run recorded fact (candidate: stamp on `BehaviorOrigin` or
     the weave), which is a design decision, not a patch.
   - *Origin search can mis-credit in adversarial compositions* — an
     origin-less weave that `adopt_ticker`s someone else's dormant
     origin-carrying thread would record runs against that behavior.
     Unreachable with the example drivers (a behavior-spawned weave's
     own primary ref is first); the clean fix is stamping behavior
     identity on the weave at spawn, same candidate as above.
   - *Idle dormant member defers retention forever* — a
     derived-but-never-run thread (weave died right after derive)
     never reaches a terminal state, so its unit never sweeps.
   - *Flush-order loss windows* (threads → weaves → behaviors):
     crash after weave flush / before behavior flush leaves a
     declared run permanently unrecorded (driver state says resolved,
     no re-declaration comes); crash after the title thread flushes
     Completed but before the weave flush leaves the driver waiting
     on a completion that never replays — the latter is the
     pre-existing scripted-weave hazard class, now also holding a run
     unrecorded.
   **Slice 3 ratified 2026-08-09** (four forks via AskUserQuestion):
   autoquery — driver-initiated knowledge retrieval, the first async
   non-thread effect. Ground truth that shaped the slice: the builtin
   "autoquery" is scheduler machinery keyed on `task.config.autoquery`,
   not builtin-driver behavior — after every successful model call it
   extracts query text from the response (reasoning-first by default),
   fires an async embed→search→rerank over hot in-scope buckets, and
   injects the formatted nudge at the next `NeedsModelCall` boundary,
   HOLDING that model call while a query is in flight so the nudge
   lands in time. On scripted threads it is half-alive: queries launch
   (the trigger is config-keyed and scripted threads inherit pod
   config) but injection and the wait-gate live in
   `step_until_blocked`'s loop head, which scripted stepping bypasses
   (`scripted_run_agent` / `continue_cycle` call `begin_model_call`
   directly) — an autoquery-enabled pod pays embed+rerank per scripted
   turn while nudges rot in the queue, injectable only stale at a
   parked turn boundary (where injection also re-delivers
   `turn_start`).
   - *Surface: a query effect; the builtin machinery gates off
     scripted threads.* New effect `query_knowledge{id, query,
     buckets?, top_k?, snippet_chars?, hot_only?}` with completion
     events `query_completed{id, query, hits}` / `query_failed{id,
     message}`; the driver owns trigger, formatting, injection, and
     dedup in Lua. `maybe_launch_knowledge_autoquery` skips
     scripted-ticked threads — no more silent embed+rerank cost with
     undeliverable nudges. REJECTED: config passthrough (making the
     builtin machinery work under driver-gated stepping puts
     scheduler-owned injection and wait-gates inside the scripted
     executors — the scheduler delaying and prefixing driver-decided
     model calls with content the driver never sees); both surfaces
     active (double-nudge risk, and the passthrough work buys nothing
     once drivers own the loop). Async-effect pattern established
     here for every future non-thread effect: driver-supplied
     correlation `id` echoed in the completion (a duplicate id while
     one is in flight refuses and journals); journaled pending at
     issue like `RunAgent`, resolved on completion; restart heals
     pending query records to `query_failed` at the weave's first
     activation (the dead-ticked-thread shape — an idempotent fact,
     not an edge); completed records are inert at load. `min_score`
     is deliberately absent: hits carry rerank scores and the driver
     filters in Lua — the loop is driver-owned, the scheduler does
     not pre-judge relevance.
   - *Injection: nudge-on-continue.* `continue_cycle` gains an
     optional `nudge` text param appended atomically as the same
     system-authored message shape `submit_server_nudge` uses, before
     the model call it triggers. `append_entry`'s mid-generation
     refusal stays untouched (turn-boundary injection already works
     through it — `NeedsModelCall` is not mid-generation). REJECTED:
     relaxing append_entry at the post-tools parked boundary (sound
     there — results are integrated — but it carves a state-specific
     exception into a deliberate guard, and `AgentBoundary` must stay
     refused or an entry splits tool_use from tool_result);
     turn-boundary-only (loses the flagship case: retrieval landing
     between tool rounds inside a working cycle). Atomicity is
     load-bearing, not convenience: a `continue_cycle` + `append_entry`
     pair in one effects list only sequences correctly for the event's
     origin thread — any other thread is stepped re-entrantly by the
     continue executor and the append lands mid-generation, refused.
     *Consequence found at implementation — the hold is a parked
     boundary and async handlers store, never move.* "Hold the
     continue" = park the tools boundary (return no effects); after a
     query resolves the scheduler steps every thread the weave ticks,
     so the parked boundary re-fires and its handler finds the stored
     nudge and continues. Moving the held thread from the
     `query_completed`/`query_failed` handler instead works live but
     breaks across a restart: a healed `query_failed` notice and the
     re-fired boundary share one drain, and the boundary event queued
     behind the notice goes stale the moment the notice's handler
     moves the thread. Contract documented on the events; both example
     patterns journal identically.
   - *Cold buckets: per-effect `hot_only`, default true.* Ambient
     nudge loops stay hot-only by authoring convention; a driver may
     explicitly request a cold-capable query (the scheduled-digest
     shape). Driver code is operator-authored and journaled — the
     `knowledge_query` tool's trust class, not ambient config's.
     REJECTED: hard hot-only (a digest driver would have to route
     retrieval through a model turn just to touch cold data).
   - *`agent_completed` gains `reasoning`* (empty string when none) —
     the same parity argument that put `text` on the event; the
     builtin's default `query_source` is reasoning-then-text and a
     thinking-heavy model's terse text would starve a text-only
     query. Reasoning is already persisted in thread journals — no
     new exposure class. Authoring note, not mechanism: drivers that
     stash reasoning into state bloat their persisted JSON.
   - Scope and bounds: bucket refs use the config grammar (bare,
     `server:name`, `pod:name`) resolved against the weave's POD
     knowledge ceiling (no participant narrowing — the driver is not
     a participant; empty means every in-scope bucket); `top_k`
     defaults 5, max 20, zero refused — the `knowledge_query` tool's
     bounds, not new numbers; `snippet_chars` bounds chunk text
     crossing into Lua, default 500 (the builtin nudge default). Hits
     carry `bucket`, `source_id`, `chunk_id`, `locator`, `score`,
     `snippet` — enough for Lua-side dedup keys matching the
     builtin's (source-or-chunk per bucket).
   **Carrying case:** titled_chat grows an `autoquery` bool knob
   (default false) implementing the builtin loop in Lua: query on
   tool-bearing responses from reasoning-then-text, hold `continue`
   until the result arrives, inject via the continue nudge, dedup
   seen hits in driver state, suppress when the model called
   `knowledge_query` itself. Extends its role as the builtin-parity
   proof. Roundtable rider deliberately skipped this slice: no
   natural per-round nudge point without new design work.
   **Slice 3 landed 2026-08-09.** `ScriptedEffect::QueryKnowledge` /
   `ScriptedEvent::{QueryCompleted, QueryFailed}` +
   `ScriptedKnowledgeHit` (event `Eq` dropped for the f32 scores);
   `ContinueCycle.nudge`; `AgentCompleted.reasoning`;
   `PersistedDriverEffect::QueryKnowledge{query_id, query, buckets}`
   (pending at issue, buckets = the labels actually queried) and
   `Continue.nudge_entry` (the journal explains the injected entry);
   `weave_query_knowledge` executor (static faults journal Failed AND
   fail the activation — the run_agent audit shape; environmental
   refusals journal Failed and queue `query_failed` onto the same
   activation's drain, the thread_derived delivery shape);
   `SchedulerCompletion::ScriptedQuery` → completion applier settles
   the record, then steps every ticked thread of the weave so parked
   boundaries re-fire (the store-don't-move mechanism); hot path
   grabs cache-loaded buckets at admission, `hot_only = false`
   snapshots (slot, serving mode) and loads inside the future — the
   tool's exact shape; builtin `maybe_launch_knowledge_autoquery`
   gates off scripted-ticked threads. Restart: the shutdown bulk
   interrupt now EXEMPTS async non-thread records (they must stay
   Pending), the load scan queues loss notices without touching the
   journal, and the record fails AT DELIVERY — crash anywhere before
   the notice lands re-derives it next load, the same
   rebuild-at-load contract as the dead-ticked scan; thread-scoped
   bulk resolvers (`fail_pending_for` / `interrupt_pending_for`)
   also skip async records, so another thread's death can no longer
   settle a flying query's record out from under it. Harness: five
   tests — end-to-end refusal-never-stalls (pins reasoning-then-text
   through real effect emission), the flagship
   hold→complete→re-fire→nudge arc incl. snippet clip, journaled
   `nudge_entry`, and dedup-to-nothing (via a manufactured in-flight
   standing in for a launch — the real engine path needs bucket +
   provider fixtures the harness doesn't have; noted, not hidden),
   duplicate-id-is-a-journaled-driver-fault, and the
   restart-heal arc pinning Pending-until-delivery. Lua tests pin the
   full effect decode and the event's Lua-side shape.
   **Reviewed (same session), verdict sound after fixes.** The
   review's HIGH was real and structural: the persist layer's
   shutdown interrupt resolved every pending record at load, so the
   heal scan (which filters on Pending) was dead code on the
   production path and a query in flight at shutdown wedged its
   driver forever — the heal test had passed only because it called
   the scan directly, bypassing persist. Fixed by the async-record
   exemption + resolve-at-delivery above (the test now runs the
   shutdown interrupt first). Also taken from review: the thread-
   scoped bulk-resolver exemption (any thread's failure matched the
   query's attribution-less record); static faults now journal
   (ratified text said "refuses and journals"; the first cut
   silently dropped); the continue-nudge entry now broadcasts a
   ThreadSnapshot like both sibling injection paths (a live viewer
   otherwise saw the reply reference material it couldn't see).
   Accepted, recorded: `nudge_entry` is a moment-in-time index
   (compaction/insert can shift it — the pre-existing
   `AppendEntry.entry_index` weakness, forensic-only); builtin
   autoquery leftovers on an adopted thread still inject once after
   a builtin→scripted ticker adoption (bounded, self-clearing);
   titled_chat's stored nudge can go stale across an external
   cancel (hits were genuinely unseen; self-limiting); the real
   launch→flight→completion path is fixture-untestable in the
   harness today (fake embed/rerank providers + a fake Bucket are
   feasible but nontrivial — a future harness investment).
   **Slice 4 ratified 2026-08-10** (four forks via AskUserQuestion):
   async dispatch callbacks via the driver contract. Ground truth
   that shaped the slice: `dispatch_thread(sync=true)` is just a
   parked tool call and already works on scripted threads — the
   slice is entirely the `sync=false` path, where the builtin
   renders a `<dispatched-thread-notification>` envelope and
   injects it as fresh input on the parent. On scripted weaves that
   path was worse than half-alive: a primary's callback rode
   `input_accepted` indistinguishable from human input (the
   roundtable would deliberate a round on an XML envelope), an
   auxiliary's callback was silently DESTROYED (the 7b input guard
   refuses auxiliaries, and both delivery paths drop the envelope
   on refusal — the queued path dequeues first), and the Function
   registry is memory-only, so restart orphans every async callback
   for all weaves even though the child survives and its terminal
   state is durable on disk.
   **Delivery ratified: driver events, builtin injection gated off
   scripted-ticked parents** (the autoquery gate-off mirror; here
   the gate is delivery-shape selection at tool registration).
   The driver decides what a callback means: append + run, feed
   coordination state, or drop. Auxiliary dispatches Just Work
   (delivery is weave-level, tagged with the dispatching thread).
   This DISSOLVES the step-9 `input_accepted` source gap rather
   than answering it: dispatch callbacks were the only machine text
   on the input path, so once they're gated off, no `source` field
   is needed. REJECTED: `source` field on `input_accepted` (keeps
   the injection the driver can't suppress or reroute; auxiliaries
   still lose callbacks); both-event-and-source (vocabulary with no
   consumer).
   **Restart ratified: journal + reconnect.** Scripted async
   dispatches journal a pending
   `DispatchCallback{tool_use_id, child_thread_id}` record on the
   dispatching weave (exempt from bulk resolvers via slice 3's
   `is_async_non_thread`), indexed by an in-memory watcher map
   keyed by child thread id, rebuilt from pending records at load.
   Child still live at load → watcher re-arms; child already
   terminal (the common case — the persister heals mid-flight
   children to Failed before `load_state`) → notice queued for the
   weave's first activation, built from the child's persisted final
   state; child gone → failure notice. Resolve-at-delivery: the
   record stays Pending until the driver hears the event, so a
   crash anywhere before delivery re-derives the notice next load.
   Unlike a query (whose future dies with the process), the
   dispatch payload is durable — undelivered completions re-stash
   losslessly instead of degrading to loss notices. Scripted async
   dispatch thereby becomes reliable across restart, EXCEEDING the
   builtin, which keeps its lossy restart until conversion.
   REJECTED: parity punt (driver waits forever on an event that
   never comes, or must track dispatch age defensively).
   **Event shape ratified: split events, raw fields.**
   `dispatch_completed{thread_id (the dispatching parent),
   tool_use_id, child_thread_id, result, usage{total_tokens,
   tool_uses, duration_ms}}` and `dispatch_failed{thread_id,
   tool_use_id, child_thread_id, message}`; cancellation folds into
   `dispatch_failed` with a cancel message. Correlation is by
   `tool_use_id`, which `agent_completed.tool_calls` already
   exposes — a driver sees itself dispatch. Drivers format their
   own notification text in Lua (visible user-role attributed
   provenance via `append_entry`). REJECTED: single event + status
   enum (breaks the `*_completed`/`*_failed` vocabulary symmetry);
   carrying the builtin's pre-rendered XML envelope (redundant
   payload; the webui's envelope-reattachment path doesn't apply to
   driver-appended attributed entries anyway).
   **Store-don't-move amended: scoped to parked boundaries.** A
   dispatch result usually arrives while the parent is QUIESCENT —
   no boundary will ever re-fire, so the strict rule would strand
   the result until the next human input. The contract now reads:
   async handlers may move quiescent threads (`append_entry` +
   `run_agent` both admit Idle/Completed — the dispatch handler
   appends the notification and runs a fresh turn, matching builtin
   behavior); when the thread is parked at a boundary the handler
   must store and let the re-fired boundary move it, exactly as
   slice 3 ratified. The driver always knows its own parking
   discipline, and `run_agent` on a parked thread refuses loudly
   (activation fault) rather than corrupting anything. After
   delivery the applier steps every ticked thread (the slice-3
   shape), so parked boundaries re-fire either way. REJECTED:
   strict store-don't-move plus a new wake/idle boundary event
   (a new event kind whose only consumer is this corner, and one
   extra driver activation of latency on every quiescent delivery).
   **Slice 4 landed 2026-08-10.**
   `ScriptedEvent::{DispatchCompleted, DispatchFailed}` +
   `ScriptedDispatchUsage`;
   `PersistedDriverEffect::DispatchCallback{parent_thread_id,
   tool_use_id, child_thread_id}` in `is_async_non_thread()` (the
   record is scheduler-authored — the model called the tool — but
   the journal owns it because the pending record is what survives
   restart). Registration: the async branch of
   `register_dispatch_thread_tool` selects delivery shape by the
   parent's ticker (scripted → `FunctionDelivery::None` + journal +
   watcher; the Function keeps creation and cascade-cancel only);
   `launch_create_thread` now RETURNS the child id and the dispatch
   path calls it directly — the registry read-back was unreliable
   when a fast child terminated during launch (the same edge the
   watcher re-checks after arming). Incidental fix: async-dispatch
   creation failure now returns a tool ERROR (previously acked
   success with an empty child id). Live delivery:
   `complete_scripted_dispatch_for_child` fires at every terminal
   fan-out site (step tail, cascade-cancel, execute_cancel_thread),
   resolves the record, delivers with weave primary as origin, and
   steps all ticked threads (the slice-3 shape). Restart:
   `heal_pending_scripted_dispatches` re-arms watchers for live
   children and derives notices from persisted final state for
   terminal/gone ones; notices drain in the activation preamble
   after query losses, resolving records at delivery; the salvage
   arm re-stashes dispatch terminals LOSSLESS in both directions.
   Weave removal purges watchers + notices at all three sites.
   titled_chat: `busy`/`dead` flags gate flush-vs-store; quiescent
   flush = `append_entry` (author "dispatch") + `run_agent`; dedupe
   by (tool_use_id, child). Harness: five tests — the flagship
   round trip (incl. the builtin followup queue staying empty), the
   busy stash, cancelled-child reporting, parent-cancel containment,
   and the restart reconnect arc (shutdown-interrupt exemption,
   Pending-until-delivery, REAL result after the crash window).
   **Reviewed (same session), verdict sound after fixes.** Fixes
   taken: the activation cap check now runs BEFORE the pop, only
   when another event is waiting — the cap-tripping event previously
   dropped unprocessed, which for loss facts and dispatch terminals
   (records already resolved in the preamble) was permanent loss;
   titled_chat marks the primary `dead` on its `thread_failed` and
   stores dispatch terminals until input revives it — previously a
   plain UI cancel of a parent with a child in flight cascaded into
   `run_agent` against the corpse, faulting the activation and
   clobbering the user's Cancelled with a driver Failed (regression
   test added); the driver dedupe keys (tool_use_id, child), robust
   to backends reusing tool ids across turns. Accepted, recorded:
   a driver FAULTING on the dispatch event itself loses that
   delivery (the record resolved pre-VM; the activation fails
   loudly, the weave needs its program fixed anyway, and the result
   stays readable on the child thread — resolving post-VM would
   invert the write-ahead discipline; same pre-existing shape as a
   fault on `query_completed`); a crash between the thread flush
   (notification appended) and the weave flush (record + dedupe
   state) re-appends the notification once at reload —
   at-least-once, the documented contract; the ticker-adoption race
   can leave SEVERAL stale builtin injections (one queued envelope
   per idle boundary), not one — same accepted class, wording
   corrected here; a dormant-but-alive child at load re-arms a
   watcher that waits for external input to revive the child —
   record stays Pending, nothing lies or leaks, near-unreachable
   since dispatched children are seeded and stepped at creation.

10. **Step 11 slice 5 ratified 2026-08-10: compaction parity for
   scripted drivers.** Ground truth that shaped the slice: the
   builtin's four policy functions were already titled_chat's Lua
   verbatim, and slices 1–4 covered titling, autoquery, and async
   dispatch — compaction is the LAST builtin capability the scripted
   contract lacks. The scripted side was deliberately stubbed:
   manual `CompactThread` refuses scripted weaves ("a driver
   composes compaction from weave primitives"), `maybe_auto_compact`
   early-returns on scripted tickers, and while the primitives exist
   (`append_entry`, `derive_thread` + seed, `advance_head`), a
   driver could not learn "compact now" (no trigger event, no usage
   visibility on `agent_completed`), could not run the configured
   Rust-regex `summary_regex`, and could not reproduce the
   continuation's inheritance (setup prefix copied verbatim so pod
   drift doesn't leak, profiles realigned, config + bindings
   carried — intricate tested Rust inside
   `finalize_builtin_compaction`). Four forks ratified:
   - *Staging: parity first.* This slice puts compaction in the
     scripted contract and titled_chat (completing the parity
     proof); the conversion proper — pod `thread_defaults.driver`
     surface, program resolution for a universal default,
     legacy-weave migration, builtin deletion — is the NEXT slice.
     REJECTED: one big slice (review surface too large); convert
     first (regression window: new threads would lose /compact and
     auto-compact until parity landed).
   - *Trigger: driver self-detect.* `agent_completed` gains usage
     payloads — `usage` (the completed call) and `thread_usage`
     (the thread's cumulative `total_usage`, the operand of the
     builtin's threshold check), each with the four `Usage` fields.
     titled_chat declares an Integer knob
     `compaction.token_threshold` (0/unset = off, matching the
     builtin's manual-only default) and compares cumulative input
     tokens at its cycle-closing boundary. The scheduler's own
     auto-trigger stays builtin-only; its scripted early-return is
     now permanent law, deleted with the builtin next slice.
     REJECTED: scheduler-emitted auto trigger (uniform config
     surface, but policy belongs to the driver; the thread-config
     `token_threshold` splitting meaning across driver kinds is the
     accepted cost); both-at-once (surface without a consumer).
   - *Resolved-config round trip.* The driver needs the thread's
     resolved compaction texts (pod-relative `prompt_file`,
     `summary_regex`, `continuation_template`) — pod-dir resolution
     is scheduler business. New effect
     `request_compaction {thread_id}`: the scheduler validates
     (weave ticks the thread, thread is the current primary,
     `compaction.enabled`) and answers with driver event
     `compaction_ready {thread_id, reason: "driver", prompt,
     summary_regex, continuation_template}`; refusal journals the
     effect failure and delivers
     `compaction_refused {thread_id, message}` so the driver can
     clear its own marker (the query_failed precedent — effects
     must not fail invisibly). The manual client message on a
     scripted weave stops refusing: it branches BEFORE Function
     registration, runs the same validation, and delivers
     `compaction_ready {reason: "manual"}`; refusals bounce to the
     client as a correlated Error, but success sends NO positive
     ack (resolved at review, replacing this entry's earlier "and
     acks" wording) — the visible UX is the driver appending the
     summary prompt and streaming the turn, and a client wanting a
     correlated ack is a UI-slice protocol question, not a driver
     one. The Function registry's `CompactThread` (and its
     terminal shape) stays builtin-only and dies with the builtin. Idle-ness is NOT
     checked scheduler-side for scripted weaves — the driver knows
     its own parking discipline and stores the request until the
     cycle closes (better than the builtin, which rejects non-idle
     outright).
   - *Continuation inheritance: granular reference-fields on
     `derive_thread`.* Three optional by-reference copy directives,
     each naming a thread the weave references: `setup_from` (copy
     the source's setup prefix verbatim — system prompt + tool
     manifest — realigning the default responder's frozen profile;
     exclusive with `system_prompt`/`disable_tools`, the
     combination refuses at execution), `config_from` (copy
     participants, profiles, model, max_tokens, max_turns,
     tunables, compaction, autoquery, and the origin marker;
     explicit `model`/`backend`/`max_turns` layer on top),
     `bindings_from` (copy backend + named host_env + mcp_hosts).
     The machinery is `finalize_builtin_compaction`'s snapshot
     code refactored into shared helpers — the builtin finalize
     consumes the same helpers until it dies. titled_chat's
     compaction derive sets all three to the old head; its title
     derive keeps using none. REJECTED: a single `inherit_from`
     bundle (opinionated composite; composable pieces chosen —
     "much easier to get subtly wrong" accepted as the cost of
     flexibility); inline content fields (Lua holding manifest
     bytes).
   - *Extraction: Rust-regex helper.* The Lua environment gains
     `regex_capture(pattern, text)` — group-1 capture (whole match
     when the pattern has no groups) or nil + error message,
     backed by the linear-time regex crate (no catastrophic
     backtracking from driver-supplied patterns). The configured
     `summary_regex` rides `compaction_ready` and runs verbatim;
     the default's end-anchored nested-tag handling is preserved.
     REJECTED: hand-rolled Lua extraction (custom `summary_regex`
     configs silently unsupported; Lua patterns cannot express the
     end-anchor).
   - *titled_chat flow* (the parity-proof shape): threshold
     crossing or `compaction_ready` while busy/dead stores a
     wanted-marker; at quiescence, append the prompt (author
     "user") + `run_agent`, mark prompted; the summary turn's
     `agent_completed` extracts via `regex_capture` — success
     derives the continuation (all three `_from` refs + seeded
     continuation template), `thread_derived` advances the head,
     updates `state.primary`, and runs the first turn; extraction
     failure finishes the cycle and clears the marker (builtin
     parity: thread left Completed, no continuation, no retry).
     `compaction_refused` and a primary `thread_failed` clear the
     marker. Restart mid-summary-turn abandons the compaction
     (thread healed to Failed → `thread_failed` clears) — the
     builtin's documented restart contract, unchanged.

   **Slice 5 landed 2026-08-10** (same session as ratification).
   Contract: `usage`/`thread_usage` on `agent_completed`
   (`ScriptedCallUsage`, the four protocol Usage fields);
   `compaction_ready`/`compaction_refused` events;
   `request_compaction` effect (journal record
   `PersistedDriverEffect::RequestCompaction` — synchronous, never
   rests Pending); `derive_thread` grew
   `setup_from`/`config_from`/`bindings_from` with the builtin
   finalize's machinery refactored into shared helpers
   (`inherited_config_override`, `inherited_bindings_request`,
   `setup_prefix_snapshot`, `apply_setup_snapshot` in
   compaction.rs) — the builtin path verified field-for-field
   unchanged; `_from` validation failures journal a Failed
   DeriveThread record before faulting; `regex_capture` injected
   into the sandbox. Manual routing branches in
   `apply_client_message` ahead of Function registration;
   `deliver_manual_compaction` runs the driver and steps ticked
   threads origin-first; `weave_request_compaction` answers in the
   same activation drain. titled_chat: `compaction.token_threshold`
   Integer knob + the cp machine. Harness: seven tests (manual roll
   incl. a pod-drift trap on the setup copy, threshold self-detect,
   extraction-failure retry, three refusal shapes, restart
   abandonment + retry, exclusivity fault journaling,
   input-supersedes) + three Lua contract tests.
   **Reviewed (same session), verdict sound after fixes.** Fixes
   taken: (1) titled_chat's `input_accepted` on the primary now
   sets `busy` (the cycle begins at acceptance — resource warmup
   parks a thread in a state that admits `append_entry` yet
   refuses `run_agent`, so a manual compact in that window
   previously faulted the primary) and CLEARS any in-flight
   compaction (input supersedes: a reply to the user's message is
   never regex-tested as a summary, and a marker wedged by a
   faulted roll — driver faults fire no `thread_failed`, so
   nothing else clears it — now heals on the next message instead
   of blocking compaction forever); regression test added. (2)
   `regex_capture` hardened: the instruction budget meters Lua
   instructions, not Rust time, so both operands are bounded
   (pattern 64 KiB compiled, text 1 MiB) and arguments arrive as
   Lua strings with UTF-8 refusals as error RETURNS — previously
   a non-UTF-8 argument raised an uncatchable conversion error
   that faulted the activation; group-1 semantics made strict (a
   non-participating group returns nil, not the whole match). (3)
   The manual-path ack contradiction resolved in favor of the
   code: no positive ack (see the amended bullet above). Accepted,
   recorded: a steady stream of dispatch terminals starves a held
   compaction (each close prefers the flush; every deferral does
   real work); the scripted continuation carries no placeholder
   title (the builtin's rode `send_user_message`'s derive_title —
   cosmetic); the fork-mid-compaction guard is builtin-only
   (scripted cp is opaque JSON; forking mid-summary copies the
   prompt into the fork — revisit when the builtin and its guard
   die next slice); `thread_derived`/`compaction_ready` are not in
   the activation-cap salvage arm (a >16-VM-call activation can
   drop them; titled_chat's flows never approach the cap —
   pre-existing class shared with title derives); a torn flush
   (threads before weaves) can leave `cp="prompted"` on disk with
   no summary turn — the next close fails extraction and
   self-heals, and input clears it too.

## Open questions (flagged, not ratified)

- Entry storage: copies on pollution (accepted initially; fan-out is small)
  vs. a pod-level entry store with threads as ordered entry-refs plus
  rendering directives (recovers deduplication; revisit if duplication
  bites).
- Presentation block vocabulary growth beyond the ratified minimal three
  (markdown panel, entry-range references) as drivers demand.
- Final naming ("weave" provisional; protocol `Conversation` struct may want
  a `Transcript` rename if "conversation" is ever surfaced for the tier).
  The weave-first client protocol rework belongs to the same step.

## Non-negotiable invariants

- Persisted weaves snapshot the normalized definition and driver/program
  version needed to explain their behavior later.
- Participant and effect scopes only narrow the owning pod/thread ceiling.
- Provider requests are built only from the target thread's own materialized
  log — no computed cross-log views feed a model.
- Provider-private replay never crosses thread boundaries by default.
- An active thread has exactly one ticker; ticker changes are journaled.
- External effects remain scheduler-admitted, journaled before I/O, and
  auditable.
- Presentation references content; it cannot conceal or fabricate it.
  Drill-down to raw threads is always available.
- Existing thread JSON and single-agent clients remain readable throughout
  the staged migration.
