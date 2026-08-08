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
