# Configurable Threads and Weaves

Status: **steps 1–5 landed** — participant foundation, execution profiles,
the compatibility driver with its durable effect journal, and the weave
entity itself: every thread is coordinated by a singleton weave, driver
policy is inverted out of `Thread` (boundary outcomes routed through the
ticking weave by the scheduler), and driver state/journal persist at
`<pod>/weaves/<weave_id>.json`. The architectural target was re-ratified
2026-07-31 into the pod/weave/thread model below; steps 6+ have not
started.

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
run_agent(thread, limits)                           -- tick one turn
call_tool(thread, tool, arguments)
derive_thread(definition, seed, relationship)       -- fork/compact/check/...
adopt_ticker(thread) / release_ticker(thread)
update_presentation(structure)
emit_event(payload)
await_input(selector)
finish(outcome)
```

`run_agent` remains a compound effect containing the tested
model → tools → same-model loop; decomposing its sub-turns is not required
before multi-thread arrangements work. Relationship metadata on
`derive_thread`/`append_entry` generalizes and subsumes the existing
compaction lineage and fork/dispatch parent links.

Drivers are scripted (Lua) creations at the pod level; the built-in
compatibility driver is the degenerate program.

## Presentation

The driver assembles and updates a typed presentation data structure that
fixed machinery in the UI renders — the driver composes blocks, it does not
paint. The structure must be replayable: a pure function of persisted driver
state, or maintained through journaled `update_presentation` effects, so a
reconnecting client rebuilds the display without private channel state.

The block vocabulary starts deliberately minimal and grows only as real
drivers demand:

- `primary_transcript(thread_ref)` — the conversation head. A compaction
  roll is the driver advancing this pointer along a journaled
  `compaction` edge; the old context stays reachable via drill-down.
- `thread_list([...])` — auxiliary referenced threads (checkers, subagents).
- `status(text)` and a markdown panel.

A malformed or erroring presentation falls back to the drill-down thread
list; presentation can degrade, ground truth cannot.

## Clients

The Kotlin Android app is deprecated (ruling 2026-07-31). whisper-agent-
damascene-ui gains a mobile-responsive layout system and is wrapped as the
Android app. Wire-protocol growth for weaves therefore does not need to
preserve the hand-mirrored Kotlin protocol layer.

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
   `release_ticker`.
7. First scripted (Lua) driver plus the minimal presentation vocabulary.
   Exercise case: the auto-mode permission checker — a derived thread with
   curated seed, custom prompt, tools disabled, intercepting tool admission —
   chosen because it stresses exactly the cases the rejected projection
   model could not express.
8. Move compaction onto weave machinery: head-advance along a `compaction`
   edge; delete the compaction-specific in-flight bit, internal originator,
   state hook, lineage field, and wire lifecycle.
9. Title generation, autoquery, behavior startup, and dispatch callbacks as
   weave drivers, as concrete cases justify.

## Open questions (flagged, not ratified)

- Entry storage: copies on pollution (accepted initially; fan-out is small)
  vs. a pod-level entry store with threads as ordered entry-refs plus
  rendering directives (recovers deduplication; revisit if duplication
  bites).
- Final presentation block vocabulary beyond the minimal set.
- Wire shape for weave subscription/streaming (clients subscribe to a weave;
  thread events route into presentation slots via run/participant ids).
- Final naming ("weave" provisional; protocol `Conversation` struct may want
  a `Transcript` rename if "conversation" is ever surfaced for the tier).

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
