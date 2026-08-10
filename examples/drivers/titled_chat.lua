-- Titled chat: the degenerate single-agent chat loop plus model-based
-- title generation (migration step 11, slice 1).
--
-- The chat half is deliberately the builtin driver's shape made
-- literal in Lua: run a turn at every turn boundary, dispatch
-- requested tools, continue after results, finish when a reply asks
-- for nothing — the "builtin compatibility driver is the degenerate
-- program" claim as a working artifact.
--
-- The title half: after the primary's first completed reply, derive a
-- one-turn, tools-off title thread seeded with the opening exchange,
-- run it, and set_title the primary with its cleaned reply. The
-- scheduler's first-input truncation title stands until then and
-- STAYS standing if the title model fails — set_title is
-- last-write-wins, so titling degrades gracefully. The title thread
-- lingers as a referenced auxiliary (relationship "title"): the
-- title's provenance stays inspectable in drill-down.
--
-- Knobs: `title.model` ({backend, model}) — unset falls through to the
-- pod's default model. `autoquery` (boolean, default false) — when on,
-- every tool-bearing reply fires a knowledge query built from the
-- reply's reasoning (falling back to its text), the continue after
-- tools holds until the query resolves, and unseen hits ride into the
-- next model call as a nudge: the builtin autoquery loop (step 11
-- slice 3) expressed in driver code.
--
-- Async dispatch (step 11 slice 4): when the model dispatches a child
-- with sync=false, its terminal arrives as dispatch_completed /
-- dispatch_failed. Quiescent primary: append the notification as an
-- attributed entry and run a fresh turn (the builtin injection,
-- expressed in driver code). Mid-cycle: store, and flush when the
-- cycle finishes — the movement contract on dispatch_completed in
-- driver/lua.rs.
--
-- Compaction (step 11 slice 5): the builtin flow composed from weave
-- primitives. Trigger is driver-owned — the `compaction.token_threshold`
-- knob compares the thread's cumulative input tokens at each cycle
-- close (the builtin auto-trigger's operand) and issues
-- request_compaction; the client's manual compact arrives as the same
-- compaction_ready event with reason "manual". The event carries the
-- thread's RESOLVED prompt/regex/template. Flow: append the prompt at
-- quiescence and run the summary turn; extract the <summary> with
-- regex_capture (the configured Rust regex verbatim); derive the
-- continuation inheriting the old head's setup, config, and bindings
-- (setup_from/config_from/bindings_from — pod drift must not leak
-- across the roll); advance_head; run the continuation seeded with the
-- filled-in template. Extraction failure matches the builtin: the head
-- stays Completed, no continuation, no retry. `state.cp` phases:
-- "req" (request in flight) -> "ready" (config held, waiting for
-- quiescence) -> "prompted" (summary turn running) -> "derive"
-- (continuation being created) -> nil.

local TITLE_PROMPT = "You write conversation titles. Reply with the "
  .. "title only: 3 to 8 words, no quotes, no trailing punctuation."

-- Truncate to `limit` characters on a UTF-8 boundary.
local function clip(text, limit)
  text = text or ""
  if utf8.len(text) and utf8.len(text) > limit then
    return string.sub(text, 1, utf8.offset(text, limit + 1) - 1)
  end
  return text
end

-- Last `limit` characters on a UTF-8 boundary — the query wants the
-- TAIL of a reasoning trace (its end sits closest to the next action).
local function clip_tail(text, limit)
  text = text or ""
  local n = utf8.len(text)
  if n and n > limit then
    return string.sub(text, utf8.offset(text, n - limit + 1))
  end
  return text
end

-- The builtin's dedup key: source when the chunk has one, chunk id
-- otherwise, scoped per bucket.
local function hit_key(hit)
  if hit.source_id ~= "" then
    return hit.bucket .. "\tsource:" .. hit.source_id
  end
  return hit.bucket .. "\tchunk:" .. hit.chunk_id
end

local function format_nudge(hits)
  local out = "A hot knowledge bucket surfaced material related to the "
    .. "current reasoning trace. Use `knowledge_query` if you need "
    .. "more context.\n"
  for i, hit in ipairs(hits) do
    local title = hit.source_id ~= "" and hit.source_id
      or ("chunk " .. hit.chunk_id)
    if hit.locator then title = title .. " (" .. hit.locator .. ")" end
    out = out .. i .. ". [" .. hit.bucket .. "] " .. title .. "\n"
      .. hit.snippet .. "\n"
  end
  return out
end

-- Append every stored dispatch notification to the primary and run a
-- fresh turn on it. Callers guarantee the primary is quiescent (busy
-- unset): append_entry and run_agent both admit an idle, completed, or
-- input-holding thread, and refuse loudly mid-cycle.
local function flush_dispatches(state)
  local effects = {}
  for _, text in ipairs(state.disp_pending) do
    effects[#effects + 1] = { kind = "append_entry",
      thread_id = state.primary, author = "dispatch", text = text }
  end
  state.disp_pending = nil
  effects[#effects + 1] = { kind = "run_agent", thread_id = state.primary }
  state.busy = true
  return effects
end

-- Append the held compaction prompt to the quiescent primary and run
-- the summary turn. Callers guarantee quiescence (busy unset); the
-- author is "user" for builtin parity — the model sees the prompt as
-- an ordinary user message.
local function start_summary_turn(state)
  state.cp = "prompted"
  state.busy = true
  return {
    { kind = "append_entry", thread_id = state.primary, author = "user",
      text = state.cp_prompt },
    { kind = "run_agent", thread_id = state.primary },
  }
end

local function clear_compaction(state)
  state.cp = nil
  state.cp_prompt = nil
  state.cp_regex = nil
  state.cp_template = nil
end

-- Trim, unquote, collapse whitespace, drop a trailing period, clip.
-- The FINAL trim matters: unquoting can reveal whitespace ('" "'),
-- and the scheduler trims before its empty-title refusal — the Lua
-- emptiness guard must agree or a whitespace title fails the effect.
local function clean_title(text)
  local t = string.match(text or "", "^%s*(.-)%s*$")
  t = string.match(t, '^"(.*)"$') or string.match(t, "^'(.*)'$") or t
  t = string.gsub(t, "%s+", " ")
  if string.sub(t, -1) == "." then t = string.sub(t, 1, -2) end
  t = string.match(t, "^%s*(.-)%s*$")
  return clip(t, 60)
end

function describe()
  return {
    label = "Titled chat",
    description = "Single-agent chat whose title is written by a model "
      .. "after the first reply.",
    knobs = {
      { id = "title.model", label = "Title model", type = "model" },
      { id = "autoquery", label = "Knowledge autoquery", type = "boolean",
        default = false },
      -- Absent = manual-only, the builtin token_threshold=None default.
      { id = "compaction.token_threshold",
        label = "Auto-compaction threshold (tokens)", type = "integer",
        min = 1 },
    },
  }
end

function on_event(state, event, config)
  local k = event.kind

  if k == "input_accepted" then
    state.primary = state.primary or event.thread_id
    if event.thread_id == state.primary then
      -- Input revives a dead primary; stored dispatch notifications
      -- flush at the revived cycle's finish.
      state.dead = nil
      -- The cycle effectively begins at acceptance: the turn is
      -- coming but may lag (resource warmup parks the thread in a
      -- state that admits append_entry yet refuses run_agent), so
      -- anything that would move the primary from here to turn_start
      -- must store, not start. Review fix, slice 5.
      state.busy = true
      -- Input supersedes an in-flight compaction the same way it
      -- supersedes a model call: a held request, a running summary
      -- turn, or a wedged marker from a faulted roll all clear — the
      -- reply to THIS message must never be regex-tested as a
      -- summary, and the threshold simply re-requests at the next
      -- close. Review fix, slice 5.
      clear_compaction(state)
      if not state.opening then
        state.opening = clip(event.text, 500)
      end
    end
    return { state = state }
  end

  if k == "turn_start" then
    if state.primary == nil then
      -- Driver-run turn on a seeded thread that never saw
      -- input_accepted: the conversation head is whoever turns first.
      state.primary = event.thread_id
    end
    if event.thread_id == state.primary then
      -- Cycle opening: dispatch terminals arriving from here to the
      -- finish store instead of moving the thread. A turn also proves
      -- the primary is alive again.
      state.busy = true
      state.dead = nil
    end
    return { effects = { { kind = "run_agent", thread_id = event.thread_id } },
             state = state }
  end

  if k == "agent_completed" then
    -- The title came back: name the head, close the title thread.
    if state.title_thread and event.thread_id == state.title_thread then
      state.titled = true
      -- One-shot: unmap the title thread so a later external cancel of
      -- the lingering auxiliary can't re-enter the title branches and
      -- declare the run twice.
      state.title_thread = nil
      local effects = { { kind = "finish_cycle", thread_id = event.thread_id } }
      local title = clean_title(event.text)
      if #title > 0 then
        effects[#effects + 1] = { kind = "set_title",
          thread_id = state.primary, title = title }
      end
      -- Reply delivered, title resolved: the triggered unit of work
      -- is done. For a behavior-spawned weave this records the run
      -- (step 11 slice 2); elsewhere it journals and moves nothing.
      effects[#effects + 1] = { kind = "complete_run" }
      return { effects = effects, state = state }
    end

    if event.thread_id ~= state.primary then
      -- A thread this driver never ran completed a turn; nothing to
      -- coordinate.
      return { state = state }
    end

    if #event.tool_calls > 0 then
      local effects = { { kind = "dispatch_tools", thread_id = event.thread_id } }
      if config and config.autoquery then
        -- The builtin's suppression rule: a model that queries
        -- knowledge itself needs no ambient nudge this round.
        local explicit = false
        for _, call in ipairs(event.tool_calls) do
          if call.name == "knowledge_query"
            or call.name == "drain_knowledge_nudges" then
            explicit = true
          end
        end
        if explicit then
          state.aq_nudge = nil
        elseif not state.aq_pending then
          -- Reasoning-then-text, tail-clipped: the builtin's default
          -- query source.
          local q = event.reasoning ~= "" and event.reasoning or event.text
          q = clip_tail(q, 4000)
          if #q > 0 then
            state.aq_n = (state.aq_n or 0) + 1
            local id = "aq-" .. state.aq_n
            state.aq_pending = id
            effects[#effects + 1] = { kind = "query_knowledge", id = id,
              query = q, top_k = 1 }
          end
        end
      end
      return { effects = effects, state = state }
    end

    if state.cp == "prompted" then
      -- The summary turn came back. Extract with the CONFIGURED regex
      -- verbatim; success rolls the weave, failure matches the builtin
      -- (head stays Completed, no continuation, no retry).
      local effects = { { kind = "finish_cycle", thread_id = event.thread_id } }
      local body = regex_capture(state.cp_regex, event.text)
      if body then
        -- Keep busy set: the roll is one continuous busy period, so
        -- dispatch terminals keep storing until the continuation runs
        -- (they flush at ITS first close).
        state.cp = "derive"
        local seed = string.gsub(state.cp_template, "{{summary}}",
          function() return body end)
        effects[#effects + 1] = { kind = "derive_thread",
          relationship = "compaction",
          setup_from = state.primary,
          config_from = state.primary,
          bindings_from = state.primary,
          seed = { { author = "user", text = seed } },
          source_thread_id = state.primary }
      else
        clear_compaction(state)
        state.busy = nil
        if state.disp_pending and #state.disp_pending > 0 then
          for _, e in ipairs(flush_dispatches(state)) do
            effects[#effects + 1] = e
          end
        end
      end
      return { effects = effects, state = state }
    end

    -- Cycle closing: the primary is quiescent once finish_cycle
    -- applies, so stored dispatch notifications can flush below.
    state.busy = nil
    local effects = { { kind = "finish_cycle", thread_id = event.thread_id } }
    if not state.title_requested then
      -- First completed reply: ask a (cheap, knob-configured) model
      -- for a real title. One shot — a dead title thread is not
      -- replaced; the truncation placeholder simply stands.
      state.title_requested = true
      local knob = config and config["title.model"]
      effects[#effects + 1] = { kind = "derive_thread",
        relationship = "title",
        system_prompt = TITLE_PROMPT,
        model = knob and knob.model or nil,
        backend = knob and knob.backend or nil,
        disable_tools = true,
        max_turns = 1,
        seed = { { author = "user", text = "Title this conversation:\n\n"
          .. "User: " .. (state.opening or "") .. "\n\n"
          .. "Assistant: " .. clip(event.text, 500) } },
        source_thread_id = event.thread_id }
    end
    if state.disp_pending and #state.disp_pending > 0 then
      -- Dispatch terminals landed mid-cycle; deliver them now that
      -- the cycle is closing (finish, then append + fresh turn). A
      -- held compaction waits — the flush cycle's close serves it.
      for _, e in ipairs(flush_dispatches(state)) do
        effects[#effects + 1] = e
      end
    elseif state.cp == "ready" then
      -- A compaction held while the cycle ran; the head is quiescent
      -- once finish_cycle applies, so the summary turn starts now.
      for _, e in ipairs(start_summary_turn(state)) do
        effects[#effects + 1] = e
      end
    elseif not state.cp then
      -- Self-detected trigger (the builtin auto-compaction operand:
      -- cumulative input tokens vs the knob). The answer arrives as
      -- compaction_ready in this same activation's drain.
      local threshold = config and config["compaction.token_threshold"]
      if threshold and event.thread_usage.input_tokens > threshold then
        state.cp = "req"
        effects[#effects + 1] = { kind = "request_compaction",
          thread_id = event.thread_id }
      end
    end
    return { effects = effects, state = state }
  end

  if k == "thread_derived" then
    if event.relationship == "title" then
      state.title_thread = event.thread_id
      return { effects = { { kind = "run_agent", thread_id = event.thread_id } },
               state = state }
    end
    if event.relationship == "compaction" then
      -- The roll: the continuation becomes the head (the old one is
      -- demoted to a dormant auxiliary by advance_head) and runs its
      -- seeded first turn. Stored dispatch notifications now target
      -- the new head and flush at its first close.
      state.primary = event.thread_id
      clear_compaction(state)
      state.busy = true
      return { effects = {
        { kind = "advance_head", thread_id = event.thread_id },
        { kind = "run_agent", thread_id = event.thread_id },
      }, state = state }
    end
    return { state = state }
  end

  if k == "compaction_ready" then
    if event.thread_id ~= state.primary
      or state.cp == "prompted" or state.cp == "derive" then
      -- Stale head, or already mid-flow (events can re-deliver).
      return { state = state }
    end
    state.cp_prompt = event.prompt
    state.cp_regex = event.summary_regex
    state.cp_template = event.continuation_template
    if state.busy or state.dead then
      -- Mid-cycle or corpse: hold. The close (or the revived cycle's
      -- close) starts the summary turn.
      state.cp = "ready"
      return { state = state }
    end
    return { effects = start_summary_turn(state), state = state }
  end

  if k == "compaction_refused" then
    -- Only the driver's own request can be refused this way (manual
    -- refusals bounce to the client); drop the in-flight marker so a
    -- later crossing re-requests.
    if event.thread_id == state.primary and state.cp == "req" then
      state.cp = nil
    end
    return { state = state }
  end

  if k == "tools_completed" then
    if event.thread_id == state.primary and config and config.autoquery then
      if state.aq_pending then
        -- Hold the continue until the query resolves — the builtin's
        -- wait-gate, expressed as a parked boundary. The query
        -- handlers only store facts; this boundary re-fires once the
        -- query resolves (the scheduler steps the weave's ticked
        -- threads after delivery) and the continue flows from here.
        return { state = state }
      end
      if state.aq_nudge then
        local nudge = state.aq_nudge
        state.aq_nudge = nil
        return { effects = { { kind = "continue_cycle",
          thread_id = event.thread_id, nudge = nudge } }, state = state }
      end
    end
    return { effects = { { kind = "continue_cycle", thread_id = event.thread_id } },
             state = state }
  end

  -- Store-don't-move (see the contract on query_completed in
  -- driver/lua.rs): both query handlers record the outcome and issue
  -- NO thread effects — the parked tools boundary re-fires and acts on
  -- what they stored, which stays correct when a restart drains a
  -- healed query_failed and the re-fired boundary back to back.
  if k == "query_completed" then
    if event.id ~= state.aq_pending then
      return { state = state }
    end
    state.aq_pending = nil
    state.aq_seen = state.aq_seen or {}
    local fresh = {}
    for _, hit in ipairs(event.hits) do
      local key = hit_key(hit)
      if not state.aq_seen[key] then
        state.aq_seen[key] = true
        fresh[#fresh + 1] = hit
      end
    end
    if #fresh > 0 then
      state.aq_nudge = format_nudge(fresh)
    end
    return { state = state }
  end

  if k == "query_failed" then
    if event.id == state.aq_pending then
      -- The query died (refused, engine error, or lost to a restart);
      -- the chat must not stall on retrieval that was only ever
      -- opportunistic.
      state.aq_pending = nil
    end
    return { state = state }
  end

  if k == "dispatch_completed" or k == "dispatch_failed" then
    -- Terminal facts can re-deliver across a restart (the record
    -- resolves at delivery; a crash before the flush re-derives the
    -- notice): dedupe by (tool call, child) — the scheduler-side
    -- correlation pair, robust to a backend reusing tool_use_ids
    -- across turns.
    local seen_key = event.tool_use_id .. "\t" .. event.child_thread_id
    state.disp_seen = state.disp_seen or {}
    if state.disp_seen[seen_key] then
      return { state = state }
    end
    state.disp_seen[seen_key] = true
    local text
    if k == "dispatch_completed" then
      text = "[dispatched thread " .. event.child_thread_id
        .. " completed]\n" .. event.result
    else
      text = "[dispatched thread " .. event.child_thread_id
        .. " failed: " .. event.message .. "]"
    end
    state.disp_pending = state.disp_pending or {}
    state.disp_pending[#state.disp_pending + 1] = text
    if state.busy or state.dead or not state.primary then
      -- Mid-cycle, dead, or headless: store only. Moving the primary
      -- from here would fault against a parked boundary or a corpse
      -- (a cancelled parent's cascade cancels its child, and that
      -- child's dispatch_failed arrives right behind the death
      -- notice); input revives the primary and the notifications
      -- flush at the revived cycle's finish.
      return { state = state }
    end
    return { effects = flush_dispatches(state), state = state }
  end

  if k == "thread_failed" then
    if state.title_thread and event.thread_id == state.title_thread then
      -- The title model died; keep the truncation placeholder and
      -- don't retry (title_requested stays set). The chat itself
      -- delivered its reply, so the triggered run still completed —
      -- titling is cosmetic.
      state.title_thread = nil
      return { effects = { { kind = "complete_run" } }, state = state }
    end
    if event.thread_id == state.primary then
      -- The head is a corpse (failed or cancelled) until input
      -- revives it: dispatch terminals must store, not flush — a
      -- run_agent against it would fault the activation and clobber
      -- the cancel with a driver failure.
      state.dead = true
      -- A dying summary turn abandons the compaction (the builtin's
      -- restart contract: heal-to-Failed never resumes a summary);
      -- a held request dies with the head too — the user compacts
      -- again after input revives it.
      clear_compaction(state)
    end
    return { state = state }
  end

  return { state = state }
end

function present(state)
  local blocks = {}
  if state.primary then
    blocks[#blocks + 1] = { kind = "primary_transcript",
      thread_id = state.primary }
  end
  -- Surface the title thread only while it works; once titled (or
  -- dead) it drops from the display but stays reachable via the
  -- weave's drill-down list — curate, never conceal.
  if state.title_thread and not state.titled then
    blocks[#blocks + 1] = { kind = "thread_list",
      thread_ids = { state.title_thread } }
  end
  return blocks
end
