-- Roundtable: the first real multi-agent conversation driver
-- (migration step 9; tangled-threads shape).
--
-- The thread this weave ticks (the primary) is a MINUTES view: the
-- user speaks into it, and every voice's reply is appended back into
-- it attributed to the speaker. Each voice lives in its own derived
-- thread — a private context that accumulates the same conversation
-- from its own seat: its replies are native model turns there, and
-- everyone else's words arrive as attributed entries (the request
-- projection wraps those in <participant-message> markers).
--
-- Policy: one-pass round-robin. Per accepted input every voice speaks
-- once, in CAST order; later voices see earlier replies from the same
-- round. The round closes the primary's cycle at the end, reopening
-- its compose box. Input stacked while a round runs queues, and
-- rounds run back-to-back until it drains.
--
-- Failure: a voice thread dying mid-round (provider failure, cancel —
-- the scheduler delivers `thread_failed`) skips its turn; the dead
-- thread is unmapped and the next round derives a replacement with a
-- fresh context (a Failed thread refuses run_agent, so replacement is
-- the only revival; the replacement does not remember earlier
-- rounds). A restart mid-round heals in-flight threads to Failed and
-- the scheduler replays those deaths as thread_failed facts at the
-- weave's first activation — the primary's death voids the stale
-- round, so the next input starts cleanly. thread_failed may
-- re-report a death this driver already handled; every handler here
-- is a no-op for unmapped threads.
--
-- The cast is program-declared: edit CAST, copy this file under
-- <pod>/drivers/, and pick it at thread creation. Per voice: `id`
-- (the attribution everyone sees), `prompt` (its private system
-- prompt), and optional `model` (nil = the pod default). At creation
-- the new-thread form additionally offers one model knob per voice
-- (describe() below) — a chosen {backend, model} overrides the CAST
-- entry, so a cross-provider panel needs no program edit.
--
-- v1 limits: voices run with tools disabled.

local CAST = {
  {
    id = "optimist",
    prompt = "You are `optimist`, one voice at a small roundtable with "
      .. "the user and other panelists. Other panelists' words arrive "
      .. "wrapped in <participant-message> markers. Champion what could "
      .. "work: find the promising angle and build the strongest version "
      .. "of the idea under discussion. If another panelist has spoken, "
      .. "engage their point directly. Stay concrete and brief — a short "
      .. "conversational paragraph, no headings.",
  },
  {
    id = "skeptic",
    prompt = "You are `skeptic`, one voice at a small roundtable with "
      .. "the user and other panelists. Other panelists' words arrive "
      .. "wrapped in <participant-message> markers. Probe what could "
      .. "fail: surface the weakest assumption and say what evidence "
      .. "would settle it. If another panelist has spoken, engage their "
      .. "strongest point, not a strawman. Stay concrete and brief — a "
      .. "short conversational paragraph, no headings.",
  },
}

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

-- Trim, unquote, collapse whitespace, drop a trailing period, clip.
local function clean_title(text)
  local t = string.match(text or "", "^%s*(.-)%s*$")
  t = string.match(t, '^"(.*)"$') or string.match(t, "^'(.*)'$") or t
  t = string.gsub(t, "%s+", " ")
  if string.sub(t, -1) == "." then t = string.sub(t, 1, -2) end
  return clip(t, 60)
end

local function voice_of(state, thread_id)
  if not state.voices then return nil end
  for _, voice in ipairs(CAST) do
    if state.voices[voice.id] == thread_id then return voice.id end
  end
  return nil
end

-- Speak `text` into the minutes and into every other voice's context,
-- attributed to `speaker`.
local function speak(state, speaker, text, from_thread, effects)
  effects[#effects + 1] = { kind = "append_entry",
    thread_id = state.primary, author = speaker, text = text,
    source_thread_id = from_thread }
  for _, voice in ipairs(CAST) do
    if voice.id ~= speaker and state.voices[voice.id] then
      effects[#effects + 1] = { kind = "append_entry",
        thread_id = state.voices[voice.id], author = speaker, text = text,
        source_thread_id = from_thread }
    end
  end
end

-- Give every voice the user's words, then hand the first voice the
-- floor. Only called with a complete cast (ensure_cast).
local function start_round(state, text)
  local effects = {}
  state.queue = {}
  for _, voice in ipairs(CAST) do
    effects[#effects + 1] = { kind = "append_entry",
      thread_id = state.voices[voice.id], author = "user", text = text,
      source_thread_id = state.primary }
    state.queue[#state.queue + 1] = voice.id
  end
  state.speaking = table.remove(state.queue, 1)
  effects[#effects + 1] = { kind = "run_agent",
    thread_id = state.voices[state.speaking] }
  return effects
end

-- Derive any missing voice threads (first round, or replacements for
-- dead voices), or start the round directly when the cast is whole.
-- When derives are needed the input waits in state.awaiting until
-- thread_derived drains them. A per-voice model knob (creation-frozen
-- config) beats the CAST default; replacements re-read the same knob,
-- so a configured voice keeps its provider across deaths.
local function ensure_cast(state, text, config)
  state.voices = state.voices or {}
  local derives = {}
  for _, voice in ipairs(CAST) do
    if not state.voices[voice.id] then
      local knob = config and config["voice." .. voice.id .. ".model"]
      derives[#derives + 1] = { kind = "derive_thread",
        relationship = "voice:" .. voice.id,
        system_prompt = voice.prompt,
        model = knob and knob.model or voice.model,
        backend = knob and knob.backend or nil,
        disable_tools = true,
        source_thread_id = state.primary }
    end
  end
  if #derives > 0 then
    state.deriving = #derives
    state.awaiting = text
    return derives
  end
  return start_round(state, text)
end

-- Advance the floor after the current speaker's turn resolved (reply
-- or death): run the next voice, or close the round — opening the
-- next one immediately if input stacked up (the primary's cycle stays
-- open across back-to-back rounds; one finish closes however many
-- stacked submissions opened it).
local function advance_floor(state, effects, config)
  state.speaking = table.remove(state.queue, 1)
  if state.speaking then
    effects[#effects + 1] = { kind = "run_agent",
      thread_id = state.voices[state.speaking] }
    return
  end
  -- First round closed: ask a (knob-configured) model to title the
  -- minutes. One shot — a dead title thread is not replaced; the
  -- truncation placeholder stands (set_title is last-write-wins).
  if not state.title_requested then
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
        .. "Panelist: " .. (state.first_reply or "") } },
      source_thread_id = state.primary }
  end
  if state.pending and #state.pending > 0 then
    for _, e in ipairs(ensure_cast(state, table.remove(state.pending, 1), config)) do
      effects[#effects + 1] = e
    end
  else
    effects[#effects + 1] = { kind = "finish_cycle",
      thread_id = state.primary }
  end
end

function on_event(state, event, config)
  local k = event.kind

  if k == "input_accepted" then
    state.primary = state.primary or event.thread_id
    if event.thread_id ~= state.primary then
      -- The scheduler only admits external input on the primary;
      -- guard anyway.
      return { state = state }
    end
    if not state.opening then
      state.opening = clip(event.text, 500)
    end
    if state.speaking or (state.deriving or 0) > 0 then
      state.pending = state.pending or {}
      state.pending[#state.pending + 1] = event.text
      return { state = state }
    end
    return { effects = ensure_cast(state, event.text, config), state = state }
  end

  if k == "thread_derived" then
    if event.relationship == "title" then
      state.title_thread = event.thread_id
      return { effects = { { kind = "run_agent", thread_id = event.thread_id } },
               state = state }
    end
    local voice_id = string.match(event.relationship or "", "^voice:(.+)$")
    if voice_id and state.voices then
      state.voices[voice_id] = event.thread_id
      state.deriving = state.deriving - 1
      if state.deriving == 0 then
        local text = state.awaiting
        state.awaiting = nil
        return { effects = start_round(state, text), state = state }
      end
    end
    return { state = state }
  end

  if k == "agent_completed" then
    if state.title_thread and event.thread_id == state.title_thread then
      local effects = { { kind = "finish_cycle", thread_id = event.thread_id } }
      local title = clean_title(event.text)
      if #title > 0 then
        effects[#effects + 1] = { kind = "set_title",
          thread_id = state.primary, title = title }
      end
      return { effects = effects, state = state }
    end
    local speaker = voice_of(state, event.thread_id)
    if not speaker then return { state = state } end
    if speaker ~= state.speaking then
      -- A superseded voice's reply — its round died under it (primary
      -- cancelled). Close its cycle so a later round can run it
      -- again; the floor has moved on and the reply is dropped.
      return { effects = { { kind = "finish_cycle", thread_id = event.thread_id } },
               state = state }
    end
    local effects = { { kind = "finish_cycle", thread_id = event.thread_id } }
    if not state.first_reply then
      state.first_reply = clip(event.text, 500)
    end
    speak(state, speaker, event.text, event.thread_id, effects)
    advance_floor(state, effects, config)
    return { effects = effects, state = state }
  end

  if k == "thread_failed" then
    if state.title_thread and event.thread_id == state.title_thread then
      -- The title model died; keep the truncation placeholder and
      -- don't retry (title_requested stays set).
      state.title_thread = nil
      return { state = state }
    end
    if event.thread_id == state.primary then
      -- The minutes thread died (cancel or failure): the round is
      -- void. Fresh input heals the primary and starts a new round;
      -- queued input rides along until then.
      state.speaking = nil
      state.queue = nil
      return { state = state }
    end
    local voice_id = voice_of(state, event.thread_id)
    if not voice_id then return { state = state } end
    -- The voice's thread is dead: unmap it so the next round derives
    -- a replacement, drop it from this round's rotation, and if it
    -- held the floor, move on without its reply.
    state.voices[voice_id] = nil
    if state.queue then
      for i, id in ipairs(state.queue) do
        if id == voice_id then table.remove(state.queue, i) break end
      end
    end
    if voice_id == state.speaking then
      local effects = {}
      advance_floor(state, effects, config)
      return { effects = effects, state = state }
    end
    return { state = state }
  end

  -- turn_start re-fires while the primary is parked at its input
  -- boundary; the conversation is driven off input_accepted,
  -- agent_completed, and thread_failed, so every other boundary
  -- parks.
  return { state = state }
end

-- Configuration declaration (step 10): one model knob per cast seat,
-- generated from CAST so the program stays the single source of truth.
-- All optional — an unset knob leaves that voice on its CAST `model`
-- (or the pod default).
function describe()
  local knobs = {}
  for _, voice in ipairs(CAST) do
    knobs[#knobs + 1] = {
      id = "voice." .. voice.id .. ".model",
      label = voice.id .. " model",
      type = "model",
    }
  end
  knobs[#knobs + 1] = {
    id = "title.model",
    label = "Title model",
    type = "model",
  }
  return {
    label = "Roundtable",
    description = "One-pass round-robin panel: each voice keeps a "
      .. "private thread, replies pollinate through the minutes view.",
    knobs = knobs,
  }
end

-- Presentation: the minutes is the head; while a voice holds the
-- floor say who, and the cast stays reachable for drill-down.
function present(state)
  local blocks = {}
  if state.primary then
    blocks[#blocks + 1] = { kind = "primary_transcript",
      thread_id = state.primary }
  end
  if state.speaking and state.queue then
    blocks[#blocks + 1] = { kind = "status",
      text = "round in flight: " .. state.speaking .. " speaking ("
        .. (#CAST - #state.queue) .. "/" .. #CAST .. ")" }
  end
  local voices = {}
  for _, voice in ipairs(CAST) do
    local tid = state.voices and state.voices[voice.id]
    if tid then voices[#voices + 1] = tid end
  end
  if #voices > 0 then
    blocks[#blocks + 1] = { kind = "thread_list", thread_ids = voices }
  end
  return blocks
end
