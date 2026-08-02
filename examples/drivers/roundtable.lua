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
-- The cast is program-declared: edit CAST, copy this file under
-- <pod>/drivers/, and pick it at thread creation. Per voice: `id`
-- (the attribution everyone sees), `prompt` (its private system
-- prompt), and optional `model` (nil = the pod default).
--
-- v1 limits: voices run with tools disabled; a voice thread failing
-- mid-round stalls the round (fresh input starts a new one).

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
    if voice.id ~= speaker then
      effects[#effects + 1] = { kind = "append_entry",
        thread_id = state.voices[voice.id], author = speaker, text = text,
        source_thread_id = from_thread }
    end
  end
end

-- Give every voice the user's words, then hand the first voice the
-- floor.
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
  return { effects = effects, state = state }
end

function on_event(state, event)
  local k = event.kind

  if k == "input_accepted" then
    state.primary = state.primary or event.thread_id
    if event.thread_id ~= state.primary then
      -- The scheduler only admits external input on the primary;
      -- guard anyway.
      return { state = state }
    end
    if state.speaking or (state.deriving or 0) > 0 then
      state.pending = state.pending or {}
      state.pending[#state.pending + 1] = event.text
      return { state = state }
    end
    if not state.voices then
      -- First input: derive one thread per voice; the round starts
      -- once the whole cast exists.
      state.voices = {}
      state.deriving = #CAST
      state.pending = { event.text }
      local effects = {}
      for _, voice in ipairs(CAST) do
        effects[#effects + 1] = { kind = "derive_thread",
          relationship = "voice:" .. voice.id,
          system_prompt = voice.prompt,
          model = voice.model,
          disable_tools = true,
          source_thread_id = state.primary }
      end
      return { effects = effects, state = state }
    end
    return start_round(state, event.text)
  end

  if k == "thread_derived" then
    local voice_id = string.match(event.relationship or "", "^voice:(.+)$")
    if voice_id and state.voices then
      state.voices[voice_id] = event.thread_id
      state.deriving = state.deriving - 1
      if state.deriving == 0 then
        return start_round(state, table.remove(state.pending, 1))
      end
    end
    return { state = state }
  end

  if k == "agent_completed" then
    local speaker = voice_of(state, event.thread_id)
    if not speaker or speaker ~= state.speaking then
      -- Not the floor-holding voice (or a re-delivered boundary).
      return { state = state }
    end
    local effects = { { kind = "finish_cycle", thread_id = event.thread_id } }
    speak(state, speaker, event.text, event.thread_id, effects)
    state.speaking = table.remove(state.queue, 1)
    if state.speaking then
      effects[#effects + 1] = { kind = "run_agent",
        thread_id = state.voices[state.speaking] }
    elseif state.pending and #state.pending > 0 then
      -- Stacked input: the next round starts without closing the
      -- primary's cycle (one finish closes however many stacked
      -- submissions opened it).
      local nxt = start_round(state, table.remove(state.pending, 1))
      for _, e in ipairs(nxt.effects) do effects[#effects + 1] = e end
    else
      effects[#effects + 1] = { kind = "finish_cycle",
        thread_id = state.primary }
    end
    return { effects = effects, state = state }
  end

  -- turn_start re-fires while the primary is parked at its input
  -- boundary; the conversation is driven off input_accepted and
  -- agent_completed, so every other boundary parks.
  return { state = state }
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
