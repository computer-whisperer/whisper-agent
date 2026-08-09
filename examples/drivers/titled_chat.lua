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
-- Knob: `title.model` ({backend, model}) — unset falls through to the
-- pod's default model.

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
    },
  }
end

function on_event(state, event, config)
  local k = event.kind

  if k == "input_accepted" then
    state.primary = state.primary or event.thread_id
    if event.thread_id == state.primary and not state.opening then
      state.opening = clip(event.text, 500)
    end
    return { state = state }
  end

  if k == "turn_start" then
    if state.primary == nil then
      -- Driver-run turn on a seeded thread that never saw
      -- input_accepted: the conversation head is whoever turns first.
      state.primary = event.thread_id
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
      return { effects = { { kind = "dispatch_tools", thread_id = event.thread_id } },
               state = state }
    end

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
    return { effects = effects, state = state }
  end

  if k == "thread_derived" then
    if event.relationship == "title" then
      state.title_thread = event.thread_id
      return { effects = { { kind = "run_agent", thread_id = event.thread_id } },
               state = state }
    end
    return { state = state }
  end

  if k == "tools_completed" then
    return { effects = { { kind = "continue_cycle", thread_id = event.thread_id } },
             state = state }
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
