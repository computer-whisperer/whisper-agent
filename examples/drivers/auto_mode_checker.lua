-- Auto-mode permission checker (migration step 7 exercise case).
--
-- Drives a primary chat thread like the builtin single-agent driver,
-- but intercepts every tool request: a single-turn checker thread is
-- derived (curated seed, custom prompt, tools disabled) and asked to
-- approve or deny each call before anything executes. This is exactly
-- the shape the rejected projection model could not express — the
-- checker grows per-check on its own axis, in its own context.
--
-- Contract notes (see src/runtime/driver/lua.rs): on_event must be
-- idempotent at boundaries; all mutable state lives in the returned
-- `state` table.

-- A call is approved only by a whole verdict line reading exactly
-- "ALLOW <id>" (or "ALLOW ALL"). Substring matching would fail open:
-- "I cannot ALLOW ALL of these" must not approve anything, and an id
-- that prefixes another id must not approve both.
local function verdict_allows(verdict, tool_use_id)
  for line in string.gmatch(verdict, "[^\r\n]+") do
    local trimmed = string.match(line, "^%s*(.-)%s*$")
    if trimmed == "ALLOW ALL" or trimmed == "ALLOW " .. tool_use_id then
      return true
    end
  end
  return false
end

function on_event(state, event)
  local k = event.kind

  if k == "input_accepted" then
    -- Fresh cycle. Any in-flight interception is void — the scheduler
    -- heals the primary out of its parked boundary on superseding
    -- input. `primary` survives for present().
    return { state = { primary = event.thread_id } }
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
    -- A checker verdict came back.
    if state.checking and event.thread_id == state.checking.checker then
      local verdict = event.text
      local decisions = {}
      for i, call in ipairs(state.checking.calls) do
        local allow = verdict_allows(verdict, call.tool_use_id)
        local decision = { tool_use_id = call.tool_use_id, allow = allow }
        if not allow then
          decision.message = "denied by permission checker"
        end
        decisions[i] = decision
      end
      local primary = state.checking.primary
      state.checking = nil
      return { effects = {
        { kind = "finish_cycle", thread_id = event.thread_id },
        { kind = "resolve_tools", thread_id = primary, decisions = decisions },
      }, state = state }
    end

    -- Re-delivered primary boundary while a check is in flight: hold.
    if state.checking and event.thread_id == state.checking.primary then
      return { state = state }
    end

    if #event.tool_calls == 0 then
      return { effects = { { kind = "finish_cycle", thread_id = event.thread_id } },
               state = state }
    end

    -- The primary requested tools: intercept and derive a checker.
    local question = "The agent requests these tool calls:\n"
    for _, call in ipairs(event.tool_calls) do
      question = question .. "- id " .. call.tool_use_id .. ": " .. call.name .. "\n"
    end
    question = question ..
      "Reply ALLOW <id> per approved call, ALLOW ALL, or DENY."
    state.checking = { primary = event.thread_id, calls = event.tool_calls }
    return { effects = {
      { kind = "derive_thread", relationship = "check", disable_tools = true,
        system_prompt = "You are a strict permission checker. Reply only with ALLOW/DENY lines.",
        seed = { { author = "user", text = question } },
        source_thread_id = event.thread_id },
    }, state = state }
  end

  if k == "thread_derived" then
    if event.relationship == "check" and state.checking then
      state.checking.checker = event.thread_id
      return { effects = { { kind = "run_agent", thread_id = event.thread_id } },
               state = state }
    end
    return { state = state }
  end

  if k == "tools_completed" then
    return { effects = { { kind = "continue_cycle", thread_id = event.thread_id } },
             state = state }
  end

  return { state = state }
end

-- Presentation: a pure function of the state on_event returned. The
-- primary transcript is always the head; while a check is in flight,
-- surface a status line and the checker thread. A finished checker
-- drops from the display but stays reachable via the weave's
-- drill-down thread list — curate, never conceal.
function present(state)
  local blocks = {}
  if state.primary then
    blocks[#blocks + 1] =
      { kind = "primary_transcript", thread_id = state.primary }
  end
  if state.checking then
    blocks[#blocks + 1] = { kind = "status",
      text = "permission check in flight ("
        .. #state.checking.calls .. " tool calls)" }
    if state.checking.checker then
      blocks[#blocks + 1] =
        { kind = "thread_list", thread_ids = { state.checking.checker } }
    end
  end
  return blocks
end
