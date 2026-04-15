# Tool-Use Protocol Formats Across Provider APIs

How the three major hosted LLM providers (Anthropic, OpenAI, Google Gemini) expose tool/function calling on the wire. Written with an eye toward what whisper-agent's provider-agnostic tool abstraction has to translate between.

## Convergence

All three have stabilized on the same *semantic* shape:

1. **Schema declaration** — tools are declared on every request as a list, each tool providing a name, description, and a **JSON Schema** for its input parameters.
2. **Structured tool-call emission** — the assistant's response includes one or more structured tool-call blocks (not text-to-regex-parse), interleaved with text and thinking content where supported.
3. **Result injection** — the harness executes the tool, then sends the result back as a new message with provider-specific block/role markers referencing the call's ID.
4. **Parallel calls** — a single assistant turn may emit multiple tool calls, which the harness must execute and collect before the next turn.

Cosmetic wire shapes differ; semantics are identical. A shim layer of ~20 lines of per-provider mapping is enough, *except* for streaming-partial-inputs, which is genuinely three different formats.

## Anthropic (Messages API)

**Declaration** on every request:

```json
{
  "tools": [{
    "name": "Bash",
    "description": "Executes a bash command...",
    "input_schema": {
      "type": "object",
      "properties": {
        "command": {"type": "string"},
        "timeout": {"type": "number"}
      },
      "required": ["command"]
    }
  }]
}
```

**Assistant output** — `tool_use` content block interleaved with `text`/`thinking`:

```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "Checking the directory."},
    {"type": "tool_use",
     "id": "toolu_01ABC",
     "name": "Bash",
     "input": {"command": "ls -la"}}
  ],
  "stop_reason": "tool_use"
}
```

**Result back** — `tool_result` block in a user-role message:

```json
{"role": "user", "content": [
  {"type": "tool_result",
   "tool_use_id": "toolu_01ABC",
   "content": "total 8\ndrwxr-xr-x ..."}
]}
```

`tool_choice` options: `auto | any | tool | none`.

## OpenAI (Chat Completions / Responses)

**Declaration** — wrapped in a `function` envelope:

```json
{"tools": [{
  "type": "function",
  "function": {
    "name": "Bash",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": {...},
      "required": [...]
    }
  }
}]}
```

**Assistant output** — `tool_calls` on the assistant message. Note `arguments` is a **JSON string**, not an object:

```json
{"role": "assistant", "content": null, "tool_calls": [{
  "id": "call_abc",
  "type": "function",
  "function": {
    "name": "Bash",
    "arguments": "{\"command\":\"ls -la\"}"
  }
}]}
```

**Result back** — separate `tool` role message per call:

```json
{"role": "tool",
 "tool_call_id": "call_abc",
 "content": "total 8\n..."}
```

`tool_choice` options: `auto | required | none | {specific-function}`.

## Google Gemini

**Declaration** — `functionDeclarations` inside `tools`:

```json
{"tools": [{
  "functionDeclarations": [{
    "name": "Bash",
    "description": "...",
    "parameters": {"type": "object", ...}
  }]
}]}
```

**Assistant output** — `functionCall` as a part. `args` is an **object** (unlike OpenAI's string):

```json
{"role": "model", "parts": [
  {"functionCall": {
    "name": "Bash",
    "args": {"command": "ls -la"}
  }}
]}
```

**Result back** — `functionResponse` as a user-role part:

```json
{"role": "user", "parts": [
  {"functionResponse": {
    "name": "Bash",
    "response": {"content": "total 8\n..."}
  }}
]}
```

`tool_choice` equivalent: `FunctionCallingConfig.mode: AUTO | ANY | NONE | VALIDATED`.

## Gotchas when unifying

- **Argument-encoding asymmetry.** OpenAI's `arguments` is a JSON *string*; Anthropic's `input` and Gemini's `args` are already-parsed *objects*. A unified representation must parse OpenAI's string before passing to the tool runner.
- **Streaming partial inputs.** Three incompatible delta formats:
  - Anthropic: `input_json_delta` chunks inside `content_block_delta` events.
  - OpenAI: partial chunks on `tool_calls[i].function.arguments`.
  - Gemini: partial `functionCall` parts.
  - All three require buffer-and-reparse to reconstruct a complete invocation before the tool can fire.
- **Tool result shape.** Anthropic/Gemini keep results inside message content blocks; OpenAI uses a dedicated `tool` role per call. Semantically equivalent, structurally different — affects message-history replay logic.
- **Tool-choice semantics.** Superficially similar enum names (`any`/`required`/`ANY`) mean the same thing — force a tool call — but the opt-out and specific-function syntaxes diverge.

## Server-side vs. client-side tools

Anthropic is shipping built-in tools that execute *in their runtime* rather than in the client:

- `bash_20250124`, `text_editor_20250124` — filesystem/shell inside Anthropic's sandbox.
- `computer_20250124` — screen capture + mouse/keyboard in their VM.
- `web_search_20250305` — search and fetch from their infra.

These are identified by versioned type strings and are declared alongside normal custom tools. They never reach the client harness; the model calls them, Anthropic's runtime executes them, results come back in the next turn.

**Whisper-agent will not use server-side tools.** The whole point is that actions must happen on the user's machine — local filesystem, local k8s cluster, local obsidian vault, local POSIX hosts. Claude Code works the same way: it declares `Bash`/`Read`/`Edit`/etc. as plain custom tools and executes them client-side. Whisper-agent inherits that posture.

## Implication for whisper-agent

A provider-agnostic tool abstraction is tractable because the semantics are identical. The shim translates:

- JSON Schema input spec → declaration format.
- Tool call → `(name, id, input_object)` triple.
- Tool result → `(id, content)` pair, re-wrapped per provider.
- Streaming partials → buffer-and-reparse to a completed invocation.

The *one* legitimately hard part is streaming: three different delta formats, each requiring provider-specific state machines. Everything else is cosmetic.
