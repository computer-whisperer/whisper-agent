# Codex (`codex exec`) — Wire Capture

Capture: `codex.mitm`, CLI version `0.118.0`. 11 model turns, **all over a
single WebSocket** rather than a sequence of HTTP requests.

## Transport

- `wss://chatgpt.com/backend-api/codex/responses`
- HTTP/1.1 Upgrade, then 2792 WebSocket frames over the lifetime of the run.
  - 11 client → server frames (every `response.create` is one frame).
  - 2781 server → client frames (event deltas).
- Subprotocol gated by header `openai-beta: responses_websockets=2026-02-06`.
- `Authorization: Bearer <REDACTED JWT>` plus `Sec-WebSocket-Key`. JWT is an
  OAuth ID token tied to a ChatGPT account, not a `sk-…` API key.

This is structurally different from every other capture: there is no per-turn
HTTP request/response pair — turns are just *messages* in a long-lived stream.

## Endpoint Map

| Count | Endpoint | Purpose |
|---|---|---|
| 1 (WS, 2792 msgs) | `chatgpt.com/backend-api/codex/responses` | agent loop |
| 3 | `POST chatgpt.com/backend-api/wham/apps` | "WHAM" telemetry |
| 3 | `POST ab.chatgpt.com/otlp/v1/metrics` | OTLP metrics |
| 1 | `GET chatgpt.com/backend-api/codex/models` | model list |
| 1 | `GET chatgpt.com/backend-api//connectors/directory/list` | connectors |
| 1 | `GET chatgpt.com/backend-api/plugins/featured` | plugin index |
| 4 | `GET api.github.com/repos/openai/plugins/...` and `codeload.github.com/...` | plugin source download |
| 33 | `GET index.crates.io/...` | unrelated cargo |

Notably *very low* control-plane chatter compared to gemini.

## Request (Client→Server) Frame

Every turn is one `response.create` JSON frame. Top-level keys (15):

```
client_metadata, generate, include, input, instructions, model,
parallel_tool_calls, prompt_cache_key, reasoning, store, stream, text,
tool_choice, tools, type
```

Notable fields:

- `model: "gpt-5.3-codex"`.
- `reasoning: {"effort":"xhigh"}`. Reasoning summaries are not streamed back as
  plain text; they are returned as `reasoning` output items with an
  `encrypted_content` blob (server-signed, opaque to the client) plus
  optional summary text.
- `include: ["reasoning.encrypted_content"]` — opt-in to receive the encrypted
  reasoning block so it can be passed back into the next request, preserving
  the model's chain of thought across turns without exposing it client-side.
- `tool_choice: "auto"`, `parallel_tool_calls: true`.
- `text: {"verbosity":"low"}`.
- `instructions`: a single 12,343-char string (no list-of-blocks structure).
  Begins:
  > You are Codex, a coding agent based on GPT-5. You and the user share the
  > same workspace and collaborate to achieve the user's goals.
  > # Personality …
- `prompt_cache_key`: opaque ULID-shaped string (redacted) used to bind the
  request to a server-side cached prefix.
- `store: false` — the responses are not retained server-side after the
  conversation ends.

### Tool Declarations (77 total!)

Three tool *types* coexist in the array:

- `function` (×75) — JSON-schema-typed callable, OpenAI Responses API shape.
  Tool list includes the actual coding tools (`exec_command`, `write_stdin`,
  `update_plan`, `view_image`, `request_user_input`, `tool_suggest`,
  `spawn_agent`, `send_input`, `resume_agent`, `wait_agent`, `close_agent`,
  `list_mcp_resources`, `list_mcp_resource_templates`, `read_mcp_resource`)
  plus 60+ `mcp__codex_apps__github_*` tools auto-imported from the GitHub
  MCP plugin (search/fetch/create issues/PRs/comments/reactions/...).
- `custom` (×1) — `apply_patch`, defined by a **Lark grammar** rather than a
  JSON schema (see below).
- `web_search` (×1) — server-side builtin, name field omitted.

Representative `function` tool:

```jsonc
{
  "type": "function",
  "name": "exec_command",
  "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
  "strict": false,
  "parameters": {
    "type": "object",
    "properties": {
      "cmd":                  {"type":"string", "description":"Shell command to execute."},
      "justification":        {"type":"string", "description":"Only set if sandbox_permissions is \"require_escalated\". …"},
      "login":                {"type":"boolean","description":"Whether to run the shell with -l/-i semantics. Defaults to true."},
      "max_output_tokens":    {"type":"number"},
      "prefix_rule":          {"type":"array","items":{"type":"string"}},
      "sandbox_permissions":  {"type":"string"},
      "yield_time_ms":        {"type":"number"}
    }
  }
}
```

Representative `custom` tool — note **`format.type:"grammar"`** with a Lark
definition, not a JSON schema. The model emits raw text matching the grammar:

```jsonc
{
  "type": "custom",
  "name": "apply_patch",
  "description": "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
  "format": {
    "type": "grammar",
    "syntax": "lark",
    "definition": "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\n…"
  }
}
```

This is qualitatively different from JSON-schema tool calling — it's
constrained-decode against a formal grammar. The harness must understand
`response.custom_tool_call_input.delta` events (delta is a *grammar
fragment*, not JSON) and reassemble the raw text rather than parsing JSON.

## Response (Server→Client) Event Types

Server frame counts in this run:

| count | type |
|---|---|
| 2314 | `response.custom_tool_call_input.delta` |
| 290  | `response.output_text.delta` |
| 67   | `response.function_call_arguments.delta` |
| 26   | `response.output_item.added` |
| 26   | `response.output_item.done` |
| 11   | `response.created` |
| 11   | `response.in_progress` |
| 11   | `response.completed` |
| 6    | `response.function_call_arguments.done` |
| 5    | `response.content_part.added` |
| 5    | `response.output_text.done` |
| 5    | `response.content_part.done` |
| 3    | `response.custom_tool_call_input.done` |
| 1    | `codex.rate_limits` |

The grammar-tool deltas (`response.custom_tool_call_input.delta`, 2314 of
them) dominate; that's the model writing out the `apply_patch` grammar one
fragment at a time.

Representative envelope (`response.completed`, large fields elided):

```json
{"type":"response.completed",
 "response":{"id":"resp_…","object":"response","created_at":1776201951,
   "status":"completed","completed_at":1776201951,"error":null,
   "instructions":"<12343 chars omitted>",
   "model":"gpt-5.3-codex","output":"<N output items omitted>",
   "parallel_tool_calls":true,"prompt_cache_key":"<REDACTED>"}}
```

A reasoning output item arrives as:

```json
{"type":"response.output_item.added",
 "item":{"id":"rs_…","type":"reasoning",
         "encrypted_content":"<REDACTED>"}}
```

A tool-call arg delta:

```json
{"type":"response.custom_tool_call_input.delta","delta":"***",
 "item_id":"ctc_…","output_index":3,"sequence_number":57}
```

Of note: every delta carries an `obfuscation: "<random>"` field (purpose
unclear — likely sequence-tampering detection or padding; redacted in our
extract).

## Tool Results Returned to Model

Tool results are sent in the *next* `response.create` frame as part of
`input`, with `type: "function_call_output"` items keyed by `call_id`:

```jsonc
{ "input": [
    { "type": "function_call",        "call_id": "fc_…", "name": "exec_command",
      "arguments": "{\"cmd\":\"ls -la\"}"},
    { "type": "function_call_output", "call_id": "fc_…",
      "output": "<stdout/stderr blob>"}
] }
```

For the custom_tool grammar path the call/result types become `custom_tool_call`
and `custom_tool_call_output`. Reasoning items (`type: "reasoning"`) with the
encrypted blob are also passed through unchanged.

## Turn-by-turn Summary

| # | Status | input_tokens | cached | output | reasoning | Tool calls |
|---|---|---:|---:|---:|---:|---|
| 1  | completed | 15,598 | 0      | 0     | 0     | — (warmup, empty `input`) |
| 2  | completed | 17,683 | 15,488 | 466   | 385   | exec_command `ls -la` |
| 3  | completed | 18,275 | 18,048 | 43    | 10    | exec_command `cargo init …` |
| 4  | completed | 18,423 | 18,304 | 44    | 21    | exec_command `cat Cargo.toml` |
| 5  | completed | 18,557 | 18,432 | 36    | 13    | exec_command `cat src/main.rs` |
| 6  | completed | 18,665 | 18,560 | 2,964 | 2,868 | apply_patch (Cargo.toml) |
| 7  | completed | 21,667 | 21,504 | 2,223 | 9     | apply_patch (full game in src/main.rs) |
| 8  | completed | 23,928 | 23,808 | 290   | 228   | exec_command `cargo check` (yield_time_ms=120000) |
| 9  | completed | 24,570 | 24,192 | 700   | 561   | apply_patch (timing fix) |
| 10 | completed | 25,308 | 25,216 | 37    | 9     | exec_command `cargo check` |
| 11 | completed | 25,452 | 25,216 | 328   | 193   | — (final summary) |

**Turn 1 is empty-input warmup**: zero input items, zero output tokens, but
the request still ships full `instructions` + `tools` and the server bills
15,598 input tokens (no cache hit yet). Subsequent turns achieve >97% cache
hits. This appears to be a deliberate cache-priming pattern.

The user prompt + developer/permissions/skills/plugins context is sent in
**turn 2** (3 items: developer message with 4 sub-blocks, user environment
context, then the actual user prompt).

## Token / Usage Totals

| metric | value |
|---|---|
| input_tokens | 228,126 |
| cached input tokens | 208,768 |
| output_tokens (incl. 4,297 reasoning) | 7,131 |

(Cache utilization: ~91.5%.) Reasoning tokens are part of `output_tokens`.
Notable spike at turn 6: 2,868 reasoning tokens before emitting any visible
text — model "thought hard" before the first apply_patch.

## Telemetry / Non-Agent Traffic

| class | count |
|---|---|
| WHAM apps (chatgpt) | 3 |
| OTLP metrics | 3 |
| GitHub plugin install | 4 |
| chatgpt.com models/connectors/plugins | 3 |
| (cargo + crates.io) | 33 |

Total non-loop control traffic: 13. Markedly leaner than gemini's
~36 admin POSTs.

## Provider-Specific Surprises

1. **Single persistent WebSocket** — the unit of work is a *frame*, not a
   request. Any harness that models "one HTTP roundtrip per turn" cannot
   wrap codex without adapting to a long-lived bidirectional channel.
2. **Custom grammar tool calls.** `apply_patch` is constrained to a Lark
   grammar; the model emits raw text fragments through
   `response.custom_tool_call_input.delta` events. There is no JSON to parse.
3. **Encrypted reasoning passthrough.** The chain of thought is returned as
   an opaque encrypted blob (`reasoning.encrypted_content`), which the client
   must echo back into the next turn's `input` to preserve thinking — without
   ever being able to read it.
4. **Empty warmup turn**: turn 1 carries instructions + tools + zero input
   items, returns zero output, costs 15,598 input tokens. Pure prompt-cache
   priming. Without this, every cache hit afterward would be a cold start.
5. **77 declared tools**, dominated by 60+ MCP-imported `mcp__codex_apps__github_*`
   functions. Tool count alone is not a useful comparison metric across
   providers.
6. **Per-delta `obfuscation` field** of unknown purpose; redacted here as a
   precaution. May be tamper-detection nonces.
