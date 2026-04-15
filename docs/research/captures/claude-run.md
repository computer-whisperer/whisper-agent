# Claude Code (`claude -p`) — Wire Capture

Capture: `claude.mitm`, CLI version `2.1.108`, prompt =
"create minimal snake game in Rust with crossterm, cargo init, cargo check".
17 total POSTs to the Anthropic Messages endpoint. The first one is **not**
agent-loop traffic — it is a session-title generation against a smaller model.

## Transport

- `POST https://api.anthropic.com/v1/messages?beta=true`
- Request body: `application/json`
- Response body: SSE (`text/event-stream`) — one stream per turn
- One TCP connection per turn (HTTP/2 multiplexed); no persistent socket.
- Each turn re-sends the entire conversation: messages array grows by 2 per
  turn (assistant turn N → tool_result for turn N+1).

## Auth

- Single `Authorization` header, OAuth-style bearer (value redacted in capture).
  No `x-api-key`; Claude Code's CLI authenticates via OAuth account, not a
  raw API key.

## Endpoint Map

| Count | Endpoint | Purpose |
|---|---|---|
| 17 | `POST api.anthropic.com/v1/messages` | agent loop |
| 8  | `POST api.anthropic.com/api/event_logging/v2/batch` | first-party telemetry |
| 6  | `POST http-intake.logs.us5.datadoghq.com/api/v2/logs` | Datadog logs |
| 3  | `GET api.anthropic.com/mcp-registry/v0/servers` | MCP discovery |
| 2  | `POST mcp-proxy.anthropic.com/v1/mcp/<srv>` | MCP proxy calls |
| 1  | `GET api.anthropic.com/api/oauth/account/settings` | OAuth bootstrap |
| 1  | `GET api.anthropic.com/api/claude_cli/bootstrap` | CLI bootstrap |
| 1  | `GET api.anthropic.com/api/claude_code_penguin_mode` | feature flag |
| 1  | `GET api.anthropic.com/api/claude_code_grove` | feature flag |
| 1  | `GET api.anthropic.com/v1/mcp_servers` | MCP listing |
| 1  | `GET api.anthropic.com/api/eval/sdk-…` | eval (A/B?) |
| 2  | `GET downloads.claude.ai/claude-code-releases/...` | plugin update |
| 50 | `GET index.crates.io/...` and `static.crates.io/...` | unrelated cargo |

## Request Body

The agent-loop turns use these top-level keys (turn 1 — the title-gen call —
omits `context_management` and `thinking`, includes only 9 keys):

```
context_management, max_tokens, messages, metadata, model,
output_config, stream, system, temperature, thinking, tools
```

Notable:

- `model: "claude-opus-4-6"` for the agent loop. Turn 1 uses
  `"claude-haiku-4-5-20251001"` for title generation.
- `max_tokens: 32000`, `temperature: 1`, `stream: true`.
- `output_config: {"effort":"high"}` — opaque hint.
- `metadata: {"user_id": <REDACTED>}`.
- `system` is **a list of 4 text blocks**, total 26310 chars on the agent
  loop request:
  - block 0 (85 ch): `x-anthropic-billing-header: cc_version=2.1.108.3c4; cc_entrypoint=sdk-cli; cch=…` — billing/telemetry tags, smuggled into the system prompt as a text block. Surprising place for it.
  - block 1 (62 ch): `"You are a Claude agent, built on Anthropic's Claude Agent SDK."`
  - block 2 (9.9 KB): the agent system prompt proper, opens with security framing and tool-usage rules.
  - block 3 (16.2 KB): output-formatting / harness conventions starting with `# Text output (does not apply to tool calls)`.
- None of the four blocks set `cache_control` explicitly — caching is implicit.

### Tool Declarations (10)

All 10 declared tools have `type: "custom"` (no anthropic-server tools like
`bash_20250124` are used by Claude Code itself):

| Name | Purpose | Params (top-level) |
|---|---|---|
| `Agent` | spawn sub-agent | description, isolation, model, prompt, run_in_background, subagent_type |
| `Bash` | shell exec | command, dangerouslyDisableSandbox, description, rerun, run_in_background, timeout |
| `Edit` | string-replace edit | file_path, new_string, old_string, replace_all |
| `Glob` | filename pattern match | path, pattern |
| `Grep` | ripgrep wrapper | -A, -B, -C, -i, -n, context, glob, head_limit, multiline, offset, output_mode, path, pattern, type |
| `Read` | file read | file_path, limit, offset, pages |
| `ScheduleWakeup` | self-pace /loop | delaySeconds, prompt, reason |
| `Skill` | invoke local skill | args, skill |
| `ToolSearch` | fetch deferred-tool schemas | max_results, query |
| `Write` | file write | content, file_path |

Representative full declaration (`Bash`, abbreviated):

```jsonc
{
  "name": "Bash",
  "description": "Executes a given bash command and returns its output.\n\n…",
  "input_schema": {
    "type": "object",
    "properties": {
      "command":         {"type":"string"},
      "description":     {"type":"string"},
      "timeout":         {"type":"number"},
      "run_in_background": {"type":"boolean"},
      "rerun":           {"type":"string"},
      "dangerouslyDisableSandbox": {"type":"boolean"}
    },
    "required": ["command"]
  }
}
```

## Response / Event Shape

SSE event types, in order they appear during a turn:

```
message_start          ← envelope + initial usage (input_tokens, cache_*)
content_block_start    ← per content block (text, tool_use, thinking)
ping                   ← keepalive
content_block_delta    ← {delta:{type:text_delta|input_json_delta|thinking_delta,…}}
content_block_stop
message_delta          ← {delta:{stop_reason,…}, usage:{output_tokens,…}}
message_stop
```

Tool-use start example (verbatim, sanitized):

```json
{"type":"content_block_start","index":1,
 "content_block":{"type":"tool_use","id":"toolu_…","name":"Bash","input":{},
                  "caller":{"type":"direct"}}}
```

Tool arguments stream as `input_json_delta` partial JSON strings inside
`content_block_delta` events; the harness must concatenate and parse on
`content_block_stop`. The `caller.type=direct` field hints at sub-agent /
delegation routing inside the SDK.

## Tool Results Returned to Model

Every subsequent turn's `messages` array includes tool results as a `user`-role
message containing a list of `tool_result` content blocks:

```jsonc
{ "role": "user",
  "content": [
    { "type": "tool_result",
      "tool_use_id": "toolu_…",
      "is_error": false,
      "content": [ { "type": "text", "text": "Creating binary (application) package …" } ]
    }
  ]
}
```

Multiple tool results may be batched in one user-role message but in this run
each turn returned exactly one (no parallel tool calls were issued).

## Turn-by-turn Summary

| # | Model | Input | Tool calls | Stop | usage in/out/cache_create/cache_read |
|---|---|---|---|---|---|
| 1 | haiku-4-5 | user prompt (title-gen system) | — | end_turn | 429 / 18 / 0 / 0 |
| 2 | opus-4-6 | user prompt + reminders | Bash `cargo init` | tool_use | 3 / 116 / 7006 / 11859 |
| 3 |  | tool_result(init OK) | Bash `cargo add crossterm` | tool_use | 1 / 90 / 165 / 18865 |
| 4 |  | tool_result(SSL error) | Read Cargo.toml | tool_use | 1 / 71 / 189 / 19030 |
| 5 |  | tool_result(file) | Edit Cargo.toml | tool_use | 1 / 142 / 129 / 19219 |
| 6 |  | tool_result(edited) | Write src/main.rs (~2.3k out tokens — full game) | tool_use | 1 / 2259 / 198 / 19348 |
| 7 |  | tool_result(error: "must read first") | Read src/main.rs | tool_use | 1 / 71 / 2305 / 19546 |
| 8 |  | tool_result(stub) | Write src/main.rs (full game, 2.2k out tokens) | tool_use | 1 / 2192 / 2412 / 19546 |
| 9 |  | tool_result(written) | Bash `cargo check` | tool_use | 1 / 112 / 2229 / 21958 |
| 10 |  | tool_result(SSL error) | Bash diagnose certs (thinking_used=true) | tool_use | 1 / 260 / 357 / 24187 |
| 11 |  | tool_result(cert paths) | Bash retry with `CARGO_HTTP_CAINFO` | tool_use | 1 / 121 / 311 / 24544 |
| 12 |  | tool_result(still SSL) | Bash `curl` to test | tool_use | 1 / 101 / 278 / 24855 |
| 13 |  | tool_result(curl OK) | Bash write `.cargo/config.toml` | tool_use | 1 / 185 / 131 / 25133 |
| 14 |  | tool_result(SSL error) | Bash rewrite cargo config | tool_use | 1 / 167 / 342 / 25264 |
| 15 |  | tool_result(SSL error) | Bash inspect env | tool_use | 1 / 120 / 324 / 25606 |
| 16 |  | tool_result(found mitmproxy CA path) | Bash write cargo config with mitm CA | tool_use | 1 / 182 / 248 / 25930 |
| 17 |  | tool_result(cargo check OK) | — (final summary text) | end_turn | 1 / 150 / 575 / 26178 |

Claude is the only agent that actually completed `cargo check` cleanly — it
diagnosed mitmproxy's MITM cert and pointed cargo at it.

## Token / Cache Totals (all 17 turns)

| metric | value |
|---|---|
| input_tokens | 447 |
| output_tokens | 6357 |
| cache_creation_input_tokens | 17,199 |
| cache_read_input_tokens | 351,068 |

Implicit prompt caching is doing essentially all the heavy lifting:
**~98% of "input" content is being served from cache** across the 16-turn
agent loop. This pattern only works because the system prompt (26 KB) and
tool declarations are byte-stable across turns and the prefix grows
monotonically.

## Telemetry / Non-Agent Traffic

| class | count |
|---|---|
| Anthropic event_logging | 8 |
| Datadog logs | 6 |
| MCP registry / proxy | 6 |
| OAuth | 1 |
| Plugin downloads | 2 |
| Bootstrap / feature flags | 3 |
| Other | 1 |

Roughly **23 telemetry/admin requests for 17 model requests** — heavier
side-traffic ratio than codex; lighter than gemini.

## Provider-Specific Surprises

1. **First turn is on a different model** (`claude-haiku-4-5`) for session
   title generation. A naïve "claude run = N requests" count over-states the
   agent loop turn count by 1.
2. **Billing header is sent as the first system text block**, not as an HTTP
   header. Code that stripped non-prose system blocks for caching could
   accidentally drop telemetry signal — or, worse, code that passed the system
   array straight from a multi-tenant proxy could leak this data.
3. `output_config: {"effort":"high"}` and `thinking` are top-level request
   fields independent of the documented `extended_thinking` API parameter.
4. Tool calls have a `caller: {"type":"direct"}` discriminator on the
   `tool_use` content block, presumably distinguishing direct calls from
   sub-agent invocations.
5. Tool results return as a `user`-role message — the model never receives an
   explicit "tool" role.
