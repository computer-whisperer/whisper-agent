# Agentic CLI Wire-Capture Comparison

Three CLIs were given the same prompt — "create a minimal snake game in Rust
with crossterm, cargo init, cargo check" — and their network traffic was
captured via mitmproxy. This doc compares the captured traffic across
providers for the purpose of designing whisper-agent's
provider-agnostic agent loop.

Per-provider deep dives:
[claude](captures/claude-run.md), [codex](captures/codex-run.md),
[gemini](captures/gemini-run.md).

---

## 1. Side-by-side at a glance

|                        | Claude Code (`claude -p`) | Codex (`codex exec`) | Gemini CLI (`gemini -p --yolo`) |
|---|---|---|---|
| CLI version            | 2.1.108 | 0.118.0 | 0.36.0 |
| Endpoint               | `POST api.anthropic.com/v1/messages?beta=true` | `wss://chatgpt.com/backend-api/codex/responses` | `POST cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse` |
| Transport              | **HTTP/2 POST + SSE response** | **Single persistent WebSocket** | **HTTP/2 POST + SSE response** |
| Auth                   | OAuth bearer in `Authorization` | OAuth JWT bearer in `Authorization` (+ `openai-beta` subprotocol header) | Google OAuth bearer in `Authorization` |
| Model                  | `claude-opus-4-6` (turn 1: `claude-haiku-4-5` for title-gen) | `gpt-5.3-codex` | `gemini-3-flash-preview` |
| Total wire requests/turns | 17 (1 title-gen + 16 agent loop) | 11 (1 warmup + 10 agent loop) over 1 WS, 2792 frames | 19 SSE requests, of which 6 were rate-limit retries → 13 real turns |
| Tool-call format       | `tool_use` content block with streamed `input_json_delta` JSON | `function_call` (JSON-arg streaming) **and** `custom_tool_call` (Lark-grammar streaming) | `functionCall` part with whole-object `args` (no streaming) |
| Tool result format     | `user`-role message with `tool_result` content blocks (keyed by `tool_use_id`) | next `response.create` carries `function_call_output` items keyed by `call_id` | next `contents` user turn carries `functionResponse` parts keyed **by tool name** |
| Tools declared         | 10 (all `type:"custom"` JSON-schema) | 77 (75 function + 1 custom-grammar + 1 web_search builtin) | 16 (in `tools[0].functionDeclarations`) |
| Total input tokens     | 447 (uncached) + 17,199 cache_creation + 351,068 cache_read | 228,126 (208,768 cached) | 136,638 (92,708 cached) |
| Total output tokens    | 6,357 | 7,131 (incl. 4,297 reasoning) | 4,905 + 2,839 thoughts |
| Prompt cache hit rate  | ~98% (implicit, no cache_control set) | ~92% (server-managed `prompt_cache_key`) | ~68% (`cachedContent`) |
| Telemetry / non-loop POSTs | 23 (event_logging, datadog, mcp_registry, plugin updates) | 13 (wham/apps, otlp metrics, github plugin install) | **38** (play.googleapis.com/log, recordCodeAssistMetrics, retrieveUserQuota, etc.) |
| MCP integration        | active — 3 GETs to `mcp-registry/v0/servers`, 2 POSTs to `mcp-proxy.anthropic.com` | dormant in this run, but 60+ `mcp__codex_apps__github_*` tools were *declared* (autoloaded github MCP) | not observed |
| System prompt format   | **list of 4 text blocks**, 26,310 ch total (block 0 is a billing telemetry header!) | single 12,343-ch string in `instructions` | single 24,411-ch string in `systemInstruction.parts[0].text` |
| Reasoning visibility   | `thinking` content blocks streamed as `thinking_delta` text | opaque `encrypted_content` blob, must be echoed back unchanged | `thought:true` text parts + opaque `thoughtSignature` blob |
| Final task outcome     | `cargo check` succeeded (worked around mitmproxy by configuring CA cert) | `cargo check` succeeded | gave up after 6 SSL workaround attempts; `Cargo.lock` and `target/` not produced |

---

## 2. Structural vs cosmetic differences

A lot of the differences are renames — every provider has *some* notion of
"tool", "result", "system prompt", "stop reason". But several differences are
genuinely structural and force the harness into provider-specific code paths.

### Genuinely structural

1. **Transport: WebSocket vs HTTP-SSE.** Codex uses a single bidirectional
   WebSocket; the unit of work is a *frame*, not a request. There is no per-turn
   request/response pair. A naïve `async fn turn(req) -> resp` interface
   cannot wrap codex without adapting to a long-lived stream. **This is the
   one divergence the harness genuinely cannot paper over** — see §6.

2. **Tool-call modality.** All three support JSON-schema function calls, but
   codex additionally exposes **`custom` tools defined by a Lark grammar**
   (`apply_patch`). The model emits raw text fragments matching the grammar
   through `response.custom_tool_call_input.delta` events. There is no JSON
   to parse and no input_schema to validate against — the parser is the
   grammar itself. Whisper-agent will need either to declare grammar tools
   are codex-only, or to model the abstraction as "tool input is bytes; some
   tools use JSON, some use grammar."

3. **Reasoning passthrough.** Each provider handles model "thinking"
   differently and the harness has to make a choice per provider:
   - Claude: thinking is plain `text_delta`-style content (stream as text).
   - Codex: thinking is an opaque encrypted blob that must be **echoed back**
     into the next request's `input` to preserve the chain of thought across
     turns. The client cannot read it.
   - Gemini: thinking is `thought:true` text parts (filterable) **plus** a
     `thoughtSignature` blob that must be **echoed back** on the carrying
     functionCall part. The client *can* read the prose but must preserve
     the signature.
   Whisper-agent must treat reasoning as "an opaque token the model wants
   to receive back next turn" rather than just "extra text."

4. **Tool-result identity.** Anthropic and OpenAI key tool results by a
   call-specific ID (`tool_use_id`, `call_id`). Gemini keys by **tool name**.
   This means a parallel-tool-call abstraction that issues two `read_file`
   calls concurrently has no way to disambiguate the gemini results.
   Whisper-agent should either disable parallel calls of the same name on
   gemini or carry a side-channel mapping.

5. **Cache strategy.** Claude relies on implicit prefix caching (no
   `cache_control` markers used by Claude Code itself). Codex sends an
   explicit `prompt_cache_key` (server-side managed) and even **fires a
   warmup turn with empty `input` to prime the cache** before the first real
   turn. Gemini emits a `cachedContentTokenCount` but in this run it cold-misses
   on the first agent turn and sometimes randomly thereafter (~68% hit).
   The harness can ignore this for correctness, but cost-modeling code must
   account for the difference.

### Cosmetic (just renames)

- "system prompt" / "instructions" / "systemInstruction" — same thing.
- "tools" array in all three; field-naming differs.
- "stop_reason" / "status" / "finishReason" — same thing.
- Token-usage field names differ (`input_tokens` vs `prompt_tokens` vs
  `promptTokenCount`) but mean the same.

---

## 3. Tool declaration formats — same tool, three syntaxes

The shell-execution tool in each system, abbreviated:

**Claude** (`Bash`, declared as `type:"custom"`, anthropic JSON-schema):

```jsonc
{
  "name": "Bash",
  "description": "Executes a given bash command and returns its output…",
  "input_schema": {
    "type": "object",
    "properties": {
      "command":           {"type":"string"},
      "description":       {"type":"string"},
      "timeout":           {"type":"number"},
      "run_in_background": {"type":"boolean"},
      "rerun":             {"type":"string"}
    },
    "required": ["command"]
  }
}
```

**Codex** (`exec_command`, OpenAI Responses-API function tool):

```jsonc
{
  "type": "function",
  "name": "exec_command",
  "description": "Runs a command in a PTY, returning output or a session ID…",
  "strict": false,
  "parameters": {
    "type": "object",
    "properties": {
      "cmd":                 {"type":"string"},
      "justification":       {"type":"string"},
      "login":               {"type":"boolean"},
      "max_output_tokens":   {"type":"number"},
      "prefix_rule":         {"type":"array","items":{"type":"string"}},
      "sandbox_permissions": {"type":"string"},
      "yield_time_ms":       {"type":"number"}
    }
  }
}
```

**Gemini** (`run_shell_command`, inside `tools[0].functionDeclarations`):

```jsonc
{
  "name": "run_shell_command",
  "description": "This tool executes a given shell command as `bash -c <command>`…",
  "parametersJsonSchema": {
    "type": "object",
    "properties": {
      "command":     {"type":"string"},
      "description": {"type":"string"},
      "dir_path":    {"type":"string"},
      "is_background": {"type":"boolean"}
    },
    "required": ["command"]
  }
}
```

The structural differences are minor:

- field name: `input_schema` (anthropic) vs `parameters` (openai) vs
  `parametersJsonSchema` (gemini)
- envelope: anthropic flat, openai requires `type:"function"`, gemini requires
  wrapping in a `functionDeclarations` group.

A `Tool { name, description, json_schema }` abstraction in whisper-agent can
serialize to any of these with a thin per-provider serializer. The harder
question is the `apply_patch`-style grammar tool, which doesn't fit a JSON-schema
abstraction at all.

The Codex `apply_patch` tool, for completeness:

```jsonc
{
  "type": "custom",
  "name": "apply_patch",
  "description": "Use the `apply_patch` tool to edit files. This is a FREEFORM tool…",
  "format": {
    "type": "grammar",
    "syntax": "lark",
    "definition": "start: begin_patch hunk+ end_patch\n…"
  }
}
```

---

## 4. Tool results — three different injection points

The wire shape of "here is the output of the last tool call" matters for the
harness: it determines what the conversation-state object looks like and how
events compose.

**Claude.** Append a `user`-role message whose `content` is a list of one or
more `tool_result` blocks. Tool results coexist with regular text blocks in
the same user turn:

```jsonc
{ "role": "user",
  "content": [
    { "type": "tool_result",
      "tool_use_id": "toolu_…",
      "is_error": false,
      "content": [{"type":"text","text":"Creating binary (application) package…"}]
    }
  ]
}
```

**Codex.** Send the next `response.create` frame with `input` containing the
prior `function_call` *and* a `function_call_output` item (paired by `call_id`).
Reasoning items must also be carried over verbatim:

```jsonc
{ "type": "response.create",
  "input": [
    {"type":"reasoning",            "id":"rs_…", "encrypted_content":"<opaque>"},
    {"type":"function_call",        "call_id":"fc_…","name":"exec_command",
                                    "arguments":"{\"cmd\":\"ls -la\"}"},
    {"type":"function_call_output", "call_id":"fc_…","output":"<stdout>"}
  ]
}
```

**Gemini.** Append a `user`-role `content` with `functionResponse` parts
(keyed by tool *name*, not call ID):

```jsonc
{ "role": "user",
  "parts": [
    { "functionResponse": {
        "name": "run_shell_command",
        "response": {"output": "Updating crates.io index\nerror: …"}
    }}
  ]
}
```

The harness can present these uniformly as
`Message::ToolResult { call_id, tool_name, content, is_error }` and serialize
each provider's shape in the adapter — but the underlying *call ID vs name*
asymmetry leaks through and constrains the parallel-tool-call surface.

---

## 5. Streaming shape: SSE vs WebSocket — concrete parser implications

The differences here are bigger than they look at first.

**SSE (claude / gemini)**: each line is `event: <name>` and `data: <json>`,
separated by blank lines. A turn is one HTTP response body. The parser is a
trivial state machine. Closing the response = end of turn.

```
event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"cmd\":"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}
```

Each SSE event delivers a complete JSON object; partial-JSON streaming exists
inside the `partial_json` field, and the harness must concatenate
`input_json_delta` strings to reassemble tool args.

Gemini's SSE is even simpler: each `data:` is one full
`GenerateContentResponse`. There is no token-by-token tool-arg streaming —
parts arrive whole.

**WebSocket (codex)**: each frame is one JSON object (binary or text — codex
uses text frames with utf-8 JSON). There is no SSE envelope; no `event:` line.
The parser must:
- track which `response_id` is currently open,
- distinguish 14+ event types (response.created/in_progress/completed,
  response.output_item.added/done, response.content_part.added/done,
  response.output_text.delta/done, response.function_call_arguments.delta/done,
  response.custom_tool_call_input.delta/done, codex.rate_limits, …),
- aggregate deltas keyed by `item_id` rather than by index,
- handle interleaved frames belonging to multiple in-flight items
  (`output_index` field).

Because the WebSocket is persistent, the harness must also handle:
- send-side framing of `response.create` and `function_call_output` frames,
- sequence numbering (`sequence_number` on every server frame),
- connection liveness / reconnect with state replay.

A naïve "read the response body until done, then parse" loop does not work for
codex.

---

## 6. Harness architecture implications for whisper-agent

The unit of work the abstraction should expose is **"asynchronous stream of
typed events"**, not "request/response pair." Concretely, something like:

```rust
trait Provider {
    type Conversation;

    /// Send a turn (user message and/or pending tool results) and receive
    /// a stream of events for the assistant's reply.
    fn turn(
        &self,
        conv: &mut Self::Conversation,
        new_input: Vec<InputItem>,
    ) -> impl Stream<Item = Event> + Send;
}

enum Event {
    TextDelta { text: String },
    Thinking { text: String, opaque_signature: Option<Bytes> },
    ToolCallStart { call_id: CallId, name: String, modality: ToolModality },
    ToolCallArgDelta { call_id: CallId, fragment: Bytes },
    ToolCallDone { call_id: CallId, full_args: Bytes },
    StopReason(StopReason),
    Usage(Usage),
    RateLimitInfo(RateLimitInfo),  // codex.rate_limits / gemini retry hints
}
```

Such a stream interface natively handles both:
- HTTP-SSE: each turn opens a new substream that ends with the response body,
- WebSocket: each turn opens a logical substream demuxed from the persistent
  socket by `response_id` + `sequence_number`.

The **transport split is the only structural divergence the harness cannot
hide.** It surfaces in two places:
- *connection lifecycle*: HTTP providers need no setup; WebSocket providers
  need `connect` / `reconnect` / `keepalive` handling.
- *backpressure model*: HTTP providers naturally end-of-stream when the body
  closes; WebSocket providers must observe `response.completed` to know a
  turn is done. Stream timeouts must be per-turn, not per-connection.

I recommend modeling this as a `Provider::Session` that holds the connection
(noop for HTTP, owned WebSocket for codex) and exposes `Session::turn(...) ->
Stream<Event>` rather than `Provider::turn(...) -> Stream<Event>` directly.

### Where a single trait covers all three

- **Tool declaration:** `Tool { name, description, json_schema }` plus a
  separate enum branch for codex's grammar tools.
- **Conversation state:** `Vec<Message>` where `Message` is a sum type
  spanning user-text, assistant-text, tool-call, tool-result, thinking,
  signature. The serializer per provider knows how to project this onto
  anthropic content blocks / openai input items / gemini contents-with-parts.
- **Tool result injection:** uniform at the API level
  (`Message::ToolResult { call_id, tool_name, content, is_error }`),
  serialized differently per provider.
- **Token usage:** uniform `Usage { input, cached, output, reasoning }`,
  populated from each provider's variant fields.
- **Stop reason:** uniform `StopReason { EndTurn, ToolUse, MaxTokens, Error }`.
- **Streaming events:** the `Event` enum above is the harness's contract.

### Where provider-specific code is unavoidable

- **Connection lifecycle**: HTTP-per-turn vs persistent WebSocket.
- **Cache management**: `cache_control` markers (anthropic), `prompt_cache_key`
  + warmup-empty-turn pattern (codex), implicit-with-misses (gemini).
- **Reasoning continuity**: each provider's "echo this opaque blob back next
  turn" mechanism is wire-shaped differently and must be plumbed by adapters.
- **Grammar tools**: codex-only.
- **Parallel tool calls**: must be disabled or call-name-de-duped on gemini.
- **Rate-limit retry**: codex emits `codex.rate_limits` info frames; gemini
  emits Google RPC `RATE_LIMIT_EXCEEDED` errors with `RetryInfo.retryDelay`;
  anthropic uses HTTP 429 with `retry-after`. The harness should normalize
  these into a single `RateLimitInfo` event but the adapter is not trivial.
- **Sub-agent / MCP tool routing**: Claude's `caller:{type:"direct"}` field
  and codex's `mcp__*` tool-name prefix imply provider-specific
  MCP-result fan-in plumbing. Out of scope here.

### Recommendation

**Yes, expose "stream of events" as the unit of work, not "request/response
pair."** This is the only way to integrate codex without an awkward
WebSocket-to-Future bridge per turn. It is not particularly more complex for
HTTP providers — SSE already maps cleanly to a stream. The harness design
should also separate `Session` (transport / connection / cache key) from
`Conversation` (message history / thread of tool calls) so that codex's
single-WebSocket-many-conversations and anthropic's
single-conversation-many-HTTP-roundtrips are both expressible.

---

## 7. Cross-cutting findings worth knowing

1. **Telemetry is louder than the agent loop in two of three providers.**
   Gemini in particular fires 38 admin/log POSTs per 13 model turns. A
   privacy-conscious wrapper has to MITM or strip these — the model-turn
   surface alone tells you nothing about how chatty the CLI is.

2. **First-turn anomalies in two of three.** Claude uses a different model
   (haiku) for session-title generation before the agent loop starts; codex
   fires an empty-input warmup turn purely to prime its server-side prompt
   cache. Naïve "agent loop = N turns" math is wrong by 1 in both cases.

3. **Six of gemini's 19 wire turns were rate-limit retries** within a single
   short run, with quota replenishing every 0.5–6.5 s. Gemini's prompt-tier
   per-minute quota is much tighter than the others.

4. **The captured Claude system prompt smuggles a billing/telemetry header
   into the first system text block.** Code that handles multi-tenant
   forwarding of system prompts could leak this; code that strips
   "non-content" system blocks could break Anthropic's tracking.

5. **Encrypted reasoning is now the norm, not the exception.** Codex
   (`encrypted_content`) and gemini (`thoughtSignature`) both ship opaque
   server-signed reasoning blobs that the client must round-trip without
   inspection. Whisper-agent needs a first-class concept of
   "opaque continuation token" attached to assistant turns.

6. **Tool counts vary by 7.7×** (claude: 10, gemini: 16, codex: 77). Most of
   codex's tool count is auto-imported MCP plugin functions
   (`mcp__codex_apps__github_*`). Tool count alone is not a meaningful
   provider comparison.
