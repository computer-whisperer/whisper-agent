# Anthropic Messages API — Research

Scope: Anthropic's HTTP API (`POST /v1/messages`) as documented at `platform.claude.com/docs/en/api/messages` (formerly `docs.claude.com/en/api/messages`; host redirect observed April 2026). Written for a provider-agnostic backend; focus is wire surface, not SDK ergonomics.

Verified against docs as of 2026-04-14. Claims flagged `(unverified)` were not explicitly confirmed in the pages read.

---

## 1. Request surface

Endpoint: `POST https://api.anthropic.com/v1/messages`
Required headers: `x-api-key`, `anthropic-version: 2023-06-01`, `content-type: application/json`. Beta features gate on `anthropic-beta: <header>` (see §9).
Source: <https://platform.claude.com/docs/en/api/messages>

Top-level request body:

```json
{
  "model": "claude-opus-4-6",
  "messages": [{"role": "user|assistant", "content": "string | ContentBlock[]"}],
  "system": "string | TextBlock[]",
  "max_tokens": 1024,
  "temperature": 0.0,
  "top_p": 0.9,
  "top_k": 40,
  "stop_sequences": ["..."],
  "metadata": {"user_id": "opaque-string"},
  "stream": false,
  "tools": [],
  "tool_choice": {"type": "auto"},
  "thinking": {"type": "enabled", "budget_tokens": 10000},
  "service_tier": "auto"
}
```

Content-block input types (all carry optional `cache_control`):

- `text` — `{type, text}`
- `image` — `{type, source: {type: "base64"|"url", media_type, data|url}}` (jpeg/png/gif/webp)
- `document` — PDF (base64/url) or `text/plain`; supports `title`, `context`, `citations: {enabled: true}`
- `tool_use` — `{type, id, name, input}` (assistant-authored)
- `tool_result` — `{type, tool_use_id, content, is_error?}` (user-authored)
- `thinking` — `{type, thinking, signature}` (pass-back for multi-turn; see §3)
- `redacted_thinking` — `{type, data}` (opaque blob; rare on current models)
- `server_tool_use` — Anthropic-executed (web_search, web_fetch, code_execution, etc.)

Notes: `messages` requires alternating `user`/`assistant` turns (consecutive same-role turns are merged). Final `assistant` turn prefills the response. `system` may be string or array of text blocks (array form enables per-block `cache_control`).

---

## 2. Tool use

Source: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference>

### Custom tool declaration

```json
{
  "type": "custom",
  "name": "get_weather",
  "description": "...",
  "input_schema": {"type": "object", "properties": {...}, "required": [...]},
  "cache_control": {"type": "ephemeral"},
  "strict": true,
  "defer_loading": false,
  "eager_input_streaming": false,
  "input_examples": [{...}]
}
```

`strict: true` guarantees schema conformance on both tool name and input. `defer_loading: true` excludes from initial system prompt (paired with tool_search).

### Tool-call loop

Model emits assistant message with `stop_reason: "tool_use"` containing one or more `tool_use` blocks. Caller executes, returns a user turn with `tool_result` blocks keyed by `tool_use_id` (`is_error: true` signals failure).

### `tool_choice`

```json
{"type": "auto"}                                       // default; model decides
{"type": "any"}                                        // must call some tool
{"type": "tool", "name": "get_weather"}                // force specific tool
{"type": "none"}                                       // disable tools for turn
```

Add `"disable_parallel_tool_use": true` to `auto`/`any` to suppress parallel calls (default allows them).

### Anthropic-provided tools (versioned `type` strings)

Server-side (executed by Anthropic; results materialize as `server_tool_use` + result blocks):
- `web_search_20260209` / `web_search_20250305` (GA)
- `web_fetch_20260209` / `web_fetch_20250910` (GA)
- `code_execution_20260120` / `code_execution_20250825` (GA; bash + files); `code_execution_20250522` (legacy, Python-only)
- `advisor_20260301` (beta `advisor-tool-2026-03-01`)
- `tool_search_tool_regex_20251119` / `tool_search_tool_bm25_20251119` (GA)
- `mcp_toolset` (beta `mcp-client-2025-11-20`)

Client-side (Anthropic-schema, caller executes):
- `bash_20250124` (GA)
- `text_editor_20250728` / `text_editor_20250124` (GA; model-keyed)
- `computer_20251124` / `computer_20250124` (beta `computer-use-2025-11-24`)
- `memory_20250818` (GA)

---

## 3. Extended thinking

Source: <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>

```json
{"thinking": {"type": "enabled", "budget_tokens": 10000, "display": "summarized|omitted"}}
{"thinking": {"type": "adaptive"}}   // recommended on Opus/Sonnet 4.6
{"thinking": {"type": "disabled"}}
```

- `budget_tokens`: minimum 1024; must be `< max_tokens` except when interleaved thinking is enabled (then it's a cumulative budget across all thinking blocks). Billed as output tokens.
- `display: "summarized"` (default on Claude 4): response contains summarized `thinking` blocks but billing reflects full thinking tokens. `"omitted"`: empty `thinking` field with opaque `signature` (fastest TTFT, still billed).
- `signature`: cryptographic token on every `thinking` block; caller MUST echo thinking blocks back unmodified alongside any subsequent `tool_result` in the same tool-use cycle, or the API errors. Outside tool cycles the API auto-strips prior-turn thinking.
- `redacted_thinking` `{type, data}`: opaque encrypted blob when thinking is withheld; must also be preserved verbatim on round-trip.
- Interleaved thinking (between/after tool calls): beta header `interleaved-thinking-2025-05-14` on Opus/Sonnet 4.5 and 4.1; automatic (no header) on 4.6-era models; deprecated on 4.6 in favor of `adaptive`.
- Tool-choice constraint with thinking: only `auto` or `none` supported; `any`/`tool` rejected.

---

## 4. Prompt caching

Source: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>

```json
{"cache_control": {"type": "ephemeral", "ttl": "5m"}}   // or "1h"
```

- Only `type: "ephemeral"` exists. TTL `"5m"` (default, 1.25x write multiplier) or `"1h"` (2x write multiplier). Reads priced at 0.1x base input.
- Up to 4 cache breakpoints per request, placeable on: last tool in `tools`, `system` text blocks, any `messages` content block (text/image/document/tool_use/tool_result). `thinking` blocks cannot carry breakpoints directly but are cached when inside a cached assistant turn.
- Minimum cacheable prefix: 1024 tokens (Sonnet 4.5, Opus 4.1, Sonnet 4, Sonnet 3.7), 2048 (Sonnet 4.6, Haiku 3.5), 4096 (Opus 4.6, Opus 4.5, Haiku 4.5). Below the floor, caching silently no-ops.
- Mixed TTLs: 1h breakpoints must precede 5m breakpoints in the rendered prefix.
- Usage accounting on response:

```json
"usage": {
  "input_tokens": 50,
  "cache_creation_input_tokens": 248,
  "cache_read_input_tokens": 100000,
  "output_tokens": 503,
  "cache_creation": {"ephemeral_5m_input_tokens": 148, "ephemeral_1h_input_tokens": 100}
}
```

Total billed input = `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. Lookback window for matching prior cache entries is 20 blocks. Cache is org-scoped (workspace-scoped starting 2026-02-05, per docs).

---

## 5. Response surface

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-4-6",
  "content": [ /* text | thinking | tool_use | server_tool_use | ... */ ],
  "stop_reason": "end_turn|max_tokens|stop_sequence|tool_use",
  "stop_sequence": null,
  "usage": { /* see §4 */ },
  "container": {"id": "...", "expires_at": "..."}   // present iff code_execution used
}
```

`stop_reason` values observed in docs: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`. Additional values such as `pause_turn`, `refusal` appear in some contexts (unverified from the pages read; worth a direct check if relied upon).

---

## 6. Streaming (SSE)

Source: <https://platform.claude.com/docs/en/build-with-claude/streaming>

Set `"stream": true`. Event sequence per turn:

1. `message_start` — `{type, message: {...empty content...}}`
2. For each content block (indexed): `content_block_start` → N×`content_block_delta` → `content_block_stop`
3. `message_delta` — `{type, delta: {stop_reason, stop_sequence}, usage: {output_tokens}}` (cumulative usage)
4. `message_stop`
5. Intermixed: `ping`, `error` (e.g. `overloaded_error`)

Delta variants inside `content_block_delta`:

```json
{"type": "text_delta", "text": "..."}
{"type": "input_json_delta", "partial_json": "{\"location\": \"San Fra"}   // tool_use input
{"type": "thinking_delta", "thinking": "..."}
{"type": "signature_delta", "signature": "..."}                             // just before thinking block stop
{"type": "citations_delta", "citation": {...}}                              // (unverified exact shape)
```

Tool input arrives as partial-JSON string fragments; accumulate and parse at `content_block_stop`. With `display: "omitted"` on thinking, no `thinking_delta` events fire — only a single `signature_delta`. Fine-grained (per-token) tool input streaming requires `eager_input_streaming: true` on the tool.

---

## 7. Logit / logprob access

**Not available.** The Messages API does not accept or return `logprobs`, `top_logprobs`, `logit_bias`, or any per-token probability distribution. This is a stated product stance, not an oversight — Anthropic has not exposed logprobs on any Claude model (web search April 2026 confirms: "Anthropic's API does not expose token logprobs"). Third-party community shims (e.g. `anerli/anthropic-logprobs`) exist but they reconstruct proxies from multiple sampled calls, not real logits.

Implication for whisper-agent's abstraction: any `logprobs`/distribution field in the provider-agnostic interface must be nullable and explicitly `null` for Anthropic responses. Do not attempt to fake it from Anthropic output.

---

## 8. Token IDs on wire

**Not available.** Neither requests nor responses transport integer token IDs. All content is strings/bytes (text) or structured content blocks; tokenization is opaque server-side.

The `POST /v1/messages/count_tokens` endpoint exists (same request schema as `/v1/messages`, minus `max_tokens`), but returns only a scalar count:

```json
{"input_tokens": 12345}
```

No token IDs, no per-segment breakdown, no alignment back to input spans. Source: <https://platform.claude.com/docs/en/api/messages-count-tokens>.

No `logit_bias` or ban-token mechanism exists — `stop_sequences` operates on string matches, not token IDs.

---

## 9. Beta / experimental features

Enabled via `anthropic-beta: <header>` (comma-separable). Selected current surface:

- **Message Batches API** — `POST /v1/messages/batches`. Async, 50% discount, most batches finish <1h. Source: <https://platform.claude.com/docs/en/build-with-claude/batch-processing>. Not ZDR-eligible.
- **Files API** — upload + reference by `file_id` in `document`/`image` sources. (Referenced across docs; specific beta header unverified.)
- **Citations** — `citations: {enabled: true}` on `document` blocks; response text blocks carry `citations` array; streams `citations_delta`. GA on most models.
- **PDF support** — `document` block with `application/pdf` source (GA).
- **Code execution tool** — server sandbox (`code_execution_20260120`); produces `container` object in response.
- **Web search / web fetch** — server tools, GA.
- **Memory tool** — `memory_20250818` client-side tool, GA.
- **Context editing** — tool-result clearing, thinking-block clearing. Beta.
- **Server-side compaction** — auto-summarizes older turns; beta on Opus/Sonnet 4.6.
- **1M-token context window** — GA on Claude Mythos Preview, Opus 4.6, Sonnet 4.6 (per context-windows docs; earlier models cap at 200k). The earlier `context-1m-2025-08-07` beta header gated this on Sonnet 4 (unverified whether still required on 4.6).
- **Extended output** — `output-300k-2026-03-24` beta header raises output cap to 300k for Opus/Sonnet 4.6 in Batch API.
- **Interleaved thinking** — `interleaved-thinking-2025-05-14` (see §3).
- **MCP connector** — `mcp-client-2025-11-20`.
- **Computer use** — `computer-use-2025-11-24` / `computer-use-2025-01-24`.
- **Advisor tool** — `advisor-tool-2026-03-01`.

---

## 10. What is NOT available

| Capability | Status on Anthropic API |
| --- | --- |
| `logprobs` / `top_logprobs` | Not exposed, product stance |
| Full logit distributions | Not exposed |
| Integer token IDs (input or output wire) | Not exposed; `count_tokens` returns counts only |
| `logit_bias` / token steering | Not exposed |
| `n > 1` / parallel sampling per request | Not supported (caller must issue N requests) |
| Custom samplers (min_p, typical_p, mirostat, etc.) | Only `temperature`, `top_p`, `top_k`, `stop_sequences` |
| Seed / deterministic sampling | No public `seed` parameter (unverified whether an internal determinism guarantee exists) |
| LoRA / fine-tune upload | Not offered on first-party API |
| First-party embeddings | Not offered; Anthropic points users to Voyage AI partnership |
| Raw KV-cache handle / prefix export | Not exposed; prompt caching is opaque server-managed |
| Speculative decoding hooks | Not exposed |
| Echo / prompt-logprobs | Not exposed |

Anything that requires reaching below the string-level API surface (logits, token IDs, samplers, cache tensors) will need to be either modeled as `Optional`/`None` for the Anthropic provider in whisper-agent's abstraction, or served exclusively from whisper-tensor's own engine.
