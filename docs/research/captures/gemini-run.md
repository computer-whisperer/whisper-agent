# Gemini CLI (`gemini -p --yolo`) — Wire Capture

Capture: `gemini.mitm`, CLI version `0.36.0`. 19 SSE requests to the Code
Assist endpoint, **of which 6 were RATE_LIMIT_EXCEEDED 429s** that the harness
transparently retried — only 13 are real model turns.

## Transport

- `POST https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse`
- Request body: `application/json`
- Response body: SSE stream of JSON objects (Google's classic `alt=sse` shape;
  each `data:` line carries one full `GenerateContentResponse`).
- One TCP/HTTP request per turn (HTTP/2 multiplexed), no persistent socket.
- This is the **internal "Code Assist" endpoint**, not the public
  `generativelanguage.googleapis.com` Gemini API. Wrapper field naming differs
  from the public docs.

## Auth

- `Authorization: Bearer <REDACTED>` — Google OAuth access token (refreshed
  via `POST oauth2.googleapis.com/token`, 1 request observed).
- `Authorization` and `x-goog-user-project` are the only auth headers.

## Endpoint Map

| Count | Endpoint | Purpose |
|---|---|---|
| 19 | `POST cloudcode-pa.googleapis.com/v1internal:streamGenerateContent` | agent loop (incl. 6 retries) |
| 11 | `POST play.googleapis.com/log` | Google Play telemetry |
| 10 | `POST cloudcode-pa.googleapis.com/v1internal:recordCodeAssistMetrics` | per-event usage metrics |
| 6  | `POST cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota` | quota check |
| 4  | `POST cloudcode-pa.googleapis.com/v1internal:fetchAdminControls` | admin policy |
| 2  | `POST cloudcode-pa.googleapis.com/v1internal:loadCodeAssist` | session bootstrap |
| 2  | `POST cloudcode-pa.googleapis.com/v1internal:listExperiments` | feature flags |
| 3  | `POST oauth2.googleapis.com/{token,tokeninfo}` | OAuth refresh + verify |

This is the noisiest control-plane of the three CLIs.

## Request Body

Top-level shape is double-wrapped — the standard `contents` /
`systemInstruction` / `tools` payload is nested under `request`:

```jsonc
{
  "model": "gemini-3-flash-preview",
  "project": "<REDACTED>",
  "user_prompt_id": "<REDACTED>",
  "request": {
    "session_id": "<REDACTED>",
    "systemInstruction": { "parts": [{"text": "..."}] },
    "contents": [ { "role": "user"|"model", "parts": [...] }, ... ],
    "tools":    [ { "functionDeclarations": [...] } ],
    "generationConfig": {...}
  }
}
```

Top-level keys: `model, project, request, user_prompt_id`.
Inner-`request` keys: `contents, generationConfig, session_id, systemInstruction, tools`.

`systemInstruction` is a single 24,411-char block (one `parts[0].text`). Begins:

> You are Gemini CLI, an autonomous CLI agent specializing in software
> engineering tasks. Your primary goal is to help users safely and effectively.
> # Core Mandates …

### Tool Declarations (16)

All tools live in a single `tools[0].functionDeclarations` array — Gemini does
not have JSON-schema-tools-as-array-of-typed-tools shape; it has *tool groups*
each holding a list of `functionDeclarations`. Schemas use the
`parametersJsonSchema` field (recent Gemini moved away from the older
`parameters` field that used a Gemini-specific schema dialect).

| Name | Purpose |
|---|---|
| `list_directory` | dir listing |
| `read_file` | file read |
| `grep_search` | regex search |
| `glob` | filename glob |
| `replace` | string replace |
| `write_file` | file write |
| `run_shell_command` | shell exec (`bash -c`) |
| `web_fetch` | URL extraction |
| `save_memory` | persistent memory |
| `google_web_search` | grounded search builtin |
| `enter_plan_mode` / `exit_plan_mode` | plan-mode toggle |
| `codebase_investigator` | sub-agent: codebase analysis |
| `cli_help` | sub-agent: gemini CLI Q&A |
| `generalist` | sub-agent: general-purpose |
| `activate_skill` | invoke local skill |

Representative full declaration (`read_file`):

```jsonc
{
  "name": "read_file",
  "description": "Reads and returns the content of a specified file. To maintain context efficiency, you MUST use 'start_line' and 'end_line' for targeted, surgical reads of specific sections. …",
  "parametersJsonSchema": {
    "type": "object",
    "properties": {
      "file_path":  {"type":"string", "description":"The path to the file to read."},
      "start_line": {"type":"number", "description":"Optional: The 1-based line number to start reading from."},
      "end_line":   {"type":"number", "description":"Optional: The 1-based line number to end reading at (inclusive)."}
    },
    "required": ["file_path"]
  }
}
```

## Response / Event Shape

Each SSE `data:` line carries one full `GenerateContentResponse` wrapped in a
top-level `response` object. Keys: `candidates, createTime, modelVersion,
responseId, usageMetadata`.

Representative event (verbatim, sanitized — note the `"thought": true` part
marking a *thinking* span, distinct from regular text):

```json
{"response":{
  "candidates":[
    {"content":{
       "role":"model",
       "parts":[
         {"thought": true,
          "text":"**Developing Snake in Rust**\n\nI've initiated a new Rust project …"},
         {"functionCall":{"name":"run_shell_command",
                          "args":{"command":"cargo init",
                                  "description":"Initialize a new Cargo project."}}}
       ]},
     "finishReason":"STOP"}
  ],
  "responseId":"…",
  "modelVersion":"gemini-3-flash-preview",
  "usageMetadata":{"promptTokenCount":6635,"candidatesTokenCount":37,
                   "thoughtsTokenCount":95,"cachedContentTokenCount":0,
                   "totalTokenCount":6767}
}}
```

`functionCall.args` arrives as a fully-formed JSON object (no streaming
deltas — gemini emits whole content parts per chunk, not token-by-token).
`thoughtSignature` (server-signed reasoning blob, redacted) appears on the
*subsequent* function-call part for thinking continuity, analogous to
codex's `encrypted_content`.

`finishReason` values seen: `STOP` (success) and `null` (rate-limit error
where the response carries an `error` envelope instead of `candidates`).

## Tool Results Returned to Model

Tool results are appended to `contents[]` as a `user`-role turn whose `parts`
contain `functionResponse` objects keyed **by tool name** (not call ID):

```jsonc
{ "role": "user",
  "parts": [
    { "functionResponse": {
        "name": "run_shell_command",
        "response": { "output": "Updating crates.io index\nerror: download …" }
    }}
  ]
}
```

This means parallel tool calls of the same name are not unambiguously
result-able from this format alone — though Gemini doesn't seem to issue
parallel calls in this run.

## Turn-by-turn Summary

(Successful turns only; 6 RATE_LIMIT_EXCEEDED retries omitted.)

| # | prompt | cached | cand | thoughts | Tool calls |
|---|---:|---:|---:|---:|---|
| 1  | 6,635  | 0      | 37    | 95    | run_shell_command `cargo init` |
| 2  | 6,729  | 5,876  | 45    | 15    | run_shell_command `cargo add crossterm` |
| 4  | 8,724  | 0      | 48    | 31    | read_file Cargo.toml |
| 6  | 6,953  | 5,862  | 81    | 25    | write_file Cargo.toml |
| 7  | 7,109  | 5,864  | 1,630 | 104   | write_file src/main.rs (full game) |
| 9  | 12,390 | 7,620  | 415   | 175   | replace (drop `rand` dep) |
| 10 | 11,036 | 6,361  | 544   | 146   | replace (more) |
| 11 | 11,903 | 9,580  | 37    | 9     | run_shell_command `cargo check` |
| 13 | 12,089 | 9,586  | 74    | 378   | write_file `.cargo/config.toml` (SSL workaround) |
| 14 | 14,986 | 11,484 | 49    | 11    | run_shell_command retry cargo check |
| 15 | 12,474 | 9,381  | 77    | 118   | run_shell_command `CARGO_HTTP_SSL_VERIFY=false cargo check` |
| 17 | 12,771 | 9,348  | 47    | 1,019 | run_shell_command `rm .cargo/config.toml` |
| 19 | 12,839 | 11,746 | 1,821 | 713   | — (final summary, no tool call) |

The skipped turns (3, 5, 8, 12, 16, 18) all returned an
`@type: type.googleapis.com/google.rpc.ErrorInfo` payload with
`reason: RATE_LIMIT_EXCEEDED` and a `quotaResetDelay` of 0.5–6.5 s; the
harness retried at the suggested delay.

Gemini did **not** complete `cargo check` cleanly — turns 11–17 are a long
debugging arc trying to bypass the mitmproxy SSL interception, eventually
giving up. The work directory has `Cargo.toml` + `src/main.rs` but no
`Cargo.lock` or `target/`.

## Token / Cache Totals (across all 19 SSE requests, including retries)

| metric | value |
|---|---|
| promptTokenCount | 136,638 |
| candidatesTokenCount | 4,905 |
| thoughtsTokenCount | 2,839 |
| cachedContentTokenCount | 92,708 |
| totalTokenCount | 144,382 |

`cachedContentTokenCount / promptTokenCount ≈ 68%`. Notably, the **first turn
of the run gets zero cache hit** (cold prefix), and **turn 4 also caches zero
tokens** even though all earlier turns are byte-stable — looks like a cache
miss caused by message-array growth crossing some threshold; turn 6 onwards
the cache rebuilds and serves consistently. Cache utilization is markedly
worse than Claude (≈98%) or Codex (≈92%).

## Telemetry / Non-Agent Traffic

| class | count |
|---|---|
| `play.googleapis.com/log` | 11 |
| `recordCodeAssistMetrics` | 10 |
| `retrieveUserQuota` | 6 |
| `fetchAdminControls` | 4 |
| `loadCodeAssist` | 2 |
| `listExperiments` | 2 |
| OAuth | 3 |

Total: 38 non-loop POSTs (vs 23 for Claude, 13 for Codex). Heavy admin /
quota / metrics overhead suggests this surface is shared with internal Google
Code Assist and inherits its observability conventions.

## Provider-Specific Surprises

1. **Double-wrapped request body**: standard Gemini content/tools fields are
   nested under `request`, with `model`/`project`/`user_prompt_id` floated to
   the outer level — implementations targeting the public Gemini API will need
   adapter glue for this internal endpoint.
2. **Six rate-limit retries in a single 13-turn run**, with the model's
   per-minute quota appearing to reset every ~10–15 s. The CLI wrapper
   transparently retries and the model surface stays available, but a naïve
   "request count = turn count" metric over-counts by ~50%.
3. **Whole-part responses, not token deltas**: a `functionCall` arrives in
   one SSE chunk, not as streaming JSON fragments. Simpler to parse than
   anthropic's `input_json_delta` or codex's grammar-fragment deltas.
4. **`thought: true` parts** are returned in the same `parts` array as
   regular text — the harness must filter them out of user-visible output.
   `thoughtSignature` blobs (analogous to codex `encrypted_content`) appear
   on later parts and must be echoed back.
5. **Tool results are keyed by tool name** in `functionResponse.name`, not
   by a unique call ID — parallel calls of the same name would be ambiguous.
6. **Heaviest control-plane** of the three: 38 admin requests for 13 model
   turns. Quota retrievals are particularly chatty (every few turns).
7. The CLI **gave up** when it couldn't bypass mitmproxy's SSL — final turn
   produced a summary explaining the failure rather than continuing to
   iterate, which is a different failure mode than Claude (which diagnosed
   the mitmproxy CA and pointed cargo at it).
