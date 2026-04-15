# Industry LLM Completion APIs — State of Play (April 2026)

Survey of hosted LLM APIs (OpenAI, Google Gemini) and self-hosted inference
servers (vLLM, llama.cpp, TGI, SGLang, Ollama, LM Studio), scoped to what
surfaces are *stable* vs *experimental*, and what power-user features (logit
distributions, token-ID I/O, guided decoding, speculative decoding, LoRA
hot-swap) are accessible through each.

Anthropic is covered by a sibling document.

---

## 1. OpenAI APIs

OpenAI currently operates two parallel completion surfaces.

### 1.1 Chat Completions (`/v1/chat/completions`) — legacy but widely supported

Still the de-facto industry "shape." Request fields:

```jsonc
{
  "model": "gpt-4o-2024-08-06",
  "messages": [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}],
  "tools": [{"type": "function", "function": {...json-schema...}}],
  "tool_choice": "auto" | "none" | "required" | {"type":"function","function":{"name":"..."}},
  "response_format": {"type":"json_schema","json_schema":{...}},
  "temperature": 0..2, "top_p": 0..1,
  "n": 1, "stop": ["..."] ,
  "logit_bias": {"<token_id>": -100..100},
  "logprobs": true, "top_logprobs": 0..20,
  "seed": 12345,
  "stream": true,
  "max_completion_tokens": 4096
}
```

Response: `choices[*].{message, finish_reason, logprobs}` plus
`usage.{prompt_tokens, completion_tokens, total_tokens}`. When `logprobs:true`,
`choices[*].logprobs.content[*]` returns `{token, logprob, bytes, top_logprobs:[{token,logprob,bytes},...]}`
for each generated token, with `top_logprobs` capped at **20** alternatives
per position (was 5 originally, raised in 2024). No access to the full vocab
distribution.

`logit_bias` takes **token IDs as keys**, so clients have to tokenize
client-side with `tiktoken` to know which IDs to bias. This is the only place
in the hosted OpenAI API where token IDs cross the wire, and it's *input
only* — no endpoint accepts `prompt_token_ids` or returns `token_ids`.
([docs](https://platform.openai.com/docs/api-reference/chat/create))

**Structured outputs** via `response_format: {type:"json_schema", strict:true}`
guarantee schema adherence (not just valid JSON) on `gpt-4o-2024-08-06+`. It is
enforced server-side via constrained decoding.
([guide](https://platform.openai.com/docs/guides/structured-outputs))

### 1.2 Responses API (`/v1/responses`) — successor, 2025+

Released March 2025 as the agentic replacement for Chat Completions +
Assistants. Differences:

- **Stateful** via `store:true` + `previous_response_id` — server retains
  conversation and reasoning trace between calls; clients no longer have to
  re-send full history.
- **Built-in tools** hosted by OpenAI: `web_search`, `file_search`,
  `computer_use`, `code_interpreter`, `image_generation`, plus remote **MCP**
  servers, invoked in-loop within a single request.
- **Reasoning items** (`o3`, `o4-mini`, `gpt-5` families): internal
  chain-of-thought is preserved server-side and *never* returned to client;
  clients get summaries via `reasoning.summary`.
- **Background mode** for long o-series runs: kick off async, poll or stream.
- **`include` parameter** selectively adds extra output
  (`message.output_text.logprobs`, `reasoning.encrypted_content`,
  `file_search_call.results`, etc.).
- Input/output is item-oriented (`input_text`, `input_image`, `function_call`,
  `function_call_output`, `reasoning`) rather than the flat messages array.

Historically the Responses API did *not* support `logprobs` at launch; it was
added later as an `include` option for non-reasoning models. Reasoning models
still do not expose logprobs.
([new-tools blog](https://openai.com/index/new-tools-and-features-in-the-responses-api/),
[migrate guide](https://platform.openai.com/docs/guides/migrate-to-responses))

### 1.3 Other OpenAI endpoints

| Endpoint                | Status                                                           |
| ----------------------- | ---------------------------------------------------------------- |
| `/v1/embeddings`        | Stable. `text-embedding-3-*`. Supports batch.                    |
| `/v1/batch`             | 50% discount, 24h SLA. Supports chat, responses, embeddings, moderations. |
| `/v1/files`             | Upload for batch / file_search / fine-tuning.                    |
| `/v1/fine_tuning/jobs`  | SFT, DPO, and now reinforcement fine-tuning (RFT).               |
| `/v1/realtime`          | GA; WebSocket and WebRTC for low-latency voice/text. The original Beta variant sunsets 2026-05-07. |
| `/v1/audio/{speech,transcriptions,translations}` | Stable. `tts-1`, `whisper-1`, `gpt-4o-transcribe`. |
| `/v1/assistants`        | **Deprecated**; sunset 2026-08-26. Replaced by Responses + Conversations. |

([deprecations](https://developers.openai.com/api/docs/deprecations))

---

## 2. Google Gemini API

Single primary endpoint `:generateContent` (and `:streamGenerateContent`),
with Vertex AI and AI Studio variants.

```jsonc
{
  "contents": [
    {"role":"user","parts":[{"text":"..."},{"inline_data":{"mime_type":"image/png","data":"<b64>"}}]}
  ],
  "systemInstruction": {"parts":[{"text":"..."}]},
  "generationConfig": {
    "temperature": 0..2, "topP": 0..1, "topK": 1..40,
    "candidateCount": 1..8, "maxOutputTokens": 8192,
    "stopSequences": ["..."], "seed": 42,
    "responseMimeType": "application/json",
    "responseSchema": {...openapi-subset...},
    "responseLogprobs": true, "logprobs": 1..20,
    "thinkingConfig": {"includeThoughts": true, "thinkingBudget": 0..24576,
                       "thinkingLevel": "low"|"medium"|"high"}
  },
  "safetySettings": [{"category":"HARM_CATEGORY_*","threshold":"BLOCK_*"}],
  "tools": [{"functionDeclarations":[{"name":"...","description":"...","parameters":{...}}]},
            {"googleSearch": {}}, {"codeExecution": {}}],
  "toolConfig": {"functionCallingConfig": {"mode":"AUTO"|"ANY"|"NONE"}}
}
```

**Logprobs** are supported on Vertex (`responseLogprobs:true`, `logprobs:1..20`)
and return the chosen-token logprob plus up to 20 alternatives per position —
functionally equivalent to OpenAI's cap.
([blog](https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/),
[reference](https://ai.google.dev/api/generate-content))

**Thinking**: Gemini 2.5 series uses `thinkingBudget` (explicit token budget);
Gemini 3 uses `thinkingLevel` ("low/medium/high"). `includeThoughts:true`
returns summary thought parts marked with `thought:true` inside the candidate
parts array. ([thinking](https://ai.google.dev/gemini-api/docs/thinking))

**Response shape** is noticeably different from OpenAI. The model returns
`candidates[*].content.parts[*]`, where each part is a discriminated union
(`text`, `functionCall`, `inlineData`, `executableCode`, `codeExecutionResult`,
or a thought-summary `text` with `thought:true`). Usage is in
`usageMetadata.{promptTokenCount, candidatesTokenCount, cachedContentTokenCount,
totalTokenCount}`.

**No token-ID I/O.** No `prompt_token_ids`, no logit_bias. Google runs
`/countTokens` and a separate tokenizer offering, but tokens never cross the
completion wire as IDs.

---

## 3. "OpenAI-compatible" as the de-facto standard

By 2026 the Chat Completions JSON shape is effectively the REST protocol for
local LLM inference. Servers implementing `/v1/chat/completions` (and usually
`/v1/completions`, `/v1/embeddings`, `/v1/models`):

- **vLLM** — full compat plus extensive non-standard extensions.
- **llama.cpp `llama-server`** — OpenAI-compat + richer native `/completion`.
- **SGLang** — OpenAI-compat; the scripting DSL is separate.
- **TGI** — Messages API OpenAI-compat. **Project is in maintenance mode as
  of 2025**; HuggingFace now recommends vLLM / SGLang / llama.cpp going
  forward. ([TGI docs](https://huggingface.co/docs/text-generation-inference/en/index))
- **Ollama** — `/v1/*` mirrors OpenAI on port 11434.
- **LM Studio** — `/v1/*` on port 1234, OpenAI-compat.
- **Text Generation WebUI**, **LocalAI**, **KoboldCpp** — same story.

### Common non-standard extensions (several of these originated in vLLM and
propagated)

| Extension                   | Purpose                                                   |
| --------------------------- | --------------------------------------------------------- |
| `guided_json` / `guided_regex` / `guided_grammar` / `guided_choice` | Constrained decoding via Outlines / lm-format-enforcer / xgrammar. |
| `guided_decoding_backend`   | Pick backend (outlines / lm-format-enforcer / xgrammar).  |
| `prompt_logprobs`           | Per-prompt-token logprobs (not just generated tokens).    |
| `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty` | Samplers beyond OpenAI's. |
| `best_of`, `use_beam_search`, `length_penalty` | Parallel sampling / beam. |
| `ignore_eos`, `skip_special_tokens`, `include_stop_str_in_output` | Fine control over decoding termination and text. |
| `return_token_ids` (vLLM ≥ 0.10.2) | Return `prompt_token_ids` and per-choice `token_ids` alongside strings. |
| `top_n_tokens` (TGI)        | Full per-position top-N with logprob.                     |

### Token IDs across the wire

Only self-hosted servers bother. **vLLM** accepts `prompt_token_ids` on the
native engine interface and (as of v0.10.2) returns them via
`return_token_ids:true` to eliminate retokenization drift in agent RL.
**llama.cpp** has first-class `/tokenize` and `/detokenize` endpoints and
`/completion` accepts pre-tokenized `prompt` arrays of IDs. **TGI** exposes
prefill token details via `decoder_input_details:true`. No hosted provider
does any of this.
([vLLM agent-lightning blog](https://vllm.ai/blog/agent-lightning))

---

## 4. Self-hosted servers — richer surfaces

### 4.1 vLLM

- **Logprobs**: `logprobs:N` up to the full vocab (server-gated by
  `--max-logprobs`); returns strings + IDs + logprobs. `prompt_logprobs:N`
  returns the same for every prompt token — used for rescoring and training
  data collection.
- **Guided decoding**: JSON schema / regex / grammar / choice, backed by
  Outlines, lm-format-enforcer, or xgrammar, switchable per request.
- **Speculative decoding**: `--speculative_model`, draft+target, EAGLE /
  Medusa / n-gram drafters; works with structured outputs as of 2025.
- **LoRA**: `--enable-lora`, `--max-loras`, `--max-lora-rank`. LoRA adapters
  hot-swap per request via `model` field; adapters keyed into prefix-cache
  hash so KV reuse is LoRA-safe.
- **Prefix caching**: `--enable-prefix-caching`, radix-style; automatic.
- **Beam search**: `use_beam_search:true` + `best_of`.
- **Native engine** (non-HTTP): full `SamplingParams` surface; accepts
  `prompt_token_ids`.

### 4.2 llama.cpp `llama-server`

- `/completion`: the native, non-OpenAI path with the richest surface —
  `n_probs` (top-N per token), `logit_bias` (by ID or by text substring),
  `grammar` (GBNF), `json_schema`, `mirostat` v1/v2, `dry_*`, `samplers`
  ordering, `cache_prompt`, `slot_id` pinning.
- `/v1/chat/completions`: OpenAI-compat; some native options (`grammar`,
  `cache_prompt`) accessible as extensions.
- `/tokenize`, `/detokenize`, `/embedding`, `/slots`, `/props`, `/metrics`.
- `reasoning_format` parses thinking content for DeepSeek-style models.
- Runs single-model per process; no dynamic LoRA hot-swap at the server
  level (LoRA is loaded at startup).

### 4.3 TGI (HuggingFace, maintenance mode)

- `/generate`, `/generate_stream`, plus OpenAI `/v1/chat/completions`
  Messages API.
- `top_n_tokens`: top-N logprob alternates per output token.
- `decoder_input_details:true`: prefill token details (IDs + logprobs).
- `grammar` parameter for guided JSON / regex.
- Watermarking, tensor-parallel, Paged/Flash attention.
- **In maintenance**: HF now contributes to vLLM/SGLang instead. Not
  recommended for new deployments.

### 4.4 SGLang

- OpenAI-compat server with extensions mirroring vLLM's guided decoding.
- **xgrammar** backend: the fastest open-source structured-output engine
  (up to 10× JSON decoding vs alternatives).
- **RadixAttention**: prefix KV cache organized as a radix tree, giving
  near-free reuse across forks/branches.
- **SGLang DSL** (Python-embedded): `gen`, `fork`, `choices`, control flow,
  structured parallelism. This is *not* JSON-over-HTTP — it's a program
  model for multi-branch prompting where the server understands the DAG.
- Used at scale (xAI, LinkedIn, NVIDIA) as of 2026.

---

## 5. Industry convergence — assessment

**Has the industry regressed to pure text-in / text-out?** For hosted APIs,
effectively yes. OpenAI and Gemini (and by extension Anthropic) exchange
**strings** plus a small envelope of structured sub-types (tool calls, image
parts, thought summaries). Token IDs never cross the wire in either
direction; logit distributions are truncated to top-20; full-vocab logits are
not exposed by any major hosted provider and will not be, because they're
fingerprints of proprietary model weights.

**Token IDs between client and server**: only in self-hosted. vLLM and
llama.cpp are the gold standard. This gap is load-bearing for RL and
distillation workloads — which is precisely why vLLM shipped
`return_token_ids`.

**Logit distributions**: truncated top-N (20 on hosted, unlimited on
self-hosted, subject to `--max-logprobs`). vLLM and llama.cpp expose the
cleanest interfaces; TGI's `top_n_tokens` is equivalent but the project is
frozen.

**Minimum common surface** that every serious provider has converged to:

1. Messages with `{role, content}`, roles ⊇ `{system, user, assistant, tool}`.
2. Multi-turn conversations via message list (stateless) or thread/response
   id (stateful).
3. Function/tool calling: JSON-schema-declared tools, model emits
   `tool_call`s, client returns `tool_result`s.
4. Streaming via SSE — token deltas with tool-call deltas.
5. Structured outputs via JSON schema (guaranteed on OpenAI, Gemini, vLLM,
   llama.cpp, SGLang, TGI).
6. `usage.{prompt_tokens, completion_tokens, total_tokens}` (or vendor
   equivalent).
7. `temperature`, `top_p`, `stop`, `max_tokens`, `seed`.

**Gap between "OpenAI-compatible" and what a power-user needs**: significant.

- Full-vocab logprobs / prompt_logprobs (for RL advantages, distillation).
- Token-ID I/O (for drift-free RL, exact replay, custom tokenization).
- Guided grammar beyond JSON (regex, CFG).
- Logit processors (arbitrary server-side logit manipulation).
- Speculative decoding config, draft model selection.
- LoRA hot-swap keyed per request.
- Multi-draft / branched sampling with cache sharing (RadixAttention-style).
- Access to reasoning traces as first-class data, not summaries.

None of these are part of the OpenAI wire format and none are being
standardized. Any abstraction that targets "OpenAI-compatible" alone will
force whisper-agent to throw away most of whisper-tensor's unique value.

---

## Primary sources

- OpenAI API — https://platform.openai.com/docs/api-reference
- OpenAI Responses — https://openai.com/index/new-tools-and-features-in-the-responses-api/
- OpenAI deprecations — https://developers.openai.com/api/docs/deprecations
- Gemini generateContent — https://ai.google.dev/api/generate-content
- Gemini thinking — https://ai.google.dev/gemini-api/docs/thinking
- Gemini logprobs — https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/
- vLLM OpenAI server — https://docs.vllm.ai/en/stable/serving/openai_compatible_server/
- vLLM token IDs — https://vllm.ai/blog/agent-lightning
- vLLM speculative decoding — https://docs.vllm.ai/en/latest/features/speculative_decoding/
- llama.cpp server — https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- TGI — https://huggingface.co/docs/text-generation-inference/en/index
- SGLang — https://sgl-project.github.io/ , https://www.lmsys.org/blog/2024-01-17-sglang/
- Ollama / LM Studio OpenAI-compat — https://lmstudio.ai/docs/developer
