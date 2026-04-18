# Claude Code `/compact` — Wire Capture

How Claude Code implements conversation compaction, observed via mitmproxy
against CLI `2.1.114` on 2026-04-18. Raw flows in the scratch dir at
`compaction-research/claude-capture.flows` (not checked in — contains OAuth
tokens). Payload JSON for the key calls was extracted and inspected; this doc
summarizes.

Captured session: 1 user prompt ("take a look at …/snake-superclient.html, make
a client that beats it"), 4 Opus agent turns, then `/compact`. 12 wire calls
to `/v1/messages?beta=true` variants in total.

## TL;DR

Compaction is **one `/v1/messages?beta=true` call** that sends the whole
unsummarized conversation plus an appended user message containing an explicit
"summarize yourself" instruction. The response is plain text: an
`<analysis>…</analysis>` block followed by a `<summary>…</summary>` block
with nine mandated sections. That summary text then becomes the seed of a
fresh session.

No special endpoint, no server-side "summarize this session" primitive —
compaction is entirely driven by a prompt and run against the same model and
tool catalog as normal turns.

## Mechanism

1. The CLI constructs a normal-looking `/v1/messages` request with the full
   existing conversation untouched.
2. It appends **one new `user` message** whose text is the compaction
   instruction (see Appendix A).
3. It sets `max_tokens: 20000`, `thinking: {"type":"adaptive"}`,
   `output_config: {"effort":"xhigh"}`, and a server-side
   `context_management.edits` hint (see below).
4. The model streams back a `<analysis>…</analysis><summary>…</summary>`
   text response. No tool calls.
5. On session resume, the `<summary>` body is rendered as the first user
   message of a new session, prefixed with "This session is being continued
   from a previous conversation…". (This part is not in the capture — the
   user exited before resuming — but it matches what we see in new sessions'
   initial context.)

Because the prior conversation is identical to the prefix the server has
already cached from earlier turns, the compaction call is cheap: the response
reports `cache_read_input_tokens: 60533` vs `cache_creation: 5209` — only the
new instruction block paid creation cost.

## Request Shape (flow #10)

Top-level keys:

```
model, messages, system, tools,
max_tokens, thinking, output_config, stream,
context_management, metadata
```

| key | value | notes |
|---|---|---|
| `model` | `"claude-opus-4-7"` | same as the agent loop |
| `max_tokens` | `20000` | 20k summary budget |
| `stream` | `true` | SSE response |
| `thinking` | `{"type":"adaptive"}` | extended thinking enabled |
| `output_config` | `{"effort":"xhigh"}` | cranked up from normal `"high"` |
| `context_management` | `{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}` | **see below** |
| `tools` | 10 entries | full catalog still advertised despite "do not call" prompt |
| `system` | 4 text blocks, 26 254 chars | identical to normal-turn system, first block is the `x-anthropic-billing-header: cc_version=…` line |
| `messages` | 9 entries | 8 from the prior conversation + 1 new user instruction |

### `context_management.edits.clear_thinking_20251015`

New beta field, dated form `clear_thinking_20251015`. Tells the server to strip
prior `thinking` content blocks from cached messages while keeping everything
else (`"keep":"all"` here refers to messages, not thinking). This is
presumably a cache-friendly way to drop reasoning tokens from a long
conversation without invalidating the prefix hash — the server applies the
edit on its side and the client never has to rewrite messages.

The response echoes `context_management: {"applied_edits": []}` which on this
turn meant no edits actually ran — likely because thinking blocks were already
absent from the cached prefix.

### `messages` array

```
msg[0] user       [5 text blocks]        — original session init (system-reminders + prompt)
msg[1] assistant  [thinking, tool_use, tool_use]
msg[2] user       [tool_result, tool_result]
msg[3] assistant  [thinking, tool_use, tool_use]
msg[4] user       [tool_result, tool_result]
msg[5] assistant  [thinking, text, tool_use]
msg[6] user       [tool_result]
msg[7] assistant  [thinking, text]
msg[8] user       str 5581B               ← the compaction instruction
```

Everything up to `msg[7]` is the untouched turn-by-turn trace of the session.
`msg[8]` is the only thing the CLI added for compaction.

Note `msg[0]` is a multi-block user turn: Claude Code expands the session's
system reminders into separate text blocks alongside the user prompt:

- block 0 (797 ch): `<system-reminder>` listing deferred tools
- block 1 (2987 ch): `<system-reminder>` listing available skills
- block 2 (1209 ch): `<system-reminder>` auto-mode notice
- block 3 (8312 ch): `<system-reminder>` containing `claudeMd` context
- block 4 (154 ch): the actual user-typed prompt

The compaction instruction at `msg[8]` is a single plain string (not blocks).

## The Compaction Prompt

Appendix A has it verbatim. Key structural features:

1. **Hard lead** with `CRITICAL: Respond with TEXT ONLY. Do NOT call any
   tools.` Then enumerates forbidden tools (`Read, Bash, Grep, Glob, Edit,
   Write, or ANY other tool`) and warns
   `Tool calls will be REJECTED and will waste your only turn — you will fail
   the task.` This redundancy is presumably defense-in-depth against models
   that tool-call reflexively — the tools are still declared in the request,
   so the model *could* call them.
2. Requires an **`<analysis>` block then a `<summary>` block**.
3. The `<analysis>` block is prompted to walk the conversation
   chronologically and enumerate (per message) user intent, approach,
   decisions, code/file details, errors+fixes, and specifically called-out
   user feedback.
4. The `<summary>` block must contain **nine named sections in a fixed
   order**:
   1. Primary Request and Intent
   2. Key Technical Concepts
   3. Files and Code Sections
   4. Errors and fixes
   5. Problem Solving
   6. All user messages (explicitly: "List ALL user messages that are not tool results")
   7. Pending Tasks
   8. Current Work
   9. Optional Next Step
5. Section 9 is instructed to include **direct verbatim quotes** from the
   latest messages — explaining why compacted sessions consistently reproduce
   the last user message word-for-word.
6. Section 6 ("All user messages") is what keeps the user's intent across the
   boundary even when individual messages didn't surface into later actions.
7. An example scaffold is inlined, showing the exact `<analysis>…</analysis>
   <summary>…</summary>` output structure.
8. A per-session "Compact Instructions" extension hook is documented in the
   prompt itself — it invites extra `## Compact Instructions` text from the
   session context. Claude Code surfaces this as user-configurable compaction
   guidance.
9. A closing `REMINDER: Do NOT call any tools.` line reinforces (1).

## Response Shape

Plain text, streamed via SSE. In the captured session:

- `input_tokens: 1762` (new)
- `cache_creation_input_tokens: 5209` (the new instruction block)
- `cache_read_input_tokens: 60533` (the entire prior conversation, cached)
- `output_tokens: 4559` (the summary itself)
- `stop_reason: "end_turn"`

The stream is purely `text_delta` events inside a single content block; no
`thinking_delta`, no `tool_use`. The model obeyed the no-tool directive.

The generated `<summary>` in this capture followed the 9-section template
exactly — including literal direct-quote blocks in section 9 ("If there is a
next step, include direct quotes from the most recent conversation showing
exactly what task you were working on and where you left off").

## Session Resumption

Not directly captured (user exited after compaction). Extrapolated from
observed behavior in prior Claude Code sessions:

- A new `/v1/messages` request is made with a fresh `messages` array.
- `messages[0]` (user) contains the `<summary>` body prefixed with text like
  "This session is being continued from a previous conversation that ran out
  of context. The summary below covers the earlier portion of the
  conversation." — the same wording that appeared in our own session's
  post-compaction context.
- Any files re-referenced in the summary may be reloaded into the new
  session; the two `/v1/messages/count_tokens?beta=true` calls observed in
  the trailing moments of the capture (sizing the `.html` files referenced
  in the summary) are consistent with this.

## Auxiliary Wire Observations (adjacent to compaction)

Three things in the capture are worth noting even though they are not part of
compaction itself.

### 1. Parallel security-monitor classifier

One `/v1/messages` call during the session (flow #6) was a **server-side
safety classifier** running in parallel with the main turn, not part of
compaction:

- Same model (`claude-opus-4-7`), `max_tokens: 64`, `stop_sequences: ["</block>"]`
- 29 674-char dedicated safety system prompt ("You are a security monitor for
  autonomous AI coding agents.")
- Input was the recent action to evaluate; output was `<block>no</block>` or
  `<block>yes</block><reason>…</reason>`.
- Invoked separately from the agent turn — the capture shows it slotted
  between normal turns.

Relevant here only as a design reference: Anthropic clearly runs a second
model as a blocking-style gate on agent actions. Whisper-agent's permission
system is a different shape (user prompts) but this is a pattern worth
keeping in mind when we later design autonomous/headless behavior gates.

### 2. `count_tokens` pre-flight

Two `POST /v1/messages/count_tokens?beta=true` calls were issued right after
compaction, sized with the snake-client HTML bodies (33 KB and 20 KB req
bodies, ~21 B responses). These are used to decide whether a Read-tool result
will fit within a token budget before actually returning the content to the
model. Not compaction-related but an interesting budget-control primitive
the main API exposes.

### 3. Token cost envelope

Whole-session totals across the 9 `/v1/messages` agent calls + 1 compaction
call:

- `input_tokens` (new): O(10²) — essentially nothing each turn beyond the
  newly-appended deltas, because the prefix is cached.
- `cache_read_input_tokens`: dominates; 60 533 alone on the compaction turn.
- `cache_creation_input_tokens`: 5 209 on the compaction turn (the
  instruction) vs 2 694 earlier turns — caching is doing the heavy lifting.

So compaction is **cheap** if you've been caching the main loop. The cost of
/compact ≈ cost of one additional turn of the size of its instruction block
plus the output-token cost of the summary (4 559 out tokens in this session).

## Surprises

1. **Tools stay advertised.** The compaction request keeps all 10 tools in
   `tools`. The prompt tells the model not to use them. They could have been
   dropped to harden the no-tool contract and reduce prompt size — instead
   the implementation leaves them in (presumably to preserve the identical
   cache prefix).
2. **Nine hard-coded sections.** The compacted summary is not free-form — it
   has a fixed schema with named sections. This is what makes Claude's
   compacted sessions so consistent in feel.
3. **"All user messages" is section 6.** The summary is structurally
   required to reproduce the user's conversation, not just distil it. This
   is load-bearing for long sessions where a user instruction several turns
   back is the only reason a later action is happening.
4. **Direct-quote mandate in "Next Step".** The verbatim quote is an
   explicit anti-drift mechanism — the compaction prompt specifically says
   "This should be verbatim to ensure there's no drift in task
   interpretation."
5. **`clear_thinking_20251015` is a new beta.** Dated name suggests an
   active-in-beta server-side feature. Worth checking docs for what edits
   are available — probably the right tool for long-running sessions that
   want to trim thinking without re-uploading messages.
6. **`output_config.effort = "xhigh"`.** Normal turns used `"high"`; the
   compaction turn upgraded to `"xhigh"`. This is an undocumented knob.
7. **Summary produced in a single turn.** No multi-turn refinement, no
   chunking. The model reads the whole conversation and writes the summary
   in one shot, bounded by `max_tokens: 20000`.
8. **Extension hook in the prompt.** The prompt explicitly mentions a
   `## Compact Instructions` extension block. This is the mechanism that
   user-level "compaction hints" get routed through.

## Implications for Whisper-Agent (sketch only)

Design constraints this capture makes clearer:

- A compaction primitive is **a special prompt, not a special endpoint** —
  whisper-agent can implement it purely in the scheduler/thread layer
  without any backend-adapter changes.
- It wants the **same model as the main thread** (to match tone and
  continuity) and the **same tool catalog** advertised (to keep the cache
  prefix).
- The summary **schema matters** — a free-form summary loses the "what was
  the user trying to do" thread. We probably want an analogous 6-to-9
  section schema, tuned to whisper-agent's domain (pods, behaviors, tasks).
- **Section "All user messages" is load-bearing**, especially for our
  long-running behavior threads where user intent may be far upstream.
- The **resumption seed** is just a user message containing the summary —
  fits cleanly into our existing ThreadConfig + messages model.
- Token budget: ~20k output tokens is a safe summary ceiling.

Open questions for the design pass:

- Trigger policy. Claude Code has an explicit `/compact` command; it also
  appears to auto-compact near context limits (we didn't capture an auto
  path). Where does whisper-agent want that decision to live — user,
  scheduler, thread?
- Behavior thread compaction. Long-running scheduled threads may compact
  repeatedly across days. Does each compaction carry forward the previous
  compacted summary, or does the summary get stored separately and threaded
  through the behavior's state?
- Tool result handling. Big tool results (file reads, `grep` output) are
  the usual reason for blowing context. Does compaction implicitly handle
  them, or do we want a separate tool-result-truncation primitive that runs
  before compaction ever triggers?

These belong in the design doc, not here.

---

## Appendix A — Verbatim Compaction Instruction (`msg[8]` of flow #10)

Text as received by the model. 5 581 chars. Whitespace preserved.

```
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>


REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task.
```

## Appendix B — How To Re-Capture

If we want another capture in the future (e.g. to catch auto-compaction):

```bash
# Terminal A — proxy
cd /home/christian/workspace/whisper-agent/compaction-research
mitmweb --mode regular@8888 --web-port 8889 \
    --set save_stream_file=./claude-capture.flows \
    --set web_open_browser=false

# Terminal B — driver
HTTPS_PROXY=http://127.0.0.1:8888 \
HTTP_PROXY=http://127.0.0.1:8888 \
NODE_EXTRA_CA_CERTS=$HOME/.mitmproxy/mitmproxy-ca-cert.pem \
claude
```

Then browse flows at `http://127.0.0.1:8889`. To post-process:

```bash
# summary of all /v1/messages calls
mitmdump -nr claude-capture.flows -s summarize.py

# dump specific flows to JSON files
FLOWS=6,10 mitmdump -nr claude-capture.flows -s extract.py
```

`summarize.py` and `extract.py` live alongside the `.flows` file.
