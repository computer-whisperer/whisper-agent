#!/usr/bin/env bash
#
# Probe OpenAI Responses API behavior to validate the design assumptions
# behind a `previous_response_id`-based stateful continuation refactor.
#
# Pick auth mode via $1: `api_key` (env: OPENAI_API_KEY) or `codex`
# (default; reads ~/.codex/auth.json). Prints each test's request summary
# plus the response status and key extracted fields.
#
# What we're trying to learn:
#   1) Does store=true + prev_id actually thread context server-side?
#      (Send turn 2 with ONLY a delta; expect the model to reference turn 1.)
#   2) What does the API do if we send full replay AND prev_id together?
#      (Duplicate, error, or harmless?)
#   3) What's the exact error body when prev_id is invalid?
#   4) Does the chatgpt-subscription host honor prev_id? (Documented, but
#      it's been wrong before — image-gen 404, gpt-5 rejection.)
#   5) Does an image_generation_call output item round-trip via prev_id?
#      (Turn 2: "make it brighter" without re-sending the image.)
#
# Output is verbose-but-structured: each test prints `==> testN <description>`
# and shows status, response.id, output[0] shape, and a truncated body.

set -uo pipefail

MODE="${1:-codex}"
case "$MODE" in
    api_key)
        : "${OPENAI_API_KEY:?set OPENAI_API_KEY to use api_key mode}"
        BASE="https://api.openai.com/v1"
        BEARER="$OPENAI_API_KEY"
        EXTRA_HEADER=()
        ;;
    codex)
        AUTH_FILE="${CODEX_AUTH_FILE:-$HOME/.codex/auth.json}"
        if [[ ! -f "$AUTH_FILE" ]]; then
            echo "codex mode: $AUTH_FILE not found" >&2
            exit 2
        fi
        if ! command -v jq >/dev/null 2>&1; then
            echo "codex mode requires jq" >&2
            exit 2
        fi
        BEARER="$(jq -r '.tokens.access_token // empty' "$AUTH_FILE")"
        ACCOUNT_ID="$(jq -r '.tokens.account_id // empty' "$AUTH_FILE")"
        if [[ -z "$BEARER" ]]; then
            echo "codex mode: no .tokens.access_token in $AUTH_FILE" >&2
            exit 2
        fi
        BASE="https://chatgpt.com/backend-api/codex"
        EXTRA_HEADER=()
        if [[ -n "$ACCOUNT_ID" ]]; then
            EXTRA_HEADER=(-H "chatgpt-account-id: $ACCOUNT_ID")
        fi
        ;;
    *)
        echo "usage: $0 [api_key|codex]" >&2
        exit 2
        ;;
esac

# ChatGPT-subscription host rejects stream:false. We always stream and
# extract the response from response.completed events.
STREAM=true
if [[ "$MODE" == api_key ]]; then
    # api.openai.com works with either; pick non-stream for simpler parsing
    # in the probe.
    STREAM=false
fi

# Pick a model for the chat driver. On Codex we have to use what /models
# advertises (gpt-5 plain isn't accepted any more); api_key uses gpt-4o
# which has been stable.
if [[ "$MODE" == codex ]]; then
    MODEL_RESP="$(curl -s "$BASE/models?client_version=99.0.0" \
        -H "Authorization: Bearer $BEARER" "${EXTRA_HEADER[@]}")"
    MODEL="$(echo "$MODEL_RESP" | jq -r '
        .models // [] | map(select(.supported_in_api and .visibility=="list"))
        | .[0].slug // empty')"
    if [[ -z "$MODEL" ]]; then
        echo "could not pick a model from $BASE/models — response:" >&2
        echo "$MODEL_RESP" | head -c 1000 >&2
        exit 3
    fi
else
    MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
fi
echo "==> probing $BASE with model=$MODEL stream=$STREAM"

# ---------- helpers ----------

# Fire one POST /responses; print status, then dump the parsed body
# nicely. Sets RESPONSE_ID and OUTPUT_TEXT for the caller.
RESPONSE_ID=""
OUTPUT_TEXT=""
LAST_BODY=""
post_responses() {
    local desc="$1"; shift
    local body="$1"; shift
    echo
    echo "==> $desc"
    echo "  REQUEST:"
    echo "$body" | jq -C '.' | sed 's/^/    /'

    local accept_h=("-H" "accept: application/json")
    if [[ "$STREAM" == true ]]; then
        accept_h=("-H" "accept: text/event-stream")
    fi

    local raw
    raw="$(curl -sS -w '\nHTTPSTATUS=%{http_code}' \
        -X POST "$BASE/responses" \
        -H "Authorization: Bearer $BEARER" \
        -H "content-type: application/json" \
        "${accept_h[@]}" \
        "${EXTRA_HEADER[@]}" \
        --data "$body")"
    local status="${raw##*HTTPSTATUS=}"
    local body_text="${raw%HTTPSTATUS=*}"
    LAST_BODY="$body_text"

    echo "  STATUS: $status"

    # If streaming, extract the `data:` line of the response.completed event.
    local final_payload="$body_text"
    if [[ "$STREAM" == true ]]; then
        final_payload="$(echo "$body_text" \
            | awk '
                /^data: / { sub(/^data: /, ""); buf = buf $0 "\n"; next }
                /^$/ { if (buf ~ /"type":"response\.completed"/) { print buf; exit } buf = ""; next }
            ')"
    fi

    if [[ -z "$final_payload" ]]; then
        echo "  (no response.completed event found; raw stream below)"
        echo "$body_text" | head -c 1500 | sed 's/^/    /'
        RESPONSE_ID=""
        OUTPUT_TEXT=""
        return
    fi

    # Pretty-print the final response.
    if echo "$final_payload" | jq -e . > /dev/null 2>&1; then
        # Capture id and any output_text content.
        RESPONSE_ID="$(echo "$final_payload" | jq -r '
            (.response.id // .id // empty)')"
        OUTPUT_TEXT="$(echo "$final_payload" | jq -r '
            ((.response.output // .output) // [])
            | map(select(.type=="message"))
            | first.content // []
            | map(select(.type=="output_text"))
            | first.text // empty')"
        echo "  RESPONSE.id: $RESPONSE_ID"
        echo "  OUTPUT.types: $(echo "$final_payload" | jq -c '
            ((.response.output // .output) // []) | map(.type)')"
        if [[ -n "$OUTPUT_TEXT" ]]; then
            echo "  TEXT (first 200 chars): ${OUTPUT_TEXT:0:200}"
        fi
        echo "  ERROR: $(echo "$final_payload" | jq -c '.error // .response.error // null')"
    else
        echo "  (non-JSON body, first 1500 bytes:)"
        echo "$body_text" | head -c 1500 | sed 's/^/    /'
        RESPONSE_ID=""
        OUTPUT_TEXT=""
    fi
}

# ---------- tests ----------

# Test 1: baseline. store=true, no prev_id, capture response.id.
post_responses "test1: baseline turn 1, store=true" "$(jq -nc \
    --arg model "$MODEL" --argjson stream "$STREAM" '{
    model: $model,
    instructions: "Reply concisely.",
    input: [
        {type:"message", role:"user", content:[
            {type:"input_text", text:"My favorite color is teal. Just acknowledge."}]}
    ],
    store: true,
    stream: $stream
}')"
ID1="$RESPONSE_ID"

if [[ -z "$ID1" ]]; then
    echo
    echo "ABORT: turn 1 did not return a response.id; bail." >&2
    exit 4
fi

# Test 2: delta-only follow-up. Input is JUST the new message.
post_responses "test2: turn 2 with prev_id, DELTA-ONLY input" "$(jq -nc \
    --arg model "$MODEL" --arg pid "$ID1" --argjson stream "$STREAM" '{
    model: $model,
    input: [
        {type:"message", role:"user", content:[
            {type:"input_text", text:"What did I just say my favorite color was? One word."}]}
    ],
    previous_response_id: $pid,
    store: true,
    stream: $stream
}')"

# Test 3: prev_id + full replay (the more "defensive" approach).
post_responses "test3: turn 2 with prev_id AND full replay (defensive shape)" "$(jq -nc \
    --arg model "$MODEL" --arg pid "$ID1" --argjson stream "$STREAM" '{
    model: $model,
    input: [
        {type:"message", role:"user", content:[
            {type:"input_text", text:"My favorite color is teal. Just acknowledge."}]},
        {type:"message", role:"user", content:[
            {type:"input_text", text:"What did I just say my favorite color was? One word."}]}
    ],
    previous_response_id: $pid,
    store: true,
    stream: $stream
}')"

# Test 4: invalid prev_id. Capture exact error.
post_responses "test4: invalid prev_id (TTL miss / not found)" "$(jq -nc \
    --arg model "$MODEL" --argjson stream "$STREAM" '{
    model: $model,
    input: [
        {type:"message", role:"user", content:[
            {type:"input_text", text:"Hi."}]}
    ],
    previous_response_id: "resp_does_not_exist_zzz_99999",
    store: true,
    stream: $stream
}')"

# Test 5: image_generation, turn 1.
post_responses "test5: image_generation turn 1 (force tool_choice)" "$(jq -nc \
    --arg model "$MODEL" --argjson stream "$STREAM" '{
    model: $model,
    instructions: "Invoke image_generation exactly once.",
    input: [
        {type:"message", role:"user", content:[
            {type:"input_text", text:"a small cartoon owl sitting on a branch"}]}
    ],
    tools: [{type:"image_generation", size:"1024x1024", quality:"low"}],
    tool_choice: {type:"image_generation"},
    store: true,
    stream: $stream
}')"
IMG_ID="$RESPONSE_ID"

if [[ -n "$IMG_ID" ]]; then
    # Test 6: edit the image, turn 2, prev_id, delta-only input.
    post_responses "test6: image edit via prev_id, delta-only" "$(jq -nc \
        --arg model "$MODEL" --arg pid "$IMG_ID" --argjson stream "$STREAM" '{
        model: $model,
        instructions: "Invoke image_generation exactly once.",
        input: [
            {type:"message", role:"user", content:[
                {type:"input_text", text:"now make it nighttime, with a moon visible"}]}
        ],
        tools: [{type:"image_generation", size:"1024x1024", quality:"low"}],
        tool_choice: {type:"image_generation"},
        previous_response_id: $pid,
        store: true,
        stream: $stream
    }')"
fi

echo
echo "==> done. The interesting bits to skim:"
echo "    - test2 should reference 'teal' if delta-only chaining works"
echo "    - test3 status & whether it errors / duplicates"
echo "    - test4 error body shape"
echo "    - test6 status: did the second image generation accept prev_id?"
