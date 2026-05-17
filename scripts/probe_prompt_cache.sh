#!/usr/bin/env bash
#
# Probe /responses prompt-cache behavior to settle two questions:
#
#   (Q1) When we send byte-identical request bodies back-to-back, is
#        cached_tokens stably near-full, or does it bounce? If it
#        bounces, OpenAI's load balancer is routing successive requests
#        to different cache shards — i.e. the cache miss is *routing*,
#        not a non-deterministic prefix on our side.
#
#   (Q2) Does `prompt_cache_key` change the picture? Per OpenAI's docs
#        it's a routing-affinity hint combined with the prefix hash.
#        Test:
#          - Whether the field is even accepted (codex route silently
#            drops some fields; api_key route is stricter).
#          - Whether identical-body bursts stabilize at near-full
#            cached_tokens when the key is present.
#          - Whether a *different* key on the same prefix still hits.
#
# Pick auth mode via $1: `api_key` (env: OPENAI_API_KEY) or `codex`
# (default; reads ~/.codex/auth.json). Output is one row per request,
# grouped by scenario: input_tokens, cached_tokens, cache_read%.

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

# ChatGPT-subscription host rejects stream:false. api.openai.com accepts
# both; pick stream off there for simpler JSON parsing.
STREAM=true
if [[ "$MODE" == api_key ]]; then
    STREAM=false
fi

# Pick a model. On codex, the picker mirrors the live /models filter
# from the main probe — first user-visible, API-supported slug.
if [[ "$MODE" == codex ]]; then
    MODEL_RESP="$(curl -s "$BASE/models?client_version=99.0.0" \
        -H "Authorization: Bearer $BEARER" "${EXTRA_HEADER[@]}")"
    MODEL="$(echo "$MODEL_RESP" | jq -r '
        .models // [] | map(select(.supported_in_api and .visibility=="list"))
        | .[0].slug // empty')"
    if [[ -z "$MODEL" ]]; then
        echo "could not pick a codex model — response:" >&2
        echo "$MODEL_RESP" | head -c 800 >&2
        exit 3
    fi
else
    MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
fi
echo "==> probing $BASE  model=$MODEL  stream=$STREAM"

# ---------- prefix construction ----------
#
# Want >>1024 tokens (the cache floor) but capped on cost. Build a chunk
# of pseudo-content with a per-run nonce so each script invocation gets
# a fresh prefix, then repeat it to reach the target size.
#
# ~4 chars/token English → ~4 KB ≈ 1k tokens. Aim for ~10 KB / ~2.5k
# tokens to comfortably exceed the cache floor and have headroom for
# the divergence point. The nonce sits inside the body to defeat
# prior-run caching but is identical across all requests in a single
# run, so within-run cache hits are still legal.

RUN_NONCE="$(date +%s%N)-$$-$RANDOM"
CHUNK="The following is a fictional reference document for testing prompt caching behavior."
CHUNK="$CHUNK Run nonce: $RUN_NONCE."
CHUNK="$CHUNK It contains no useful information and should be ignored by the model."

# Repeat a lipsum-ish line until we cross ~9 KB. Generated deterministically
# from a hash of the run nonce so each run is unique-but-stable.
LIPSUM_LINE="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam."

INSTRUCTIONS="$CHUNK"
for _ in $(seq 1 60); do
    INSTRUCTIONS+=" $LIPSUM_LINE"
done
INSTRUCTIONS+=" When asked, reply with the single word OK."

echo "==> instructions: $(printf %s "$INSTRUCTIONS" | wc -c) chars (~$(($(printf %s "$INSTRUCTIONS" | wc -c) / 4)) tokens estimated)"

# ---------- helper: one request, print one row ----------

# Args: <scenario_label> <body_json>
# Prints: "  <label>  in=N  cached=K  cache%=X.X%  status=200  id=resp_..."
post_one() {
    local label="$1"; shift
    local body="$1"; shift

    local accept_h=("-H" "accept: application/json")
    if [[ "$STREAM" == true ]]; then
        accept_h=("-H" "accept: text/event-stream")
    fi

    local raw status body_text
    raw="$(curl -sS -w '\nHTTPSTATUS=%{http_code}' \
        -X POST "$BASE/responses" \
        -H "Authorization: Bearer $BEARER" \
        -H "content-type: application/json" \
        "${accept_h[@]}" \
        "${EXTRA_HEADER[@]}" \
        --data "$body")"
    status="${raw##*HTTPSTATUS=}"
    body_text="${raw%HTTPSTATUS=*}"

    # Extract the response.completed JSON (or use body_text directly for non-stream).
    local final_payload="$body_text"
    if [[ "$STREAM" == true ]]; then
        final_payload="$(echo "$body_text" \
            | awk '
                /^data: / { sub(/^data: /, ""); buf = buf $0 "\n"; next }
                /^$/ { if (buf ~ /"type":"response\.completed"/) { print buf; exit } buf = ""; next }
            ')"
    fi

    if [[ -z "$final_payload" ]] || ! echo "$final_payload" | jq -e . >/dev/null 2>&1; then
        printf "  %-22s status=%s  (non-JSON body, first 300 chars below)\n" \
            "$label" "$status"
        echo "$body_text" | head -c 300 | sed 's/^/      /'
        echo
        return
    fi

    # Pluck the usage fields. Path depends on whether the completed event is
    # wrapped in `{ response: { ... } }` (stream) or the top-level body is
    # already the response (non-stream).
    local in_toks cached_toks resp_id err
    in_toks=$(echo "$final_payload" | jq -r '
        ((.response // .) | .usage.input_tokens) // 0')
    cached_toks=$(echo "$final_payload" | jq -r '
        ((.response // .) | .usage.input_tokens_details.cached_tokens) // 0')
    resp_id=$(echo "$final_payload" | jq -r '
        ((.response // .) | .id) // "?"')
    err=$(echo "$final_payload" | jq -c '
        ((.response // .) | .error) // null')

    local pct="0.0"
    if [[ "$in_toks" -gt 0 ]]; then
        pct=$(awk "BEGIN { printf \"%.1f\", ($cached_toks / $in_toks) * 100 }")
    fi
    printf "  %-22s status=%s  in=%-6s cached=%-6s cache%%=%s%%  id=%s  err=%s\n" \
        "$label" "$status" "$in_toks" "$cached_toks" "$pct" "$resp_id" "$err"
}

# ---------- body builders ----------
#
# Build the body in JSON-clean form via jq so embedded quotes/newlines
# in INSTRUCTIONS don't break things.

base_body_no_key() {
    jq -nc \
        --arg model "$MODEL" \
        --arg instr "$INSTRUCTIONS" \
        --argjson stream "$STREAM" '{
        model: $model,
        instructions: $instr,
        input: [
            {type:"message", role:"user", content:[
                {type:"input_text", text:"Say OK."}
            ]}
        ],
        store: false,
        stream: $stream
    }'
}

base_body_with_key() {
    local key="$1"
    jq -nc \
        --arg model "$MODEL" \
        --arg instr "$INSTRUCTIONS" \
        --arg pck "$key" \
        --argjson stream "$STREAM" '{
        model: $model,
        instructions: $instr,
        input: [
            {type:"message", role:"user", content:[
                {type:"input_text", text:"Say OK."}
            ]}
        ],
        store: false,
        stream: $stream,
        prompt_cache_key: $pck
    }'
}

# ---------- scenarios ----------

N=5  # bursts per scenario; small to keep cost low

echo
echo "==> S1: $N identical requests, NO prompt_cache_key"
echo "    (control: stability of cached_tokens under load-balancer routing)"
for i in $(seq 1 "$N"); do
    post_one "S1.req${i}" "$(base_body_no_key)"
done

echo
echo "==> S2: $N identical requests, SAME prompt_cache_key='probe-$RUN_NONCE-A'"
echo "    (test: does affinity hint stabilize the burst?)"
KEY_A="probe-$RUN_NONCE-A"
for i in $(seq 1 "$N"); do
    post_one "S2.req${i}" "$(base_body_with_key "$KEY_A")"
done

echo
echo "==> S3: 1 request with DIFFERENT key='probe-$RUN_NONCE-B', same prefix"
echo "    (test: does a different key still hit the existing cache, or partition?)"
KEY_B="probe-$RUN_NONCE-B"
post_one "S3.req1" "$(base_body_with_key "$KEY_B")"

echo
echo "==> S4: 1 more request with KEY_A again (after the different-key probe)"
echo "    (test: does KEY_A's affinity survive the intervening KEY_B request?)"
post_one "S4.req1" "$(base_body_with_key "$KEY_A")"

echo
echo "==> S5: 1 request with prompt_cache_key='' (empty string)"
echo "    (sanity: does empty value 400 / is it treated as 'unset'?)"
post_one "S5.req1" "$(base_body_with_key "")"

echo
echo "==> S6: 1 request with a clearly-bogus key (special chars, long)"
echo "    (sanity: does the codex route validate format?)"
BOGUS_KEY='!@#$%^&*()_+-=[]{}|;:'\''",.<>/?'"$(printf 'X%.0s' $(seq 1 200))"
post_one "S6.req1" "$(base_body_with_key "$BOGUS_KEY")"

echo
echo "==> done. Interpretation guide:"
echo "    - S1: if cached% bounces between low/high → load-balancer routing instability"
echo "          (the production bug). If stably high → premise wrong; non-determinism is"
echo "          elsewhere."
echo "    - S2 vs S1: if S2 stabilizes near 100%, prompt_cache_key fixes the routing."
echo "    - S3: if cached% is still high, the key biases routing but doesn't partition"
echo "          the cache. If low, the key partitions strictly (i.e. wrong key = miss)."
echo "    - S4 vs S2: if S4 drops, the intervening different-key request bumped KEY_A"
echo "          off its affine shard. If high, sticky."
echo "    - S5/S6: 200 means silently accepted; 400 means the route validates."
