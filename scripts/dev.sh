#!/usr/bin/env bash
#
# Dev convenience: build everything and run the whisper-agent stack.
#
# `whisper-agent-host-daemon` is the host-env provider — each thread that
# binds a host_env gets its own landlock-isolated MCP host, provisioned
# on demand by the daemon over a WebSocket dial-in to the scheduler at
# /v1/host_env_link. No "fallback MCP"; threads without a bound host env
# run with only shared MCP tools (web fetch / search).
#
# All runtime artifacts (workspace root, audit log, persisted pods/threads)
# land under ./sandbox/ at the repo root. Gitignored; preserved across runs
# so persistence survives a restart.
#
# Env overrides:
#   SANDBOX            default: $REPO_ROOT/sandbox
#   LISTEN_SERVER      default: 127.0.0.1:8080
#   LISTEN_FETCH       default: 127.0.0.1:9830  (web-fetch MCP daemon)
#   LISTEN_SEARCH      default: 127.0.0.1:9831  (web-search MCP daemon)
#   LISTEN_IMAGEGEN    default: 127.0.0.1:9832  (image-gen MCP daemon)
#   IMAGEGEN_BACKEND   default: openai          (name of the [backends.X] entry
#                                               in whisper-agent.toml whose auth
#                                               drives image generation. Must be
#                                               kind = "openai_responses".)
#
# Flags:
#   --skip-wasm       skip the wasm-pack build (use when only Rust code changed)
#   --no-fetch        skip starting the web-fetch daemon (no `web_fetch` tool)
#   --no-search       skip starting the web-search daemon (no `web_search` tool).
#                     Auto-set when BRAVE_API_KEY is absent from whisper-agent.toml's
#                     [secrets] table.
#   --no-imagegen     skip starting the image-gen daemon (no `image_generate` tool).
#                     Auto-set when the daemon fails to come up (missing backend,
#                     wrong kind, missing API key, etc.) — we try-and-skip rather
#                     than duplicating config validation here in shell.
#
# Config discovery: the whisper-agent binary searches (in order)
# $XDG_CONFIG_HOME/whisper-agent/whisper-agent.toml,
# $HOME/.config/whisper-agent/whisper-agent.toml, then ./whisper-agent.toml.
# This script relies on that — it does not pass `--config`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SANDBOX="${SANDBOX:-$REPO_ROOT/sandbox}"
LISTEN_SERVER="${LISTEN_SERVER:-127.0.0.1:8080}"
LISTEN_FETCH="${LISTEN_FETCH:-127.0.0.1:9830}"
LISTEN_SEARCH="${LISTEN_SEARCH:-127.0.0.1:9831}"
LISTEN_IMAGEGEN="${LISTEN_IMAGEGEN:-127.0.0.1:9832}"
IMAGEGEN_BACKEND="${IMAGEGEN_BACKEND:-openai}"

SKIP_WASM=0
USE_FETCH=1
USE_SEARCH=1
USE_IMAGEGEN=1
for arg in "$@"; do
    case "$arg" in
        --skip-wasm)    SKIP_WASM=1 ;;
        --no-fetch)     USE_FETCH=0 ;;
        --no-search)    USE_SEARCH=0 ;;
        --no-imagegen)  USE_IMAGEGEN=0 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
    esac
done

mkdir -p "$SANDBOX"

# Pre-shared bearer the host-daemon presents on the WS upgrade. Generated
# on first run, preserved across restarts so running processes keep
# authenticating. The same file is referenced by --auth-daemon (server side)
# and --token-file (daemon side). Rotate by deleting and restarting both.
DAEMON_TOKEN_FILE="$SANDBOX/host-daemon-token"
if [[ ! -s "$DAEMON_TOKEN_FILE" ]]; then
    echo "==> generating host-daemon token at $DAEMON_TOKEN_FILE"
    umask 077
    head -c 32 /dev/urandom | od -An -vtx1 | tr -d ' \n' > "$DAEMON_TOKEN_FILE"
    echo >> "$DAEMON_TOKEN_FILE"
    chmod 600 "$DAEMON_TOKEN_FILE"
fi

# Build the webui wasm bundle BEFORE the main binary. whisper-agent
# embeds `crates/whisper-agent-webui/pkg/` at compile time via
# rust-embed, so building the binary against a stale pkg/ baked in
# the previous wasm output and the fresh wasm-pack rewrite wouldn't
# take effect until the NEXT dev.sh run. Putting wasm-pack first
# keeps wasm changes and the serving binary in lockstep.
if [[ "$SKIP_WASM" -eq 0 ]]; then
    echo "==> building whisper-agent-webui (wasm)"
    RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
        wasm-pack build crates/whisper-agent-webui --target web
else
    echo "==> skipping wasm build (--skip-wasm)"
fi

# Build the main binary so we can use `whisper-agent config env` to
# resolve any [secrets] declared in the active whisper-agent.toml. That
# result feeds the auto-skip decision below, so it has to happen before we
# pick the sibling-daemon package list.
echo "==> building whisper-agent (release)"
cargo build --release -p whisper-agent

# shellcheck disable=SC1090
eval "$("$REPO_ROOT/target/release/whisper-agent" config env)"

# Skip the search daemon silently if no key is configured — the rest of the
# stack works fine without it.
if [[ "$USE_SEARCH" -eq 1 && -z "${BRAVE_API_KEY:-}" ]]; then
    echo "==> BRAVE_API_KEY absent from [secrets] — skipping web-search daemon"
    USE_SEARCH=0
fi

PACKAGES="-p whisper-agent-mcp-host -p whisper-agent-host-daemon"
if [[ "$USE_FETCH" -eq 1 ]]; then
    PACKAGES="$PACKAGES -p whisper-agent-mcp-fetch"
fi
if [[ "$USE_SEARCH" -eq 1 ]]; then
    PACKAGES="$PACKAGES -p whisper-agent-mcp-search"
fi
if [[ "$USE_IMAGEGEN" -eq 1 ]]; then
    PACKAGES="$PACKAGES -p whisper-agent-mcp-imagegen"
fi

echo "==> building ($PACKAGES) (release)"
# shellcheck disable=SC2086
cargo build --release $PACKAGES

CHILD_PIDS=()
cleanup() {
    echo
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "==> stopping pid=$pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

SHARED_HOST_ARGS=()
if [[ "$USE_FETCH" -eq 1 ]]; then
    echo "==> starting whisper-agent-mcp-fetch on $LISTEN_FETCH"
    "$REPO_ROOT/target/release/whisper-agent-mcp-fetch" \
        --listen "$LISTEN_FETCH" &
    CHILD_PIDS+=($!)

    for _ in $(seq 1 20); do
        if curl -sf -X POST "http://$LISTEN_FETCH/mcp" \
            -H 'content-type: application/json' \
            -d '{"jsonrpc":"2.0","id":1,"method":"ping"}' > /dev/null 2>&1
        then break; fi
        sleep 0.25
    done

    SHARED_HOST_ARGS+=(--shared-mcp-host "fetch=http://$LISTEN_FETCH/mcp")
fi

if [[ "$USE_SEARCH" -eq 1 ]]; then
    echo "==> starting whisper-agent-mcp-search on $LISTEN_SEARCH"
    "$REPO_ROOT/target/release/whisper-agent-mcp-search" \
        --listen "$LISTEN_SEARCH" &
    CHILD_PIDS+=($!)

    for _ in $(seq 1 20); do
        if curl -sf -X POST "http://$LISTEN_SEARCH/mcp" \
            -H 'content-type: application/json' \
            -d '{"jsonrpc":"2.0","id":1,"method":"ping"}' > /dev/null 2>&1
        then break; fi
        sleep 0.25
    done

    SHARED_HOST_ARGS+=(--shared-mcp-host "search=http://$LISTEN_SEARCH/mcp")
fi

# image-gen daemon: reads OpenAI auth from whisper-agent.toml's
# [backends.$IMAGEGEN_BACKEND] entry. If the backend is missing, the wrong
# kind, or its credentials don't resolve, the daemon exits during startup
# rather than half-running. We detect that here by polling /mcp for a few
# seconds; if it never answers, drop the wiring and continue without
# `image_generate`.
if [[ "$USE_IMAGEGEN" -eq 1 ]]; then
    echo "==> starting whisper-agent-mcp-imagegen on $LISTEN_IMAGEGEN (backend=$IMAGEGEN_BACKEND)"
    "$REPO_ROOT/target/release/whisper-agent-mcp-imagegen" \
        --listen "$LISTEN_IMAGEGEN" \
        --backend "$IMAGEGEN_BACKEND" &
    IMAGEGEN_PID=$!
    CHILD_PIDS+=($IMAGEGEN_PID)

    IMAGEGEN_READY=0
    for _ in $(seq 1 20); do
        if ! kill -0 "$IMAGEGEN_PID" 2>/dev/null; then
            break # daemon exited (missing backend, bad auth, etc.)
        fi
        if curl -sf -X POST "http://$LISTEN_IMAGEGEN/mcp" \
            -H 'content-type: application/json' \
            -d '{"jsonrpc":"2.0","id":1,"method":"ping"}' > /dev/null 2>&1
        then
            IMAGEGEN_READY=1
            break
        fi
        sleep 0.25
    done

    if [[ "$IMAGEGEN_READY" -eq 1 ]]; then
        SHARED_HOST_ARGS+=(--shared-mcp-host "imagegen=http://$LISTEN_IMAGEGEN/mcp")
    else
        echo "==> imagegen daemon failed to come up — continuing without image_generate" \
             "(set IMAGEGEN_BACKEND or pass --no-imagegen to silence)"
    fi
fi

echo "==> starting whisper-agent-host-daemon dialing $LISTEN_SERVER"
# Daemon dials into /v1/host_env_link with exponential-backoff retry,
# so it's fine to start before the server is up — it'll connect once
# the server's listener is ready.
"$REPO_ROOT/target/release/whisper-agent-host-daemon" \
    --server-url "ws://$LISTEN_SERVER/v1/host_env_link" \
    --token-file "$DAEMON_TOKEN_FILE" \
    --mcp-host-bin "$REPO_ROOT/target/release/whisper-agent-mcp-host" \
    --probe-workspace "$SANDBOX" &
CHILD_PIDS+=($!)

echo "==> starting whisper-agent on $LISTEN_SERVER (host-env workspace=$SANDBOX)"
echo "    open http://$LISTEN_SERVER/ in a browser"
# `local-landlock` is the daemon name we admit via --auth-daemon; the
# synthesized default pod's host env binds to it with this workspace.
"$REPO_ROOT/target/release/whisper-agent" serve \
    --listen "$LISTEN_SERVER" \
    --auth-daemon "local-landlock=$DAEMON_TOKEN_FILE" \
    --default-host-env-provider "local-landlock" \
    --default-host-env-workspace "$SANDBOX" \
    --audit-log "$SANDBOX/audit.jsonl" \
    --pods-root "$SANDBOX/pods" \
    --buckets-root "$SANDBOX/buckets" \
    "${SHARED_HOST_ARGS[@]}"
