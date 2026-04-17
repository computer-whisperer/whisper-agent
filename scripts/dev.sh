#!/usr/bin/env bash
#
# Dev convenience: build everything and run the whisper-agent stack.
#
# The sandbox daemon is the host-env provider — each thread that binds
# a host_env gets its own landlock-isolated MCP host, provisioned on
# demand. No "fallback MCP"; threads without a bound host env run with
# only shared MCP tools (web fetch / search).
#
# All runtime artifacts (workspace root, audit log, persisted pods/threads)
# land under ./sandbox/ at the repo root. Gitignored; preserved across runs
# so persistence survives a restart.
#
# Env overrides:
#   SANDBOX           default: $REPO_ROOT/sandbox
#   LISTEN_SERVER     default: 127.0.0.1:8080
#   LISTEN_SANDBOX    default: 127.0.0.1:9810  (sandbox daemon)
#   LISTEN_FETCH      default: 127.0.0.1:9830  (web-fetch MCP daemon)
#   LISTEN_SEARCH     default: 127.0.0.1:9831  (web-search MCP daemon)
#
# Flags:
#   --skip-wasm       skip the wasm-pack build (use when only Rust code changed)
#   --no-fetch        skip starting the web-fetch daemon (no `web_fetch` tool)
#   --no-search       skip starting the web-search daemon (no `web_search` tool).
#                     Auto-set when BRAVE_API_KEY is unset in CLOUD_KEYS.txt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SANDBOX="${SANDBOX:-$REPO_ROOT/sandbox}"
LISTEN_SERVER="${LISTEN_SERVER:-127.0.0.1:8080}"
LISTEN_SANDBOX="${LISTEN_SANDBOX:-127.0.0.1:9810}"
LISTEN_FETCH="${LISTEN_FETCH:-127.0.0.1:9830}"
LISTEN_SEARCH="${LISTEN_SEARCH:-127.0.0.1:9831}"

SKIP_WASM=0
USE_FETCH=1
USE_SEARCH=1
for arg in "$@"; do
    case "$arg" in
        --skip-wasm)    SKIP_WASM=1 ;;
        --no-fetch)     USE_FETCH=0 ;;
        --no-search)    USE_SEARCH=0 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
    esac
done

if [[ ! -f CLOUD_KEYS.txt ]]; then
    echo "error: CLOUD_KEYS.txt not found at repo root" >&2
    echo "       create it with a single line: ANTHROPIC_API_KEY=sk-ant-..." >&2
    exit 1
fi
# shellcheck disable=SC1091
source CLOUD_KEYS.txt
export ANTHROPIC_API_KEY
export BRAVE_API_KEY="${BRAVE_API_KEY:-}"

if [[ "$USE_SEARCH" -eq 1 && -z "$BRAVE_API_KEY" ]]; then
    echo "==> BRAVE_API_KEY unset in CLOUD_KEYS.txt — skipping web-search daemon"
    USE_SEARCH=0
fi

mkdir -p "$SANDBOX"

PACKAGES="-p whisper-agent -p whisper-agent-mcp-host -p whisper-agent-sandbox"
if [[ "$USE_FETCH" -eq 1 ]]; then
    PACKAGES="$PACKAGES -p whisper-agent-mcp-fetch"
fi
if [[ "$USE_SEARCH" -eq 1 ]]; then
    PACKAGES="$PACKAGES -p whisper-agent-mcp-search"
fi

echo "==> building ($PACKAGES) (release)"
# shellcheck disable=SC2086
cargo build --release $PACKAGES

if [[ "$SKIP_WASM" -eq 0 ]]; then
    echo "==> building whisper-agent-webui (wasm)"
    RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
        wasm-pack build crates/whisper-agent-webui --target web
else
    echo "==> skipping wasm build (--skip-wasm)"
fi

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

echo "==> starting whisper-agent-sandbox on $LISTEN_SANDBOX"
"$REPO_ROOT/target/release/whisper-agent-sandbox" \
    --listen "$LISTEN_SANDBOX" \
    --mcp-host-bin "$REPO_ROOT/target/release/whisper-agent-mcp-host" &
CHILD_PIDS+=($!)

for _ in $(seq 1 20); do
    if curl -sf "http://$LISTEN_SANDBOX/health" > /dev/null 2>&1; then break; fi
    sleep 0.25
done

echo "==> starting whisper-agent on $LISTEN_SERVER (host-env workspace=$SANDBOX)"
echo "    open http://$LISTEN_SERVER/ in a browser"
# `local-landlock` is the catalog name we register the sandbox daemon
# under; the synthesized default pod's host env binds to it with this
# workspace.
"$REPO_ROOT/target/release/whisper-agent" serve \
    --listen "$LISTEN_SERVER" \
    --config "$REPO_ROOT/whisper-agent.toml" \
    --host-env-provider "local-landlock=http://$LISTEN_SANDBOX" \
    --default-host-env-provider "local-landlock" \
    --default-host-env-workspace "$SANDBOX" \
    --audit-log "$SANDBOX/audit.jsonl" \
    --pods-root "$SANDBOX/pods" \
    "${SHARED_HOST_ARGS[@]}"
