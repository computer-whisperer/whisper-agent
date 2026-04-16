#!/usr/bin/env bash
#
# Dev convenience: build everything and run the whisper-agent stack.
#
# By default uses the sandbox daemon — each task gets its own
# landlock-isolated MCP host, provisioned on demand. Pass --no-sandbox
# to fall back to a single shared MCP host (old behavior).
#
# All runtime artifacts (workspace root, audit log, persisted tasks) land
# under ./sandbox/ at the repo root. The directory is gitignored and preserved
# across runs so task-JSON persistence survives a restart.
#
# Env overrides:
#   SANDBOX           default: $REPO_ROOT/sandbox
#   LISTEN_SERVER     default: 127.0.0.1:8080
#   LISTEN_SANDBOX    default: 127.0.0.1:9810  (sandbox daemon)
#   LISTEN_FETCH      default: 127.0.0.1:9830  (web-fetch MCP daemon)
#   LISTEN_MCP        default: 127.0.0.1:8800  (standalone MCP, --no-sandbox only)
#
# Flags:
#   --skip-wasm       skip the wasm-pack build (use when only Rust code changed)
#   --no-sandbox      use a single shared MCP host instead of the sandbox daemon
#   --no-fetch        skip starting the web-fetch daemon (no `web_fetch` tool)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SANDBOX="${SANDBOX:-$REPO_ROOT/sandbox}"
LISTEN_SERVER="${LISTEN_SERVER:-127.0.0.1:8080}"
LISTEN_SANDBOX="${LISTEN_SANDBOX:-127.0.0.1:9810}"
LISTEN_FETCH="${LISTEN_FETCH:-127.0.0.1:9830}"
LISTEN_MCP="${LISTEN_MCP:-127.0.0.1:8800}"

SKIP_WASM=0
USE_SANDBOX=1
USE_FETCH=1
for arg in "$@"; do
    case "$arg" in
        --skip-wasm)    SKIP_WASM=1 ;;
        --no-sandbox)   USE_SANDBOX=0 ;;
        --no-fetch)     USE_FETCH=0 ;;
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

mkdir -p "$SANDBOX"

PACKAGES="-p whisper-agent -p whisper-agent-mcp-host"
if [[ "$USE_SANDBOX" -eq 1 ]]; then
    PACKAGES="$PACKAGES -p whisper-agent-sandbox"
fi
if [[ "$USE_FETCH" -eq 1 ]]; then
    PACKAGES="$PACKAGES -p whisper-agent-mcp-fetch"
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

    # Wait for fetch daemon to accept connections (cheap MCP ping).
    for _ in $(seq 1 20); do
        if curl -sf -X POST "http://$LISTEN_FETCH/mcp" \
            -H 'content-type: application/json' \
            -d '{"jsonrpc":"2.0","id":1,"method":"ping"}' > /dev/null 2>&1
        then break; fi
        sleep 0.25
    done

    SHARED_HOST_ARGS+=(--shared-mcp-host "fetch=http://$LISTEN_FETCH/mcp")
fi

if [[ "$USE_SANDBOX" -eq 1 ]]; then
    # ---------- Sandboxed mode ----------
    echo "==> starting whisper-agent-sandbox on $LISTEN_SANDBOX"
    "$REPO_ROOT/target/release/whisper-agent-sandbox" \
        --listen "$LISTEN_SANDBOX" \
        --mcp-host-bin "$REPO_ROOT/target/release/whisper-agent-mcp-host" &
    CHILD_PIDS+=($!)

    for _ in $(seq 1 20); do
        if curl -sf "http://$LISTEN_SANDBOX/health" > /dev/null 2>&1; then break; fi
        sleep 0.25
    done

    echo "==> starting whisper-agent on $LISTEN_SERVER (sandbox workspace=$SANDBOX)"
    echo "    open http://$LISTEN_SERVER/ in a browser"
    "$REPO_ROOT/target/release/whisper-agent" serve \
        --listen "$LISTEN_SERVER" \
        --config "$REPO_ROOT/whisper-agent.toml" \
        --sandbox-daemon-url "http://$LISTEN_SANDBOX" \
        --sandbox-workspace "$SANDBOX" \
        --audit-log "$SANDBOX/audit.jsonl" \
        --state-dir "$SANDBOX/tasks" \
        "${SHARED_HOST_ARGS[@]}"
else
    # ---------- Standalone MCP host mode (legacy) ----------
    echo "==> starting whisper-agent-mcp-host on $LISTEN_MCP (workspace=$SANDBOX)"
    "$REPO_ROOT/target/release/whisper-agent-mcp-host" \
        --workspace-root "$SANDBOX" \
        --listen "$LISTEN_MCP" &
    CHILD_PIDS+=($!)

    for _ in $(seq 1 20); do
        if curl -sf -X POST "http://$LISTEN_MCP/mcp" \
            -H 'content-type: application/json' \
            -d '{"jsonrpc":"2.0","id":1,"method":"ping"}' > /dev/null 2>&1
        then break; fi
        if ! kill -0 "${CHILD_PIDS[0]}" 2>/dev/null; then
            echo "error: whisper-agent-mcp-host exited before accepting connections" >&2
            exit 1
        fi
        sleep 0.25
    done

    echo "==> starting whisper-agent on $LISTEN_SERVER"
    echo "    open http://$LISTEN_SERVER/ in a browser"
    "$REPO_ROOT/target/release/whisper-agent" serve \
        --listen "$LISTEN_SERVER" \
        --config "$REPO_ROOT/whisper-agent.toml" \
        --mcp-host-url "http://$LISTEN_MCP/mcp" \
        --audit-log "$SANDBOX/audit.jsonl" \
        --state-dir "$SANDBOX/tasks" \
        "${SHARED_HOST_ARGS[@]}"
fi
