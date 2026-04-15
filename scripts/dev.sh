#!/usr/bin/env bash
#
# Dev convenience: build everything, run whisper-agent-mcp-host in the background,
# and run whisper-agent in the foreground. Ctrl-C tears down both.
#
# All runtime artifacts (mcp-host workspace root, audit log, persisted tasks) land
# under ./sandbox/ at the repo root. The directory is gitignored and preserved across
# runs so task-JSON persistence survives a restart.
#
# Env overrides:
#   SANDBOX         default: $REPO_ROOT/sandbox
#   LISTEN_SERVER   default: 127.0.0.1:8080
#   LISTEN_MCP      default: 127.0.0.1:8800
#
# Flags:
#   --skip-wasm     skip the wasm-pack build (use when only Rust code changed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SANDBOX="${SANDBOX:-$REPO_ROOT/sandbox}"
LISTEN_SERVER="${LISTEN_SERVER:-127.0.0.1:8080}"
LISTEN_MCP="${LISTEN_MCP:-127.0.0.1:8800}"

SKIP_WASM=0
for arg in "$@"; do
    case "$arg" in
        --skip-wasm) SKIP_WASM=1 ;;
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

echo "==> building whisper-agent and whisper-agent-mcp-host (release)"
cargo build --release -p whisper-agent -p whisper-agent-mcp-host

if [[ "$SKIP_WASM" -eq 0 ]]; then
    echo "==> building whisper-agent-webui (wasm)"
    RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
        wasm-pack build crates/whisper-agent-webui --target web
else
    echo "==> skipping wasm build (--skip-wasm)"
fi

MCP_PID=""
cleanup() {
    if [[ -n "${MCP_PID:-}" ]] && kill -0 "$MCP_PID" 2>/dev/null; then
        echo
        echo "==> stopping whisper-agent-mcp-host (pid=$MCP_PID)"
        kill "$MCP_PID" 2>/dev/null || true
        wait "$MCP_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "==> starting whisper-agent-mcp-host on $LISTEN_MCP (workspace=$SANDBOX)"
"$REPO_ROOT/target/release/whisper-agent-mcp-host" \
    --workspace-root "$SANDBOX" \
    --listen "$LISTEN_MCP" &
MCP_PID=$!

# Wait for the mcp-host to start accepting requests (~5s timeout).
for _ in $(seq 1 20); do
    if curl -sf -X POST "http://$LISTEN_MCP/mcp" \
        -H 'content-type: application/json' \
        -d '{"jsonrpc":"2.0","id":1,"method":"ping"}' > /dev/null 2>&1
    then
        break
    fi
    if ! kill -0 "$MCP_PID" 2>/dev/null; then
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
    --state-dir "$SANDBOX/tasks"
