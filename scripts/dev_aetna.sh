#!/usr/bin/env bash
#
# Dev convenience for the aetna native client (whisper-agent-desktop-aetna).
#
# Runs the existing dev.sh stack in the background to bring up the
# server + daemons, waits for the listen port, then launches the
# desktop-aetna binary in the foreground. Closing the desktop window
# returns control to the shell; the EXIT trap then tears the server
# stack down.
#
# Companion to scripts/dev.sh: dev.sh runs the server (and bakes in
# the browser-ui wasm bundle for users hitting http://$LISTEN_SERVER);
# this script adds the native desktop client on top.
#
# Env overrides:
#   LISTEN_SERVER  default: 127.0.0.1:8080  (passed through to dev.sh)
#
# Flags:
#   --include-wasm  also build the browser-ui wasm bundle (default: skip,
#                   since the native desktop client doesn't need it — only
#                   browser users on http://$LISTEN_SERVER do)
#   any other       passed through to dev.sh (e.g. --no-fetch, --no-search)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LISTEN_SERVER="${LISTEN_SERVER:-127.0.0.1:8080}"

# Default to skipping the wasm build — the aetna native binary doesn't
# need the browser-ui pkg/. Pass --include-wasm to opt in (e.g. when
# also testing the browser ui side-by-side).
DEV_ARGS=(--skip-wasm)
PASSTHROUGH=()
for arg in "$@"; do
    case "$arg" in
        --include-wasm)
            DEV_ARGS=()
            ;;
        *)
            PASSTHROUGH+=("$arg")
            ;;
    esac
done

# Build the desktop-aetna binary up front. dev.sh blocks once the
# server starts; we don't want to be still compiling our binary at
# that point.
echo "==> building whisper-agent-desktop-aetna (release)"
cargo build --release -p whisper-agent-desktop-aetna

# Launch dev.sh in its own session so we can clean up the entire
# subtree (server + daemons + dev.sh's own EXIT trap) by signaling
# the process group. `setsid` makes DEV_PID the leader of a new
# process group with pgid==pid, so `kill -- -$DEV_PID` reaches
# every descendant.
echo "==> starting backend stack via dev.sh ${DEV_ARGS[*]:-} ${PASSTHROUGH[*]:-}"
LISTEN_SERVER="$LISTEN_SERVER" \
    setsid "$SCRIPT_DIR/dev.sh" "${DEV_ARGS[@]}" "${PASSTHROUGH[@]}" &
DEV_PID=$!

cleanup() {
    if kill -0 "$DEV_PID" 2>/dev/null; then
        echo
        echo "==> stopping dev.sh stack (pgid=$DEV_PID)"
        kill -- -"$DEV_PID" 2>/dev/null || true
        wait "$DEV_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Wait for the server's listen port to start accepting connections.
# 60s budget — covers a cold compile + slow first scheduler init.
HOST="${LISTEN_SERVER%:*}"
PORT="${LISTEN_SERVER##*:}"
echo "==> waiting for $HOST:$PORT"
READY=0
for _ in $(seq 1 240); do
    if (echo > "/dev/tcp/$HOST/$PORT") 2>/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$DEV_PID" 2>/dev/null; then
        echo "==> dev.sh exited before $HOST:$PORT opened" >&2
        exit 1
    fi
    sleep 0.25
done
if [[ "$READY" -ne 1 ]]; then
    echo "==> server never opened $HOST:$PORT (timed out)" >&2
    exit 1
fi

echo "==> launching whisper-agent-desktop-aetna against http://$LISTEN_SERVER"
# Pass an explicit empty token so the binary's login-form auto-skip
# matches: it requires both `--server` and a token source to skip the
# form. Loopback servers ignore the (empty) Authorization header
# anyway. Drop this flag once we want the script to exercise the
# login flow itself.
"$REPO_ROOT/target/release/whisper-agent-desktop-aetna" \
    --server "http://$LISTEN_SERVER" \
    --token ""
