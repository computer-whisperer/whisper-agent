#!/usr/bin/env bash
#
# Container entrypoint for the whisper-agent server image.
#
# Starts the shared MCP sidecars (web fetch, optionally web search) on
# loopback, then exec's `whisper-agent serve` listening dual-stack on
# port 8080. The agent serves the embedded webui at `/` and the
# multiplexed WebSocket protocol at `/ws`.
#
# Layout (matches the Dockerfile):
#   /etc/whisper-agent/whisper-agent.toml   — config, mounted RO
#   /var/lib/whisper-agent/{audit.jsonl,pods/} — data, mounted RW
#
# Overrides via env (rarely needed; the defaults match the image
# layout):
#   WHISPER_AGENT_CONFIG          /etc/whisper-agent/whisper-agent.toml
#   WHISPER_AGENT_PODS_ROOT       /var/lib/whisper-agent/pods
#   WHISPER_AGENT_AUDIT_LOG       /var/lib/whisper-agent/audit.jsonl
#   WHISPER_AGENT_LISTEN          [::]:8080
#   WHISPER_AGENT_LISTEN_FETCH    127.0.0.1:9830
#   WHISPER_AGENT_LISTEN_SEARCH   127.0.0.1:9831
#
# Extra args passed to this script flow through to `whisper-agent
# serve`, so the orchestrator can append flags like
# `--host-env-provider name=url` and `--host-env-provider-token
# name=path` without overriding the entrypoint.
#
# tini is PID 1 (set by the Dockerfile ENTRYPOINT) and reaps zombies;
# restart of crashed sidecars is the orchestrator's job.

set -euo pipefail

CONFIG="${WHISPER_AGENT_CONFIG:-/etc/whisper-agent/whisper-agent.toml}"
PODS_ROOT="${WHISPER_AGENT_PODS_ROOT:-/var/lib/whisper-agent/pods}"
AUDIT_LOG="${WHISPER_AGENT_AUDIT_LOG:-/var/lib/whisper-agent/audit.jsonl}"
LISTEN="${WHISPER_AGENT_LISTEN:-[::]:8080}"
LISTEN_FETCH="${WHISPER_AGENT_LISTEN_FETCH:-127.0.0.1:9830}"
LISTEN_SEARCH="${WHISPER_AGENT_LISTEN_SEARCH:-127.0.0.1:9831}"

if [[ ! -f "$CONFIG" ]]; then
    echo "entrypoint: config not found at $CONFIG" >&2
    echo "entrypoint: mount whisper-agent.toml into /etc/whisper-agent/" >&2
    exit 1
fi

# Lift [secrets] from the toml into the env so sibling daemons see them.
# Same trick scripts/dev.sh uses; the binary's `config env` subcommand
# emits `export KEY='VALUE'` lines for every entry in [secrets].
eval "$(whisper-agent config env --config "$CONFIG")"

EXTRA_ARGS=()

# Web fetch: no API key required, always start.
echo "entrypoint: starting whisper-agent-mcp-fetch on $LISTEN_FETCH"
whisper-agent-mcp-fetch --listen "$LISTEN_FETCH" &
EXTRA_ARGS+=(--shared-mcp-host "fetch=http://${LISTEN_FETCH}/mcp")

# Web search: needs BRAVE_API_KEY in [secrets]. Skip silently if absent
# — the rest of the agent works fine without it.
if [[ -n "${BRAVE_API_KEY:-}" ]]; then
    echo "entrypoint: starting whisper-agent-mcp-search on $LISTEN_SEARCH"
    whisper-agent-mcp-search --listen "$LISTEN_SEARCH" &
    EXTRA_ARGS+=(--shared-mcp-host "search=http://${LISTEN_SEARCH}/mcp")
else
    echo "entrypoint: BRAVE_API_KEY absent from [secrets] — skipping web-search sidecar"
fi

exec whisper-agent serve \
    --config "$CONFIG" \
    --listen "$LISTEN" \
    --pods-root "$PODS_ROOT" \
    --audit-log "$AUDIT_LOG" \
    "${EXTRA_ARGS[@]}" \
    "$@"
