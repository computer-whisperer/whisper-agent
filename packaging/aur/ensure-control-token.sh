#!/usr/bin/env bash
#
# Generate a control-plane bearer token for whisper-agent-sandbox if the
# token file does not yet exist. Idempotent — safe to run on every
# service start.
#
# Usage: ensure-control-token <path>
#
# The token is 64 hex chars (256 bits of /dev/urandom). The file is
# created mode 0600, owned by whoever runs this script (the systemd
# unit runs it as the whisper-agent service user).

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <token-file>" >&2
    exit 2
fi

TOKEN_FILE="$1"

if [[ -s "$TOKEN_FILE" ]]; then
    exit 0
fi

# install(1) creates the file with the requested mode atomically; head
# then appends the token bytes. umask is set defensively for the rare
# case where install isn't available.
umask 077
install -m 0600 /dev/null "$TOKEN_FILE"
head -c 32 /dev/urandom | od -An -vtx1 | tr -d ' \n' >> "$TOKEN_FILE"
echo >> "$TOKEN_FILE"
