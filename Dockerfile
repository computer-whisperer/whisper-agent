# syntax=docker/dockerfile:1.7
#
# whisper-agent server image.
#
# Bundles whisper-agent (with the webui embedded) plus the shared MCP
# sidecars (web fetch, web search) into a single debian-slim image.
# The host-side sandbox stack ships separately as the AUR package
# whisper-agent-host — this image does NOT contain whisper-agent-sandbox
# or whisper-agent-mcp-host.
#
# Volumes:
#   /etc/whisper-agent/        config (RO mount, must contain whisper-agent.toml)
#   /var/lib/whisper-agent/    runtime data (RW mount, holds pods/ and audit.jsonl)
#
# Build:   podman build -t whisper-agent .
# Inspect: podman run --rm -it whisper-agent /usr/local/bin/whisper-agent --help

ARG RUST_VERSION=1.92
ARG DEBIAN_VERSION=bookworm

# ---- chef base ---------------------------------------------------------------
# Shared base for planner and builder: pinned cargo-chef, the wasm32
# target, and a wasm-pack binary.
FROM rust:${RUST_VERSION}-${DEBIAN_VERSION} AS chef
WORKDIR /app
RUN cargo install cargo-chef --locked --version 0.1.77 \
    && rustup target add wasm32-unknown-unknown \
    && curl -sSfL https://rustwasm.github.io/wasm-pack/installer/init.sh | sh

# ---- planner: emit the dependency recipe ------------------------------------
# Reads only Cargo.toml + Cargo.lock content, so the resulting recipe.json
# is bit-identical across source-only changes — the builder's `cargo
# chef cook` step gets a layer cache hit until deps actually change.
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ---- builder: cook deps, build wasm bundle, build server binaries -----------
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Cook host-target deps for the whole workspace. The wasm crate is
# built separately by wasm-pack below; its (smaller) dep tree
# recompiles each build, but the bulk of compile time lives here.
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
# wasm bundle first — the agent crate embeds it via rust-embed at
# compile time. RUSTFLAGS picks the wasm-compatible getrandom backend.
RUN RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
    wasm-pack build crates/whisper-agent-webui --target web --release
# Server binaries (with the freshly-built webui baked in).
RUN cargo build --release \
    -p whisper-agent \
    -p whisper-agent-mcp-fetch \
    -p whisper-agent-mcp-search

# ---- runtime ----------------------------------------------------------------
FROM debian:${DEBIAN_VERSION}-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

# Fixed UID — keeps PVC ownership stable across image rebuilds.
RUN groupadd --system --gid 10001 whisper-agent \
    && useradd --system --uid 10001 --gid whisper-agent \
        --no-create-home --shell /usr/sbin/nologin whisper-agent

COPY --from=builder /app/target/release/whisper-agent          /usr/local/bin/whisper-agent
COPY --from=builder /app/target/release/whisper-agent-mcp-fetch  /usr/local/bin/whisper-agent-mcp-fetch
COPY --from=builder /app/target/release/whisper-agent-mcp-search /usr/local/bin/whisper-agent-mcp-search
COPY packaging/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# /etc/whisper-agent — empty, expected to be overlaid with a ConfigMap/Secret
# /var/lib/whisper-agent — owned by the runtime user; expected to be a PVC
RUN mkdir -p /etc/whisper-agent /var/lib/whisper-agent \
    && chown -R whisper-agent:whisper-agent /var/lib/whisper-agent

USER whisper-agent
WORKDIR /var/lib/whisper-agent

ENV RUST_LOG=info,whisper_agent=info,tower_http=info
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
