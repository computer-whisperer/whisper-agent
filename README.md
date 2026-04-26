# whisper-agent

A self-hosted, headless agent loop. The conversation runs on a server you control; tools that touch the outside world (filesystems, processes, web fetches, search) cross a network boundary into separate MCP servers, never running in-process with the loop.

What you get out of the box:

- A web UI (egui+wasm, served by the agent itself) for driving long-lived conversations from any browser.
- Pluggable LLM backends: Anthropic, OpenAI Chat + Responses, Gemini, and any openai-compatible local endpoint (Ollama, LM Studio, llama.cpp).
- One conversation can reach into multiple managed POSIX hosts simultaneously, each gated by per-thread Landlock-isolated MCP servers.
- Optional shared MCP tools for web fetch and Brave web search.
- A CBOR-over-WebSocket protocol for native clients (an Android client lives in `android/`).

If you want a Claude Code / Codex / Gemini CLI-style agent that runs on a *server* instead of your laptop and can act on hosts other than the one running the loop, this is for you.

See [`docs/design_headless_loop.md`](docs/design_headless_loop.md) for the rationale.

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────┐
│  whisper-agent (server)                                     │
│  • pod / thread / resource scheduler                        │
│  • provider clients (Anthropic, OpenAI, Gemini, openai-     │
│    compatible local)                                        │
│  • serves the webui (egui+wasm) and a CBOR/WebSocket        │
│    protocol for clients                                     │
└──────────────┬──────────────────────────────────────────────┘
               │ MCP over streamable HTTP
   ┌───────────┴──────────────────────────┐
   │                                      │
┌──┴──────────────────────┐   ┌───────────┴─────────────────┐
│ shared MCP daemons      │   │ per-host sandbox daemon     │
│  • whisper-agent-mcp-   │   │  • whisper-agent-sandbox    │
│    fetch (HTTP fetch)   │   │    (landlock provisioner)   │
│  • whisper-agent-mcp-   │   │  • spawns whisper-agent-    │
│    search (Brave API)   │   │    mcp-host per thread      │
└─────────────────────────┘   └─────────────────────────────┘
```

**Server-side** (one container in deployment): `whisper-agent`, `whisper-agent-webui` (wasm bundle served by the agent), `whisper-agent-mcp-fetch`, `whisper-agent-mcp-search`.

**Host-side** (installed on each managed POSIX host as a systemd service): `whisper-agent-sandbox` and the `whisper-agent-mcp-host` binary it spawns.

## Quick start (5 minutes, local development)

Stand up the full stack on your laptop, talking to Anthropic with your API key. Everything binds to loopback; the sandbox writes its workspace under `./sandbox/` (gitignored).

### Prerequisites

- A recent stable Rust toolchain — `rustup default stable` if you don't have one.
- `wasm-pack` for the webui bundle: `cargo install wasm-pack`.
- Linux 5.13+ (the sandbox uses Landlock).
- An Anthropic API key.

### Steps

```sh
# 1. Seed your config from the template (gitignored, holds your key).
cp whisper-agent.toml.example whisper-agent.toml

# 2. Set your Anthropic key in the environment — the example config reads it
#    from $ANTHROPIC_API_KEY by default.
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Build everything and start the stack.
./scripts/dev.sh

# 4. Open http://127.0.0.1:8080/ in a browser.
```

`dev.sh` builds every crate (release), builds the webui wasm bundle, starts the sandbox daemon and the shared `web_fetch` MCP daemon, then starts `whisper-agent`. Runtime artifacts (workspace, audit log, persisted pods/threads, sandbox control token) land under `./sandbox/`. Pass `--help` for flags (`--skip-wasm`, `--no-fetch`, `--no-search`).

To also enable web search, drop a Brave Search API key into `whisper-agent.toml`'s `[secrets]` table as `BRAVE_API_KEY = "BSA..."` and re-run `dev.sh` — the search daemon auto-starts when the key is present.

### First steps in the webui

The server boots with a synthesized **default pod** wired to a Landlock-isolated workspace at `./sandbox/`. Open the webui, start a new thread, and type — the assistant has `read_file` / `write_file` / `bash` tools scoped to the workspace, plus `web_fetch` (and `web_search` if you set the Brave key).

The conceptual layout is **pods → threads → resources**: pods are persistent project directories with `pod.toml` capability rules, threads are individual conversations within a pod, and resources (backends, MCP hosts, host envs) live in the scheduler and are referenced by name. See [`docs/design_pod_thread_scheduler.md`](docs/design_pod_thread_scheduler.md).

## Configuration

`whisper-agent.toml.example` is the canonical reference — every section is documented inline. Briefly:

- `[backends.<name>]` — one table per LLM endpoint. `kind` selects the wire format; `auth` is a tagged union (`api_key` with `value` or `env`, plus subscription auth modes for OpenAI Codex and Gemini-CLI).
- `[secrets]` — env vars exported to sibling daemons via `whisper-agent config env` (e.g. `BRAVE_API_KEY` for web search).
- `[shared_mcp_hosts]` — singleton MCP server URLs all threads can dial. `dev.sh` registers the dev-stack ones via CLI flags; populate this for non-dev deployments.
- `[[host_env_providers]]` — sandbox-daemon URLs (`name`, `url`, optional `token_file`). Imported once into the durable runtime catalog at `<pods_root>/../host_env_providers.toml`; after that, the WebUI's Providers tab is the source of truth for add / update / remove.
- `[[auth.clients]]` / `[[auth.admins]]` — bearer tokens for non-loopback access. **Loopback always bypasses auth**; configure these only when exposing the server externally.
- `[embedding_providers.<name>]` / `[rerank_providers.<name>]` — for knowledge buckets; see [`docs/design_knowledge_db.md`](docs/design_knowledge_db.md).

Config search precedence: `--config <path>` → `$XDG_CONFIG_HOME/whisper-agent/whisper-agent.toml` → `$HOME/.config/whisper-agent/whisper-agent.toml` → `./whisper-agent.toml`.

## Production deployment

### Server-side (Docker / Kubernetes)

The repository ships a `Dockerfile` that bundles `whisper-agent`, the embedded webui, and the shared MCP sidecars (web fetch, web search) into one debian-slim image:

```sh
podman build -t whisper-agent .
```

The image expects:

- `/etc/whisper-agent/whisper-agent.toml` — config, mounted RW (so the webui can hot-swap backend credentials).
- `/var/lib/whisper-agent/` — runtime data (pods, audit log, durable provider catalog), mounted RW with uid 10001.

The entrypoint (`packaging/docker/entrypoint.sh`) starts the fetch sidecar unconditionally and the search sidecar when `BRAVE_API_KEY` is present in `[secrets]`. TLS is opt-in via `WHISPER_AGENT_TLS_CERT` / `WHISPER_AGENT_TLS_KEY` env vars (mount the standard k8s `kubernetes.io/tls` secret).

### Host-side (Arch / AUR)

The `whisper-agent-host` AUR package (PKGBUILD in `packaging/aur/`) installs `whisper-agent-sandbox` as a systemd service on a managed POSIX host. The systemd unit auto-generates a control-token at `/var/lib/whisper-agent/sandbox-control-token` on first start; copy that token to your central `whisper-agent` server (e.g. as a Kubernetes secret mounted as `--host-env-provider-token`) so the server can authenticate and provision per-thread MCP hosts.

```sh
# On each managed host:
sudo pacman -S whisper-agent-host    # or: makepkg -si in packaging/aur/
sudo systemctl enable --now whisper-agent-sandbox.service
sudo cat /var/lib/whisper-agent/sandbox-control-token   # copy to server-side secret
```

## Crates

| Crate | Role |
| --- | --- |
| `whisper-agent` (root) | The agent loop, HTTP/WS server, webui host. |
| `whisper-agent-webui` | Browser chat UI (egui + wasm), built with `wasm-pack`. |
| `whisper-agent-protocol` | CBOR-over-WebSocket protocol shared by server and webui. |
| `whisper-agent-sandbox` | Per-host daemon that provisions Landlock-isolated MCP host instances per thread. |
| `whisper-agent-mcp-host` | MCP server exposing a controlled slice of POSIX (read/write_file, bash). One per thread, spawned by the sandbox daemon. |
| `whisper-agent-mcp-fetch` | Shared MCP server exposing a guarded `web_fetch` tool. |
| `whisper-agent-mcp-search` | Shared MCP server exposing `web_search` (Brave Search API). |
| `whisper-agent-mcp-proto` | MCP / JSON-RPC types shared by the MCP servers. |
| `whisper-agent-desktop` | Native desktop client — egui app sharing UI code with `whisper-agent-webui`. |

Native Android client: see `android/`.

## Documentation

Design docs in [`docs/`](docs/):

- [`design_headless_loop.md`](docs/design_headless_loop.md) — why the loop is server-side
- [`design_pod_thread_scheduler.md`](docs/design_pod_thread_scheduler.md) — pods, threads, and resources
- [`design_permissions.md`](docs/design_permissions.md) — tool-boundary patterns
- [`design_functions.md`](docs/design_functions.md) — unified Function model (design draft)
- [`design_behaviors.md`](docs/design_behaviors.md) — autonomous pod behaviors (design)
- [`design_knowledge_db.md`](docs/design_knowledge_db.md) — embedded retrieval / knowledge buckets

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
