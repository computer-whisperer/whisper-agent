# whisper-agent

A server-resident agent loop that drives long-lived LLM conversations and reaches into managed POSIX hosts via MCP. The loop runs centrally; every operation that touches the outside world (filesystem, processes, services) crosses an MCP tool boundary, never running in-process with the loop. See [`docs/design_headless_loop.md`](docs/design_headless_loop.md) for the rationale.

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

## Crates

| Crate | Role |
| --- | --- |
| `whisper-agent` (root) | The agent loop, HTTP/WS server, webui host. |
| `whisper-agent-webui` | Browser chat UI (egui + wasm), built with `wasm-pack`. |
| `whisper-agent-protocol` | CBOR-over-WebSocket protocol shared by server and webui. |
| `whisper-agent-sandbox` | Per-host daemon that provisions landlock-isolated MCP host instances per thread. |
| `whisper-agent-mcp-host` | MCP server exposing a controlled slice of POSIX (read/write_file, bash). One per thread, spawned by the sandbox daemon. |
| `whisper-agent-mcp-fetch` | Shared MCP server exposing a guarded `web_fetch` tool. |
| `whisper-agent-mcp-search` | Shared MCP server exposing `web_search` (Brave Search API). |
| `whisper-agent-mcp-proto` | MCP / JSON-RPC types shared by the MCP servers. |

## Quickstart (development)

Prerequisites: a recent Rust toolchain, `wasm-pack`, and Linux 5.13+ (landlock).

```sh
./scripts/dev.sh
# open http://127.0.0.1:8080/
```

`dev.sh` builds every crate (release), builds the webui wasm bundle, starts the sandbox daemon and any shared MCP daemons configured by your secrets, then starts `whisper-agent`. Runtime artifacts (workspace, audit log, persisted pods/threads, sandbox control token) land under `./sandbox/`. Pass `--help` for flags (`--skip-wasm`, `--no-fetch`, `--no-search`).

## Configuration

`whisper-agent serve` searches for `whisper-agent.toml` in (in order):

1. `--config <path>` (explicit; errors if missing)
2. `$XDG_CONFIG_HOME/whisper-agent/whisper-agent.toml`
3. `$HOME/.config/whisper-agent/whisper-agent.toml`
4. `./whisper-agent.toml`

Without a config file, the server falls back to a single Anthropic backend driven by `--anthropic-api-key` (or `ANTHROPIC_API_KEY`) and `--model`.

The TOML file declares:

- a `[backends.<name>]` table per LLM endpoint (Anthropic, OpenAI Chat, OpenAI Responses, Gemini, openai-compatible)
- `[secrets]` — exported into the environment for sibling daemons via `whisper-agent config env`
- `[shared_mcp_hosts]` — named shared-MCP daemon URLs
- `[[host_env_providers]]` — sandbox-daemon seed entries (with optional `token_file`); imported once into the durable runtime catalog at `<pods_root>/../host_env_providers.toml`, after which the WebUI's Providers tab is the source of truth for add / update / remove

## Documentation

Design docs in [`docs/`](docs/):

- [`design_headless_loop.md`](docs/design_headless_loop.md) — why the loop is server-side
- [`design_pod_thread_scheduler.md`](docs/design_pod_thread_scheduler.md) — pods, threads, and resources
- [`design_permissions.md`](docs/design_permissions.md) — tool-boundary patterns
- [`design_functions.md`](docs/design_functions.md) — unified Function model (design draft)
- [`design_behaviors.md`](docs/design_behaviors.md) — autonomous pod behaviors (design)

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
