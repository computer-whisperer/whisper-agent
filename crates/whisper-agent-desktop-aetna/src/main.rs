//! whisper-agent-desktop-aetna — native window around the aetna chat UI.
//!
//! Sibling of `whisper-agent-desktop` (egui). Same WebSocket transport
//! (tokio + tungstenite, CBOR over `/ws`); different render stack
//! (`aetna-winit-wgpu` instead of eframe).
//!
//! Two phases, mirroring the egui sibling's `RootApp`:
//!   - **Login**: a [`LoginApp`] form collects server URL + token.
//!     Submission writes to `$XDG_CONFIG_HOME/whisper-agent/desktop.toml`
//!     (when "Remember" is checked) and transitions to Connected.
//!   - **Connected**: spawns the WS task and wraps a [`ChatApp`] in a
//!     [`DesktopApp`] that drains the inbound tokio channel each frame.
//!
//! CLI flags still override config. If both `--server` and a token
//! source are supplied (or loaded from config), the login screen is
//! skipped — `--login` forces it back.
//!
//! Threading:
//!   - main thread: winit event loop + aetna `Runner` paint
//!   - tokio pool: `ws_loop` reads/writes frames
//!   - bridge: a tokio `mpsc` carries inbound `InboundEvent`s into the
//!     UI thread; `App::before_build` drains the queue every frame
//!     before the rebuild reads its state.
//!
//! Wakeup caveat: aetna-winit-wgpu doesn't currently expose a
//! cross-thread "wake the event loop" primitive (no winit
//! `EventLoopProxy` re-export). To keep the WS task's frames visible
//! within a frame, we run with a fixed redraw cadence
//! (`HostConfig::redraw_interval`). This is a temporary scaffold —
//! ideally aetna grows a wakeup channel and we drop the cadence.

mod config;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aetna_core::prelude::Rect;
use aetna_core::{App, BuildCx, El, Selection, Theme, UiEvent};
use anyhow::{Context, Result, anyhow};
use clap::Parser;
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::{
    client::IntoClientRequest,
    http::header::{AUTHORIZATION, HeaderValue},
    protocol::Message,
};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use whisper_agent_aetna_ui::{
    ChatApp, Inbound, InboundEvent, LoginApp, LoginInput, SendFn, SubmitFn,
};
use whisper_agent_protocol::{ClientToServer, decode_from_server, encode_to_server};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "whisper-agent-desktop-aetna: native window around the whisper-agent aetna ui"
)]
struct Args {
    /// Base URL of a running whisper-agent server, e.g. `https://host:8443`
    /// or `http://127.0.0.1:8080`. The client appends `/ws` to derive the
    /// websocket endpoint, swapping the scheme to `ws`/`wss` as needed.
    /// Overrides the value saved in the config file.
    #[arg(long)]
    server: Option<String>,

    /// Bearer token for the server's `[[auth.clients]]` gate. Required for
    /// non-loopback servers; loopback servers accept any value (or none).
    /// Overrides the value saved in the config file.
    #[arg(long, env = "WA_TOKEN", conflicts_with = "token_file")]
    token: Option<String>,

    /// Read the bearer token from a single-line file — keeps the token
    /// off shell history and out of `ps` listings. Same semantic as
    /// `--token`; mutually exclusive.
    #[arg(long)]
    token_file: Option<PathBuf>,

    /// Force the login screen even when CLI flags or saved config
    /// would otherwise auto-connect. Useful for changing servers.
    #[arg(long)]
    login: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,whisper_agent_desktop_aetna=info")),
        )
        .init();

    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let args = Args::parse();
    let cli_token = resolve_cli_token(&args)?;
    let saved = config::load().unwrap_or_else(|e| {
        warn!("could not load saved config: {e:#}");
        config::Config::default()
    });

    // CLI overrides saved config for this session. An empty saved
    // field is treated as missing.
    let initial_server = args.server.clone().or_else(|| non_empty(saved.server));
    let initial_token = cli_token.or_else(|| non_empty(saved.token));

    let rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("build tokio runtime")?,
    );

    // RootApp owns the phase switch. `phase_signal` is how a successful
    // login submission inside `LoginApp::on_event` reaches us — the
    // submit callback writes here, and the next `before_build` reads
    // it and swaps from Login to Connected.
    let phase_signal: Arc<Mutex<Option<LoginInput>>> = Arc::new(Mutex::new(None));

    // Decide the initial phase based on what we have to start with.
    let initial_phase = match (args.login, initial_server.clone(), initial_token.clone()) {
        // Have everything and the user didn't ask for the form —
        // skip straight to Connected.
        (false, Some(server), Some(token)) => Phase::start_connected(server, token, rt.clone()),
        // Anything else: render the form, prefilled where possible.
        (_, server, token) => {
            let signal = phase_signal.clone();
            let submit: SubmitFn = Box::new(move |input: LoginInput| {
                if let Ok(mut slot) = signal.lock() {
                    *slot = Some(input);
                }
            });
            Phase::Login(LoginApp::new(server, token, submit))
        }
    };

    let root = RootApp {
        rt: rt.clone(),
        phase: initial_phase,
        phase_signal,
    };

    let viewport = Rect::new(0.0, 0.0, 1200.0, 800.0);
    // 30Hz cadence so WS frames get drained promptly without us
    // depending on user input to redraw. See the wakeup-caveat note at
    // the top of the file — this trades idle CPU for visibility into
    // background-arrived state.
    let host_config =
        aetna_winit_wgpu::HostConfig::default().with_redraw_interval(Duration::from_millis(33));
    aetna_winit_wgpu::run_with_config("whisper-agent", viewport, root, host_config)
        .map_err(|e| anyhow!("aetna runner: {e}"))?;
    Ok(())
}

fn non_empty(s: Option<String>) -> Option<String> {
    s.filter(|v| !v.trim().is_empty())
}

/// Top-level aetna [`App`] that switches between the login form and
/// the connected chat UI. Only one WS task runs at a time (in the
/// Connected phase). Mirrors `whisper-agent-desktop`'s `RootApp` so
/// users get the same flow whether they pick the egui or aetna client.
struct RootApp {
    rt: Arc<tokio::runtime::Runtime>,
    phase: Phase,
    /// `LoginApp::submit` writes here from inside `on_event`. The
    /// next `before_build` poll moves the credentials onto the main
    /// thread, persists them, spawns the WS task, and flips `phase`
    /// to `Connected`.
    phase_signal: Arc<Mutex<Option<LoginInput>>>,
}

#[allow(clippy::large_enum_variant)]
enum Phase {
    Login(LoginApp),
    Connected(DesktopApp),
}

impl Phase {
    /// Spin up the bridges + ws task for `(server, token)` and wrap a
    /// fresh `ChatApp` in `DesktopApp`. Used both for the
    /// auto-connect path (CLI / config gave us creds) and for the
    /// post-login transition.
    fn start_connected(server: String, token: String, rt: Arc<tokio::runtime::Runtime>) -> Phase {
        let ws_url = match derive_ws_url(&server) {
            Ok(u) => u,
            Err(e) => {
                warn!("invalid server URL {server:?}: {e:#}");
                // Fall back to the form so the user can fix it.
                let signal: Arc<Mutex<Option<LoginInput>>> = Arc::new(Mutex::new(None));
                let signal_for_submit = signal.clone();
                let submit: SubmitFn = Box::new(move |input: LoginInput| {
                    if let Ok(mut slot) = signal_for_submit.lock() {
                        *slot = Some(input);
                    }
                });
                let mut form = LoginApp::new(Some(server), non_empty(Some(token)), submit);
                form.set_error(format!("invalid server URL: {e:#}"));
                return Phase::Login(form);
            }
        };
        info!(url = %ws_url, "connecting");

        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel::<InboundEvent>();
        let (outbound_tx, outbound_rx) = mpsc::unbounded_channel::<ClientToServer>();

        let token_for_task = if token.is_empty() { None } else { Some(token) };
        let _ws_handle = rt.spawn(async move {
            match ws_loop(ws_url, token_for_task, outbound_rx, inbound_tx).await {
                Ok(()) => info!("ws loop exited"),
                Err(e) => error!("ws loop error: {e:#}"),
            }
        });

        let inbound_staging: Arc<Mutex<Vec<InboundEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let staging_for_bouncer = inbound_staging.clone();
        let _bouncer = rt.spawn(async move {
            let mut rx = inbound_rx;
            while let Some(event) = rx.recv().await {
                if let Ok(mut q) = staging_for_bouncer.lock() {
                    q.push(event);
                }
            }
        });

        let send_fn: SendFn = Box::new(move |msg: ClientToServer| {
            if let Err(e) = outbound_tx.send(msg) {
                warn!("outbound channel closed: {e}");
            }
        });

        let inbound_shared = Inbound::default();
        let inbound_for_app = inbound_shared.clone();
        let inner = ChatApp::new(inbound_for_app, send_fn);

        Phase::Connected(DesktopApp {
            inner,
            inbound_sink: inbound_shared,
            inbound_staging,
            _rt: rt,
        })
    }
}

impl App for RootApp {
    fn before_build(&mut self) {
        // Phase swap: drain the login signal and (if a submission
        // landed last frame) persist creds + spawn the WS task on
        // the way to the Connected phase.
        let pending = self
            .phase_signal
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        if let Some(input) = pending {
            persist_login(&input);
            self.phase = Phase::start_connected(input.server, input.token, self.rt.clone());
        }
        match &mut self.phase {
            Phase::Login(app) => app.before_build(),
            Phase::Connected(app) => app.before_build(),
        }
    }

    fn build(&self, cx: &BuildCx) -> El {
        match &self.phase {
            Phase::Login(app) => app.build(cx),
            Phase::Connected(app) => app.build(cx),
        }
    }

    fn on_event(&mut self, event: UiEvent) {
        match &mut self.phase {
            Phase::Login(app) => app.on_event(event),
            Phase::Connected(app) => app.on_event(event),
        }
    }

    fn selection(&self) -> Selection {
        match &self.phase {
            Phase::Login(app) => app.selection(),
            Phase::Connected(app) => app.selection(),
        }
    }

    fn theme(&self) -> Theme {
        match &self.phase {
            Phase::Login(app) => app.theme(),
            Phase::Connected(app) => app.theme(),
        }
    }
}

/// Persist a successful login submission to the user's config dir.
/// `remember = false` clears the token but keeps the server URL —
/// matches the egui sibling.
fn persist_login(input: &LoginInput) {
    let cfg = config::Config {
        server: Some(input.server.clone()),
        token: if input.remember && !input.token.is_empty() {
            Some(input.token.clone())
        } else {
            None
        },
    };
    if let Err(e) = config::save(&cfg) {
        warn!("could not save config: {e:#}");
    } else if let Ok(p) = config::path() {
        info!("saved login to {}", p.display());
    }
}

/// Wraps [`ChatApp`] with the cross-thread bridge. Drains the std-side
/// staging buffer into the App's [`Inbound`] queue at the start of each
/// frame, before delegating the build/event hooks to the inner App.
struct DesktopApp {
    inner: ChatApp,
    inbound_sink: Inbound,
    inbound_staging: Arc<Mutex<Vec<InboundEvent>>>,
    _rt: Arc<tokio::runtime::Runtime>,
}

impl App for DesktopApp {
    fn before_build(&mut self) {
        if let Ok(mut staging) = self.inbound_staging.lock()
            && !staging.is_empty()
        {
            let drained = std::mem::take(&mut *staging);
            let mut sink = self.inbound_sink.borrow_mut();
            for ev in drained {
                sink.push_back(ev);
            }
        }
        self.inner.before_build();
    }

    fn build(&self, cx: &BuildCx) -> El {
        self.inner.build(cx)
    }

    fn on_event(&mut self, event: UiEvent) {
        self.inner.on_event(event);
    }

    fn selection(&self) -> Selection {
        self.inner.selection()
    }

    fn theme(&self) -> Theme {
        self.inner.theme()
    }
}

async fn ws_loop(
    url: String,
    token: Option<String>,
    mut outbound_rx: mpsc::UnboundedReceiver<ClientToServer>,
    inbound_tx: mpsc::UnboundedSender<InboundEvent>,
) -> Result<()> {
    let mut req = url
        .as_str()
        .into_client_request()
        .context("parse ws URL into client request")?;
    if let Some(tok) = token {
        let value = HeaderValue::from_str(&format!("Bearer {tok}"))
            .context("build Authorization header")?;
        req.headers_mut().insert(AUTHORIZATION, value);
    }

    let (ws, _resp) = tokio_tungstenite::connect_async(req)
        .await
        .context("ws upgrade")?;
    let (mut sink, mut stream) = ws.split();

    let _ = inbound_tx.send(InboundEvent::ConnectionOpened);

    loop {
        tokio::select! {
            out = outbound_rx.recv() => match out {
                Some(msg) => {
                    let bytes = match encode_to_server(&msg) {
                        Ok(b) => b,
                        Err(e) => {
                            error!("encode_to_server failed: {e}");
                            continue;
                        }
                    };
                    if let Err(e) = sink.send(Message::Binary(bytes.into())).await {
                        warn!("ws send failed: {e}");
                        break;
                    }
                }
                None => break,
            },
            frame = stream.next() => match frame {
                Some(Ok(Message::Binary(bytes))) => match decode_from_server(&bytes) {
                    Ok(ev) => {
                        let _ = inbound_tx.send(InboundEvent::Wire(ev));
                    }
                    Err(e) => error!("decode_from_server failed: {e}"),
                },
                Some(Ok(Message::Close(reason))) => {
                    let detail = reason
                        .map(|c| format!("code {} {}", c.code, c.reason))
                        .unwrap_or_else(|| "peer closed".into());
                    let _ = inbound_tx.send(InboundEvent::ConnectionClosed { detail });
                    break;
                }
                Some(Ok(_)) => {}
                Some(Err(e)) => {
                    let _ = inbound_tx.send(InboundEvent::ConnectionError {
                        detail: e.to_string(),
                    });
                    break;
                }
                None => {
                    let _ = inbound_tx.send(InboundEvent::ConnectionClosed {
                        detail: "stream ended".into(),
                    });
                    break;
                }
            },
        }
    }
    Ok(())
}

fn resolve_cli_token(args: &Args) -> Result<Option<String>> {
    if let Some(tok) = &args.token {
        return Ok(Some(tok.clone()));
    }
    if let Some(path) = &args.token_file {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read token file {}", path.display()))?;
        return Ok(Some(raw.trim().to_string()));
    }
    Ok(None)
}

fn derive_ws_url(server: &str) -> Result<String> {
    let parsed = url::Url::parse(server).context("parse --server as URL")?;
    let ws_scheme = match parsed.scheme() {
        "http" | "ws" => "ws",
        "https" | "wss" => "wss",
        other => return Err(anyhow!("unsupported scheme `{other}` in --server")),
    };
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("--server has no host"))?;
    let mut out = format!("{ws_scheme}://{host}");
    if let Some(port) = parsed.port() {
        out.push(':');
        out.push_str(&port.to_string());
    }
    out.push_str("/ws");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::derive_ws_url;

    #[test]
    fn http_maps_to_ws() {
        assert_eq!(
            derive_ws_url("http://127.0.0.1:8080").unwrap(),
            "ws://127.0.0.1:8080/ws"
        );
    }

    #[test]
    fn https_maps_to_wss() {
        assert_eq!(
            derive_ws_url("https://example.com").unwrap(),
            "wss://example.com/ws"
        );
    }

    #[test]
    fn unsupported_scheme_errors() {
        assert!(derive_ws_url("ftp://whatever").is_err());
    }
}
