//! whisper-agent-desktop — native window around the shared egui UI.
//!
//! The UI layer (`ChatApp`) lives in `whisper-agent-webui` as a portable
//! library; this binary supplies the WebSocket transport (via
//! `tokio-tungstenite`) and the eframe window. The wire protocol is
//! unchanged from the browser webui — same CBOR frames over the same
//! `/ws` endpoint.
//!
//! Two phases:
//!   - `Login`: egui form collects server URL + token. Submission
//!     writes to `$XDG_CONFIG_HOME/whisper-agent/desktop.toml` (if
//!     "Remember" is checked) and transitions to `Connected`.
//!   - `Connected`: spawns the ws task and wraps `ChatApp` in a
//!     `DesktopApp` that drains the inbound tokio channel each frame.
//!
//! CLI flags still override config. If both `--server` and a token
//! source are supplied (or loaded from config), the login screen is
//! skipped.
//!
//! Threading:
//!   - main thread: eframe window + render
//!   - tokio pool: `ws_loop` reads/writes frames
//!   - bridge: two `tokio::sync::mpsc` unbounded channels, plus a
//!     `OnceLock<egui::Context>` the WS task uses to `request_repaint`
//!     when a frame arrives so the UI wakes without polling.

mod config;
mod login;

use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};
use std::{cell::RefCell, path::PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use eframe::App as _;
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::{
    client::IntoClientRequest,
    http::header::{AUTHORIZATION, HeaderValue},
    protocol::Message,
};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use whisper_agent_protocol::{ClientToServer, decode_from_server, encode_to_server};
use whisper_agent_webui::{ChatApp, Inbound, InboundEvent, SendFn};

use crate::login::{LoginForm, Submission};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "whisper-agent-desktop: native window around the whisper-agent webui"
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
                .unwrap_or_else(|_| EnvFilter::new("info,whisper_agent_desktop=info")),
        )
        .init();

    // Pin rustls's process-level CryptoProvider before any TLS code
    // runs. tokio-tungstenite's `rustls-tls-native-roots` pulls rustls
    // in with both `ring` and `aws-lc-rs` reachable from transitive
    // deps, so rustls refuses to auto-pick and panics on first TLS
    // use. `.ok()` swallows the "already installed" error in case some
    // library beat us to it.
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

    // The WS task needs a handle to the egui context so it can wake the
    // UI when a frame arrives. `run_native` hands us the context inside
    // the app-creator closure; we drop it into this OnceLock from there.
    let ctx_holder: Arc<OnceLock<egui::Context>> = Arc::new(OnceLock::new());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("whisper-agent")
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    let rt_for_creator = rt.clone();
    let ctx_for_creator = ctx_holder.clone();
    eframe::run_native(
        "whisper-agent",
        options,
        Box::new(move |cc| {
            let _ = ctx_for_creator.set(cc.egui_ctx.clone());
            cc.egui_ctx.set_zoom_factor(1.1);

            let mut root = RootApp::new(rt_for_creator, ctx_for_creator);
            // Skip the login form when both values are known AND the
            // user didn't pass `--login`. Any other combination lands
            // on the form, prefilled with whatever we do have.
            match (args.login, initial_server, initial_token) {
                (false, Some(server), Some(token)) => root.start_connection(server, token),
                (_, server, token) => {
                    root.phase = Phase::Login(LoginForm::new(server, token));
                }
            }

            Ok(Box::new(root) as Box<dyn eframe::App>)
        }),
    )
    .map_err(|e| anyhow!("eframe: {e}"))?;

    Ok(())
}

fn non_empty(s: Option<String>) -> Option<String> {
    s.filter(|v| !v.trim().is_empty())
}

/// Top-level eframe app: owns the tokio runtime and the egui context
/// handle, and switches between the login form and the connected chat
/// UI. Only one ws task runs at a time (in the Connected phase).
struct RootApp {
    rt: Arc<tokio::runtime::Runtime>,
    ctx_holder: Arc<OnceLock<egui::Context>>,
    phase: Phase,
}

enum Phase {
    Login(LoginForm),
    // ChatApp is ~4KB and LoginForm is ~80B; box the large arm to
    // keep the enum itself small (clippy::large_enum_variant).
    Connected(Box<DesktopApp>),
}

impl RootApp {
    fn new(rt: Arc<tokio::runtime::Runtime>, ctx_holder: Arc<OnceLock<egui::Context>>) -> Self {
        // `phase` is overwritten by the caller before run_native actually
        // runs a frame, so this placeholder never renders.
        Self {
            rt,
            ctx_holder,
            phase: Phase::Login(LoginForm::new(None, None)),
        }
    }

    /// Spawn the ws task for `(server, token)` and transition to the
    /// connected phase. `token` may be empty — we only attach the
    /// Authorization header when it's non-empty.
    fn start_connection(&mut self, server: String, token: String) {
        let ws_url = match derive_ws_url(&server) {
            Ok(u) => u,
            Err(e) => {
                let mut form = LoginForm::new(Some(server), non_empty(Some(token)));
                form.set_error(format!("invalid server URL: {e:#}"));
                self.phase = Phase::Login(form);
                return;
            }
        };
        info!(url = %ws_url, "connecting");

        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel::<InboundEvent>();
        let (outbound_tx, outbound_rx) = mpsc::unbounded_channel::<ClientToServer>();

        let ctx_for_task = self.ctx_holder.clone();
        let token_for_task = if token.is_empty() { None } else { Some(token) };
        self.rt.spawn(async move {
            match ws_loop(
                ws_url,
                token_for_task,
                outbound_rx,
                inbound_tx,
                ctx_for_task,
            )
            .await
            {
                Ok(()) => info!("ws loop exited"),
                Err(e) => error!("ws loop error: {e:#}"),
            }
        });

        // Shared Rc<RefCell<VecDeque>> the WS-bridged events land in. Both
        // `DesktopApp` (pushes from the tokio channel) and `ChatApp` (drains
        // during render) hold clones.
        let inbound_shared: Inbound = Rc::new(RefCell::new(VecDeque::new()));
        let inbound_for_app = inbound_shared.clone();
        let inbound_for_desktop = inbound_shared;

        let send_fn: SendFn = Box::new(move |msg: ClientToServer| {
            if let Err(e) = outbound_tx.send(msg) {
                warn!("outbound channel closed: {e}");
            }
        });

        let chat_app = ChatApp::new(inbound_for_app, send_fn);
        self.phase = Phase::Connected(Box::new(DesktopApp {
            inner: chat_app,
            inbound_rx,
            inbound_sink: inbound_for_desktop,
        }));
    }
}

impl eframe::App for RootApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let submission = match &mut self.phase {
            Phase::Login(form) => form.show(ui),
            Phase::Connected(app) => {
                app.ui(ui, frame);
                None
            }
        };
        if let Some(Submission {
            server,
            token,
            remember,
        }) = submission
        {
            if remember {
                let cfg = config::Config {
                    server: Some(server.clone()),
                    token: (!token.is_empty()).then(|| token.clone()),
                };
                if let Err(e) = config::save(&cfg) {
                    warn!("could not save config: {e:#}");
                } else if let Ok(p) = config::path() {
                    info!("saved login to {}", p.display());
                }
            }
            self.start_connection(server, token);
        }
    }
}

/// Wraps `ChatApp` with the cross-thread bridge: drains the tokio channel
/// into the shared `Inbound` queue before delegating each frame to the
/// portable UI code.
struct DesktopApp {
    inner: ChatApp,
    inbound_rx: mpsc::UnboundedReceiver<InboundEvent>,
    inbound_sink: Inbound,
}

impl DesktopApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        while let Ok(event) = self.inbound_rx.try_recv() {
            self.inbound_sink.borrow_mut().push_back(event);
        }
        self.inner.ui(ui, frame);
    }
}

async fn ws_loop(
    url: String,
    token: Option<String>,
    mut outbound_rx: mpsc::UnboundedReceiver<ClientToServer>,
    inbound_tx: mpsc::UnboundedSender<InboundEvent>,
    ctx: Arc<OnceLock<egui::Context>>,
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
    request_repaint(&ctx);

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
                        request_repaint(&ctx);
                    }
                    Err(e) => error!("decode_from_server failed: {e}"),
                },
                Some(Ok(Message::Close(reason))) => {
                    let detail = reason
                        .map(|c| format!("code {} {}", c.code, c.reason))
                        .unwrap_or_else(|| "peer closed".into());
                    let _ = inbound_tx.send(InboundEvent::ConnectionClosed { detail });
                    request_repaint(&ctx);
                    break;
                }
                Some(Ok(_)) => {}
                Some(Err(e)) => {
                    let _ = inbound_tx.send(InboundEvent::ConnectionError {
                        detail: e.to_string(),
                    });
                    request_repaint(&ctx);
                    break;
                }
                None => {
                    let _ = inbound_tx.send(InboundEvent::ConnectionClosed {
                        detail: "stream ended".into(),
                    });
                    request_repaint(&ctx);
                    break;
                }
            },
        }
    }
    Ok(())
}

fn request_repaint(ctx: &Arc<OnceLock<egui::Context>>) {
    if let Some(c) = ctx.get() {
        c.request_repaint();
    }
}

/// Load the bearer token from whichever CLI source the user picked.
/// Returns `None` when neither `--token` nor `--token-file` was given;
/// the caller then falls back to saved config.
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

/// Turn a user-friendly `--server http://host:port` into the matching
/// `ws://host:port/ws` (or `wss://` for `https://`). Rejects anything
/// else so a bad flag fails fast instead of producing an opaque connect
/// error later.
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
    fn ws_scheme_passes_through() {
        assert_eq!(
            derive_ws_url("ws://localhost:1234").unwrap(),
            "ws://localhost:1234/ws"
        );
    }

    #[test]
    fn unsupported_scheme_errors() {
        assert!(derive_ws_url("ftp://whatever").is_err());
    }

    #[test]
    fn missing_scheme_errors() {
        assert!(derive_ws_url("127.0.0.1:8080").is_err());
    }
}
