//! whisper-agent-desktop-aetna — native window around the aetna chat UI.
//!
//! Sibling of `whisper-agent-desktop` (egui). Same WebSocket transport
//! (tokio + tungstenite, CBOR over `/ws`); different render stack
//! (`aetna-winit-wgpu` instead of eframe).
//!
//! Stage 1 surface: `--server` + `--token` are required (no in-app
//! login form yet — that arrives once the aetna text-input plumbing
//! lands). Outbound messages are still wired through so the wire path
//! exists end-to-end; only the rendered UI is the connection-status
//! banner.
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

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aetna_core::{App, BuildCx, El, Rect, Theme, UiEvent};
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
use whisper_agent_aetna_ui::{ChatApp, InboundEvent, SendFn};
use whisper_agent_protocol::{ClientToServer, decode_from_server, encode_to_server};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "whisper-agent-desktop-aetna: native window around the whisper-agent aetna ui (stage 1 scaffold)"
)]
struct Args {
    /// Base URL of a running whisper-agent server, e.g. `https://host:8443`
    /// or `http://127.0.0.1:8080`. The client appends `/ws` to derive the
    /// websocket endpoint, swapping the scheme to `ws`/`wss` as needed.
    #[arg(long)]
    server: String,

    /// Bearer token for the server's `[[auth.clients]]` gate. Required for
    /// non-loopback servers; loopback servers accept any value (or none).
    #[arg(long, env = "WA_TOKEN", conflicts_with = "token_file")]
    token: Option<String>,

    /// Read the bearer token from a single-line file — keeps the token
    /// off shell history and out of `ps` listings.
    #[arg(long)]
    token_file: Option<PathBuf>,
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
    let token = resolve_token(&args)?;

    let ws_url = derive_ws_url(&args.server)?;
    info!(url = %ws_url, "connecting");

    // Build the cross-thread bridge. The WS task is async (tokio); the
    // UI is sync (winit). The tokio mpsc carries inbound events into a
    // thread-safe staging buffer; `App::before_build` drains it into
    // the per-frame `Inbound` (Rc<RefCell>) the App reads from.
    let (inbound_tx, inbound_rx) = mpsc::unbounded_channel::<InboundEvent>();
    let (outbound_tx, outbound_rx) = mpsc::unbounded_channel::<ClientToServer>();

    // The std_mpsc::Sender side is what the UI thread polls; tokio's
    // mpsc::Receiver lives on the WS-bridge bouncer thread (below).
    let inbound_staging: Arc<Mutex<Vec<InboundEvent>>> = Arc::new(Mutex::new(Vec::new()));

    // Tokio runtime owns the WS task. Held in an `Arc` only so we can
    // hand a clone into the bouncer thread; `run_aetna_app` blocks the
    // main thread until the window closes, after which `_rt` drops and
    // the WS task is canceled.
    let rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("build tokio runtime")?,
    );

    let token_for_task = token.filter(|t| !t.is_empty());
    let _ws_handle = rt.spawn(async move {
        match ws_loop(ws_url, token_for_task, outbound_rx, inbound_tx).await {
            Ok(()) => info!("ws loop exited"),
            Err(e) => error!("ws loop error: {e:#}"),
        }
    });

    // Bouncer: drain the tokio mpsc and push into the std-side staging
    // buffer the UI thread reads. The UI thread can't poll a tokio
    // `mpsc::Receiver` without a tokio context, so this runs on the
    // tokio runtime and bridges the two worlds through a `Mutex<Vec>`.
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

    let inbound_shared = whisper_agent_aetna_ui::Inbound::default();
    // Shared with the App for its per-frame drain.
    let inbound_for_app = inbound_shared.clone();
    let inner = ChatApp::new(inbound_for_app, send_fn);

    // Keep the rt alive for the duration of the UI thread by capturing
    // it in the App wrapper. Dropping `rt` cancels the WS task.
    let app = DesktopApp {
        inner,
        inbound_sink: inbound_shared,
        inbound_staging,
        _rt: rt,
    };

    let viewport = Rect::new(0.0, 0.0, 1200.0, 800.0);
    // 30Hz cadence so WS frames get drained promptly without us
    // depending on user input to redraw. See the wakeup-caveat note at
    // the top of the file — this trades idle CPU for visibility into
    // background-arrived state.
    let host_config =
        aetna_winit_wgpu::HostConfig::default().with_redraw_interval(Duration::from_millis(33));
    aetna_winit_wgpu::run_with_config("whisper-agent", viewport, app, host_config)
        .map_err(|e| anyhow!("aetna runner: {e}"))?;
    Ok(())
}

/// Wraps [`ChatApp`] with the cross-thread bridge. Drains the std-side
/// staging buffer into the App's [`Inbound`] queue at the start of each
/// frame, before delegating the build/event hooks to the inner App.
struct DesktopApp {
    inner: ChatApp,
    inbound_sink: whisper_agent_aetna_ui::Inbound,
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

fn resolve_token(args: &Args) -> Result<Option<String>> {
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
