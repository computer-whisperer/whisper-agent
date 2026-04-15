//! whisper-agent-webui — browser chat UI.
//!
//! `app.rs` is platform-agnostic egui rendering. The wasm-only `web_entry` module below
//! opens the WebSocket, wires the inbound message queue and the outbound send closure,
//! and hands both to the [`ChatApp`].

mod app;

pub use app::ChatApp;

#[cfg(target_arch = "wasm32")]
mod web_entry {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use web_sys::{BinaryType, CloseEvent, ErrorEvent, MessageEvent, WebSocket};
    use whisper_agent_protocol::{
        ClientToServer, ServerToClient, SessionState, decode_from_server, encode_to_server,
    };

    use super::app::{ChatApp, Inbound, SendFn};

    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
        eframe::WebLogger::init(log::LevelFilter::Info).ok();

        let web_options = eframe::WebOptions::default();

        wasm_bindgen_futures::spawn_local(async move {
            let document = web_sys::window()
                .expect("no window")
                .document()
                .expect("no document");
            let canvas = document
                .get_element_by_id("the_canvas_id")
                .expect("missing #the_canvas_id")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("#the_canvas_id is not a canvas");

            let result = eframe::WebRunner::new()
                .start(
                    canvas,
                    web_options,
                    Box::new(|cc| {
                        let (inbound, send_fn) = open_websocket(cc.egui_ctx.clone());
                        Ok(Box::new(ChatApp::new(inbound, send_fn)))
                    }),
                )
                .await;

            if let Err(e) = result {
                log::error!("eframe start failed: {e:?}");
            }
        });
    }

    fn open_websocket(egui_ctx: egui::Context) -> (Inbound, SendFn) {
        let inbound: Inbound = Rc::new(RefCell::new(VecDeque::new()));

        let url = ws_url();
        log::info!("connecting to {url}");
        let socket = WebSocket::new(&url).expect("WebSocket::new");
        socket.set_binary_type(BinaryType::Arraybuffer);

        // onopen → push a Status::Connected event so the UI shows the right state.
        {
            let inbound_for_open = inbound.clone();
            let ctx_for_open = egui_ctx.clone();
            let cb = Closure::wrap(Box::new(move |_: JsValue| {
                inbound_for_open.borrow_mut().push_back(ServerToClient::Status {
                    state: SessionState::Connected,
                    detail: None,
                });
                ctx_for_open.request_repaint();
            }) as Box<dyn FnMut(JsValue)>);
            socket.set_onopen(Some(cb.as_ref().unchecked_ref()));
            cb.forget();
        }

        // onmessage → decode CBOR and push to inbound queue.
        {
            let inbound_for_msg = inbound.clone();
            let ctx_for_msg = egui_ctx.clone();
            let cb = Closure::wrap(Box::new(move |e: MessageEvent| {
                let data = e.data();
                if let Ok(buf) = data.dyn_into::<js_sys::ArrayBuffer>() {
                    let array = js_sys::Uint8Array::new(&buf);
                    let bytes = array.to_vec();
                    match decode_from_server(&bytes) {
                        Ok(event) => {
                            inbound_for_msg.borrow_mut().push_back(event);
                            ctx_for_msg.request_repaint();
                        }
                        Err(err) => log::error!("decode_from_server failed: {err}"),
                    }
                } else {
                    log::warn!("received non-binary ws frame; ignoring");
                }
            }) as Box<dyn FnMut(MessageEvent)>);
            socket.set_onmessage(Some(cb.as_ref().unchecked_ref()));
            cb.forget();
        }

        // onerror → surface as an error event in the UI.
        {
            let inbound_for_err = inbound.clone();
            let ctx_for_err = egui_ctx.clone();
            let cb = Closure::wrap(Box::new(move |e: ErrorEvent| {
                log::error!("ws error: {}", e.message());
                inbound_for_err.borrow_mut().push_back(ServerToClient::Error {
                    message: format!("websocket error: {}", e.message()),
                });
                ctx_for_err.request_repaint();
            }) as Box<dyn FnMut(ErrorEvent)>);
            socket.set_onerror(Some(cb.as_ref().unchecked_ref()));
            cb.forget();
        }

        // onclose → flip status to Closed (mapped via Error variant since SessionState
        // doesn't have a Closed variant; we fake it by emitting Error and a system note).
        {
            let inbound_for_close = inbound.clone();
            let ctx_for_close = egui_ctx.clone();
            let cb = Closure::wrap(Box::new(move |e: CloseEvent| {
                log::warn!("ws closed: code={} reason={}", e.code(), e.reason());
                inbound_for_close.borrow_mut().push_back(ServerToClient::Error {
                    message: format!("websocket closed (code {})", e.code()),
                });
                ctx_for_close.request_repaint();
            }) as Box<dyn FnMut(CloseEvent)>);
            socket.set_onclose(Some(cb.as_ref().unchecked_ref()));
            cb.forget();
        }

        let socket_for_send = socket.clone();
        let send_fn: SendFn = Box::new(move |msg: ClientToServer| {
            match encode_to_server(&msg) {
                Ok(bytes) => {
                    if let Err(e) = socket_for_send.send_with_u8_array(&bytes) {
                        log::error!("ws send failed: {e:?}");
                    }
                }
                Err(e) => log::error!("encode_to_server failed: {e}"),
            }
        });

        (inbound, send_fn)
    }

    fn ws_url() -> String {
        let location = web_sys::window().expect("no window").location();
        let host = location.host().unwrap_or_else(|_| "127.0.0.1:8080".into());
        let proto = if location
            .protocol()
            .map(|p| p == "https:")
            .unwrap_or(false)
        {
            "wss"
        } else {
            "ws"
        };
        format!("{proto}://{host}/ws")
    }
}
