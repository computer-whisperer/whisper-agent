//! whisper-agent-webui — browser chat UI.
//!
//! `app.rs` is platform-agnostic egui rendering. The wasm-only `web_entry` module below
//! opens the WebSocket, wires the inbound message queue and the outbound send closure,
//! and hands both to the [`ChatApp`].

mod app;
mod cron_preview;
mod editor;
#[cfg(target_arch = "wasm32")]
mod fonts;

pub use app::{AttachmentIngress, ChatApp, Inbound, InboundEvent, SendFn};

#[cfg(target_arch = "wasm32")]
mod web_entry {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;
    use web_sys::{BinaryType, CloseEvent, ErrorEvent, MessageEvent, WebSocket};
    use whisper_agent_protocol::{ClientToServer, decode_from_server, encode_to_server};

    use super::app::{AttachmentIngress, ChatApp, Inbound, InboundEvent, SendFn};

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
                        // Bump everything up — egui's defaults are tuned
                        // for native desktop, browser canvas reads small
                        // at 1.0. 1.5 lands closer to a comfortable
                        // reading size without pushing the layout into
                        // tablet mode.
                        cc.egui_ctx.set_zoom_factor(1.2);
                        super::fonts::install(&cc.egui_ctx);
                        // Wire the `bytes://` image loader so inline
                        // attachment thumbnails and conversation-history
                        // images decode without hitting the network.
                        egui_extras::install_image_loaders(&cc.egui_ctx);
                        let (inbound, send_fn) = open_websocket(cc.egui_ctx.clone());
                        let app = ChatApp::new(inbound, send_fn);
                        // eframe's web backend doesn't deliver drop
                        // events to egui's input queue on this build
                        // (empirical: `i.raw.dropped_files` stays
                        // empty through drops). Install our own JS
                        // listeners at document-body scope — they
                        // catch drops anywhere on the page and push
                        // through the same attachment-ingress the
                        // file-picker button uses.
                        let ingress = app.attachment_ingress(cc.egui_ctx.clone());
                        install_drop_handlers(ingress.clone());
                        install_paste_handler(ingress);
                        Ok(Box::new(app))
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

        // onopen → tell the app the connection is up.
        {
            let inbound_for_open = inbound.clone();
            let ctx_for_open = egui_ctx.clone();
            let cb = Closure::wrap(Box::new(move |_: JsValue| {
                inbound_for_open
                    .borrow_mut()
                    .push_back(InboundEvent::ConnectionOpened);
                ctx_for_open.request_repaint();
            }) as Box<dyn FnMut(JsValue)>);
            socket.set_onopen(Some(cb.as_ref().unchecked_ref()));
            cb.forget();
        }

        // onmessage → decode CBOR and push the wire event to the inbound queue.
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
                            inbound_for_msg
                                .borrow_mut()
                                .push_back(InboundEvent::Wire(event));
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

        // onerror → surface as a connection-error event.
        {
            let inbound_for_err = inbound.clone();
            let ctx_for_err = egui_ctx.clone();
            let cb = Closure::wrap(Box::new(move |e: ErrorEvent| {
                log::error!("ws error: {}", e.message());
                inbound_for_err
                    .borrow_mut()
                    .push_back(InboundEvent::ConnectionError {
                        detail: e.message(),
                    });
                ctx_for_err.request_repaint();
            }) as Box<dyn FnMut(ErrorEvent)>);
            socket.set_onerror(Some(cb.as_ref().unchecked_ref()));
            cb.forget();
        }

        // onclose → flip status.
        {
            let inbound_for_close = inbound.clone();
            let ctx_for_close = egui_ctx.clone();
            let cb = Closure::wrap(Box::new(move |e: CloseEvent| {
                log::warn!("ws closed: code={} reason={}", e.code(), e.reason());
                inbound_for_close
                    .borrow_mut()
                    .push_back(InboundEvent::ConnectionClosed {
                        detail: format!("code {}", e.code()),
                    });
                ctx_for_close.request_repaint();
            }) as Box<dyn FnMut(CloseEvent)>);
            socket.set_onclose(Some(cb.as_ref().unchecked_ref()));
            cb.forget();
        }

        let socket_for_send = socket.clone();
        let send_fn: SendFn = Box::new(move |msg: ClientToServer| match encode_to_server(&msg) {
            Ok(bytes) => {
                if let Err(e) = socket_for_send.send_with_u8_array(&bytes) {
                    log::error!("ws send failed: {e:?}");
                }
            }
            Err(e) => log::error!("encode_to_server failed: {e}"),
        });

        (inbound, send_fn)
    }

    /// Install document-level HTML5 drag-drop handlers so dropped
    /// files on the whisper-agent page land in the compose area's
    /// attachment queue. Runs once at app startup; the forgotten
    /// closures live for the lifetime of the page. Scoped to
    /// `document.body` (not the canvas) so the drop zone is the
    /// whole visible UI — users aim at their compose box but may
    /// release slightly off-target, and a body-level target forgives
    /// that.
    ///
    /// `dragover` `preventDefault` is required: without it the
    /// browser refuses the drop outright (cursor stays "not-allowed").
    /// `drop` `preventDefault` is required to stop the browser from
    /// navigating away to show the dropped image.
    fn install_drop_handlers(ingress: AttachmentIngress) {
        let Some(window) = web_sys::window() else {
            log::warn!("install_drop_handlers: no window; skipping");
            return;
        };
        let Some(document) = window.document() else {
            log::warn!("install_drop_handlers: no document; skipping");
            return;
        };
        let Some(body) = document.body() else {
            log::warn!("install_drop_handlers: no body; skipping");
            return;
        };

        let dragover_cb = Closure::wrap(Box::new(|e: web_sys::DragEvent| {
            e.prevent_default();
        }) as Box<dyn FnMut(web_sys::DragEvent)>);
        if let Err(e) =
            body.add_event_listener_with_callback("dragover", dragover_cb.as_ref().unchecked_ref())
        {
            log::warn!("failed to install dragover listener: {e:?}");
        }
        dragover_cb.forget();

        let drop_cb = Closure::wrap(Box::new(move |e: web_sys::DragEvent| {
            e.prevent_default();
            let Some(dt) = e.data_transfer() else {
                log::warn!("drop event had no DataTransfer");
                return;
            };
            let Some(files) = dt.files() else {
                log::warn!("drop DataTransfer carried no files");
                return;
            };
            let len = files.length();
            log::info!("drop: {len} file(s) received");
            for i in 0..len {
                let Some(file) = files.item(i) else {
                    continue;
                };
                let name = file.name();
                let ingress = ingress.clone();
                // File.arrayBuffer() returns a Promise<ArrayBuffer>;
                // wrap as a JsFuture so we can await it.
                let promise = file.array_buffer();
                wasm_bindgen_futures::spawn_local(async move {
                    match wasm_bindgen_futures::JsFuture::from(promise).await {
                        Ok(buffer) => {
                            let array = js_sys::Uint8Array::new(&buffer);
                            ingress.push(array.to_vec(), name);
                        }
                        Err(err) => {
                            log::warn!("failed to read dropped file {name}: {err:?}");
                        }
                    }
                });
            }
        }) as Box<dyn FnMut(web_sys::DragEvent)>);
        if let Err(e) =
            body.add_event_listener_with_callback("drop", drop_cb.as_ref().unchecked_ref())
        {
            log::warn!("failed to install drop listener: {e:?}");
        }
        drop_cb.forget();
    }

    /// Install a document-level `paste` listener that extracts
    /// image items from the clipboard and routes them through the
    /// same attachment ingress as the file picker / drag-drop. Mirrors
    /// the drag-drop takeover: egui's text-paste support sees the
    /// clipboard's text payload, but it doesn't surface file/image
    /// blobs, so we read those out of `ClipboardEvent.clipboardData`
    /// directly.
    ///
    /// Note: we *don't* `preventDefault` here. egui still wants the
    /// paste event for text typed into the compose area; suppressing
    /// it would break Ctrl+V on plain text. Browsers happily deliver
    /// the same event to multiple listeners.
    fn install_paste_handler(ingress: AttachmentIngress) {
        let Some(window) = web_sys::window() else {
            log::warn!("install_paste_handler: no window; skipping");
            return;
        };
        let Some(document) = window.document() else {
            log::warn!("install_paste_handler: no document; skipping");
            return;
        };
        let Some(body) = document.body() else {
            log::warn!("install_paste_handler: no body; skipping");
            return;
        };

        let cb = Closure::wrap(Box::new(move |e: web_sys::ClipboardEvent| {
            let Some(dt) = e.clipboard_data() else {
                return;
            };
            let items = dt.items();
            let len = items.length();
            for i in 0..len {
                let Some(item) = items.get(i) else { continue };
                if item.kind() != "file" {
                    continue;
                }
                let mime = item.type_();
                if !mime.starts_with("image/") {
                    continue;
                }
                let Ok(Some(file)) = item.get_as_file() else {
                    continue;
                };
                let name = if file.name().is_empty() {
                    format!("clipboard-image.{}", mime_to_ext(&mime))
                } else {
                    file.name()
                };
                log::info!("paste: image kind={mime} name={name}");
                let promise = file.array_buffer();
                let ingress = ingress.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    match wasm_bindgen_futures::JsFuture::from(promise).await {
                        Ok(buffer) => {
                            let array = js_sys::Uint8Array::new(&buffer);
                            ingress.push(array.to_vec(), name);
                        }
                        Err(err) => {
                            log::warn!("paste: failed to read clipboard image: {err:?}");
                        }
                    }
                });
            }
        }) as Box<dyn FnMut(web_sys::ClipboardEvent)>);
        if let Err(e) = body.add_event_listener_with_callback("paste", cb.as_ref().unchecked_ref())
        {
            log::warn!("failed to install paste listener: {e:?}");
        }
        cb.forget();
    }

    /// Best-effort MIME → extension lookup for naming pasted-from-
    /// clipboard images (Chromium hands us a blank `File.name`).
    /// Falls back to "bin" for anything we don't recognize, since the
    /// downstream MIME sniffer reads the bytes anyway.
    fn mime_to_ext(mime: &str) -> &'static str {
        match mime {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/gif" => "gif",
            "image/webp" => "webp",
            "image/heic" => "heic",
            "image/heif" => "heif",
            _ => "bin",
        }
    }

    fn ws_url() -> String {
        let location = web_sys::window().expect("no window").location();
        let host = location.host().unwrap_or_else(|_| "127.0.0.1:8080".into());
        let proto = if location.protocol().map(|p| p == "https:").unwrap_or(false) {
            "wss"
        } else {
            "ws"
        };
        format!("{proto}://{host}/ws")
    }
}
