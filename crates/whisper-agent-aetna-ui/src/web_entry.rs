//! Wasm browser entry for the aetna chat UI.
//!
//! The `#[wasm_bindgen(start)]` function below runs on bundle load. The
//! actual wgpu+winit host shell (surface bring-up, ResizeObserver,
//! frame loop, keydown preventDefault for `/` / `'` / etc., text-paste
//! routing to focused inputs, sRGB view-format fixups) lives in
//! [`aetna_web`]; we just call [`aetna_web::start_with_config`] with
//! our [`ChatApp`] and hold onto the returned [`WebHandle`] so the
//! whisper-agent-specific browser glue can wake the renderer after
//! pushing work into ChatApp-owned queues.
//!
//! What this module still owns:
//! - WebSocket plumbing to `/ws`, decoding inbound binary frames into
//!   [`InboundEvent::Wire`], encoding outbound [`ClientToServer`].
//! - Document-level drag-and-drop file handling — dropped files land
//!   in the compose area's [`AttachmentIngress`].
//! - A capture-phase paste listener that intercepts *image* clipboard
//!   items before the upstream text-only listener runs. Text pastes
//!   continue through aetna-web's built-in path (which forwards to
//!   the focused Aetna text input).
//!
//! Auth: the static `assets/index.html` probes `/auth/check` before it
//! imports this bundle and bounces to `login.html` on 401. Once the
//! session cookie is good, the wasm bundle loads with the cookie
//! attached implicitly on the WS upgrade; we don't surface a login
//! form in-canvas the way the native client does.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use aetna_core::Rect;
use aetna_web::{WebHandle, WebHostConfig, start_with_config};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{BinaryType, CloseEvent, ErrorEvent, MessageEvent, WebSocket};
use whisper_agent_protocol::{ClientToServer, decode_from_server, encode_to_server};

use crate::app::{AttachmentIngress, ChatApp, Inbound, InboundEvent, SendFn};

/// Logical viewport supplied to `WebHostConfig`. aetna-web's
/// `ResizeObserver` reads the canvas's CSS box on first observe and
/// reconfigures the surface to match, so this is effectively just a
/// pre-layout fallback for the brief window before the first observer
/// fire — but it should still be a reasonable chat-shaped rectangle so
/// any initial paint isn't a 900×640 showcase-demo letterbox.
const VIEWPORT: Rect = Rect {
    x: 0.0,
    y: 0.0,
    w: 1200.0,
    h: 800.0,
};
/// The HTML `id` of the canvas the bundled `index.html` ships. aetna-web
/// defaults to `aetna_canvas`; we override via `with_canvas_id` so the
/// existing host page keeps working.
const CANVAS_ID: &str = "the_canvas_id";

#[wasm_bindgen(start)]
pub fn start() {
    // Open the WebSocket eagerly so its connect handshake races aetna-web's
    // async wgpu adapter request. ChatApp needs the (inbound, send_fn) pair
    // at construction time; the inbound queue is shared with the WS callbacks
    // we wire up below.
    let inbound: Inbound = Rc::new(RefCell::new(VecDeque::new()));
    let socket = open_socket();
    let send_fn = build_send_fn(socket.clone());

    let app = ChatApp::new(inbound.clone(), send_fn);
    let ingress = app.attachment_ingress();

    // Hand the app to aetna-web. `start_with_config` schedules the
    // event loop on browser microtasks and returns immediately with a
    // `WebHandle`. The handle's `request_redraw` is safe to call before
    // the surface has finished its async setup — it buffers and flushes
    // on first ready frame.
    let config = WebHostConfig::new(VIEWPORT).with_canvas_id(CANVAS_ID);
    let handle = start_with_config(config, app);

    // Now wire the whisper-agent-specific browser glue. Each listener
    // owns a clone of the handle so it can wake the renderer after
    // pushing work into ChatApp-owned shared state, without having to
    // wait for unrelated user input to trigger a frame.
    wire_socket_handlers(&socket, inbound, handle.clone());
    install_drop_handlers(ingress.clone(), handle.clone());
    install_image_paste_handler(ingress, handle);
}

// ============================================================
// WebSocket plumbing
// ============================================================

fn open_socket() -> WebSocket {
    let url = ws_url();
    log::info!("connecting to {url}");
    let socket = WebSocket::new(&url).expect("WebSocket::new");
    socket.set_binary_type(BinaryType::Arraybuffer);
    socket
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

fn build_send_fn(socket: WebSocket) -> SendFn {
    Box::new(move |msg: ClientToServer| match encode_to_server(&msg) {
        Ok(bytes) => {
            if let Err(e) = socket.send_with_u8_array(&bytes) {
                log::error!("ws send failed: {e:?}");
            }
        }
        Err(e) => log::error!("encode_to_server failed: {e}"),
    })
}

fn wire_socket_handlers(socket: &WebSocket, inbound: Inbound, handle: WebHandle) {
    {
        let inbound = inbound.clone();
        let handle = handle.clone();
        let cb = Closure::wrap(Box::new(move |_: JsValue| {
            inbound
                .borrow_mut()
                .push_back(InboundEvent::ConnectionOpened);
            handle.request_redraw();
        }) as Box<dyn FnMut(JsValue)>);
        socket.set_onopen(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
    }

    {
        let inbound = inbound.clone();
        let handle = handle.clone();
        let cb = Closure::wrap(Box::new(move |e: MessageEvent| {
            let data = e.data();
            if let Ok(buf) = data.dyn_into::<js_sys::ArrayBuffer>() {
                let array = js_sys::Uint8Array::new(&buf);
                let bytes = array.to_vec();
                match decode_from_server(&bytes) {
                    Ok(event) => {
                        inbound.borrow_mut().push_back(InboundEvent::Wire(event));
                        handle.request_redraw();
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

    {
        let inbound = inbound.clone();
        let handle = handle.clone();
        let cb = Closure::wrap(Box::new(move |e: ErrorEvent| {
            log::error!("ws error: {}", e.message());
            inbound
                .borrow_mut()
                .push_back(InboundEvent::ConnectionError {
                    detail: e.message(),
                });
            handle.request_redraw();
        }) as Box<dyn FnMut(ErrorEvent)>);
        socket.set_onerror(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
    }

    {
        let cb = Closure::wrap(Box::new(move |e: CloseEvent| {
            log::warn!("ws closed: code={} reason={}", e.code(), e.reason());
            inbound
                .borrow_mut()
                .push_back(InboundEvent::ConnectionClosed {
                    detail: format!("code {}", e.code()),
                });
            handle.request_redraw();
        }) as Box<dyn FnMut(CloseEvent)>);
        socket.set_onclose(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
    }
}

// ============================================================
// Drag-drop and image-paste listeners
// ============================================================

/// Install document-body drag-drop handlers so dropped files land in
/// the compose area's attachment staging queue. `dragover` must
/// `preventDefault` (otherwise the browser refuses the drop with a
/// not-allowed cursor); `drop` must `preventDefault` to stop the
/// browser from navigating away to display the dropped file.
fn install_drop_handlers(ingress: AttachmentIngress, handle: WebHandle) {
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
            let handle = handle.clone();
            let promise = file.array_buffer();
            wasm_bindgen_futures::spawn_local(async move {
                match wasm_bindgen_futures::JsFuture::from(promise).await {
                    Ok(buffer) => {
                        let array = js_sys::Uint8Array::new(&buffer);
                        ingress.push(array.to_vec(), name);
                        handle.request_redraw();
                    }
                    Err(err) => {
                        log::warn!("failed to read dropped file {name}: {err:?}");
                    }
                }
            });
        }
    }) as Box<dyn FnMut(web_sys::DragEvent)>);
    if let Err(e) = body.add_event_listener_with_callback("drop", drop_cb.as_ref().unchecked_ref())
    {
        log::warn!("failed to install drop listener: {e:?}");
    }
    drop_cb.forget();
}

/// Install a *capture-phase* paste listener that handles image
/// clipboard items only. aetna-web installs its own paste listener
/// (bubble phase) that always pulls `text/plain` and routes it to the
/// focused text input — but Chromium screenshots often expose both an
/// image item AND a text representation, and we want the image to
/// win. Registering in capture phase guarantees we run before any
/// bubble-phase listener regardless of registration order; if we
/// handle an image we `stopImmediatePropagation` so the upstream
/// text-paste path doesn't also fire for the same event. For
/// text-only clipboard payloads we do nothing — propagation continues
/// and aetna-web routes the text as usual.
fn install_image_paste_handler(ingress: AttachmentIngress, handle: WebHandle) {
    let Some(window) = web_sys::window() else {
        log::warn!("install_image_paste_handler: no window; skipping");
        return;
    };
    let Some(document) = window.document() else {
        log::warn!("install_image_paste_handler: no document; skipping");
        return;
    };

    let cb = Closure::wrap(Box::new(move |e: web_sys::ClipboardEvent| {
        let Some(dt) = e.clipboard_data() else {
            return;
        };
        let items = dt.items();
        let len = items.length();
        let mut handled_image = false;
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
            handled_image = true;
            let name = if file.name().is_empty() {
                format!("clipboard-image.{}", mime_to_ext(&mime))
            } else {
                file.name()
            };
            log::info!("paste: image kind={mime} name={name}");
            let promise = file.array_buffer();
            let ingress = ingress.clone();
            let handle = handle.clone();
            wasm_bindgen_futures::spawn_local(async move {
                match wasm_bindgen_futures::JsFuture::from(promise).await {
                    Ok(buffer) => {
                        let array = js_sys::Uint8Array::new(&buffer);
                        ingress.push(array.to_vec(), name);
                        handle.request_redraw();
                    }
                    Err(err) => {
                        log::warn!("paste: failed to read clipboard image: {err:?}");
                    }
                }
            });
        }

        if handled_image {
            // Eat the event so aetna-web's bubble-phase text listener
            // doesn't also fire for the same paste (which would inject
            // the text/plain alternative the OS attached alongside the
            // image, e.g. the file path on Windows).
            e.prevent_default();
            e.stop_immediate_propagation();
        }
        // Text-only payloads fall through to aetna-web's listener.
    }) as Box<dyn FnMut(web_sys::ClipboardEvent)>);

    // `add_event_listener_with_callback_and_bool` registers with
    // `useCapture: true`, so this fires during capture phase — before
    // any bubble-phase listener on the same target or its ancestors,
    // regardless of registration order.
    if let Err(e) = document.add_event_listener_with_callback_and_bool(
        "paste",
        cb.as_ref().unchecked_ref(),
        true,
    ) {
        log::warn!("failed to install image-paste capture listener: {e:?}");
    }
    cb.forget();
}

/// Best-effort MIME -> extension for naming pasted clipboard images
/// when Chromium hands us a blank `File.name`.
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
