//! Wasm browser entry for the aetna chat UI.
//!
//! The `#[wasm_bindgen(start)]` function below runs on bundle load,
//! opens a WebSocket against `/ws`, builds a [`ChatApp`], installs
//! document-level drag-drop + clipboard-paste listeners (winit
//! doesn't deliver these to the canvas reliably across browsers),
//! and hands the app to an aetna [`Host`] driving a `<canvas
//! id="the_canvas_id">` from the host HTML page.
//!
//! Auth: the static `assets/index.html` probes `/auth/check` before
//! it imports this bundle and bounces to `login.html` on 401. Once
//! the session cookie is good, the wasm bundle loads with the
//! cookie attached implicitly on the WS upgrade; we don't surface a
//! login form in-canvas the way the native client does.
//!
//! The bulk of this module is a lift of the wgpu+winit host shell
//! from `aetna-web/src/lib.rs` — aetna ships a reusable browser host
//! for the showcase demo but doesn't yet expose it as a public API
//! the way `aetna-winit-wgpu::run_with_config` does for native. The
//! eventual cleanup is to upstream `aetna-web::start_with::<A>` and
//! call into that; until then we carry the boilerplate here so the
//! pivot can land without coordinated cross-repo work.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

use aetna_core::{
    App, BuildCx, Cursor, FrameTrigger, HostDiagnostics, KeyModifiers, Palette, PointerButton,
    Rect, UiKey,
};
use aetna_wgpu::{MsaaTarget, PrepareTimings, Runner};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{BinaryType, CloseEvent, ErrorEvent, MessageEvent, WebSocket};
use web_time::Instant;
use whisper_agent_protocol::{ClientToServer, decode_from_server, encode_to_server};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::window::{CursorIcon, Window, WindowId};

use crate::app::{AttachmentIngress, ChatApp, Inbound, InboundEvent, SendFn};

/// Default logical viewport. The first ResizeObserver fire (which
/// arrives synchronously on `observe()`) immediately overrides this
/// with the canvas's actual CSS-laid-out size.
const VIEWPORT: Rect = Rect {
    x: 0.0,
    y: 0.0,
    w: 1200.0,
    h: 800.0,
};
const SAMPLE_COUNT: u32 = 4;
const CANVAS_ID: &str = "the_canvas_id";
const FRAME_LOG_INTERVAL: u32 = 60;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    // Shared gfx slot — host writes on async-setup completion; JS
    // callbacks read it to call `window.request_redraw()` after
    // pushing inbound events so the next frame drains the queue
    // without waiting for unrelated user input.
    let gfx_slot: Rc<RefCell<Option<Gfx>>> = Rc::new(RefCell::new(None));

    // Open the WS first so it's connecting in parallel with the
    // host's async adapter request.
    let (inbound, send_fn) = open_websocket(gfx_slot.clone());
    let app = ChatApp::new(inbound, send_fn);

    // Drag-drop + clipboard-paste land in the same staging queue the
    // native file picker uses (via `AttachmentIngress`). Wake the
    // canvas after each push so the staged thumbnail appears
    // immediately instead of on the next user input.
    let ingress = app.attachment_ingress();
    install_drop_handlers(ingress.clone(), gfx_slot.clone());
    install_paste_handler(ingress, gfx_slot.clone());

    let event_loop = EventLoop::new().expect("EventLoop::new");
    let host = Host::<ChatApp>::new(VIEWPORT, app, gfx_slot);
    // spawn_app hands control to the browser raf loop; native uses
    // run_app(...) which blocks the calling thread instead.
    event_loop.spawn_app(host);
}

// ============================================================
// WebSocket plumbing
// ============================================================

fn open_websocket(gfx_slot: Rc<RefCell<Option<Gfx>>>) -> (Inbound, SendFn) {
    let inbound: Inbound = Rc::new(RefCell::new(VecDeque::new()));

    let url = ws_url();
    log::info!("connecting to {url}");
    let socket = WebSocket::new(&url).expect("WebSocket::new");
    socket.set_binary_type(BinaryType::Arraybuffer);

    {
        let inbound = inbound.clone();
        let slot = gfx_slot.clone();
        let cb = Closure::wrap(Box::new(move |_: JsValue| {
            inbound
                .borrow_mut()
                .push_back(InboundEvent::ConnectionOpened);
            wake(&slot);
        }) as Box<dyn FnMut(JsValue)>);
        socket.set_onopen(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
    }

    {
        let inbound = inbound.clone();
        let slot = gfx_slot.clone();
        let cb = Closure::wrap(Box::new(move |e: MessageEvent| {
            let data = e.data();
            if let Ok(buf) = data.dyn_into::<js_sys::ArrayBuffer>() {
                let array = js_sys::Uint8Array::new(&buf);
                let bytes = array.to_vec();
                match decode_from_server(&bytes) {
                    Ok(event) => {
                        inbound.borrow_mut().push_back(InboundEvent::Wire(event));
                        wake(&slot);
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
        let slot = gfx_slot.clone();
        let cb = Closure::wrap(Box::new(move |e: ErrorEvent| {
            log::error!("ws error: {}", e.message());
            inbound
                .borrow_mut()
                .push_back(InboundEvent::ConnectionError {
                    detail: e.message(),
                });
            wake(&slot);
        }) as Box<dyn FnMut(ErrorEvent)>);
        socket.set_onerror(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
    }

    {
        let inbound = inbound.clone();
        let slot = gfx_slot.clone();
        let cb = Closure::wrap(Box::new(move |e: CloseEvent| {
            log::warn!("ws closed: code={} reason={}", e.code(), e.reason());
            inbound
                .borrow_mut()
                .push_back(InboundEvent::ConnectionClosed {
                    detail: format!("code {}", e.code()),
                });
            wake(&slot);
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

/// Request a redraw if the host has finished async setup. Pre-setup
/// inbound events queue normally and get drained by the first frame
/// the host produces from its own `request_redraw` after setup.
fn wake(slot: &Rc<RefCell<Option<Gfx>>>) {
    if let Some(gfx) = slot.borrow().as_ref() {
        gfx.window.request_redraw();
    }
}

// ============================================================
// Drag-drop and clipboard-paste listeners
// ============================================================

/// Install document-body drag-drop handlers so dropped files land in
/// the compose area's attachment staging queue. `dragover` must
/// `preventDefault` (otherwise the browser refuses the drop with a
/// not-allowed cursor); `drop` must `preventDefault` to stop the
/// browser from navigating away to display the dropped file.
fn install_drop_handlers(ingress: AttachmentIngress, gfx_slot: Rc<RefCell<Option<Gfx>>>) {
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
            let slot = gfx_slot.clone();
            let promise = file.array_buffer();
            wasm_bindgen_futures::spawn_local(async move {
                match wasm_bindgen_futures::JsFuture::from(promise).await {
                    Ok(buffer) => {
                        let array = js_sys::Uint8Array::new(&buffer);
                        ingress.push(array.to_vec(), name);
                        wake(&slot);
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

/// Install a document-level `paste` listener that extracts image
/// items from the clipboard. We don't `preventDefault` — egui's
/// sibling depended on letting text paste continue to the focused
/// input; aetna's text inputs read paste through their own handler
/// driven by `key_down` + clipboard, so the same restraint applies.
fn install_paste_handler(ingress: AttachmentIngress, gfx_slot: Rc<RefCell<Option<Gfx>>>) {
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
            let slot = gfx_slot.clone();
            wasm_bindgen_futures::spawn_local(async move {
                match wasm_bindgen_futures::JsFuture::from(promise).await {
                    Ok(buffer) => {
                        let array = js_sys::Uint8Array::new(&buffer);
                        ingress.push(array.to_vec(), name);
                        wake(&slot);
                    }
                    Err(err) => {
                        log::warn!("paste: failed to read clipboard image: {err:?}");
                    }
                }
            });
        }
    }) as Box<dyn FnMut(web_sys::ClipboardEvent)>);
    if let Err(e) = body.add_event_listener_with_callback("paste", cb.as_ref().unchecked_ref()) {
        log::warn!("failed to install paste listener: {e:?}");
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

// ============================================================
// Browser wgpu+winit host shell
// ----
// Lifted from `aetna-web/src/lib.rs`. Differences from upstream:
//   - The gfx slot is constructor-injected so JS callbacks outside
//     the Host (WebSocket + drag-drop) can call request_redraw on
//     the same window after pushing events.
//   - The viewport default is sized for a chat app (1200x800), not
//     the showcase demo (900x640).
// Everything else — adapter pick, sRGB view fixup, ResizeObserver
// wiring, two-lane redraw triggers — is the upstream behavior. See
// the aetna-web source for the rationale comments.
// ============================================================

#[derive(Default)]
struct FrameStats {
    build_us: u64,
    prepare_us: u64,
    submit_us: u64,
    inter_us: u64,
    layout_us: u64,
    draw_ops_us: u64,
    paint_us: u64,
    gpu_upload_us: u64,
    snapshot_us: u64,
    samples: u32,
    last_frame_start: Option<Instant>,
}

impl FrameStats {
    fn record(
        &mut self,
        frame_start: Instant,
        t1: Instant,
        t2: Instant,
        t3: Instant,
        prep: PrepareTimings,
    ) {
        self.build_us += (t1 - frame_start).as_micros() as u64;
        self.prepare_us += (t2 - t1).as_micros() as u64;
        self.submit_us += (t3 - t2).as_micros() as u64;
        self.layout_us += prep.layout.as_micros() as u64;
        self.draw_ops_us += prep.draw_ops.as_micros() as u64;
        self.paint_us += prep.paint.as_micros() as u64;
        self.gpu_upload_us += prep.gpu_upload.as_micros() as u64;
        self.snapshot_us += prep.snapshot.as_micros() as u64;
        if let Some(prev) = self.last_frame_start {
            self.inter_us += (frame_start - prev).as_micros() as u64;
        }
        self.last_frame_start = Some(frame_start);
        self.samples += 1;
        if self.samples >= FRAME_LOG_INTERVAL {
            self.flush();
        }
    }

    fn flush(&mut self) {
        let n = self.samples as u64;
        let inter_n = (self.samples.saturating_sub(1)) as u64;
        let build = self.build_us / n;
        let prepare = self.prepare_us / n;
        let submit = self.submit_us / n;
        let layout = self.layout_us / n;
        let draw_ops = self.draw_ops_us / n;
        let paint = self.paint_us / n;
        let gpu_upload = self.gpu_upload_us / n;
        let snapshot = self.snapshot_us / n;
        let cpu = build + prepare + submit;
        let inter = self.inter_us.checked_div(inter_n).unwrap_or(0);
        let util = (cpu * 100).checked_div(inter).unwrap_or(0);
        log::info!(
            "frame[{n}] inter={:.2}ms cpu={:.2}ms util={util}% | build={:.2} prepare={:.2} (layout={:.2} draw_ops={:.2} paint={:.2} gpu={:.2} snapshot={:.2}) submit={:.2}",
            inter as f64 / 1000.0,
            cpu as f64 / 1000.0,
            build as f64 / 1000.0,
            prepare as f64 / 1000.0,
            layout as f64 / 1000.0,
            draw_ops as f64 / 1000.0,
            paint as f64 / 1000.0,
            gpu_upload as f64 / 1000.0,
            snapshot as f64 / 1000.0,
            submit as f64 / 1000.0,
        );
        self.build_us = 0;
        self.prepare_us = 0;
        self.submit_us = 0;
        self.inter_us = 0;
        self.layout_us = 0;
        self.draw_ops_us = 0;
        self.paint_us = 0;
        self.gpu_upload_us = 0;
        self.snapshot_us = 0;
        self.samples = 0;
    }
}

fn locate_canvas() -> web_sys::HtmlCanvasElement {
    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");
    document
        .get_element_by_id(CANVAS_ID)
        .unwrap_or_else(|| panic!("missing #{CANVAS_ID} canvas element"))
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("#the_canvas_id is not a canvas")
}

fn measure_canvas(canvas: &web_sys::HtmlCanvasElement) -> (u32, u32) {
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0)
        .max(1.0);
    let css_w = canvas.client_width().max(1) as f64;
    let css_h = canvas.client_height().max(1) as f64;
    let phys_w = (css_w * dpr).round() as u32;
    let phys_h = (css_h * dpr).round() as u32;
    (phys_w, phys_h)
}

fn apply_canvas_size(canvas: &web_sys::HtmlCanvasElement, gfx: &mut Gfx, phys_w: u32, phys_h: u32) {
    canvas.set_width(phys_w);
    canvas.set_height(phys_h);
    if gfx.config.width == phys_w && gfx.config.height == phys_h {
        return;
    }
    gfx.config.width = phys_w;
    gfx.config.height = phys_h;
    gfx.surface.configure(&gfx.device, &gfx.config);
    gfx.renderer.set_surface_size(phys_w, phys_h);
    let extent = surface_extent(&gfx.config);
    if !gfx.msaa.matches(extent) {
        gfx.msaa = MsaaTarget::new(&gfx.device, gfx.render_format, extent, SAMPLE_COUNT);
    }
}

struct Host<A: App> {
    viewport: Rect,
    app: A,
    gfx: Rc<RefCell<Option<Gfx>>>,
    last_pointer: Option<(f32, f32)>,
    modifiers: KeyModifiers,
    stats: FrameStats,
    last_cursor: Cursor,
    next_trigger: FrameTrigger,
    last_frame_at: Option<Instant>,
    frame_index: u64,
    last_prepared_size: Option<(u32, u32)>,
    backend: Rc<RefCell<&'static str>>,
    _resize_closure: Option<Closure<dyn FnMut()>>,
    _resize_observer: Option<web_sys::ResizeObserver>,
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: Runner,
    msaa: MsaaTarget,
    render_format: wgpu::TextureFormat,
}

fn surface_extent(config: &wgpu::SurfaceConfiguration) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    }
}

impl<A: App> Host<A> {
    fn new(viewport: Rect, app: A, gfx: Rc<RefCell<Option<Gfx>>>) -> Self {
        Self {
            viewport,
            app,
            gfx,
            last_pointer: None,
            modifiers: KeyModifiers::default(),
            stats: FrameStats::default(),
            last_cursor: Cursor::Default,
            next_trigger: FrameTrigger::Initial,
            last_frame_at: None,
            frame_index: 0,
            last_prepared_size: None,
            backend: Rc::new(RefCell::new("?")),
            _resize_closure: None,
            _resize_observer: None,
        }
    }
}

fn backend_label(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "DX12",
        wgpu::Backend::Gl => "WebGL2",
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        wgpu::Backend::Noop => "noop",
    }
}

fn srgb_view_of(format: wgpu::TextureFormat) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    match format {
        F::Rgba8Unorm => Some(F::Rgba8UnormSrgb),
        F::Bgra8Unorm => Some(F::Bgra8UnormSrgb),
        _ => None,
    }
}

impl<A: App + 'static> ApplicationHandler for Host<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.borrow().is_some() {
            return;
        }
        let canvas = locate_canvas();

        let attrs = Window::default_attributes().with_canvas(Some(canvas.clone()));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let (initial_w, initial_h) = measure_canvas(&canvas);
        canvas.set_width(initial_w);
        canvas.set_height(initial_h);

        // Keep the canvas backing buffer tracking its CSS box size for
        // the lifetime of the page. ResizeObserver fires once on
        // observe() with the initial size, then again on every change.
        let canvas_for_observer = canvas.clone();
        let window_for_observer = window.clone();
        let gfx_for_observer = self.gfx.clone();
        let resize_closure: Closure<dyn FnMut()> = Closure::new(move || {
            let (phys_w, phys_h) = measure_canvas(&canvas_for_observer);
            let mut gfx_borrow = gfx_for_observer.borrow_mut();
            if let Some(gfx) = gfx_borrow.as_mut() {
                apply_canvas_size(&canvas_for_observer, gfx, phys_w, phys_h);
            } else {
                canvas_for_observer.set_width(phys_w);
                canvas_for_observer.set_height(phys_h);
            }
            drop(gfx_borrow);
            window_for_observer.request_redraw();
        });
        let observer = web_sys::ResizeObserver::new(resize_closure.as_ref().unchecked_ref())
            .expect("ResizeObserver::new failed");
        observer.observe(&canvas);
        self._resize_closure = Some(resize_closure);
        self._resize_observer = Some(observer);

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL;
        let instance = wgpu::Instance::new(instance_desc);
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");

        let viewport = self.viewport;
        let shaders = self.app.shaders();
        let theme = self.app.theme();
        let gfx_slot = self.gfx.clone();
        let backend_slot = self.backend.clone();
        let window_for_async = window.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("no compatible adapter");

            let info = adapter.get_info();
            log::info!(
                "aetna-ui: adapter selected — backend={:?} name={:?} driver={:?} device_type={:?}",
                info.backend,
                info.name,
                info.driver,
                info.device_type,
            );
            *backend_slot.borrow_mut() = backend_label(info.backend);

            let downlevel = adapter.get_downlevel_capabilities();
            let per_sample_shading = downlevel
                .flags
                .contains(wgpu::DownlevelFlags::MULTISAMPLED_SHADING);
            if !per_sample_shading {
                log::info!(
                    "aetna-ui: adapter lacks DownlevelFlags::MULTISAMPLED_SHADING; \
                     shaders will downlevel `@interpolate(perspective, sample)` to per-pixel-centre interpolation"
                );
            }

            let limits =
                wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("whisper_agent_aetna_ui::device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("request_device");

            let surface_caps = surface.get_capabilities(&adapter);
            let format = surface_caps
                .formats
                .iter()
                .copied()
                .find(|f| f.is_srgb())
                .unwrap_or(surface_caps.formats[0]);
            let render_format = srgb_view_of(format).unwrap_or(format);
            let view_formats = if render_format != format {
                vec![render_format]
            } else {
                Vec::new()
            };
            log::info!(
                "aetna-ui: surface format {:?} (sRGB? {}) → render view {:?}; offered {:?}",
                format,
                format.is_srgb(),
                render_format,
                surface_caps.formats,
            );

            let inner = window_for_async.inner_size();
            let want_copy_src = surface_caps.usages.contains(wgpu::TextureUsages::COPY_SRC);
            let usage = if want_copy_src {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
            } else {
                log::warn!(
                    "aetna-ui: surface does not advertise COPY_SRC; backdrop-sampling \
                     shaders will paint nothing on this backend"
                );
                wgpu::TextureUsages::RENDER_ATTACHMENT
            };
            let present_mode = if surface_caps
                .present_modes
                .contains(&wgpu::PresentMode::Fifo)
            {
                wgpu::PresentMode::Fifo
            } else {
                surface_caps.present_modes[0]
            };
            let config = wgpu::SurfaceConfiguration {
                usage,
                format,
                width: inner.width.max(1),
                height: inner.height.max(1),
                present_mode,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats,
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&device, &config);

            let mut renderer = Runner::with_caps(
                &device,
                &queue,
                render_format,
                SAMPLE_COUNT,
                per_sample_shading,
            );
            renderer.set_theme(theme);
            renderer.set_surface_size(config.width, config.height);
            for s in shaders {
                if s.samples_backdrop && !want_copy_src {
                    continue;
                }
                renderer.register_shader_with(
                    &device,
                    s.name,
                    s.wgsl,
                    s.samples_backdrop,
                    s.samples_time,
                );
            }

            let msaa = MsaaTarget::new(
                &device,
                render_format,
                surface_extent(&config),
                SAMPLE_COUNT,
            );
            *gfx_slot.borrow_mut() = Some(Gfx {
                window: window_for_async.clone(),
                surface,
                device,
                queue,
                config,
                renderer,
                msaa,
                render_format,
            });
            let _ = viewport;
            window_for_async.request_redraw();
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let mut gfx_borrow = self.gfx.borrow_mut();
        let Some(gfx) = gfx_borrow.as_mut() else {
            return;
        };
        let scale = gfx.window.scale_factor() as f32;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                gfx.config.width = size.width.max(1);
                gfx.config.height = size.height.max(1);
                gfx.surface.configure(&gfx.device, &gfx.config);
                gfx.renderer
                    .set_surface_size(gfx.config.width, gfx.config.height);
                let extent = surface_extent(&gfx.config);
                if !gfx.msaa.matches(extent) {
                    gfx.msaa =
                        MsaaTarget::new(&gfx.device, gfx.render_format, extent, SAMPLE_COUNT);
                }
                self.next_trigger = FrameTrigger::Resize;
                gfx.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let lx = position.x as f32 / scale;
                let ly = position.y as f32 / scale;
                self.last_pointer = Some((lx, ly));
                let moved = gfx.renderer.pointer_moved(lx, ly);
                for event in moved.events {
                    self.app.on_event(event);
                }
                if moved.needs_redraw {
                    self.next_trigger = FrameTrigger::Pointer;
                    gfx.window.request_redraw();
                }
            }

            WindowEvent::CursorLeft { .. } => {
                self.last_pointer = None;
                for event in gfx.renderer.pointer_left() {
                    self.app.on_event(event);
                }
                self.next_trigger = FrameTrigger::Pointer;
                gfx.window.request_redraw();
            }

            // Browser drag/drop is handled by our document-level
            // listeners (see install_drop_handlers); winit's
            // HoveredFile / DroppedFile arms never fire on wasm32.
            WindowEvent::HoveredFile(path) => {
                let (lx, ly) = self.last_pointer.unwrap_or((0.0, 0.0));
                for event in gfx.renderer.file_hovered(path, lx, ly) {
                    self.app.on_event(event);
                }
                self.next_trigger = FrameTrigger::Pointer;
                gfx.window.request_redraw();
            }

            WindowEvent::HoveredFileCancelled => {
                for event in gfx.renderer.file_hover_cancelled() {
                    self.app.on_event(event);
                }
                self.next_trigger = FrameTrigger::Pointer;
                gfx.window.request_redraw();
            }

            WindowEvent::DroppedFile(path) => {
                let (lx, ly) = self.last_pointer.unwrap_or((0.0, 0.0));
                for event in gfx.renderer.file_dropped(path, lx, ly) {
                    self.app.on_event(event);
                }
                self.next_trigger = FrameTrigger::Pointer;
                gfx.window.request_redraw();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let Some(button) = pointer_button(button) else {
                    return;
                };
                let Some((lx, ly)) = self.last_pointer else {
                    return;
                };
                match state {
                    ElementState::Pressed => {
                        for event in gfx.renderer.pointer_down(lx, ly, button) {
                            self.app.on_event(event);
                        }
                        self.next_trigger = FrameTrigger::Pointer;
                        gfx.window.request_redraw();
                    }
                    ElementState::Released => {
                        for event in gfx.renderer.pointer_up(lx, ly, button) {
                            self.app.on_event(event);
                        }
                        self.next_trigger = FrameTrigger::Pointer;
                        gfx.window.request_redraw();
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let Some((lx, ly)) = self.last_pointer else {
                    return;
                };
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y * 50.0,
                    MouseScrollDelta::PixelDelta(p) => -(p.y as f32) / scale,
                };
                if gfx.renderer.pointer_wheel(lx, ly, dy) {
                    self.next_trigger = FrameTrigger::Pointer;
                    gfx.window.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = key_modifiers(modifiers.state());
                gfx.renderer.set_modifiers(self.modifiers);
            }

            WindowEvent::KeyboardInput {
                event:
                    key_event @ winit::event::KeyEvent {
                        state: ElementState::Pressed,
                        ..
                    },
                is_synthetic: false,
                ..
            } => {
                if let Some(key) = map_key(&key_event.logical_key) {
                    for event in gfx.renderer.key_down(key, self.modifiers, key_event.repeat) {
                        self.app.on_event(event);
                    }
                }
                if let Some(text) = &key_event.text
                    && let Some(event) = gfx.renderer.text_input(text.to_string())
                {
                    self.app.on_event(event);
                }
                self.next_trigger = FrameTrigger::Keyboard;
                gfx.window.request_redraw();
            }
            WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                if let Some(event) = gfx.renderer.text_input(text) {
                    self.app.on_event(event);
                }
                self.next_trigger = FrameTrigger::Keyboard;
                gfx.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();
                let frame = match gfx.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                    wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                        gfx.surface.configure(&gfx.device, &gfx.config);
                        return;
                    }
                    other => {
                        log::error!("surface unavailable: {other:?}");
                        return;
                    }
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(gfx.render_format),
                    ..Default::default()
                });

                let last_frame_dt = self
                    .last_frame_at
                    .map(|t| frame_start.duration_since(t))
                    .unwrap_or(std::time::Duration::ZERO);
                self.last_frame_at = Some(frame_start);
                let trigger = std::mem::take(&mut self.next_trigger);
                let scale_factor = gfx.window.scale_factor() as f32;
                let viewport_rect = Rect::new(
                    0.0,
                    0.0,
                    gfx.config.width as f32 / scale_factor,
                    gfx.config.height as f32 / scale_factor,
                );
                let current_size = (gfx.config.width, gfx.config.height);
                let paint_only = trigger == FrameTrigger::ShaderPaint
                    && Some(current_size) == self.last_prepared_size;

                let (prepare, palette, t_after_build, t_after_prepare) = if paint_only {
                    let palette = gfx.renderer.theme().palette().clone();
                    let t_after_build = Instant::now();
                    let prepare =
                        gfx.renderer
                            .repaint(&gfx.device, &gfx.queue, viewport_rect, scale_factor);
                    let t_after_prepare = Instant::now();
                    (prepare, palette, t_after_build, t_after_prepare)
                } else {
                    self.frame_index = self.frame_index.wrapping_add(1);
                    let diagnostics = HostDiagnostics {
                        backend: *self.backend.borrow(),
                        surface_size: (gfx.config.width, gfx.config.height),
                        scale_factor,
                        msaa_samples: SAMPLE_COUNT,
                        frame_index: self.frame_index,
                        last_frame_dt,
                        trigger,
                    };
                    self.app.before_build();
                    let theme = self.app.theme();
                    let cx = BuildCx::new(&theme)
                        .with_ui_state(gfx.renderer.ui_state())
                        .with_diagnostics(&diagnostics);
                    let mut tree = self.app.build(&cx);
                    let palette = theme.palette().clone();
                    gfx.renderer.set_theme(theme);
                    gfx.renderer.set_hotkeys(self.app.hotkeys());
                    gfx.renderer.set_selection(self.app.selection());
                    gfx.renderer.push_toasts(self.app.drain_toasts());
                    gfx.renderer
                        .push_focus_requests(self.app.drain_focus_requests());
                    gfx.renderer
                        .push_scroll_requests(self.app.drain_scroll_requests());
                    let t_after_build = Instant::now();
                    let prepare = gfx.renderer.prepare(
                        &gfx.device,
                        &gfx.queue,
                        &mut tree,
                        viewport_rect,
                        scale_factor,
                    );
                    let t_after_prepare = Instant::now();

                    let cursor = gfx.renderer.ui_state().cursor(&tree);
                    if cursor != self.last_cursor {
                        gfx.window.set_cursor(winit_cursor(cursor));
                        self.last_cursor = cursor;
                    }
                    self.last_prepared_size = Some(current_size);
                    (prepare, palette, t_after_build, t_after_prepare)
                };

                let mut encoder =
                    gfx.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("whisper_agent_aetna_ui::encoder"),
                        });
                gfx.renderer.render(
                    &gfx.device,
                    &mut encoder,
                    &frame.texture,
                    &view,
                    Some(&gfx.msaa.view),
                    wgpu::LoadOp::Clear(bg_color(&palette)),
                );
                gfx.queue.submit(Some(encoder.finish()));
                frame.present();
                let t_after_submit = Instant::now();

                self.stats.record(
                    frame_start,
                    t_after_build,
                    t_after_prepare,
                    t_after_submit,
                    prepare.timings,
                );

                if prepare.next_layout_redraw_in.is_some() {
                    self.next_trigger = FrameTrigger::Animation;
                    gfx.window.request_redraw();
                } else if prepare.next_paint_redraw_in.is_some() {
                    self.next_trigger = FrameTrigger::ShaderPaint;
                    gfx.window.request_redraw();
                }
                let _ = self.viewport;
            }
            _ => {}
        }
    }
}

fn map_key(key: &Key) -> Option<UiKey> {
    match key {
        Key::Named(NamedKey::Enter) => Some(UiKey::Enter),
        Key::Named(NamedKey::Escape) => Some(UiKey::Escape),
        Key::Named(NamedKey::Tab) => Some(UiKey::Tab),
        Key::Named(NamedKey::Space) => Some(UiKey::Space),
        Key::Named(NamedKey::ArrowUp) => Some(UiKey::ArrowUp),
        Key::Named(NamedKey::ArrowDown) => Some(UiKey::ArrowDown),
        Key::Named(NamedKey::ArrowLeft) => Some(UiKey::ArrowLeft),
        Key::Named(NamedKey::ArrowRight) => Some(UiKey::ArrowRight),
        Key::Named(NamedKey::Backspace) => Some(UiKey::Backspace),
        Key::Named(NamedKey::Delete) => Some(UiKey::Delete),
        Key::Named(NamedKey::Home) => Some(UiKey::Home),
        Key::Named(NamedKey::End) => Some(UiKey::End),
        Key::Named(NamedKey::PageUp) => Some(UiKey::PageUp),
        Key::Named(NamedKey::PageDown) => Some(UiKey::PageDown),
        Key::Character(s) => Some(UiKey::Character(s.to_string())),
        Key::Named(named) => Some(UiKey::Other(format!("{named:?}"))),
        _ => None,
    }
}

fn pointer_button(b: MouseButton) -> Option<PointerButton> {
    match b {
        MouseButton::Left => Some(PointerButton::Primary),
        MouseButton::Right => Some(PointerButton::Secondary),
        MouseButton::Middle => Some(PointerButton::Middle),
        _ => None,
    }
}

fn key_modifiers(mods: winit::keyboard::ModifiersState) -> KeyModifiers {
    KeyModifiers {
        shift: mods.shift_key(),
        ctrl: mods.control_key(),
        alt: mods.alt_key(),
        logo: mods.super_key(),
    }
}

fn winit_cursor(cursor: Cursor) -> CursorIcon {
    match cursor {
        Cursor::Default => CursorIcon::Default,
        Cursor::Pointer => CursorIcon::Pointer,
        Cursor::Text => CursorIcon::Text,
        Cursor::NotAllowed => CursorIcon::NotAllowed,
        Cursor::Grab => CursorIcon::Grab,
        Cursor::Grabbing => CursorIcon::Grabbing,
        Cursor::Move => CursorIcon::Move,
        Cursor::EwResize => CursorIcon::EwResize,
        Cursor::NsResize => CursorIcon::NsResize,
        Cursor::NwseResize => CursorIcon::NwseResize,
        Cursor::NeswResize => CursorIcon::NeswResize,
        Cursor::ColResize => CursorIcon::ColResize,
        Cursor::RowResize => CursorIcon::RowResize,
        Cursor::Crosshair => CursorIcon::Crosshair,
        _ => CursorIcon::Default,
    }
}

fn bg_color(palette: &Palette) -> wgpu::Color {
    let c = palette.background;
    wgpu::Color {
        r: srgb_to_linear(c.r as f64 / 255.0),
        g: srgb_to_linear(c.g as f64 / 255.0),
        b: srgb_to_linear(c.b as f64 / 255.0),
        a: c.a as f64 / 255.0,
    }
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
