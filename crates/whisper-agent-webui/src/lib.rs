//! whisper-agent-webui — browser chat UI.
//!
//! Step 5 of the MVP: scaffold only — renders the chat layout, no networking yet.
//! WebSocket wiring lands in step 6.

mod app;

pub use app::ChatApp;

#[cfg(target_arch = "wasm32")]
mod web_entry {
    use wasm_bindgen::prelude::*;

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
                    Box::new(|_cc| Ok(Box::new(super::ChatApp::default()))),
                )
                .await;

            if let Err(e) = result {
                log::error!("eframe start failed: {e:?}");
            }
        });
    }
}
