//! Modal dialogs and editor windows.
//!
//! One submodule per modal (or cluster of related modals). Each
//! exposes a free render function plus, where the modal needs to
//! dispatch wire messages or open another modal, an event enum the
//! parent reduces back into `ChatApp` mutations.
//!
//! Pattern: state is passed in by reference (typically the
//! `Option<...ModalState>` slot the modal lives in), no `ChatApp`
//! access from inside the renderer. The parent (`app.rs`) owns the
//! slots, opens them by writing `Some(...)`, and lets the renderer
//! clear them on close. Mirrors the row-renderer convention in
//! `widgets`.

mod behavior_editor;
mod buckets;
mod fork;
mod new_behavior;
mod new_pod;
mod pod_editor;
mod provider_editor;
mod settings;
mod viewers;

pub(super) use behavior_editor::{BehaviorEditorEvent, render_behavior_editor_modal};
#[allow(unused_imports)]
pub(super) use buckets::BucketsEvent;
pub(super) use buckets::render_buckets_modal;
pub(super) use fork::{ForkEvent, render_fork_modal};
pub(super) use new_behavior::{NewBehaviorEvent, render_new_behavior_modal};
pub(super) use new_pod::{NewPodEvent, render_new_pod_modal};
pub(super) use pod_editor::{PodEditorEvent, render_pod_editor_modal};
pub(super) use provider_editor::{ProviderEditorEvent, render_provider_editor_modal};
pub(super) use settings::{SettingsEvent, render_settings_modal};
pub(super) use viewers::{
    FileViewerEvent, render_file_viewer_modal, render_image_lightbox_modal,
    render_json_viewer_modal,
};
