//! Pattern-based classifiers for pod-relative filenames. Shared between
//! the agent-facing `pod_*_file` tools and server-side handlers (e.g. the
//! webui's directory-listing endpoint) so both surfaces agree on what's
//! runtime state that clients must not overwrite.

use crate::pod;
use crate::pod::behaviors::{BEHAVIOR_STATE, BEHAVIORS_DIR, validate_behavior_id};

/// Pod-relative directory name holding the agent's persistent memory
/// index + per-memory files. Siblings of `behaviors/` and `threads/`.
pub const MEMORY_DIR: &str = "memory";
/// Filename of the index that lists every individual memory file.
/// Auto-injected into each thread's conversation log at creation time.
pub const MEMORY_INDEX: &str = "MEMORY.md";

/// Return the filename (basename + extension, single path component) when
/// `filename` matches `memory/<name>.md`. Nested subdirs or non-`.md`
/// suffixes return `None` — memory is intentionally flat.
pub fn parse_memory_path(filename: &str) -> Option<&str> {
    let rest = filename.strip_prefix(&format!("{MEMORY_DIR}/"))?;
    if rest.is_empty() || rest.contains('/') || rest.starts_with('.') {
        return None;
    }
    if !rest.ends_with(".md") {
        return None;
    }
    Some(rest)
}

/// Split `behaviors/<id>/<suffix>` into `(id, suffix)` when `filename`
/// matches that shape. Both parts must be a single path component — nested
/// subdirs return `None`.
pub fn parse_behavior_path(filename: &str) -> Option<(&str, &str)> {
    let rest = filename.strip_prefix(&format!("{BEHAVIORS_DIR}/"))?;
    let (id, suffix) = rest.split_once('/')?;
    if id.is_empty() || suffix.is_empty() || suffix.contains('/') {
        return None;
    }
    Some((id, suffix))
}

/// Return the thread id for a path like `threads/<id>.json`, or None.
pub fn parse_thread_path(filename: &str) -> Option<&str> {
    let rest = filename.strip_prefix(&format!("{}/", pod::THREADS_DIR))?;
    if rest.contains('/') || rest.starts_with('.') {
        return None;
    }
    let id = rest.strip_suffix(".json")?;
    if id.is_empty() {
        return None;
    }
    Some(id)
}

/// Pod-relative path reachable for reads but not writes — runtime state
/// the scheduler owns (thread JSONs, pod_state.json, per-behavior
/// state.json). Both the `pod_write_file` tool and any webui write
/// endpoint must reject these paths.
pub fn is_readonly_path(filename: &str) -> bool {
    if filename == pod::POD_STATE_JSON {
        return true;
    }
    if parse_thread_path(filename).is_some() {
        return true;
    }
    if let Some((id, suffix)) = parse_behavior_path(filename)
        && suffix == BEHAVIOR_STATE
        && validate_behavior_id(id).is_ok()
    {
        return true;
    }
    false
}
