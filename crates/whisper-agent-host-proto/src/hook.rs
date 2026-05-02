//! Hook event tagging — the future-extension surface for behavior
//! integration, audit streams, resource warnings, etc.
//!
//! Schema is intentionally open in v1: the [`crate::Frame::HookEvent`]
//! payload is `serde_json::Value` and the kind is a free-form string.
//! Once the behavior/hook integration in
//! `docs/design_behaviors.md` is concrete enough to pin down, we
//! either formalize [`HookKind`] as an enum or split into typed
//! frames; today's open shape doesn't paint us into a corner.

use serde::{Deserialize, Serialize};

/// Tag for a [`crate::Frame::HookEvent`]. Recipients ignore unknown
/// kinds rather than failing — a daemon may emit kinds the scheduler
/// hasn't learned about yet, and vice-versa.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct HookKind(pub String);

impl HookKind {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for HookKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_kind_round_trips() {
        let k = HookKind::new("tool_started");
        let mut bytes = Vec::new();
        ciborium::into_writer(&k, &mut bytes).unwrap();
        let back: HookKind = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(k, back);
    }
}
