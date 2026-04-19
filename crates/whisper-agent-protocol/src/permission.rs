//! Permission-model wire types shared between server and webui.
//!
//! `Disposition` and `AllowMap<T>` are the two data types used to
//! express "what the scope admits." Full documentation of the model
//! lives in `docs/design_functions.md`; this module carries only the
//! shapes the wire protocol needs.
//!
//! The richer `PermissionScope` / `PodsScope` types — including
//! per-pod operation enums and narrowing composition for all fields —
//! live in the main `whisper-agent` crate's `permission` module, which
//! re-exports these two so downstream code sees one type.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// What the scope admits for a given item.
///
/// The derive-order `Allow < AllowWithPrompt < Deny` is the
/// restrictiveness ordering used by narrowing composition — the *more*
/// restrictive disposition wins when two scopes intersect.
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    #[default]
    Allow,
    AllowWithPrompt,
    Deny,
}

impl Disposition {
    /// Combine two dispositions under narrowing composition: more
    /// restrictive wins.
    pub fn narrow(self, other: Self) -> Self {
        self.max(other)
    }

    /// Does this disposition admit the operation at all (with or
    /// without prompting)?
    pub fn admits(self) -> bool {
        !matches!(self, Self::Deny)
    }

    /// Does this disposition require user prompting before the
    /// operation proceeds?
    pub fn requires_prompt(self) -> bool {
        matches!(self, Self::AllowWithPrompt)
    }
}

/// Per-item disposition with a default for unlisted items.
///
/// Narrowing composition picks the more restrictive disposition per
/// item: for any `k`, the result's disposition is
/// `narrow(self.disposition(k), other.disposition(k))`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AllowMap<T: Ord + Clone> {
    #[serde(default)]
    pub default: Disposition,
    #[serde(default = "BTreeMap::new", skip_serializing_if = "BTreeMap::is_empty")]
    pub overrides: BTreeMap<T, Disposition>,
}

impl<T: Ord + Clone> AllowMap<T> {
    /// All items admitted with `Allow`.
    pub fn allow_all() -> Self {
        Self {
            default: Disposition::Allow,
            overrides: BTreeMap::new(),
        }
    }

    /// All items denied.
    pub fn deny_all() -> Self {
        Self {
            default: Disposition::Deny,
            overrides: BTreeMap::new(),
        }
    }

    /// Disposition for `item`: override if present, otherwise default.
    pub fn disposition(&self, item: &T) -> Disposition {
        self.overrides.get(item).copied().unwrap_or(self.default)
    }

    /// Narrow `self` with `other`: more-restrictive disposition wins
    /// per item. Post-narrow default is `narrow(self.default, other.default)`.
    /// Overrides that equal the post-narrow default are elided to keep
    /// the map small.
    pub fn narrow(&self, other: &Self) -> Self {
        let default = self.default.narrow(other.default);
        let all_keys: std::collections::BTreeSet<&T> = self
            .overrides
            .keys()
            .chain(other.overrides.keys())
            .collect();
        let mut overrides = BTreeMap::new();
        for k in all_keys {
            let merged = self.disposition(k).narrow(other.disposition(k));
            if merged != default {
                overrides.insert(k.clone(), merged);
            }
        }
        Self { default, overrides }
    }

    /// Upgrade a specific item's disposition to `Allow`, ignoring
    /// whatever the underlying default/override said. Used for
    /// per-tool remember-approval overrides.
    pub fn set_allow(&mut self, item: T) {
        if self.default == Disposition::Allow {
            self.overrides.remove(&item);
        } else {
            self.overrides.insert(item, Disposition::Allow);
        }
    }
}

impl<T: Ord + Clone> Default for AllowMap<T> {
    fn default() -> Self {
        Self::allow_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disposition_ordering() {
        assert!(Disposition::Allow < Disposition::AllowWithPrompt);
        assert!(Disposition::AllowWithPrompt < Disposition::Deny);
    }

    #[test]
    fn allowmap_narrow_picks_more_restrictive() {
        use Disposition::*;
        let a: AllowMap<String> = AllowMap {
            default: Allow,
            overrides: [("bash".into(), AllowWithPrompt)].into_iter().collect(),
        };
        let b: AllowMap<String> = AllowMap {
            default: AllowWithPrompt,
            overrides: [("bash".into(), Deny)].into_iter().collect(),
        };
        let r = a.narrow(&b);
        assert_eq!(r.default, AllowWithPrompt);
        assert_eq!(r.disposition(&"bash".into()), Deny);
    }

    #[test]
    fn allowmap_set_allow_removes_override_when_default_is_allow() {
        use Disposition::*;
        let mut m: AllowMap<String> = AllowMap {
            default: Allow,
            overrides: [("bash".into(), AllowWithPrompt)].into_iter().collect(),
        };
        m.set_allow("bash".into());
        assert!(m.overrides.is_empty());
    }

    #[test]
    fn allowmap_set_allow_inserts_override_when_default_is_not_allow() {
        use Disposition::*;
        let mut m: AllowMap<String> = AllowMap {
            default: AllowWithPrompt,
            overrides: BTreeMap::new(),
        };
        m.set_allow("bash".into());
        assert_eq!(m.disposition(&"bash".into()), Allow);
    }
}
