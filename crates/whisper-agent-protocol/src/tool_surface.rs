//! Tool-surface configuration — how many tools the model sees in
//! its wire `tools:` array vs. as names-only prose vs. fetched
//! on demand. See `docs/design_permissions_rework.md` for why this
//! exists (prompt-cache coherence + MCP-attach mid-conversation +
//! escalation vocabulary discovery).
//!
//! Composed from `pod.thread_defaults.tool_surface` (base) narrowed
//! by `behavior.scope.tool_surface` (optional per-behavior replace).

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Full tool-surface policy for a thread. Carried on `ThreadDefaults`
/// as the pod baseline and referenced from `BehaviorScope` as an
/// optional full-replace override.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ToolSurface {
    /// Which tools get full schemas in the wire `tools:` array (and
    /// thus the thread's `Role::Tools` message at position 1).
    /// Everything outside this set is still callable — the model
    /// either learned the name from `initial_listing` or from a
    /// mid-conversation activation message, and fetches the schema
    /// via `describe_tool` on demand.
    #[serde(default = "default_core_tools")]
    pub core_tools: CoreTools,
    /// What to include in the system-prompt-appended tool catalog
    /// snapshot at thread creation.
    #[serde(default = "default_initial_listing")]
    pub initial_listing: InitialListing,
    /// How mid-conversation tool additions (escalation AddTool grants,
    /// MCP hosts attached after thread start) are surfaced to the
    /// model. Append-only in every case — position-1 `Role::Tools` is
    /// never rewritten after thread seed (that would bust the prompt
    /// cache).
    #[serde(default = "default_activation_surface")]
    pub activation_surface: ActivationSurface,
}

impl Default for ToolSurface {
    fn default() -> Self {
        Self {
            core_tools: default_core_tools(),
            initial_listing: default_initial_listing(),
            activation_surface: default_activation_surface(),
        }
    }
}

/// Which tools get full schemas in the thread's wire `tools:` field.
///
/// On disk accepts either the string `"all"` (unrestricted — every
/// admissible tool gets a full schema, matching pre-rework behavior)
/// or a list of tool names. Names that aren't admissible by the
/// thread's scope are silently dropped at seed time — the policy says
/// "include these IF admissible," not "admit these."
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreTools {
    /// Every scope-admitted tool's full schema lands in `Role::Tools`.
    /// Reproduces the original catalog-dumping behavior and is still
    /// useful when the pod has few tools or the author explicitly
    /// wants "nothing deferred."
    All,
    /// Only these names get full schemas. The rest appear in the
    /// system-prompt listing (if `initial_listing = AllNames`) with
    /// schemas fetched on demand via `describe_tool`.
    Named(Vec<String>),
}

impl Serialize for CoreTools {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::All => s.serialize_str("all"),
            Self::Named(v) => v.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for CoreTools {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Shape {
            Str(String),
            Vec(Vec<String>),
        }
        match Shape::deserialize(d)? {
            Shape::Str(s) if s == "all" => Ok(Self::All),
            Shape::Str(s) => Err(serde::de::Error::custom(format!(
                "unknown core_tools sentinel `{s}` (expected \"all\" or a list)"
            ))),
            Shape::Vec(v) => Ok(Self::Named(v)),
        }
    }
}

fn default_core_tools() -> CoreTools {
    CoreTools::Named(vec![
        "describe_tool".to_string(),
        "find_tool".to_string(),
        "request_escalation".to_string(),
    ])
}

/// What tool-catalog content gets appended to the system prompt at
/// thread seed time.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum InitialListing {
    /// Enumerate every admissible tool — name + one-line description,
    /// grouped by source (builtins / host-env MCP / shared MCP), plus
    /// a trailing "available via escalation" section when the thread
    /// has an interactive channel. This is the default: it solves
    /// discovery at the cost of a few hundred tokens of prose, vs.
    /// the kilobytes that full schemas would cost.
    #[default]
    AllNames,
    /// Only a summary: counts by group, plus the core-tools names.
    /// Useful when the pod has so many tools that even the name list
    /// is uncomfortably long; the model relies on `find_tool` to
    /// discover what exists.
    CoreOnly,
    /// No listing appended — the model learns about tools only via
    /// `find_tool`, or never. Discovery-first.
    None,
}

/// How mid-conversation additions are surfaced to the model. In
/// every case the conversation gains a new `Role::System` block at
/// the tail — position-1 `Role::Tools` is never edited after seed.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ActivationSurface {
    /// Append a brief `"Tools newly available: foo, bar. Call
    /// describe_tool for schemas."` message. Defers schema cost
    /// until the model actually needs it; schemas cache naturally
    /// once fetched.
    #[default]
    Announce,
    /// Append the newly-available tools' full schemas inline in a
    /// system message. Trades upfront cost for zero round-trips to
    /// `describe_tool`. Right choice when a single activation is
    /// known to be immediately relevant.
    InjectSchema,
}

fn default_initial_listing() -> InitialListing {
    InitialListing::AllNames
}

fn default_activation_surface() -> ActivationSurface {
    ActivationSurface::Announce
}

impl ToolSurface {
    /// Compose a pod-baseline surface with an optional behavior
    /// override. A `Some` override replaces the baseline wholesale;
    /// `None` leaves the baseline untouched. Matches how
    /// `BehaviorThreadOverride` works — behaviors either inherit or
    /// replace, no per-field narrowing.
    pub fn compose(&self, behavior_override: Option<&ToolSurface>) -> ToolSurface {
        behavior_override.cloned().unwrap_or_else(|| self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_small_core_plus_all_names() {
        let s = ToolSurface::default();
        match &s.core_tools {
            CoreTools::Named(v) => {
                assert_eq!(v.len(), 3);
                assert!(v.iter().any(|n| n == "describe_tool"));
                assert!(v.iter().any(|n| n == "find_tool"));
                assert!(v.iter().any(|n| n == "request_escalation"));
            }
            CoreTools::All => panic!("default should be a small named set, not All"),
        }
        assert_eq!(s.initial_listing, InitialListing::AllNames);
        assert_eq!(s.activation_surface, ActivationSurface::Announce);
    }

    #[test]
    fn core_tools_all_serializes_as_string() {
        let v = serde_json::to_value(CoreTools::All).unwrap();
        assert_eq!(v, serde_json::json!("all"));
    }

    #[test]
    fn core_tools_named_serializes_as_list() {
        let v = serde_json::to_value(CoreTools::Named(vec!["a".into(), "b".into()])).unwrap();
        assert_eq!(v, serde_json::json!(["a", "b"]));
    }

    #[test]
    fn core_tools_deserializes_string_all() {
        let v: CoreTools = serde_json::from_value(serde_json::json!("all")).unwrap();
        assert_eq!(v, CoreTools::All);
    }

    #[test]
    fn core_tools_deserializes_list() {
        let v: CoreTools = serde_json::from_value(serde_json::json!(["x", "y"])).unwrap();
        assert_eq!(v, CoreTools::Named(vec!["x".to_string(), "y".to_string()]));
    }

    #[test]
    fn core_tools_rejects_unknown_string() {
        let err = serde_json::from_value::<CoreTools>(serde_json::json!("everything")).unwrap_err();
        assert!(err.to_string().contains("unknown core_tools sentinel"));
    }

    #[test]
    fn compose_uses_override_when_some() {
        let base = ToolSurface::default();
        let over = ToolSurface {
            core_tools: CoreTools::All,
            initial_listing: InitialListing::None,
            activation_surface: ActivationSurface::InjectSchema,
        };
        let composed = base.compose(Some(&over));
        assert_eq!(composed.core_tools, CoreTools::All);
        assert_eq!(composed.initial_listing, InitialListing::None);
        assert_eq!(composed.activation_surface, ActivationSurface::InjectSchema);
    }

    #[test]
    fn compose_preserves_base_when_none() {
        let base = ToolSurface::default();
        let composed = base.compose(None);
        assert_eq!(composed, base);
    }

    #[test]
    fn tool_surface_round_trips_through_json() {
        let s = ToolSurface {
            core_tools: CoreTools::Named(vec!["describe_tool".into()]),
            initial_listing: InitialListing::CoreOnly,
            activation_surface: ActivationSurface::InjectSchema,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: ToolSurface = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }
}
